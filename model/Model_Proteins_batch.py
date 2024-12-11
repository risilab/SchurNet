import torch
import ptens as p
from torch.nn import functional as F, Sequential, ModuleList, Linear, BatchNorm1d, ReLU, Parameter
from typing import Callable, Union, List
from functools import reduce
from argparse import Namespace


from data_cleaning.data_loader import get_data_handler
from data_cleaning.utils import get_model_size, get_out_dim
from data_cleaning.feature_encoders import get_node_encoder, get_edge_encoder
from data_cleaning.Transforms import PreAddIndex, StandardPreprocessing

from torch_geometric.nn import global_add_pool, global_mean_pool

_inner_mlp_mult = 2

ptens0_type = Union[p.batched_ptensors0b,p.batched_subgraphlayer0b]
ptens1_type = Union[p.batched_ptensors1b,p.batched_subgraphlayer1b]

class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, hidden_dim) -> None:
        super().__init__()

        self.node_mlp = Sequential(
            Linear(4 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.edge_mlp = Sequential(
            Linear(3 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
   
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep: ptens0_type, edge_rep: ptens1_type,G):
        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node2edge,G,self.edge) #nc * 2

        edge_new = p.batched_subgraphlayer1b.cat_channels(edge_rep,node2edge) #nc * 3
        edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge_new,G,self.node)

        edge_out = self.edge_mlp(edge_new.torch()) #nc * 3 -> nc
        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],-1)) #nc * 4 -> nc
        node_out = p.batched_subgraphlayer0b.like(node_rep,node_out)
        edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        return node_out, edge_out

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout) -> None:
        super().__init__()
        self.edge_node = Edge_node(hidden_dim)

    def forward(self, node_rep: ptens0_type, edge_rep: ptens1_type,G):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out = self.edge_node(node_rep,edge_rep,G)

        #residual connection
        # node_out = node_rep + node_out
        # edge_out = edge_rep + edge_out
        return node_out, edge_out


class Model_Proteins_batch(torch.nn.Module):
    def __init__(self, hidden_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, ds_name,dataset,device = 'cuda') -> None:
        print("Running: Model_Proteins_batch\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,hidden_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,hidden_dim)
        
        self.conv_layers = ModuleList([ConvLayer(hidden_dim,dropout) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(2 * hidden_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU()
                                    )

        self.dropout = dropout
        self.lin = Linear(dense_dim,out_dim)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        
    def forward(self, data) -> torch.Tensor:
        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)

        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x.flatten()))
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr.flatten()))
        edge_rep = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge)

        for conv_layer in self.conv_layers:
            node_rep, edge_rep = conv_layer(node_rep, edge_rep, G)

        # node_outs = []
        node_outs = [node_rep.torch()]
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node).torch())
        reps = torch.cat(node_outs,-1) #nc * 2
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps) # size = [num_graphs,out_dim]
    
# ---- Main --------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print("test")
    device = 'cuda'

    args = Namespace(
        ds_name='MUTAG',
        batch_size=2,
        eval_batch_size=2,
        num_folds=10,
        fold_idx = 0
    )

    data_handler = get_data_handler(PreAddIndex(),args)

    model = Model_Proteins_batch(hidden_dim=4,dense_dim=4,out_dim=1,num_layers=4,dropout=0.,readout=global_add_pool,ds_name=args.ds_name,dataset=data_handler.ds).to(device)

    loss_fn = torch.nn.L1Loss()
    for data in data_handler.train_dataloader():
        # print("batch is:", data)
        data.to(device)
        pred = model(data)
        loss = loss_fn(pred,data.y)
        print(f"loss: {loss}")
        loss.backward()
        break
