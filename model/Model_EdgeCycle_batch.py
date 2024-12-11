import torch
import ptens as p
from torch.nn import functional as F
from torch.nn import Sequential,ModuleList,Linear,BatchNorm1d,ReLU, Parameter
from typing import Callable, Union
from torch_geometric.nn import global_add_pool, global_mean_pool
from data_cleaning.utils import get_model_size
from data_cleaning.feature_encoders import get_node_encoder, get_edge_encoder
from argparse import Namespace
from train import get_loss_fn
from data_cleaning.Transforms import PreAddIndex
from data_cleaning.data_loader import get_data_handler

_inner_mlp_mult = 2


class Edge_edge(torch.nn.Module):
    '''Edge and edge message passing'''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(5 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,edge_rep,G):
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge) #nc * 2
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge2edge,G,self.edge) #nc * 4

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),edge2edge.torch()],-1)) #nc * 5 -> nc
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        return edge_out
    
class Edge_Cycle(torch.nn.Module):
    '''node, edge and cycle message passing'''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(11 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.cycle_mlp = Sequential(
            Linear(5 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
       
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycle5=p.subgraph.cycle(5)
        self.cycle6=p.subgraph.cycle(6)

    def forward(self,edge_rep,cycle_rep,G):
        edge2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.cycle5) #nc * 2
        edge2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle5,G,self.cycle5) #nc * 4
        edge2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.cycle6) #nc * 2
        edge2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle6,G,self.cycle6) #nc * 4

        cycle_in = p.batched_subgraphlayer1b.cat(edge2cycle5,edge2cycle6) 
        # cycle_in = p.ptensors1b.linmaps(cycle_in) #nc * 6
        cycle_new = p.batched_subgraphlayer1b.cat_channels(cycle_in,cycle_rep) #nc * 5; nc * 2

        cycle2edge = p.batched_subgraphlayer1b.gather_from_ptensors(cycle_new,G,self.edge) #nc * 10; nc * 4

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch()],-1)) # nc * 11 -> nc; nc * 5 -> nc
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        cycle_out = self.cycle_mlp(cycle_new.torch()) #nc * 5 -> nc; nc * 2
        cycle_out = p.batched_ptensors1b.like(cycle_rep,cycle_out)
        return edge_out, cycle_out

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout) -> None:
        super().__init__()
        self.mlp = Sequential(
            Linear(2*hidden_dim,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.edge2edge = Edge_edge(hidden_dim)
        self.edge_cycle = Edge_Cycle(hidden_dim)

    def forward(self, edge_rep, cycle_rep,G):
        edge_out = self.edge2edge(edge_rep, G)
        edge_out2, cycle_out = self.edge_cycle(edge_rep, cycle_rep, G)
        #edge_out, edge_out2 is torch.tensor

        edge_out = self.mlp(torch.cat([edge_out,edge_out2],-1))
        edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)

        #residual connection
        # edge_out = edge_rep + edge_out
        # cycle_out = cycle_rep + cycle_out 
        return edge_out, cycle_out


class Model_EdgeCycle_batch(torch.nn.Module):
    def __init__(self, hidden_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, ds_name,dataset,device = 'cuda') -> None:
        print("Running: Model_EdgeCycle_batch\n")
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
        self.edge_mlp1 = p.Linear(2 * hidden_dim,hidden_dim * _inner_mlp_mult)
        self.edge_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.cycle_mlp1 = p.Linear(2 * hidden_dim,hidden_dim * _inner_mlp_mult)
        self.cycle_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.dropout = dropout
        self.lin = Linear(dense_dim,out_dim)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycle5=p.subgraph.cycle(5)
        self.cycle6=p.subgraph.cycle(6)

        self.device = device
        
    def forward(self, data) -> torch.Tensor:
        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)

        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x.flatten()))
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr.flatten()))
        
        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge)

        edge_rep = p.batched_subgraphlayer1b.cat_channels(node2edge,edge2edge) #nc * 2
        edge_rep = self.edge_mlp1(edge_rep).relu()
        edge_rep = self.edge_mlp2(edge_rep).relu() #nc = hidden_dim

        node2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.cycle5)
        node2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(node2cycle5,G,self.cycle5) #nc * 2
        node2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.cycle6)
        node2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(node2cycle6,G,self.cycle6) #nc * 2

        cycle_rep = p.batched_subgraphlayer1b.cat(node2cycle5,node2cycle6)
        cycle_rep = self.cycle_mlp1(cycle_rep).relu()
        cycle_rep = self.cycle_mlp2(cycle_rep).relu() #nc = hidden_dim


        for conv_layer in self.conv_layers:
            edge_rep, cycle_rep = conv_layer(edge_rep, cycle_rep, G)

        node_out_1 = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node)
        node_out_2 = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node)
        reps = torch.cat([node_out_1.torch(),node_out_2.torch()],-1) #nc * 2
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps)
    

