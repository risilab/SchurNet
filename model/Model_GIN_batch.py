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

class Node_node(torch.nn.Module):
    '''Node to node message passing
        replicate of GINEConv
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()

        self.node_mlp = Sequential(
            Linear(hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.epsilon = Parameter(torch.Tensor([0.0]))

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep: ptens0_type, edge_attr: ptens0_type,G, data):
        # print("----------------starting MPNN---------------")
        # print("node_rep: ",node_rep)
        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep,G,self.edge)
        node2edge = edge_attr + node2edge

        # node2edge = node2edge.relu(0)
        node_new = p.batched_subgraphlayer0b.gather_from_ptensors(node2edge,G,self.node)

        node_out = self.node_mlp(node_new.torch() + (1 + self.epsilon - data.degree.reshape((-1,1))) * node_rep.torch())
        # node_out = self.node_mlp(torch.cat([node_new.torch(),node_rep.torch()],dim=-1))
        node_out = p.batched_subgraphlayer0b.like(node_rep,node_out)
        return node_out

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout) -> None:
        super().__init__()
        self.node_node = Node_node(hidden_dim)

    def forward(self, node_rep: ptens0_type, edge_attr,G, data):
        node_out = self.node_node(node_rep,edge_attr,G, data)

        #residual connection
        # node_out = node_rep + node_out
        # edge_out = edge_rep + edge_out
        return node_out


class Model_GIN_batch(torch.nn.Module):
    def __init__(self, hidden_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, ds_name,dataset,device = 'cuda') -> None:
        print("Running: Model_GIN_batch\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,hidden_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,hidden_dim)

        # self.init_mlp = Sequential(
        #                     Linear(hidden_dim,hidden_dim * _inner_mlp_mult,False),
        #                     BatchNorm1d(hidden_dim*_inner_mlp_mult),
        #                     ReLU(),
        #                     Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
        #                     BatchNorm1d(hidden_dim),
        #                     ReLU()
        #                 )

        self.conv_layers = ModuleList([ConvLayer(hidden_dim,dropout) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(hidden_dim,dense_dim,False),
                                    BatchNorm1d(dense_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim,False),
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

        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x))
        edge_attr = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr))
        edge_attr = p.batched_subgraphlayer0b.gather_from_ptensors(edge_attr,G,self.edge)

        for conv_layer in self.conv_layers:
            node_rep = conv_layer(node_rep, edge_attr, G, data)

        reps = node_rep.torch()
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)

        return self.lin(reps) # size = [num_graphs,out_dim]

