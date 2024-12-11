from typing import Callable, Optional, Union
import typing

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F

from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm
import ptens as p

from torch_geometric.datasets import ZINC
from torch_geometric.datasets import TUDataset

from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_geometric.nn import GINConv, GINEConv

from .node_edge import NodeEdge

from .node_edge_stable import Edge_node
from .edge_cycle import EdgeCycle

import copy

# import files from previous directory
import sys
import os

sys.path.append("..")
# from utils import global_mean_pool

# save_dir = "test_GIN_edge"
# os.makedirs(save_dir, exist_ok=True)
from model.model_utils import write_to_file, get_mlp_invertible



# class NodeEdgeCycle(torch.nn.Module):
    # def __init__(self,
    #              eps: float = 0.0,
    #              train_eps: bool = True,
    #              node_hidden_dim: int = 128,
    #              node_dense_dim: int = 128,
    #              edge_hidden_dim: int = 128,
    #              edge_dense_dim: int = 128,
    #              included_cycles: typing.List[int] = [5, 6],
    #              inversed_order: bool = False,
    #              activation: str = "ReLU",
    #              skip_connection: bool = False):
    #     super(NodeEdgeCycle, self).__init__()

#         self.node_edge = NodeEdge(node_hidden_dim=node_hidden_dim,
#                                   node_dense_dim=node_dense_dim,
#                                   edge_hidden_dim=edge_hidden_dim,
#                                   edge_dense_dim=edge_dense_dim,
#                                   eps=eps,
#                                   train_eps=train_eps)
#         self.edge_cycle = EdgeCycle(hidden_dim=edge_hidden_dim, dense_dim=edge_dense_dim, eps=eps, train_eps=train_eps, included_cycles=included_cycles)

#         print("initializing a model with included cycles:", included_cycles) 

#         self.combine_edge_mlp = get_mlp_invertible(2 * edge_hidden_dim, edge_dense_dim, edge_hidden_dim, 2, inverse=inversed_order, activation=activation)
#         self.skip_connection = skip_connection
        
        

#     def forward(self, data, G, x, edge_attr, cycle_attr):
#         node_out, edge_attr_1 = self.node_edge(data, x, G, edge_attr)
#         edge_attr_2, cycle_out = self.edge_cycle(data, edge_attr, G, cycle_attr)
#         edge_attr_torch =  self.combine_edge_mlp(torch.cat([edge_attr_1.torch(), edge_attr_2.torch()], dim=-1))
#         edge_out = p.batched_subgraphlayer0b.like(edge_attr, edge_attr_torch)

#         if self.skip_connection:
#             edge_out = edge_out + edge_attr
#             node_out = node_out + x
#             cycle_out = cycle_out + cycle_attr

        
#         return node_out, edge_out, cycle_out


class NodeEdgeCycle(torch.nn.Module):
    def __init__(self,
                 eps: float = 0.0,
                 train_eps: bool = True,
                 node_hidden_dim: int = 128,
                 node_dense_dim: int = 128,
                 edge_hidden_dim: int = 128,
                 edge_dense_dim: int = 128,
                 cycle_hidden_dim: int = 128,
                 cycle_dense_dim: int = 128,
                 included_cycles: typing.List[int] = [5, 6],
                 inversed_order: bool = False,
                 activation: str = "ReLU",
                 skip_connection: bool = False):
        super(NodeEdgeCycle, self).__init__()
        # self.edge_node = Edge_node(rep_dim)
        # self.edge_node = Edge_node(node_hidden_dim)
        self.edge_node = Edge_node(
            node_hidden_dim)
        # self.edge_cycle = Edge_cycle(node_hidden_dim, included_cycles)
        self.edge_cycle = EdgeCycle(edge_hidden_dim=edge_hidden_dim, edge_dense_dim=edge_dense_dim, cycle_hidden_dim=cycle_hidden_dim, cycle_dense_dim=cycle_dense_dim, eps=eps, train_eps=train_eps, included_cycles=included_cycles)
        
        self.edge_mlp = get_mlp_invertible(node_hidden_dim + edge_hidden_dim, edge_dense_dim, edge_hidden_dim,2)
        print("under updated model, the included cycles are:", included_cycles)

    def forward(self, data, G, node_rep, edge_rep,cycle_rep):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out1 = self.edge_node(node_rep,edge_rep,G, data)
        edge_out2, cycle_out = self.edge_cycle(data, edge_rep, G, cycle_rep)
        
        #edge_out1, edge_out2 has dim = rep_dim; edge_out has dim = 2 * rep_dim 
        edge_out = self.edge_mlp(torch.cat([edge_out1.torch(),edge_out2.torch()],dim=-1))
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        #residual connection

        node_out = node_rep + node_out
        edge_out = edge_rep + edge_out
        cycle_out = cycle_rep + cycle_out
        return node_out, edge_out, cycle_out







# from ..model_utils import get_mlp
# _inner_mlp_mult = 2



# class Edge_node(torch.nn.Module):
#     '''Edge and edge + node message passing'''
#     def __init__(self, rep_dim) -> None:
#         super().__init__()

#         self.node_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
#         self.edge_mlp_0 = get_mlp(3 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
        
#         self.node=p.subgraph.trivial()
#         self.edge=p.subgraph.edge()

#     def forward(self,node_rep, edge_rep,G,data):
#         # print("----------------starting MPNN---------------")
#         node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep,G,self.edge) #nc

#         edge_out = self.edge_mlp_0(torch.cat([edge_rep.torch(),node2edge.torch()],dim=-1)) #nc * 3 -> nc
#         edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)
        
#         edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge_out,G,self.node) # nc

#         node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],dim = -1)) #nc * 2 -> nc
#         node_out = p.batched_subgraphlayer0b.like(node_rep,node_out)
#         return node_out, edge_out