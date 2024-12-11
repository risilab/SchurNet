from typing import Callable, Optional, Union
import typing

import torch
from torch.nn import functional as F
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

from .autobahn.edge_cycle_autobahn import EdgeCycleAutobahn
from .autobahn.edge_cycle_autobahn_on_linmap import EdgeCycleAutobahnOnLinmap
from .autobahn.edge_cycle_autobahn_lvl import EdgeCycleAutobahnLvl

import copy

# import files from previous directory
import sys
import os

sys.path.append("..")
# from utils import global_mean_pool

# save_dir = "test_GIN_edge"
# os.makedirs(save_dir, exist_ok=True)
from model.model_utils import write_to_file, get_mlp_invertible





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
                 autobahn_channels: int = 128,
                 cycle_linmap_autobahn: bool = False,
                 lvl_autobahn: bool = False,
                 included_cycles: typing.List[int] = [5, 6],
                 subgraphs: typing.Optional[typing.List[p.subgraph]] = None,
                 inversed_order: bool = False,
                 activation: str = "ReLU",
                 skip_connection: bool = False,
                 dropout: float = 0.0,
                 bn_momentum: float = 0.1,
                 bn_eps: float = 1e-5):
        super(NodeEdgeCycle, self).__init__()
        # self.edge_node = Edge_node(rep_dim)
        # self.edge_node = Edge_node(node_hidden_dim)
        self.edge_node = Edge_node(
            node_hidden_dim)
        # self.edge_cycle = Edge_cycle(node_hidden_dim, included_cycles)
        print("subgraphs:", subgraphs)
        edge_cycle_conv = EdgeCycleAutobahn
        if cycle_linmap_autobahn:
            edge_cycle_conv = EdgeCycleAutobahnOnLinmap

        if lvl_autobahn:
            edge_cycle_conv = EdgeCycleAutobahnLvl

        self.edge_cycle = edge_cycle_conv(edge_hidden_dim=edge_hidden_dim,
                                          edge_dense_dim=edge_dense_dim,
                                          cycle_hidden_dim=cycle_hidden_dim,
                                          cycle_dense_dim=cycle_dense_dim,
                                          autobahn_channels=autobahn_channels,
                                          eps=eps,
                                          train_eps=train_eps,
                                          included_cycles=included_cycles,
                                          subgraphs=subgraphs,
                                          bn_eps=bn_eps,
                                          bn_momentum=bn_momentum)

        self.edge_mlp = get_mlp_invertible(node_hidden_dim + edge_hidden_dim, edge_dense_dim, edge_hidden_dim,2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        print("under updated model, the included cycles are:", included_cycles)

        self.dropout = dropout
        self.skip_connection = skip_connection

    def forward(self, data, G, node_rep, edge_rep,cycle_rep):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out1 = self.edge_node(node_rep,edge_rep,G, data)
        edge_out2, cycle_out = self.edge_cycle(data, edge_rep, G, cycle_rep)

        edge_out = self.edge_mlp(torch.cat([edge_out1,edge_out2],dim=-1))

        node_out = F.dropout(node_out, p=self.dropout, training=self.training)
        cycle_out = F.dropout(cycle_out, p=self.dropout, training=self.training)

        edge_out = F.dropout(edge_out, p=self.dropout, training=self.training)

        edge_out = p.batched_subgraphlayer0b.like(edge_rep, edge_out)

        cycle_out = p.batched_ptensors1b.like(cycle_rep, cycle_out)
        #edge_out1, edge_out2 has dim = rep_dim; edge_out has dim = 2 * rep_dim
        node_out = p.batched_subgraphlayer0b.like(node_rep, node_out)

        #residual connection

        if self.skip_connection:
            edge_out = edge_out + edge_rep
            node_out = node_out + node_rep
            cycle_out = cycle_out + cycle_rep
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
