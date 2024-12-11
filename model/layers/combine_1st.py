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

from .node_edge_1st import NodeEdge
from .edge_cycle_1st import EdgeCycle

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
                 included_cycles: typing.List[int] = [5, 6],
                 inversed_order: bool = False,
                 activation: str = "ReLU",
                 skip_connection: bool = False):
        super(NodeEdgeCycle, self).__init__()

        self.node_edge = NodeEdge(node_hidden_dim=node_hidden_dim,
                                  node_dense_dim=node_dense_dim,
                                  edge_hidden_dim=edge_hidden_dim,
                                  edge_dense_dim=edge_dense_dim,
                                  eps=eps,
                                  train_eps=train_eps)
        self.edge_cycle = EdgeCycle(hidden_dim=edge_hidden_dim, dense_dim=edge_dense_dim, eps=eps, train_eps=train_eps, included_cycles=included_cycles)

        print("initializing a model with included cycles:", included_cycles) 

        self.combine_edge_mlp = get_mlp_invertible(2 * edge_hidden_dim, edge_dense_dim, edge_hidden_dim, 2, inverse=inversed_order, activation=activation)
        self.skip_connection = skip_connection
        
        

    def forward(self, data, G, x, edge_attr, cycle_attr):
        node_out, edge_attr_1 = self.node_edge(data, x, G, edge_attr)
        edge_attr_2, cycle_out = self.edge_cycle(data, edge_attr, G, cycle_attr)
        edge_attr_torch =  self.combine_edge_mlp(torch.cat([edge_attr_1.torch(), edge_attr_2.torch()], dim=-1))
        edge_out = p.batched_subgraphlayer1b.like(edge_attr, edge_attr_torch)

        if self.skip_connection:
            edge_out = edge_out + edge_attr
            node_out = node_out + x
            cycle_out = cycle_out + cycle_attr

        
        return node_out, edge_out, cycle_out