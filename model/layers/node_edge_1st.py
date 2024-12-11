import torch

from typing import Callable, Optional, Union

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

import copy

# import files from previous directory
import sys
import os
from ..model_utils import get_mlp_invertible


class NodeEdge(torch.nn.Module):

    def __init__(self,
                 edge_hidden_dim: int = 128,
                 node_hidden_dim: int = 128,
                 edge_dense_dim: int = 128,
                 node_dense_dim: int = 128,
                 eps: float = 0.0,
                 train_eps: bool = True):
        super(NodeEdge, self).__init__()
        self.node_mlp_1 = get_mlp_invertible(node_hidden_dim, node_dense_dim, node_hidden_dim, 2) 

        self.node_mlp_2 = get_mlp_invertible(edge_hidden_dim, node_dense_dim, node_hidden_dim, 2) 

        self.edge_mlp_1 = get_mlp_invertible(edge_hidden_dim + node_hidden_dim, edge_dense_dim + node_hidden_dim, edge_hidden_dim, 2)
        self.edge_mlp_2 = get_mlp_invertible(edge_hidden_dim, edge_dense_dim, edge_hidden_dim, 2)

        if train_eps:
            self.eps_node = torch.nn.Parameter(torch.Tensor([[0.]]),
                                               requires_grad=True)
            self.eps_edge = torch.nn.Parameter(torch.Tensor([[0.]]),
                                               requires_grad=True)
        else:
            self.eps_node = torch.nn.Parameter(torch.Tensor([[eps]]))
            self.eps_edge = torch.nn.Parameter(torch.Tensor([[eps]]))

        self.edge = p.subgraph.edge()
        self.node = p.subgraph.trivial()

    def forward(self, data, x, G, edge_attr=None):
        '''
        If edge weight is not set to None, we will do GINEconv instead of GINConv
        '''

        # message passing on vertices

        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(
            x, G, self.edge)
        if edge_attr is not None:

            edge_out = torch.cat(
                [edge_attr.torch(), node2edge.torch()], dim=-1)
            # print("edge out shape:", edge_out.shape)
            edge_out = self.edge_mlp_1(edge_out)
            # check device for edge out
            edge_out = self.edge_mlp_2((1 + self.eps_edge) * edge_attr.torch() +
                                       edge_out)

            edge_out = p.batched_subgraphlayer1b.like(edge_attr, edge_out)
            # print("edge out final shape:", edge_out.torch().shape)
            # write_to_file(edge_out, os.path.join(save_dir, "edge_out.txt"))

            node2edge = edge_out
        neighbor = p.batched_subgraphlayer0b.gather_from_ptensors(
            node2edge, G, self.node)
        neighbor = self.node_mlp_2(neighbor.torch())

        # remove the self loop in neighbor
        degree_vector = data.degree.view(-1, 1).float()
        neighbor = neighbor - x.torch(
        ) * degree_vector  # TODO: think about a way to reduce the neighbor effect

        node_out = self.node_mlp_1((1 + self.eps_node) * x.torch() + neighbor)

        node_out = p.batched_subgraphlayer0b.like(x, node_out)

        # now, consider message passing on edges

        if edge_attr is not None:
            return node_out, edge_out
        else:
            return node_out