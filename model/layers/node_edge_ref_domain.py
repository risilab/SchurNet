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

from model.model_utils import write_to_file
save_dir = "test_node_edge_ref_domain"
os.makedirs(save_dir, exist_ok=True)



class NodeEdgeRefDomain(torch.nn.Module):

    def __init__(self,
                 nn: Callable[[Tensor], Tensor],
                 eps: float = 0.0,
                 train_eps: bool = True,
                 hidden_dim: int = 128,
                 dense_dim: int = 128):
        super(NodeEdgeRefDomain, self).__init__()
        self.node_mlp = nn
        # self.edge_mlp_1 = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim * 2, dense_dim),
        #     torch.nn.BatchNorm1d(dense_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(dense_dim, hidden_dim),
        #     torch.nn.BatchNorm1d(hidden_dim),
        #     torch.nn.ReLU(),
        # )
        # self.edge_mlp_2 = copy.deepcopy(nn)

        if train_eps:
            self.eps_node = torch.nn.Parameter(torch.Tensor([[eps]]),
                                               requires_grad=True)
            self.eps_edge = torch.nn.Parameter(torch.Tensor([[eps]]),
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




        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(
            x, G, self.edge)

        

        if edge_attr is not None:
            edge_1 = p.batched_subgraphlayer1b.gather_from_ptensors(edge_attr, G, self.edge)

            node2edge_first_order = p.batched_subgraphlayer1b.gather_from_ptensors(x, G, self.edge)
            a,b = node2edge_first_order.torch().shape        
            assert a % 2 == 0
            swapped = node2edge_first_order.torch().reshape(-1, 2, b)[:, [1,0], :].reshape(-1, b)

            node2edge_first_order_swapped = p.batched_subgraphlayer1b.like(node2edge_first_order, swapped)
            # repeat edge_attr twice:
            edge_attr_1_torch = edge_attr.torch().repeat_interleave(2, dim=0).to(node2edge_first_order.torch().device)



            edge_attr_1 = p.batched_subgraphlayer1b.like(edge_1, edge_attr_1_torch)


            sum_result = node2edge_first_order_swapped + edge_attr_1
            node2edge = sum_result.relu(0)

        neighbor = p.batched_subgraphlayer0b.gather_from_ptensors(
            node2edge, G, self.node)

        # remove the self loop in neighbor
        # degree_vector = data.degree.view(-1, 1).float()
        # neighbor = neighbor.torch() - x.torch(
        # ) * degree_vector  # TODO: think about a way to reduce the neighbor effect

        node_out = self.node_mlp((1 + self.eps_node) * x.torch() + neighbor.torch())

        node_out = p.batched_subgraphlayer0b.like(x, node_out)


        return node_out