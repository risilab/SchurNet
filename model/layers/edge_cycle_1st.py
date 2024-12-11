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


import copy

# import files from previous directory
import sys
import os

sys.path.append("..")
# from utils import global_mean_pool

save_dir = "test_GIN_cycle_1st_order_edge"
os.makedirs(save_dir, exist_ok=True)
from ..model_utils import write_to_file, get_mlp_invertible
class EdgeCycle(torch.nn.Module):
    def __init__(self,
                 eps: float = 0.0,
                 train_eps: bool = True,
                 hidden_dim: int = 128,
                 dense_dim: int = 128,
                 included_cycles: typing.List[int] = [5, 6]):
        super(EdgeCycle, self).__init__()
        self.edge_mlp_1 = get_mlp_invertible(hidden_dim * 5, dense_dim,hidden_dim, num_layers=2, activation="ReLU", inverse=False)
        
        self.edge_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
        )
        self.cycle_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_dim, hidden_dim * 2),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.ReLU(),
        )
        self.cycle_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
        )

        if train_eps:
            self.eps_edge_1 = torch.nn.Parameter(torch.Tensor([[0.]]),
                                               requires_grad=True)
            self.eps_edge_2 = torch.nn.Parameter(torch.Tensor([[0.]]),
                                               requires_grad=True)
            self.eps_cycle = torch.nn.Parameter(torch.Tensor([[0.]]),
                                               requires_grad=True)
        else:
            self.eps_edge = torch.nn.Parameter(torch.Tensor([[eps]]))
            self.eps_cycle = torch.nn.Parameter(torch.Tensor([[eps]]))

        self.edge = p.subgraph.edge()
        self.cycles = [p.subgraph.cycle(i) for i in included_cycles]
        self.included_cycles = included_cycles

    def forward(self, data, edge_attr, G, cycle_attr=None):

        # print("edge_attr", edge_attr)


        edge2cycles_lst_1 = [p.batched_subgraphlayer1b.gather_from_ptensors(
            edge_attr, G, cycle, min_overlaps=1) for cycle in self.cycles] # completely covered by the cycle
        edge2cycles_lst_2 = [p.batched_subgraphlayer1b.gather_from_ptensors(
            edge_attr, G, cycle, min_overlaps=2) for cycle in self.cycles] # completely covered by the cycle

        
        edge2cycles_1 = p.batched_subgraphlayer1b.cat(*edge2cycles_lst_1)
        edge2cycles_2 = p.batched_subgraphlayer1b.cat(*edge2cycles_lst_2)
        cycle_linmap = p.batched_subgraphlayer1b.cat(*[p.batched_subgraphlayer1b.gather_from_ptensors(cycle_attr, G, cycle, min_overlaps=size) for size, cycle in zip(self.included_cycles, self.cycles)]) # TODO: maybe we need to just do linmaps here, because a linmap strictly forbids smoothing information from bigger cycles
        lift_aggr = p.batched_subgraphlayer1b.cat_channels(
            edge2cycles_2, edge2cycles_1)
        # cycle_linmap = p.batched_subgraphlayer1b.cat(*[p.batched_subgraphlayer1b.gather_from_ptensors(cycle_attr, G, cycle) for cycle in self.cycles])

        write_to_file(cycle_attr, os.path.join(save_dir, "cycle_attr.ptens"))
        write_to_file(cycle_linmap, os.path.join(save_dir, "cycle_linmap.ptens"))
        cycle_out = self.cycle_mlp_2((1 + self.eps_cycle) * cycle_linmap.torch() + self.cycle_mlp_1(lift_aggr.torch()))
        
        lvl_aggr_cycle_ptens = p.batched_subgraphlayer1b.cat_channels(
            lift_aggr, cycle_attr) # sum of mlps vs mlps of sum here?
        lvl_aggr_cycle_torch = self.edge_mlp_1(lvl_aggr_cycle_ptens.torch())
        

        lvl_aggr_cycle_ptens = p.batched_ptensors1b.like(lvl_aggr_cycle_ptens, lvl_aggr_cycle_torch)

        
        
        
        
        lvl_aggr_1 = p.batched_subgraphlayer1b.gather_from_ptensors(
            lvl_aggr_cycle_ptens, G, self.edge, min_overlaps=1) # TODO: verify the lvl aggr
        

        lvl_aggr_2 = p.batched_subgraphlayer1b.gather_from_ptensors(
            lvl_aggr_cycle_ptens, G, self.edge, min_overlaps=2)
        
        edge_linmap = p.batched_subgraphlayer1b.gather_from_ptensors(
            edge_attr, G, self.edge, min_overlaps=2)
        edge_out = self.edge_mlp_2((1 + self.eps_edge_1) * edge_linmap.torch() +
                                    (1 + self.eps_edge_2) * lvl_aggr_1.torch() + lvl_aggr_2.torch())
        

        edge_out = p.batched_subgraphlayer1b.like(edge_attr, edge_out)

        cycle_out = p.batched_ptensors1b.like(cycle_attr, cycle_out)

        return edge_out, cycle_out
