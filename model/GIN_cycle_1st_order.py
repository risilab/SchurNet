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

# from .layers.node_edge import NodeEdge
# from .layers.edge_cycle import EdgeCycle
from .layers.combine_1st import NodeEdgeCycle

import copy

# import files from previous directory
import sys
import os

sys.path.append("..")
# from utils import global_mean_pool

# save_dir = "test_GIN_edge"
# os.makedirs(save_dir, exist_ok=True)
from model.model_utils import write_to_file, get_mlp_invertible

savedir = "test_GIN_cycle_new"
os.makedirs(savedir, exist_ok=True)


class GIN_cycle(torch.nn.Module):

    def __init__(self,
                 dataset,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 dense_dim: int = 128,
                 dropout_rate=0.1,
                 out_dim: int = 1,
                 include_edge_features: bool = True,
                 eps: float = 0,
                 readout: Callable = global_add_pool,
                 included_cycles: typing.List = [5, 6],
                 inversed_order: bool = False,
                 activation: str = "ReLU"):
        super(GIN_cycle, self).__init__()
        ## Initialization Step

        # embed the nodes into hidden_dim dimension
        self.node_hidden_dim = hidden_dim
        self.edge_hidden_dim = hidden_dim * 2
        self.node_dense_dim = dense_dim
        self.edge_dense_dim = dense_dim * 3
        self.node_embedding = torch.nn.Embedding(dataset.num_node_attr,
                                                 self.node_hidden_dim)
        if hasattr(dataset, 'num_edge_features') and include_edge_features:
            self.edge_embedding = torch.nn.Embedding(dataset.num_edge_attr,
                                                     self.edge_hidden_dim)

        self.included_cycles = included_cycles

        self.cycles = [p.subgraph.cycle(i) for i in self.included_cycles]

        self.cycle_mlp = get_mlp_invertible(hidden_dim * 2, dense_dim, hidden_dim * 2, num_layers=3, inverse=inversed_order, activation=activation)

        self.num_cycles = len(self.included_cycles)

        self.num_layers = num_layers

        # self.initialization = NodeEdge(            node_hidden_dim=self.node_hidden_dim,
        #     node_dense_dim=self.node_dense_dim,
        #     edge_hidden_dim=self.edge_hidden_dim,
        #     edge_dense_dim=self.edge_dense_dim,
        #                                eps=eps,
        #                                train_eps=True)
        # self.initialization_cycles = EdgeCycle(                                               eps=eps,
        #                                        train_eps=True,
        #                                        hidden_dim=self.edge_hidden_dim,
        #                                        dense_dim=self.edge_dense_dim,
        #                                        included_cycles=self.included_cycles)
        self.initialization = NodeEdgeCycle(eps=eps, train_eps=True, node_hidden_dim=self.node_hidden_dim, node_dense_dim=self.node_dense_dim, edge_hidden_dim=self.edge_hidden_dim, edge_dense_dim=self.edge_dense_dim, included_cycles=self.included_cycles, inversed_order=inversed_order, activation=activation)
        ## Aggregation Layers

        self.conv_layers = torch.nn.ModuleList(
            [NodeEdgeCycle(eps=eps, train_eps=True, node_hidden_dim=self.node_hidden_dim, node_dense_dim=self.node_dense_dim, edge_hidden_dim=self.edge_hidden_dim, edge_dense_dim=self.edge_dense_dim, included_cycles=self.included_cycles, inversed_order=inversed_order, activation=activation) for _ in range(num_layers)])

        _edge_mlp_mult = 1 if include_edge_features else 0
        _cycle_mlp_mult = 1 if included_cycles else 0
        self.lin1 = get_mlp_invertible(in_dim=(_edge_mlp_mult + _cycle_mlp_mult) * self.edge_hidden_dim + self.node_hidden_dim,
                                       hid_dim=dense_dim,
                                       out_dim=dense_dim,
                                       num_layers=2,
                                       inverse=inversed_order,
                                       activation=activation)
        
        self.lin2 = get_mlp_invertible(dense_dim, dense_dim, hidden_dim, num_layers=2, inverse=inversed_order, activation=activation)        

        self.final_lin = torch.nn.Linear(hidden_dim, out_dim, False)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.include_edge_features = include_edge_features
        self.readout = readout

        self.node = p.subgraph.trivial()
        self.edge = p.subgraph.edge()

    def forward(self, data):
        x, _, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(
            data, "edge_attr") and self.include_edge_features else None

        graphid_list = data.idx.tolist()
        G = p.batched_ggraph.from_cache(graphid_list)

        # x = self.node_embedding(x).to(x.device)
        x = self.node_embedding(x)
        if self.include_edge_features:
            edge_attr = self.edge_embedding(edge_attr)

        x = p.batched_subgraphlayer0b.from_vertex_features(graphid_list, x)
        if self.include_edge_features:
            # how about 0th order?
            edge_attr_0 = p.batched_subgraphlayer0b.from_edge_features(
                graphid_list, edge_attr)
            
            edge_attr_0 = p.batched_subgraphlayer0b.gather_from_ptensors(
                edge_attr_0, G, self.edge, min_overlaps=2) # correct the edge attr reference domain
            
            edge_attr = p.batched_subgraphlayer0b.from_edge_features(
                graphid_list, edge_attr)
            
            edge_attr = p.batched_subgraphlayer1b.gather_from_ptensors(
                edge_attr, G, self.edge, min_overlaps=2) # correct the edge attr reference domain
            

        cycle_rep = [
            p.batched_subgraphlayer1b.gather_from_ptensors(x, G, cycle)
            for cycle in self.cycles
        ]
        cycle_rep = [
            p.batched_subgraphlayer1b.gather_from_ptensors(cycle_ptens, G, cycle, min_overlaps=3)
            for cycle_ptens, cycle in zip(cycle_rep,self.cycles)
        ]
        cycle_rep = p.batched_subgraphlayer1b.cat(*cycle_rep)
        cycle_rep_mlp = self.cycle_mlp(cycle_rep.torch())
        cycle_rep = p.batched_ptensors1b.like(cycle_rep, cycle_rep_mlp)

        # check out the shape of the cycles:

        x, edge_attr, cycle_rep = self.initialization(data, G, x, edge_attr, cycle_rep)

        for conv in self.conv_layers:
            x, edge_attr, cycle_rep = conv(data, G, x, edge_attr, cycle_rep)
            


        if edge_attr is not None and self.include_edge_features:
            edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(
                edge_attr, G, self.node)

            cycle2node = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep, G, self.node)

            x = torch.cat([x.torch(), edge2node.torch()], dim=-1)
            x = torch.cat([x, cycle2node.torch()], dim=-1)
            x = self.readout(x, batch)
        else:
            x = self.readout(x.torch(), batch)
        ## Classification Head

        # check lin1 device
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return self.final_lin(x)
