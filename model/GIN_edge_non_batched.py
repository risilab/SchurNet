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

from .layers.node_edge_non_batched import NodeEdge
import copy

# import files from previous directory
import sys
import os

sys.path.append("..")
# from utils import global_mean_pool

# save_dir = "test_GIN_edge"
# os.makedirs(save_dir, exist_ok=True)
# from model.model_utils import write_to_file




class GIN_edge(torch.nn.Module):

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
                 inversed_order: bool = False,
                 activation: str = "ReLU"):
        super().__init__()
        ## Initialization Step

        # embed the nodes into hidden_dim dimension
        self.node_embedding = torch.nn.Embedding(dataset.num_node_attr,
                                                 hidden_dim)
        if hasattr(dataset, 'num_edge_features') and include_edge_features:
            self.edge_embedding = torch.nn.Embedding(dataset.num_edge_attr,
                                                     hidden_dim)

        self.GINConv = NodeEdge

        if not inversed_order:
            self.initialization = self.GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, dense_dim),
                torch.nn.BatchNorm1d(dense_dim),
                getattr(torch.nn, activation)(),
                torch.nn.Linear(dense_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                getattr(torch.nn, activation)(),
            ),
                                               eps=eps,
                                               train_eps=True,
                                               hidden_dim=hidden_dim,
                                               dense_dim=dense_dim)
        else:
            self.initialization = self.GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, dense_dim),
                getattr(torch.nn, activation)(),
                torch.nn.BatchNorm1d(dense_dim),
                torch.nn.Linear(dense_dim, hidden_dim),
                getattr(torch.nn, activation)(),
                torch.nn.BatchNorm1d(hidden_dim),
            ),
                                               eps=eps,
                                               train_eps=True,
                                               hidden_dim=hidden_dim,
                                               dense_dim=dense_dim)
        ## Aggregation Layers
        self.mp_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if not inversed_order:
                self.mp_layers.append(
                    self.GINConv(torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, dense_dim, False),
                        torch.nn.BatchNorm1d(dense_dim),
                        getattr(torch.nn, activation)(),
                        torch.nn.Linear(dense_dim, hidden_dim, False),
                        torch.nn.BatchNorm1d(hidden_dim),
                        getattr(torch.nn, activation)(),
                    ),
                                 eps=eps,
                                 train_eps=True,
                                 hidden_dim=hidden_dim,
                                 dense_dim=dense_dim))
            else:
                self.mp_layers.append(
                    self.GINConv(torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, dense_dim, False),
                        getattr(torch.nn, activation)(),
                        torch.nn.BatchNorm1d(dense_dim),
                        torch.nn.Linear(dense_dim, hidden_dim, False),
                        getattr(torch.nn, activation)(),
                        torch.nn.BatchNorm1d(hidden_dim),
                    ),
                                 eps=eps,
                                 train_eps=True,
                                 hidden_dim=hidden_dim,
                                 dense_dim=dense_dim))
        _edge_mlp_mult = 2 if include_edge_features else 1
        if not inversed_order:
            self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * _edge_mlp_mult, dense_dim, False),
                torch.nn.BatchNorm1d(dense_dim),
                getattr(torch.nn, activation)(),
                torch.nn.Linear(dense_dim, dense_dim, False),
                torch.nn.BatchNorm1d(dense_dim),
                getattr(torch.nn, activation)(),
            )
            self.lin2 = torch.nn.Sequential(
                torch.nn.Linear(dense_dim, dense_dim, False),
                torch.nn.BatchNorm1d(dense_dim),
                getattr(torch.nn, activation)(),
                torch.nn.Linear(dense_dim, hidden_dim, False),
                torch.nn.BatchNorm1d(hidden_dim),
                getattr(torch.nn, activation)(),
            )
        else:
            self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * _edge_mlp_mult, dense_dim, False),
                getattr(torch.nn, activation)(),
                torch.nn.BatchNorm1d(dense_dim),
                torch.nn.Linear(dense_dim, dense_dim, False),
                getattr(torch.nn, activation)(),
                torch.nn.BatchNorm1d(dense_dim),
            )
            self.lin2 = torch.nn.Sequential(
                torch.nn.Linear(dense_dim, dense_dim, False),
                getattr(torch.nn, activation)(),
                torch.nn.BatchNorm1d(dense_dim),
                torch.nn.Linear(dense_dim, hidden_dim, False),
                getattr(torch.nn, activation)(),
                torch.nn.BatchNorm1d(hidden_dim),
            )

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

        # graphid_list = data.idx.tolist()
        # G = p.batched_ggraph.from_cache(graphid_list)
        G=p.ggraph.from_edge_index(data.edge_index.float().to('cpu'),data.num_nodes)

        edges=data.edge_index.transpose(1,0).tolist()


        x = self.node_embedding(x)
        if self.include_edge_features:
            edge_attr = self.edge_embedding(edge_attr)

        # x = p.subgraphlayer0b.from_vertex_features(graphid_list, x)
        x = p.ptensors0b.from_matrix(x)
        if self.include_edge_features:
            # edge_attr = p.subgraphlayer0b.from_edge_features(
                # graphid_list, edge_attr)
            edge_attr = p.ptensors0b.from_matrix(edge_attr, edges)
            edge_attr = p.subgraphlayer0b.gather_from_ptensors(edge_attr, G, self.edge)

        if edge_attr is not None and self.include_edge_features:
            x, edge_attr = self.initialization(data, x, G, edge_attr=edge_attr)
        else:
            x = self.initialization(data, x, G, edge_attr=edge_attr)

        for conv in self.mp_layers:
            if edge_attr is not None and self.include_edge_features:
                x, edge_attr = conv(data, x, G, edge_attr=edge_attr)
            else:
                x = conv(x, G, edge_attr=edge_attr)
        if edge_attr is not None and self.include_edge_features:
            edge2node = p.subgraphlayer0b.gather_from_ptensors(
                edge_attr, G, self.node)
            x = torch.cat([x.torch(), edge2node.torch()], dim=-1)
            x = self.readout(x, batch) # global_add_pool
        else:
            x = self.readout(x.torch(), batch)
        ## Classification Head

        # check lin1 device
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return self.final_lin(x)
