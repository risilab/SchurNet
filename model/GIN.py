from typing import Callable, Optional, Union, List

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

# import files from previous directory
import sys

# from utils import global_mean_pool

from .layers.node_edge_ref_domain import NodeEdgeRefDomain

import os
from model.model_utils import write_to_file, reference_domain_with_batch, reorder_ptensors
from torch_geometric.utils import coalesce
from torch_geometric.utils import unbatch_edge_index

save_dir = "test_ref_domain"
os.makedirs(save_dir, exist_ok=True)


class GIN(torch.nn.Module):

    def __init__(self,
                 dataset,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 dense_dim: int = 128,
                 dropout_rate=0.1,
                 ptens_conv: bool = True,
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
        GINConv_geometric = GINConv if dataset.num_edge_attr == 0 or not include_edge_features else GINEConv
        self.GINConv = NodeEdgeRefDomain if ptens_conv else GINConv_geometric
        # torch activation function based on the string

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
                                               train_eps=True)
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
                                               train_eps=True)
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
                                 train_eps=True))
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
                                 train_eps=True))
        if not inversed_order:
            self.lin1 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, dense_dim, False),
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
                torch.nn.Linear(hidden_dim, dense_dim, False),
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

    def forward(self, data):
        if self.GINConv == NodeEdgeRefDomain:
            return self.forward_ptens(data)
        elif self.GINConv == GINConv or self.GINConv == GINEConv:
            return self.forward_torch(data)
        else:
            raise ValueError("GINConv not found")

    def forward_torch(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(
            data, "edge_attr") and self.include_edge_features else None

        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(
            edge_attr) if edge_attr is not None else None

        if self.include_edge_features:
            x = self.initialization(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.initialization(x, edge_index)
        for conv in self.mp_layers:
            if self.include_edge_features:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
        x = self.readout(x, batch)
        ## Classification Head

        # check lin1 device
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return self.final_lin(x)

        # return F.log_softmax(x, dim=-1)

    def forward_ptens(self, data):
        x, _, batch = data.x, data.edge_index, data.batch
        if self.include_edge_features and hasattr(data, "edge_attr"):
            edge_attr = data.edge_attr
            # process edge attr to be aligned with the domain of edge2edge
        else:
            edge_attr = None

        graphid_list = data.idx.tolist()
        G = p.batched_ggraph.from_cache(graphid_list)

        # x = self.node_embedding(x).to(x.device)
        x = self.node_embedding(x)
        if self.include_edge_features:
            edge_attr = self.edge_embedding(edge_attr)

        x = p.batched_subgraphlayer0b.from_vertex_features(graphid_list, x)
        if self.include_edge_features:
            edge_attr = p.batched_subgraphlayer0b.from_edge_features(
                graphid_list, edge_attr)
            self.edge = p.subgraph.edge()
            edge2edge = p.batched_subgraphlayer0b.gather_from_ptensors(
                edge_attr, G, self.edge)


            edge_attr_reference_domain = edge_attr.get_atoms().torch()

            # get edge2edge reference domain
            edge2edge_reference_domain = edge2edge.get_atoms().torch()

            edge_attr_reference_domain = reference_domain_with_batch(edge_attr_reference_domain, device=data.edge_index.device)
            edge2edge_reference_domain = reference_domain_with_batch(edge2edge_reference_domain, device=data.edge_index.device)
            

            reordered_edge_attr = reorder_ptensors(edge_attr.torch(), edge_attr_reference_domain, edge2edge_reference_domain)

            edge_attr = p.batched_subgraphlayer0b.like(edge2edge,
                                                       reordered_edge_attr)

        x = self.initialization(data, x, G, edge_attr=edge_attr)

        for conv in self.mp_layers:
            x = conv(data, x, G, edge_attr=edge_attr)
        x = self.readout(x.torch(), batch)
        ## Classification Head

        # check lin1 device
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return self.final_lin(x)
