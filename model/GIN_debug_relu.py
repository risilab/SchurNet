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

from torch_geometric.nn import global_mean_pool

from torch_geometric.nn import GINConv, GINEConv


# import files from previous directory
import sys
sys.path.append("..")
# from utils import global_mean_pool



class GINConv_ptens(torch.nn.Module):

    def __init__(self,
                 nn: Callable[[Tensor], Tensor],
                 eps: float = 0.0,
                 train_eps: bool = True):
        super(GINConv_ptens, self).__init__()
        self.nn = nn
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([[eps]]), requires_grad=True)
        else:
            self.eps = torch.nn.Parameter(torch.Tensor([[eps]]))
        self.edge = p.subgraph.edge()
        self.node = p.subgraph.trivial()

    def forward(self, data, x, G, edge_attr=None):
        '''
        If edge weight is not set to None, we will do GINEconv instead of GINConv
        '''

        # message phase:

        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(
            x, G, self.edge)
        if edge_attr is not None:
            edge2edge = p.batched_subgraphlayer0b.gather_from_ptensors(edge_attr, G, self.edge)

            node2edge = node2edge + edge2edge
            node2edge = node2edge.relu(0)
            
        neighbor = p.batched_subgraphlayer0b.gather_from_ptensors(node2edge, G, self.node) 

        # remove the self loop in neighbor
        degree_vector = data.degree.view(-1, 1).float()
        neighbor = neighbor.torch() - x.torch() * degree_vector


        out = (1 + self.eps) * x.torch() + neighbor

        return p.batched_subgraphlayer0b.like(x, self.nn(out))




class GIN(torch.nn.Module):

    def __init__(self, dataset, num_layers: int=4, hidden_dim:int =128, dense_dim: int=128, dropout_rate=0.1, ptens_conv: bool = True, out_dim: int = 1, include_edge_features: bool = True):
        super().__init__()
        ## Initialization Step

        # embed the nodes into hidden_dim dimension
        self.node_embedding = torch.nn.Embedding(dataset.num_node_attr, hidden_dim)
        if hasattr(dataset, 'num_edge_features') and include_edge_features:
            self.edge_embedding = torch.nn.Embedding(dataset.num_edge_attr,
                                                     hidden_dim)
        GINConv_geometric = GINConv if dataset.num_edge_attr == 0 else GINEConv

        self.GINConv = GINConv_ptens if ptens_conv else GINConv_geometric
        self.initialization = self.GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
        ),
                                            eps=0.01,
                                            train_eps=True)
        ## Aggregation Layers
        self.mp_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.mp_layers.append(
                self.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, dense_dim),
                    torch.nn.BatchNorm1d(dense_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(dense_dim, hidden_dim),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(),
                ),
                              eps=0.01,
                              train_eps=True))
        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_dim, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
        )
        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(dense_dim, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
        )
        self.final_lin = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.include_edge_features = include_edge_features

    def forward(self, data):
        if self.GINConv == GINConv_ptens:
            return self.forward_ptens(data)
        elif self.GINConv == GINConv or self.GINConv == GINEConv:
            return self.forward_torch(data)
        else:
            raise ValueError("GINConv not found")
    
    
    def forward_torch(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and self.include_edge_features else None
        print("edge attr is None: ", edge_attr is None)


        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr) if edge_attr is not None else None


        x = self.initialization(x, edge_index, edge_attr=edge_attr)
        for conv in self.mp_layers:
            x = conv(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        ## Classification Head

        # check lin1 device
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return self.final_lin(x)
        
        # return F.log_softmax(x, dim=-1) 

    def forward_ptens(self, data):
        x, _, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and self.include_edge_features else None

        

        graphid_list = data.idx.tolist()
        G = p.batched_ggraph.from_cache(graphid_list)


        # x = self.node_embedding(x).to(x.device)
        x = self.node_embedding(x)
        if self.include_edge_features:
            edge_attr = self.edge_embedding(edge_attr)

        x = p.batched_subgraphlayer0b.from_vertex_features(graphid_list, x)
        if self.include_edge_features:
            edge_attr = p.batched_subgraphlayer0b.from_edge_features(graphid_list, edge_attr)

        x = self.initialization(data, x, G, edge_attr=edge_attr)

        for conv in self.mp_layers:
            x = conv(data, x, G, edge_attr=edge_attr)
        x = global_mean_pool(x.torch(), batch)
        ## Classification Head

        # check lin1 device
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        
        return self.final_lin(x)