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
import os
# from utils import global_mean_pool
def write_to_file(data, filename):
    with open(filename, 'w') as f:
        print(data, file=f)
save_dir = "test_linmaps"
os.makedirs(save_dir, exist_ok=True)


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

        
        write_to_file(x, os.path.join(save_dir, "x.txt"))
        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(
            x, G, self.edge)
        write_to_file(node2edge, os.path.join(save_dir, "node2edge.txt"))
        
        
        node2edge_1 = p.batched_subgraphlayer1b.gather_from_ptensors(x, G, self.edge)
        write_to_file(node2edge_1, os.path.join(save_dir, "node2edge_1.txt"))

        linmap_result = p.batched_subgraphlayer1b.linmaps(node2edge_1)
        write_to_file(linmap_result, os.path.join(save_dir, "linmap_result.txt"))

        
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




class GIN_test(torch.nn.Module):

    def __init__(self, dataset, num_layers: int=4, hidden_dim:int =128, dense_dim: int=128, dropout_rate=0.1, ptens_conv: bool = True, out_dim: int = 1, include_edge_features: bool = False):
        super().__init__()
        ## Initialization Step

        # embed the nodes into hidden_dim dimension
        self.node_embedding = torch.nn.Embedding(dataset.num_node_attr, hidden_dim)
        if hasattr(dataset, 'num_edge_features') and include_edge_features:
            self.edge_embedding = torch.nn.Embedding(dataset.num_edge_attr,
                                                     hidden_dim)
        GINConv_geometric = GINConv if dataset.num_edge_attr == 0 or not include_edge_features else GINEConv
        self.GINConv_ptens = GINConv_ptens 
        self.GINConv_geometric = GINConv_geometric
        self.initialization_ptens = self.GINConv_ptens(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, dense_dim),
            torch.nn.BatchNorm1d(dense_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
        ),
                                            eps=0.01,
                                            train_eps=True)
        self.initialization_geometric = self.GINConv_geometric(torch.nn.Sequential(
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
        # self.mp_layers = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     self.mp_layers.append(
        #         self.GINConv(torch.nn.Sequential(
        #             torch.nn.Linear(hidden_dim, dense_dim),
        #             torch.nn.BatchNorm1d(dense_dim),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(dense_dim, hidden_dim),
        #             torch.nn.BatchNorm1d(hidden_dim),
        #             torch.nn.ReLU(),
        #         ),
        #                       eps=0.01,
        #                       train_eps=True))
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and self.include_edge_features else None

        graphid_list = data.idx.tolist()
        G = p.batched_ggraph.from_cache(graphid_list)

        x = self.node_embedding(x)

        import copy
        x_ptens = torch.tensor(x)
        x_geometric = torch.tensor(x)

        x_ptens = p.batched_subgraphlayer0b.from_vertex_features(graphid_list, x_ptens) 
        # ---geometric---

        x_ptens = self.initialization_ptens(data, x_ptens, G, edge_attr=edge_attr)
        x_geometric = self.initialization_geometric(x_geometric, edge_index)
        assert torch.allclose(x_ptens.torch(), x_geometric), "initialization not equal"
        x = global_mean_pool(x_ptens, batch)
        ## Classification Head

        # check lin1 device
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        # ---geometric---
        return self.final_lin(x)
        
        # return F.log_softmax(x, dim=-1) 
