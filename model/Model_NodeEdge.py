import torch
import ptens as p
from torch.nn import functional as F, Sequential, ModuleList, Linear, BatchNorm1d, ReLU, Parameter
from typing import Callable, Union, List
from functools import reduce
from argparse import Namespace

from data_cleaning.feature_encoders import get_node_encoder, get_edge_encoder

from torch_geometric.nn import global_add_pool, global_mean_pool
from .model_utils import write_to_file

_inner_mlp_mult = 2

class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, hidden_dim) -> None:
        super().__init__()

        self.node_mlp = Sequential(
            Linear(2 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.edge_mlp = Sequential(
            Linear(3 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
   
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep, edge_rep,G):
        # print("----------------starting MPNN---------------")
        # write_to_file(node_rep,"output/node_rep_org.txt")
        node2edge = p.subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        # print("node2edge.shape: ",node2edge.torch().shape)
        # write_to_file(node2edge,"output/node2edge_org.txt")
        # node2edge = p.subgraphlayer1b.gather_from_ptensors(node2edge,G,self.edge) #nc * 2
        node2edge = p.subgraphlayer1b.linmaps(node2edge) #nc * 2
        # write_to_file(node2edge,"output/node2edge_after_gather_org.txt")

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),node2edge.torch()],-1)) #nc * 3 -> nc
        edge_out = p.subgraphlayer1b.like(edge_rep,edge_out)

        edge2node = p.subgraphlayer0b.gather_from_ptensors(edge_out,G,self.node)

        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],-1)) #nc * 2 -> nc
        node_out = p.ptensors0b.like(node_rep,node_out)
        return node_out, edge_out

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout) -> None:
        super().__init__()
        self.edge_node = Edge_node(hidden_dim)

    def forward(self, node_rep, edge_rep,G):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out = self.edge_node(node_rep,edge_rep,G)

        #residual connection
        # node_out = node_rep + node_out
        # edge_out = edge_rep + edge_out
        return node_out, edge_out


class Model_NodeEdge(torch.nn.Module):
    def __init__(self, hidden_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, ds_name,dataset,device = 'cuda') -> None:
        print("Running: Model_NodeEdge\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,hidden_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,hidden_dim)
        
        self.conv_layers = ModuleList([ConvLayer(hidden_dim,dropout) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(2 * hidden_dim,dense_dim,False),
                                    BatchNorm1d(dense_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim,False),
                                    BatchNorm1d(dense_dim),
                                    ReLU()
                                    )

        self.dropout = dropout
        self.lin = Linear(dense_dim,out_dim)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        
    def forward(self, data) -> torch.Tensor:
        G=p.ggraph.from_edge_index(data.edge_index.float().to('cpu'),data.num_nodes)

        edges=data.edge_index.transpose(1,0).tolist()
        node_rep = p.ptensors0b.from_matrix(self.node_embedding(data.x.flatten()))
        edge_rep = p.ptensors0b.from_matrix(self.edge_embedding(data.edge_attr.flatten()),edges)
        edge_rep = p.subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge)
        # write_to_file(edge_rep,"output/edge_rep_org.txt")

        for conv_layer in self.conv_layers:
            node_rep, edge_rep = conv_layer(node_rep, edge_rep, G)

        # node_outs = []
        node_outs = [node_rep.torch()]
        node_outs.append(p.subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node).torch())
        reps = torch.cat(node_outs,-1) #nc * 2
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps) # size = [num_graphs,out_dim]
    
