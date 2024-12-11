import torch
import ptens as p
from torch.nn import functional as F, Sequential, ModuleList, Linear, BatchNorm1d, ReLU, Parameter
from typing import Callable, Union, List
from functools import reduce
from argparse import Namespace

from data_cleaning.feature_encoders import get_node_encoder, get_edge_encoder

from torch_geometric.nn import global_add_pool, global_mean_pool
from .model_utils import write_to_file

from .model_utils import get_mlp

_inner_mlp_mult = 2
_cycle_sizes = [3,4,5,6,7,8]
# _cycle_sizes = [5,6]

class Edge_cycle(torch.nn.Module):
    def __init__(self, rep_dim,bn_eps,bn_momentum) -> None:
        super().__init__()
        self.cycle_mlp = get_mlp(5 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2,bn_eps,bn_momentum)     
        self.edge_mlp = get_mlp(3 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2,bn_eps,bn_momentum)
      
        self.edge=p.subgraph.edge()
        self.node=p.subgraph.trivial()
        self.cycles = [p.subgraph.cycle(i) for i in _cycle_sizes]
        
    def forward(self,edge_rep,cycle_rep,G,data):
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,cycle) for cycle in self.cycles]
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle,G,cycle) 
                       for cycle, edge2cycle in zip(self.cycles,edge2cycles)] #in principle should do linmaps, nc * 4
        edge2cycle = p.batched_subgraphlayer1b.cat(*edge2cycles)
        cycle_new = cycle_rep
        # cycles_new = [p.batched_subgraphlayer1b.gather_from_ptensors(cycle_rep,G,cycle) for cycle in self.cycles]
        # cycle_new = p.batched_subgraphlayer1b.cat(*cycles_new)
       
        cycle_out = self.cycle_mlp(torch.cat([cycle_new.torch(),edge2cycle.torch()],dim=-1)) #nc * 6 -> nc
        cycle_out = p.batched_ptensors1b.like(cycle_rep,cycle_out)
        
        cycle2edge = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_out,G,self.edge)
        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch()],dim=-1)) #nc * 3 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)
        
        return edge_out, cycle_out

class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, rep_dim, bn_eps,bn_momentum) -> None:
        super().__init__()

        self.node_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2,bn_eps,bn_momentum)
        self.edge_mlp_0 = get_mlp(3 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2,bn_eps,bn_momentum)
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep, edge_rep,G,data):
        # print("----------------starting MPNN---------------")
        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep,G,self.edge) #nc

        edge_out = self.edge_mlp_0(torch.cat([edge_rep.torch(),node2edge.torch()],dim=-1)) #nc * 3 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)
        
        edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge_out,G,self.node) # nc

        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],dim = -1)) #nc * 2 -> nc
        node_out = p.batched_subgraphlayer0b.like(node_rep,node_out)
        return node_out, edge_out

class ConvLayer(torch.nn.Module):
    def __init__(self, rep_dim: int, dropout, bn_eps,bn_momentum) -> None:
        super().__init__()
        self.edge_node = Edge_node(rep_dim,bn_eps,bn_momentum)
        self.edge_cycle = Edge_cycle(rep_dim,bn_eps,bn_momentum)
        
        self.edge_mlp = get_mlp(2 * rep_dim,2 * rep_dim * _inner_mlp_mult,2 * rep_dim,2,bn_eps,bn_momentum)

    def forward(self, node_rep, edge_rep,cycle_rep,G, data):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out1 = self.edge_node(node_rep,edge_rep,G, data)
        edge_out2, cycle_out = self.edge_cycle(edge_rep,cycle_rep,G,data)
        
        #edge_out1, edge_out2 has dim = rep_dim; edge_out has dim = 2 * rep_dim 
        edge_out = self.edge_mlp(torch.cat([edge_out1.torch(),edge_out2.torch()],dim=-1))
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        #residual connection
        # node_out = node_rep + node_out
        # edge_out = edge_rep + edge_out
        return node_out, edge_out, cycle_out


class Model_EdgeCycle_0th_batch(torch.nn.Module):
    def __init__(self, rep_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, ds_name,dataset, bn_eps,bn_momentum,device = 'cuda') -> None:
        print("Running: Model_EdgeCycle_0th_batch\n\n")
        print("bn_momentum:",bn_momentum)
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,rep_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,rep_dim * 2)
        self.cycle_mlp = get_mlp(2*rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2,bn_eps,bn_momentum)
        
        self.conv_layers = ModuleList([ConvLayer(rep_dim,dropout, bn_eps,bn_momentum) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(4 * rep_dim,2 * rep_dim,False),
                                    BatchNorm1d(2 * rep_dim,eps=bn_eps,momentum=bn_momentum),
                                    ReLU(),
                                    Linear(2 * rep_dim,rep_dim,False),
                                    BatchNorm1d(rep_dim,eps=bn_eps,momentum=bn_momentum),
                                    ReLU()
                                    )

        self.dropout = dropout
        self.lin = Linear(rep_dim,out_dim)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycles = [p.subgraph.cycle(i) for i in _cycle_sizes]

        
    def forward(self, data) -> torch.Tensor:
        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)

        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x.flatten()))
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr.flatten()))
        edge_rep = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.edge)
        
        cycle_reps = [p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,cycle) for cycle in self.cycles]
        cycle_tmp = p.batched_subgraphlayer1b.cat(*cycle_reps)
        cycle_reps = [p.batched_subgraphlayer1b.gather_from_ptensors(cycle_rep,G,cycle) 
                       for cycle, cycle_rep in zip(self.cycles,cycle_reps)] #in principle should do linmaps, nc * 2
        cycle_rep = p.batched_subgraphlayer1b.cat(*cycle_reps)
        cycle_rep2 = self.cycle_mlp(cycle_rep.torch())
        cycle_rep = p.batched_ptensors1b.like(cycle_tmp,cycle_rep2)

        for conv_layer in self.conv_layers:
            node_rep, edge_rep, cycle_rep = conv_layer(node_rep, edge_rep, cycle_rep,G, data)

        # node_outs = []
        node_outs = [node_rep.torch()]
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node).torch())
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node).torch())
        reps = torch.cat(node_outs,-1) #nc * 2
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps) # size = [num_graphs,out_dim]
    
