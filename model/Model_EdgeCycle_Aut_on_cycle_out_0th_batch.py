import torch
import ptens as p
from torch.nn import functional as F, Sequential, ModuleList, Linear, BatchNorm1d, ReLU, Parameter
from typing import Callable, Union, List
from functools import reduce
from argparse import Namespace

from data_cleaning.feature_encoders import get_node_encoder, get_edge_encoder

from torch_geometric.nn import global_add_pool, global_mean_pool
from .model_utils import write_to_file,get_mlp

_inner_mlp_mult = 2

class Edge_cycle(torch.nn.Module):
    def __init__(self, rep_dim, num_channels, cycle_sizes) -> None:
        super().__init__()
        # self.batchnorm = BatchNorm1d(num_channels * rep_dim)
        self.cycle_mlp_2 = Sequential(
            Linear(2 * rep_dim,rep_dim * _inner_mlp_mult,True),
            ReLU(),
            Linear(rep_dim * _inner_mlp_mult,rep_dim,True),
            ReLU()
        )
        self.cycle_mlp = get_mlp((num_channels + 2) * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,3)  # one more layer mlp
        self.edge_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
        self.edge=p.subgraph.edge()
        self.node=p.subgraph.trivial()
        self.cycle_sizes = cycle_sizes
        self.cycles = [p.subgraph.cycle(i) for i in cycle_sizes]
        self.cycle_autobahns =ModuleList([p.Autobahn(rep_dim,rep_dim * num_channels,cycle) 
                                          for cycle in self.cycles])
        
        
    def forward(self,edge_rep,cycles_rep,G,data):
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,cycle) for cycle in self.cycles]

        cycles_new = [p.batched_subgraphlayer1b.cat_channels(cycle_rep,edge2cycle)
                      for cycle_rep,edge2cycle in zip(cycles_rep,edge2cycles)] #nc * 2
        cycles_new = [self.cycle_mlp_2(cycle_new.torch()) for cycle_new in cycles_new] #nc * 2 -> nc
        cycles_new = [p.batched_subgraphlayer1b.like(cycle_rep,cycle_new) 
                      for cycle_rep,cycle_new in zip(cycles_rep,cycles_new)]
        cycles_aut = [cycle_autobahn(cycle_new)
                       for cycle_autobahn, cycle_new in zip(self.cycle_autobahns,cycles_new)] #nc * num_channels
        # cycles_aut = [F.relu(self.batchnorm(cycle_aut.torch())) for cycle_aut in cycles_aut] # do batch_norm to the feature after autobahn to improve stability
        cycles_linmap = [p.batched_subgraphlayer1b.gather_from_ptensors(cycle_new,G,cycle,min_overlaps=size) 
                         for cycle_new, cycle, size in zip(cycles_new,self.cycles,self.cycle_sizes)] #nc * 2

        cycles_out = [self.cycle_mlp(torch.cat([cycle_linmap.torch(),cycle_aut.torch()],dim=-1))
                      for cycle_linmap, cycle_aut in zip(cycles_linmap,cycles_aut)] #nc * (num_channels + 2) -> nc
        cycles_out = [p.batched_subgraphlayer1b.like(cycle_rep,cycle_out) for cycle_rep,cycle_out in zip(cycles_rep,cycles_out)]
        cycle_out = p.batched_subgraphlayer1b.cat(*cycles_out)        
        
        cycle2edge = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_out,G,self.edge)
        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch()],dim=-1)) #nc * 2 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)
        
        return edge_out, cycles_out

class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, rep_dim) -> None:
        super().__init__()

        self.node_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
        self.edge_mlp_0 = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep, edge_rep,G,data):
        # print("----------------starting MPNN---------------")
        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep,G,self.edge)

        edge_out = self.edge_mlp_0(torch.cat([edge_rep.torch(),node2edge.torch()],dim=-1))
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)
        
        edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge_out,G,self.node)

        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],dim = -1)) #nc * 2 -> nc
        node_out = p.batched_subgraphlayer0b.like(node_rep,node_out)
        return node_out, edge_out

class ConvLayer(torch.nn.Module):
    def __init__(self, rep_dim, num_channels,dropout,cycle_sizes) -> None:
        super().__init__()
        self.edge_node = Edge_node(rep_dim)
        self.edge_cycle = Edge_cycle(rep_dim, num_channels, cycle_sizes)
        
        self.edge_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,3)  # one more layer mlp

    def forward(self, node_rep, edge_rep,cycles_rep,G, data):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out = self.edge_node(node_rep,edge_rep,G, data)        
        edge_out2, cycles_out = self.edge_cycle(edge_rep,cycles_rep,G,data)
        
        edge_out = self.edge_mlp(torch.cat([edge_out.torch(),edge_out2.torch()],dim=-1))
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        #residual connection
        # node_out = node_rep + node_out
        # edge_out = edge_rep + edge_out
        return node_out, edge_out, cycles_out


class Model_EdgeCycle_Aut_0th_batch(torch.nn.Module):
    def __init__(self, rep_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, num_channels,ds_name,dataset,cycle_sizes,device = 'cuda') -> None:
        print("Running: Model_EdgeCycle_Aut_0th_batch\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,rep_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,rep_dim) 
        self.cycle_mlp = get_mlp(2*rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
        
        self.conv_layers = ModuleList([ConvLayer(rep_dim,num_channels,dropout,cycle_sizes) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(3 * rep_dim,2 * rep_dim,False),
                                    BatchNorm1d(2 * rep_dim),
                                    ReLU(),
                                    Linear(2 * rep_dim,rep_dim,False),
                                    BatchNorm1d(rep_dim),
                                    ReLU()
                                    )

        self.dropout = dropout
        self.lin = Linear(rep_dim,out_dim)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycles = [p.subgraph.cycle(i) for i in cycle_sizes]
        self.cycle_sizes = cycle_sizes

        
    def forward(self, data) -> torch.Tensor:
        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)

        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x.flatten()))
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr.flatten()))
        edge_rep = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.edge,min_overlaps=2)
        
        cycles_rep = [p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,cycle) for cycle in self.cycles]
        cycles_rep2 = [p.batched_subgraphlayer1b.gather_from_ptensors(cycle_rep,G,cycle,min_overlaps=size) 
                         for cycle_rep, cycle, size in zip(cycles_rep,self.cycles,self.cycle_sizes)] #linmaps
        cycles_rep2 = [self.cycle_mlp(cycle_rep.torch()) for cycle_rep in cycles_rep2] 
        cycles_rep = [p.batched_subgraphlayer1b.like(cycle_tmp,cycle_rep2) for cycle_tmp,cycle_rep2 in zip(cycles_rep,cycles_rep2)]

        for conv_layer in self.conv_layers:
            node_rep, edge_rep, cycles_rep = conv_layer(node_rep, edge_rep, cycles_rep,G, data)

        # node_outs = []
        node_outs = [node_rep.torch()]
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node).torch())
        cycle_rep = p.batched_subgraphlayer1b.cat(*cycles_rep)
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node).torch())
        reps = torch.cat(node_outs,-1) #nc * 2
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps) # size = [num_graphs,out_dim]
    
