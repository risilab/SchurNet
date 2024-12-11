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
_cycle_sizes = [3,4,5,6,7,8]

ptens0_type = Union[p.batched_ptensors0b,p.batched_subgraphlayer0b]
ptens1_type = Union[p.batched_ptensors1b,p.batched_subgraphlayer1b]

class Edge_cycle(torch.nn.Module):
    def __init__(self, rep_dim, num_channels) -> None:
        super().__init__()
        self.cycle_mlp = Sequential(
            Linear((num_channels + 1) * rep_dim,rep_dim * _inner_mlp_mult,False),
            BatchNorm1d(rep_dim*_inner_mlp_mult),
            ReLU(),
            Linear(rep_dim * _inner_mlp_mult,rep_dim,False),
            BatchNorm1d(rep_dim),
            ReLU()
        )
        self.edge_mlp = Sequential(
            Linear(2 * rep_dim,rep_dim * _inner_mlp_mult,False),
            BatchNorm1d(rep_dim*_inner_mlp_mult),
            ReLU(),
            Linear(rep_dim * _inner_mlp_mult,rep_dim,False),
            BatchNorm1d(rep_dim),
            ReLU()
        )
        self.edge=p.subgraph.edge()
        self.node=p.subgraph.trivial()
        self.cycles = [p.subgraph.cycle(i) for i in _cycle_sizes]
        self.cycle_autobahns = ModuleList(
            [ModuleList([p.Autobahn(rep_dim,rep_dim,cycle) for cycle in self.cycles]) for _ in range(num_channels)]
        )
        
        
    def forward(self,edge_rep,cycle_rep,G,data):
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,cycle) for cycle in self.cycles]
        edge2cycles = [[autobahn(edge2cycle)
                       for autobahn, edge2cycle in zip(one_channel_aut,edge2cycles)] for one_channel_aut in self.cycle_autobahns] #in principle should do linmaps, nc * 2
        
        edge2cycle_multi_ch = [p.batched_subgraphlayer1b.cat(*edge2cycles_one_ch).torch() for edge2cycles_one_ch in edge2cycles]
        edge2cycle = torch.cat(edge2cycle_multi_ch,dim=-1) #nc * num_channels
       
        cycle_out = self.cycle_mlp(torch.cat([cycle_rep.torch(),edge2cycle],dim=-1)) #nc * (num_channels + 1) -> nc
        cycle_out = p.batched_ptensors1b.like(cycle_rep,cycle_out)
        
        cycle2edge = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_out,G,self.edge)
        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch()],dim=-1)) #nc * 2 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)
        
        return edge_out, cycle_out

class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, rep_dim) -> None:
        super().__init__()

        self.node_mlp = Sequential(
            Linear(2 * rep_dim,rep_dim * _inner_mlp_mult,False),
            BatchNorm1d(rep_dim*_inner_mlp_mult),
            ReLU(),
            Linear(rep_dim * _inner_mlp_mult,rep_dim,False),
            BatchNorm1d(rep_dim),
            ReLU()
        )
        self.edge_mlp_0 = Sequential(
                Linear(2 * rep_dim,rep_dim * _inner_mlp_mult,False),
                BatchNorm1d(rep_dim*_inner_mlp_mult),
                ReLU(),
                Linear(rep_dim * _inner_mlp_mult,rep_dim,False),
                BatchNorm1d(rep_dim),
                ReLU()
            )
        
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
    def __init__(self, rep_dim: int, num_channels,dropout) -> None:
        super().__init__()
        self.edge_node = Edge_node(rep_dim)
        self.edge_cycle = Edge_cycle(rep_dim, num_channels)
        
        self.edge_mlp = Sequential(
                Linear(2 * rep_dim,rep_dim * _inner_mlp_mult,False),
                BatchNorm1d(rep_dim*_inner_mlp_mult),
                ReLU(),
                Linear(rep_dim * _inner_mlp_mult,rep_dim,False),
                BatchNorm1d(rep_dim),
                ReLU()
            )

    def forward(self, node_rep, edge_rep,cycle_rep,G, data):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out = self.edge_node(node_rep,edge_rep,G, data)
        edge_out2, cycle_out = self.edge_cycle(edge_rep,cycle_rep,G,data)
        
        edge_out = self.edge_mlp(torch.cat([edge_out.torch(),edge_out2.torch()],dim=-1))
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        #residual connection
        # node_out = node_rep + node_out
        # edge_out = edge_rep + edge_out
        return node_out, edge_out, cycle_out


class Model_EdgeCycle_Aut_0th_batch_old(torch.nn.Module):
    def __init__(self, rep_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, num_channels,ds_name,dataset,cycle_sizes,device = 'cuda') -> None:
        print("Running: Model_EdgeCycle_Aut_0th_batch_old\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,rep_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,rep_dim)
        self.cycle_mlp = Sequential(
            Linear(2 * rep_dim,rep_dim * _inner_mlp_mult,False),
            BatchNorm1d(rep_dim*_inner_mlp_mult),
            ReLU(),
            Linear(rep_dim * _inner_mlp_mult,rep_dim,False),
            BatchNorm1d(rep_dim),
            ReLU()
        )
        
        self.conv_layers = ModuleList([ConvLayer(rep_dim,num_channels,dropout) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(3 * rep_dim,dense_dim,False),
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
    
