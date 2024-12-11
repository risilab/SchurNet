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
from .branched_cycles import get_branched_5_cycles, get_branched_6_cycles, get_subgraph_size

_inner_mlp_mult = 2


class Edge_branched_cycle(torch.nn.Module):
    '''do Autobahn operation on branched cycles'''
    def __init__(self, rep_dim, num_channels,use_branched_5_cycles) -> None:
        super().__init__()
        self.br_cycle_mlp = get_mlp((3 + num_channels) * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,3) # one more layer mlp
        self.edge_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)

        self.edge=p.subgraph.edge()
        self.node=p.subgraph.trivial()
        self.br_cycles = list(get_branched_6_cycles().values())
        if use_branched_5_cycles:
            self.br_cycles.extend(list(get_branched_5_cycles().values()))
        self.br_cycle_sizes = [get_subgraph_size(S) for S in self.br_cycles]
        
        self.br_cycle_autobahns = ModuleList([p.Autobahn(rep_dim,rep_dim * num_channels,br_cycle) 
                                          for br_cycle in self.br_cycles])

    def forward(self,edge_rep,br_cycle_rep,G,data):
        edge2br_cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,br_cycle,min_overlaps=2) 
                       for br_cycle in self.br_cycles]

        edge2br_cycles_aut = [autobahn(edge2br_cycle)
                       for autobahn, edge2br_cycle in zip(self.br_cycle_autobahns,edge2br_cycles)]
        edge2br_cycle_aut = p.batched_subgraphlayer1b.cat(*edge2br_cycles_aut) #nc * num_channels
        edge2br_cycles_linmap = [p.batched_subgraphlayer1b.gather_from_ptensors(edge2br_cycle,G,br_cycle,min_overlaps=size)  #min_overlaps=3 to make sure only cycle to itself
                       for br_cycle,size, edge2br_cycle in zip(self.br_cycles, self.br_cycle_sizes,edge2br_cycles)] #in principle should do linmaps, nc * 4
        edge2br_cycle_linmap = p.batched_subgraphlayer1b.cat(*edge2br_cycles_linmap) #nc * 2
        
        br_cycle_out = self.br_cycle_mlp(torch.cat([br_cycle_rep.torch(),edge2br_cycle_linmap.torch(),edge2br_cycle_aut.torch()],dim=-1)) #nc * 3 -> nc
        br_cycle_out = p.batched_ptensors1b.like(br_cycle_rep,br_cycle_out)

        br_cycle2edge = p.batched_subgraphlayer0b.gather_from_ptensors(br_cycle_out,G,self.edge,min_overlaps=2)
        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),br_cycle2edge.torch()],dim=-1)) #nc * 2 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        return edge_out, br_cycle_out

class Edge_cycle(torch.nn.Module):
    def __init__(self, rep_dim,cycle_sizes,gather_overlap) -> None:
        super().__init__()
        self.cycle_mlp = get_mlp(3 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,3) # one more layer mlp
        self.edge_mlp = get_mlp(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)

        self.edge=p.subgraph.edge()
        self.node=p.subgraph.trivial()
        self.cycles = [p.subgraph.cycle(i) for i in cycle_sizes]
        self.cycle_sizes = cycle_sizes
        
        self.gather_overlap = gather_overlap

    def forward(self,edge_rep,cycle_rep,G,data):
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,cycle,min_overlaps=self.gather_overlap) for cycle in self.cycles]
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle,G,cycle,min_overlaps=size)  #min_overlaps=3 to make sure only cycle to itself
                       for cycle, size,edge2cycle in zip(self.cycles,self.cycle_sizes,edge2cycles)] #in principle should do linmaps, nc * 4
        edge2cycle = p.batched_subgraphlayer1b.cat(*edge2cycles) #nc * 2

        cycle_out = self.cycle_mlp(torch.cat([cycle_rep.torch(),edge2cycle.torch()],dim=-1)) #nc * 3 -> nc
        cycle_out = p.batched_ptensors1b.like(cycle_rep,cycle_out)

        cycle2edge = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_out,G,self.edge,min_overlaps=self.gather_overlap)
        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch()],dim=-1)) #nc * 2 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        return edge_out, cycle_out

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
        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep,G,self.edge) #nc

        edge_out = self.edge_mlp_0(torch.cat([edge_rep.torch(),node2edge.torch()],dim=-1)) #nc * 2 -> nc
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge_out,G,self.node) # nc

        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],dim = -1)) #nc * 2 -> nc
        node_out = p.batched_subgraphlayer0b.like(node_rep,node_out)
        return node_out, edge_out

class ConvLayer(torch.nn.Module):
    def __init__(self, rep_dim: int,num_channels, dropout,cycle_sizes,use_branched_5_cycles, gather_overlap) -> None:
        super().__init__()
        self.edge_node = Edge_node(rep_dim)
        self.edge_cycle = Edge_cycle(rep_dim,cycle_sizes,gather_overlap)
        self.edge_br_cycle = Edge_branched_cycle(rep_dim, num_channels, use_branched_5_cycles)

        self.edge_mlp = get_mlp(3 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,3) #one more layer mlp

    def forward(self, node_rep, edge_rep,cycle_rep,br_cycle_rep,G, data):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        node_out, edge_out1 = self.edge_node(node_rep,edge_rep,G, data)
        edge_out2, cycle_out = self.edge_cycle(edge_rep,cycle_rep,G,data)
        edge_out3, br_cycle_out = self.edge_br_cycle(edge_rep,br_cycle_rep,G,data)

        #edge_out1, edge_out2 has dim = rep_dim; edge_out has dim = 2 * rep_dim
        edge_out = self.edge_mlp(torch.cat([edge_out1.torch(),edge_out2.torch(),edge_out3.torch()],dim=-1))
        edge_out = p.batched_subgraphlayer0b.like(edge_rep,edge_out)

        #residual connection
        # node_out = node_rep + node_out
        # edge_out = edge_rep + edge_out
        return node_out, edge_out, cycle_out, br_cycle_out


class Model_EdgeCycle_Aut_on_branch_0th_batch_small(torch.nn.Module):
    def __init__(self, rep_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, ds_name,num_channels,dataset,cycle_sizes,use_branched_5_cycles,gather_overlap,device = 'cuda') -> None:
        print("Running: Model_EdgeCycle_Aut_on_branch_0th_batch_small\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,rep_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,rep_dim)
        self.cycle_mlp = get_mlp(2*rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
        self.br_cycle_mlp = get_mlp(2*rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)

        self.conv_layers = ModuleList([ConvLayer(rep_dim,num_channels,dropout,cycle_sizes,use_branched_5_cycles,gather_overlap) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(4 * rep_dim,2 * rep_dim,False),
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
        self.br_cycles = list(get_branched_6_cycles().values())
        if use_branched_5_cycles:
            self.br_cycles.extend(list(get_branched_5_cycles().values()))
        self.br_cycle_sizes = [get_subgraph_size(S) for S in self.br_cycles]
        print("br_cycle_sizes: ", self.br_cycle_sizes)

    def forward(self, data) -> torch.Tensor:
        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)

        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x.flatten()))
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr.flatten()))
        edge_rep = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.edge, min_overlaps = 2) #min_overlaps=2 to make sure only edge to itself

        cycle_reps = [p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,cycle) for cycle in self.cycles]
        cycle_tmp = p.batched_subgraphlayer1b.cat(*cycle_reps)
        cycle_reps = [p.batched_subgraphlayer1b.gather_from_ptensors(cycle_rep,G,cycle,min_overlaps=size)
                       for cycle, size,cycle_rep in zip(self.cycles,self.cycle_sizes,cycle_reps)] #in principle should do linmaps, nc * 2
        cycle_rep = p.batched_subgraphlayer1b.cat(*cycle_reps)
        cycle_rep2 = self.cycle_mlp(cycle_rep.torch())
        cycle_rep = p.batched_ptensors1b.like(cycle_tmp,cycle_rep2)
        
        br_cycle_reps = [p.batched_subgraphlayer1b.gather_from_ptensors(node_rep, G, br_cycle) for br_cycle in self.br_cycles]
        br_cycle_tmp = p.batched_subgraphlayer1b.cat(*br_cycle_reps)
        br_cycle_reps = [p.batched_subgraphlayer1b.gather_from_ptensors(br_cycle_rep, G, br_cycle, min_overlaps=size)
                        for br_cycle, size, br_cycle_rep in zip(self.br_cycles, self.br_cycle_sizes, br_cycle_reps)] #in principle should do linmaps, nc * 2
        br_cycle_rep = p.batched_subgraphlayer1b.cat(*br_cycle_reps)
        br_cycle_rep2 = self.br_cycle_mlp(br_cycle_rep.torch())
        br_cycle_rep = p.batched_ptensors1b.like(br_cycle_tmp, br_cycle_rep2)

        for conv_layer in self.conv_layers:
            node_rep, edge_rep, cycle_rep, br_cycle_rep = conv_layer(node_rep, edge_rep, cycle_rep,br_cycle_rep,G, data)

        # node_outs = []
        node_outs = [node_rep.torch()]
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node).torch())
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node).torch())
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(br_cycle_rep,G,self.node).torch())
        reps = torch.cat(node_outs,-1) #nc * 2
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)

        return self.lin(reps) # size = [num_graphs,out_dim]

