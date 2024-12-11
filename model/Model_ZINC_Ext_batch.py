import torch
import ptens as p
from torch.nn import functional as F
from torch.nn import Sequential,ModuleList,Linear,BatchNorm1d,ReLU, Parameter
from typing import Callable, Union, List
from functools import reduce
from Autobahn_ptens.data_cleaning.feature_encoders import get_node_encoder, get_edge_encoder

_inner_mlp_mult = 2

ptens0_type = Union[p.batched_ptensors0b,p.batched_subgraphlayer0b]
ptens1_type = Union[p.batched_ptensors1b,p.batched_subgraphlayer1b]

def get_path_subgraph(min_len,max_len):
    eddge_index = [[i,i+1] for i in range(min_len)]
    path_subgraphs = []
    for i in range(min_len,max_len + 1):
        path_i = p.subgraph.from_edge_index(torch.tensor(eddge_index).float().transpose(0,1))
        path_subgraphs.append(path_i)
        eddge_index.append([i,i+1])
        # print(f"i: {path_i}")
    return path_subgraphs
    

class Edge_edge(torch.nn.Module):
    '''Edge and edge message passing'''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(5 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        # I think could also just use constand 1/sqrt(2)
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,edge_rep: ptens1_type,G):
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge) #nc * 2
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge2edge,G,self.edge) #nc * 4

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),edge2edge.torch()],-1)) #nc * 5 -> nc
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        return edge_out
class Cycle_path(torch.nn.Module):
    '''Cycle and path message passing'''
    def __init__(self, hidden_dim,cycle_sizes) -> None:
        super().__init__()
        self.cycles = [p.subgraph.cycle(k) for k in cycle_sizes]
        self.cycle_autobahns = ModuleList([p.Autobahn(2 * hidden_dim,hidden_dim,cycle) for cycle in self.cycles])
        # self.cycle_autobahns = ModuleList([p.Linear(2 * hidden_dim,hidden_dim) for cycle in self.cycles])

        self.paths = get_path_subgraph(3,6)
        self.paths_autobahns = ModuleList([p.Autobahn(2 * hidden_dim, hidden_dim, path) for path in self.paths])
        # self.paths_autobahns = ModuleList([p.Linear(2 * hidden_dim, hidden_dim) for path in self.paths])

    
    def forward(self,cycle_reps: List[ptens1_type], path_reps: List[ptens1_type],G):
        #cycle to path

        cycles2paths = [[p.batched_subgraphlayer1b.gather_from_ptensors(cycle_rep,G,path) for cycle_rep in cycle_reps]
                         for path in self.paths] #nc * 2
        path_outs = [reduce(lambda x,y: x+y,cycles2path) for cycles2path in cycles2paths] #gather info from all intersected cycles
        path_outs = [autobahn(path_out) 
                       for autobahn,path_out in zip(self.paths_autobahns,path_outs)] #use autobahn to process/broacast info 
        
        #path to cycle
        paths2cycles = [[p.batched_subgraphlayer1b.gather_from_ptensors(path_rep,G,cycle) for path_rep in path_reps]
                         for cycle in self.cycles] #nc * 2
        cycle_outs = [reduce(lambda x,y: x+y,paths2cycle) for paths2cycle in paths2cycles] #gather info from all intersected paths
        cycle_outs = [autobahn(cycle_out) 
                       for autobahn,cycle_out in zip(self.cycle_autobahns,cycle_outs)] #use autobahn to process/broacast info 

        return cycle_outs, path_outs

class Edge_Cycle(torch.nn.Module):
    '''node, edge and cycle message passing'''
    def __init__(self, hidden_dim, cycle_sizes) -> None:
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(9 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.cycle_mlp = Sequential(
            Linear(2 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.path_mlp = Sequential(
            Linear(2 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        # I think could also just use constand 1/sqrt(2)
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

        self.cycles = [p.subgraph.cycle(k) for k in cycle_sizes]
        self.cycle_autobahns = ModuleList([p.Autobahn(2 * hidden_dim,hidden_dim,cycle) for cycle in self.cycles])
        # self.cycle_autobahns = ModuleList([p.Linear(2 * hidden_dim,hidden_dim) for cycle in self.cycles])

        self.paths = get_path_subgraph(3,6)
        self.paths_autobahns = ModuleList([p.Autobahn(2 * hidden_dim, hidden_dim, path) for path in self.paths])
        # self.paths_autobahns = ModuleList([p.Linear(2 * hidden_dim, hidden_dim) for path in self.paths])


    def forward(self,edge_rep: ptens1_type,cycle_reps: List[ptens1_type], path_reps: List[ptens1_type],G):
        #edge to cyles
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,cycle) 
                       for cycle in self.cycles] #nc * 2
        # print(f"edge2cycles: ")
        # for edge2cycle in edge2cycles:
        #     print(edge2cycle)
        edge2cycles = [autobahn(edge2cycle) 
                       for autobahn,edge2cycle in zip(self.cycle_autobahns,edge2cycles)]
        # print("after autobahn")
        print("type edge2cycles",type(edge2cycles[0]))
        print("type cycle_reps",type(cycle_reps[0]))
        print("edge2cycles: ",edge2cycles[0].torch().shape)
        print("cycle_reps: ",cycle_reps[0].torch().shape)
        cycle_news = [p.batched_subgraphlayer1b.cat_channels(cycle_in,cycle_rep) 
                         for cycle_in,cycle_rep in zip(edge2cycles,cycle_reps)] #nc * 2
        cycles2edge = [p.batched_subgraphlayer1b.gather_from_ptensors(cycle_new,G,self.edge) 
                       for cycle_new in cycle_news] #nc * 4
        cycle2edge = reduce(lambda x,y: x+y, cycles2edge) 

        #edge2paths
        edge2paths = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,path) 
                       for path in self.paths] #nc * 2
        edge2paths = [autobahn(edge2path) 
                       for autobahn,edge2path in zip(self.paths_autobahns,edge2paths)]
        path_news = [p.batched_subgraphlayer1b.cat_channels(path_in,path_rep) 
                         for path_in,path_rep in zip(edge2paths,path_reps)] #nc * 2

        paths2edge = [p.batched_subgraphlayer1b.gather_from_ptensors(path_new,G,self.edge) 
                       for path_new in path_news] #nc * 4
        path2edge = reduce(lambda x,y: x+y, paths2edge) 

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch(), path2edge.torch()],-1)) #nc * 9 -> nc
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        
        # cycle_outs = cycle_news
        # path_outs = path_news

        cycle_outs = [self.cycle_mlp(cycle_new.torch()) 
                      for cycle_new in cycle_news] #nc * 2 -> nc; all cycles share the same mlp
        cycle_outs = [p.batched_subgraphlayer1b.like(cycle_rep,cycle_out) 
                      for cycle_rep,cycle_out in zip(cycle_reps,cycle_outs)]
        path_outs = [self.path_mlp(path_new.torch()) 
                      for path_new in path_news] #nc * 2 -> nc; all paths share the same mlp
        path_outs = [p.batched_subgraphlayer1b.like(path_rep,path_out) 
                      for path_rep,path_out in zip(path_reps,path_outs)]
        return edge_out, cycle_outs, path_outs

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout, cycle_sizes) -> None:
        super().__init__()
        self.mlp = Sequential(
            Linear(2*hidden_dim,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.cycle_mlp = Sequential(
            Linear(2 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.path_mlp = Sequential(
            Linear(2 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.edge2edge = Edge_edge(hidden_dim)
        self.edge_cycle = Edge_Cycle(hidden_dim,cycle_sizes)
        self.path_cycle = Cycle_path(hidden_dim,cycle_sizes)

    def forward(self, edge_rep: ptens1_type, cycle_reps: List[ptens1_type], path_reps: List[ptens1_type],G):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        edge_out = self.edge2edge(edge_rep, G)
        edge_out2, cycle_outs, path_outs = self.edge_cycle(edge_rep, cycle_reps,path_reps, G)
        cycle_outs2, path_outs2 = self.path_cycle(cycle_outs,path_outs, G)
        #edge_out, edge_out2 is torch.tensor

        edge_out = self.mlp(torch.cat([edge_out,edge_out2],-1))
        edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)

        cycle_news = [p.batched_subgraphlayer1b.cat_channels(cycle_out2,cycle_out) 
                         for cycle_out2,cycle_out in zip(cycle_outs2,cycle_outs)] #nc * 2
        cycle_outs = [self.cycle_mlp(cycle_new.torch()) 
                      for cycle_new in cycle_news] #nc * 2 -> nc; all cycles share the same mlp
        cycle_outs = [p.batched_subgraphlayer1b.like(cycle_rep,cycle_out) 
                      for cycle_rep,cycle_out in zip(cycle_reps,cycle_outs)]
        path_news = [p.batched_subgraphlayer1b.cat_channels(path_out2,path_out) 
                         for path_out2,path_out in zip(path_outs2,path_outs)] #nc * 2
        path_outs = [self.path_mlp(path_new.torch()) 
                      for path_new in path_news] #nc * 2 -> nc; all paths share the same mlp
        path_outs = [p.batched_subgraphlayer1b.like(path_rep,path_out) 
                      for path_rep,path_out in zip(path_reps,path_outs)]

        #residual connection
        edge_out = edge_rep + edge_out
        cycle_outs = [cycle_rep + cycle_out for cycle_rep, cycle_out in zip(cycle_reps,cycle_outs)]
        path_outs = [path_rep + path_out for path_rep, path_out in zip(path_reps,path_outs)]
        return edge_out, cycle_outs, path_outs


class Model_ZINC_Ext_batch(torch.nn.Module):
    def __init__(self, rep_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, num_channels,ds_name,dataset,cycle_sizes,device = 'cuda') -> None:
        print("Running: Model_ZINC_Ext_batch\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,rep_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,rep_dim) 
        
        self.conv_layers = ModuleList([ConvLayer(rep_dim,dropout,cycle_sizes) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(3 * rep_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU()
                                    )
        self.lin = Linear(dense_dim,out_dim)
        
        self.edge_mlp1 = p.Linear(2 * rep_dim,rep_dim * _inner_mlp_mult)
        self.edge_mlp2 = p.Linear(rep_dim * _inner_mlp_mult,rep_dim)

        self.cycle_mlp1 = p.Linear(rep_dim,rep_dim * _inner_mlp_mult)
        self.cycle_mlp2 = p.Linear(rep_dim * _inner_mlp_mult,rep_dim)

        self.path_mlp1 = p.Linear(rep_dim,rep_dim * _inner_mlp_mult)
        self.path_mlp2 = p.Linear(rep_dim * _inner_mlp_mult,rep_dim)

        self.dropout = dropout
        

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycles = [p.subgraph.cycle(k) for k in cycle_sizes]
        self.paths = get_path_subgraph(3,6)

        self.device = device
        
    def forward(self, data) -> torch.Tensor:
        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)

        edges=data.edge_index.transpose(1,0).tolist()
        # node_rep = p.batched_ptensors0b.from_matrix(self.node_embedding(data.x.flatten()),data.graph_size.tolist())
        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x.flatten()))
        # print(f"data.edge_index.shape : {data.edge_index.shape}")
        # print(f"data.edge_attr.shape: {data.edge_attr.shape}")
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr.flatten()))
        # print(f"edge_rep:\n {edge_rep}")

        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge)

        edge_rep = p.batched_subgraphlayer1b.cat_channels(node2edge,edge2edge) #nc * 2
        edge_rep = self.edge_mlp1(edge_rep).relu()
        edge_rep = self.edge_mlp2(edge_rep).relu() #nc = hidden_dim

        node2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,cycle_k) for cycle_k in self.cycles]
        #didn't do autobahn here. a linmap would be prefered, but omitted
        cycle_reps = [self.cycle_mlp1(cycle_rep).relu() for cycle_rep in node2cycles] 
        cycle_reps = [self.cycle_mlp2(cycle_rep).relu() for cycle_rep in cycle_reps] #nc = hidden_dim

        node2paths = [p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,path_k) for path_k in self.paths]
        #didn't do autobahn here. a linmap would be preferred, but omitted
        path_reps = [self.path_mlp1(path_rep).relu() for path_rep in node2paths] 
        path_reps = [self.path_mlp2(path_rep).relu() for path_rep in path_reps] #nc = hidden_dim

        for conv_layer in self.conv_layers:
            edge_rep, cycle_reps, path_reps = conv_layer(edge_rep, cycle_reps, path_reps, G)

        # print("path sizes")
        # for length, path_out in zip(range(3,7),path_reps):
        #     print(f"path of length: {length}")
        #     print(f"path_out.shape: {path_out.torch().shape}")
        #     print(f"num of paths: {path_out.torch().shape[0] / (length + 1)}")

        # print("cycle sizes")
        # for length, cycle_out in zip(range(5,7),cycle_reps):
        #     print(f"cycle of length: {length}")
        #     print(f"cycle_out.shape: {cycle_out.torch().shape}")
        #     print(f"num of cycles: {cycle_out.torch().shape[0] / (length)}")

        # print(f"Edges\n {data.edge_index.shape}")
        # print(f"Edge_out.shape\n {edge_rep.torch().shape}")
        # print(f"Nodes\n {data.num_nodes}")


        cycle_rep = p.batched_subgraphlayer1b.cat(*cycle_reps)  
        path_rep = p.batched_subgraphlayer1b.cat(*path_reps)
        node_outs = [p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node).torch()]
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(path_rep,G,self.node).torch())
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node).torch())
        reps = torch.cat(node_outs,-1) #nc * 3
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps)
    

