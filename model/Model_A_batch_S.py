import torch
import ptens as p
from torch.nn import functional as F
from torch.nn import Sequential,ModuleList,Linear,BatchNorm1d,ReLU, Parameter
from typing import Callable, Union, List
from Autobahn_ptens.data_cleaning.data_loader import get_dataloader
from torch_geometric.nn import global_add_pool, global_mean_pool
from data_cleaning.utils import Counter
from data_cleaning.graph_cache import PreAddIndex, cache_graph, GraphTransform
from functools import reduce

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
            Linear(hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.linear = Linear(4 * hidden_dim,hidden_dim)
        self.eps = Parameter(torch.tensor(0.1))
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,edge_rep: ptens1_type,G):
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge) #nc * 2
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge2edge,G,self.edge) #nc * 4
        # edge2edge = p.batched_subgraphlayer1b.linmaps(edge2edge)
        #combine feature using linear combination
        edge2edge = self.linear(edge2edge.torch())

        #GIN ways of combining old and new representation
        edge_out = self.edge_mlp(edge2edge + (1 + self.eps) * edge_rep.torch()) 
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        return edge_out

class Edge_Cycle(torch.nn.Module):
    '''node, edge and cycle message passing'''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.edge_linear = Linear(4 * hidden_dim,hidden_dim)

        self.cycle_mlp = Sequential(
            Linear(hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

        self.cycles = [p.subgraph.cycle(k) for k in [5,6]]
        # self.cycle_autobahns = ModuleList([p.Autobahn(2 * hidden_dim,hidden_dim,cycle) for cycle in self.cycles])
        self.cycle_linears = ModuleList([p.Linear(4 * hidden_dim,hidden_dim) for cycle in self.cycles])
        self.cycle_eps = Parameter(torch.tensor(0.1))
        self.edge_eps = Parameter(torch.tensor(0.1))
        

    def forward(self,edge_rep: ptens1_type,cycle_reps: List[ptens1_type],G):
        #edge to cyles
        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,cycle) 
                       for cycle in self.cycles] #nc * 2
        # edge2cycles = [autobahn(edge2cycle) 
        #                for autobahn,edge2cycle in zip(self.cycle_autobahns,edge2cycles)]
        # print("after autobahn")

        edge2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle,G,cycle) 
                       for edge2cycle, cycle in zip(edge2cycles,self.cycles)] #nc * 4
        edge2cycles = [linear(edge2cycle) 
                       for linear, edge2cycle in zip(self.cycle_linears,edge2cycles)] #nc 
        cycles2edge = [p.batched_subgraphlayer1b.cat_channels(cycle_in,cycle_rep) 
                         for cycle_in,cycle_rep in zip(edge2cycles,cycle_reps)] #nc * 2
        cycles2edge = [p.batched_subgraphlayer1b.gather_from_ptensors(cycle_new,G,self.edge) 
                       for cycle_new in cycles2edge] #nc * 4
        cycle2edge = reduce(lambda x,y: x+y, cycles2edge)
        cycle2edge = self.edge_linear(cycle2edge.torch())

        edge_out = self.edge_mlp(cycle2edge + edge_rep.torch() * (1 + self.edge_eps)) #nc -> nc
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)

        cycle_outs = [self.cycle_mlp(cycle_in.torch() + cycle_rep.torch() * (1 + self.cycle_eps)) 
                      for cycle_in, cycle_rep in zip(edge2cycles,cycle_reps)] #nc -> nc; all cycles share the same mlp
        cycle_outs = [p.batched_subgraphlayer1b.like(cycle_rep,cycle_out) 
                      for cycle_rep,cycle_out in zip(cycle_reps,cycle_outs)]
        return edge_out, cycle_outs

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout) -> None:
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

        self.edge2edge = Edge_edge(hidden_dim)
        self.edge_cycle = Edge_Cycle(hidden_dim)

    def forward(self, edge_rep: ptens1_type, cycle_reps: List[ptens1_type],G):
        '''
        cycle_reps is a list of cycle_rep of different lenghts
        '''
        edge_out = self.edge2edge(edge_rep, G)
        edge_out2, cycle_outs = self.edge_cycle(edge_rep, cycle_reps, G)
        #edge_out, edge_out2 is torch.tensor

        edge_out = self.mlp(torch.cat([edge_out,edge_out2],-1))
        edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        
        #residual connection
        edge_out = edge_rep + edge_out
        cycle_outs = [cycle_rep + cycle_out for cycle_rep, cycle_out in zip(cycle_reps,cycle_outs)]
        return edge_out, cycle_outs


class Model_Andrew_batch_S(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dense_dim: int, num_layers: int, dropout,
                 readout: Callable[[torch.Tensor,torch.Tensor],torch.Tensor], device = 'cuda') -> None:
        print("Running: Model_Andrew_batch_Simple\n\n")
        super().__init__()
        self.node_embedding = torch.nn.Embedding(22,embedding_dim)
        self.edge_embedding = torch.nn.Embedding(4,embedding_dim)
        
        self.conv_layers = ModuleList([ConvLayer(hidden_dim,dropout) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear((1 + 2) * hidden_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU()
                                    )
        self.edge_mlp1 = p.Linear(2 * embedding_dim,hidden_dim * _inner_mlp_mult)
        self.edge_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.cycle_mlp1 = p.Linear(embedding_dim,hidden_dim * _inner_mlp_mult)
        self.cycle_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.path_mlp1 = p.Linear(embedding_dim,hidden_dim * _inner_mlp_mult)
        self.path_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.dropout = dropout
        self.lin = Linear(dense_dim,1)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycles = [p.subgraph.cycle(k) for k in [5,6]]
        self.paths = get_path_subgraph(3,6)


        self.device = device
        
    def forward(self, data) -> torch.Tensor:
        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)

        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list,self.node_embedding(data.x.flatten()))
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list,self.edge_embedding(data.edge_attr.flatten()))

        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge)


        edge_rep = p.batched_subgraphlayer1b.cat_channels(node2edge,edge2edge) #nc * 2
        edge_rep = self.edge_mlp1(edge_rep).relu()
        edge_rep = self.edge_mlp2(edge_rep).relu() #nc = hidden_dim

        node2cycles = [p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,cycle_k) for cycle_k in self.cycles]
        #didn't do autobahn here. a linmap would be prefered, but omitted
        cycle_reps = [self.cycle_mlp1(cycle_rep).relu() for cycle_rep in node2cycles] 
        cycle_reps = [self.cycle_mlp2(cycle_rep).relu() for cycle_rep in cycle_reps] #nc = hidden_dim


        for conv_layer in self.conv_layers:
            edge_rep, cycle_reps = conv_layer(edge_rep, cycle_reps, G)

        # node_outs = []
        node_outs = [p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node).torch() for cycle_rep in cycle_reps]
        node_outs.append(p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node).torch())
        reps = torch.cat(node_outs,-1) #nc * (1 + num_cycles)
        reps = self.readout(reps,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps).flatten()
    
# ---- Main --------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print("test")
    device = 'cuda'

    counter = Counter(0)
    train_loader = get_dataloader('/home/qingqi/Ptensor_Ext/Autobahn_ptens/zinc', 'train', 2, device, cache=True, transform=GraphTransform(), pre_transform=PreAddIndex(counter))

    model = Model_Andrew_batch_S(embedding_dim=2,hidden_dim=4,dense_dim=4,num_layers=4,dropout=0.,readout=global_add_pool).to(device)

    loss_fn = torch.nn.L1Loss()
    for data in train_loader:
        # print("batch is:", data)
        data.to(device)
        pred = model(data)
        loss = loss_fn(pred,data.y)
        print(f"loss: {loss}")
        loss.backward()
        break
