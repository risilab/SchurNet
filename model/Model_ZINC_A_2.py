import torch
import ptens as p
from torch.nn import functional as F
from torch.nn import Sequential,ModuleList,Linear,BatchNorm1d,ReLU, Parameter, Dropout
from typing import Callable, Union
from Autobahn_ptens.data_cleaning.data_loader import get_dataloader
from torch_geometric.nn import global_add_pool, global_mean_pool

_inner_mlp_mult = 2

ptens0_type = Union[p.ptensors0b,p.subgraphlayer0b]
ptens1_type = Union[p.ptensors1b,p.subgraphlayer1b]

class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, hidden_dim, dropout_rate) -> None:
        super().__init__()

        self.node_mlp = Sequential(
            Linear(7 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        self.edge_mlp = Sequential(
            Linear(7 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.edge_dropout = Dropout(dropout_rate)
        self.node_dropout = Dropout(dropout_rate)

        # I think could also just use constand 1/sqrt(2)
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep: ptens0_type, edge_rep: ptens1_type,G):
        node2edge = p.subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        node2edge = p.subgraphlayer1b.gather_from_ptensors(node2edge,G,self.edge) #nc * 2

        edge2edge = p.subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge) #nc * 2
        edge2edge = p.subgraphlayer1b.gather_from_ptensors(edge2edge,G,self.edge) #nc * 4

        edge2node = p.subgraphlayer1b.cat_channels(edge2edge, node2edge) #nc * 6
        edge2node = p.subgraphlayer0b.gather_from_ptensors(edge2node,G,self.node)

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),edge2edge.torch(), node2edge.torch()],-1)) #nc * 7 -> nc
        edge_out = self.edge_dropout(edge_out)
        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],-1)) #nc * 7 -> nc
        # node_out = self.node_dropout(node_out)
        node_out = p.ptensors0b.like(node_rep,node_out)
        # edge_out = p.subgraphlayer1b.like(edge_rep,edge_out)
        return node_out, edge_out
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
        edge2edge = p.subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge) #nc * 2
        edge2edge = p.subgraphlayer1b.gather_from_ptensors(edge2edge,G,self.edge) #nc * 4

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),edge2edge.torch()],-1)) #nc * 5 -> nc
        # edge_out = p.subgraphlayer1b.like(edge_rep,edge_out)
        return edge_out
    
class Edge_Cycle(torch.nn.Module):
    '''node, edge and cycle message passing'''
    def __init__(self, hidden_dim, dropout_rate) -> None:
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(11 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )
        self.edge_dropout = Dropout(dropout_rate)
        self.cycle_dropout = Dropout(dropout_rate)

        self.cycle_mlp = Sequential(
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
        self.cycle5=p.subgraph.cycle(5)
        self.cycle6=p.subgraph.cycle(6)

    def forward(self,edge_rep: ptens1_type,cycle_rep: ptens1_type,G):
        edge2cycle5 = p.subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.cycle5) #nc * 2
        edge2cycle5 = p.subgraphlayer1b.gather_from_ptensors(edge2cycle5,G,self.cycle5) #nc * 4
        edge2cycle6 = p.subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.cycle6) #nc * 2
        edge2cycle6 = p.subgraphlayer1b.gather_from_ptensors(edge2cycle6,G,self.cycle6) #nc * 4

        cycle_in = p.subgraphlayer1b.cat(edge2cycle5,edge2cycle6) 
        # cycle_in = p.ptensors1b.linmaps(cycle_in) #nc * 6
        cycle_new = p.subgraphlayer1b.cat_channels(cycle_in,cycle_rep) #nc * 5

        cycle2edge = p.subgraphlayer1b.gather_from_ptensors(cycle_new,G,self.edge) #nc * 10

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch()],-1)) # nc * 11 -> nc
        edge_out = self.edge_dropout(edge_out)
        # edge_out = p.subgraphlayer1b.like(edge_rep,edge_out)
        cycle_out = self.cycle_mlp(cycle_new.torch()) #nc * 5 -> nc
        # cycle_out = self.cycle_dropout(cycle_out)
        cycle_out = p.subgraphlayer1b.like(cycle_rep,cycle_out)
        return edge_out, cycle_out

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate) -> None:
        super().__init__()
        self.mlp = Sequential(
            Linear(2*hidden_dim,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        # self.edge2edge = Edge_edge(hidden_dim)
        self.edge_node = Edge_node(hidden_dim, dropout_rate)
        self.edge_cycle = Edge_Cycle(hidden_dim, dropout_rate)

    def forward(self, node_rep: ptens0_type,edge_rep: ptens1_type, cycle_rep: ptens1_type,G):
        # edge_out = self.edge2edge(edge_rep, G)
        node_out, edge_out = self.edge_node(node_rep,edge_rep,G)
        edge_out2, cycle_out = self.edge_cycle(edge_rep, cycle_rep, G)
        #edge_out, edge_out2 is torch.tensor

        edge_out = self.mlp(torch.cat([edge_out,edge_out2],-1))
        edge_out = p.subgraphlayer1b.like(edge_rep,edge_out)
        
        #residual connection
        node_out = node_rep + node_out
        edge_out = edge_rep + edge_out
        cycle_out = cycle_rep + cycle_out
        return node_out, edge_out, cycle_out


class Model_Andrew_2(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dense_dim: int, num_layers: int, dropout_rate,
                 readout: Callable[[torch.Tensor,torch.Tensor],torch.Tensor], device = 'cuda') -> None:
        print("Using Net_Andrew_2, added node rep")
        super().__init__()
        self.node_embedding = torch.nn.Embedding(22,embedding_dim)
        self.edge_embedding = torch.nn.Embedding(4,embedding_dim)
        
        self.conv_layers = ModuleList([ConvLayer(hidden_dim,dropout_rate) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(3 * hidden_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim),
                                    BatchNorm1d(dense_dim),
                                    ReLU()
                                    )
        self.node_mlp1 = p.Linear(embedding_dim,hidden_dim * _inner_mlp_mult)
        self.node_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.edge_mlp1 = p.Linear(2 * embedding_dim,hidden_dim * _inner_mlp_mult)
        self.edge_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.cycle_mlp1 = p.Linear(2 * embedding_dim,hidden_dim * _inner_mlp_mult)
        self.cycle_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.dropout_rate = dropout_rate
        self.dropout = Dropout(dropout_rate)
        self.lin = Linear(dense_dim,1)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycle5=p.subgraph.cycle(5)
        self.cycle6=p.subgraph.cycle(6)

        self.device = device
        
    def forward(self, data) -> torch.Tensor:
        G=p.ggraph.from_edge_index(data.edge_index.float().to('cpu'),data.num_nodes)

        edges=data.edge_index.transpose(1,0).tolist()
        node_rep = p.ptensors0b.from_matrix(self.node_embedding(data.x.flatten()))
        edge_rep = p.ptensors0b.from_matrix(self.edge_embedding(data.edge_attr.flatten()),edges)
        
        node2edge = p.subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        edge2edge = p.subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge)


        edge_rep = p.subgraphlayer1b.cat_channels(node2edge,edge2edge) #nc * 2
        edge_rep = self.edge_mlp1(edge_rep).relu()
        edge_rep = self.edge_mlp2(edge_rep).relu() #nc = hidden_dim

        node2cycle5 = p.subgraphlayer1b.gather_from_ptensors(node_rep,G,self.cycle5)
        node2cycle5 = p.subgraphlayer1b.gather_from_ptensors(node2cycle5,G,self.cycle5) #nc * 2
        node2cycle6 = p.subgraphlayer1b.gather_from_ptensors(node_rep,G,self.cycle6)
        node2cycle6 = p.subgraphlayer1b.gather_from_ptensors(node2cycle6,G,self.cycle6) #nc * 2

        cycle_rep = p.subgraphlayer1b.cat(node2cycle5,node2cycle6)
        cycle_rep = self.cycle_mlp1(cycle_rep).relu()
        cycle_rep = self.cycle_mlp2(cycle_rep).relu() #nc = hidden_dim

        # node_rep = p.subgraphlayer1b.gather_from_ptensors(node_rep,G,self.node)
        node_rep = self.node_mlp1(node_rep).relu()
        node_rep = self.node_mlp2(node_rep).relu() #nc = hidden_dim

        # node_out_1 = p.subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node)
        # node_out_2 = p.subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node)
        # reps = torch.cat([node_rep.torch(), node_out_1.torch(),node_out_2.torch()],-1) #nc = hidden_dim * 3
        for conv_layer in self.conv_layers:
            node_rep, edge_rep, cycle_rep = conv_layer(node_rep, edge_rep, cycle_rep, G)

        node_out_1 = p.subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node)
        node_out_2 = p.subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node)
        reps = torch.cat([node_rep.torch(), node_out_1.torch(),node_out_2.torch()],-1) #nc = hidden_dim * 3

        reps = self.readout(reps,data.batch) # nc = hidden_dim * 2 * (1 + num_layer)

        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        reps = self.dropout(reps)
        
        return self.lin(reps).flatten()
    
# ---- Main --------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print("test")
    device = 'cuda'

    train_loader = get_dataloader('zinc','train',2,device)
    model = Model_Andrew_2(embedding_dim=2,hidden_dim=4,dense_dim=4,num_layers=4,dropout_rate=0.2,readout=global_add_pool).to(device)

    loss_fn = torch.nn.L1Loss()
    for data in train_loader:
        data.to(device)
        pred = model(data)
        loss = loss_fn(pred,data.y)
        print(f"loss: {loss}")
        loss.backward()
        break
