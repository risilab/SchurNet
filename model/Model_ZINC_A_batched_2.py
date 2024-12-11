import torch
import ptens as p
from torch.nn import functional as F
from torch.nn import Sequential,ModuleList,Linear,BatchNorm1d,ReLU, Parameter
from typing import Callable, Union
from Autobahn_ptens.data_cleaning.data_loader import get_dataloader
from torch_geometric.nn import global_add_pool, global_mean_pool
from typing import List
from data_cleaning.utils import Counter
from data_cleaning.graph_cache import PreAddIndex, cache_graph, GraphTransform



_inner_mlp_mult = 2

ptens0_type = Union[p.batched_ptensors0b,p.batched_subgraphlayer0b]
ptens1_type = Union[p.batched_ptensors1b,p.batched_subgraphlayer1b]

class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, hidden_dim) -> None:
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

        # I think could also just use constand 1/sqrt(2)
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)
        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep: ptens0_type, edge_rep: ptens1_type,G):
        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node2edge,G,self.edge) #nc * 2

        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge) #nc * 2
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge2edge,G,self.edge) #nc * 4

        edge2node = p.batched_subgraphlayer1b.cat_channels(edge2edge, node2edge) #nc * 6
        edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge2node,G,self.node)

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),edge2edge.torch(), node2edge.torch()],-1)) #nc * 7 -> nc
        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],-1)) #nc * 7 -> nc
        node_out = p.batched_ptensors0b.like(node_rep,node_out)
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
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
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge) #nc * 2
        edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge2edge,G,self.edge) #nc * 4

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),edge2edge.torch()],-1)) #nc * 5 -> nc
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        return edge_out
    
class Edge_Cycle(torch.nn.Module):
    '''node, edge and cycle message passing'''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.edge_mlp = Sequential(
            Linear(11 * hidden_dim,hidden_dim * _inner_mlp_mult,False),
            BatchNorm1d(hidden_dim*_inner_mlp_mult),
            ReLU(),
            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

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
        edge2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.cycle5) #nc * 2
        edge2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle5,G,self.cycle5) #nc * 4
        edge2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.cycle6) #nc * 2
        edge2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle6,G,self.cycle6) #nc * 4

        cycle_in = p.batched_subgraphlayer1b.cat(edge2cycle5,edge2cycle6) 
        # cycle_in = p.batched_ptensors1b.linmaps(cycle_in) #nc * 6
        cycle_new = p.batched_subgraphlayer1b.cat_channels(cycle_in,cycle_rep) #nc * 5

        cycle2edge = p.batched_subgraphlayer1b.gather_from_ptensors(cycle_new,G,self.edge) #nc * 10

        edge_out = self.edge_mlp(torch.cat([edge_rep.torch(),cycle2edge.torch()],-1)) # nc * 11 -> nc
        # edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        cycle_out = self.cycle_mlp(cycle_new.torch()) #nc * 5 -> nc
        cycle_out = p.batched_subgraphlayer1b.like(cycle_rep,cycle_out)
        return edge_out, cycle_out

class ConvLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout) -> None:
        super().__init__()
        self.mlp = Sequential(
            Linear(2*hidden_dim,hidden_dim,False),
            BatchNorm1d(hidden_dim),
            ReLU()
        )

        # self.edge2edge = Edge_edge(hidden_dim)
        self.edge_node = Edge_node(hidden_dim)
        self.edge_cycle = Edge_Cycle(hidden_dim)

    def forward(self, node_rep: ptens0_type,edge_rep: ptens1_type, cycle_rep: ptens1_type,G):
        # edge_out = self.edge2edge(edge_rep, G)
        node_out, edge_out = self.edge_node(node_rep,edge_rep,G)
        edge_out2, cycle_out = self.edge_cycle(edge_rep, cycle_rep, G)
        #edge_out, edge_out2 is torch.tensor

        edge_out = self.mlp(torch.cat([edge_out,edge_out2],-1))
        edge_out = p.batched_subgraphlayer1b.like(edge_rep,edge_out)
        return node_out, edge_out, cycle_out


class Net_Andrew_batch_2(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dense_dim: int, num_layers: int, dropout,
                 readout: Callable[[torch.Tensor,torch.Tensor],torch.Tensor], device = 'cuda') -> None:
        print("Using Net_Andrew_2, added node rep")
        super().__init__()
        self.node_embedding = torch.nn.Embedding(22,embedding_dim)
        self.edge_embedding = torch.nn.Embedding(4,embedding_dim)
        
        self.conv_layers = ModuleList([ConvLayer(hidden_dim,dropout) for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear((1 + num_layers) * 3 * hidden_dim,dense_dim),
                                    BatchNorm1d(hidden_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim),
                                    BatchNorm1d(hidden_dim),
                                    ReLU()
                                    )
        self.node_mlp1 = p.Linear(embedding_dim,hidden_dim * _inner_mlp_mult)
        self.node_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.edge_mlp1 = p.Linear(2 * embedding_dim,hidden_dim * _inner_mlp_mult)
        self.edge_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.cycle_mlp1 = p.Linear(2 * embedding_dim,hidden_dim * _inner_mlp_mult)
        self.cycle_mlp2 = p.Linear(hidden_dim * _inner_mlp_mult,hidden_dim)

        self.dropout = dropout
        self.lin = Linear(dense_dim,1)

        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()
        self.cycle5=p.subgraph.cycle(5)
        self.cycle6=p.subgraph.cycle(6)

        self.device = device
    
    def forward(self, data) -> torch.Tensor:
        # G=p.batched_ggraph.from_edge_index(data.edge_index.float().to('cpu'),data.num_nodes)
        # print("idx size:", data.idx.shape)
        # print("idx is:", data.idx[:10])
        # print("graph size:", data.graph_size.shape)
        # print("graph is:", data.graph_size[:10])
        # print("edge size:", data.edge_size.shape)
        # print("edge is:", data.edge_size[:10])
        G_test = p.batched_ggraph.from_cache([0])
        print("test finished. G_test:", G_test)
        G = p.batched_ggraph.from_cache(data.idx.tolist())
        

        edges=data.edge_index.transpose(1,0).tolist()
        node_rep = p.batched_ptensors0b.from_matrix(self.node_embedding(data.x.flatten()), data.graph_size.tolist())

        
        # print("edge attr dimension:", data.edge_attr.shape)

        # print("edge attr flatten dimension:", data.edge_attr.flatten().shape)
        edge_emb = self.edge_embedding(data.edge_attr.flatten())
        print("edge_embedding dimension: ", edge_emb.shape)
        print("data edge size dimension:", data.edge_size.shape)
        print("sum of edge sizes:", data.edge_size.sum())

        # print("data dimension: ", data.x.shape)

        # print("data graph sizes", data.graph_size.sum())
        # print("data sum of edge sizes: ", data.edge_size.sum())

        # edge_rep_raw_ptens = p.ptensors0b.from_matrix(self.edge_embedding(data.edge_attr.flatten()),edges)

        # edge_rep = p.batched_ptensors0b.from_matrix(edge_rep_raw_ptens, data.edge_size.tolist()) 
        # print(f"edge_rep: {edge_rep}")

        node2edge = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.edge)
        # edge2edge = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep,G,self.edge)


        edge_rep = p.batched_subgraphlayer1b.cat_channels(node2edge,node2edge) #nc * 2
        edge_rep = self.edge_mlp1(edge_rep).relu()
        edge_rep = self.edge_mlp2(edge_rep).relu() #nc = hidden_dim

        node2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.cycle5)
        node2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(node2cycle5,G,self.cycle5) #nc * 2
        node2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.cycle6)
        node2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(node2cycle6,G,self.cycle6) #nc * 2

        cycle_rep = p.batched_subgraphlayer1b.cat(node2cycle5,node2cycle6)
        cycle_rep = self.cycle_mlp1(cycle_rep).relu()
        cycle_rep = self.cycle_mlp2(cycle_rep).relu() #nc = hidden_dim

        # node_rep = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep,G,self.node)
        node_rep = self.node_mlp1(node_rep).relu()
        node_rep = self.node_mlp2(node_rep).relu() #nc = hidden_dim

        node_out_1 = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node)
        node_out_2 = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node)
        reps = torch.cat([node_rep.torch(), node_out_1.torch(),node_out_2.torch()],-1) #nc = hidden_dim * 3
        for conv_layer in self.conv_layers:
            node_rep, edge_rep, cycle_rep = conv_layer(node_rep, edge_rep, cycle_rep, G)
            node_out_1 = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node)
            node_out_2 = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep,G,self.node)
            reps = torch.cat([reps, node_rep.torch(), node_out_1.torch(),node_out_2.torch()],-1) #nc += hidden_dim * 3

        reps = self.readout(reps,data.batch) # nc = hidden_dim * 2 * (1 + num_layer)

        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps).flatten()
    
# ---- Main --------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print("test")
    device = 'cuda'
    counter = Counter(0)
    train_loader = get_dataloader('zinc', 'train', 12, device, cache=True, transform=GraphTransform(), pre_transform=PreAddIndex(counter))

    model = Net_Andrew_batch_2(embedding_dim=2,hidden_dim=4,dense_dim=4,num_layers=4,dropout=0.,readout=global_add_pool).to(device)

    loss_fn = torch.nn.L1Loss()
    for data in train_loader:
        print("batch is:", data)
        data.to(device)
        pred = model(data)
        loss = loss_fn(pred,data.y)
        print(f"loss: {loss}")
        loss.backward()
        break
