import torch
import ptens as p
from torch.nn import functional as F
from torch.nn import Sequential,ModuleList,Linear,BatchNorm1d,ReLU, Parameter
from typing import Callable, Union, List

from torch_geometric.nn.conv.supergat_conv import is_undirected
from loader import get_dataloader
from torch_geometric.nn import global_add_pool, global_mean_pool
from utils import Counter
from graph_cache import PreAddIndex, cache_graph, GraphTransform
from functools import reduce
from torch_geometric.utils import to_undirected

_inner_mlp_mult = 4

ptens0_type = Union[p.batched_ptensors0b,p.batched_subgraphlayer0b]
ptens1_type = Union[p.batched_ptensors1b,p.batched_subgraphlayer1b]

class NodeEdgeLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, eps: float, momentum: float, GIN_conv: bool = True) -> None:
        super().__init__()
        # self.lift_mlp = Sequential(
        #     Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
        #     BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
        #     ReLU(True),
        #     Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
        #     BatchNorm1d(hidden_channels,eps,momentum),
        #     ReLU(True)
        # )
        self.lift_mlp = p.Linear(hidden_channels, hidden_channels,False)
        # self.lvl_mlp_1 = Sequential(
        #     Linear(2*hidden_channels,hidden_channels,False),
        #     BatchNorm1d(hidden_channels,eps,momentum),
        #     ReLU(True)
        # )
        self.lvl_mlp_1 = p.Linear(2*hidden_channels,hidden_channels,False)
        
        # self.lvl_mlp_2 = Sequential(
        #     Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
        #     BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
        #     ReLU(True),
        #     Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False),
        #     BatchNorm1d(hidden_channels,eps,momentum),
        #     ReLU(True)
        # )

        self.lvl_mlp_2 = p.Linear(hidden_channels,hidden_channels,False)
        # self.lvl_mlp_2_1 = p.Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False)
        self.GIN_conv = GIN_conv
        self.epsilon1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)

        self.node = p.subgraph.trivial()
        self.edge = p.subgraph.edge()

    def forward(self, node_rep: ptens0_type, edge_rep: ptens0_type, G : p.batched_ggraph) -> tuple[ptens0_type,ptens0_type]:
        # print("inside node edge layer forward method")
        # print("node rep dimension:", node_rep.torch().shape)
        # print("edge rep dimension:", edge_rep.torch().shape)
        lift_aggr :ptens0_type = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep, G, self.edge)
        # print("lift_aggr dimension first:", lift_aggr.torch().shape)
        # lift_aggr = p.batched_subgraphlayer0b.gather_from_ptensors(lift_aggr, G, self.edge)
        # aggregate information from nodes to edges

        
        # print("lift_aggr dimension second:", lift_aggr.torch().shape)
        edge_rep = p.batched_subgraphlayer0b.cat_channels(lift_aggr,edge_rep)
        # print("edge rep dimension after cat_channels:", edge_rep.torch().shape)
        edge_rep = self.lvl_mlp_1(edge_rep).relu()

        # message pass information back to nodes
        lvl_aggr = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep, G, self.node) 
        

        # print("lvl_aggr dimension:", lvl_aggr.torch().shape)
        # print("node rep dimension:", node_rep.torch().shape)
        
        if self.GIN_conv:
            node_rep_torch = (1 + self.epsilon1) * node_rep.torch()
            node_rep = p.batched_subgraphlayer0b.like(node_rep, node_rep_torch)
            edge_rep_torch = (1 + self.epsilon2) * edge_rep.torch()
            edge_rep = p.batched_subgraphlayer0b.like(edge_rep, edge_rep_torch)
        # node_out = self.lvl_mlp_2((1 + self.epsilon1) * node_rep.torch()+ lvl_aggr.torch())
        # node_out = self.lvl_mlp_2_1(self.lvl_mlp_2_0((1 + self.epsilon1) * node_rep+ lvl_aggr))
        node_out = self.lvl_mlp_2(node_rep + lvl_aggr).relu()
        edge_out = self.lift_mlp(edge_rep + lift_aggr).relu()
            

        # edge_out = self.lift_mlp1(self.lift_mlp0((1 + self.epsilon2) * edge_rep+ lift_aggr))
        return node_out, edge_out

class EdgeCycleLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, eps: float, momentum: float, GIN_conv = True) -> None:
        super().__init__()
        # self.lift_mlp = Sequential(
        #     Linear(2*hidden_channels,hidden_channels*_inner_mlp_mult,False),
        #     BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
        #     ReLU(True),
        #     Linear(hidden_channels*_inner_mlp_mult,2 * hidden_channels,False), # TODO: add a parameter for the output dimension (depending on how mnany cycles are processed)
        #     BatchNorm1d(2 * hidden_channels,eps,momentum),
        #     ReLU(True)
        # )
        self.lift_mlp = p.Linear(2*hidden_channels,2 * hidden_channels,False)
        self.lvl_mlp_1 = p.Linear(4*hidden_channels,hidden_channels,False)
        # self.lvl_mlp_1 = Sequential(
        #     Linear(3*hidden_channels,hidden_channels,False),
        #     BatchNorm1d(hidden_channels,eps,momentum),
        #     ReLU(True)
        # )
        # self.lvl_mlp_2 = Sequential(
        #     Linear(hidden_channels,hidden_channels*_inner_mlp_mult,False),
        #     BatchNorm1d(hidden_channels*_inner_mlp_mult,eps,momentum),
        #     ReLU(True),
        #     Linear(hidden_channels*_inner_mlp_mult,hidden_channels,False), 
        #     BatchNorm1d(hidden_channels,eps,momentum),
        #     ReLU(True)
        # )
        self.lvl_mlp_2 = p.Linear(hidden_channels,hidden_channels,False)
        self.epsilon1_1 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon1_2 = Parameter(torch.tensor(0.),requires_grad=True)
        self.epsilon2 = Parameter(torch.tensor(0.),requires_grad=True)

        self.cycle5 = p.subgraph.cycle(5)
        self.cycle6 = p.subgraph.cycle(6)
        self.edge = p.subgraph.edge()
        self.node = p.subgraph.trivial()

        self.GIN_conv = GIN_conv

    def forward(self, edge_rep: ptens0_type, cycle_rep: ptens1_type, G : p.batched_ggraph) -> tuple[ptens0_type,ptens0_type]:
        # gather information from nodes to cycles
        # encode cycles

        edge2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep, G, self.cycle5)
        edge2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(edge_rep, G, self.cycle6)
        cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle5, G, self.cycle5)
        cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(edge2cycle6, G, self.cycle6)
        lift_aggr = p.batched_subgraphlayer1b.cat(cycle5,cycle6)
        

        lvl_aggr_edge = self.lvl_mlp_1(p.batched_subgraphlayer1b.cat_channels(lift_aggr, cycle_rep)).relu()
        # print("lvl aggr edge dimension:", lvl_aggr_edge.torch().shape)

        
        # propagate information from cycles to edges
        lvl_aggr = p.batched_subgraphlayer0b.gather_from_ptensors(lvl_aggr_edge, G, self.edge)

        # TODO: figure out exactly what to do here. The current implementation is unclear in terms of the channel dimensions
        # print("edge rep dimension:", edge_rep.torch().shape)
        # print("lvl aggr dimension:", lvl_aggr.torch().shape)

        intermediate = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep, G, self.edge)
        linmap_cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(intermediate, G, self.cycle5)
        linmap_cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(intermediate, G, self.cycle6)
        linmap = p.batched_subgraphlayer1b.cat(linmap_cycle5, linmap_cycle6)
        
        if self.GIN_conv:
            edge_rep_torch = (1 + self.epsilon1_1) * edge_rep.torch()
            edge_rep = p.batched_subgraphlayer0b.like(edge_rep, edge_rep_torch)
            lvl_aggr_torch = (1 + self.epsilon1_2) * lvl_aggr.torch()
            lvl_aggr = p.batched_subgraphlayer0b.like(lvl_aggr, lvl_aggr_torch)

            linmap_torch = (1 + self.epsilon2) * linmap.torch()
            linmap = p.batched_ptensors1b.like(linmap, linmap_torch)



        # edge_out = self.lvl_mlp_2((1 + self.epsilon1_1) * edge_rep.torch() + (1 + self.epsilon1_2) * lvl_aggr.torch())
        edge_out = self.lvl_mlp_2(edge_rep +  lvl_aggr).relu()
        
        # # TODO: is this linmap1_1 in ptensors?
        # # How do we implement a 
        # print("cycle rep dimension:", cycle_rep.torch().shape)
        # intermediate = p.batched_subgraphlayer1b.gather_from_ptensors(cycle_rep, G, self.edge)
        # print("intermediate dimension:", intermediate.torch().shape)
        # intermediate0 = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep, G, self.edge)
        # print("intermediate0 dimension:", intermediate0.torch().shape)
        # linmap_cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(intermediate, G, self.cycle5)
        # print("linmap cycle5 dimension:", linmap_cycle5.torch().shape)
        # linmap_cycle5_0 = p.batched_subgraphlayer1b.gather_from_ptensors(intermediate0, G, self.cycle5)
        # print("linmap cycle5_0 dimension:", linmap_cycle5_0.torch().shape)
        # linmap_cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(intermediate, G, self.cycle6)
        # print("linmap cycle6 dimension:", linmap_cycle6.torch().shape)
        # linmap_cycle6_0 = p.batched_subgraphlayer1b.gather_from_ptensors(intermediate0, G, self.cycle6)
        # print("linmap cycle6_0 dimension:", linmap_cycle6_0.torch().shape)
        # print("cat channels dimension:", p.batched_subgraphlayer1b.cat_channels(linmap_cycle5, linmap_cycle6).torch().shape)
        # cycle_out = self.lift_mlp((1 + self.epsilon2) * p.batched_subgraphlayer1b.cat_channels(linmap_cycle5, linmap_cycle6) + lift_aggr)
        #
        # cycle_out = self.lift_mlp((1 + self.epsilon2) * linmap.torch() + lift_aggr.torch())

        cycle_out =  self.lift_mlp(linmap + lift_aggr).relu()

        return edge_out, cycle_out 


    


class ModelLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, dropout: float,eps: float, momentum: float, reduce_ptensors: str, include_cycle_cycle: bool, GIN_conv: bool = True) -> None:
        super().__init__()
        self.node_edge = NodeEdgeLayer(hidden_channels, eps, momentum, GIN_conv=GIN_conv)
        
        self.edge_cycle = EdgeCycleLayer(hidden_channels, eps, momentum, GIN_conv=GIN_conv)
        
        # self.mlp = Sequential(
        #     Linear(2*hidden_channels,hidden_channels,False),
        #     BatchNorm1d(hidden_channels,eps,momentum),
        #     ReLU(True)
        # )
        self.mlp = p.Linear(2*hidden_channels,hidden_channels,False)
        self.dropout = dropout
        self.include_cycle_cycle = include_cycle_cycle
        # if include_cycle_cycle:
            # self.cycle_cycle = TransferLayer1_1(hidden_channels,eps,momentum)
            # self.cycle_cycle = CycleCycleLayer(hidden_channels, eps, momentum, reduce_ptensors)
            # self.mlp_cycle = Sequential(
            #     Linear(2*hidden_channels,hidden_channels,False),
            #     BatchNorm1d(hidden_channels,eps,momentum),
            #     ReLU(True)
            # )
        
    def forward(self, node_rep: torch.Tensor, edge_rep: torch.Tensor, cycle_rep: torch.Tensor, G:p.batched_ggraph) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        node_out, edge_out_1 = self.node_edge(node_rep, edge_rep, G)
        edge_out_2, cycle_out = self.edge_cycle(edge_rep, cycle_rep, G)


        # edge_out = self.mlp(torch.cat([edge_out_1,edge_out_2],-1))
        edge_out = self.mlp(p.batched_subgraphlayer0b.cat_channels(edge_out_1,edge_out_2)).relu()


        # if self.include_cycle_cycle:
            # cycle_out_2 = self.cycle_cycle(cycle_rep,data[(('cycles','cycles'),1)])
            # cycle_out = self.mlp_cycle(cycle_out_2)

        # node_out = F.dropout(node_out,self.dropout,self.training)
        # edge_out = F.dropout(edge_out,self.dropout,self.training)
        # cycle_out = F.dropout(cycle_out,self.dropout,self.training)
        #
        return node_out, edge_out, cycle_out

class Model_Simple(torch.nn.Module):
    def __init__(self, embedding_dim=64,  hidden_channels=64, conv_layers_num=2, eps = 0.1, dropout = 0.5, momentum = 0.1, reduce_ptensors = "sum", include_cycle_cycle = False, GIN_conv = True) -> None:
        super().__init__()
        assert hidden_channels == embedding_dim # TODO: maybe just disregard the hidden_channels parameter
        self.node_embedding = torch.nn.Embedding(22, embedding_dim)
        self.edge_embedding = torch.nn.Embedding(4, embedding_dim)

        self.final_mlp = Sequential(
            Linear(4 * hidden_channels, 4 * hidden_channels), # TODO: fix dimension
            BatchNorm1d(4 * hidden_channels, eps, momentum),
            ReLU(),
            Linear(4 * hidden_channels, 1)
        )
        # self.final_mlp = p.Linear(4 * hidden_channels, 1, False)

        self.eps = Parameter(torch.Tensor([eps]))
        self.layers = ModuleList([
            ModelLayer(hidden_channels, dropout, eps, momentum, reduce_ptensors=reduce_ptensors, include_cycle_cycle=include_cycle_cycle, GIN_conv=GIN_conv) for _ in range(conv_layers_num)
            ]) 
        

        self.edge = p.subgraph.edge()
        self.node = p.subgraph.trivial()
        self.cycle5 = p.subgraph.cycle(5)
        self.cycle6 = p.subgraph.cycle(6)

        self.cycle_mlp = p.Linear(2*hidden_channels,2 * hidden_channels,False)
        # self.eps_node = Parameter(torch.tensor(0.),requires_grad=True)
        # self.eps_edge = Parameter(torch.tensor(0.),requires_grad=True)
        # self.eps_cycle = Parameter(torch.tensor(0.),requires_grad=True)
        #
        # self.node_mlp = Sequential(
        #     Linear(hidden_channels, hidden_channels),
        #     BatchNorm1d(hidden_channels),
        #     ReLU(),
        #     Linear(hidden_channels, hidden_channels)
        # )
        #
        # self.edge_mlp = Sequential(
        #     Linear(hidden_channels, hidden_channels),
        #     BatchNorm1d(hidden_channels),
        #     ReLU(),
        #     Linear(hidden_channels, hidden_channels)
        # )
        # self.cycle_mlp = Sequential(
        #     Linear(hidden_channels, hidden_channels),
        #     BatchNorm1d(hidden_channels),
        #     ReLU(),
        #     Linear(hidden_channels, hidden_channels),
        # )

    def forward(self, data):

        graphid_list = data.idx.tolist()
        G=p.batched_ggraph.from_cache(graphid_list)
        
        # read from cached graphs
        
        node_rep = p.batched_subgraphlayer0b.from_vertex_features(graphid_list, self.node_embedding(data.x.flatten()))
        edge_rep = p.batched_subgraphlayer0b.from_edge_features(graphid_list, self.edge_embedding(data.edge_attr.flatten()))

        # print("test gather operation 0b and 1b")
        node2edge0 = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep, G, self.edge)
        node2edge1 = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep, G, self.edge)
        # print("node2edge0 dimension:", node2edge0.torch().shape)
        # print("node2edge1 dimension:", node2edge1.torch().shape)
        
        # make a copy of the node_rep
        # node_in = node_rep.clone()
        # edge_in = edge_rep.clone()
        
        # print("node_rep type:", type(node_rep))
        # print("node rep dimension:", node_rep.torch().shape)


        # TODO: compute cycle rep now
        # print("cycle 5 type:", type(cycle5))
        # print("cycle 6 type:", type(cycle6))
        # print("cycle 5 dimension:", cycle5.torch().shape)
        # print("cycle 6 dimension:", cycle6.torch().shape)
        node2cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep, G, self.cycle5)
        node2cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(node_rep, G, self.cycle6)

        cycle5 = p.batched_subgraphlayer1b.gather_from_ptensors(node2cycle5, G, self.cycle5)
        cycle6 = p.batched_subgraphlayer1b.gather_from_ptensors(node2cycle6, G, self.cycle6)

        cycle_rep = p.batched_subgraphlayer1b.cat(cycle5,cycle6)
        # print("cycle rep dimension:", cycle_rep.torch().shape)
        # cycle_rep = cycle5
    
        # cycle_in = cycle_rep.clone()

        # performing message passing
        for layer in self.layers: 
            # node_out, edge_out, cycle_out = layer(node_rep, edge_rep, cycle_rep, G)
            # node_rep = p.batched_subgraphlayer0b.like(node_rep, node_out)
            # edge_rep = p.batched_subgraphlayer0b.like(edge_rep, edge_out)
            # TODO: debug this following line
            # cycle_rep = p.batched_subgraphlayer1b.like(self.cycle_mlp(cycle_rep).relu(), cycle_out)
            # cycle_rep = p.batched_ptensors1b.like(cycle_rep, cycle_out)
            node_rep, edge_rep, cycle_rep = layer(node_rep, edge_rep, cycle_rep, G)

        # add edge2node and node2node
        # edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep,G,self.node)
        #
        # edge_rep = p.batched_subgraphlayer1b.cat_channels(node2edge, edge2edge)
        

        # finalizing model, stacking the outputs
        # maybe don't add epsilons here
        # node_out = self.node_mlp((1 + self.eps_node) * node_in + node_rep)
        # edge_out = self.edge_mlp((1 + self.eps_edge) * edge_in + edge_rep)
        # cycle_out = self.cycle_mlp((1 + self.eps_cycle) * cycle_in + cycle_rep)
        
        node_out1 = p.batched_subgraphlayer0b.gather_from_ptensors(edge_rep, G, self.node) # nc
        node_out2 = p.batched_subgraphlayer0b.gather_from_ptensors(cycle_rep, G, self.node) # nc * 2

        reps = torch.cat([node_rep.torch(), node_out1.torch(),node_out2.torch()],-1) # nc + nc + nc * 2
        # reps = p.batched_subgraphlayer0b.cat_channels(node_rep, node_out1)
        # reps = p.batched_subgraphlayer0b.cat_channels(reps, node_out2)
        reps = global_add_pool(reps, data.batch)
        # reps = p.batched_subgraphlayer0b.like(reps, reps_torch)
        out = self.final_mlp(reps)
        return out.flatten()

        
        

        


if __name__ == "__main__":
    from loader import get_dataloader
    from utils import Counter
    counter = Counter(0)
    dataloader = get_dataloader(
                        'zinc',
                        'train',
                        batch_size=32,
                        pin_memory_device='cuda',
                        cache=True,
                        transform=GraphTransform(is_undirected=True),
                        pre_transform=PreAddIndex(counter), 
                        ratio=0.1)
    model = Model_Simple(embedding_dim=64, hidden_channels=64, conv_layers_num=2, eps = 0.1, dropout = 0., momentum = 0.1, reduce_ptensors = "sum", include_cycle_cycle = False).to('cuda')
    # print out number of parameters in the Model_Simple
    print("number of parameters:", sum(p.numel() for p in model.parameters()))
    for data in dataloader:
        data.to('cuda')
        out = model(data)
        loss = F.mse_loss(out, data.y)
        print("loss is: ", loss.detach().item())
        break
    print("done")
    test_linear = p.Linear(1,1,False) 
    
    random_input = torch.rand(1,1)
    out = test_linear(random_input).relu()
