import torch
from ..model_utils import get_mlp, get_mlp_invertible
import ptens as p
_inner_mlp_mult = 2



class Edge_node(torch.nn.Module):
    '''Edge and edge + node message passing'''
    def __init__(self, rep_dim) -> None:
        super().__init__()

        self.node_mlp = get_mlp_invertible(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)
        self.edge_mlp_0 = get_mlp_invertible(2 * rep_dim,rep_dim * _inner_mlp_mult,rep_dim,2)

        
        self.node=p.subgraph.trivial()
        self.edge=p.subgraph.edge()

    def forward(self,node_rep, edge_rep,G,data):
        node2edge = p.batched_subgraphlayer0b.gather_from_ptensors(node_rep,G,self.edge) #nc

        edge_out = self.edge_mlp_0(torch.cat([edge_rep.torch(),node2edge.torch()],dim=-1))
        edge_out_ptens = p.batched_subgraphlayer0b.like(edge_rep,edge_out)
        
        edge2node = p.batched_subgraphlayer0b.gather_from_ptensors(edge_out_ptens,G,self.node) 

        node_out = self.node_mlp(torch.cat([node_rep.torch(),edge2node.torch()],dim = -1)) 
        return node_out, edge_out