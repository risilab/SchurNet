import torch
from torch.nn import functional as F, Sequential, ModuleList, Linear, BatchNorm1d, ReLU, Parameter 
from data_cleaning.feature_encoders import get_node_encoder, get_edge_encoder
from torch_geometric.nn.conv import GINConv, GINEConv

_inner_mlp_mult = 2

class Model_GIN(torch.nn.Module):
    '''standard GIN model from pytorch geometric'''
    def __init__(self, hidden_dim: int, dense_dim: int, out_dim: int, num_layers: int, dropout,
                 readout, ds_name,dataset,device = 'cuda') -> None:
        print("Running: Model_GIN\n\n")
        super().__init__()
        self.node_embedding = get_node_encoder(ds_name,dataset,hidden_dim)
        self.edge_embedding = get_edge_encoder(ds_name,dataset,hidden_dim)
        
        # self.init_mlp = Sequential(
        #                     Linear(hidden_dim,hidden_dim * _inner_mlp_mult,False),
        #                     BatchNorm1d(hidden_dim*_inner_mlp_mult),
        #                     ReLU(),
        #                     Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
        #                     BatchNorm1d(hidden_dim),
        #                     ReLU()
        #                 )
        
        self.conv_layers = ModuleList([GINConv(Sequential(
                                            Linear(hidden_dim,hidden_dim * _inner_mlp_mult,False),
                                            BatchNorm1d(hidden_dim*_inner_mlp_mult),
                                            ReLU(),
                                            Linear(hidden_dim * _inner_mlp_mult,hidden_dim,False),
                                            BatchNorm1d(hidden_dim),
                                            ReLU()
                                        ),train_eps=True) 
                                       for _ in range(num_layers)])
        self.readout = readout

        self.final_mlp = Sequential(Linear(hidden_dim,dense_dim,False),
                                    BatchNorm1d(dense_dim),
                                    ReLU(),
                                    Linear(dense_dim,dense_dim,False),
                                    BatchNorm1d(dense_dim),
                                    ReLU()
                                    )

        self.dropout = dropout
        self.lin = Linear(dense_dim,out_dim)
        
    def forward(self, data) -> torch.Tensor:

        node_rep = self.node_embedding(data.x.flatten())
        # node_rep = self.init_mlp(node_rep)
        # edge_rep = self.edge_embedding(data.edge_attr.flatten())
     
        for conv in self.conv_layers:
            # node_rep = conv(node_rep,data.edge_index,edge_rep)
            node_rep = conv(node_rep,data.edge_index)
            
        node_out = node_rep
        # print(f"node_out.shape: {node_out.shape}")
        reps = self.readout(node_out,data.batch)
        # print(reps)

        reps = self.final_mlp(reps) # size = [num_graphs ,dense_dim]
        # reps = F.dropout(reps,self.dropout,self.training)
        
        return self.lin(reps) # size = [num_graphs,out_dim]
    
