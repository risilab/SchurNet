from .data_loader import _tu_datasets
from torch.nn import Embedding
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_bond_feature_dims
import torch


class ToxBondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(ToxBondEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        full_bond_features_dims = get_bond_feature_dims()
        for i, dim in enumerate(full_bond_features_dims):
            # add one additional bond type, for manmade
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
        return bond_embedding


def get_node_encoder(ds_name,dataset,hidden_dim):
    if ds_name in _tu_datasets:
        num_node_attr = dataset.num_node_attr
        return Embedding(num_node_attr,hidden_dim)
    elif ds_name in ['ZINC','ZINC-Full']:
        return Embedding(dataset.num_node_attr,hidden_dim)
    elif ds_name == 'ogbg-molhiv':
        return AtomEncoder(hidden_dim)
    elif ds_name == 'ogbg-moltox21':
        return AtomEncoder(hidden_dim)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
        
    
def get_edge_encoder(ds_name,dataset,hidden_dim):
    if ds_name in _tu_datasets:
        num_edge_attr = dataset.num_edge_attr
        return Embedding(num_edge_attr,hidden_dim)
    elif ds_name in ['ZINC','ZINC-Full']:
        return Embedding(dataset.num_edge_attr,hidden_dim)
    elif ds_name == 'ogbg-molhiv':
        return BondEncoder(hidden_dim)
    elif ds_name == 'ogbg-moltox21':
        # return BondEncoder(hidden_dim)
        return ToxBondEncoder(hidden_dim) # for adding the manmade bond type
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
