import torch
from torch.nn import functional as F, Sequential, ModuleList, Linear, BatchNorm1d, ReLU, Parameter
from torchviz import make_dot  # for visualizing the model
import random
import numpy as np
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import ptens as p
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from .branched_cycles import get_branched_5_cycles, get_branched_6_cycles

def get_mlp(in_dim,hid_dim,out_dim,num_layers, bn_eps=1e-5,bn_momentum=0.1):
    if num_layers == 2:
        return Sequential(
            Linear(in_dim,hid_dim,False),
            BatchNorm1d(hid_dim,eps=bn_eps,momentum=bn_momentum),
            ReLU(),
            Linear(hid_dim,out_dim,False),
            BatchNorm1d(out_dim,eps=bn_eps,momentum=bn_momentum),
            ReLU()
        )
    elif num_layers > 2:
        layers = []
        layers.append(Linear(in_dim, hid_dim, False))
        layers.append(BatchNorm1d(hid_dim, eps=bn_eps, momentum=bn_momentum))
        layers.append(ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(Linear(hid_dim, hid_dim, False))
            layers.append(BatchNorm1d(hid_dim,  eps=bn_eps, momentum=bn_momentum))
            layers.append(ReLU())
        
        layers.append(Linear(hid_dim, out_dim, False))
        layers.append(BatchNorm1d(out_dim, eps=bn_eps, momentum=bn_momentum ))
        layers.append(ReLU())
        
        return Sequential(*layers)
    else:
        raise("Not Implemented Error")

class SkipConnectionBlock(torch.nn.Module):
    # TODO: debug to verify this
    def __init__(self, in_dim, out_dim, activation: torch.nn, inverse : bool = False):
        super(SkipConnectionBlock, self).__init__()
        
        if not inverse:
            self.layer = torch.nn.Sequential(
                Linear(in_dim, out_dim, False),
                BatchNorm1d(out_dim),
                activation(),
            )
        else:
            self.layer = torch.nn.Sequential(
                Linear(in_dim, out_dim, False),
                activation(),
                BatchNorm1d(out_dim),
            )
    def forward(self, x):
        return x + self.layer(x)  

def get_block(in_dim, out_dim, activation: torch.nn, inverse : bool = False, eps=1e-5, momentum=0.1):
    if not inverse:
        return torch.nn.Sequential(
            Linear(in_dim, out_dim, False),
            BatchNorm1d(out_dim, eps=eps, momentum=momentum),
            activation(),
        )
    else:
        return torch.nn.Sequential(
            Linear(in_dim, out_dim, False),
            activation(),
            BatchNorm1d(out_dim, eps=eps, momentum=momentum),
        )

def get_mlp_invertible(in_dim, hid_dim, out_dim, num_layers, inverse : bool = False, activation : str = "ReLU", bn_eps=1e-5, bn_momentum=0.1): 
    activation = getattr(torch.nn, activation)
    assert num_layers >= 1, "Number of layers should be at least 1"
    if num_layers == 1:
        print("Number of layers is 1, hidden dimension is not used")
        return get_block(in_dim, out_dim, activation, inverse, bn_eps, bn_momentum)
    else:
        layers = [get_block(in_dim, hid_dim, activation, inverse, bn_eps, bn_momentum)] + [get_block(hid_dim, hid_dim, activation, inverse, bn_eps, bn_momentum) for _ in range(num_layers - 2)] + [get_block(hid_dim, out_dim, activation, inverse, bn_eps, bn_momentum)]
        layers = torch.nn.Sequential(*layers)
        return layers
        

    
def fix_weight(model, w):
    '''fix the weight of the model to scalar w'''
    for name, param in model.named_parameters():
        param.data.fill_(w)

def visualize_architecture(model, train_loader,device='cuda'):
    make_dot(model(next(iter(train_loader)).to(device)),
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True).render("attached", format="png")
    
def write_to_file(data, filename):
    with open(filename, 'w') as f:
        print(data, file=f)

def reference_domain_with_batch(reference_domain: List[int], device=torch.device("cuda")): 
    updated_reference_domain = torch.cat([
        torch.cat([
            torch.tensor(batch, device=device, dtype=torch.float),
            torch.full((len(batch), 1), idx, dtype=torch.float, device=device)
        ],
                dim=1)
        for idx, batch in enumerate(reference_domain)
    ])
    return updated_reference_domain

def reorder_ptensors(input_ptensors, current_ref_domain, target_ref_domain):
    
    def hash_indices(indices, prime_factors):
        """Optimized hash function to uniquely map multi-dimensional indices to single integers."""
        prime_factors = prime_factors.view(-1, 1)
        return torch.matmul(indices, prime_factors).view(-1)

    # Choose prime numbers for hashing each dimension
    prime_factors = torch.tensor([10007, 10009, 10037], dtype=torch.float, device=input_ptensors.device)

    
    current_hashed = hash_indices(current_ref_domain, prime_factors).long()
    target_hashed = hash_indices(target_ref_domain, prime_factors).long()
    num_edges = current_ref_domain.size(0)
    max_hash = current_hashed.max().item() + 1
    mapping = torch.full((max_hash,), -1, dtype=torch.long, device=input_ptensors.device)
    # Use `scatter` to create mapping from current hashed indices to `edge_attr` indices
    mapping.scatter_(0, current_hashed, torch.arange(num_edges, dtype=torch.long, device=input_ptensors.device))

    # Map target hashed indices to `edge_attr` indices
    mapped_indices = mapping[target_hashed]

    # Reorder `edge_attr` using mapped indices
    reordered_edge_attr = input_ptensors[mapped_indices]
    return reordered_edge_attr

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If you are using CUDA, you should also set the following
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    # For deterministic operations (might impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    # torch.set_default_dtype(torch.float64) # TODO: bugging the training file
    

def get_subgraphs(subgraphs_list):
    subgraphs = []
    branched_5_cycles = get_branched_5_cycles()
    branched_6_cycles = get_branched_6_cycles()
    for subgraph in subgraphs_list:
        adj_matrix = generate_benzene_adjacency_matrix()
        if subgraph == "substituent 1":
            adj_matrix = add_substituent(adj_matrix, 0)
        elif subgraph == "substituent 2":
            adj_matrix = add_substituent(adj_matrix, 2)
        elif subgraph == "substituent 3":
            adj_matrix = add_substituent(adj_matrix, 4)
        elif subgraph == "napthalene":
            adj_matrix = generate_naphthalene_adjacency_matrix()
        elif subgraph in branched_5_cycles:
            subgraphs.append(branched_5_cycles[subgraph])
        elif subgraph in branched_6_cycles:
            subgraphs.append(branched_6_cycles[subgraph])
        else:
            print("Unimplemented subgraphs")
        subgraphs.append(get_subgraph(adj_matrix))
    return subgraphs


def get_subgraph(adj_matrix):
    '''
    return a ptensors subgraph
    '''
    
    edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix))
    edge_index = edge_index.float()


    graph = p.subgraph.from_edge_index(edge_index)
    return graph

def generate_benzene_adjacency_matrix():
    # Create a 6-cycle graph for benzene (C6H6)
    adj_matrix = np.zeros((6, 6), dtype=int)
    for i in range(6):
        adj_matrix[i][(i + 1) % 6] = 1
        adj_matrix[(i + 1) % 6][i] = 1
    return adj_matrix

def add_substituent(adj_matrix, position):
    # Add a substituent to a specified position in the graph
    n = len(adj_matrix)
    new_adj_matrix = np.zeros((n + 1, n + 1), dtype=int)
    new_adj_matrix[:n, :n] = adj_matrix
    new_adj_matrix[position, n] = 1
    new_adj_matrix[n, position] = 1
    return new_adj_matrix

def generate_naphthalene_adjacency_matrix():
    # Naphthalene (C10H8) - two fused benzene rings
    adj_matrix = np.zeros((10, 10), dtype=int)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (2, 6), (6, 7), (7, 8), (8, 9), (9, 4)]
    for (i, j) in edges:
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1
    return adj_matrix

def generate_pyridine_adjacency_matrix():
    # Pyridine (C5H5N) - benzene with one nitrogen atom
    adj_matrix = np.zeros((6, 6), dtype=int)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    for (i, j) in edges:
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1
    return adj_matrix

def plot_graph(adj_matrix, title="Graph"):
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.kamada_kawai_layout(G)  # You can also try nx.spring_layout(G)
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_color='lightblue', with_labels=True, node_size=700, font_weight='bold', edge_color='k')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    benzene_adj_matrix = generate_benzene_adjacency_matrix()
    benzene_1_sub_adj_matrix = add_substituent(benzene_adj_matrix, 0)
    benzene_2_sub_adj_matrix = add_substituent(benzene_1_sub_adj_matrix, 2)
    benzene_3_sub_adj_matrix = add_substituent(benzene_2_sub_adj_matrix, 4)
    naphthalene_adj_matrix = generate_naphthalene_adjacency_matrix()
    pyridine_adj_matrix = generate_pyridine_adjacency_matrix()

    plot_graph(benzene_adj_matrix, "Benzene (C6H6)")
    plot_graph(benzene_1_sub_adj_matrix, "Benzene with One Substituent")
    plot_graph(benzene_2_sub_adj_matrix, "Benzene with Two Substituents")
    plot_graph(benzene_3_sub_adj_matrix, "Benzene with Three Substituents")
    plot_graph(naphthalene_adj_matrix, "Naphthalene (C10H8)")
    plot_graph(pyridine_adj_matrix, "Pyridine (C5H5N)")

    subgraphs = ["substituent 1", "substituent 2", "substituent 3", "napthalene"],

    subgraphs = get_subgraphs(subgraphs)

    # verify the adjacency matrix of the subgraphs
    for subgraph in subgraphs:
        print(subgraph.adjacency_matrix)

