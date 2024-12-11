from typing import List, Optional
import torch
import ptens as p
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, DataLoader
from typing import Literal
from typing import overload
from typing import overload, Union
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree
from ogb.utils.features import get_bond_feature_dims


class Counter():
    def __init__(self, start=0):
        self.value = start

    def get(self):
        ret = self.value
        self.value += 1
        return ret

class PreAddIndex(BaseTransform):
    def __init__(self, counter = Counter(0)) -> None:
        super().__init__()
        self.counter = counter

    def __call__(self, data: Data) -> Data:
        data.idx = self.counter.get()
        return data

def zero_degree_handle(data, deg, ds_name):
    if ds_name == 'ogbg-moltox21':
        add_virtual_edges(data, deg)
    elif ds_name == 'ogbg-molhiv':
        remove_zero_degree_nodes(data, deg)
    else:
        print("Error: ds_name is not specified when trying to deal with degree 0 nodes")


def add_virtual_edges(data, deg):
    print("data before adding pair\n", data.x)
    print("data.edge_index: ", data.edge_index)
    print("data.edge_attr: ", data.edge_attr)
    print("data degree:", data.degree)
    full_bond_feature_dims = torch.tensor([get_bond_feature_dims()])
    print("full bond feature dims", full_bond_feature_dims)
    initial_num_nodes = data.num_nodes

    mask = deg == 0

    # Create a new node for each node with degree 0
    new_x = data.x[mask]
    new_deg = data.degree[mask] + torch.tensor([1])
    deg[mask] = 1  # set the degree of those nodes with degree 0 to 1
    data.num_nodes += int(mask.sum().item(
    ))  # update the number of nodes, but making sure it is an int
    data.x = torch.cat([data.x, new_x])
    data.degree = torch.cat([data.degree, new_deg])

    # find the indices for those nodes with degree 0
    indices = torch.where(mask)[0]
    # Create a new edge for each node with degree 0, and shift the labels to match the new nodes

    new_edges = torch.stack([
        indices,
        torch.arange(initial_num_nodes, initial_num_nodes + mask.sum())
    ],
                            dim=0)
    # also add the inverse edges
    new_edges = torch.cat([new_edges, new_edges.flip(0)], dim=1)
    print("new_edges: ", new_edges)
    print("new_edges shape: ", new_edges.shape)

    # Add the new edges to the edge_index
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    
    num_new_edges = new_edges.size(1)
    new_edge_attr = full_bond_feature_dims.repeat(num_new_edges, 1)

    # for those newly added edges, we need to add the corresponding edge_attr to data.edge_attr
    data.edge_attr = torch.cat(
        [data.edge_attr, new_edge_attr], dim=0
    )  # appending the full bond feature dims to the edge_attr, as the last type of possible combination (all of them being special type that is used only for deg 0 nodes)
    print("data.edge_attr after update: ", data.edge_attr)
    print("data after adding pair\n", data.x)
    print('edge index after adding pair: ', data.edge_index)
    print("data.edge_index: ", data.edge_index)
    print("data degree:", data.degree)

def remove_zero_degree_nodes(data, deg):

    print(data)
    print("data.edge_index: ", data.edge_index)

    mask = deg > 0
    data.x = data.x[mask]
    data.degree = data.degree[mask]
    data.num_nodes = data.x.size(0)

    # Remap edge_index to account for removed nodes
    _, inverse_indices = torch.unique(data.edge_index[0], return_inverse=True)
    data.edge_index[0] = inverse_indices

    _, inverse_indices = torch.unique(data.edge_index[1], return_inverse=True)
    data.edge_index[1] = inverse_indices

    print("data after removing self-loop\n", data)
    print("data.edge_index: ", data.edge_index)

#################################################################################################################################
# dataset specific transforms
#################################################################################################################################
encoding_flags = Literal['OGB', 'degree', None]
label_type = Literal['single-dim', 'multi-class']

class StandardPreprocessing(BaseTransform):
    def __init__(self, label: label_type, node_encoding: encoding_flags = None, edge_encoding: encoding_flags = None, ds_name = None) -> None:
        """NOTE: encoding just gives some flags for how you want it to process."""
        super().__init__()
        self.label = label
        self.node_encoding = node_encoding
        self.edge_encoding = edge_encoding
        self.total_nodes_deleted = 0
        self.total_data_deleted = 0
        self.ds_name = ds_name

    def __call__(self, data):
        # node proc
        x: Tensor = data.x if data.x is not None else None
        deg: Tensor = None
        deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.int32)
        if x is None:
            if self.node_encoding == 'degree':
                x = deg
            else:
                x = torch.zeros(data.num_nodes, dtype=torch.int32)
        elif self.node_encoding != 'OGB':
            # we want to ensure standard feature form. (use single scalar instead of one-hot encoding)
            if x.ndim == 2:
                if x.size(1) > 1:
                    x = x.argmax(1)
                else:
                    x = x.flatten()
            else:
                x = x.long()
            #for OGB dataset, we don't change the data.x since we'll use OGB's AtomEncoder

        data.x = x #shape = [num_nodes, ]
        data.degree = deg

        # # Store the initial number of nodes
        initial_num_nodes = data.num_nodes

        # # # remove node with degree 0
        # if (deg == 0).any():
        #     print(data)
        #     print("data.edge_index: ", data.edge_index)
        #     print("data degree:", data.degree)

        #     mask = deg > 0
        #     data.x = data.x[mask]
        #     data.degree = data.degree[mask]
        #     data.num_nodes = data.x.size(0)


        #     # Remap edge_index to account for removed nodes
        #     _, inverse_indices = torch.unique(data.edge_index[0], return_inverse=True)
        #     data.edge_index[0] = inverse_indices

        #     _, inverse_indices = torch.unique(data.edge_index[1], return_inverse=True)
        #     data.edge_index[1] = inverse_indices

        #                 # Calculate the number of deleted nodes
        #     nodes_deleted = initial_num_nodes - data.num_nodes
        #     self.total_nodes_deleted += nodes_deleted
        #     print("data after removing self-loop\n", data)
        #     print("data.edge_index: ", data.edge_index)

        # for all nodes with degree 0, make a copy of this node and connect them
        deg_zero = False
        if (deg == 0).any():
            if self.ds_name is not None and (self.ds_name == 'ogbg-moltox21' or self.ds_name == 'ogbg-molhiv'):
                zero_degree_handle(data, deg, ds_name=self.ds_name)
            else:
                print("Error: ds_name is not specified when trying to deal with degree 0 nodes")


        # edge proc
        edge_attr: Tensor = data.edge_attr if data.edge_attr is not None else None
        if edge_attr is None:
            if self.edge_encoding == 'degree':
                if deg is None:
                    deg = degree(data.edge_index[0], data.num_nodes,dtype=torch.int32)
                #edge_index.shape = [2, num_edges]
                edge_attr = deg[data.edge_index].transpose(1, 0)
                edge_attr = edge_attr[:,0] + edge_attr[:,1] #shape = [num_edges, ]
            else:
                edge_attr = torch.zeros(data.edge_index.size(1), dtype=torch.int32)

        elif self.edge_encoding != 'OGB' and edge_attr.ndim == 2:
            if edge_attr.size(1) > 1:
                edge_attr = edge_attr.argmax(1)
            else:
                edge_attr = edge_attr.flatten()
        data.edge_attr = edge_attr #shape = [num_edges, ]

        # graph labels
        y: Tensor = data.y
        if self.label == 'multi-class': # for multi-class classification
            if y.ndim > 1:
                y = y.squeeze()
            y = y.long()
        elif self.label == 'single-dim': # for regression or binary classification
            y = y.view(-1, 1).float()
        data.y = y

        # add one self-loop for node with degree 0
        # zero_degree_nodes = torch.where(data.degree == 0)[0]
        # if zero_degree_nodes.size(0) > 0:
        #     print(data)
        #     data.edge_index = torch.cat([data.edge_index, zero_degree_nodes.view(1, -1).expand(2,-1)], dim=1)
        #     data.edge_attr = torch.cat([data.edge_attr, torch.zeros(zero_degree_nodes.size(0), data.edge_attr.size(1),dtype=torch.int32)], dim=0)
        #     data.degree = degree(data.edge_index[0], data.num_nodes, dtype=torch.int32)
        #     data.num_edges = data.edge_index.size(1)
        #     print("data after adding self-loop\n", data)
        #     print("data.edge_index: ", data.edge_index)

        return data
