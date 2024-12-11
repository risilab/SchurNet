from __future__ import annotations
import torch
from torch import Tensor
from typing import Literal, TypeAlias, Union, List, Tuple, Dict
import ptens as p
from ptens_base import atomspack
import os
import importlib.util

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from tqdm import tqdm

import yaml
import argparse
from omegaconf import DictConfig
from hydra.utils import instantiate
import random
import numpy as np
from torchviz import make_dot  # for visualizing the model
import wandb



ptensors_layer_type : TypeAlias = Union[p.ptensors0,p.ptensors1,p.ptensors2]
domain_list_type : TypeAlias = Union[atomspack,List[List[int]]]

def get_out_dim(ds) -> int:
    multi = {
        'ogbg-moltox21'     : 12,
        'peptides-struct'   : 11,

        # tudatasets
        'ENZYMES'           : 6 ,
        'COLLAB'            : 3 ,
        'IMDB-MULTI'        : 3 ,

    }
    if ds in multi:
        return multi[ds]
    else:
        return 1



def get_path_subgraph(min_len,max_len):
    eddge_index = [[i,i+1] for i in range(min_len)]
    path_subgraphs = []
    for i in range(min_len,max_len + 1):
        path_i = p.subgraph.from_edge_index(torch.tensor(eddge_index).float().transpose(0,1))
        path_subgraphs.append(path_i)
        eddge_index.append([i,i+1])
        # print(f"i: {path_i}")
    return path_subgraphs

def get_run_path(base_dir: str = 'logs',run_id = None) -> str:
    if run_id:
        return os.path.join(base_dir,run_id), run_id
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        run_id = 0
    else:
        contents = os.listdir(base_dir)
        contents = [int(c) for c in contents]
        if not contents:
            run_id = 0
        else:
            run_id = 1 + max(*contents) if len(contents) > 1 else 1 + contents[0]
    run_path = f"{base_dir}/{run_id}/"
    os.makedirs(run_path, exist_ok=True) # in get run path, we do not create the actual run path as it may interefere with resuming runs logic
    return run_path, run_id

def get_latest_run(base_dir: str = 'logs') -> str:
    if not os.path.exists(base_dir):
        raise Exception(f"Directory {base_dir} does not exist.")
    contents = os.listdir(base_dir)
    contents = [int(c) for c in contents]
    if not contents:
        raise Exception(f"No runs found in {base_dir}")
    return str(max(*contents))

def get_group(args, run_id) -> str:
    if args.debug:
        wandb_group = 'debug'
    elif args.cross_validation:
        wandb_group = f'{args.ds_name}_{args.model_name}/10_fold_cv_{run_id}'
    else:
        wandb_group = f'{args.ds_name}_{args.model_name}/single_run'
    return wandb_group
    

def str_to_graph(name: str) -> str:
    r"""
    Takes in a string of the form "[KPC]_[1-9][0-9]*" and computes the following:
        First letter:
            K: complete graph
            P: path
            C: cycle
        Subscript: number of nodes in returned graph.
    """
    graph_type : Literal['K','P','C']= name[0]
    num_nodes = int(name[2:])
    if graph_type == 'K':
        return p.graph.from_matrix(torch.ones(num_nodes,num_nodes,dtype=torch.float32) - torch.eye(num_nodes))
    else:
        if graph_type == 'P':
            edge_index = torch.stack([torch.arange(num_nodes-1),torch.arange(1,num_nodes)])
        elif graph_type == 'C':
            edge_index = torch.stack([torch.arange(num_nodes),torch.arange(1,num_nodes + 1) % num_nodes])
        else:
            raise Exception(f'Graph type "{graph_type}" not recognized.')
        edge_index = edge_index.float()
        edge_index = torch.cat([edge_index,edge_index.flip(0)],-1)
        return p.graph.from_edge_index(edge_index)

def get_model_size(model):
        # Calculate the total size of parameters in bytes
        total_size_bytes = sum(p.numel() * p.element_size()
                            for p in model.parameters())
        # Convert bytes to megabytes
        total_size_mb = total_size_bytes / (1024**2)
        print(f"Total parameter size: {total_size_mb:.2f} MB")
        return total_size_mb
    
def get_num_batches(batch_size, total_samples):
    num_batches = total_samples // batch_size 
    if total_samples % batch_size != 0:
        num_batches += 1  
    return num_batches

def get_graph_size(data):
    '''
    Given a DataBatch, find the number of vertices of the graph
    '''
    return data.x.shape[0]

def trace_handler(p, run_path, use_wandb=False):
    output1 = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    print(output1)
    output2 = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
    print(output2)
    p.export_chrome_trace(f"{run_path}/{str(p.step_num)}.json")


def write_hyperparameters_to_yaml(args, yaml_file_path):
    # Convert args to dictionary
    args_dict = vars(args)

    # Write dictionary to yaml file
    with open(yaml_file_path, 'w') as file:
        yaml.dump(args_dict, file)

def load_hyperparameters_from_yaml(cfg: DictConfig, args: argparse.Namespace):
    '''
    cfg: DictConfig object containing the hyperparameters (refer to hydra use)
    args: argparse.Namespace object to store the hyperparameters. Need to be initialized and passed as an argument for this function
    '''
    dataset = cfg["dataset"]
    experiment_settings = cfg["experiment_settings"]
    training_hyperparameters = cfg["training_hyperparameters"]
    model = cfg["model"]


    # Load hyperparameters from DictConfig object
    for key, value in dataset.items():
        setattr(args, key, value)
    for key, value in experiment_settings.items():
        setattr(args, key, value)
    for key, value in training_hyperparameters.items():
        setattr(args, key, value)
    for key, value in model.items():
        setattr(args, key, value)
    return args


def model_factory(data_handler, args, **kwargs):
    '''
    Create the model from the configuration file
    data_handler: the data handler object used to specify the dataset
    args: the arguments parsed from the command line (TODO: remove args and only use cfg)
    kwargs: the model specific arguments loaded from model yaml file
    '''
    model_name = kwargs.pop("model_name")
    file_name = kwargs.pop("file_name")
    print("model name is:", model_name)
    print("file name is:", file_name)
    module = __import__(f"model.{file_name}", fromlist=[file_name])
    model_class = getattr(module, model_name)

    # if model intialization rquirs dataset specific parameters in args,
    if 'dataset' in model_class.__init__.__code__.co_varnames:
        dataset = data_handler.ds # warning: maybe change this to model_kwargs['dataset']
        kwargs['dataset'] = dataset
    
    if 'ds_name' in model_class.__init__.__code__.co_varnames:
        kwargs['ds_name'] = args.ds_name

    if 'out_dim' in model_class.__init__.__code__.co_varnames:
        kwargs['out_dim'] = args.out_dim

    if 'readout' in model_class.__init__.__code__.co_varnames:
        # check if torch_geometric.nn has the readout function specified by the yaml file
        import torch_geometric.nn
        readout = getattr(torch_geometric.nn, kwargs['readout'])
        if readout is None:
            raise ValueError(
                f"readout function {kwargs['readout']} not found in torch_geometric.nn"
            )
        # TODO: have a separate readout folder for readout functions
        kwargs['readout'] = readout
    # check if there are unnessary arguments in kwargs
    for key in list(kwargs.keys()):
        if key not in model_class.__init__.__code__.co_varnames:
            print(
                f"WARNING: removing {key} from kwargs. Please remove it from the config file."
            )
            kwargs.pop(key)

    # create a configuration for the model
    cfg = {"_target_": f"model.{file_name}.{model_name}", **kwargs}

    # Instantiate the model from the configuration
    model = instantiate(cfg)
    print(f"loaded model: {model_name} with kwargs: {kwargs}")

    return model


def get_run_from_resume_id(entity, project, run_id):
    api = wandb.Api()
    # entity, project = "gnn-explore", "Autobahn_ptens_richard"
    run = api.run(entity + "/" + project + "/" + run_id)
    return run


