import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from typing import Literal
import numpy as np
import ptens as p
import tqdm
from typing import Union
from .Transforms import StandardPreprocessing, encoding_flags, label_type
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
from torch_geometric.transforms import Compose
from torch_geometric.transforms import BaseTransform
from torch.utils.data import ConcatDataset
from torch_geometric.data import InMemoryDataset
from ogb.graphproppred import PygGraphPropPredDataset

_tu_datasets = ['MUTAG','ENZYMES','PROTEINS','COLLAB','IMDB-BINARY','REDDIT_BINARY','IMDB-MULTI','NCI1','NCI109','PTC_MR']
tu_dataset_type = Literal['MUTAG','ENZYMES','PROTEINS','COLLAB','IMDB-BINARY','REDDIT_BINARY','IMDB-MULTI','NCI1','NCI109','PTC_MR']
tu_dataset_type_list: list[str] = ['MUTAG','ENZYMES','PROTEINS','COLLAB','IMDB-BINARY','REDDIT_BINARY','IMDB-MULTI','NCI1','NCI109','PTC_MR']
dataset_type_list: list[str] = ['ZINC','ZINC-Full','ogbg-molhiv','peptides-struct','graphproperty','ogbg-moltox21',*tu_dataset_type_list]
dataset_type = Union[
    Literal['ZINC','ZINC-Full','ogbg-molhiv','peptides-struct','graphproperty','ogbg-moltox21'],tu_dataset_type
    ]

def cache_graph(dataset):
    for data in dataset:
        p.ggraph.from_edge_index(data.edge_index.float()).cache(data.idx)

class DataHandler():
    ds : InMemoryDataset #store the whole dataset
    splits: dict[Literal['train','val','test'],InMemoryDataset] #store the splits of dataset, get train, val, test via here
    batch_size: dict[Literal['train','val','test'],int]
    ltype: label_type
    def __init__(self, root: str, train_batch_size, val_batch_size, test_batch_size, pre_transform, ltype: label_type, node_enc: encoding_flags, edge_enc: encoding_flags, ds_name=None) -> None:
        # self.transform = PreprocessTransform()
        self.ltype = ltype
        self.pre_transform = Compose(
            [
                StandardPreprocessing(ltype,node_enc,edge_enc, ds_name=ds_name),
                pre_transform
            ]
        )
        self.root = root
        self.batch_size = {
            'train' : train_batch_size,
            'val' : val_batch_size,
            'test' : test_batch_size
        }
    def _get_dataloader(self, split: Literal['train','val','test'],shuffle: bool=False, distributed: bool=False):
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(self.splits[split], shuffle=True)
            return DataLoader(dataset=self.splits[split],batch_size=self.batch_size[split],shuffle=False, num_workers=16, pin_memory=True, sampler=sampler)
        else:
            return DataLoader(dataset=self.splits[split],batch_size=self.batch_size[split],shuffle=shuffle, num_workers=8, pin_memory=True)
    def set_fold_idx(self, idx: int):
        pass
    def prepare_data(self):...

    def train_dataloader(self, distributed:bool = False):
        return self._get_dataloader('train',True, distributed=distributed)
    def test_dataloader(self, distributed:bool = False):
        return self._get_dataloader('test', distributed=distributed)
    def val_dataloader(self, distributed:bool = False):
        return self._get_dataloader('val', distributed=distributed)
    
class OGBGDatasetHandler(DataHandler):
    def __init__(self, root: str, ds_name, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform) -> None:
        if ds_name == 'ogbg-molhiv':
            ltype = 'single-dim' #single-dim for OGB.HIV
        else:
            ltype = 'multi-label' #multi-class for OGB.Tox21
        super().__init__(root, train_batch_size, val_batch_size, test_batch_size, pre_transform,ltype,'OGB','OGB', ds_name=ds_name)
        self.ds_name = ds_name
        self.prepare_data()


    def prepare_data(self):
        ds = PygGraphPropPredDataset(name = self.ds_name, root = self.root,pre_transform=self.pre_transform)
        cache_graph(ds)
        split_idx = ds.get_idx_split() 
        self.splits = {
            'train': ds[split_idx['train']],
            'val': ds[split_idx['valid']],
            'test': ds[split_idx['test']]
        }
        self.ds = ds

    def _get_splits(self):        
        return self.splits
class ZINCDatasetHandler(DataHandler):
    def __init__(self, root: str, ds_name, subset, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform) -> None:
        use_degree = None
        ltype = 'single-dim'
        super().__init__(root, train_batch_size, val_batch_size, test_batch_size, pre_transform,ltype,use_degree,use_degree, ds_name=ds_name)
        self.ds_name = ds_name
        self.subset = subset
        self.prepare_data()

    def filter_valid_data(self, dataset):
        return [data for data in dataset if data is not None] 
    def prepare_data(self):
        self.splits = {
            split : ZINC(self.root + '/ZINC', self.subset, split, pre_transform=self.pre_transform)
            for split in ['train','val','test']
        }
        for split in ['train','val','test']:
            print(f"len of {split} dataset: {len(self.splits[split])}")
            print(f"data.idx: {self.splits[split][0].idx}")
            cache_graph(self.splits[split])
            
        ds = self.splits['train']
        ds.num_node_attr = 28
        ds.num_edge_attr = 4
        print(f"num_node_attr: {ds.num_node_attr}, num_edge_attr: {ds.num_edge_attr}")        
        self.ds = ds

    def _get_splits(self):        
        return self.splits
class TUDatasetHandler(DataHandler):
    def __init__(self, root: str, ds_name: tu_dataset_type, train_batch_size: int, val_batch_size: int, test_batch_size: int, pre_transform, num_folds=None, seed=0) -> None:
        # use_degree: encoding_flags = None if ds_name == 'REDDIT_BINARY' else "degree"
        use_degree = 'degree'
        # use_degree = None
        if ds_name in ['COLLAB','IMDB-MULTI','ENZYMES']:
            ltype = 'multi-class'
        else:
            ltype = 'single-dim'
        super().__init__(root, train_batch_size, val_batch_size, test_batch_size, pre_transform,ltype,use_degree,use_degree, ds_name=ds_name)
        self.num_folds = num_folds
        self.seed = seed
        self.ds_name = ds_name
        self.split_idx = 0
        self.prepare_data()

    def prepare_data(self):
        ds = TUDataset(self.root,self.ds_name,pre_transform=self.pre_transform,use_edge_attr=True,use_node_attr=True)
        cache_graph(ds)
        ds.num_node_attr = ds.x.max().item() + 1
        ds.num_edge_attr = ds.edge_attr.max().item() + 1
        print(f"num_node_attr: {ds.num_node_attr}, num_edge_attr: {ds.num_edge_attr}")

        # get splits that each class maintain the same occurences in each fold
        skf = StratifiedKFold(self.num_folds,shuffle=True,random_state=self.seed) 
        self.split_indices = list(skf.split(np.zeros(len(ds)),ds.y))
        self.ds = ds
    
    def set_fold_idx(self, idx: int):
        self.split_idx = idx
        self.splits = self._get_splits()

    def _get_splits(self):        
        # adapted from GIN
        train_idx, test_idx = self.split_indices[self.split_idx] #get the split for one fold
        print(f"train_idx: {train_idx}\ntest_idx: {test_idx}")
        return {
                'train' : self.ds[train_idx],
                'val' : self.ds[test_idx],
                'test' : self.ds[test_idx],
            }
    
def get_data_handler(pre_transform, args) -> DataHandler:
    ds_name = args.ds_name
    ds_path = './data/'
    handlerArgs = {
        'root' : ds_path,
        'train_batch_size' : args.batch_size,
        'val_batch_size' : args.eval_batch_size,
        'test_batch_size' : args.eval_batch_size,
        'pre_transform' : pre_transform,
    }
    if ds_name in ['ZINC','ZINC-Full']:
        data_handler = ZINCDatasetHandler(ds_name=ds_name,subset = ds_name != 'ZINC-Full',**handlerArgs)
    elif ds_name in ['ogbg-molhiv', 'ogbg-moltox21']:
        data_handler = OGBGDatasetHandler(ds_name=ds_name,**handlerArgs)
    elif ds_name in _tu_datasets:
        data_handler = TUDatasetHandler(ds_name=ds_name,num_folds=args.num_folds,seed=0,**handlerArgs)
        data_handler.set_fold_idx(args.fold_idx)
    else:
        raise NotImplementedError(ds_name)
    return data_handler