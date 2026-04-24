import os
import torch
import pickle as pkl
from tqdm import tqdm
import networkx as nx 
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from .graph_enumerations import cycle_mapping_index, clique_mapping_index

class HomDataset(Dataset):
    """Homomorphism dataset class. Only support cycles and cliques at the moment."""
    def __init__(
        self, 
        name: str,
        root_path: str = './data/',
        cycle_length_bound: int = 10, 
        clique_size_bound: int = 5,
        transform = None
        ):
        self.name = name
        self.root_path = root_path
        self.cycle_length_bound = cycle_length_bound
        self.clique_size_bound = clique_size_bound
        self.num_classes = None
        self.raw_data = None
        self.num_features = None
        self.is_pyg = False
        self.transform = None

        if os.path.exists(os.path.join(root_path, name+'.pt')):
            print(f"Loading processed data from {os.path.join(root_path, name+'.pt')}...") 
            self.num_classes, self.num_features, self.data = torch.load(os.path.join(root_path, name+'.pt'))
        else:
            print(f"Reading raw data from {os.path.join(root_path, name+'.pkl')}...")
            if name.lower() in ['exp', 'csl', 'sr25']:
                with open(os.path.join(root_path, name+'.pkl'), 'rb') as f:
                    self.num_classes, self.num_features, self.raw_data = pkl.load(f)
            else:  # Assume TUDataset
                self.raw_data = TUDataset(root=self.root_path, name=self.name, use_node_attr=True)
                self.num_features = self.raw_data.num_node_features
                self.num_classes = self.raw_data.num_classes
                self.is_pyg = True
            self.data = []
            self.process()

    def process(self):
        assert self.raw_data is not None, 'Failed to load raw data.'

        mapping_index_dict = dict()
        _tmp_data = []
        _all_patterns = set()

        desc_str = 'Creating homomorphism mappings...'
        for i, g in enumerate(tqdm(self.raw_data, desc=desc_str)):
            if self.is_pyg:
                x = g.x 
                y = g.y
                g = to_networkx(g, to_undirected=True)
            else:
                x = g.graph['x']
                y = g.graph['y']
            mappings = cycle_mapping_index(g, length_bound=self.cycle_length_bound)
            mappings.update(clique_mapping_index(g, size_bound=self.clique_size_bound))
            _all_patterns.update(mappings.keys())
            _tmp_data.append((x, y, mappings))

        desc_str = 'Finalizing...'
        for (x, y, mappings) in tqdm(_tmp_data, desc=desc_str):
            for k in _all_patterns:
                if k not in mappings:
                    mappings[k] = None 
            gdata = Data(x=x, y=y, mapping_index_dict=mappings)
            self.data.append(gdata)

        torch.save((self.num_classes, self.num_features, self.data), os.path.join(self.root_path, self.name+'.pt'))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def hom_collate(data_list):
    batch = Batch()
    keys = data_list[0].keys()
    num_nodes_list = [data.num_nodes for data in data_list]
    node_offsets = torch.cumsum(torch.tensor([0] + num_nodes_list[:-1]), dim=0)
    mapping_keys = list(data_list[0]['mapping_index_dict'].keys())
    batch['mapping_index_dict'] = dict()
    batch.batch = torch.cat([torch.full((num_nodes,), i, dtype=torch.long) for i, num_nodes in enumerate(num_nodes_list)])
    for mapping_key in mapping_keys:
        stacked_tensors = []
        for i, data in enumerate(data_list):
            tensor = data['mapping_index_dict'][mapping_key]
            if tensor is not None:
                adjusted_tensor = tensor + node_offsets[i]
                stacked_tensors.append(adjusted_tensor)
        if stacked_tensors:
            batch['mapping_index_dict'][mapping_key] = torch.cat(stacked_tensors, dim=0)
    batch['x'] = torch.cat([data['x'] for data in data_list], dim=0)
    batch['y'] = torch.cat([data['y'] for data in data_list], dim=0)
    batch.batch_size = len(data_list)

    return batch


class HomDataLoader(DataLoader):
    """Simple wrapper for a dataloader with `hom_collate` function"""
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch = None,
        exclude_keys = None,
        collate_fn = hom_collate,
        **kwargs,
        ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

