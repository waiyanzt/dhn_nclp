import os
import pickle as pkl

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from .graph_enumerations import clique_mapping_index, cycle_mapping_index


class HomDataset(Dataset):
    """Original DHN graph-classification dataset adapter.

    PyG ``TUDataset`` graphs (including ENZYMES and PROTEINS) and the
    repository's pickled synthetic datasets are converted to ``Data`` objects
    with precomputed cycle/clique homomorphism mappings.  The processed tuple
    is cached as ``<root_path>/<name>.pt``, matching the original DHN layout.
    """

    def __init__(
        self,
        name: str,
        root_path: str = "./data/",
        cycle_length_bound: int = 10,
        clique_size_bound: int = 5,
        transform=None,
    ):
        self.name = name
        self.root_path = root_path
        self.cycle_length_bound = cycle_length_bound
        self.clique_size_bound = clique_size_bound
        self.num_classes = None
        self.raw_data = None
        self.num_features = None
        self.is_pyg = False
        self.transform = transform

        cache_path = os.path.join(root_path, name + ".pt")
        if os.path.exists(cache_path):
            print(f"Loading processed data from {cache_path}...")
            try:
                payload = torch.load(cache_path, weights_only=False)
            except TypeError:
                payload = torch.load(cache_path)
            self.num_classes, self.num_features, self.data = payload
            return

        if name.lower() in {"exp", "csl", "sr25"}:
            raw_path = os.path.join(root_path, name + ".pkl")
            print(f"Reading raw data from {raw_path}...")
            with open(raw_path, "rb") as handle:
                self.num_classes, self.num_features, self.raw_data = pkl.load(
                    handle
                )
        else:
            print(f"Reading PyG TUDataset {name} under {root_path}...")
            self.raw_data = TUDataset(
                root=self.root_path,
                name=self.name,
                use_node_attr=True,
            )
            self.num_features = self.raw_data.num_node_features
            self.num_classes = self.raw_data.num_classes
            self.is_pyg = True

        self.data = []
        self.process()

    def process(self):
        if self.raw_data is None:
            raise RuntimeError("Failed to load raw graph-classification data")

        staged = []
        all_patterns = set()
        for graph_data in tqdm(
            self.raw_data, desc="Creating homomorphism mappings..."
        ):
            if self.is_pyg:
                x = graph_data.x
                y = graph_data.y
                graph = to_networkx(graph_data, to_undirected=True)
            else:
                graph = graph_data
                x = graph.graph["x"]
                y = graph.graph["y"]
            mappings = cycle_mapping_index(
                graph, length_bound=self.cycle_length_bound
            )
            mappings.update(
                clique_mapping_index(
                    graph, size_bound=self.clique_size_bound
                )
            )
            all_patterns.update(mappings)
            staged.append((x, y, mappings))

        for x, y, mappings in tqdm(staged, desc="Finalizing..."):
            for pattern in all_patterns:
                mappings.setdefault(pattern, None)
            self.data.append(
                Data(x=x, y=y, mapping_index_dict=mappings)
            )

        os.makedirs(self.root_path, exist_ok=True)
        cache_path = os.path.join(self.root_path, self.name + ".pt")
        torch.save(
            (self.num_classes, self.num_features, self.data), cache_path
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return self.transform(sample) if self.transform else sample


def hom_collate(data_list):
    """Combine multiple HomData-style PyG `Data` objects into a single `Batch`,
    offsetting `mapping_index` tensors by node-count so they refer to the right
    rows in the concatenated `x`.
    """
    batch = Batch()
    num_nodes_list = [data.num_nodes for data in data_list]
    node_offsets = torch.cumsum(torch.tensor([0] + num_nodes_list[:-1]), dim=0)
    mapping_keys = list(data_list[0]['mapping_index_dict'].keys())
    batch['mapping_index_dict'] = dict()
    batch.batch = torch.cat([
        torch.full((num_nodes,), i, dtype=torch.long)
        for i, num_nodes in enumerate(num_nodes_list)
    ])
    for mapping_key in mapping_keys:
        stacked_tensors = []
        for i, data in enumerate(data_list):
            tensor = data['mapping_index_dict'][mapping_key]
            if tensor is not None:
                stacked_tensors.append(tensor + node_offsets[i])
        if stacked_tensors:
            batch['mapping_index_dict'][mapping_key] = torch.cat(stacked_tensors, dim=0)
    batch['x'] = torch.cat([data['x'] for data in data_list], dim=0)
    batch['y'] = torch.cat([data['y'] for data in data_list], dim=0)
    batch.batch_size = len(data_list)
    return batch


class HomDataLoader(DataLoader):
    """DataLoader using DHN's mapping-aware graph collation."""

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn=hom_collate,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )


class NodeClassDataset:
    """Loads a precomputed node-classification graph.

    Expects a `.pt` file produced by `preprocess.imdb.node_classification`
    (or any equivalent preprocessor) containing:
        {
            'data':         PyG Data with x, y, edge_index, train/val/test_mask,
                            mapping_index_dict, batch, batch_size,
            'num_features': int,
            'num_classes':  int,
        }
    """
    def __init__(self, path: str):
        payload = torch.load(path)
        self.data: Data = payload['data']
        self.num_features: int = payload['num_features']
        self.num_classes: int = payload['num_classes']
