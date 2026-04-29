import torch
from torch_geometric.data import Batch, Data


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


class NodeClassDataset:
    """Loads a precomputed node-classification graph.

    Expects a `.pt` file produced by `preprocess/preprocess_IMDB_for_dhn_nc.py`
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
