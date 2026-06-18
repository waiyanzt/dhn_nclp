"""Convert MAGNN-style IMDB heterogeneous preprocessed data into a single
homogeneous PyG Data file ready for DHN node classification.

Run AFTER one of the `preprocess_IMDB_star*.py` scripts has produced the
heterogeneous IMDB output. This script flattens the hetero graph (treats all
node types as one), enumerates {p1, c2, p3} homomorphism mappings once over
the full graph, and saves the bundle to a single .pt file.

INPUT  (set IN_DIR below):
    adjM.npz                  - sparse (N, N) adjacency over all nodes
    node_types.npy            - (N,) int node-type per node (movies = 0)
    features_{i}.npz          - sparse features for type-i nodes
    labels.npy                - (M,) int genre labels for movies
    train_val_test_idx.npz    - dict with train_idx / val_idx / test_idx
                                (indices into the labels array)

OUTPUT (set OUT_PATH below):
    A torch.save'd dict with keys: data, num_features, num_classes
"""
import os
import sys

import numpy as np
import scipy.sparse
import torch
import networkx as nx
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhn.graph_enumerations import (
    cycle_mapping_index,
    path_mapping_index,
    single_node_mapping_index,
)


IN_DIR = 'data/preprocessed/IMDB_preprocessed_star/'
OUT_PATH = 'data/preprocessed/IMDB_dhn_nc.pt'
PATTERNS = ['p1', 'c2', 'p3']

PATTERN_FNS = {
    'p1': lambda g: single_node_mapping_index(g),
    'c2': lambda g: cycle_mapping_index(g, length_bound=2),
    'p3': lambda g: path_mapping_index(g),
}


def main():
    print(f"Loading from {IN_DIR}")

    # 1. adjacency
    adj = scipy.sparse.load_npz(os.path.join(IN_DIR, 'adjM.npz')).tocoo()
    edge_index = torch.from_numpy(np.vstack([adj.row, adj.col])).long()

    # 2. node types
    type_mask = np.load(os.path.join(IN_DIR, 'node_types.npy'))
    num_nodes = int(type_mask.shape[0])
    num_types = int(type_mask.max()) + 1

    # 3. features: scatter each type's features back into the right rows
    feats_per_type = []
    feat_dim = None
    for i in range(num_types):
        f_i = scipy.sparse.load_npz(os.path.join(IN_DIR, f'features_{i}.npz')).toarray()
        if feat_dim is None:
            feat_dim = f_i.shape[1]
        elif f_i.shape[1] != feat_dim:
            raise ValueError(
                f"features_{i}.npz has dim {f_i.shape[1]} but features_0 has {feat_dim}; "
                f"unified feature dim required for flat-DHN"
            )
        feats_per_type.append(f_i)

    full_feats = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    for i in range(num_types):
        type_i_idx = np.where(type_mask == i)[0]
        full_feats[type_i_idx] = feats_per_type[i]
    x = torch.from_numpy(full_feats)

    # 4. labels: only for movie (type-0) nodes
    movie_indices = np.where(type_mask == 0)[0]
    target_labels = np.load(os.path.join(IN_DIR, 'labels.npy'))
    assert target_labels.shape[0] == movie_indices.shape[0], (
        f"label/movie count mismatch: {target_labels.shape[0]} labels vs "
        f"{movie_indices.shape[0]} type-0 nodes"
    )
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[movie_indices] = torch.from_numpy(target_labels).long()
    num_classes = int(target_labels.max()) + 1

    # 5. masks: provided indices are local to the labels array, so map through movie_indices
    idx_data = np.load(os.path.join(IN_DIR, 'train_val_test_idx.npz'))
    train_local = idx_data['train_idx']
    val_local = idx_data['val_idx']
    test_local = idx_data['test_idx']
    train_global = movie_indices[train_local]
    val_global = movie_indices[val_local]
    test_global = movie_indices[test_local]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_global] = True
    val_mask[val_global] = True
    test_mask[test_global] = True

    print(f"Graph:   nodes={num_nodes} types={num_types} features={feat_dim}")
    print(f"Targets: {len(movie_indices)} movies, {num_classes} classes")
    print(f"Splits:  train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}")

    # 6. build undirected, deduped NetworkX graph for enumeration
    print("Building NetworkX graph (undirected, deduped)...")
    nxg = nx.Graph()
    nxg.add_nodes_from(range(num_nodes))
    seen = set()
    for i, j in zip(adj.row.tolist(), adj.col.tolist()):
        if i == j:
            continue
        e = (i, j) if i < j else (j, i)
        seen.add(e)
    nxg.add_edges_from(seen)
    print(f"  edges (undirected): {nxg.number_of_edges()}")

    # 7. enumerate patterns
    print(f"Enumerating patterns {PATTERNS}...")
    mapping_index_dict = {}
    for p in PATTERNS:
        mapping_index_dict.update(PATTERN_FNS[p](nxg))
    for k, v in mapping_index_dict.items():
        shape = None if v is None else tuple(v.shape)
        print(f"  {k}: {shape}")

    # 8. assemble Data; attach batch attrs because DHN.forward reads them
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        mapping_index_dict=mapping_index_dict,
    )
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    data.batch_size = 1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save({
        'data': data,
        'num_features': feat_dim,
        'num_classes': num_classes,
    }, OUT_PATH)
    print(f"Saved bundle to {OUT_PATH}")


if __name__ == '__main__':
    main()
