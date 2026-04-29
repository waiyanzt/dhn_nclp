"""Synthetic-graph sanity tests for DHN node classification.

Verifies enumeration + forward pass on three minimal cases:
  - 1 node, no edges       (P1 only; P2 and P3 absent)
  - 2 nodes, 1 edge        (P1 + P2; P3 = closed 2-walks)
  - 3-node path 0-1-2      (all three patterns)

Run from the project root:
    python tests/test_simple_graphs.py
"""
import os
import sys

import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhn.graph_enumerations import (
    cycle_mapping_index,
    path_mapping_index,
    single_node_mapping_index,
)
from dhn.datasets import hom_collate
from dhn.models import DHN


FEAT_DIM = 4
HIDDEN = 8
NUM_CLASSES = 2


def build_data(g, x, y):
    mappings = {}
    mappings.update(single_node_mapping_index(g))
    mappings.update(cycle_mapping_index(g, length_bound=2))
    mappings.update(path_mapping_index(g))
    return Data(x=x, y=y, mapping_index_dict=mappings)


def make_model():
    return DHN(
        out_dim=NUM_CLASSES,
        layers_config=[
            {
                'p1': (FEAT_DIM, HIDDEN, 1),
                'c2': (FEAT_DIM, HIDDEN, 2),
                'p3': (FEAT_DIM, HIDDEN, 3),
            },
        ],
        act_module=nn.ReLU,
        agg=None,
        inplace=False,
    )


def test_single_node():
    g = nx.Graph()
    g.add_node(0)
    x = torch.randn(1, FEAT_DIM)
    y = torch.tensor([0])
    data = build_data(g, x, y)

    p1 = data.mapping_index_dict['p1']
    c2 = data.mapping_index_dict['c2']
    p3 = data.mapping_index_dict['p3']
    assert p1 is not None and tuple(p1.shape) == (1, 1), f"p1 wrong: {p1}"
    assert c2 is None, f"c2 should be None on edgeless graph, got {c2}"
    assert p3 is None, f"p3 should be None on edgeless graph, got {p3}"

    batch = hom_collate([data])
    out = make_model()(batch)
    assert tuple(out.shape) == (1, NUM_CLASSES), f"out shape: {out.shape}"
    print(f"[ok] single node       -> out {tuple(out.shape)}")


def test_two_nodes_one_edge():
    g = nx.Graph()
    g.add_edge(0, 1)
    x = torch.randn(2, FEAT_DIM)
    y = torch.tensor([0, 1])
    data = build_data(g, x, y)

    p1 = data.mapping_index_dict['p1']
    c2 = data.mapping_index_dict['c2']
    p3 = data.mapping_index_dict['p3']
    assert p1 is not None and tuple(p1.shape) == (2, 1)
    assert c2 is not None and tuple(c2.shape) == (2, 2), f"c2 wrong: {c2}"
    # 2-walks: 0-1-0 and 1-0-1 (closed walks count as homomorphisms)
    assert p3 is not None and tuple(p3.shape) == (2, 3), f"p3 wrong: {p3}"

    batch = hom_collate([data])
    out = make_model()(batch)
    assert tuple(out.shape) == (2, NUM_CLASSES), f"out shape: {out.shape}"
    print(f"[ok] 2 nodes + 1 edge  -> out {tuple(out.shape)}")


def test_three_node_path():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    x = torch.randn(3, FEAT_DIM)
    y = torch.tensor([0, 1, 0])
    data = build_data(g, x, y)

    p1 = data.mapping_index_dict['p1']
    c2 = data.mapping_index_dict['c2']
    p3 = data.mapping_index_dict['p3']
    assert p1 is not None and tuple(p1.shape) == (3, 1)
    # 2 undirected edges x both orientations = 4 rows
    assert c2 is not None and tuple(c2.shape) == (4, 2), f"c2 wrong: {c2}"
    # 2-walks rooted at each node:
    #   from 0: (0,1,0), (0,1,2)
    #   from 1: (1,0,1), (1,2,1)
    #   from 2: (2,1,0), (2,1,2)
    # = 6 total
    assert p3 is not None and tuple(p3.shape) == (6, 3), f"p3 wrong: {p3}"

    batch = hom_collate([data])
    out = make_model()(batch)
    assert tuple(out.shape) == (3, NUM_CLASSES), f"out shape: {out.shape}"
    print(f"[ok] 3-node path       -> out {tuple(out.shape)}")


if __name__ == '__main__':
    test_single_node()
    test_two_nodes_one_edge()
    test_three_node_path()
    print("\nall tests passed")
