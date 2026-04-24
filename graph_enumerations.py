import torch
import numpy as np
import networkx as nx
from itertools import permutations
from collections import defaultdict


def cycle_mapping_index(nxg, length_bound=10):
    base_cycles = [*nx.simple_cycles(nxg, length_bound=length_bound)]
    index_dict = defaultdict(list)
    for c in base_cycles:
        index_dict[f'c{len(c)}'].append(c)
        index_dict[f'c{len(c)}'].append([*reversed(c)])
    index_dict['c2'] = list(nxg.edges())
    result = dict()
    for k, v in index_dict.items():
        result[k] = torch.tensor(np.vstack([np.roll(v, i, axis=1) for i in range(1, int(k[1:])+1)])).long()
    for i in range(2, length_bound+1):
        if f'c{i}' not in result:
            result[f'c{i}'] = None
    return result


def clique_mapping_index(nxg, size_bound=5):
    base_cliques = [k for k in nx.clique.enumerate_all_cliques(nxg) if len(k) > 2 and len(k) <= size_bound]
    index_dict = defaultdict(list)
    for c in base_cliques:
        index_dict[f'k{len(c)}'].extend(permutations(c))
    result = dict()
    for k, v in index_dict.items():
        result[k] = torch.tensor(v).long()
    for i in range(3, size_bound+1):
        if f'k{i}' not in result:
            result[f'k{i}'] = None
    return result