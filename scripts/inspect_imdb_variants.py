"""One-off diagnostic: characterize each IMDB_preprocessed_star* variant.

Answers three questions:
  Q1. What node types exist in each variant? (Is there a genre node?)
  Q2. Which variant (V1/V2/V3/V4) does each directory implement?
      V1: Link <-> Movie only
      V2: Link <-> Movie + Link <-> Director
      V3: Link <-> Movie + Link <-> Actor
      V4: Link <-> Movie + Link <-> Actor + Link <-> Director
  Q3. Is the adjacency stored symmetrically?

Run from repo root:
    uv run python scripts/inspect_imdb_variants.py
"""
import os
import numpy as np
import scipy.sparse

DIRS = [
    'data/preprocessed/IMDB_preprocessed_star',
    'data/preprocessed/IMDB_preprocessed_star_t',
    'data/preprocessed/IMDB_preprocessed_star_t_2',
    'data/preprocessed/IMDB_preprocessed_star_t_3',
]

# From the NC preprocessor we know type 0 = movie. The other type ids
# we infer from node counts: ~4180 = movie or link (1:1), ~2081 = director,
# remainder = actor.
COUNT_HINTS = {4180: 'movie_or_link', 2081: 'director'}


def guess_type_name(count, already_taken):
    if count == 4180:
        return 'link' if 'movie' in already_taken else 'movie'
    if abs(count - 2081) < 5:
        return 'director'
    return 'actor'


def inspect(d):
    print(f"\n=== {d} ===")
    adj = scipy.sparse.load_npz(os.path.join(d, 'adjM.npz')).tocsr()
    types = np.load(os.path.join(d, 'node_types.npy'))

    # Q1: types present
    unique, counts = np.unique(types, return_counts=True)
    print(f"  Node-type counts: {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"  Total nodes: {types.shape[0]},  num types: {len(unique)}")

    # Heuristic name mapping for readability
    name_of = {0: 'movie'}  # confirmed by NC preprocessor
    taken = {'movie'}
    for t in unique:
        if t == 0:
            continue
        c = int(counts[unique.tolist().index(t)])
        n = guess_type_name(c, taken)
        name_of[int(t)] = n
        taken.add(n)
    print(f"  Inferred type names: {name_of}")

    # Q3: adjacency symmetry
    sym = (adj != adj.T).nnz == 0
    print(f"  Adjacency symmetric: {sym}  (nnz={adj.nnz})")

    # Q2: edge counts between every (type_a, type_b) pair, undirected nnz
    print(f"  Cross-type edge counts (one direction; double if symmetric):")
    ulist = unique.tolist()
    for i, t1 in enumerate(ulist):
        for t2 in ulist[i+1:]:
            idx1 = np.where(types == t1)[0]
            idx2 = np.where(types == t2)[0]
            n = adj[idx1][:, idx2].nnz
            n1 = name_of.get(int(t1), f'type{t1}')
            n2 = name_of.get(int(t2), f'type{t2}')
            marker = '   <-- nonzero' if n > 0 else ''
            print(f"    {n1:<10s} <-> {n2:<10s}: {n}{marker}")


for d in DIRS:
    inspect(d)

print("""
Use the link-connection pattern to map dir -> variation:
  V1: link <-> movie only
  V2: link <-> movie + link <-> director
  V3: link <-> movie + link <-> actor
  V4: link <-> movie + link <-> actor + link <-> director
""")
