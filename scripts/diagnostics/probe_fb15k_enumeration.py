"""Predict DHN p3 enumeration cost on FB15k-237 (no_changes) WITHOUT building it.

p3 row count = sum_v deg(v)^2 over the undirected, simple entity graph (relations
collapsed; relation type matters to the decoder, not to structural enumeration).
FB15k-237 has heavy hub entities, so this checks tractability before we commit.
"""
import numpy as np
import pandas as pd

PATH = "data/raw/fb15k237_augmented_inverse/fb15k237_augmented_inverse/no_changes/data.txt"

df = pd.read_csv(PATH, sep="\t", names=["h", "r", "t"], dtype=str)
ents = pd.unique(pd.concat([df["h"], df["t"]], ignore_index=True))
eid = {e: i for i, e in enumerate(ents)}
h = df["h"].map(eid).to_numpy()
t = df["t"].map(eid).to_numpy()

# undirected simple edges (collapse multi-relations + direction, drop self-loops)
lo = np.minimum(h, t)
hi = np.maximum(h, t)
mask = lo != hi
pairs = np.unique(np.stack([lo[mask], hi[mask]], axis=1), axis=0)

deg = np.zeros(len(ents), dtype=np.int64)
np.add.at(deg, pairs[:, 0], 1)
np.add.at(deg, pairs[:, 1], 1)

p3 = int((deg.astype(np.int64) ** 2).sum())
print(f"entities:            {len(ents):,}")
print(f"relations:           {df['r'].nunique()}")
print(f"triples:             {len(df):,}")
print(f"undirected edges:    {len(pairs):,}   -> c2 rows ~= {2*len(pairs):,}")
print(f"max entity degree:   {deg.max():,}")
print(f"top-5 hub degrees:   {sorted(deg.tolist(), reverse=True)[:5]}")
print(f"p3 rows = sum deg^2: {p3:,}")
print(f"p3 tensor est:       {p3 * 3 * 8 / 1e9:.2f} GB (int64, 3 cols)")
