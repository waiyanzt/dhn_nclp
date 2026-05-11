"""Build a homogeneous-flat DHN bundle for IMDb link prediction.

Mirrors Bishwash's RGCN-LP preprocessor (KC_scripts/MAGNN/preprocess_IMDB_rgcn_lp.py)
to guarantee graph-content parity with the RGCN baseline, then flattens the
heterograph to a single homogeneous PyG Data with DHN pattern enumerations.

Target-edge handling (leakage prevention by construction):
  - md / mg / ml: only TRAIN positive target edges are added to the graph;
    val/test target edges are never added.

Auxiliary genre nodes (added to the graph for ALL tasks):
  - Three genre nodes (Action/Comedy/Drama) are appended to the global node
    space and connected to every movie via movie-genre edges derived from
    movie_metadata.csv. For mg, genre is the LP target, so only TRAIN
    movie-genre edges are added (val/test are the held-out positives). For
    md and ml, genre is auxiliary structural context — every movie carries
    its genre edge regardless of split (no leakage since genre is not the
    target). Bishwash's RGCN runner has the movie-genre relation wired up
    for every task; including these edges matches that intent.

`x` is None: the trainer creates nn.Embedding(num_nodes, in_dim) for headline
RGCN parity.

Usage:
    python preprocess/preprocess_IMDB_dhn_lp.py --task md --variant v1
    python preprocess/preprocess_IMDB_dhn_lp.py --task ml --variant v1,v2,v3,v4
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhn.graph_enumerations import (  # noqa: E402
    cycle_mapping_index,
    path_mapping_index,
    single_node_mapping_index,
)

SEED = 1566911444
NUM_GENRES = 3
PATTERNS = ['p1', 'c2', 'p3']

PATTERN_FNS = {
    'p1': lambda g: single_node_mapping_index(g),
    'c2': lambda g: cycle_mapping_index(g, length_bound=2),
    'p3': lambda g: path_mapping_index(g),
}


# ---- Bishwash's CSV readers (verbatim, for parity) ------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_imdb_frame_md_mg(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (movies, labels). Labels are genre ids 0/1/2 (Action/Comedy/Drama),
    matching Bishwash's CMPNN convention. Used as the LP target for mg and as
    the source of auxiliary movie-genre edges for md.
    """
    movies = (
        pd.read_csv(csv_path, encoding="utf-8")
        .drop_duplicates(subset=["movie_imdb_link"])
        .dropna(axis=0, subset=["actor_1_name", "director_name"])
        .reset_index(drop=True)
    )
    labels = np.full(len(movies), -1, dtype=int)
    for idx, genres in movies["genres"].items():
        for g in str(genres).split("|"):
            if g == "Action":
                labels[idx] = 0
                break
            elif g == "Comedy":
                labels[idx] = 1
                break
            elif g == "Drama":
                labels[idx] = 2
                break
    keep = labels != -1
    return movies[keep].reset_index(drop=True), labels[keep]


def read_imdb_frame_ml(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (movies, labels). Labels are genre ids 0/1/2. Used as the source
    of auxiliary movie-genre edges for ml (genre is not the ml target, so
    every movie carries its genre edge without leakage).
    """
    movies = pd.read_csv(csv_path, encoding="utf-8")
    movies = movies.drop_duplicates(subset="movie_imdb_link").dropna(
        subset=["movie_imdb_link", "actor_1_name", "director_name", "genres"]
    ).reset_index(drop=True)
    labels = np.full(len(movies), -1, dtype=np.int64)
    for i, genres in movies["genres"].astype(str).items():
        for g in genres.split("|"):
            g = g.strip()
            if g == "Action":
                labels[i] = 0
                break
            elif g == "Comedy":
                labels[i] = 1
                break
            elif g == "Drama":
                labels[i] = 2
                break
    keep = np.where(labels >= 0)[0]
    return movies.iloc[keep].reset_index(drop=True), labels[keep]


def _directors_actors_mg(movies: pd.DataFrame) -> tuple[list, list]:
    directors = sorted(set(movies["director_name"].dropna()))
    actors = sorted(set(
        movies["actor_1_name"].dropna().tolist()
        + movies["actor_2_name"].dropna().tolist()
        + movies["actor_3_name"].dropna().tolist()
    ))
    return directors, actors


def _directors_actors_ml(movies: pd.DataFrame) -> tuple[list, list]:
    directors = sorted(set(movies["director_name"].dropna().tolist()))
    actors = sorted(set(movies["actor_1_name"].dropna().tolist()))
    return directors, actors


# ---- Bishwash's negative samplers (verbatim) ------------------------------

def sample_negs_md(pos, n_directors, k, rng):
    N = len(pos)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (_m, d_true) in enumerate(pos):
        cand = [d for d in range(n_directors) if d != int(d_true)]
        if not cand:
            cand = list(range(n_directors))
        neg[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return neg


def sample_negs_mg(pos, k, rng):
    all_g = np.arange(NUM_GENRES, dtype=np.int64)
    N = len(pos)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (_m, g_true) in enumerate(pos):
        gt = int(g_true)
        cand = all_g[all_g != gt]
        neg[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return neg


def sample_negs_ml(pos, n_movies, true_set, k, rng):
    L_all = np.arange(n_movies, dtype=np.int64)
    N = len(pos)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (mL, l_true) in enumerate(pos):
        mL, l_true = int(mL), int(l_true)
        cand = [x for x in L_all if (mL, int(x)) not in true_set]
        if not cand:
            cand = list(L_all)
        neg[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return neg


# ---- Hetero-edge builders (mirror Bishwash's v1-v4 rules) -----------------

def _add(d: dict, key: str, u: int, v: int) -> None:
    if key not in d:
        d[key] = ([], [])
    d[key][0].append(int(u))
    d[key][1].append(int(v))


def build_typed_edges_md(movies, variant, train_pos, dmap, amap, labels):
    raw: dict = {}
    train_md = {(int(r[0]), int(r[1])) for r in train_pos}

    if variant == "v1":
        for mi, row in movies.iterrows():
            mi = int(mi)
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add(raw, "movie-actor", mi, amap[row[acol]])
            _add(raw, "movie-link", mi, mi)
        for m, d in train_md:
            _add(raw, "movie-director", m, d)
    elif variant == "v3":
        for mi, row in movies.iterrows():
            mi = int(mi)
            _add(raw, "movie-link", mi, mi)
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add(raw, "link-actor", mi, amap[row[acol]])
        for m, d in train_md:
            _add(raw, "movie-director", m, d)
    else:
        raise ValueError(f"md variant must be v1 or v3, got {variant}")

    # Auxiliary movie-genre edges from labels.npy (genre is not the md target).
    for mi in range(len(movies)):
        _add(raw, "movie-genre", mi, int(labels[mi]))
    return raw


def build_typed_edges_mg(movies, variant, train_pos, dmap, amap):
    raw: dict = {}
    train_mg = {(int(r[0]), int(r[1])) for r in train_pos}

    if variant == "v1":
        for mi, row in movies.iterrows():
            mi = int(mi)
            _add(raw, "movie-director", mi, dmap[row["director_name"]])
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add(raw, "movie-actor", mi, amap[row[acol]])
            _add(raw, "movie-link", mi, mi)
        for m, g in train_mg:
            _add(raw, "movie-genre", m, g)
    elif variant == "v2":
        for mi, row in movies.iterrows():
            mi = int(mi); li = mi
            _add(raw, "link-director", li, dmap[row["director_name"]])
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add(raw, "link-actor", li, amap[row[acol]])
            _add(raw, "movie-link", mi, li)
        for m, g in train_mg:
            _add(raw, "movie-genre", m, g)
    elif variant == "v3":
        for mi, row in movies.iterrows():
            mi = int(mi); li = mi
            _add(raw, "movie-link", mi, li)
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add(raw, "link-actor", li, amap[row[acol]])
            _add(raw, "movie-director", mi, dmap[row["director_name"]])
        for m, g in train_mg:
            _add(raw, "movie-genre", m, g)
    elif variant == "v4":
        for mi, row in movies.iterrows():
            mi = int(mi); li = mi
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add(raw, "movie-actor", mi, amap[row[acol]])
            _add(raw, "movie-link", mi, li)
            _add(raw, "link-director", li, dmap[row["director_name"]])
        for m, g in train_mg:
            _add(raw, "movie-genre", m, g)
    else:
        raise ValueError(f"mg variant must be v1-v4, got {variant}")
    return raw


def build_typed_edges_ml(movies, variant, train_pos, dmap, amap, labels):
    raw: dict = {}
    train_ml = {(int(r[0]), int(r[1])) for r in train_pos}

    if variant == "v1":
        for i, row in movies.iterrows():
            i = int(i)
            _add(raw, "movie-director", i, dmap[row["director_name"]])
            _add(raw, "movie-actor", i, amap[row["actor_1_name"]])
            _add(raw, "movie-link", i, i)
        for m, l in train_ml:
            _add(raw, "movie-link", m, l)
    elif variant == "v2":
        for i, row in movies.iterrows():
            i = int(i); li = i
            _add(raw, "link-director", li, dmap[row["director_name"]])
            _add(raw, "link-actor", li, amap[row["actor_1_name"]])
            _add(raw, "movie-link", i, li)
        for m, l in train_ml:
            _add(raw, "movie-link", m, l)
    elif variant == "v3":
        for i, row in movies.iterrows():
            i = int(i); li = i
            _add(raw, "movie-link", i, li)
            _add(raw, "link-actor", li, amap[row["actor_1_name"]])
            _add(raw, "movie-director", i, dmap[row["director_name"]])
        for m, l in train_ml:
            _add(raw, "movie-link", m, l)
    elif variant == "v4":
        for i, row in movies.iterrows():
            i = int(i); li = i
            _add(raw, "movie-actor", i, amap[row["actor_1_name"]])
            _add(raw, "movie-link", i, li)
            _add(raw, "link-director", li, dmap[row["director_name"]])
        for m, l in train_ml:
            _add(raw, "movie-link", m, l)
    else:
        raise ValueError(f"ml variant must be v1-v4, got {variant}")

    # Auxiliary movie-genre edges from labels.npy (genre is not the ml target).
    for mi in range(len(movies)):
        _add(raw, "movie-genre", mi, int(labels[mi]))
    return raw


# ---- Flatten hetero -> homogeneous edge_index -----------------------------

def flatten_to_global(typed_edges: dict, offsets: dict) -> torch.Tensor:
    """Convert {'movie-actor': ([u_locals], [v_locals]), ...} to a global
    edge_index (2, 2E) with reverse edges added (PyG undirected convention).
    """
    src_chunks, dst_chunks = [], []
    for rel_key, (u_locals, v_locals) in typed_edges.items():
        if not u_locals:
            continue
        u_type, v_type = rel_key.split("-")
        if u_type not in offsets or v_type not in offsets:
            raise ValueError(f"Unknown node type(s) in relation {rel_key!r}; offsets has {list(offsets)}")
        u = np.asarray(u_locals, dtype=np.int64) + offsets[u_type]
        v = np.asarray(v_locals, dtype=np.int64) + offsets[v_type]
        src_chunks.append(u)
        dst_chunks.append(v)
    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    src_all = np.concatenate([src, dst])
    dst_all = np.concatenate([dst, src])
    return torch.from_numpy(np.vstack([src_all, dst_all]))


# ---- Main per-(task, variant) preprocessing -------------------------------

def preprocess_one(csv_path: Path, shared_npz: Path, task: str, variant: str,
                   out_path: str, neg_k: int, seed: int) -> None:
    set_seed(seed)
    print(f"\n=== DHN-LP preprocess | task={task} variant={variant} ===", flush=True)

    # 1. Splits + movies
    z = np.load(shared_npz)
    train_pos = z["train_pos"].astype(np.int64)
    val_pos = z["val_pos"].astype(np.int64)
    test_pos = z["test_pos"].astype(np.int64)

    if task == "ml":
        movies, labels = read_imdb_frame_ml(str(csv_path))
        directors, actors = _directors_actors_ml(movies)
    else:
        movies, labels = read_imdb_frame_md_mg(str(csv_path))
        directors, actors = _directors_actors_mg(movies)

    M = len(movies)
    Dn = len(directors)
    An = len(actors)
    Gn = NUM_GENRES
    dmap = {n: i for i, n in enumerate(directors)}
    amap = {n: i for i, n in enumerate(actors)}

    # Sanity: split local-ids must fit the movie space
    for split_name, arr in (("train_pos", train_pos), ("val_pos", val_pos), ("test_pos", test_pos)):
        m_max = int(arr[:, 0].max()) if len(arr) else -1
        assert m_max < M, f"{split_name}: movie_local={m_max} out of range (M={M})"

    # 2. Global id offsets: movies -> directors -> actors -> links -> genres
    # Genre nodes are present for all tasks; they are the LP target for mg and
    # auxiliary structural context for md/ml.
    offsets = {
        "movie": 0,
        "director": M,
        "actor": M + Dn,
        "link": M + Dn + An,
        "genre": M + Dn + An + M,
    }
    num_nodes_total = M + Dn + An + M + Gn

    print(f"  Counts: movies={M}, directors={Dn}, actors={An}, links={M}, genres={Gn}")
    print(f"  Total nodes: {num_nodes_total}")
    print(f"  Splits: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")

    # 3. Build typed edges. Target edges (the LP positives) come from train_pos
    # only — val/test target edges are never added. Auxiliary movie-genre edges
    # come from labels (every movie carries its genre) for md and ml.
    if task == "md":
        typed_edges = build_typed_edges_md(movies, variant, train_pos, dmap, amap, labels)
    elif task == "mg":
        typed_edges = build_typed_edges_mg(movies, variant, train_pos, dmap, amap)
    else:
        typed_edges = build_typed_edges_ml(movies, variant, train_pos, dmap, amap, labels)

    print("  Typed edge counts:")
    for k, (u, _) in typed_edges.items():
        print(f"    {k:<18s} {len(u)}")

    # 4. Flatten to homogeneous edge_index
    edge_index = flatten_to_global(typed_edges, offsets)
    print(f"  Flat edge_index: {tuple(edge_index.shape)} (includes reverse direction)")

    # 5. NetworkX undirected graph for pattern enumeration
    print("  Building NetworkX graph (undirected, deduped)...")
    nxg = nx.Graph()
    nxg.add_nodes_from(range(num_nodes_total))
    seen = set()
    for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if i == j:
            continue
        e = (i, j) if i < j else (j, i)
        seen.add(e)
    nxg.add_edges_from(seen)
    print(f"    edges (undirected, deduped): {nxg.number_of_edges()}")

    # 6. Enumerate patterns
    print(f"  Enumerating patterns {PATTERNS}...")
    mapping_index_dict = {}
    for p in PATTERNS:
        mapping_index_dict.update(PATTERN_FNS[p](nxg))
    for k, v in mapping_index_dict.items():
        shape = None if v is None else tuple(v.shape)
        print(f"    {k}: {shape}")

    # 7. Translate splits to global ids
    if task == "md":
        tgt_offset = offsets["director"]
    elif task == "mg":
        tgt_offset = offsets["genre"]
    else:
        tgt_offset = offsets["link"]

    def to_global_pos(pos):
        out = np.empty_like(pos)
        out[:, 0] = pos[:, 0] + offsets["movie"]
        out[:, 1] = pos[:, 1] + tgt_offset
        return torch.from_numpy(out).long()

    splits_global = {
        "train_pos": to_global_pos(train_pos),
        "val_pos":   to_global_pos(val_pos),
        "test_pos":  to_global_pos(test_pos),
    }

    # 8. Negatives (match Bishwash's seeding scheme)
    rng_train = np.random.RandomState(seed)
    if task == "md":
        train_neg_local = sample_negs_md(train_pos, Dn, neg_k, rng_train)
        rng_eval = np.random.RandomState(seed + 7)
        val_neg_local = sample_negs_md(val_pos, Dn, neg_k, rng_eval)
        test_neg_local = sample_negs_md(test_pos, Dn, neg_k, rng_eval)
        k_eff = neg_k
    elif task == "mg":
        train_neg_local = sample_negs_mg(train_pos, neg_k, rng_train)
        rng_eval = np.random.RandomState(seed + 7)
        val_neg_local = sample_negs_mg(val_pos, neg_k, rng_eval)
        test_neg_local = sample_negs_mg(test_pos, neg_k, rng_eval)
        k_eff = neg_k
    else:  # ml: shared rng across train/val/test, matches Bishwash
        all_true = (
            set(map(tuple, train_pos.tolist()))
            | set(map(tuple, val_pos.tolist()))
            | set(map(tuple, test_pos.tolist()))
        )
        k_eff = min(int(neg_k), max(1, M - 1))
        train_neg_local = sample_negs_ml(train_pos, M, all_true, k_eff, rng_train)
        val_neg_local = sample_negs_ml(val_pos, M, all_true, k_eff, rng_train)
        test_neg_local = sample_negs_ml(test_pos, M, all_true, k_eff, rng_train)

    def negs_to_global(neg):
        return torch.from_numpy((neg + tgt_offset).astype(np.int64))

    splits_global["train_neg"] = negs_to_global(train_neg_local)
    splits_global["val_neg"] = negs_to_global(val_neg_local)
    splits_global["test_neg"] = negs_to_global(test_neg_local)

    print(f"  Negatives: k_eff={k_eff}, train_neg shape={tuple(splits_global['train_neg'].shape)}")

    # 9. Assemble PyG Data + bundle
    data = Data(
        x=None,
        edge_index=edge_index,
        mapping_index_dict=mapping_index_dict,
    )
    data.num_nodes = num_nodes_total
    data.batch = torch.zeros(num_nodes_total, dtype=torch.long)
    data.batch_size = 1

    num_nodes_per_type = {"movie": M, "director": Dn, "actor": An, "link": M, "genre": Gn}

    bundle = {
        "data": data,
        "splits": splits_global,
        "node_offsets": offsets,
        "meta": {
            "task": task,
            "variant": variant,
            "num_nodes_per_type": num_nodes_per_type,
            "num_nodes_total": num_nodes_total,
            "neg_k": k_eff,
            "kendall_keys": {
                "md": ["movie_local", "director_local"],
                "mg": ["movie_local", "genre_id"],
                "ml": ["movie_local", "link_local"],
            }[task],
            "patterns": PATTERNS,
        },
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(bundle, out_path)
    print(f"  Saved bundle -> {out_path}")


# ---- CLI ------------------------------------------------------------------

def _parse_task(s: str) -> str:
    t = str(s).strip().lower()
    if t not in {"md", "mg", "ml"}:
        raise SystemExit("--task must be one of: md, mg, ml")
    return t


def _parse_variants(s: str, task: str) -> list[str]:
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    good = {"v1", "v3"} if task == "md" else {"v1", "v2", "v3", "v4"}
    bad = [v for v in vals if v not in good]
    if bad:
        raise SystemExit(f"Unknown variant(s) for task={task}: {bad}; expected subset of {sorted(good)}")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="IMDB DHN-LP preprocessing (homogeneous-flat, RGCN-parity).")
    ap.add_argument("--task", type=_parse_task, required=True)
    ap.add_argument("--variant", default="v1",
                    help="Comma-separated. md allows v1,v3; mg/ml allow v1-v4.")
    ap.add_argument("--csv", default="data/raw/IMDB/movie_metadata.csv")
    ap.add_argument("--shared-npz", default="",
                    help="Default: data/preprocessed/CMPNN/IMDB_<task>_shared_splits.npz")
    ap.add_argument("--out-dir", default="data/preprocessed")
    ap.add_argument("--neg-k", type=int, default=19,
                    help="Negatives per positive (md/ml=19; mg typically 2).")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    task = args.task
    variants = _parse_variants(args.variant, task)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Missing csv: {csv_path}")

    shared = Path(args.shared_npz) if args.shared_npz.strip() else Path(
        f"data/preprocessed/CMPNN/IMDB_{task}_shared_splits.npz"
    )
    if not shared.is_file():
        raise SystemExit(f"Missing shared npz: {shared}")

    if task == "mg" and (args.neg_k < 1 or args.neg_k > 2):
        print(f"[warn] mg has 3 genres; neg-k={args.neg_k} is unusual (recommended: 1-2)", flush=True)

    for v in variants:
        out_path = f"{args.out_dir}/IMDB_dhn_lp_{task}_{v}.pt"
        preprocess_one(csv_path, shared, task, v, out_path, args.neg_k, args.seed)


if __name__ == "__main__":
    main()
