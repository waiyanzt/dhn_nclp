"""Checks for the DBLP* union-graph construction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.dblp.link_prediction import PATTERNS, build_typed_edges


def fixture():
    pa = pd.DataFrame({"paper_id": [10, 11], "author_id": [20, 21]})
    pt = pd.DataFrame({"paper_id": [10, 11], "term_id": [30, 31]})
    pc = pd.DataFrame({"paper_id": [10, 11], "conf_id": [40, 40]})
    pr = pd.DataFrame({"paper_id": [10, 11], "area_id": [50, 51]})
    ar = pd.DataFrame({"author_id": [20, 21], "area_id": [50, 51]})
    maps = {
        "paper": {10: 0, 11: 1},
        "author": {20: 0, 21: 1},
        "term": {30: 0, 31: 1},
        "conf": {40: 0},
        "area": {50: 0, 51: 1},
    }
    train_pos = np.asarray([[0, 0], [1, 0]], dtype=np.int64)
    return pa, pt, pc, pr, ar, train_pos, maps


def test_baselines_have_one_area_relation_family():
    args = fixture()
    expected = {
        "v1": "paper-area",
        "v2": "conf-area",
        "v3": "author-area",
    }
    for variant, area_relation in expected.items():
        edges = build_typed_edges(variant, *args)
        area_keys = {key for key in edges if key.endswith("-area")}
        assert area_keys == {area_relation}


def test_universal_has_all_area_relation_families():
    edges = build_typed_edges("universal", *fixture())
    assert set(edges) == {
        "paper-author",
        "paper-term",
        "paper-conf",
        "paper-area",
        "conf-area",
        "author-area",
    }


def test_venue_area_uses_training_target_edges_only():
    pa, pt, pc, pr, ar, _train_pos, maps = fixture()
    train_pos = np.asarray([[0, 0]], dtype=np.int64)
    edges = build_typed_edges(
        "v2", pa, pt, pc, pr, ar, train_pos, maps
    )
    conf_area = edges["conf-area"]
    assert list(zip(*conf_area)) == [(0, 0)]


def test_dblp_uses_scalable_pattern_set():
    assert PATTERNS == ["p1", "c2"]


if __name__ == "__main__":
    test_baselines_have_one_area_relation_family()
    test_universal_has_all_area_relation_families()
    test_venue_area_uses_training_target_edges_only()
    test_dblp_uses_scalable_pattern_set()
    print("DBLP invariant preprocessing tests passed")
