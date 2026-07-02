"""Checks for the standardized WordNet DHN pattern set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.wordnet.link_prediction import PATTERNS


def test_wordnet_uses_scalable_pattern_set():
    assert PATTERNS == ["p1", "c2"]


if __name__ == "__main__":
    test_wordnet_uses_scalable_pattern_set()
    print("WordNet pattern configuration test passed")
