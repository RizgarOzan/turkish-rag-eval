"""The README's stem-length table and the claims under it must be what
results/stem/ says (docs/measure_stem.py writes it)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from turkish_rag_eval.metrics import paired_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
ROWS = json.loads((ROOT / "results" / "stem" / "summary.json").read_text(encoding="utf-8"))
LENGTHS = (None, 3, 4, 5, 6, 7, 8)


def cell(chunking: str, length) -> dict:
    return next(r for r in ROWS if r["chunking"] == chunking and r["stem_length"] == length)


def change(chunking: str, length) -> tuple[float, float, float]:
    a, b = cell(chunking, length)["ndcg@10_per_query"], cell(chunking, 5)["ndcg@10_per_query"]
    low, high = paired_bootstrap_ci(a, b)
    return sum(a) / len(a) - sum(b) / len(b), low, high


def test_every_table_row_matches_the_results():
    for chunking in ("fixed", "sentence", "hierarchical"):
        scores = " | ".join(f"{cell(chunking, n)['ndcg@10']:.3f}" for n in LENGTHS)
        assert f"| {chunking} | {scores} |" in README, chunking


def test_five_is_the_best_score_on_every_chunking():
    for chunking in ("fixed", "sentence", "hierarchical"):
        best = max(LENGTHS, key=lambda n: cell(chunking, n)["ndcg@10"])
        assert best == 5, chunking


def test_the_quoted_intervals_are_the_computed_ones():
    for chunking, length in (("hierarchical", 4), ("sentence", 6)):
        d, low, high = change(chunking, length)
        assert f"{d:+.3f} [{low:+.3f}, {high:+.3f}]" in README, (chunking, length)


def test_no_stemming_and_six_are_clearly_worse_and_three_or_four_are_not():
    for chunking in ("fixed", "sentence", "hierarchical"):
        assert change(chunking, None)[2] < 0, chunking
        assert change(chunking, 6)[2] < 0, chunking
    for chunking in ("sentence", "hierarchical"):
        for length in (3, 4):
            assert change(chunking, length)[2] > 0, (chunking, length)
