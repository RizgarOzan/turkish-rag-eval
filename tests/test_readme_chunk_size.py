"""The README's chunk-size tables and the claims under them must be what
results/chunk_size/ says (docs/measure_chunk_size.py writes it)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from turkish_rag_eval.metrics import paired_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
ROWS = json.loads((ROOT / "results" / "chunk_size" / "summary.json").read_text(encoding="utf-8"))
DENSE = json.loads((ROOT / "results" / "chunk_size" / "multilingual-e5-small.json")
                   .read_text(encoding="utf-8"))
SIZES = (300, 500, 700, 1000, 1500)
CHUNKINGS = ("fixed", "sentence", "hierarchical")


def cell(chunking: str, size: int, rows: list = ROWS) -> dict:
    return next(r for r in rows if r["chunking"] == chunking and r["size"] == size)


def change(chunking: str, size: int, rows: list = ROWS) -> tuple[float, float, float]:
    a = cell(chunking, size, rows)["ndcg@10_per_query"]
    b = cell(chunking, 700, rows)["ndcg@10_per_query"]
    qids = sorted(set(a) & set(b))
    a, b = [a[q] for q in qids], [b[q] for q in qids]
    low, high = paired_bootstrap_ci(a, b)
    return sum(a) / len(a) - sum(b) / len(b), low, high


def test_every_table_row_matches_the_results():
    for chunking in CHUNKINGS:
        scores = " | ".join(f"{cell(chunking, n)['ndcg@10']:.3f}" for n in SIZES)
        assert f"| {chunking} | {scores} |" in README, chunking


def test_all_58_questions_are_scored_at_every_size():
    for row in ROWS:
        assert row["queries_scored"] == 58, (row["chunking"], row["size"])


def test_the_chunk_counts_in_the_text_are_the_measured_ones():
    assert f"{cell('fixed', 300)['chunks']:,} fixed chunks at 300" in README
    assert f"{cell('fixed', 1500)['chunks']:,} for fixed" in README
    assert f"{cell('hierarchical', 1500)['chunks']:,} chunks\nat 1500" in README


def test_the_quoted_intervals_are_the_computed_ones():
    for chunking in ("fixed", "hierarchical"):
        d, low, high = change(chunking, 1500)
        assert f"1500: {d:+.3f} [{low:+.3f}, {high:+.3f}]" in README, chunking


def test_300_is_clearly_worse_fixed_climbs_and_hierarchical_is_flat():
    for chunking in CHUNKINGS:
        assert change(chunking, 300)[2] < 0, chunking
    assert change("fixed", 1500)[1] > 0
    for size in (1000, 1500):
        low, high = change("hierarchical", size)[1:]
        assert low < 0 < high, size


def test_the_dense_table_matches_its_results():
    for chunking in CHUNKINGS:
        scores = " | ".join(f"{cell(chunking, n, DENSE)['ndcg@10']:.3f}" for n in SIZES)
        assert f"| {chunking} | {scores} |" in README, chunking
    assert all(row["queries_scored"] == 58 for row in DENSE)


def test_no_dense_change_against_700_clears_zero():
    changes = {(c, n): change(c, n, DENSE) for c in CHUNKINGS for n in SIZES if n != 700}
    for key, (_, low, high) in changes.items():
        assert low < 0 < high, key
    largest = max(changes, key=lambda k: changes[k][0])
    assert largest == ("fixed", 1500)
    d, low, high = changes[largest]
    assert f"fixed at\n1500, {d:+.3f} [{low:+.3f}, {high:+.3f}]" in README
