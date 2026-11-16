"""docs/results.md's BM25 k1/b table must be what results/bm25_params/ says
(docs/measure_bm25.py writes it), and the defaults must stay the defaults."""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from turkish_rag_eval.metrics import paired_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
ROWS = json.loads((ROOT / "results" / "bm25_params" / "summary.json").read_text(encoding="utf-8"))


def cells(chunking: str) -> list[dict]:
    return [r for r in ROWS if r["chunking"] == chunking]


def default(chunking: str) -> dict:
    return next(r for r in cells(chunking) if r["k1"] == 1.5 and r["b"] == 0.75)


def test_the_default_cell_is_the_published_stem5_score():
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    for chunking in ("fixed", "sentence", "hierarchical"):
        published = next(r["ndcg@10"] for r in summary
                         if r["chunking"] == chunking and r["retriever"] == "bm25_stem5")
        assert default(chunking)["ndcg@10"] == pytest.approx(published, abs=1e-9)


def test_every_table_row_matches_the_results():
    for chunking in ("fixed", "sentence", "hierarchical"):
        best = max(cells(chunking), key=lambda r: r["ndcg@10"])
        d = default(chunking)
        low, high = paired_bootstrap_ci(best["ndcg@10_per_query"], d["ndcg@10_per_query"])
        change = best["ndcg@10"] - d["ndcg@10"]
        row = (f"| {chunking} | {d['ndcg@10']:.3f} | {best['ndcg@10']:.3f} "
               f"({best['k1']}, {best['b']}) | {change:+.3f} [{low:+.3f}, {high:+.3f}] |")
        assert row in DOC, chunking
        assert low < 0, f"{chunking}: a tuned cell now separates from the default; rewrite the prose"
