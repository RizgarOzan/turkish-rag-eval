"""The README's reranking table must be what results/rerank/ says.

Every cell, every paired-bootstrap interval and both latencies are rebuilt
from the per-query files, so a re-run that moves a number turns this red
until the README follows.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from turkish_rag_eval.metrics import ndcg_at_k, paired_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "rerank" / "intfloat__multilingual-e5-small__BAAI__bge-reranker-v2-m3"
README = (ROOT / "README.md").read_text(encoding="utf-8")
CHUNKINGS = ("fixed", "sentence", "hierarchical")


def per_query(chunking: str, retriever: str) -> list[float]:
    rows = json.loads((RUN / f"perquery_{chunking}_{retriever}.json").read_text(encoding="utf-8"))
    return [ndcg_at_k(r["relevance"], r["total_relevant"], 10) for r in rows if r["total_relevant"] > 0]


def summary(chunking: str, retriever: str) -> dict:
    rows = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    return next(r for r in rows if r["chunking"] == chunking and r["retriever"] == retriever)


def row(chunking: str) -> str:
    base, rerank = per_query(chunking, "hybrid_rrf"), per_query(chunking, "hybrid_rerank")
    low, high = paired_bootstrap_ci(rerank, base)
    gain = sum(rerank) / len(rerank) - sum(base) / len(base)
    p95_base = summary(chunking, "hybrid_rrf")["latency_ms_p95"]
    p95_rerank = summary(chunking, "hybrid_rerank")["latency_ms_p95"] / 1000
    return (f"| {chunking} | {sum(base) / len(base):.3f} | {sum(rerank) / len(rerank):.3f} | "
            f"{gain:+.3f} [{low:+.3f}, {high:+.3f}] | {p95_base:.0f} ms → {p95_rerank:.1f} s |")


def test_every_reranking_row_matches_the_results():
    for chunking in CHUNKINGS:
        assert row(chunking) in README, chunking


def test_every_gain_is_clear_of_zero_as_the_text_claims():
    assert "Every gain is clear of zero" in README
    for chunking in CHUNKINGS:
        low, _ = paired_bootstrap_ci(per_query(chunking, "hybrid_rerank"),
                                     per_query(chunking, "hybrid_rrf"))
        assert low > 0, chunking


def test_the_run_is_what_the_text_says():
    rows = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    assert {r["model"] for r in rows} == {"intfloat/multilingual-e5-small"}
    assert all(r["queries"] == 58 for r in rows)
