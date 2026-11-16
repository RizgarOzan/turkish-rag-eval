"""Sweeps BM25's k1 and b on the 5-character stem.

The harness uses rank_bm25's defaults (k1=1.5, b=0.75), never tuned. This asks
whether the BM25 number every other finding is compared against is a weak
baseline. No model, no download. Rows land in results/bm25_params/summary.json
with the per-question nDCG@10 for the paired bootstrap against the default.

Usage:  python docs/measure_bm25.py
"""

import json
from pathlib import Path

from turkish_rag_eval import run_eval
from turkish_rag_eval.chunking import STRATEGIES
from turkish_rag_eval.corpus import load_corpus
from turkish_rag_eval.gold import load_gold, resolve_gold
from turkish_rag_eval.metrics import ndcg_at_k
from turkish_rag_eval.retrieval import SparseRetriever

ROOT = Path(__file__).resolve().parent.parent
K1S = (0.6, 0.9, 1.2, 1.5, 2.0, 3.0)
BS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> int:
    docs = load_corpus(run_eval.CORPUS).docs
    gold = load_gold(resolve_gold(None))
    rows = []
    for strategy, chunker in STRATEGIES.items():
        chunks = [c for doc in docs for c in chunker(doc)]
        for k1 in K1S:
            for b in BS:
                summary, per_query = run_eval.evaluate(
                    SparseRetriever(chunks, stem_length=5, k1=k1, b=b), chunks, gold, 10)
                rows.append({
                    "chunking": strategy,
                    "k1": k1,
                    "b": b,
                    "queries_scored": summary["queries_scored"],
                    "ndcg@10": summary["ndcg@10"],
                    "recall@5": summary["recall@5"],
                    "mrr": summary["mrr"],
                    "ndcg@10_per_query": [
                        ndcg_at_k(q["relevance"], q["total_relevant"], 10)
                        for q in per_query if q["total_relevant"] > 0],
                })
                print(f"{strategy:12s} k1={k1} b={b}  nDCG@10={summary['ndcg@10']:.3f}")
    out = ROOT / "results" / "bm25_params"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
