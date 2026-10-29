"""Sweeps the BM25 stem length: 3 to 8 characters, and no stemming.

The harness fixes the prefix at 5, and finding 2 rests on it, so this asks
whether 5 is a good choice or just the first one tried. BM25 only, so it needs
no model and no download. Rows land in results/stem/summary.json with the
per-question nDCG@10, which tests/test_readme_stem.py uses for the paired
bootstrap against 5.

Usage:  python docs/measure_stem.py
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
LENGTHS = (None, 3, 4, 5, 6, 7, 8)


def main() -> int:
    docs = load_corpus(run_eval.CORPUS).docs
    gold = load_gold(resolve_gold(None))
    rows = []
    for strategy, chunker in STRATEGIES.items():
        chunks = [c for doc in docs for c in chunker(doc)]
        for length in LENGTHS:
            summary, per_query = run_eval.evaluate(
                SparseRetriever(chunks, stem_length=length), chunks, gold, 10)
            rows.append({
                "chunking": strategy,
                "stem_length": length,
                "queries_scored": summary["queries_scored"],
                "ndcg@10": summary["ndcg@10"],
                "recall@5": summary["recall@5"],
                "mrr": summary["mrr"],
                "ndcg@10_per_query": [
                    ndcg_at_k(q["relevance"], q["total_relevant"], 10)
                    for q in per_query if q["total_relevant"] > 0],
            })
            print(f"{strategy:12s} stem={length}  nDCG@10={summary['ndcg@10']:.3f}")
    out = ROOT / "results" / "stem"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
