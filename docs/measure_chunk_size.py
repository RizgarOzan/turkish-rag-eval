"""Sweeps the chunk size: 300 to 1500 characters on all three chunkings.

Every chunker takes size=700 and nothing in the README says why. Without an
argument it is BM25 only (stem 5), so it needs no model, and the rows land in
results/chunk_size/summary.json. With a model name it runs that model dense
only and writes results/chunk_size/<model>.json instead, so the BM25 curve
can be checked against a retriever that does not count words. Each row keeps
the per-question nDCG@10 keyed by qid, which tests/test_readme_chunk_size.py
uses for the paired bootstrap against 700.

A span that straddles a chunk boundary has no relevant chunk and drops out
of the score, so each row also records how many of the 58 questions were
scored and how many chunks the corpus became.

Usage:  python docs/measure_chunk_size.py [model] [--embedding-cache DIR]
"""

import argparse
import json
from pathlib import Path

from turkish_rag_eval import run_eval
from turkish_rag_eval.chunking import STRATEGIES
from turkish_rag_eval.corpus import load_corpus
from turkish_rag_eval.gold import load_gold, resolve_gold
from turkish_rag_eval.metrics import ndcg_at_k
from turkish_rag_eval.retrieval import SparseRetriever

ROOT = Path(__file__).resolve().parent.parent
SIZES = (300, 500, 700, 1000, 1500)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?")
    parser.add_argument("--embedding-cache", type=Path)
    args = parser.parse_args()

    docs = load_corpus(run_eval.CORPUS).docs
    gold = load_gold(resolve_gold(None))
    if args.model:
        from sentence_transformers import SentenceTransformer

        from turkish_rag_eval.models import prefixes_for
        from turkish_rag_eval.retrieval import DenseRetriever
        model = SentenceTransformer(args.model)
        query_prefix, doc_prefix = prefixes_for(args.model)

        def retriever(chunks):
            return DenseRetriever(chunks, model, query_prefix, doc_prefix,
                                  cache_dir=args.embedding_cache, model_name=args.model)
    else:
        def retriever(chunks):
            return SparseRetriever(chunks, stem_length=5)

    rows = []
    for strategy, chunker in STRATEGIES.items():
        for size in SIZES:
            chunks = [c for doc in docs for c in chunker(doc, size=size)]
            summary, per_query = run_eval.evaluate(retriever(chunks), chunks, gold, 10)
            rows.append({
                "chunking": strategy,
                "size": size,
                "chunks": len(chunks),
                "queries_scored": summary["queries_scored"],
                "ndcg@10": summary["ndcg@10"],
                "recall@5": summary["recall@5"],
                "mrr": summary["mrr"],
                "ndcg@10_per_query": {
                    q["qid"]: ndcg_at_k(q["relevance"], q["total_relevant"], 10)
                    for q in per_query if q["total_relevant"] > 0},
            })
            print(f"{strategy:12s} size={size:5d}  chunks={len(chunks):5d}  "
                  f"scored={summary['queries_scored']}  nDCG@10={summary['ndcg@10']:.3f}")
    out = ROOT / "results" / "chunk_size"
    out.mkdir(parents=True, exist_ok=True)
    name = args.model.split("/")[-1] + ".json" if args.model else "summary.json"
    (out / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
