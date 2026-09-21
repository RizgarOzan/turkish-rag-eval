"""Run every (chunking strategy x retriever) combination over the gold set.

Relevance is defined at the *answer span* level, not the chunk level: a gold
item records which document holds the answer and a short verbatim span from
it. A retrieved chunk counts as relevant when it comes from that document and
contains the span. That definition survives changing the chunker, which is
the whole point - otherwise every strategy would need its own labels.
"""

import argparse
import json
import statistics
import time

from .chunking import STRATEGIES
from .gold import ROOT, load_gold, normalise
from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from .models import DEFAULT_MODEL, prefixes_for, results_dir
from .retrieval import DenseRetriever, HybridRetriever, SparseRetriever

CORPUS = ROOT / "data" / "raw" / "corpus.json"
K_VALUES = (1, 3, 5, 10)


def is_relevant(chunk: dict, item: dict) -> bool:
    # normalise() folds case with a plain str.lower(), not turkish_lower(). That
    # is deliberate here and is *not* the naive-casing trap the harness studies.
    # turkish_lower() matters when a query is matched against a document: the two
    # are written independently, so surface form and casing diverge. This gate
    # instead matches a verbatim answer span against the very text it was copied
    # from, so both operands are the same run of characters. lower() is
    # context-free per character, so if the span is a substring of the body it
    # stays one after either fold - the Turkish i-rule cannot change the verdict.
    # Crucially, validate_gold.py enforces the "span appears verbatim in the
    # article" invariant with this same normalise(); using a different fold here
    # would silently desynchronise the evaluator from the check that admits gold.
    if chunk["doc_id"] != item["doc_id"]:
        return False
    return normalise(item["answer_span"]) in normalise(chunk["body"])


def evaluate(retriever, chunks: list[dict], gold: list[dict], k_max: int):
    per_query, latencies = [], []
    for item in gold:
        start = time.perf_counter()
        hits = retriever.search(item["question"], k_max)
        latencies.append((time.perf_counter() - start) * 1000)

        relevance = [is_relevant(chunks[idx], item) for idx, _ in hits]
        total_relevant = sum(1 for c in chunks if is_relevant(c, item))
        per_query.append({
            "question": item["question"],
            "total_relevant": total_relevant,
            "relevance": relevance,
            "top_score": hits[0][1] if hits else 0.0,
            "found": any(relevance),
        })

    scored = [q for q in per_query if q["total_relevant"] > 0]
    summary = {
        "queries": len(gold),
        "queries_scored": len(scored),
        "mrr": _mean(reciprocal_rank(q["relevance"]) for q in scored),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": _percentile(latencies, 95),
    }
    for k in K_VALUES:
        summary[f"recall@{k}"] = _mean(
            recall_at_k(q["relevance"], q["total_relevant"], k) for q in scored)
        summary[f"precision@{k}"] = _mean(
            precision_at_k(q["relevance"], k) for q in scored)
        summary[f"ndcg@{k}"] = _mean(
            ndcg_at_k(q["relevance"], q["total_relevant"], k) for q in scored)
    return summary, per_query


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round(pct / 100 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return ordered[max(idx, 0)]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k-max", type=int, default=10)


def run(args) -> int:
    docs = json.loads(CORPUS.read_text(encoding="utf-8"))
    gold = load_gold()
    print(f"{len(docs)} belge, {len(gold)} soru\nmodel: {args.model}\n")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    query_prefix, doc_prefix = prefixes_for(args.model)
    results = results_dir(args.model)

    rows = []
    for strategy_name, chunker in STRATEGIES.items():
        chunks = [c for doc in docs for c in chunker(doc)]
        sizes = [len(c["body"]) for c in chunks]
        print(f"[{strategy_name}] {len(chunks)} parca, "
              f"ortalama {statistics.mean(sizes):.0f} karakter")

        start = time.perf_counter()
        dense = DenseRetriever(chunks, model, query_prefix, doc_prefix)
        index_seconds = time.perf_counter() - start
        print(f"    dense indeks: {index_seconds:.0f} s")
        sparse_stem = SparseRetriever(chunks, stem_length=5)
        sparse_raw = SparseRetriever(chunks, stem_length=None)
        variants = {
            "dense": dense,
            "bm25_stem5": sparse_stem,
            "bm25_nostem": sparse_raw,
            "hybrid_rrf": HybridRetriever(dense, sparse_stem),
        }
        for variant_name, retriever in variants.items():
            summary, per_query = evaluate(retriever, chunks, gold, args.k_max)
            summary["chunking"] = strategy_name
            summary["retriever"] = variant_name
            summary["chunks"] = len(chunks)
            summary["model"] = args.model
            summary["dense_index_seconds"] = index_seconds
            rows.append(summary)
            print(f"    {variant_name:12s} "
                  f"R@5={summary['recall@5']:.3f} "
                  f"nDCG@10={summary['ndcg@10']:.3f} "
                  f"MRR={summary['mrr']:.3f} "
                  f"P95={summary['latency_ms_p95']:.0f}ms")
            results.mkdir(parents=True, exist_ok=True)
            (results / f"perquery_{strategy_name}_{variant_name}.json").write_text(
                json.dumps(per_query, ensure_ascii=False, indent=2),
                encoding="utf-8")

    rows.sort(key=lambda r: -r["ndcg@10"])
    (results / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nen iyi: {rows[0]['chunking']} + {rows[0]['retriever']} "
          f"(nDCG@10={rows[0]['ndcg@10']:.3f})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
