"""How much retrieval quality survives the way people actually type Turkish.

Every gold question is rewritten three ways and scored on hierarchical chunks,
then scored again after ``normalize.QueryNormalizer``:

- ``ascii``: no Turkish keyboard - ç ğ ı ö ş ü typed as c g i o s u.
- ``caps``: caps lock, upper-cased the Turkish way (i -> İ, ı -> I).
- ``naive_lower``: the caps query passed through Python's ``str.lower()``,
  which turns "İ" into "i" plus a combining dot and "I" into "i".

Writes ``results/stress/<model>.json`` with nDCG@10 per retriever x variant x
normalisation, plus per-query relevance so the numbers can be re-derived.
"""

import argparse
import json
from pathlib import Path

from .chunking import STRATEGIES
from .corpus import load_corpus
from .gold import ROOT, load_gold, resolve_gold
from .models import DEFAULT_MODEL, prefixes_for
from .normalize import QueryNormalizer, fold_ascii
from .retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from .run_eval import CORPUS, evaluate

_UPPER = str.maketrans("iı", "İI")

VARIANTS = {
    "original": lambda q: q,
    "ascii": fold_ascii,
    "caps": lambda q: q.translate(_UPPER).upper(),
    "naive_lower": lambda q: q.translate(_UPPER).upper().lower(),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sparse-only", action="store_true",
                        help="skip the embedding model (BM25 only, no torch)")


def score(retrievers: dict, chunks: list[dict], gold: list[dict],
          normalise: QueryNormalizer) -> dict:
    out = {}
    for name, retriever in retrievers.items():
        for variant, rewrite in VARIANTS.items():
            for fixed in (False, True):
                queries = [rewrite(item["question"]) for item in gold]
                if fixed:
                    queries = [normalise(q) for q in queries]
                g = [dict(item, question=q) for item, q in zip(gold, queries)]
                summary, per_query = evaluate(retriever, chunks, g, 10)
                key = f"{name}/{variant}" + ("/normalized" if fixed else "")
                out[key] = {
                    "ndcg@10": summary["ndcg@10"],
                    "relevance": [q["relevance"] for q in per_query],
                    "total_relevant": [q["total_relevant"] for q in per_query],
                }
                print(f"  {key:40s} nDCG@10={summary['ndcg@10']:.3f}", flush=True)
    return out


def run(args) -> int:
    docs = load_corpus(args.corpus or CORPUS).docs
    gold = load_gold(resolve_gold(args.gold))
    chunks = [c for doc in docs for c in STRATEGIES["hierarchical"](doc)]
    normalise = QueryNormalizer(doc["text"] for doc in docs)
    print(f"{len(gold)} soru, {len(chunks)} parca, model: {args.model}", flush=True)

    sparse = SparseRetriever(chunks, stem_length=5)
    retrievers: dict[str, object] = {"bm25_stem5": sparse}
    if not args.sparse_only:
        from sentence_transformers import SentenceTransformer
        query_prefix, doc_prefix = prefixes_for(args.model)
        dense = DenseRetriever(chunks, SentenceTransformer(args.model),
                               query_prefix, doc_prefix)
        retrievers["dense"] = dense
        retrievers["hybrid_rrf"] = HybridRetriever(dense, sparse)

    result = {"model": args.model, "chunking": "hierarchical",
              "queries": len(gold), "scores": score(retrievers, chunks, gold, normalise)}
    out = args.out or ROOT / "results" / "stress" / f"{args.model.replace('/', '__')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"\n{out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
