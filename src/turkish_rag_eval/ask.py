"""Put one question to every retriever and show what each one returns.

    turkish-rag-eval ask "Şeker hastalığında aşırı susama görülür mü?"
    turkish-rag-eval ask q001 --sparse-only

A question id from the gold set, or a question typed exactly as it is there,
gets its relevant chunks marked with a check (the same rule ``run`` scores
with), so you can see which retriever found the answer and at what rank. Any
other text is answered without marks.
"""

import argparse
from pathlib import Path

from .chunking import STRATEGIES
from .corpus import load_corpus
from .gold import load_gold, resolve_gold
from .models import DEFAULT_MODEL, prefixes_for
from .retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from .run_eval import ALL_RETRIEVERS, CORPUS, is_relevant

SNIPPET = 72


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="a gold question id (q001) or any question")
    parser.add_argument("-k", type=int, default=3, help="chunks per retriever (default 3)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunking", default="hierarchical", choices=list(STRATEGIES))
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--sparse-only", action="store_true",
                        help="skip the embedding model (BM25 only, no torch)")
    parser.add_argument("--embedding-cache", type=Path, default=None, metavar="DIR",
                        help="reuse dense chunk embeddings cached by `run`")


def _snippet(chunk: dict) -> str:
    text = " ".join(chunk["body"].split())
    return text if len(text) <= SNIPPET else text[:SNIPPET - 1] + "…"


def answer(retrievers: dict, chunks: list[dict], question: str,
           item: dict | None, k: int) -> str:
    lines = []
    found, missed = [], []
    for name, retriever in retrievers.items():
        lines += ["", name]
        hits = retriever.search(question, k)
        ranks = []
        for rank, (idx, _) in enumerate(hits, 1):
            chunk = chunks[idx]
            mark = ""
            if item is not None:
                mark = "✓ " if is_relevant(chunk, item) else "· "
                if mark == "✓ ":
                    ranks.append(rank)
            lines.append(f"  {rank} {mark}{chunk['heading_path']}: {_snippet(chunk)}")
        (found.append(f"{name}@{ranks[0]}") if ranks else missed.append(name))
    if item is not None:
        lines += ["", f"found by: {', '.join(found) or '-'}",
                  f"missed in top {k}: {', '.join(missed) or '-'}"]
    return "\n".join(lines)


def run(args) -> int:
    docs = load_corpus(args.corpus or CORPUS).docs
    gold = load_gold(resolve_gold(args.gold), include_drafts=True)
    item = next((g for g in gold if args.query in (g.get("qid"), g["question"])), None)
    question: str = item["question"] if item else args.query
    chunks = [c for doc in docs for c in STRATEGIES[args.chunking](doc)]

    sparse = SparseRetriever(chunks, stem_length=5)
    available: dict[str, object] = {"bm25_stem5": sparse,
                 "bm25_nostem": SparseRetriever(chunks, stem_length=None)}
    if not args.sparse_only:
        from sentence_transformers import SentenceTransformer
        query_prefix, doc_prefix = prefixes_for(args.model)
        dense = DenseRetriever(chunks, SentenceTransformer(args.model),
                               query_prefix, doc_prefix,
                               cache_dir=args.embedding_cache, model_name=args.model)
        available["dense"] = dense
        available["hybrid_rrf"] = HybridRetriever(dense, sparse)
    retrievers = {name: available[name] for name in ALL_RETRIEVERS if name in available}

    if item:
        print(f"{item.get('qid', '?')}: {question}")
        print(f"answer in: {item.get('doc_title', item['doc_id'])}  (✓ = chunk holds it)")
    else:
        print(question)
        print("not in the gold set, so nothing is marked")
    print(answer(retrievers, chunks, question, item, args.k))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
