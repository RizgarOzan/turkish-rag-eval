"""Produce abstain records for the best (chunking, retriever) pair.

"Correct" means the top-1 chunk actually contains the gold answer span - the
strictest reading, because that is the chunk a generator would quote from when
the system answers with no human in the loop.

Both confidence signals are recorded for every query: the margin of the ranked
retriever being evaluated, and the dense retriever's raw top-1 cosine, which
is available regardless of which retriever won.
"""

import json

from abstain import RESULTS, margin_confidence, report, score_confidence
from chunking import STRATEGIES
from gold import load_gold
from retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from run_eval import CORPUS, DEFAULT_MODEL, is_relevant


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    best = summary[0]
    print(f"en iyi yapilandirma: {best['chunking']} + {best['retriever']}\n")

    docs = json.loads(CORPUS.read_text(encoding="utf-8"))
    gold = load_gold()
    chunks = [c for doc in docs for c in STRATEGIES[best["chunking"]](doc)]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DEFAULT_MODEL)
    dense = DenseRetriever(chunks, model)
    sparse = SparseRetriever(chunks, stem_length=5)
    retrievers = {
        "dense": dense,
        "bm25_stem5": sparse,
        "bm25_nostem": SparseRetriever(chunks, stem_length=None),
        "hybrid_rrf": HybridRetriever(dense, sparse),
    }
    retriever = retrievers[best["retriever"]]

    records = []
    for item in gold:
        hits = retriever.search(item["question"], 5)
        dense_hits = dense.search(item["question"], 5)
        records.append({
            "qid": item["qid"],
            "question": item["question"],
            "margin": margin_confidence(hits),
            "score": score_confidence(dense_hits),
            "correct": bool(hits) and is_relevant(chunks[hits[0][0]], item),
            "top_chunk": chunks[hits[0][0]]["heading_path"] if hits else None,
        })

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "abstain_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    report(records)


if __name__ == "__main__":
    main()
