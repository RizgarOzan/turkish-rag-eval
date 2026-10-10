"""Export the corpus and the gold set as a Hugging Face / MTEB retrieval dataset.

Writes the BEIR layout that MTEB's retrieval tasks read:

    corpus.jsonl       {"_id", "title", "text", "url"}  every article, distractors included
    queries.jsonl      {"_id", "text", "answer_span"}
    qrels/test.jsonl   {"query-id", "corpus-id", "score"}
    README.md          docs/hf-dataset-card.md, whose YAML maps the three files to configs

    passages/...       the same three files at passage level (configs passages-*)

Top-level qrels are article-level. The passage set applies the harness's own
rule instead: the hierarchical chunks, and a passage is relevant when it comes
from the answering article and contains ``answer_span``. Articles average
~20 000 characters, far past a 512-token encoder, so article-level retrieval
mostly scores the lead paragraph; passages are what a RAG pipeline retrieves.
A question whose span falls in no passage (e.g. it runs across a heading) is
left out of the passage set, as the harness leaves it out of its scores.

The top-level and passage sets hold only the hand-labelled questions, the
set the README numbers come from. ``full/`` holds all of them over the full
corpus (``fetch-corpus --include-drafts``), each query marked ``"source":
"human"`` or ``"llm-draft"``; drafts are double-labelled, not human-verified.

    python src/export_hf.py [--out data/hf]
"""

import argparse
import json
import shutil
from pathlib import Path

from .chunking import chunk_hierarchical
from .gold import ROOT, load_gold
from .run_eval import CORPUS_FULL, is_relevant

CORPUS = ROOT / "data" / "raw" / "corpus.json"
CARD = ROOT / "docs" / "hf-dataset-card.md"


def query_row(g: dict, mark_source: bool) -> dict:
    row = {"_id": g["qid"], "text": g["question"], "answer_span": g["answer_span"]}
    if mark_source:
        row["source"] = g.get("source", "human")
    return row


def build(gold: list[dict], docs: list[dict],
          mark_source: bool = False) -> tuple[list, list, list]:
    ids = {d["doc_id"] for d in docs}
    missing = [f"{g['qid']} -> {g['doc_id']}" for g in gold if g["doc_id"] not in ids]
    if missing:
        raise ValueError("articles missing from the corpus (run fetch_corpus.py): "
                         + ", ".join(missing))
    corpus = [{"_id": d["doc_id"], "title": d["title"], "text": d["text"], "url": d["url"]}
              for d in docs]
    queries = [query_row(g, mark_source) for g in gold]
    qrels = [{"query-id": g["qid"], "corpus-id": g["doc_id"], "score": 1} for g in gold]
    return corpus, queries, qrels


def build_passages(gold: list[dict], docs: list[dict],
                   mark_source: bool = False) -> tuple[list, list, list]:
    build(gold, docs)  # same missing-article check
    chunks = [c for d in docs for c in chunk_hierarchical(d)]
    corpus = [{"_id": c["chunk_id"], "title": c["heading_path"], "text": c["body"],
               "url": c["url"]} for c in chunks]
    queries, qrels = [], []
    for g in gold:
        hits = [c["chunk_id"] for c in chunks if is_relevant(c, g)]
        if hits:
            queries.append(query_row(g, mark_source))
            qrels += [{"query-id": g["qid"], "corpus-id": h, "score": 1} for h in hits]
    return corpus, queries, qrels


def write(out: Path, corpus: list, queries: list, qrels: list) -> None:
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    for path, rows in [(out / "corpus.jsonl", corpus), (out / "queries.jsonl", queries),
                       (out / "qrels" / "test.jsonl", qrels)]:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "hf")


def export(out: Path, gold: list[dict], docs: list[dict], mark_source: bool) -> None:
    corpus, queries, qrels = build(gold, docs, mark_source)
    write(out, corpus, queries, qrels)
    chars = sum(len(d["text"]) for d in corpus)
    print(f"{len(corpus)} articles ({chars} characters), {len(queries)} queries, "
          f"{len(qrels)} qrels -> {out}")
    corpus, queries, qrels = build_passages(gold, docs, mark_source)
    write(out / "passages", corpus, queries, qrels)
    print(f"{len(corpus)} passages, {len(queries)} queries, {len(qrels)} qrels "
          f"-> {out / 'passages'}")


def run(args) -> int:
    export(args.out, load_gold(), json.loads(CORPUS.read_text(encoding="utf-8")), False)
    shutil.copyfile(CARD, args.out / "README.md")
    if not CORPUS_FULL.exists():
        print(f"{CORPUS_FULL} missing, full/ skipped (fetch-corpus --include-drafts)")
        return 0
    export(args.out / "full", load_gold(include_drafts=True),
           json.loads(CORPUS_FULL.read_text(encoding="utf-8")), True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
