"""Export the corpus and the gold set as a Hugging Face / MTEB retrieval dataset.

Writes the BEIR layout that MTEB's retrieval tasks read:

    corpus.jsonl       {"_id", "title", "text", "url"}  every article, distractors included
    queries.jsonl      {"_id", "text", "answer_span"}
    qrels/test.jsonl   {"query-id", "corpus-id", "score"}
    README.md          docs/hf-dataset-card.md, whose YAML maps the three files to configs

Qrels are article-level. The harness itself is stricter (a chunk must also
contain ``answer_span``), which is why every query keeps its span.

Only hand-labelled questions are exported, the same set the README numbers
come from; LLM drafts join once they are verified.

    python src/export_hf.py [--out data/hf]
"""

import argparse
import json
import shutil
from pathlib import Path

from .gold import ROOT, load_gold

CORPUS = ROOT / "data" / "raw" / "corpus.json"
CARD = ROOT / "docs" / "hf-dataset-card.md"


def build(gold: list[dict], docs: list[dict]) -> tuple[list, list, list]:
    ids = {d["doc_id"] for d in docs}
    missing = [f"{g['qid']} -> {g['doc_id']}" for g in gold if g["doc_id"] not in ids]
    if missing:
        raise ValueError("articles missing from the corpus (run fetch_corpus.py): "
                         + ", ".join(missing))
    corpus = [{"_id": d["doc_id"], "title": d["title"], "text": d["text"], "url": d["url"]}
              for d in docs]
    queries = [{"_id": g["qid"], "text": g["question"], "answer_span": g["answer_span"]}
               for g in gold]
    qrels = [{"query-id": g["qid"], "corpus-id": g["doc_id"], "score": 1} for g in gold]
    return corpus, queries, qrels


def write(out: Path, corpus: list, queries: list, qrels: list) -> None:
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    for path, rows in [(out / "corpus.jsonl", corpus), (out / "queries.jsonl", queries),
                       (out / "qrels" / "test.jsonl", qrels)]:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "hf")


def run(args) -> int:
    docs = json.loads(CORPUS.read_text(encoding="utf-8"))
    corpus, queries, qrels = build(load_gold(), docs)
    write(args.out, corpus, queries, qrels)
    shutil.copyfile(CARD, args.out / "README.md")
    chars = sum(len(d["text"]) for d in corpus)
    print(f"{len(corpus)} articles ({chars} characters), {len(queries)} queries, "
          f"{len(qrels)} qrels -> {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
