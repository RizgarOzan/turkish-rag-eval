"""Check gold-set files before they are merged.

Offline checks run over every file: required fields are non-empty strings,
qids and questions are unique across all files, contributed items name their
annotator, and each question is a paraphrase rather than a copy of its answer
span - a copied question hands BM25 an unearned win (see README).

``--online FILE ...`` additionally fetches each article those files point at
and checks that the page id, the title and the answer span all match
Turkish Wikipedia. CI passes only the files a pull request adds, so the
network work stays small.

Exit code is 1 when anything fails.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from gold import GOLD, gold_files, normalise
from turkish_text import tokenize, turkish_lower

REQUIRED = ("qid", "question", "doc_id", "doc_title", "answer_span")
# Share of the question's stemmed words that may also appear in the span.
# The original 58 questions all sit well below this.
MAX_OVERLAP = 0.6


def overlap(question: str, span: str) -> float:
    q = set(tokenize(question))
    return len(q & set(tokenize(span))) / len(q) if q else 1.0


def check_items(files: dict[str, list]) -> list[str]:
    """``files`` maps a file name to its parsed JSON content."""
    errors, qids, questions = [], {}, {}
    for name, items in files.items():
        if not isinstance(items, list):
            errors.append(f"{name}: top level must be a JSON list")
            continue
        contributed = Path(name).name != GOLD.name
        for n, item in enumerate(items, 1):
            where = f"{name} #{n}"
            if not isinstance(item, dict):
                errors.append(f"{where}: item must be a JSON object")
                continue
            fields = REQUIRED + (("annotator",) if contributed else ())
            missing = [f for f in fields
                       if not isinstance(item.get(f), str) or not item[f].strip()]
            if missing:
                errors.append(f"{where}: missing or empty {', '.join(missing)}")
                continue
            where = f"{name} {item['qid']}"
            if item["qid"] in qids:
                errors.append(f"{where}: qid already used in {qids[item['qid']]}")
            qids[item["qid"]] = name
            key = normalise(turkish_lower(item["question"]))
            if key in questions:
                errors.append(f"{where}: same question as {questions[key]}")
            questions[key] = where
            if not item["doc_id"].isdigit():
                errors.append(f"{where}: doc_id must be the numeric Wikipedia page id")
            if not item["question"].rstrip().endswith("?"):
                errors.append(f"{where}: question must end with '?'")
            share = overlap(item["question"], item["answer_span"])
            if share > MAX_OVERLAP:
                errors.append(f"{where}: {share:.0%} of the question's words come "
                              f"from the answer span - paraphrase it")
    return errors


def fetch_article(doc_id: str) -> dict | None:
    import requests
    from fetch_corpus import API, HEADERS
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "pageids": doc_id, "format": "json"}
    for attempt in range(5):  # Wikipedia answers bursts with 429
        response = requests.get(API, params=params, timeout=60, headers=HEADERS)
        if response.status_code != 429:
            break
        time.sleep(5 * (attempt + 1))
    response.raise_for_status()
    page = response.json()["query"]["pages"].get(doc_id, {})
    if "extract" not in page:
        return None
    return {"title": page["title"], "text": page["extract"]}


def check_online(items: list[dict], fetch=fetch_article, delay: float = 1.0) -> list[str]:
    from fetch_corpus import MIN_EXTRACT_CHARS
    errors, articles = [], {}
    for item in items:
        doc_id = item["doc_id"]
        if doc_id not in articles:
            articles[doc_id] = fetch(doc_id)
            time.sleep(delay)
        article, where = articles[doc_id], item["qid"]
        if article is None:
            errors.append(f"{where}: no Turkish Wikipedia article with page id {doc_id}")
            continue
        if article["title"] != item["doc_title"]:
            errors.append(f"{where}: page {doc_id} is '{article['title']}', "
                          f"not '{item['doc_title']}'")
        if len(article["text"]) < MIN_EXTRACT_CHARS:
            errors.append(f"{where}: article is too short to be in the corpus")
        if normalise(item["answer_span"]) not in normalise(article["text"]):
            errors.append(f"{where}: answer_span not found verbatim in the article")
        second = item.get("second_annotation")
        if second and normalise(second["answer_span"]) not in normalise(article["text"]):
            errors.append(f"{where}: second_annotation span not found verbatim in the article")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", nargs="*", default=[], metavar="FILE",
                        help="also check these files against Wikipedia")
    args = parser.parse_args()

    files = {}
    for path in gold_files():
        try:
            files[str(path.relative_to(GOLD.parents[2]))] = json.loads(
                path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{path.name}: invalid JSON - {exc}")
            return 1
    errors = check_items(files)
    for name in args.online:
        items = json.loads(Path(name).read_text(encoding="utf-8"))
        errors += check_online([i for i in items if isinstance(i, dict) and "doc_id" in i])

    for error in errors:
        print(error)
    total = sum(len(v) for v in files.values() if isinstance(v, list))
    print(f"{len(files)} dosya, {total} soru, {len(errors)} hata")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
