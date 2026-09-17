"""Inter-annotator agreement for second-annotated questions.

A second annotation reuses the original ``question``, ``doc_id`` and
``doc_title`` but is written by a different ``annotator`` without looking at
the first answer span (see CONTRIBUTING.md). It links back with
``"second_of": "<original qid>"``. Pairs missing that link are still found by
matching on the normalised question text, since a second annotation is
defined to keep the question unchanged.

Two measurements per pair, neither needing anything beyond the standard
library:

token_f1 - token overlap between the two raw answer spans.
kappa    - Cohen's kappa, per chunking strategy, over the binary "is this
           chunk relevant" label each span would assign to each chunk of the
           source document (same doc_id and body contains the span, exactly
           the definition run_eval.is_relevant uses to score retrievers).
           Raw agreement is inflated by the many chunks both annotators call
           irrelevant; kappa corrects for that chance agreement.

Only the standard library is used, so this runs without the heavier
retrieval dependencies (numpy, sentence-transformers) that run_eval.py needs.
"""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from chunking import STRATEGIES
from gold import ROOT, load_gold, normalise
from turkish_text import tokenize, turkish_lower

CORPUS = ROOT / "data" / "raw" / "corpus.json"


def is_relevant(chunk: dict, item: dict) -> bool:
    """Same definition as run_eval.is_relevant, kept dependency-free here."""
    if chunk["doc_id"] != item["doc_id"]:
        return False
    return normalise(item["answer_span"]) in normalise(chunk["body"])


def find_pairs(items: list[dict]) -> list[tuple[dict, dict]]:
    """Match each second annotation to the item it re-annotates."""
    by_qid = {item["qid"]: item for item in items}
    pairs, matched = [], set()
    for item in items:
        original_qid = item.get("second_of")
        if original_qid and original_qid in by_qid:
            original = by_qid[original_qid]
            pairs.append((original, item))
            matched.add(frozenset((original["qid"], item["qid"])))

    groups: dict[str, list[dict]] = {}
    for item in items:
        key = normalise(turkish_lower(item["question"]))
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        for a, b in combinations(group, 2):
            if frozenset((a["qid"], b["qid"])) in matched:
                continue
            if a.get("annotator") == b.get("annotator"):
                continue  # same annotator, or both missing one: not independent
            pairs.append((a, b))
            matched.add(frozenset((a["qid"], b["qid"])))
    return pairs


def span_f1(a: str, b: str) -> float:
    """token F1 between two raw answer spans."""
    tokens_a, tokens_b = tokenize(a), tokenize(b)
    if not tokens_a or not tokens_b:
        return float(tokens_a == tokens_b)
    overlap = sum((Counter(tokens_a) & Counter(tokens_b)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float | None:
    """Chance-corrected agreement between two binary label lists."""
    n = len(labels_a)
    if n == 0:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    rate_a, rate_b = sum(labels_a) / n, sum(labels_b) / n
    expected = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def chunk_kappa(original: dict, second: dict, doc: dict) -> dict[str, float | None]:
    result = {}
    for name, chunker in STRATEGIES.items():
        chunks = chunker(doc)
        labels_a = [is_relevant(c, original) for c in chunks]
        labels_b = [is_relevant(c, second) for c in chunks]
        result[name] = cohens_kappa(labels_a, labels_b)
    return result


def evaluate_pairs(
    pairs: list[tuple[dict, dict]], docs_by_id: dict[str, dict]
) -> list[dict]:
    rows = []
    for original, second in pairs:
        row = {
            "qid": original["qid"],
            "second_qid": second["qid"],
            "annotator_a": original.get("annotator", "gold"),
            "annotator_b": second.get("annotator", "gold"),
            "question": original["question"],
            "token_f1": span_f1(original["answer_span"], second["answer_span"]),
            "kappa": {},
        }
        doc = docs_by_id.get(original["doc_id"])
        if doc is not None:
            row["kappa"] = chunk_kappa(original, second, doc)
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    summary = {
        "pairs": len(rows),
        "mean_token_f1": _mean(r["token_f1"] for r in rows),
    }
    for name in STRATEGIES:
        values = [r["kappa"][name] for r in rows if r["kappa"].get(name) is not None]
        summary[f"mean_kappa_{name}"] = _mean(values)
    return summary


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CORPUS))
    args = parser.parse_args()

    items = load_gold()
    pairs = find_pairs(items)
    if not pairs:
        print("second_of ile veya aynı soruyla eşleşen etiketleyici çifti yok")
        return

    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        docs = json.loads(corpus_path.read_text(encoding="utf-8"))
    else:
        print(f"uyarı: {corpus_path} yok - parça düzeyinde kappa atlanıyor")
        docs = []
    docs_by_id = {d["doc_id"]: d for d in docs}

    rows = evaluate_pairs(pairs, docs_by_id)
    summary = summarise(rows)

    for row in rows:
        kappa = ", ".join(
            f"{k}={v:.2f}" for k, v in row["kappa"].items() if v is not None
        )
        print(
            f"{row['qid']} <-> {row['second_qid']} "
            f"({row['annotator_a']} vs {row['annotator_b']}): "
            f"token_f1={row['token_f1']:.2f}" + (f", kappa: {kappa}" if kappa else "")
        )
    print(
        f"\n{summary['pairs']} çift, ortalama token_f1="
        f"{summary['mean_token_f1']:.3f}"
    )
    for name in STRATEGIES:
        print(f"  ortalama kappa ({name}): {summary[f'mean_kappa_{name}']:.3f}")

    out = ROOT / "results" / "agreement.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "pairs": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
