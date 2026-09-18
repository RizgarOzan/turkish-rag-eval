"""Inter-annotator agreement for second-annotated questions.

A second annotation reuses the original ``question``, ``doc_id`` and
``doc_title`` but is written by a different ``annotator`` without looking at
the first answer span (see CONTRIBUTING.md), and links back with
``"second_of": "<original qid>"``. A question can end up with more than two
annotators, so agreement is computed over every pairwise combination and
reported as the mean per question - not just the first pair - before being
averaged again, equally per question, into the headline numbers.

Two measurements, decided in the issue #12 discussion:

iou      - Jaccard similarity of the two spans' token sets. The headline
           number: one symmetric score that, unlike F1, correctly charges the
           extra material when one span is a superset of the other (the
           common case of two annotators bracketing the same fact with
           different amounts of surrounding clause).
token_f1 - reported alongside IoU because span-extraction/QA evaluation
           traditionally looks for it.
kappa    - Cohen's kappa, per chunking strategy, over the binary "is this
           chunk relevant" label each span would assign to each chunk of the
           source document (same doc_id and body contains the span, exactly
           the definition run_eval.is_relevant uses to score retrievers).
           Kappa, not IoU, is used here because most chunks are irrelevant to
           both annotators - raw or set-based overlap would be inflated by
           that chance agreement, which kappa corrects for.

All three use `turkish_text.tokenize`, so the casing / I-ı folding is
consistent with the rest of the harness. Only the standard library is used,
so this runs without the heavier retrieval dependencies (numpy,
sentence-transformers) that run_eval.py needs.
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


def group_by_question(items: list[dict]) -> list[list[dict]]:
    """Every item answering the same question, keyed by normalised text.

    A second annotation is required to reuse the question unchanged (see
    validate_gold.py), so this alone finds every annotator of a question -
    ``second_of`` does not need to be followed to group them.
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        key = normalise(turkish_lower(item["question"]))
        groups.setdefault(key, []).append(item)
    return [group for group in groups.values() if len(group) >= 2]


def root_qid(group: list[dict]) -> str:
    """The qid other items in the group point at with ``second_of``."""
    for item in group:
        if not isinstance(item.get("second_of"), str):
            return item["qid"]
    return min(item["qid"] for item in group)


def pairwise_items(group: list[dict]) -> list[tuple[dict, dict]]:
    """Every combination in the group with independent annotators."""
    return [(a, b) for a, b in combinations(group, 2)
            if a.get("annotator") != b.get("annotator")]


def span_iou(a: str, b: str) -> float:
    """Jaccard similarity of the two spans' token sets."""
    tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 1.0


def span_f1(a: str, b: str) -> float:
    """SQuAD-style token F1 between two raw answer spans."""
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


def chunk_kappa(item_a: dict, item_b: dict, doc: dict) -> dict[str, float | None]:
    result = {}
    for name, chunker in STRATEGIES.items():
        chunks = chunker(doc)
        labels_a = [is_relevant(c, item_a) for c in chunks]
        labels_b = [is_relevant(c, item_b) for c in chunks]
        result[name] = cohens_kappa(labels_a, labels_b)
    return result


def evaluate_groups(groups: list[list[dict]],
                     docs_by_id: dict[str, dict]) -> list[dict]:
    """One row per question, averaging every annotator pair it has."""
    rows = []
    for group in groups:
        pairs = pairwise_items(group)
        if not pairs:
            continue
        pair_metrics = []
        for a, b in pairs:
            doc = docs_by_id.get(a["doc_id"])
            pair_metrics.append({
                "annotators": (a.get("annotator", "gold"), b.get("annotator", "gold")),
                "iou": span_iou(a["answer_span"], b["answer_span"]),
                "token_f1": span_f1(a["answer_span"], b["answer_span"]),
                "kappa": chunk_kappa(a, b, doc) if doc is not None else {},
            })
        rows.append({
            "qid": root_qid(group),
            "question": group[0]["question"],
            "annotators": sorted({m for p in pair_metrics for m in p["annotators"]}),
            "pairs": len(pairs),
            "mean_iou": _mean(p["iou"] for p in pair_metrics),
            "mean_token_f1": _mean(p["token_f1"] for p in pair_metrics),
            "mean_kappa": {
                name: _mean_or_none(p["kappa"].get(name) for p in pair_metrics
                                     if p["kappa"].get(name) is not None)
                for name in STRATEGIES
            },
            "pair_details": pair_metrics,
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    """Headline numbers: each question counts once, regardless of how many
    annotators (and therefore pairs) it has."""
    summary = {
        "questions": len(rows),
        "pairs": sum(r["pairs"] for r in rows),
        "mean_iou": _mean(r["mean_iou"] for r in rows),
        "mean_token_f1": _mean(r["mean_token_f1"] for r in rows),
    }
    for name in STRATEGIES:
        summary[f"mean_kappa_{name}"] = _mean_or_none(
            r["mean_kappa"][name] for r in rows if r["mean_kappa"][name] is not None)
    return summary


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CORPUS))
    args = parser.parse_args()

    groups = group_by_question(load_gold())
    if not groups:
        print("birden fazla etiketleyicisi olan soru yok")
        return

    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        docs = json.loads(corpus_path.read_text(encoding="utf-8"))
    else:
        print(f"uyarı: {corpus_path} yok - parça düzeyinde kappa atlanıyor")
        docs = []
    docs_by_id = {d["doc_id"]: d for d in docs}

    rows = evaluate_groups(groups, docs_by_id)
    summary = summarise(rows)

    for row in rows:
        kappa = ", ".join(f"{k}={v:.2f}" for k, v in row["mean_kappa"].items()
                           if v is not None)
        print(f"{row['qid']} ({', '.join(row['annotators'])}, {row['pairs']} çift): "
              f"iou={row['mean_iou']:.2f} token_f1={row['mean_token_f1']:.2f}"
              + (f", kappa: {kappa}" if kappa else ""))
    print(f"\n{summary['questions']} soru, {summary['pairs']} çift, "
          f"ortalama IoU={summary['mean_iou']:.3f} "
          f"ortalama token_f1={summary['mean_token_f1']:.3f}")
    for name in STRATEGIES:
        kappa = summary[f"mean_kappa_{name}"]
        print(f"  ortalama kappa ({name}): "
              + (f"{kappa:.3f}" if kappa is not None else "n/a"))

    out = ROOT / "results" / "agreement.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "questions": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
