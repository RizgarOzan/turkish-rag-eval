"""Decide when retrieval is too weak to answer, and escalate to a human.

In a regulated domain the useful question is not "how accurate is the system"
but "how accurate is it on the queries it chose to answer, and how many did it
choose to answer". Those move against each other, so this module sweeps the
threshold and reports the trade-off instead of picking one number and hiding
the curve.

Two confidence signals are measured, because the obvious one fails:

margin  - normalised gap between the top hit and the runner-up. Intuitive,
          and useless on RRF output: fused scores are 1/(60+rank), so the
          top-two gap is ~2% for every query, confident or not.
score   - the dense retriever's raw top-1 cosine similarity. An absolute
          measure of how close anything in the corpus came to the query,
          which is what an abstain decision actually needs.

The comparison is the point. See results/abstain_curve_*.json.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIGNALS = ("margin", "score")


def margin_confidence(hits: list[tuple[int, float]]) -> float:
    """Normalised top-1 margin in [0, 1]. 0 when the top two are tied."""
    if not hits:
        return 0.0
    top = hits[0][1]
    if top <= 0:
        return 0.0
    runner_up = hits[1][1] if len(hits) > 1 else 0.0
    return max(0.0, (top - runner_up) / top)


def score_confidence(hits: list[tuple[int, float]]) -> float:
    """Raw top-1 score, clamped to [0, 1]. Meaningful for cosine similarity."""
    if not hits:
        return 0.0
    return max(0.0, min(1.0, hits[0][1]))


def sweep(records: list[dict], signal: str, steps: int = 21) -> list[dict]:
    """Coverage / selective-accuracy curve over thresholds of one signal."""
    curve = []
    for i in range(steps):
        threshold = i / (steps - 1)
        answered = [r for r in records if r[signal] >= threshold]
        correct = sum(1 for r in answered if r["correct"])
        curve.append({
            "threshold": round(threshold, 3),
            "coverage": len(answered) / len(records) if records else 0.0,
            "selective_accuracy": correct / len(answered) if answered else None,
            "answered": len(answered),
            "escalated": len(records) - len(answered),
        })
    return curve


def pick_threshold(curve: list[dict], min_accuracy: float,
                   min_answered: int = 5) -> dict | None:
    """Highest-coverage threshold that still clears an accuracy floor.

    None means no threshold reaches the floor - a real answer, not a
    failure: the system should not run unattended at that requirement.
    """
    viable = [p for p in curve
              if p["selective_accuracy"] is not None
              and p["selective_accuracy"] >= min_accuracy
              and p["answered"] >= min_answered]
    return max(viable, key=lambda p: p["coverage"]) if viable else None


def report(records: list[dict]) -> None:
    baseline = sum(1 for r in records if r["correct"]) / len(records)
    print(f"{len(records)} sorgu | esiksiz top-1 dogrulugu {baseline:.3f}\n")

    for signal in SIGNALS:
        curve = sweep(records, signal)
        (RESULTS / f"abstain_curve_{signal}.json").write_text(
            json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"--- sinyal: {signal} ---")
        print(f"{'esik':>6} {'kapsam':>8} {'dogruluk':>9} {'cevaplanan':>11} {'insana':>7}")
        for point in curve:
            if point["answered"] == 0:
                continue
            accuracy = f"{point['selective_accuracy']:.3f}"
            print(f"{point['threshold']:>6.2f} {point['coverage']:>8.3f} "
                  f"{accuracy:>9} {point['answered']:>11} {point['escalated']:>7}")

        for floor in (0.70, 0.80, 0.90):
            choice = pick_threshold(curve, floor)
            if choice is None:
                print(f"  >= {floor:.0%} dogruluk: hicbir esik saglamiyor")
            else:
                print(f"  >= {floor:.0%} dogruluk -> esik {choice['threshold']:.2f}: "
                      f"{choice['coverage']:.0%} otomatik, "
                      f"{choice['escalated']} soru insana")
        print()


def main() -> None:
    path = RESULTS / "abstain_records.json"
    if not path.exists():
        raise SystemExit("once run_abstain.py calistirilmali")
    report(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
