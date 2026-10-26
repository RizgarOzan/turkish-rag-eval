"""Fail CI when the BM25 path gets twice as slow.

    python docs/speed_gate.py              # compare with results/speed/baseline.json
    python docs/speed_gate.py --write      # re-measure and overwrite the baseline

Three stages run on the pinned corpus: hierarchical chunking, the BM25 index
build, and the 58 gold questions searched at k=5. A CI runner and a laptop do
not run at the same speed, so each stage is timed against a fixed pure-Python
loop on the same machine and the ratio is what gets compared. The ratio still
moves: up to ~15% between runs, and the index build is ~1.6x apart on 3.13 and
3.14. So the committed baseline is the slowest of three runs on each version,
and the gate only fails past 2x of it: it catches an accidental quadratic loop
or a re-tokenised corpus, not a 10% drift. ``--write`` records one machine and
one version; to rebuild the baseline, take the slowest per stage again.
"""

import argparse
import json
import platform
import statistics
import time
from datetime import date

from turkish_rag_eval.chunking import STRATEGIES
from turkish_rag_eval.corpus import load_corpus
from turkish_rag_eval.gold import load_gold
from turkish_rag_eval.paths import ROOT
from turkish_rag_eval.retrieval import SparseRetriever
from turkish_rag_eval.run_eval import CORPUS

BASELINE = ROOT / "results" / "speed" / "baseline.json"
STAGES = ("chunk", "index", "search")
FACTOR = 2.0


def _calibration() -> int:
    counts: dict[str, int] = {}
    for i in range(200_000):
        key = str(i * 7919 % 100_003)
        counts[key] = counts.get(key, 0) + len(key)
    return sum(counts.values())


def _median_seconds(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def measure(repeat: int = 5) -> dict[str, float]:
    """Each stage's median time divided by the calibration loop's."""
    docs = load_corpus(CORPUS).docs
    questions = [g["question"] for g in load_gold()]
    chunks = [c for doc in docs for c in STRATEGIES["hierarchical"](doc)]
    retriever = SparseRetriever(chunks, stem_length=5)
    unit = _median_seconds(_calibration, repeat)
    seconds = {
        "chunk": _median_seconds(
            lambda: [c for doc in docs for c in STRATEGIES["hierarchical"](doc)], repeat),
        "index": _median_seconds(lambda: SparseRetriever(chunks, stem_length=5), repeat),
        "search": _median_seconds(lambda: [retriever.search(q, 5) for q in questions], repeat),
    }
    return {stage: round(seconds[stage] / unit, 2) for stage in STAGES}


def check(measured: dict[str, float], baseline: dict[str, float],
          factor: float = FACTOR) -> list[str]:
    """One line per stage slower than ``factor`` times its baseline ratio."""
    failures = []
    for stage, ratio in measured.items():
        if stage not in baseline:
            failures.append(f"{stage}: no baseline")
        elif ratio > factor * baseline[stage]:
            failures.append(f"{stage}: {ratio} vs baseline {baseline[stage]} "
                            f"({ratio / baseline[stage]:.1f}x, limit {factor}x)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="overwrite the baseline")
    args = parser.parse_args()

    measured = measure()
    if args.write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({
            "ratios": measured, "date": date.today().isoformat(),
            "python": platform.python_version(),
            "machine": platform.processor() or platform.machine()},
            indent=2) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE.relative_to(ROOT)}: {measured}")
        return 0
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["ratios"]
    for stage in STAGES:
        print(f"{stage:<7} {measured[stage]:>6}  baseline {baseline.get(stage)}")
    failures = check(measured, baseline)
    for line in failures:
        print(f"too slow - {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
