"""The model leaderboard, and the check that keeps it worth reading.

A leaderboard is the cheapest way to make a benchmark used rather than merely
published: people submit to leaderboards. It is also the easiest thing to make
worthless, because a table of numbers a submitter typed in is a table of
numbers a submitter typed in.

So a submission is not a row, it is a results directory - the summary plus the
per-query relevance array behind every number - and ``--check`` re-derives each
reported metric from those arrays. Getting a fabricated score past it means
fabricating a self-consistent set of per-query judgements for all twelve
configurations, and doing it against a corpus whose fingerprint is recorded.
That is not proof, but it is a great deal more than trust.

Checking comes in two strengths, mirroring how gold files are validated:
everything is checked for internal consistency, while directories named on the
command line - in CI, the ones a pull request touches - must additionally carry
provenance and match the pinned corpus. Results produced before provenance was
recorded stay readable without turning the build red.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .paths import ROOT
from .report import DEFAULT_METRIC, metric_value

#: Recomputation should be exact; this only absorbs JSON float round-tripping.
TOLERANCE = 1e-9

DEFAULT_LOCK = ROOT / "data" / "corpus.lock.json"

METRICS = ("ndcg@10", "recall@5", "mrr")


@dataclass
class Entry:
    model: str
    directory: Path
    best: dict
    rows: list[dict]

    @property
    def fingerprint(self) -> str:
        return self.best.get("corpus_fingerprint", "")

    @property
    def harness_version(self) -> str:
        return self.best.get("harness_version", "")

    @property
    def configuration(self) -> str:
        return f"{self.best['chunking']} + {self.best['retriever']}"


def result_directories(root: Path) -> list[Path]:
    """Every directory holding a summary.json: the default run and each model."""
    directories = []
    if (root / "summary.json").exists():
        directories.append(root)
    models = root / "models"
    if models.is_dir():
        directories.extend(sorted(d for d in models.iterdir()
                                  if (d / "summary.json").exists()))
    return directories


def load_entry(directory: Path, metric: str = DEFAULT_METRIC) -> Entry | None:
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not rows:
        return None
    best = max(rows, key=lambda r: r.get(metric, 0.0))
    # Directories written before the model was recorded are the default model's.
    model = best.get("model") or _model_from_directory(directory)
    return Entry(model=model, directory=directory, best=best, rows=rows)


def _model_from_directory(directory: Path) -> str:
    if directory.parent.name == "models":
        return directory.name.replace("__", "/")
    return "(default model)"


def verify(directory: Path, strict: bool = False,
           lock_path: Path | None = None) -> tuple[list[str], list[str]]:
    """Re-derive every reported metric from the per-query files.

    Returns (errors, warnings). An error means the directory's numbers do not
    follow from its own evidence; a warning means the evidence is thinner than
    a new submission is allowed to be.

    ``lock_path`` is the corpus lockfile a strict check compares against, and
    is explicit rather than read from the repository root so that verifying a
    results directory is a function of its arguments - a check that silently
    consults surrounding state gives different verdicts on the same input.
    """
    errors: list[str] = []
    warnings: list[str] = []
    name = directory.name

    try:
        rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{name}: summary.json unreadable - {exc}"], []
    if not isinstance(rows, list) or not rows:
        return [f"{name}: summary.json is not a non-empty list of rows"], []

    fingerprints = set()
    for row in rows:
        missing = [key for key in ("chunking", "retriever") if key not in row]
        if missing:
            errors.append(f"{name}: a row is missing {', '.join(missing)}")
            continue

        label = f"{row['chunking']}_{row['retriever']}"
        per_query_path = directory / f"perquery_{label}.json"
        if not per_query_path.exists():
            errors.append(f"{name}: {label} has no perquery file to support it")
            continue

        records = json.loads(per_query_path.read_text(encoding="utf-8"))
        scored = [r for r in records if r["total_relevant"] > 0]
        for metric in METRICS:
            if metric not in row:
                continue
            values = [metric_value(r, metric) for r in scored]
            recomputed = sum(values) / len(values) if values else 0.0
            if abs(recomputed - row[metric]) > TOLERANCE:
                errors.append(
                    f"{name}: {label} reports {metric}={row[metric]:.6f} but its "
                    f"per-query file gives {recomputed:.6f}")

        fingerprints.add(row.get("corpus_fingerprint", ""))

    if len(fingerprints) > 1:
        errors.append(f"{name}: rows disagree about which corpus they used "
                      f"({len(fingerprints)} fingerprints)")

    fingerprint = next(iter(fingerprints), "")
    if not fingerprint:
        message = (f"{name}: no corpus fingerprint - re-run with this version "
                   f"so the numbers say which corpus produced them")
        (errors if strict else warnings).append(message)
    elif strict:
        errors.extend(_fingerprint_against_lock(name, fingerprint, lock_path))

    if not rows[0].get("harness_version"):
        message = f"{name}: no harness version recorded"
        (errors if strict else warnings).append(message)

    return errors, warnings


def _fingerprint_against_lock(name: str, fingerprint: str,
                              lock_path: Path | None) -> list[str]:
    lock_path = lock_path or DEFAULT_LOCK
    if not lock_path.exists():
        return []
    locked = json.loads(lock_path.read_text(encoding="utf-8")).get("fingerprint")
    if locked and fingerprint != locked:
        return [f"{name}: ran on a corpus that is not the pinned one "
                f"({fingerprint[:23]}... against {locked[:23]}...)"]
    return []


def format_markdown(entries: list[Entry], metric: str = DEFAULT_METRIC) -> str:
    """The leaderboard table, best configuration per model."""
    ranked = sorted(entries, key=lambda e: -e.best.get(metric, 0.0))
    lines = [
        f"| # | Model | Best configuration | {metric} | R@5 | MRR | P95 | Index |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for position, entry in enumerate(ranked, 1):
        best = entry.best
        index_seconds = best.get("dense_index_seconds")
        index = f"{index_seconds / 60:.0f} min" if index_seconds else "-"
        latency = best.get("latency_ms_p95")
        latency_text = f"{latency:.0f} ms" if latency is not None else "-"
        lines.append(
            f"| {position} | `{entry.model}` | {entry.configuration} "
            f"| **{best.get(metric, 0):.3f}** | {best.get('recall@5', 0):.3f} "
            f"| {best.get('mrr', 0):.3f} | {latency_text} | {index} |")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--check", action="store_true",
                        help="verify every entry against its per-query files")
    parser.add_argument("--strict", nargs="*", default=None, metavar="DIR",
                        help="directories that must also carry provenance and "
                             "match the pinned corpus; in CI, the ones the pull "
                             "request touches")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK,
                        help="corpus lockfile a strict check compares against")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the table to this file")


def run(args) -> int:
    directories = result_directories(args.results)
    if not directories:
        print(f"{args.results} holds no summary.json - run "
              f"'turkish-rag-eval run' first")
        return 1

    if args.check or args.strict is not None:
        strict_paths = {Path(p).resolve() for p in (args.strict or [])}
        errors, warnings = [], []
        for directory in directories:
            directory_errors, directory_warnings = verify(
                directory, strict=directory.resolve() in strict_paths,
                lock_path=getattr(args, "lock", None))
            errors += directory_errors
            warnings += directory_warnings

        for warning in warnings:
            print(f"uyarı: {warning}")
        for error in errors:
            print(f"hata: {error}")
        print(f"{len(directories)} sonuç dizini, {len(errors)} hata, "
              f"{len(warnings)} uyarı")
        if errors:
            return 1

    entries = [e for e in (load_entry(d, args.metric) for d in directories) if e]
    table = format_markdown(entries, args.metric)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
