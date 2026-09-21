"""Turn a results directory into a decision.

``run_eval`` writes twelve rows of numbers. Twelve rows is not an answer to
the question people actually have, which is "what should I build with, and
what does the next tier up cost me". This module answers that:

- recompute every metric per query from the stored relevance arrays, so
  confidence intervals can be derived without re-running anything;
- find the best configuration, then use a *paired* bootstrap to find which
  others are indistinguishable from it;
- among those, recommend the cheapest, and say what the top of the table
  would buy and what it would cost in latency and indexing time.

The paired comparison is where the care is. Two configurations are compared
on the queries both of them scored, which is not all queries: ``total_relevant``
counts chunks containing the answer span, and a chunker that splits an article
differently produces a different count - sometimes zero. Pairing on the
intersection keeps the comparison honest and the report says how many queries
it used.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import (
    bootstrap_ci,
    ndcg_at_k,
    paired_bootstrap_ci,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    separates,
)
from .paths import ROOT

DEFAULT_METRIC = "ndcg@10"


def metric_value(record: dict, metric: str) -> float:
    """One query's score under ``metric`` from its stored relevance array."""
    name, _, k_text = metric.partition("@")
    k = int(k_text) if k_text else 0
    relevance, total = record["relevance"], record["total_relevant"]
    if name == "ndcg":
        return ndcg_at_k(relevance, total, k)
    if name == "recall":
        return recall_at_k(relevance, total, k)
    if name == "precision":
        return precision_at_k(relevance, k)
    if name == "mrr":
        return reciprocal_rank(relevance)
    raise ValueError(f"unknown metric '{metric}'")


@dataclass
class Configuration:
    chunking: str
    retriever: str
    records: list[dict]
    summary: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.chunking} + {self.retriever}"

    @property
    def scored(self) -> list[dict]:
        """Queries this configuration could be scored on at all.

        A query with no relevant chunk cannot distinguish a good ranking from a
        bad one, and ``run_eval`` excludes it from the published mean; doing
        the same here keeps this report and that table in agreement.
        """
        return [r for r in self.records if r["total_relevant"] > 0]

    def per_query(self, metric: str) -> list[float]:
        return [metric_value(r, metric) for r in self.scored]

    def mean(self, metric: str) -> float:
        values = self.per_query(metric)
        return sum(values) / len(values) if values else 0.0

    def interval(self, metric: str) -> tuple[float, float]:
        return bootstrap_ci(self.per_query(metric))

    @property
    def latency_p95(self) -> float | None:
        return self.summary.get("latency_ms_p95")

    @property
    def index_seconds(self) -> float | None:
        return self.summary.get("dense_index_seconds")


def load_configurations(results_dir: Path) -> list[Configuration]:
    """Read every ``perquery_<chunking>_<retriever>.json`` in a results dir."""
    summaries = {}
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        for row in json.loads(summary_path.read_text(encoding="utf-8")):
            summaries[(row["chunking"], row["retriever"])] = row

    configurations = []
    for path in sorted(results_dir.glob("perquery_*.json")):
        stem = path.stem[len("perquery_"):]
        # Chunking names have no underscore; retriever names do (bm25_stem5).
        chunking, _, retriever = stem.partition("_")
        configurations.append(Configuration(
            chunking=chunking,
            retriever=retriever,
            records=json.loads(path.read_text(encoding="utf-8")),
            summary=summaries.get((chunking, retriever), {}),
        ))
    return configurations


def pair(a: Configuration, b: Configuration,
         metric: str) -> tuple[list[float], list[float]]:
    """Per-query scores for both, restricted to the queries both scored.

    Keyed by question text rather than position: different chunking strategies
    drop different queries, so the two lists are not aligned by index.
    """
    a_by_question = {r["question"]: r for r in a.scored}
    b_by_question = {r["question"]: r for r in b.scored}
    shared = [q for q in a_by_question if q in b_by_question]
    return ([metric_value(a_by_question[q], metric) for q in shared],
            [metric_value(b_by_question[q], metric) for q in shared])


@dataclass
class Recommendation:
    best: Configuration
    recommended: Configuration
    tied_with_best: list[Configuration]
    metric: str
    shared_queries: int

    @property
    def best_is_recommended(self) -> bool:
        return self.recommended is self.best


def _cost(configuration: Configuration, metric: str) -> tuple[float, float, float]:
    """Sort key for 'cheapest'.

    Query latency first, then indexing time, then the metric itself - without
    that last term two configurations costing the same would be separated by
    dict order, and the report would talk anyone out of a free improvement.
    Latency is rounded to the millisecond because the difference between 24.6
    and 25.4 ms is timing jitter, not a reason to give up accuracy.
    """
    latency = configuration.latency_p95
    return (round(latency) if latency is not None else float("inf"),
            configuration.index_seconds if configuration.index_seconds is not None
            else 0.0,
            -configuration.mean(metric))


def recommend(configurations: list[Configuration],
              metric: str = DEFAULT_METRIC) -> Recommendation:
    """The cheapest configuration that the data cannot separate from the best."""
    ranked = sorted(configurations, key=lambda c: -c.mean(metric))
    best = ranked[0]

    tied, shared_queries = [best], len(best.scored)
    for other in ranked[1:]:
        best_scores, other_scores = pair(best, other, metric)
        if not best_scores:
            continue
        if not separates(paired_bootstrap_ci(best_scores, other_scores)):
            tied.append(other)
            shared_queries = min(shared_queries, len(best_scores))

    return Recommendation(
        best=best,
        recommended=min(tied, key=lambda c: _cost(c, metric)),
        tied_with_best=tied,
        metric=metric,
        shared_queries=shared_queries,
    )


def format_table(configurations: list[Configuration], metric: str) -> str:
    """Markdown table of every configuration with its interval and cost."""
    rows = sorted(configurations, key=lambda c: -c.mean(metric))
    lines = [
        f"| Chunking | Retriever | {metric} | 95% CI | P95 |",
        "|---|---|---|---|---|",
    ]
    for configuration in rows:
        low, high = configuration.interval(metric)
        latency = configuration.latency_p95
        latency_text = f"{latency:.0f} ms" if latency is not None else "-"
        lines.append(
            f"| {configuration.chunking} | {configuration.retriever} "
            f"| {configuration.mean(metric):.3f} | [{low:.3f}, {high:.3f}] "
            f"| {latency_text} |")
    return "\n".join(lines)


def format_recommendation(recommendation: Recommendation) -> str:
    metric = recommendation.metric
    best, chosen = recommendation.best, recommendation.recommended
    lines = [f"Recommended: {chosen.name}"]

    if recommendation.best_is_recommended:
        lines.append(
            f"  Highest {metric} ({best.mean(metric):.3f}) and nothing cheaper "
            f"comes close enough to prefer.")
    else:
        difference = best.mean(metric) - chosen.mean(metric)
        lines.append(
            f"  {chosen.mean(metric):.3f} {metric} against {best.mean(metric):.3f} "
            f"for {best.name}, a gap of {difference:.3f} that a paired bootstrap "
            f"over {recommendation.shared_queries} shared queries cannot "
            f"distinguish from zero.")
        if chosen.latency_p95 and best.latency_p95:
            lines.append(
                f"  It answers in {chosen.latency_p95:.0f} ms at P95 against "
                f"{best.latency_p95:.0f} ms.")
        if best.index_seconds:
            lines.append(
                f"  {best.name} also spends {best.index_seconds / 60:.0f} minutes "
                f"building its index.")

    if len(recommendation.tied_with_best) > 1:
        others = [c.name for c in recommendation.tied_with_best if c is not best]
        lines.append(f"  Indistinguishable from the top: {', '.join(others)}.")
    else:
        lines.append("  Every other configuration is measurably worse.")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=ROOT / "results",
                        help="directory holding perquery_*.json")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help=f"metric to rank on (default {DEFAULT_METRIC})")
    parser.add_argument("--markdown", action="store_true",
                        help="print only the table, for pasting into the README")


def run(args) -> int:
    configurations = load_configurations(args.results)
    if not configurations:
        print(f"{args.results} holds no perquery_*.json - run "
              f"'turkish-rag-eval run' first")
        return 1

    table = format_table(configurations, args.metric)
    if args.markdown:
        print(table)
        return 0

    recommendation = recommend(configurations, args.metric)
    queries = len(recommendation.best.scored)
    print(f"{len(configurations)} configurations, {queries} scored queries, "
          f"metric {args.metric}\n")
    print(table)
    print()
    print(format_recommendation(recommendation))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
