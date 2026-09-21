"""The report's job is to be right about what the data does and does not show.

Two things are easy to get subtly wrong and are pinned here: pairing two
configurations on the queries *both* scored (chunking changes which queries
have a relevant chunk at all), and picking a recommendation among
configurations the bootstrap cannot separate.
"""

import json

import pytest

from turkish_rag_eval.report import (
    Configuration,
    format_table,
    load_configurations,
    metric_value,
    pair,
    recommend,
)

pytest.importorskip("numpy")


def record(question, relevance, total_relevant=1):
    return {"question": question, "relevance": relevance,
            "total_relevant": total_relevant, "found": any(relevance)}


def test_metric_value_reads_the_stored_relevance():
    hit_first = record("q", [True, False, False])
    assert metric_value(hit_first, "ndcg@10") == 1.0
    assert metric_value(hit_first, "mrr") == 1.0
    assert metric_value(hit_first, "recall@3") == 1.0
    assert metric_value(hit_first, "precision@3") == pytest.approx(1 / 3)


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        metric_value(record("q", [True]), "map@10")


def test_unscorable_queries_are_excluded():
    # A query with no relevant chunk anywhere cannot rank anything, and
    # run_eval leaves it out of the published mean.
    configuration = Configuration("sentence", "dense", [
        record("a", [True, False]),
        record("b", [False, False], total_relevant=0),
    ])
    assert len(configuration.scored) == 1
    assert configuration.mean("ndcg@10") == 1.0


def test_pairing_uses_only_the_queries_both_scored():
    a = Configuration("hierarchical", "dense", [
        record("ortak", [True, False]),
        record("yalnizca-a", [True, False]),
    ])
    b = Configuration("fixed", "dense", [
        record("ortak", [False, True]),
        record("yalnizca-a", [False, False], total_relevant=0),
    ])
    a_scores, b_scores = pair(a, b, "ndcg@10")
    assert len(a_scores) == len(b_scores) == 1
    assert a_scores[0] == 1.0


def test_pairing_matches_on_question_not_position():
    # The shared query sits at a different index on each side.
    a = Configuration("x", "dense", [record("bir", [True]), record("iki", [False])])
    b = Configuration("y", "dense", [record("iki", [True]), record("bir", [False])])
    a_scores, b_scores = pair(a, b, "mrr")
    assert list(zip(a_scores, b_scores)) == [(1.0, 0.0), (0.0, 1.0)]


def _configuration(name, scores, latency, index_seconds=None):
    """A configuration whose per-query nDCG is exactly ``scores``."""
    records = [record(f"q{i}", [True] if s else [False, True])
               for i, s in enumerate(scores)]
    summary = {"latency_ms_p95": latency}
    if index_seconds is not None:
        summary["dense_index_seconds"] = index_seconds
    return Configuration(name, "retriever", records, summary)


def test_recommendation_prefers_the_cheapest_indistinguishable_option():
    # Identical rankings, so nothing separates them; one is six times faster.
    scores = [True, False] * 20
    fast = _configuration("fast", scores, latency=4)
    slow = _configuration("slow", scores, latency=25)
    recommendation = recommend([slow, fast])
    assert recommendation.recommended is fast


def test_recommendation_keeps_the_best_when_cost_is_equal():
    # Same latency: giving up accuracy for nothing would be a bug.
    better = _configuration("better", [True] * 30 + [False] * 10, latency=25)
    worse = _configuration("worse", [True] * 20 + [False] * 20, latency=25)
    recommendation = recommend([worse, better])
    assert recommendation.recommended is better
    assert recommendation.best is better


def test_a_measurably_worse_option_is_not_recommended():
    best = _configuration("best", [True] * 40, latency=25)
    poor = _configuration("poor", [False] * 40, latency=1)
    recommendation = recommend([best, poor])
    assert recommendation.recommended is best
    assert recommendation.tied_with_best == [best]


def test_retriever_names_containing_underscores_survive_the_filename(tmp_path):
    (tmp_path / "perquery_hierarchical_bm25_stem5.json").write_text(
        json.dumps([record("q", [True])]), encoding="utf-8")
    configuration = load_configurations(tmp_path)[0]
    assert configuration.chunking == "hierarchical"
    assert configuration.retriever == "bm25_stem5"


def test_load_attaches_the_matching_summary_row(tmp_path):
    (tmp_path / "perquery_sentence_dense.json").write_text(
        json.dumps([record("q", [True])]), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps([
        {"chunking": "sentence", "retriever": "dense", "latency_ms_p95": 19.4},
    ]), encoding="utf-8")
    assert load_configurations(tmp_path)[0].latency_p95 == 19.4


def test_table_reports_an_interval_per_row():
    table = format_table([_configuration("a", [True, False] * 15, latency=4)],
                         "ndcg@10")
    body = table.splitlines()[2]
    assert body.count("|") == 6
    assert "[" in body and "ms" in body


def test_table_copes_with_a_missing_latency():
    configuration = Configuration("x", "dense", [record("q", [True])])
    assert "| - |" in format_table([configuration], "ndcg@10")
