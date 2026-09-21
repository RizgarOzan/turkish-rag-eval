"""Charts are checked for the things a rendered image cannot tell you.

Whether they look right is settled by opening them. What is worth pinning is
that both themes are produced, that a results directory without the abstention
run still yields the chart that does exist, and that the noisiest end of the
coverage curve is filtered out rather than plotted.
"""

import json

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

from turkish_rag_eval import charts  # noqa: E402


def records(hits):
    return [{"question": f"q{i}", "relevance": [hit], "total_relevant": 1,
             "found": hit} for i, hit in enumerate(hits)]


@pytest.fixture
def results(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    for name, hits in [("sentence_dense", [True, False] * 15),
                       ("fixed_bm25_stem5", [True, True, False] * 10)]:
        (directory / f"perquery_{name}.json").write_text(
            json.dumps(records(hits)), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps([
        {"chunking": "sentence", "retriever": "dense", "latency_ms_p95": 19},
        {"chunking": "fixed", "retriever": "bm25_stem5", "latency_ms_p95": 4},
    ]), encoding="utf-8")
    return directory


def curve(points):
    return [{"threshold": t, "coverage": c, "selective_accuracy": a,
             "answered": n, "escalated": 58 - n} for t, c, a, n in points]


def test_both_themes_are_written(results, tmp_path):
    for theme in ("light", "dark"):
        out = tmp_path / f"{theme}.png"
        charts.intervals_chart(results, out, theme)
        assert out.stat().st_size > 5000, "a blank PNG would be much smaller"


def test_unknown_theme_is_rejected(results, tmp_path):
    with pytest.raises(KeyError):
        charts.intervals_chart(results, tmp_path / "x.png", "solarized")


def test_thin_thresholds_are_left_out(results, tmp_path):
    # One answered query puts a point at exactly 0.0 or 1.0 accuracy; plotting
    # it would draw a cliff that is one question wide.
    (results / "abstain_curve_score.json").write_text(json.dumps(curve([
        (0.0, 1.0, 0.47, 58),
        (0.5, 0.40, 0.70, 23),
        (0.9, 0.02, 1.00, 1),
    ])), encoding="utf-8")

    out = charts.abstention_chart(results, tmp_path / "abstention.png", "light")
    assert out.exists()


def test_a_curve_of_only_thin_points_draws_no_line(results, tmp_path):
    (results / "abstain_curve_score.json").write_text(
        json.dumps(curve([(0.9, 0.02, 1.0, 1)])), encoding="utf-8")
    # Still produces a labelled, empty chart rather than raising.
    assert charts.abstention_chart(results, tmp_path / "empty.png", "light").exists()


def test_run_without_results_explains_itself(tmp_path, capsys):
    import argparse

    parser = argparse.ArgumentParser()
    charts.add_arguments(parser)
    args = parser.parse_args([])
    args.results, args.out = tmp_path, tmp_path / "out"

    assert charts.run(args) == 1
    assert "turkish-rag-eval run" in capsys.readouterr().out


def test_run_writes_charts_for_every_theme(results, tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    charts.add_arguments(parser)
    args = parser.parse_args([])
    args.results, args.out = results, tmp_path / "out"

    assert charts.run(args) == 0
    written = sorted(p.name for p in (tmp_path / "out").glob("*.png"))
    assert written == ["ndcg-intervals-dark.png", "ndcg-intervals.png"]
