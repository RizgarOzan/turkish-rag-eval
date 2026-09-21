"""The submission check is the leaderboard's only defence, so test it hard.

The property it enforces: a reported metric must follow from the per-query
relevance arrays shipped alongside it. Anyone can edit a number in
summary.json; doing so and staying consistent with twelve per-query files is a
different proposition.
"""

import json

import pytest

from turkish_rag_eval.leaderboard import (
    format_markdown,
    load_entry,
    result_directories,
    verify,
)

pytest.importorskip("numpy")

FINGERPRINT = "sha256:" + "ab" * 32


def per_query(hits):
    return [{"question": f"q{i}", "relevance": [hit], "total_relevant": 1,
             "found": hit} for i, hit in enumerate(hits)]


def write_results(directory, hits=(True, True, False, False), **overrides):
    """A one-configuration results directory whose numbers are honest."""
    directory.mkdir(parents=True, exist_ok=True)
    records = per_query(hits)
    (directory / "perquery_sentence_dense.json").write_text(
        json.dumps(records), encoding="utf-8")

    accuracy = sum(hits) / len(hits)
    row = {
        "chunking": "sentence", "retriever": "dense", "model": "test/model",
        "ndcg@10": accuracy, "recall@5": accuracy, "mrr": accuracy,
        "latency_ms_p95": 19.0,
        "corpus_fingerprint": FINGERPRINT, "harness_version": "0.1.0",
    }
    row.update(overrides)
    (directory / "summary.json").write_text(json.dumps([row]), encoding="utf-8")
    return directory


def test_honest_results_pass(tmp_path):
    errors, warnings = verify(write_results(tmp_path / "r"), strict=True)
    assert errors == [] and warnings == []


def test_an_inflated_score_is_caught(tmp_path):
    directory = write_results(tmp_path / "r", **{"ndcg@10": 0.99})
    errors, _ = verify(directory)
    assert len(errors) == 1
    assert "reports ndcg@10=0.990000" in errors[0]
    assert "gives 0.500000" in errors[0]


def test_a_row_without_its_evidence_is_caught(tmp_path):
    directory = write_results(tmp_path / "r")
    (directory / "perquery_sentence_dense.json").unlink()
    errors, _ = verify(directory)
    assert "no perquery file to support it" in errors[0]


def test_unscored_queries_are_excluded_the_same_way_run_eval_excludes_them(tmp_path):
    # Two of four queries have no relevant chunk; the mean is over the other
    # two, so a summary of 0.5 would be wrong and 1.0 is right.
    directory = tmp_path / "r"
    directory.mkdir()
    records = per_query([True, True])
    records += [{"question": "x", "relevance": [False], "total_relevant": 0,
                 "found": False} for _ in range(2)]
    (directory / "perquery_sentence_dense.json").write_text(
        json.dumps(records), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps([{
        "chunking": "sentence", "retriever": "dense", "ndcg@10": 1.0,
        "corpus_fingerprint": FINGERPRINT, "harness_version": "0.1.0",
    }]), encoding="utf-8")

    errors, _ = verify(directory, strict=True)
    assert errors == []


def test_missing_provenance_warns_but_does_not_fail(tmp_path):
    directory = write_results(tmp_path / "r", corpus_fingerprint="",
                              harness_version="")
    errors, warnings = verify(directory, strict=False)
    assert errors == []
    assert len(warnings) == 2


def test_missing_provenance_fails_a_new_submission(tmp_path):
    directory = write_results(tmp_path / "r", corpus_fingerprint="",
                              harness_version="")
    errors, warnings = verify(directory, strict=True)
    assert len(errors) == 2 and warnings == []


def test_rows_disagreeing_about_the_corpus_are_caught(tmp_path):
    directory = write_results(tmp_path / "r")
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    second = dict(rows[0], chunking="fixed", corpus_fingerprint="sha256:other")
    (directory / "perquery_fixed_dense.json").write_text(
        (directory / "perquery_sentence_dense.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(rows + [second]),
                                            encoding="utf-8")

    errors, _ = verify(directory)
    assert any("disagree about which corpus" in e for e in errors)


def test_unreadable_summary_is_reported_not_raised(tmp_path):
    directory = tmp_path / "r"
    directory.mkdir()
    (directory / "summary.json").write_text("{not json", encoding="utf-8")
    errors, _ = verify(directory)
    assert "unreadable" in errors[0]


def test_empty_summary_is_rejected(tmp_path):
    directory = tmp_path / "r"
    directory.mkdir()
    (directory / "summary.json").write_text("[]", encoding="utf-8")
    errors, _ = verify(directory)
    assert "non-empty list" in errors[0]


def test_directories_found_include_the_default_run_and_each_model(tmp_path):
    write_results(tmp_path)
    write_results(tmp_path / "models" / "org__model")
    found = result_directories(tmp_path)
    assert [d.name for d in found] == [tmp_path.name, "org__model"]


def test_the_default_run_is_labelled_when_no_model_is_recorded(tmp_path):
    directory = write_results(tmp_path / "results", model=None)
    assert load_entry(directory).model == "(default model)"


def test_a_model_directory_name_becomes_the_model_id(tmp_path):
    directory = write_results(tmp_path / "models" / "intfloat__e5-small",
                              model=None)
    assert load_entry(directory).model == "intfloat/e5-small"


def test_table_ranks_by_the_metric(tmp_path):
    weak = load_entry(write_results(tmp_path / "models" / "a__weak",
                                    hits=(True, False, False, False),
                                    model="a/weak"))
    strong = load_entry(write_results(tmp_path / "models" / "b__strong",
                                      hits=(True, True, True, False),
                                      model="b/strong"))
    table = format_markdown([weak, strong])
    assert table.index("b/strong") < table.index("a/weak")
