"""Bring-your-own-corpus: the path that turns the repo from a result into a tool.

The end-to-end case runs sparse retrievers only, which is deliberate rather
than a shortcut - a BM25-only run must work on the base install, with no model
download, or "try it on your documents" is not a real invitation.
"""

import argparse
import json
import sys

import pytest

from turkish_rag_eval import run_eval
from turkish_rag_eval.gold import resolve_gold

pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")

ARTICLE = """== Diyabet ==
Diyabet, kan şekerinin uzun süre yüksek seyrettiği bir metabolizma hastalığıdır.
Tip 1 diyabet pankreasın yeterli insülin üretememesinden kaynaklanır.

== Belirtiler ==
Sık idrara çıkma ve aşırı susama en yaygın belirtilerdir.
Tedavi edilmediğinde ketoasidoz gelişebilir.
"""

QUESTION = {
    "qid": "t-1",
    "doc_id": "diyabet.md",
    "doc_title": "diyabet",
    "question": "Şeker hastalığında hangi acil tablo gelişebilir?",
    "answer_span": "Tedavi edilmediğinde ketoasidoz gelişebilir.",
}


@pytest.fixture
def workspace(tmp_path):
    corpus = tmp_path / "belgeler"
    corpus.mkdir()
    (corpus / "diyabet.md").write_text(ARTICLE, encoding="utf-8")
    gold = tmp_path / "sorular.json"
    gold.write_text(json.dumps([QUESTION], ensure_ascii=False), encoding="utf-8")
    return tmp_path, corpus, gold


def arguments(**overrides):
    parser = argparse.ArgumentParser()
    run_eval.add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_gold_resolution_accepts_a_file_or_a_directory(tmp_path):
    first = tmp_path / "bir.json"
    first.write_text("[]", encoding="utf-8")
    (tmp_path / "iki.json").write_text("[]", encoding="utf-8")
    (tmp_path / "okuma.md").write_text("yok", encoding="utf-8")

    assert resolve_gold(first) == [first]
    assert [p.name for p in resolve_gold(tmp_path)] == ["bir.json", "iki.json"]


def test_gold_resolution_rejects_an_empty_directory(tmp_path):
    (tmp_path / "bos").mkdir()
    with pytest.raises(FileNotFoundError, match="no .json question files"):
        resolve_gold(tmp_path / "bos")


def test_entirely_mismatched_ids_stop_the_run():
    # Every metric would come back at zero, which reads as "retrieval is bad"
    # rather than "these labels belong to another corpus".
    with pytest.raises(SystemExit, match="None of the 1 document ids"):
        run_eval.check_gold_against_corpus(
            [{"doc_id": "yok.md"}], [{"doc_id": "var.md"}])


def test_partially_mismatched_ids_only_warn(capsys):
    run_eval.check_gold_against_corpus(
        [{"doc_id": "var.md"}, {"doc_id": "yok.md"}], [{"doc_id": "var.md"}])
    assert "uyarı" in capsys.readouterr().out


def test_matching_ids_say_nothing(capsys):
    run_eval.check_gold_against_corpus([{"doc_id": "var.md"}], [{"doc_id": "var.md"}])
    assert capsys.readouterr().out == ""


def test_sparse_only_run_needs_no_embedding_model(workspace, monkeypatch):
    tmp_path, corpus, gold = workspace
    out = tmp_path / "out"

    # Reaching for sentence_transformers at all means a model download, so any
    # attempt has to fail loudly rather than quietly pass on a machine that
    # happens to have the model cached.
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(
                "a BM25-only run must not load an embedding model")

    monkeypatch.setitem(sys.modules, "sentence_transformers", Forbidden())

    assert run_eval.run(arguments(
        corpus=corpus, gold=gold, out=out,
        retrievers=["bm25_stem5", "bm25_nostem"])) == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {row["retriever"] for row in summary} == {"bm25_stem5", "bm25_nostem"}
    assert len(summary) == 6, "three chunkers x two retrievers"


def test_results_record_their_provenance(workspace):
    tmp_path, corpus, gold = workspace
    out = tmp_path / "out"
    run_eval.run(arguments(corpus=corpus, gold=gold, out=out,
                           retrievers=["bm25_stem5"]))

    row = json.loads((out / "summary.json").read_text(encoding="utf-8"))[0]
    assert row["corpus_fingerprint"].startswith("sha256:")
    assert row["harness_version"]
    assert row["corpus_documents"] == 1


def test_the_answer_span_is_found(workspace):
    tmp_path, corpus, gold = workspace
    out = tmp_path / "out"
    run_eval.run(arguments(corpus=corpus, gold=gold, out=out,
                           retrievers=["bm25_stem5"]))

    records = json.loads(
        (out / "perquery_hierarchical_bm25_stem5.json").read_text(encoding="utf-8"))
    assert records[0]["found"], "the only question should retrieve its own article"
