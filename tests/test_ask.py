"""The ask command puts one question to every retriever and marks the chunks
that hold the gold answer, by the same rule ``run`` scores with."""

import argparse
import json

import pytest

from turkish_rag_eval import ask

pytest.importorskip("rank_bm25")

ARTICLE = """== Diyabet ==
Diyabet, kan şekerinin uzun süre yüksek seyrettiği bir metabolizma hastalığıdır.

== Belirtiler ==
Sık idrara çıkma ve aşırı susama en yaygın belirtilerdir.

== Tedavi ==
Tedavi edilmediğinde ketoasidoz gelişebilir, bu da hastanede bakım gerektirir.
"""

QUESTION = {
    "qid": "t-1",
    "doc_id": "diyabet.md",
    "doc_title": "diyabet",
    "question": "Şeker hastalığında aşırı susama görülür mü?",
    "answer_span": "Sık idrara çıkma ve aşırı susama en yaygın belirtilerdir.",
}


def _run(tmp_path, capsys, *argv):
    corpus = tmp_path / "belgeler"
    corpus.mkdir(exist_ok=True)
    (corpus / "diyabet.md").write_text(ARTICLE, encoding="utf-8")
    gold = tmp_path / "sorular.json"
    gold.write_text(json.dumps([QUESTION], ensure_ascii=False), encoding="utf-8")
    parser = argparse.ArgumentParser()
    ask.add_arguments(parser)
    args = parser.parse_args([*argv, "--corpus", str(corpus), "--gold", str(gold),
                              "--sparse-only", "--chunking", "hierarchical"])
    assert ask.run(args) == 0
    return capsys.readouterr().out


def _rows(out: str, retriever: str) -> list[str]:
    lines = out.splitlines()
    start = lines.index(retriever) + 1
    rows = []
    for line in lines[start:]:
        if not line.startswith("  "):
            break
        rows.append(line)
    return rows


@pytest.mark.parametrize("query", ["t-1", QUESTION["question"]])
def test_a_gold_question_marks_the_chunk_holding_its_answer(tmp_path, capsys, query):
    out = _run(tmp_path, capsys, query)
    for retriever in ("bm25_stem5", "bm25_nostem"):
        rows = _rows(out, retriever)
        assert len(rows) == 3
        marked = [r for r in rows if "✓" in r]
        assert len(marked) == 1 and "aşırı susama" in marked[0]
        assert all("·" in r for r in rows if r not in marked)
    assert "found by: bm25_stem5@1, bm25_nostem@1" in out


def test_free_text_gets_answers_but_no_marks(tmp_path, capsys):
    out = _run(tmp_path, capsys, "ketoasidoz nedir")
    assert "not in the gold set" in out
    assert "✓" not in out and "·" not in out
    assert "ketoasidoz" in _rows(out, "bm25_stem5")[0]


def test_k_sets_how_many_chunks_each_retriever_shows(tmp_path, capsys):
    out = _run(tmp_path, capsys, "t-1", "-k", "1")
    assert len(_rows(out, "bm25_stem5")) == 1
