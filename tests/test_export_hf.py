import json

import pytest

from turkish_rag_eval.export_hf import build, write

DOCS = [
    {"doc_id": "1", "title": "Diyabet", "url": "https://tr.wikipedia.org/?curid=1",
     "text": "Diyabet metni."},
    {"doc_id": "2", "title": "Astım", "url": "https://tr.wikipedia.org/?curid=2",
     "text": "Astım metni."},
]
GOLD = [{"qid": "q001", "question": "Şeker hastalığı nedir?", "doc_id": "1",
         "doc_title": "Diyabet", "answer_span": "Diyabet metni"}]


def test_rows_follow_the_beir_layout():
    corpus, queries, qrels = build(GOLD, DOCS)
    assert corpus[0] == {"_id": "1", "title": "Diyabet", "text": "Diyabet metni.",
                         "url": "https://tr.wikipedia.org/?curid=1"}
    assert queries == [{"_id": "q001", "text": "Şeker hastalığı nedir?",
                        "answer_span": "Diyabet metni"}]
    assert qrels == [{"query-id": "q001", "corpus-id": "1", "score": 1}]


def test_articles_no_question_points_at_stay_as_distractors():
    corpus, _, _ = build(GOLD, DOCS)
    assert [d["_id"] for d in corpus] == ["1", "2"]


def test_question_whose_article_is_missing_from_the_corpus_fails():
    item = dict(GOLD[0], qid="q002", doc_id="99")
    with pytest.raises(ValueError, match="q002 -> 99"):
        build([item], DOCS)


def test_files_are_utf8_jsonl(tmp_path):
    write(tmp_path, *build(GOLD, DOCS))
    line = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert "Şeker" in line and json.loads(line)["_id"] == "q001"
    assert len((tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "qrels" / "test.jsonl").exists()
