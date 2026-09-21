"""Loading and identity: the two things every published number depends on.

The fingerprint is what lets a result file say which corpus produced it, so
the properties pinned here (order independence, text sensitivity, title
insensitivity) are load-bearing rather than incidental.
"""

import json

import pytest

from turkish_rag_eval.corpus import (
    Corpus,
    CorpusError,
    build_lock,
    compare_to_lock,
    fingerprint,
    load_corpus,
)

DOCS = [
    {"doc_id": "2", "title": "Astım", "text": "b" * 500, "url": ""},
    {"doc_id": "1", "title": "Diyabet", "text": "a" * 500, "url": ""},
]


def test_fingerprint_ignores_document_order():
    assert fingerprint(DOCS) == fingerprint(list(reversed(DOCS)))


def test_fingerprint_follows_the_text():
    changed = [dict(DOCS[0], text="b" * 501), DOCS[1]]
    assert fingerprint(changed) != fingerprint(DOCS)


def test_fingerprint_ignores_metadata_that_cannot_move_a_score():
    # Retrieval reads the text. A retitled article with identical body is the
    # same corpus as far as any metric is concerned.
    retitled = [dict(DOCS[0], title="Bronşiyal astım"), DOCS[1]]
    assert fingerprint(retitled) == fingerprint(DOCS)


def test_json_corpus_round_trips(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(DOCS), encoding="utf-8")
    corpus = load_corpus(path)
    assert len(corpus) == 2
    assert corpus.fingerprint == fingerprint(DOCS)


def test_documents_are_sorted_by_id(tmp_path):
    # Chunk order decides every tie-break downstream, so it must not inherit
    # whatever order the file happened to use.
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(DOCS), encoding="utf-8")
    assert [d["doc_id"] for d in load_corpus(path).docs] == ["1", "2"]


def test_beir_directory_is_read_through_corpus_jsonl(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": d["doc_id"], "title": d["title"],
                              "text": d["text"]}) for d in DOCS),
        encoding="utf-8")
    corpus = load_corpus(tmp_path)
    assert corpus.fingerprint == fingerprint(DOCS)


def test_directory_of_text_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "bir.md").write_text("a" * 500, encoding="utf-8")
    (tmp_path / "nested" / "iki.txt").write_text("b" * 500, encoding="utf-8")
    (tmp_path / "kisa.txt").write_text("cok kisa", encoding="utf-8")

    corpus = load_corpus(tmp_path)
    assert [d["doc_id"] for d in corpus.docs] == ["bir.md", "nested/iki.txt"]


def test_same_filename_in_two_folders_stays_two_documents(tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "notlar.md").write_text(folder * 300, encoding="utf-8")
    assert len(load_corpus(tmp_path)) == 2


def test_missing_path_names_the_accepted_shapes(tmp_path):
    with pytest.raises(CorpusError, match="corpus.jsonl"):
        load_corpus(tmp_path / "yok")


def test_document_without_text_is_rejected(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([{"doc_id": "1"}]), encoding="utf-8")
    with pytest.raises(CorpusError, match="doc_id or _id"):
        load_corpus(path)


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(CorpusError, match="no .txt or .md file"):
        load_corpus(tmp_path)


def test_lock_matches_its_own_corpus():
    assert compare_to_lock(build_lock(DOCS), DOCS) == []


def test_lock_reports_changed_added_and_missing():
    lock = build_lock(DOCS, revisions={"1": 12345})
    assert lock["documents"][0]["revid"] == 12345

    drifted = [
        dict(DOCS[1], text="a" * 700),                      # changed
        {"doc_id": "3", "title": "Yeni", "text": "c" * 500},  # added
    ]                                                        # doc 2 missing
    drift = compare_to_lock(lock, drifted)
    assert any("changed: Diyabet" in d and "+200" in d for d in drift)
    assert any("added: Yeni" in d for d in drift)
    assert any("missing: Astım" in d for d in drift)


def test_lock_from_a_future_version_asks_to_be_regenerated():
    lock = dict(build_lock(DOCS), version=99)
    assert "regenerate" in compare_to_lock(lock, DOCS)[0]


def test_corpus_reports_its_size():
    corpus = Corpus(docs=DOCS, source="test")
    assert corpus.chars == 1000
