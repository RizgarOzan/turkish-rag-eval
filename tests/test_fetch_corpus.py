"""The lock half of fetch_corpus, which is the half that runs without network.

Parsing is checked against a recorded API shape rather than a live request:
the point is that lastrevid survives into the document, so the lockfile can
name a revision.
"""

import json

from turkish_rag_eval import fetch_corpus

PAGES = {
    "query": {
        "pages": {
            "1234": {
                "pageid": 1234,
                "title": "Diyabet",
                "lastrevid": 987654,
                "extract": "Diyabet bir metabolizma hastalığıdır. " * 40,
            },
            "5678": {
                "pageid": 5678,
                "title": "Çok kısa",
                "lastrevid": 111,
                "extract": "kısa",  # below MIN_EXTRACT_CHARS
            },
        }
    }
}


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_extract_parsing_keeps_the_revision_id(monkeypatch):
    monkeypatch.setattr(fetch_corpus.requests, "get",
                        lambda *a, **k: _Response(PAGES))
    docs = fetch_corpus.fetch_batch(["Diyabet"])
    assert len(docs) == 1, "the too-short extract should be dropped"
    assert docs[0]["revid"] == 987654
    assert docs[0]["doc_id"] == "1234"


def _docs(text="a" * 500):
    return [{"doc_id": "1", "title": "Diyabet", "text": text, "revid": 987654}]


def test_first_run_writes_the_lock(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch_corpus, "LOCK", tmp_path / "corpus.lock.json")
    assert fetch_corpus.report_drift(_docs(), update=False) == 0

    lock = json.loads((tmp_path / "corpus.lock.json").read_text(encoding="utf-8"))
    assert lock["documents"][0]["revid"] == 987654
    assert "kilit yazildi" in capsys.readouterr().out


def test_matching_corpus_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_corpus, "LOCK", tmp_path / "corpus.lock.json")
    fetch_corpus.report_drift(_docs(), update=False)
    assert fetch_corpus.report_drift(_docs(), update=False) == 0


def test_drifted_corpus_fails_and_names_the_article(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch_corpus, "LOCK", tmp_path / "corpus.lock.json")
    fetch_corpus.report_drift(_docs(), update=False)
    capsys.readouterr()

    # Wikipedia edited the article underneath us.
    assert fetch_corpus.report_drift(_docs("a" * 640), update=False) == 1
    out = capsys.readouterr().out
    assert "changed: Diyabet" in out and "+140" in out
    assert "--update-lock" in out


def test_update_lock_accepts_the_new_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_corpus, "LOCK", tmp_path / "corpus.lock.json")
    fetch_corpus.report_drift(_docs(), update=False)
    assert fetch_corpus.report_drift(_docs("a" * 640), update=True) == 0
    assert fetch_corpus.report_drift(_docs("a" * 640), update=False) == 0
