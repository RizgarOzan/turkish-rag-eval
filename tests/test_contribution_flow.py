"""CONTRIBUTING.md's path, end to end: drop a question file into
``data/eval/contrib/``, ``validate --online`` it, and have the harness score it.

The pieces are tested alone elsewhere; this pins that they agree about the
same file. Wikipedia is replaced by a stub article.
"""

import argparse
import json

import pytest

from turkish_rag_eval import gold, run_eval, validate_gold

pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")

SPAN = "Tedavi edilmediğinde diyabetik ketoasidoz gelişebilir ve hayati tehlike yaratır."
ARTICLE = {
    "title": "Diyabet",
    "text": ("Diyabet, kan şekerinin uzun süre yüksek seyrettiği bir hastalıktır. " * 12
             + SPAN + "\n\nSık idrara çıkma ve aşırı susama en yaygın belirtilerdir. " * 6),
}
QUESTION = {
    "qid": "yourname-001",
    "question": "Şeker hastalığı tedavi görmezse hangi akut komplikasyon ortaya çıkar?",
    "doc_id": "710430",
    "doc_title": "Diyabet",
    "answer_span": SPAN,
    "annotator": "yourname",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tiny stand-in for the repo layout, with the loaders pointed at it."""
    eval_dir = tmp_path / "data" / "eval"
    contrib = eval_dir / "contrib"
    contrib.mkdir(parents=True)
    (eval_dir / "gold.json").write_text("[]", encoding="utf-8")
    for module in (gold, validate_gold):
        monkeypatch.setattr(module, "GOLD", eval_dir / "gold.json")
    monkeypatch.setattr(gold, "CONTRIB", contrib)

    online = validate_gold.check_online
    monkeypatch.setattr(validate_gold, "check_online",
                        lambda items: online(items, fetch=lambda _: ARTICLE, delay=0))

    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"doc_id": "710430", **ARTICLE}], ensure_ascii=False),
                      encoding="utf-8")
    return contrib, corpus


def validate(*online):
    parser = argparse.ArgumentParser()
    validate_gold.add_arguments(parser)
    return validate_gold.run(parser.parse_args(["--online", *map(str, online)]))


def score(corpus, out):
    parser = argparse.ArgumentParser()
    run_eval.add_arguments(parser)
    args = parser.parse_args([])
    args.corpus, args.out, args.retrievers = corpus, out, ["bm25_stem5"]
    assert run_eval.run(args) == 0
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_a_contributed_file_validates_and_is_scored(repo, tmp_path, capsys):
    contrib, corpus = repo
    path = contrib / "yourname-diyabet.json"
    path.write_text(json.dumps([QUESTION], ensure_ascii=False, indent=1), encoding="utf-8")

    assert validate(path) == 0
    assert "2 dosya, 1 soru, 0 hata" in capsys.readouterr().out

    summary = score(corpus, tmp_path / "out")
    assert {row["queries"] for row in summary} == {1}
    perquery = json.loads((tmp_path / "out" / "perquery_hierarchical_bm25_stem5.json")
                          .read_text(encoding="utf-8"))
    assert perquery[0]["qid"] == "yourname-001" and perquery[0]["found"]


def test_a_copied_question_fails_validation_before_it_can_be_scored(repo, capsys):
    contrib, _ = repo
    path = contrib / "yourname-diyabet.json"
    copied = dict(QUESTION, question="Tedavi edilmediğinde diyabetik ketoasidoz gelişebilir mi?")
    path.write_text(json.dumps([copied], ensure_ascii=False), encoding="utf-8")

    assert validate(path) == 1
    assert "paraphrase" in capsys.readouterr().out


def test_a_span_that_is_not_in_the_article_fails_online(repo, capsys):
    contrib, _ = repo
    path = contrib / "yourname-diyabet.json"
    wrong = dict(QUESTION, answer_span="Bu cümle makalede hiç geçmiyor ama etiketlenmiş.")
    path.write_text(json.dumps([wrong], ensure_ascii=False), encoding="utf-8")

    assert validate(path) == 1
    assert "not found verbatim" in capsys.readouterr().out
