"""Offline checks for the legislation domain (data/eval/mevzuat/).

The Wikipedia validator fetches live articles, which does not work for
legislation, so these questions are checked against their own pinned corpus
(data/raw/mevzuat/corpus.json) instead: every span is copied verbatim from
exactly one article, every question is a paraphrase, and the questions do not
name the regulation or the article number, which would hand BM25 the answer.
"""

import json
import re

from turkish_rag_eval.corpus import compare_to_lock, load_corpus
from turkish_rag_eval.gold import load_gold, normalise
from turkish_rag_eval.paths import ROOT
from turkish_rag_eval.turkish_text import turkish_lower
from turkish_rag_eval.validate_gold import MAX_OVERLAP, overlap

QUESTIONS = ROOT / "data" / "eval" / "mevzuat" / "llm-draft-mevzuat.json"
CORPUS = ROOT / "data" / "raw" / "mevzuat" / "corpus.json"
LOCK = ROOT / "data" / "raw" / "mevzuat" / "corpus.lock.json"

ITEMS = json.loads(QUESTIONS.read_text(encoding="utf-8"))
DOCS = load_corpus(CORPUS).docs


def test_corpus_matches_its_lockfile():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert compare_to_lock(lock, DOCS) == []


def test_every_span_is_verbatim_in_exactly_its_own_article():
    texts = {d["doc_id"]: normalise(d["text"]) for d in DOCS}
    for item in ITEMS:
        span = normalise(item["answer_span"])
        holders = [doc_id for doc_id, text in texts.items() if span in text]
        assert holders == [item["doc_id"]], item["qid"]


def test_questions_are_paraphrases():
    for item in ITEMS:
        assert overlap(item["question"], item["answer_span"]) <= MAX_OVERLAP, item["qid"]
        assert item["question"].rstrip().endswith("?"), item["qid"]


def test_questions_do_not_name_the_regulation_or_the_article():
    for item in ITEMS:
        q = turkish_lower(item["question"])
        assert not re.search(r"\d", q), item["qid"]
        assert not re.search(r"\bmadde|yönetmeli|kanun", q), item["qid"]


def test_ids_and_questions_are_unique():
    assert len({i["qid"] for i in ITEMS}) == len(ITEMS)
    assert len({normalise(turkish_lower(i["question"])) for i in ITEMS}) == len(ITEMS)


def test_items_stay_drafts_outside_the_published_numbers():
    assert all(i["source"] == "llm-draft" for i in ITEMS)
    assert load_gold([QUESTIONS]) == []
    assert len(load_gold([QUESTIONS], include_drafts=True)) == len(ITEMS)


def test_every_question_points_at_a_corpus_article():
    ids = {d["doc_id"] for d in DOCS}
    titles = {d["doc_id"]: d["title"] for d in DOCS}
    for item in ITEMS:
        assert item["doc_id"] in ids, item["qid"]
        assert item["doc_title"] == titles[item["doc_id"]], item["qid"]


def test_each_regulation_has_distractors_beyond_the_answer_articles():
    answered = {i["doc_id"] for i in ITEMS}
    by_regulation = {}
    for d in json.loads(CORPUS.read_text(encoding="utf-8")):
        by_regulation.setdefault(d["regulation"], []).append(d["doc_id"] in answered)
    assert len(by_regulation) == 4
    for flags in by_regulation.values():
        assert flags.count(False) > 5 * flags.count(True)
