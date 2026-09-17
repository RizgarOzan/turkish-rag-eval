from agreement import chunk_kappa, cohens_kappa, evaluate_pairs, find_pairs, span_f1, summarise

ORIGINAL = {
    "qid": "q001",
    "question": "Şeker hastalığı teşhisi konan kişilerin ne kadarında ketoasidoz da bulunuyor?",
    "doc_id": "710430",
    "doc_title": "Diyabet",
    "answer_span": "yaklaşık %25'i, diyabet teşhisi konulduğunda diyabetik ketoasidoz da gelişmiştir",
}
SECOND = {
    "qid": "ayse-q001",
    "question": ORIGINAL["question"],
    "doc_id": "710430",
    "doc_title": "Diyabet",
    "answer_span": "diyabetik ketoasidoz da gelişmiştir",
    "annotator": "ayse",
    "second_of": "q001",
}
DOC = {
    "doc_id": "710430",
    "title": "Diyabet",
    "url": "https://tr.wikipedia.org/?curid=710430",
    "text": ("Giriş. " + "kelime " * 200 + ORIGINAL["answer_span"]
              + " devamı. " + "kelime " * 200),
}


def test_find_pairs_links_via_second_of():
    pairs = find_pairs([ORIGINAL, SECOND])
    assert [(a["qid"], b["qid"]) for a, b in pairs] == [("q001", "ayse-q001")]


def test_find_pairs_falls_back_to_matching_question_text():
    unlinked = dict(SECOND, second_of=None, qid="ayse-q001b")
    unlinked.pop("second_of")
    pairs = find_pairs([ORIGINAL, unlinked])
    assert [(a["qid"], b["qid"]) for a, b in pairs] == [("q001", "ayse-q001b")]


def test_find_pairs_ignores_same_annotator_duplicates():
    # No second_of link: the fallback question-text match must not pair two
    # items that share an annotator (or both lack one) - that is not an
    # independent second opinion, just the same text appearing twice.
    unlinked = dict(SECOND, qid="ayse-q001b")
    unlinked.pop("second_of")
    unlinked.pop("annotator")
    pairs = find_pairs([dict(ORIGINAL), unlinked])
    assert pairs == []


def test_find_pairs_does_not_double_count():
    pairs = find_pairs([ORIGINAL, SECOND])
    assert len(pairs) == 1


def test_span_f1_identical_spans_is_one():
    assert span_f1(ORIGINAL["answer_span"], ORIGINAL["answer_span"]) == 1.0


def test_span_f1_disjoint_spans_is_zero():
    assert span_f1(ORIGINAL["answer_span"], "tamamen alakasız bir cümle") == 0.0


def test_span_f1_partial_overlap_is_between_zero_and_one():
    f1 = span_f1(ORIGINAL["answer_span"], SECOND["answer_span"])
    assert 0.0 < f1 < 1.0


def test_cohens_kappa_perfect_agreement():
    labels = [True, False, False, True, False]
    assert cohens_kappa(labels, labels) == 1.0


def test_cohens_kappa_empty_is_none():
    assert cohens_kappa([], []) is None


def test_cohens_kappa_corrects_for_chance_on_imbalanced_labels():
    # Both mostly say "not relevant"; raw agreement is high but kappa should
    # not reward the two agreeing every time chance alone predicts it.
    a = [False] * 18 + [True, True]
    b = [False] * 18 + [False, True]
    raw_agreement = sum(x == y for x, y in zip(a, b)) / len(a)
    kappa = cohens_kappa(a, b)
    assert kappa < raw_agreement


def test_chunk_kappa_returns_a_score_per_strategy():
    result = chunk_kappa(ORIGINAL, SECOND, DOC)
    assert set(result) == {"fixed", "sentence", "hierarchical"}
    assert all(v is None or -1.0 <= v <= 1.0 for v in result.values())


def test_evaluate_pairs_and_summarise():
    pairs = find_pairs([ORIGINAL, SECOND])
    rows = evaluate_pairs(pairs, {"710430": DOC})
    assert len(rows) == 1
    row = rows[0]
    assert row["qid"] == "q001" and row["second_qid"] == "ayse-q001"
    assert row["annotator_a"] == "gold" and row["annotator_b"] == "ayse"
    assert 0.0 < row["token_f1"] <= 1.0
    assert set(row["kappa"]) == {"fixed", "sentence", "hierarchical"}

    summary = summarise(rows)
    assert summary["pairs"] == 1
    assert summary["mean_token_f1"] == row["token_f1"]


def test_evaluate_pairs_without_corpus_skips_kappa():
    pairs = find_pairs([ORIGINAL, SECOND])
    rows = evaluate_pairs(pairs, {})
    assert rows[0]["kappa"] == {}
    summary = summarise(rows)
    assert summary["mean_kappa_fixed"] == 0.0
