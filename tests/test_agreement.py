from turkish_rag_eval.agreement import (
    chunk_kappa,
    cohens_kappa,
    evaluate_groups,
    group_by_question,
    pairwise_items,
    root_qid,
    span_f1,
    span_iou,
    summarise,
)

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
THIRD = {
    "qid": "mehmet-q001",
    "question": ORIGINAL["question"],
    "doc_id": "710430",
    "doc_title": "Diyabet",
    "answer_span": "diyabet teşhisi konulduğunda diyabetik ketoasidoz",
    "annotator": "mehmet",
    "second_of": "q001",
}
DOC = {
    "doc_id": "710430",
    "title": "Diyabet",
    "url": "https://tr.wikipedia.org/?curid=710430",
    "text": ("Giriş. " + "kelime " * 200 + ORIGINAL["answer_span"]
              + " devamı. " + "kelime " * 200),
}


def test_group_by_question_groups_all_annotators():
    groups = group_by_question([ORIGINAL, SECOND, THIRD])
    assert len(groups) == 1
    assert {i["qid"] for i in groups[0]} == {"q001", "ayse-q001", "mehmet-q001"}


def test_group_by_question_drops_unanswered_questions():
    lonely = {**ORIGINAL, "qid": "q999", "question": "Alakasız tek bir soru mu bu?"}
    groups = group_by_question([ORIGINAL, SECOND, lonely])
    assert len(groups) == 1


def test_root_qid_prefers_the_item_without_second_of():
    assert root_qid([SECOND, ORIGINAL, THIRD]) == "q001"


def test_pairwise_items_ignores_same_annotator_pairs():
    same_annotator = dict(SECOND, qid="ayse-q001b")
    pairs = pairwise_items([SECOND, same_annotator])
    assert pairs == []


def test_pairwise_items_covers_every_combination_for_three_annotators():
    pairs = pairwise_items([ORIGINAL, SECOND, THIRD])
    qid_pairs = {frozenset((a["qid"], b["qid"])) for a, b in pairs}
    assert qid_pairs == {
        frozenset(("q001", "ayse-q001")),
        frozenset(("q001", "mehmet-q001")),
        frozenset(("ayse-q001", "mehmet-q001")),
    }


def test_span_iou_identical_spans_is_one():
    assert span_iou(ORIGINAL["answer_span"], ORIGINAL["answer_span"]) == 1.0


def test_span_iou_disjoint_spans_is_zero():
    assert span_iou(ORIGINAL["answer_span"], "tamamen alakasız bir cümle") == 0.0


def test_span_iou_penalises_extra_material_more_than_f1_does():
    # B is (close to) a subset of A: same fact, A carries extra clause.
    a = "yaklaşık %25'i, diyabet teşhisi konulduğunda diyabetik ketoasidoz da gelişmiştir"
    b = "diyabet teşhisi konulduğunda diyabetik ketoasidoz da gelişmiştir"
    iou, f1 = span_iou(a, b), span_f1(a, b)
    assert 0.0 < iou < f1 < 1.0


def test_span_f1_identical_spans_is_one():
    assert span_f1(ORIGINAL["answer_span"], ORIGINAL["answer_span"]) == 1.0


def test_cohens_kappa_perfect_agreement():
    labels = [True, False, False, True, False]
    assert cohens_kappa(labels, labels) == 1.0


def test_cohens_kappa_empty_is_none():
    assert cohens_kappa([], []) is None


def test_cohens_kappa_corrects_for_chance_on_imbalanced_labels():
    a = [False] * 18 + [True, True]
    b = [False] * 18 + [False, True]
    raw_agreement = sum(x == y for x, y in zip(a, b)) / len(a)
    assert cohens_kappa(a, b) < raw_agreement


def test_chunk_kappa_returns_a_score_per_strategy():
    result = chunk_kappa(ORIGINAL, SECOND, DOC)
    assert set(result) == {"fixed", "sentence", "hierarchical"}
    assert all(v is None or -1.0 <= v <= 1.0 for v in result.values())


def test_evaluate_groups_averages_every_pair_not_just_the_first():
    groups = group_by_question([ORIGINAL, SECOND, THIRD])
    rows = evaluate_groups(groups, {"710430": DOC})
    assert len(rows) == 1
    row = rows[0]
    assert row["qid"] == "q001"
    assert row["annotators"] == ["ayse", "gold", "mehmet"]
    assert row["pairs"] == 3

    expected_iou = sum(p["iou"] for p in row["pair_details"]) / 3
    assert row["mean_iou"] == expected_iou
    assert set(row["mean_kappa"]) == {"fixed", "sentence", "hierarchical"}


def test_evaluate_groups_without_corpus_skips_kappa():
    groups = group_by_question([ORIGINAL, SECOND])
    rows = evaluate_groups(groups, {})
    assert rows[0]["pair_details"][0]["kappa"] == {}
    assert all(v is None for v in rows[0]["mean_kappa"].values())


def test_summarise_weighs_each_question_equally():
    # q001 has 3 annotators (3 pairs); q002 has 2 (1 pair). Both questions
    # must count once in the headline mean, not once per pair.
    other_original = {**ORIGINAL, "qid": "q002", "question": "İkinci bir soru mu bu?"}
    other_second = {**SECOND, "qid": "ayse-q002", "question": other_original["question"],
                    "second_of": "q002", "answer_span": "tamamen farklı bir ifade"}
    groups = group_by_question([ORIGINAL, SECOND, THIRD, other_original, other_second])
    rows = evaluate_groups(groups, {})
    summary = summarise(rows)
    assert summary["questions"] == 2
    assert summary["pairs"] == 4
    assert summary["mean_iou"] == sum(r["mean_iou"] for r in rows) / 2
