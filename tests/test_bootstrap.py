"""What the bootstrapper writes has to be at least as trustworthy as a
hand-written gold file, so the checks that make it so are what is tested here.

The drafter is a stub. The interesting behaviour is not the model call, it is
what happens to a draft that quotes the document wrong, copies the question
out of the span, or gets a different span from the second pass.
"""

import pytest

from turkish_rag_eval.bootstrap import (
    AGREEMENT_THRESHOLD,
    bootstrap,
    build_items,
    check_draft,
    passes_agree,
    slugify,
)

pytest.importorskip("numpy")

DOCUMENT = {
    "doc_id": "diyabet.md",
    "title": "Diyabet",
    "text": ("Diyabet, kan şekerinin uzun süre yüksek seyrettiği bir "
             "metabolizma hastalığıdır. Tedavi edilmediğinde ketoasidoz "
             "gelişebilir ve bu durum acil müdahale gerektirir."),
}
SPAN = "Tedavi edilmediğinde ketoasidoz gelişebilir"
QUESTION = "Şeker hastalığı ihmal edilirse hangi acil tablo ortaya çıkar?"


class Drafter:
    def __init__(self, drafts, second):
        self.drafts, self.second = drafts, second

    def draft(self, document, count):
        return self.drafts[:count]

    def mark(self, document, questions):
        return self.second


def test_a_good_draft_survives():
    assert check_draft(QUESTION, SPAN, DOCUMENT["text"]) is None


def test_a_span_that_is_not_in_the_document_is_rejected():
    # The most common model failure: paraphrasing instead of quoting. Such a
    # span can never mark a retrieved chunk relevant.
    problem = check_draft(QUESTION, "Ketoasidoz riski vardır.", DOCUMENT["text"])
    assert problem == "span is not in the document verbatim"


def test_a_question_copied_from_the_span_is_rejected():
    copied = "Tedavi edilmediğinde ketoasidoz gelişebilir mi?"
    assert "copied from the span" in check_draft(copied, SPAN, DOCUMENT["text"])


def test_a_question_without_a_question_mark_is_rejected():
    assert check_draft("Bu bir soru değil", SPAN, DOCUMENT["text"]) \
        == "question does not end with '?'"


def test_spans_that_are_too_short_or_too_long_are_rejected():
    assert "shorter than" in check_draft(QUESTION, "kısa", DOCUMENT["text"])
    assert "longer than" in check_draft(QUESTION, "x" * 500, DOCUMENT["text"])
    assert check_draft(QUESTION, "", DOCUMENT["text"]) == "empty answer span"


def test_agreeing_passes_produce_an_agreed_item():
    items, dropped = build_items(
        DOCUMENT, [{"question": QUESTION, "answer_span": SPAN}],
        [SPAN + " ve bu durum acil müdahale gerektirir"], prefix="draft")

    assert dropped == []
    assert items[0]["review"] == "agreed"
    assert items[0]["second_annotation"]["annotator"] == "llm-2"
    assert items[0]["source"] == "llm-draft"


def test_disagreeing_passes_are_held_for_a_human():
    # The second pass marked a different sentence: neither span contains the
    # other, so the two passes really do point somewhere different.
    items, _ = build_items(
        DOCUMENT, [{"question": QUESTION, "answer_span": SPAN}],
        ["Diyabet, kan şekerinin uzun süre yüksek seyrettiği bir "
         "metabolizma hastalığıdır"], prefix="draft")
    assert items[0]["review"] == "needs-human"


def test_a_second_span_that_is_not_verbatim_counts_as_disagreement():
    # It says nothing about agreement, so it must not be recorded as agreement.
    items, _ = build_items(
        DOCUMENT, [{"question": QUESTION, "answer_span": SPAN}],
        ["uydurulmuş bir alıntı"], prefix="draft")
    assert items[0]["review"] == "needs-human"
    assert "second_annotation" not in items[0]


def test_a_missing_second_span_counts_as_disagreement():
    items, _ = build_items(
        DOCUMENT, [{"question": QUESTION, "answer_span": SPAN}], [""],
        prefix="draft")
    assert items[0]["review"] == "needs-human"


def test_rejected_drafts_are_reported_with_a_reason():
    items, dropped = build_items(
        DOCUMENT,
        [{"question": QUESTION, "answer_span": "belgede olmayan bir cümle"}],
        ["x"], prefix="draft")
    assert items == []
    assert "not in the document verbatim" in dropped[0]


def test_qids_are_unique_and_skip_dropped_drafts():
    drafts = [
        {"question": QUESTION, "answer_span": SPAN},
        {"question": "Geçersiz?", "answer_span": "belgede yok"},
        {"question": "Bu rahatsızlık hangi vücut sürecini ilgilendirir?",
         "answer_span": "kan şekerinin uzun süre yüksek seyrettiği"},
    ]
    items, dropped = build_items(DOCUMENT, drafts, [SPAN, "", ""],
                                 prefix="draft")
    qids = [i["qid"] for i in items]
    assert len(dropped) == 1
    assert qids == ["draft-diyabet-001", "draft-diyabet-002"]


def test_a_short_second_pass_does_not_shift_spans_onto_wrong_questions():
    # If the model returns fewer spans than questions, zipping would pair
    # question 2's span with question 1 and so on, silently.
    drafts = [{"question": QUESTION, "answer_span": SPAN}]
    items, _ = build_items(DOCUMENT, drafts, [], prefix="draft")
    assert items == [], "no second span means nothing to pair, not a mispair"


def test_bootstrap_runs_over_every_document():
    items, _ = bootstrap(
        [DOCUMENT],
        Drafter([{"question": QUESTION, "answer_span": SPAN}], [SPAN]),
        per_document=1, prefix="draft")
    assert len(items) == 1
    assert items[0]["doc_id"] == "diyabet.md"


def test_slugify_folds_turkish_letters():
    assert slugify("Çölyak Hastalığı") == "colyak-hastaligi"
    assert slugify("İstanbul") == "istanbul"
    assert slugify("!!!") == "doc"


def test_agreement_threshold_is_a_ratio():
    assert 0 < AGREEMENT_THRESHOLD <= 1


def test_containment_counts_as_agreement():
    # The harness judges a chunk relevant when it CONTAINS the span, so a
    # longer span that swallows a shorter one marks the same chunks relevant.
    short = "ketoasidoz gelişebilir"
    long = "Tedavi edilmediğinde ketoasidoz gelişebilir ve bu durum acil"
    assert passes_agree(short, long)
    assert passes_agree(long, short)


def test_containment_holds_where_token_overlap_would_not():
    # This is the calibration that set the rule: all 60 double-labelled
    # questions already in data/eval/contrib are containment pairs, and 29 of
    # them fall below the Jaccard threshold. Overlap alone would send people
    # to arbitrate an already-reviewed gold set.
    from turkish_rag_eval.agreement import span_iou

    short = "Phobos ve Deimos"
    long = ("Mars'ın 1877 yılında astronom Asaph Hall tarafından keşfedilen "
            "Phobos ve Deimos adları verilmiş, düzensiz biçimli iki küçük "
            "uydusu vardır")
    assert span_iou(short, long) < AGREEMENT_THRESHOLD
    assert passes_agree(short, long)


def test_spans_pointing_elsewhere_do_not_agree():
    assert not passes_agree("Phobos ve Deimos uyduları",
                            "Mars yüzeyindeki demir oksit rengi verir")


def test_an_empty_span_never_agrees():
    assert not passes_agree("", "herhangi bir metin")
    assert not passes_agree("herhangi bir metin", "")
