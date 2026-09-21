"""The scoring split is the whole idea, so it is what gets tested.

No network: the generator and judge are arguments, and these stubs stand in
for them. What matters is that a query whose evidence was never retrieved is
never scored as if it had been, and that answering without evidence is counted
as a hallucination rather than quietly averaged away.
"""

import pytest

from turkish_rag_eval.groundedness import (
    ABSTAIN,
    Record,
    evaluate,
    format_passages,
    report,
    summarise,
)

pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")


def chunk(doc_id, body, heading="Başlık"):
    return {"doc_id": doc_id, "chunk_id": f"{doc_id}::0", "title": doc_id,
            "url": "", "heading_path": heading, "body": body,
            "embed_text": body}


SPAN = "Tedavi edilmediğinde ketoasidoz gelişebilir."
CHUNKS = [
    chunk("diyabet", f"Diyabet bir metabolizma hastalığıdır. {SPAN}"),
    chunk("astim", "Astım hava yollarının iltihaplanmasıdır."),
]
GOLD = [{"qid": "g-1", "doc_id": "diyabet", "answer_span": SPAN,
         "question": "Diyabette hangi acil tablo gelişebilir?"}]


class Retriever:
    """Returns the chunk indices it was told to, in order."""

    def __init__(self, indices):
        self.indices = indices

    def search(self, query, k):
        return [(i, 1.0) for i in self.indices[:k]]


def judge_saying(supported=True, answers=True):
    return lambda *_args: {"supported": supported, "answers_question": answers,
                           "reason": "stub"}


def never_judged(*_args):
    raise AssertionError("this query should not have reached the judge")


def test_answer_with_evidence_is_judged():
    summary, records = evaluate(
        GOLD, CHUNKS, Retriever([0]), lambda q, c: "Ketoasidoz gelişebilir.",
        judge_saying(), k=5)

    assert records[0].span_retrieved
    assert summary["grounded_when_answered"] == 1.0
    assert summary["correct_when_answered"] == 1.0
    assert summary["hallucination_rate"] is None, "no missed queries to rate"


def test_answering_without_evidence_is_a_hallucination():
    # Only the astım chunk comes back, so nothing in the context answers the
    # diabetes question - yet the generator answers anyway.
    summary, records = evaluate(
        GOLD, CHUNKS, Retriever([1]), lambda q, c: "Ketoasidoz gelişebilir.",
        never_judged, k=5)

    assert not records[0].span_retrieved
    assert records[0].hallucinated
    assert summary["hallucination_rate"] == 1.0
    assert summary["abstained_when_not_retrieved"] == 0.0


def test_declining_without_evidence_is_the_right_answer():
    summary, records = evaluate(
        GOLD, CHUNKS, Retriever([1]), lambda q, c: ABSTAIN, never_judged, k=5)

    assert records[0].abstained and not records[0].hallucinated
    assert summary["abstained_when_not_retrieved"] == 1.0
    assert summary["hallucination_rate"] == 0.0


def test_an_abstention_is_never_sent_to_the_judge():
    # There is no claim to support, so judging one would only cost money.
    evaluate(GOLD, CHUNKS, Retriever([0]), lambda q, c: ABSTAIN, never_judged,
             k=5)


def test_the_two_populations_are_reported_separately():
    # One query answered correctly with evidence, one hallucinated without.
    gold = GOLD + [{"qid": "g-2", "doc_id": "yok", "answer_span": "yok",
                    "question": "Başka bir soru?"}]

    class Mixed:
        def __init__(self):
            self.calls = 0

        def search(self, query, k):
            self.calls += 1
            return [(0, 1.0)] if self.calls == 1 else [(1, 1.0)]

    summary, _ = evaluate(gold, CHUNKS, Mixed(), lambda q, c: "Bir cevap.",
                          judge_saying(), k=5)

    assert summary["span_retrieved"] == 1
    assert summary["span_not_retrieved"] == 1
    # Pooling these would report 50% and hide both facts.
    assert summary["grounded_when_answered"] == 1.0
    assert summary["hallucination_rate"] == 1.0


def test_an_unsupported_answer_is_marked_even_when_evidence_was_present():
    summary, records = evaluate(
        GOLD, CHUNKS, Retriever([0]), lambda q, c: "Diyabet bulaşıcıdır.",
        judge_saying(supported=False, answers=False), k=5)

    assert records[0].supported is False
    assert summary["grounded_when_answered"] == 0.0


def test_abstention_is_detected_inside_a_longer_reply():
    # Models rarely emit a bare sentinel; a wrapped one still counts.
    _, records = evaluate(
        GOLD, CHUNKS, Retriever([1]),
        lambda q, c: f"Pasajlarda bu bilgi yok. {ABSTAIN}", never_judged, k=5)
    assert records[0].abstained


def test_summary_of_nothing_has_no_rates():
    summary = summarise([])
    assert summary["queries"] == 0
    assert summary["hallucination_rate"] is None
    assert summary["grounded_when_answered"] is None


def test_report_renders_missing_rates_rather_than_crashing():
    text = report(summarise([]))
    assert "n/a" in text and "UYDURDU" in text


def test_report_shows_the_hallucination_rate():
    records = [Record("a", "q", span_retrieved=False, abstained=False,
                      answer="uydurma")]
    assert "100.0%" in report(summarise(records))


def test_passages_are_numbered_with_their_heading():
    text = format_passages(CHUNKS)
    assert text.startswith("[1] Başlık")
    assert "[2] Başlık" in text
