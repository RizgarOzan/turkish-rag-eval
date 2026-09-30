"""The scoring split is the whole idea, so it is what gets tested.

No network: the generator and judge are arguments, and these stubs stand in
for them. What matters is that a query whose evidence was never retrieved is
never scored as if it had been, and that an answer the passages do not support
is counted as a hallucination rather than quietly averaged away.
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
        judge_saying(supported=False, answers=True), k=5)

    assert not records[0].span_retrieved
    assert records[0].hallucinated
    assert summary["hallucination_rate"] == 1.0
    assert summary["answered_when_not_retrieved"] == 1.0
    assert summary["abstained_when_not_retrieved"] == 0.0


def test_an_answer_from_another_passage_is_not_a_hallucination():
    # The gold span marks one place a fact is stated; articles repeat facts.
    # Measured on the 2026-09-30 run: all five answers given without the span
    # were stated in a retrieved passage, so "no span" cannot mean "made up".
    summary, records = evaluate(
        GOLD, CHUNKS, Retriever([1]), lambda q, c: "Bir cevap.",
        judge_saying(supported=True, answers=False), k=5)

    assert not records[0].hallucinated
    assert records[0].supported is True
    assert summary["hallucination_rate"] == 0.0
    assert summary["answered_when_not_retrieved"] == 1.0
    assert summary["correct_when_answered_without_span"] == 0.0


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

    verdicts = iter([True, False])

    def judge(*_args):
        supported = next(verdicts)
        return {"supported": supported, "answers_question": supported,
                "reason": "stub"}

    summary, _ = evaluate(gold, CHUNKS, Mixed(), lambda q, c: "Bir cevap.",
                          judge, k=5)

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


# --- OpenAI-compatible providers (Groq, NVIDIA) -----------------------------

from turkish_rag_eval.groundedness import (  # noqa: E402
    ChatClient,
    parse_verdict,
    split_model,
)


def test_a_provider_prefix_selects_an_openai_compatible_endpoint():
    assert split_model("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert split_model("nvidia:nvidia/nemotron-3-super-120b-a12b") == (
        "nvidia", "nvidia/nemotron-3-super-120b-a12b")
    assert split_model("claude-opus-5") == ("anthropic", "claude-opus-5")


def test_an_unknown_provider_is_refused_rather_than_sent_to_anthropic():
    with pytest.raises(ValueError):
        split_model("mistral:large")


def test_a_verdict_is_read_from_a_fenced_or_chatty_reply():
    fenced = '```json\n{"supported": true, "answers_question": false, "reason": "x"}\n```'
    chatty = 'Here is my grade: {"supported": false, "answers_question": true, "reason": "{y}"} Done.'
    assert parse_verdict(fenced) == {"supported": True, "answers_question": False, "reason": "x"}
    assert parse_verdict(chatty)["reason"] == "{y}"


def test_a_reply_without_a_verdict_is_an_error_not_a_default():
    # Defaulting to "supported" would inflate the headline rate silently.
    with pytest.raises(ValueError):
        parse_verdict("I think it is supported.")
    with pytest.raises(ValueError):
        parse_verdict('{"supported": true}')


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def post(self, url, json, headers, timeout):
        self.sent.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def ok(text, prompt=10, completion=5):
    return FakeResponse(200, {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})


def test_the_client_waits_out_a_rate_limit_and_counts_tokens():
    session = FakeSession([FakeResponse(429, headers={"retry-after": "2"}),
                           ok("Cevap.", 100, 7)])
    waits = []
    client = ChatClient("groq", "openai/gpt-oss-120b", api_key="k",
                        session=session, sleep=waits.append)

    assert client.complete("sys", "user", max_tokens=50) == "Cevap."
    assert waits == [2.0]
    assert {k: client.tokens[k] for k in ("prompt", "completion", "calls")} == {
        "prompt": 100, "completion": 7, "calls": 1}
    assert client.tokens["seconds"] >= 0
    sent = session.sent[-1]
    assert sent["url"].startswith("https://api.groq.com/")
    assert sent["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["json"]["temperature"] == 0
    # Groq answers the default python User-Agent with 403.
    assert "turkish-rag-eval" in sent["headers"]["User-Agent"]


def test_the_client_gives_up_on_a_non_retryable_error():
    client = ChatClient("nvidia", "m", api_key="k",
                        session=FakeSession([FakeResponse(401, {"error": "no"})]),
                        sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="401"):
        client.complete("sys", "user", max_tokens=50)


def test_an_empty_answer_is_an_error():
    # A reasoning model that spends its whole budget thinking returns no
    # content; scoring that as "did not abstain" would count a hallucination.
    client = ChatClient("nvidia", "m", api_key="k",
                        session=FakeSession([ok("")]), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="empty"):
        client.complete("sys", "user", max_tokens=50)


def test_a_dropped_connection_is_retried():
    # NVIDIA reset the connection 20 minutes into the first full run and the
    # whole run was lost.
    class Dropping(FakeSession):
        def post(self, url, json, headers, timeout):
            if not self.sent:
                self.sent.append(None)
                raise ConnectionResetError(10054, "reset by peer")
            return super().post(url, json, headers, timeout)

    waits = []
    client = ChatClient("nvidia", "m", api_key="k",
                        session=Dropping([ok("Cevap.")]), sleep=waits.append)
    assert client.complete("sys", "user", max_tokens=50) == "Cevap."
    assert len(waits) == 1


def test_a_cached_reply_is_not_requested_again(tmp_path):
    cache = tmp_path / "run.cache.jsonl"
    first = ChatClient("groq", "m", api_key="k", cache=cache,
                       session=FakeSession([ok("Cevap.", 100, 7)]))
    assert first.complete("sys", "user", max_tokens=50) == "Cevap."

    # A rerun after a crash: no responses queued, so any request would fail.
    again = ChatClient("groq", "m", api_key="k", cache=cache,
                       session=FakeSession([]))
    assert again.complete("sys", "user", max_tokens=50) == "Cevap."
    # The tally still describes the whole run, not just the resumed part.
    assert (again.tokens["prompt"], again.tokens["calls"]) == (100, 1)

    other = ChatClient("groq", "m", api_key="k", cache=cache,
                       session=FakeSession([ok("Başka.")]))
    assert other.complete("sys", "another question", max_tokens=50) == "Başka."
