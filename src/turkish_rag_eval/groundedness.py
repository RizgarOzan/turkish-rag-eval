"""Score the generation half: is the answer actually held up by what was retrieved.

The project is called turkish-rag-eval and measured only the R. This closes
that gap, and it does so using a label the gold set already carries - the
verbatim answer span - rather than a second round of annotation.

The design turns on one fact the harness already knows: **whether the answer
span was retrieved at all**. That splits every query into two populations that
deserve completely different questions, and collapsing them (as a single
"accuracy" would) hides the failure people actually care about:

span retrieved      the model should answer, and the answer should rest on the
                    passages. Scored for groundedness and for whether it
                    conveys the span.

span not retrieved  the gold passage is missing, so the expected behaviour is
                    to decline. An answer is still judged, because articles
                    state a fact in more than one place; an answer the
                    passages do not state is a hallucination, and the rate at
                    which that happens is the number that decides whether a
                    Turkish RAG system can be pointed at users.

Abstention is detected deterministically. The generator is instructed to reply
with a sentinel when the passages do not contain the answer, so "did it
abstain" never depends on a judge's mood; only groundedness and correctness,
which are genuinely judgements, go to a model.

Nothing here runs without an API key, and the evaluation takes a generator and
a judge as arguments, so the scoring logic is testable with stubs and a
different provider can be dropped in without touching it. A model named
``provider:model`` (``groq:``, ``nvidia:``) goes to that provider's
OpenAI-compatible endpoint; a bare name goes to Anthropic.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import STRATEGIES
from .corpus import load_corpus
from .gold import load_gold, resolve_gold
from .models import DEFAULT_MODEL
from .paths import ROOT
from .run_eval import CORPUS, is_relevant

#: Replied verbatim when the passages do not answer the question. Chosen to be
#: something no fluent Turkish answer would contain by accident.
ABSTAIN = "BILGI_YOK"

DEFAULT_JUDGE_MODEL = "claude-opus-5"
DEFAULT_ANSWER_MODEL = "claude-opus-5"

GENERATOR_SYSTEM = f"""Sen Türkçe bir soru-cevap asistanısın.

Kurallar:
- Yalnızca sana verilen pasajlardaki bilgiyi kullan. Kendi bilgini ekleme.
- Cevabı pasajlarda bulamıyorsan, tek başına şunu yaz: {ABSTAIN}
- Bulabiliyorsan, en fazla iki cümleyle ve Türkçe cevapla.
- Tahmin etme. Emin değilsen {ABSTAIN} yaz."""

JUDGE_SYSTEM = """You grade Turkish question-answering for a retrieval benchmark.

You receive a question, the passages a system retrieved, the answer it wrote,
and the gold answer span taken verbatim from the source article.

supported: every factual claim in the answer is stated in the passages. An
answer that is true in the world but absent from the passages is NOT supported.

answers_question: the answer conveys what the gold span says. Wording may
differ; the substance may not. A hedge that never states the fact is false."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "answers_question": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "answers_question", "reason"],
    "additionalProperties": False,
}


#: provider -> (chat completions URL, environment variable holding the key)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions",
               "NVIDIA_API_KEY"),
}

#: Per-provider request extras. gpt-oss on Groq reasons before answering; low
#: effort for the same reason the Anthropic generator runs at low effort.
EXTRA = {"groq:openai/gpt-oss-120b": {"reasoning_effort": "low"},
         "groq:openai/gpt-oss-20b": {"reasoning_effort": "low"}}


def split_model(name: str) -> tuple[str, str]:
    """``groq:openai/gpt-oss-120b`` -> ("groq", "openai/gpt-oss-120b")."""
    provider, sep, model = name.partition(":")
    if not sep:
        return "anthropic", name
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"use one of {sorted(PROVIDERS)} or a bare Claude name")
    return provider, model


def parse_verdict(text: str) -> dict:
    """The first JSON object in a reply that carries both verdict fields."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"supported", "answers_question"} <= value.keys():
            return value
    raise ValueError(f"no verdict in judge reply: {text[:200]!r}")


class ChatClient:
    """One OpenAI-compatible endpoint, with rate-limit waits and a token tally.

    With ``cache``, every reply is appended to that JSON-lines file and a
    request already in it is answered from there, so a run that dies halfway
    (a dropped connection, a daily quota) resumes instead of starting over.
    """

    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(self, provider: str, model: str, api_key: str | None = None,
                 session=None, sleep=time.sleep, retries: int = 8,
                 timeout: int = 600, cache: Path | None = None):
        if session is None:
            import requests
            session = requests.Session()
        self.url, env = PROVIDERS[provider]
        self.key = api_key or os.environ[env]
        self.model = model
        self.extra = EXTRA.get(f"{provider}:{model}", {})
        self.session, self.sleep = session, sleep
        self.retries, self.timeout = retries, timeout
        self.tokens = {"prompt": 0, "completion": 0, "calls": 0, "seconds": 0.0}
        self.cache, self.cached = cache, {}
        if cache and cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.cached[entry["key"]] = entry

    def tally(self, usage: dict, seconds: float) -> None:
        self.tokens["prompt"] += usage.get("prompt_tokens") or 0
        self.tokens["completion"] += usage.get("completion_tokens") or 0
        self.tokens["calls"] += 1
        self.tokens["seconds"] = round(self.tokens["seconds"] + seconds, 1)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                **self.extra}
        headers = {"Authorization": f"Bearer {self.key}",
                   # Groq answers the default python User-Agent with 403.
                   "User-Agent": "turkish-rag-eval"}
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False)
                             .encode("utf-8")).hexdigest()
        if key in self.cached:
            entry = self.cached[key]
            self.tally(entry["usage"], entry["seconds"])
            return entry["text"]

        started = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(self.url, json=body,
                                             headers=headers,
                                             timeout=self.timeout)
            except OSError:  # requests' connection errors and timeouts too
                if attempt == self.retries:
                    raise
                self.sleep(15.0 * (attempt + 1))
                continue
            if response.status_code == 200:
                break
            if response.status_code not in self.RETRYABLE or attempt == self.retries:
                raise RuntimeError(f"{self.model}: HTTP {response.status_code} "
                                   f"{response.text[:300]}")
            wait = response.headers.get("retry-after")
            # A long silent wait looks exactly like a hang; say which limit.
            print(f"{self.model}: HTTP {response.status_code}, retry-after "
                  f"{wait}: {response.text[:200]}", file=sys.stderr)
            try:
                self.sleep(min(float(wait), 300.0))
            except (TypeError, ValueError):
                self.sleep(15.0 * (attempt + 1))

        data = response.json()
        usage = data.get("usage") or {}
        seconds = round(time.monotonic() - started, 1)
        self.tally(usage, seconds)
        text = (data["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise RuntimeError(f"{self.model}: empty answer "
                               f"(finish_reason {data['choices'][0].get('finish_reason')})")
        if self.cache:
            entry = {"key": key, "text": text, "seconds": seconds,
                     "usage": {k: usage.get(k) for k in
                               ("prompt_tokens", "completion_tokens")}}
            self.cached[key] = entry
            with self.cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return text


def user_prompt(question: str, chunks: list[dict]) -> str:
    return f"Pasajlar:\n{format_passages(chunks)}\n\nSoru: {question}"


def judge_prompt(question: str, answer: str, span: str,
                 chunks: list[dict]) -> str:
    return (f"Question: {question}\n\n"
            f"Passages:\n{format_passages(chunks)}\n\n"
            f"Answer: {answer}\n\n"
            f"Gold span: {span}")


class ChatGenerator:
    def __init__(self, client: ChatClient):
        self.client = client

    def __call__(self, question: str, chunks: list[dict]) -> str:
        return self.client.complete(GENERATOR_SYSTEM,
                                    user_prompt(question, chunks), 4096)


class ChatJudge:
    def __init__(self, client: ChatClient):
        self.client = client

    def __call__(self, question: str, answer: str, span: str,
                 chunks: list[dict]) -> dict:
        # No structured-output guarantee on these endpoints, so the schema is
        # stated and the reply parsed; a reply without it raises. The budget
        # is large because a reasoning judge thinks before it writes.
        system = (f"{JUDGE_SYSTEM}\n\nReply with one JSON object only, "
                  f"matching this schema:\n{json.dumps(JUDGE_SCHEMA)}")
        return parse_verdict(self.client.complete(
            system, judge_prompt(question, answer, span, chunks), 16000))


@dataclass
class Record:
    qid: str
    question: str
    span_retrieved: bool
    abstained: bool
    answer: str
    supported: bool | None = None
    answers_question: bool | None = None
    reason: str = ""

    @property
    def hallucinated(self) -> bool:
        """Answered with a claim the retrieved passages do not state."""
        return not self.abstained and self.supported is False


def format_passages(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {chunk['heading_path']}\n{chunk['body']}"
        for i, chunk in enumerate(chunks, 1))


class AnthropicGenerator:
    """Answers a question from retrieved passages, or declines."""

    def __init__(self, client, model: str = DEFAULT_ANSWER_MODEL):
        self.client = client
        self.model = model

    def __call__(self, question: str, chunks: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=GENERATOR_SYSTEM,
            # Effort is deliberately low: the task is to read a short context
            # and either restate it or decline. Spending more here would make
            # the generator better at guessing, which is the failure mode this
            # evaluation is trying to measure.
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": user_prompt(question, chunks),
            }],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()


class AnthropicJudge:
    """Decides whether an answer is supported and whether it answers."""

    def __init__(self, client, model: str = DEFAULT_JUDGE_MODEL):
        self.client = client
        self.model = model

    def __call__(self, question: str, answer: str, span: str,
                 chunks: list[dict]) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=JUDGE_SYSTEM,
            output_config={"format": {"type": "json_schema",
                                      "schema": JUDGE_SCHEMA}},
            messages=[{
                "role": "user",
                "content": judge_prompt(question, answer, span, chunks),
            }],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


def evaluate(gold: list[dict], chunks: list[dict], retriever, generator,
             judge, k: int = 5) -> tuple[dict, list[Record]]:
    """Answer every gold question from retrieved context and score the result."""
    records = []
    for item in gold:
        hits = retriever.search(item["question"], k)
        retrieved = [chunks[index] for index, _ in hits]
        span_retrieved = any(is_relevant(chunk, item) for chunk in retrieved)

        answer = generator(item["question"], retrieved)
        abstained = ABSTAIN in answer

        record = Record(
            qid=item["qid"], question=item["question"],
            span_retrieved=span_retrieved, abstained=abstained, answer=answer)

        # Judging an abstention is pointless - there is no claim to support.
        # An answer without the gold span is still judged: the span marks one
        # place a fact is stated, and the passages may state it elsewhere.
        if not abstained:
            verdict = judge(item["question"], answer, item["answer_span"],
                            retrieved)
            record.supported = bool(verdict["supported"])
            record.answers_question = bool(verdict["answers_question"])
            record.reason = verdict.get("reason", "")

        records.append(record)
    return summarise(records), records


def summarise(records: list[Record]) -> dict:
    """Rates over each population, reported separately and never pooled."""
    found = [r for r in records if r.span_retrieved]
    missed = [r for r in records if not r.span_retrieved]
    judged = [r for r in found if r.supported is not None]
    judged_missed = [r for r in missed if r.supported is not None]

    def rate(count: int, total: int) -> float | None:
        return count / total if total else None

    return {
        "queries": len(records),
        "span_retrieved": len(found),
        "span_not_retrieved": len(missed),
        # With the evidence present: did it use it, and use only it.
        "answered_when_retrieved": rate(
            sum(1 for r in found if not r.abstained), len(found)),
        "grounded_when_answered": rate(
            sum(1 for r in judged if r.supported), len(judged)),
        "correct_when_answered": rate(
            sum(1 for r in judged if r.answers_question), len(judged)),
        # With the evidence absent: did it keep quiet, and if not, did the
        # passages at least state what it said. The last one is the headline.
        "abstained_when_not_retrieved": rate(
            sum(1 for r in missed if r.abstained), len(missed)),
        "answered_when_not_retrieved": rate(
            sum(1 for r in missed if not r.abstained), len(missed)),
        "correct_when_answered_without_span": rate(
            sum(1 for r in judged_missed if r.answers_question),
            len(judged_missed)),
        "hallucination_rate": rate(
            sum(1 for r in missed if r.hallucinated), len(missed)),
    }


def report(summary: dict) -> str:
    def percent(value) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    return "\n".join([
        f"{summary['queries']} soru: {summary['span_retrieved']} tanesinde "
        f"cevap pasajı getirildi, {summary['span_not_retrieved']} tanesinde "
        f"getirilmedi.",
        "",
        "Kanıt getirildiğinde",
        f"  cevap verdi          {percent(summary['answered_when_retrieved'])}",
        f"  pasajlara dayanıyor  {percent(summary['grounded_when_answered'])}",
        f"  soruyu cevaplıyor    {percent(summary['correct_when_answered'])}",
        "",
        "Kanıt getirilmediğinde",
        f"  doğru şekilde sustu  {percent(summary['abstained_when_not_retrieved'])}",
        f"  yine de cevap verdi  {percent(summary['answered_when_not_retrieved'])}",
        f"    soruyu cevaplıyor  {percent(summary['correct_when_answered_without_span'])}",
        f"  UYDURDU              {percent(summary['hallucination_rate'])}",
    ])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "groundedness.json")
    parser.add_argument("--chunking", default="hierarchical",
                        choices=list(STRATEGIES))
    parser.add_argument("--retriever", default="bm25_stem5",
                        choices=("bm25_stem5", "bm25_nostem", "dense"))
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="embedding model, when --retriever is dense")
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("-k", type=int, default=5,
                        help="passages given to the generator")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N questions, for a dry run")


def build_retriever(args, chunks: list[dict]):
    from .retrieval import DenseRetriever, SparseRetriever

    if args.retriever == "dense":
        from sentence_transformers import SentenceTransformer

        from .models import prefixes_for
        query_prefix, doc_prefix = prefixes_for(args.model)
        return DenseRetriever(chunks, SentenceTransformer(args.model),
                              query_prefix, doc_prefix)
    return SparseRetriever(
        chunks, stem_length=5 if args.retriever == "bm25_stem5" else None)


def build_models(args):
    """(generator, judge, chat clients) for the two model names, or an error."""
    providers = {split_model(name)[0]
                 for name in (args.answer_model, args.judge_model)}
    for provider in providers:
        env = "ANTHROPIC_API_KEY" if provider == "anthropic" else PROVIDERS[provider][1]
        if not os.environ.get(env):
            return f"{env} gerekli - bu komut bir üretici ve bir jüri modeli çağırır."

    anthropic_client = None
    if "anthropic" in providers:
        try:
            import anthropic
        except ImportError:
            return "pip install 'turkish-rag-eval[llm]'"
        anthropic_client = anthropic.Anthropic()

    clients = {}
    parts = []
    for name, chat_cls, anthropic_cls in (
            (args.answer_model, ChatGenerator, AnthropicGenerator),
            (args.judge_model, ChatJudge, AnthropicJudge)):
        provider, model = split_model(name)
        if provider == "anthropic":
            parts.append(anthropic_cls(anthropic_client, model))
        else:
            client = clients.setdefault(name, ChatClient(
                provider, model, cache=args.out.with_suffix(".cache.jsonl")))
            parts.append(chat_cls(client))
    return parts[0], parts[1], clients


def run(args) -> int:
    models = build_models(args)
    if isinstance(models, str):
        print(models)
        return 1
    generator, judge, clients = models

    corpus = load_corpus(args.corpus or CORPUS)
    gold = load_gold(resolve_gold(args.gold))
    if args.limit:
        gold = gold[:args.limit]
    chunks = [c for doc in corpus.docs for c in STRATEGIES[args.chunking](doc)]
    print(f"{len(corpus)} belge, {len(chunks)} parça, {len(gold)} soru")
    print(f"{args.chunking} + {args.retriever}, k={args.k}\n")

    summary, records = evaluate(
        gold, chunks, build_retriever(args, chunks), generator, judge, k=args.k)

    summary.update({"chunking": args.chunking, "retriever": args.retriever,
                    "k": args.k, "answer_model": args.answer_model,
                    "judge_model": args.judge_model,
                    "temperature": 0 if clients else None,
                    "date": time.strftime("%Y-%m-%d"),
                    "tokens": {name: c.tokens for name, c in clients.items()},
                    "corpus_fingerprint": corpus.fingerprint})
    # The records hold the models' own words. They stay local (gitignored):
    # the free API terms do not allow redistributing raw model output, and the
    # rates in the summary are what the benchmark reports.
    records_path = args.out.with_suffix(".records.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary}, ensure_ascii=False,
                                   indent=2), encoding="utf-8")
    records_path.write_text(json.dumps([asdict(r) for r in records],
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(report(summary))
    print(f"\n-> {args.out}\n-> {records_path} (yerel)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
