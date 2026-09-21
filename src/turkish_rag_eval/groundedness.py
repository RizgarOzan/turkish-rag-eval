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

span not retrieved  there is nothing in the context that answers the question,
                    so the only correct behaviour is to decline. Answering
                    anyway is a hallucination, and the rate at which that
                    happens is the number that decides whether a Turkish RAG
                    system can be pointed at users.

Abstention is detected deterministically. The generator is instructed to reply
with a sentinel when the passages do not contain the answer, so "did it
abstain" never depends on a judge's mood; only groundedness and correctness,
which are genuinely judgements, go to a model.

Nothing here runs without an API key, and the evaluation takes a generator and
a judge as arguments, so the scoring logic is testable with stubs and a
different provider can be dropped in without touching it.
"""

import argparse
import json
import os
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
        """Answered a question its context could not support."""
        return not self.span_retrieved and not self.abstained


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
                "content": f"Pasajlar:\n{format_passages(chunks)}\n\n"
                           f"Soru: {question}",
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
                "content": (f"Question: {question}\n\n"
                            f"Passages:\n{format_passages(chunks)}\n\n"
                            f"Answer: {answer}\n\n"
                            f"Gold span: {span}"),
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

        # Judging an abstention is pointless - there is no claim to support -
        # and judging a hallucination is settled already: the context cannot
        # support it, which is what makes it one.
        if span_retrieved and not abstained:
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
        # With the evidence absent: did it keep quiet. This is the headline.
        "abstained_when_not_retrieved": rate(
            sum(1 for r in missed if r.abstained), len(missed)),
        "hallucination_rate": rate(
            sum(1 for r in records if r.hallucinated), len(missed)),
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


def run(args) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY gerekli - bu komut bir üretici ve bir jüri "
              "modeli çağırır.")
        return 1
    try:
        import anthropic
    except ImportError:
        print("pip install 'turkish-rag-eval[llm]'")
        return 1

    corpus = load_corpus(args.corpus or CORPUS)
    gold = load_gold(resolve_gold(args.gold))
    if args.limit:
        gold = gold[:args.limit]
    chunks = [c for doc in corpus.docs for c in STRATEGIES[args.chunking](doc)]
    print(f"{len(corpus)} belge, {len(chunks)} parça, {len(gold)} soru")
    print(f"{args.chunking} + {args.retriever}, k={args.k}\n")

    client = anthropic.Anthropic()
    summary, records = evaluate(
        gold, chunks, build_retriever(args, chunks),
        AnthropicGenerator(client, args.answer_model),
        AnthropicJudge(client, args.judge_model), k=args.k)

    summary.update({"chunking": args.chunking, "retriever": args.retriever,
                    "k": args.k, "answer_model": args.answer_model,
                    "judge_model": args.judge_model,
                    "corpus_fingerprint": corpus.fingerprint})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"summary": summary, "records": [asdict(r) for r in records]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(report(summary))
    print(f"\n-> {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
