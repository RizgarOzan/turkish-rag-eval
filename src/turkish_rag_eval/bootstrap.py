"""Draft a gold set for a corpus that does not have one.

This is the wall every "point it at your own documents" benchmark hits: the
harness needs labelled questions, and nobody has labelled questions for their
own files. Writing 58 of them by hand took this project weeks.

So the drafting is done here the same way the repository's own contributed
questions were, and with the same safeguards rather than fewer:

**Two independent passes.** One pass writes questions and marks the answer
span; a second pass sees only the document and the questions, and marks the
span again from scratch. Where the two passes disagree, the question is
written out as ``"review": "needs-human"``, and ``load_gold`` refuses to load
it until a person settles it. A single-pass draft with no disagreement signal
would look exactly as confident and be worth much less.

**Every invariant checked before anything is written.** The span has to appear
verbatim in the document, or a retrieved chunk could never be judged relevant.
The question has to be a paraphrase rather than a copy of the span - a copied
question hands BM25 an unearned win and quietly inflates every sparse number
in the table. Drafts that fail are dropped with a reason, not silently kept.

What comes out is a starting point that a person reviews, not a gold set. The
honest version of the pitch is: half an hour of checking disagreements instead
of a week of writing questions.
"""

import argparse
import json
import os
import re
from pathlib import Path

from .agreement import span_f1
from .corpus import load_corpus
from .gold import normalise
from .turkish_text import turkish_lower
from .validate_gold import MAX_OVERLAP, overlap

DEFAULT_MODEL = "claude-opus-5"

#: Token-F1 floor for two spans where neither contains the other. This is the
#: rule the README already states for the contributed questions; the
#: bootstrapper applies the same one rather than inventing a second.
AGREEMENT_THRESHOLD = 0.5

#: A span long enough to be a section and short enough to be an answer.
MIN_SPAN_CHARS = 20
MAX_SPAN_CHARS = 400

DRAFT_SYSTEM = """Türkçe bir retrieval değerlendirme seti hazırlıyorsun.

Sana bir belge veriliyor. Belgeden cevaplanabilecek sorular yaz.

Her soru için:
- "question": Türkçe, soru işaretiyle biten, tek bir olguyu soran bir soru.
  Sorunun kelimelerini cevabın geçtiği cümleden KOPYALAMA - başka kelimelerle
  sor. Kopyalanmış bir soru anahtar kelime aramasına haksız avantaj verir ve
  ölçümü bozar.
- "answer_span": Cevabı içeren, belgeden BİREBİR kopyalanmış kısa bir alıntı.
  Tek cümle yeterlidir. Tek bir karakteri bile değiştirme.

Belgede cevabı olmayan soru yazma. Belgeden az sayıda ama sağlam soru
çıkarmak, çok sayıda zayıf soru çıkarmaktan iyidir."""

MARK_SYSTEM = """Türkçe bir retrieval değerlendirme setini ikinci kez
etiketliyorsun.

Sana bir belge ve o belgeye ait sorular veriliyor. Her soru için, cevabı
içeren kısmı belgeden BİREBİR kopyala. Başka birinin işaretlediğini görmüyorsun;
kendi kararını ver.

Belgede o sorunun cevabı yoksa, o soru için boş dize ("") döndür."""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer_span": {"type": "string"},
                },
                "required": ["question", "answer_span"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}

MARK_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["spans"],
    "additionalProperties": False,
}


def passes_agree(first: str, second: str) -> bool:
    """Whether the two annotation passes located the same answer.

    Containment is checked first, and it is the signal that matters: the
    harness judges a chunk relevant when the chunk *contains* the span, so two
    spans where one contains the other mark the same passages relevant and are
    the same label for every purpose the harness has. They differ only in how
    much surrounding sentence each pass swept in.

    A token-similarity floor alone would not do. On this repository's own 150
    double-labelled questions, Jaccard overlap sits below 0.6 for 46 of them -
    yet 42 of those 46 are containment pairs. The passes were almost never
    disagreeing about where the answer is, only about how much of the sentence
    to sweep in, and a similarity floor would have sent a reviewer to arbitrate
    a third of an already-reviewed set.

    The rule is calibrated against those labels rather than chosen: it
    reproduces all 150 of them exactly, including the four genuine
    disagreements, such as the two passes marking different sentences that
    both name polysomes.

    Token F1 stays as the fallback for the case containment cannot judge - two
    spans that genuinely point somewhere different - at the same 0.5 floor the
    README states for the contributed questions.
    """
    a, b = normalise(first), normalise(second)
    if a and b and (a in b or b in a):
        return True
    return span_f1(first, second) >= AGREEMENT_THRESHOLD


def slugify(text: str) -> str:
    """A qid-safe stem: ascii, lowercase, hyphenated.

    ``turkish_lower`` rather than ``str.lower()``: Python maps "İ" to an "i"
    plus a combining dot, and the dot is not in the allowed character class,
    so "İstanbul" would come out as "i-stanbul". This is the same casing trap
    the harness exists to measure, met in its own filenames.
    """
    folded = (turkish_lower(text)
              .replace("ı", "i").replace("ş", "s").replace("ğ", "g")
              .replace("ü", "u").replace("ö", "o").replace("ç", "c"))
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", folded)).strip("-") or "doc"


def check_draft(question: str, span: str, document: str) -> str | None:
    """Why this draft cannot be used, or None if it can."""
    question = question.strip()
    span = span.strip()

    if not question.endswith("?"):
        return "question does not end with '?'"
    if not span:
        return "empty answer span"
    if len(span) < MIN_SPAN_CHARS:
        return f"span shorter than {MIN_SPAN_CHARS} characters"
    if len(span) > MAX_SPAN_CHARS:
        return f"span longer than {MAX_SPAN_CHARS} characters"
    if normalise(span) not in normalise(document):
        # The most common model failure: a span that paraphrases rather than
        # quotes. It would make the question unjudgeable, so it goes.
        return "span is not in the document verbatim"
    share = overlap(question, span)
    if share > MAX_OVERLAP:
        return f"{share:.0%} of the question's words are copied from the span"
    return None


class AnthropicDrafter:
    """Two passes over one document: write questions, then re-mark the spans."""

    def __init__(self, client, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def _json(self, system: str, prompt: str, schema: dict) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=system,
            output_config={"effort": "high",
                           "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    def draft(self, document: dict, count: int) -> list[dict]:
        payload = self._json(
            DRAFT_SYSTEM,
            f"Belge başlığı: {document['title']}\n\n{document['text']}\n\n"
            f"En fazla {count} soru yaz.",
            DRAFT_SCHEMA)
        return payload["questions"][:count]

    def mark(self, document: dict, questions: list[str]) -> list[str]:
        if not questions:
            return []
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        payload = self._json(
            MARK_SYSTEM,
            f"Belge başlığı: {document['title']}\n\n{document['text']}\n\n"
            f"Sorular:\n{numbered}\n\n"
            f"{len(questions)} adet alıntıyı sırayla döndür.",
            MARK_SCHEMA)
        spans = payload["spans"]
        # A short reply would silently shift every later span onto the wrong
        # question, so pad rather than zip and lose the alignment.
        return list(spans) + [""] * (len(questions) - len(spans))


def build_items(document: dict, drafts: list[dict], second_spans: list[str],
                prefix: str, start: int = 1) -> tuple[list[dict], list[str]]:
    """Validated gold items for one document, plus the reasons for each drop."""
    items, dropped = [], []
    stem = slugify(document["title"])
    number = start

    for draft, second in zip(drafts, second_spans):
        question = draft["question"].strip()
        span = draft["answer_span"].strip()

        problem = check_draft(question, span, document["text"])
        if problem:
            dropped.append(f"{document['title']}: {problem} - {question[:60]}")
            continue

        item = {
            "qid": f"{prefix}-{stem}-{number:03d}",
            "question": question,
            "doc_id": document["doc_id"],
            "doc_title": document["title"],
            "answer_span": span,
            "annotator": "llm-1",
            "source": "llm-draft",
        }
        number += 1

        second = (second or "").strip()
        # A second span that is not verbatim tells us nothing about agreement,
        # so it is recorded as a disagreement rather than quietly accepted.
        second_valid = bool(second) and normalise(second) in normalise(
            document["text"])
        if second_valid:
            item["second_annotation"] = {"annotator": "llm-2",
                                         "answer_span": second}
            agreed = passes_agree(span, second)
        else:
            agreed = False
        item["review"] = "agreed" if agreed else "needs-human"
        items.append(item)

    return items, dropped


def bootstrap(documents: list[dict], drafter, per_document: int,
              prefix: str) -> tuple[list[dict], list[str]]:
    items, dropped = [], []
    for document in documents:
        drafts = drafter.draft(document, per_document)
        second = drafter.mark(document, [d["question"] for d in drafts])
        document_items, document_dropped = build_items(
            document, drafts, second, prefix)
        items += document_items
        dropped += document_dropped
    return items, dropped


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("corpus", type=Path,
                        help="a JSON/JSONL file, a BEIR directory, or a folder "
                             "of .txt/.md files")
    parser.add_argument("--out", type=Path, default=Path("gold-draft.json"))
    parser.add_argument("--per-doc", type=int, default=3,
                        help="questions to attempt per document (default 3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only draft from the first N documents")
    parser.add_argument("--prefix", default="draft",
                        help="qid prefix (default 'draft')")
    parser.add_argument("--model", default=DEFAULT_MODEL)


def run(args) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY gerekli - soru taslakları bir model çağırır.")
        return 1
    try:
        import anthropic
    except ImportError:
        print("pip install 'turkish-rag-eval[llm]'")
        return 1

    corpus = load_corpus(args.corpus)
    documents = corpus.docs[:args.limit] if args.limit else corpus.docs
    print(f"{len(documents)} belge, belge başına en fazla {args.per_doc} soru\n")

    items, dropped = bootstrap(
        documents, AnthropicDrafter(anthropic.Anthropic(), args.model),
        args.per_doc, args.prefix)

    for reason in dropped:
        print(f"  atlandı: {reason}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    needs_human = [i for i in items if i["review"] == "needs-human"]
    print(f"\n{len(items)} soru yazıldı, {len(dropped)} taslak atlandı.")
    print(f"{len(needs_human)} tanesinde iki etiketleme anlaşamadı ve insan "
          f"kararı bekliyor;")
    print(f"bunlar siz 'review' alanını düzeltene kadar hiçbir ölçüme girmez.")
    print(f"\n-> {args.out}")
    print(f"Sonra: turkish-rag-eval run --corpus {args.corpus} "
          f"--gold {args.out} --retrievers bm25_stem5 bm25_nostem")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
