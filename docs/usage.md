# Usage

The short command list is in the [README](../README.md#commands);
`turkish-rag-eval <command> --help` gives each command's options. Measured
numbers for everything here are in [results.md](results.md).

## Installing

Python 3.10+. Everything runs on a CPU; a GPU only speeds up encoding
([GPU table](results.md#speed-and-hardware)).

```bash
pip install turkish-rag-eval          # metrics, BM25, gold-set tooling
pip install 'turkish-rag-eval[all]'   # + dense retrieval, charts, LLM commands
```

From a checkout:

```bash
git clone https://github.com/RizgarOzan/turkish-rag-eval
cd turkish-rag-eval
pip install -e '.[all]'
turkish-rag-eval fetch-corpus --verify   # the pinned data/raw/corpus.json ships in the repo
turkish-rag-eval run              # results/summary.json + per-query files
python -m pytest -q
```

## Choosing a model

`run --model <name>` swaps the embedding model; any sentence-transformers
model works. Results for a non-default model go to
`results/models/<org>__<name>/`, so the main table is never overwritten.

`run --embedding-cache <dir>` keeps the chunk embeddings and reuses them while
model and chunk text stay the same; `stress` and `ask` read the same cache.
What it saves is measured in [results.md](results.md#speed-and-hardware).

To submit a model to the leaderboard, open a pull request adding
`results/models/<org>__<name>/` (`turkish-rag-eval run --model <org>/<name>`,
then `leaderboard --check`). CI re-derives every metric from the per-query
relevance arrays committed beside it, and checks the harness version and
corpus fingerprint.

## Your own corpus

The harness is not tied to its own articles. Point it at a folder of `.txt` or
`.md` files, a BEIR directory, or a JSON corpus:

```bash
turkish-rag-eval run --corpus ./belgelerim --gold ./sorular.json                      --retrievers bm25_stem5 bm25_nostem
```

A BM25-only run needs no model download and no torch, so the base install is
enough to find out which chunker to use and whether stemming is worth it on
your documents.

If you have no labelled questions, `bootstrap` drafts some (the name means
drafting a gold set here, not statistical resampling):

```bash
export ANTHROPIC_API_KEY=...
turkish-rag-eval bootstrap ./belgelerim --out gold-draft.json --per-doc 3
```

It runs the same procedure the contributed questions in this repository used.
One pass writes questions and marks the answer span; a second pass sees only
the document and the questions and marks the span again. Where the passes
disagree the item is written as `"review": "needs-human"` and **never loads**
until a person settles it. Spans that are not verbatim, and questions copied
out of their own answer, are dropped with a reason.

What you get is a draft to review, not a gold set.

## Groundedness with other models

`groundedness` takes model names prefixed `groq:`, `nvidia:` or `openai:`
(a bare name goes to Anthropic). Keys can sit in a `.env` file (gitignored) in
the directory you run from; `openai:` follows `OPENAI_BASE_URL`, so any
OpenAI-compatible endpoint, a local router included, works.

## Stress test

```bash
turkish-rag-eval stress --model intfloat/multilingual-e5-small
turkish-rag-eval stress --sparse-only        # BM25 only, no model download
turkish-rag-eval run --normalize             # the same fix in front of a normal run
```

## One question at a time

`ask` takes a gold question id, or any question you type, and prints what each
retriever returns for it. For a gold question the chunks that hold the answer
get a ✓, by the same rule `run` scores with:

```text
$ turkish-rag-eval ask q034
q034: Bacak damarında pıhtı kollarda da olur mu?
answer in: Derin ven trombozu  (✓ = chunk holds it)

dense
  1 · Romatoid artrit > Belirti ve bulgular > Eklemler: Artrit sinoviyal zarın iltihaplanmasını içerir. Eklemler şişer, hassasl…
  2 · Ateroskleroz: İlerlemiş aterosklerozda görülen başka bir yaygın senaryo ise kladikasy…
  3 · Gut > Tarihçe: Kurban yatağına yatar ve sağlıklı bir şekilde uykuya dalar. Sabaha karş…

bm25_stem5
  1 · Derin ven trombozu: Derin ven trombozu (DVT), çoğunlukla bacaklarda veya pelviste olmak üze…
  2 · Derin ven trombozu > Patofizyoloji: Kan damarları hasar gördüğünde (hemostaz) kan kaybını en aza indirmek i…
  3 · Derin ven trombozu > Korunma: Genel popülasyonda kan pıhtılarının önlenmesi için, uzun süre otururken…

bm25_nostem
  1 · Derin ven trombozu > Tanı: Derin ven trombozu şüphesi için kompresyon ultrasonografisi standart ta…
  2 · Şizofreni > Hastalığın oluşum nedenleri > Beyinde yapısal ve işlevsel bozukluklar: Beynin belli bir ya da birkaç bölgesinde birinci derece sorumlu özgül b…
  3 · Osteoporoz > Osteoporozun başlıca etkileri: Bunların yanı sıra, omurların veya kol-bacak kemiklerinin kırılma tehli…

hybrid_rrf
  1 ✓ Derin ven trombozu > Belirtiler ve semptomlar: Semptomlar klasik olarak bir bacağı etkiler ve tipik olarak saatler vey…
  2 · Yüksek tansiyon > Belirtiler ve semptomlar > İkincil hipertansiyon: Renal arter stenozu (RAS, böbreği besleyen arterlerde daralma), orta ha…
  3 · Derin ven trombozu: Derin ven trombozu (DVT), çoğunlukla bacaklarda veya pelviste olmak üze…

found by: hybrid_rrf@1
missed in top 3: dense, bm25_stem5, bm25_nostem
```

This one shows why the hybrid wins on average. BM25 finds the right article
but not the section, the default MiniLM model drifts to other diseases, and
only the fused ranking puts the symptoms section first. `-k` sets how many
chunks each retriever shows, `--sparse-only` skips the embedding model, and
`--embedding-cache` reuses what `run` already encoded.

## Serving

`serve` puts one retriever behind HTTP: stemmed BM25 by default, or with
`--model` the hybrid of that model and BM25. The request format and the load
test are in [results.md](results.md#serving).
