# turkish-rag-eval

[![PyPI](https://img.shields.io/pypi/v/turkish-rag-eval)](https://pypi.org/project/turkish-rag-eval/)
[![validate](https://github.com/RizgarOzan/turkish-rag-eval/actions/workflows/validate.yml/badge.svg)](https://github.com/RizgarOzan/turkish-rag-eval/actions/workflows/validate.yml)
[![licence: MIT + CC BY-SA 4.0](https://img.shields.io/badge/licence-MIT%20%2B%20CC%20BY--SA%204.0-blue)](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/NOTICE.md)
[![dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-RizgarOzan%2Fturkish--rag--eval-yellow)](https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval)

**Which parts of a RAG pipeline actually earn their cost on an agglutinative
language?** Three chunking strategies × four retrievers × six embedding
models, measured on a hand-labelled Turkish gold set, with confidence
intervals and a cost column — then pointed at your own documents.

```bash
pip install turkish-rag-eval
turkish-rag-eval run --corpus ./belgelerim --gold ./sorular.json
turkish-rag-eval report
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RizgarOzan/turkish-rag-eval/main/docs/media/ask-q034-dark.gif">
  <img alt="turkish-rag-eval ask q034: four retrievers answer one Turkish question side by side; only the hybrid ranking puts the chunk that holds the answer first" src="https://raw.githubusercontent.com/RizgarOzan/turkish-rag-eval/main/docs/media/ask-q034.gif">
</picture>

One question, four retrievers: only the hybrid finds the symptoms section of
the right article ([how to read this](#one-question-at-a-time)).

**Contents:** [Findings](#five-findings) · [Results](#results) ·
[Groundedness](#generation-groundedness) · [Leaderboard](#leaderboard) · [Stress test](#stress-test) ·
[Why not an existing benchmark?](#why-not-an-existing-benchmark) ·
[Your own corpus](#your-own-corpus) · [Running it](#running-it) ·
[Gold set](#gold-set) · [Status](#status) · [Limits](#limits) ·
[Contribute](#contribute) · [Design notes](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md)

## Five findings

**1. The embedding model matters more than any pipeline choice.** Every model
trained for retrieval beats stemmed BM25 (0.494) on its own. The best,
[Mursit-Large-TR-Retrieval](https://huggingface.co/newmindai/Mursit-Large-TR-Retrieval),
reaches **0.781** nDCG@10 on hierarchical chunks. A small E5 the same size as
the default goes from 0.501 to 0.642 with nothing else changed, at about the
same speed (22 ms against 23 ms). The popular default model is the weak link, not dense retrieval.
See the [Leaderboard](#leaderboard).

**2. Turkish stemming is the cheapest real win.** Truncating tokens to a
5-character prefix before BM25 lifts nDCG@10 by 23–29% on every chunking
strategy, and a paired bootstrap puts every one of those gains clear of zero
(`hierarchical +0.111, 95% CI [+0.026, +0.204]`). "diyabet", "diyabetin",
"diyabete", "diyabetli" are four surface forms of one concept; an unstemmed
index almost never matches the query's form.

**3. The top of the table is a tie, and the tie decides on cost.** With 58
queries, a paired bootstrap cannot separate the best configuration from the
other two hybrid ones; the remaining nine are measurably worse. So the
real choice at the top is price: `sentence + hybrid_rrf` gives the same
quality at 27 ms instead of 37 ms.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RizgarOzan/turkish-rag-eval/main/docs/charts/ndcg-intervals-dark.png">
  <img alt="nDCG@10 per configuration with 95% bootstrap intervals; the best cannot be told apart from the other two hybrid configurations, the remaining nine are measurably worse" src="https://raw.githubusercontent.com/RizgarOzan/turkish-rag-eval/main/docs/charts/ndcg-intervals.png">
</picture>

**4. The intuitive confidence signal is the useless one.** For deciding when
*not* to answer, the obvious measure — how far ahead the top hit is — is worse
than answering everything (0.25 selective accuracy against a 0.47 baseline).
RRF fuses ranks as `1/(60+rank)`, so the top-two gap is ~2% on every query,
confident or not. The dense retriever's raw cosine works; the margin does not.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RizgarOzan/turkish-rag-eval/main/docs/charts/abstention-dark.png">
  <img alt="Coverage against selective accuracy for two confidence signals; the top-1 margin performs worse than answering everything" src="https://raw.githubusercontent.com/RizgarOzan/turkish-rag-eval/main/docs/charts/abstention.png">
</picture>

**5. How a question is typed can cost more than the model choice.** Typed
in capitals, the default model's nDCG@10 falls from 0.501 to 0.051, a bigger
loss than the 0.280 that separates it from the best model; typed
without Turkish letters ("sigara icmek"), stemmed BM25 drops from
0.494 to 0.288. Even the best model, Mursit, loses
0.142 in capitals. A small normaliser that puts the Turkish letters back
from the corpus's own words and lowers caps the Turkish way recovers most of
it (BM25 0.288 → 0.477) and leaves correctly typed questions untouched.
See the [stress test](#stress-test).

Longer write-up of the first result:
[English](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/blog/2026-09-19-bm25-turkish-en.md) ·
[Türkçe](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/blog/2026-09-19-bm25-turkish-tr.md).

## Results

58 queries, default embedding model
`paraphrase-multilingual-MiniLM-L12-v2`, CPU only. Intervals are 95% bootstrap
over queries; `turkish-rag-eval report` regenerates this table.

| Chunking | Retriever | nDCG@10 | 95% CI | R@5 | MRR | P95 |
|---|---|---|---|---|---|---|
| **hierarchical** | **hybrid_rrf** | **0.613** | [0.508, 0.716] | 0.690 | 0.559 | 37 ms |
| sentence | hybrid_rrf | 0.558 | [0.453, 0.662] | 0.621 | 0.500 | 27 ms |
| fixed | hybrid_rrf | 0.517 | [0.404, 0.629] | 0.578 | 0.484 | 31 ms |
| sentence | bm25_stem5 | 0.510 | [0.403, 0.621] | 0.638 | 0.462 | 4 ms |
| hierarchical | dense | 0.501 | [0.396, 0.606] | 0.569 | 0.441 | 30 ms |
| hierarchical | bm25_stem5 | 0.494 | [0.386, 0.605] | 0.569 | 0.444 | 8 ms |
| fixed | bm25_stem5 | 0.476 | [0.371, 0.585] | 0.526 | 0.421 | 4 ms |
| sentence | dense | 0.461 | [0.356, 0.567] | 0.552 | 0.410 | 21 ms |
| fixed | dense | 0.446 | [0.341, 0.555] | 0.491 | 0.402 | 25 ms |
| sentence | bm25_nostem | 0.411 | [0.308, 0.515] | 0.483 | 0.356 | 4 ms |
| fixed | bm25_nostem | 0.387 | [0.290, 0.487] | 0.414 | 0.319 | 3 ms |
| hierarchical | bm25_nostem | 0.383 | [0.285, 0.482] | 0.500 | 0.319 | 6 ms |

`turkish-rag-eval report` does not stop at the table — it names the cheapest
configuration the data cannot separate from the best:

```
Recommended: sentence + hybrid_rrf
  0.558 ndcg@10 against 0.613 for hierarchical + hybrid_rrf, a gap of 0.056
  that a paired bootstrap over 58 shared queries cannot distinguish from zero.
  It answers in 27 ms at P95 against 37 ms.
```

Comparisons are **paired**: both systems answered the same queries, so
resampling per-query differences removes the variance of some queries being
harder than others. It matters, and this data shows it in the sharpest way.
`hierarchical + hybrid_rrf` beats `fixed + hybrid_rrf` by 0.097 and beats
`sentence + bm25_stem5` by 0.104 — yet only the **larger** gap clears zero,
because its per-query differences are so much steadier (sd 0.34 against 0.40).
Ranking by the gap alone gets this backwards.

Hierarchical chunking only helps the dense retriever, and fixed-size chunking
— the most common default — lost on every retriever. More in the
[design notes](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#what-else-the-numbers-say).

### Generation: groundedness

The R above feeds a G: the 58 questions answered from the top 5
`hierarchical + bm25_stem5` passages by `groq:openai/gpt-oss-120b`, judged by
a model from another family, `nvidia:nvidia/nemotron-3-super-120b-a12b`, at
temperature 0 on `2026-09-30`. The gold span was among the passages for
33 of 58 questions, and the two groups are scored apart:

| Span retrieved (33) | |
|---|---|
| answered when the span was retrieved | 93.9% |
| answer rests on the passages | 96.8% |
| answer conveys the gold span | 96.8% |

| Span not retrieved (25) | |
|---|---|
| declined when the span was not retrieved | 80.0% |
| answered anyway | 20.0% |
| of those, answer is right | 60.0% |
| hallucinated (not in the passages) | 4.0% |

The generator mostly declines when the evidence is missing, and when it does
answer, the answer is usually stated in some other retrieved passage — one of
25 is not. The first scoring rule called every answer without the span a
hallucination and reported 20%; reading the five by hand showed four stated in
a retrieved passage, so answers without the span now go to the judge too
([why](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#groundedness)).
16 verdicts were checked by hand; 14 agreed. Of the other two, the judge was
too strict once (an answer inverted "not recommended unless below 7 g/dl") and
too lenient once (it accepted "hyperuricaemia" as the cause of gout).

Four more generators on the same questions, passages and judge
(`2026-10-12`), sorted by the hallucination rate:

| generator | answered (span retrieved, 33) | rests on the passages | conveys the span | declined (span missing, 25) | hallucinated |
|---|---|---|---|---|---|
| `groq:qwen/qwen3.8-27b` | 97.0% | 100.0% | 96.9% | 80.0% | 0.0% (0/25) |
| `groq:openai/gpt-oss-120b` (above) | 93.9% | 96.8% | 96.8% | 80.0% | 4.0% (1/25) |
| `nvidia:moonshotai/kimi-k3` | 100.0% | 97.0% | 97.0% | 76.0% | 4.0% (1/25) |
| `nvidia:z-ai/glm-5.3-flash` | 100.0% | 97.0% | 100.0% | 76.0% | 4.0% (1/25) |
| `groq:openai/gpt-oss-20b` | 90.9% | 96.7% | 93.3% | 76.0% | 8.0% (2/25) |

Every model hallucinated on 0 to 2 of the 25 missing-span questions. At this
size one answer is 4 points, so the table does not rank them; the same run on
the 242 drafted questions (100+ missing-span cases per model) is what would.
`z-ai/glm-5.3-flash` reasons before answering and NVIDIA's endpoint has no
switch for that, so it runs with a 16k token budget (at 4k some answers came
back empty). Per-model summaries sit in
[`results/groundedness/`](https://github.com/RizgarOzan/turkish-rag-eval/tree/main/results/groundedness).

This row depends on hosted APIs and cannot be reproduced offline; the models
may change behind the same name. Only the rates are committed
([`results/groundedness.json`](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/results/groundedness.json)) —
the free API terms do not allow redistributing raw model output.

To rerun it with other models, name them `groq:`, `nvidia:` or `openai:`
(a bare name goes to Anthropic). Keys can sit in a `.env` file (gitignored) in
the directory you run from; `openai:` follows `OPENAI_BASE_URL`, so any OpenAI-compatible
endpoint, a local router included, works.

## Leaderboard

Dense `nDCG@10` per chunking strategy, the hybrid (dense + stemmed BM25, RRF)
on hierarchical chunks, and the cost of each model on one 16-thread CPU,
measured 2026-09-18. Stemmed BM25 alone scores 0.476 / 0.510 / 0.494.
`turkish-rag-eval leaderboard` rebuilds this from `results/models/`.

| Model | Params | Turkish-only | Dense fixed | Dense sentence | Dense hierarchical | Hybrid hierarchical | Embed corpus | Query P95 |
|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 (default) | 118 M | no | 0.446 | 0.461 | 0.501 | 0.613 | 2 min | 23 ms |
| [emrecan/bert-base-turkish-cased-mean-nli-stsb-tr](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr) | 111 M | yes | 0.408 | 0.431 | 0.497 | 0.654 | 4 min | 43 ms |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 118 M | no | 0.654 | 0.644 | 0.642 | 0.639 | 3 min | 22 ms |
| [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 278 M | no | 0.631 | 0.677 | 0.668 | 0.642 | 10 min | 49 ms |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 568 M | no | 0.766 | 0.767 | — | — | > 45 min | — |
| **[newmindai/Mursit-Large-TR-Retrieval](https://huggingface.co/newmindai/Mursit-Large-TR-Retrieval)** | 404 M | yes | **0.746** | **0.740** | **0.781** | 0.672 | 35 min | 214 ms |

The two Turkish-only models are the most-downloaded Turkish entries in the
Hugging Face `sentence-similarity` category. `bge-m3` was stopped after 45
minutes, before the hierarchical chunks; its two numbers come from that
partial run. `google/embeddinggemma-300m` is gated behind a licence click and
was not run. Per-query results for every completed model are in
`results/models/`.

> The timing columns come from one machine before v0.1.0, which is why they
> are comparable with each other and not with the main table above. The four
> non-default models were re-run on 2026-10-18 with the stable tie-break and
> the sentence splitter fix: no `dense` figure moved, and in `Hybrid
> hierarchical` e5-base went 0.648 → 0.642 and Mursit 0.673 → 0.672 (the
> default row's went 0.607 → 0.613 earlier). The "+0.006" this note used to
> predict was wrong; the tie-break moves a fused score either way.

**The top of this table is settled; the middle is not.** Dense hierarchical,
with 95% bootstrap intervals over the 58 queries: MiniLM 0.501 [0.396, 0.606],
emrecan 0.497 [0.394, 0.601], e5-small 0.642 [0.542, 0.738], e5-base 0.668
[0.568, 0.765], Mursit 0.781 [0.701, 0.856]. The intervals overlap, but paired
over the same queries Mursit beats the runner-up e5-base by +0.113
[+0.029, +0.202]. The two E5 models cannot be told apart (+0.026
[-0.039, +0.094]), and neither can MiniLM and emrecan (+0.004
[-0.123, +0.135]).

**Hybrid fusion only pays for a weak dense model.** RRF lifts the small
default by +0.112 but pulls Mursit down from 0.781 to 0.672, a paired loss of
-0.109 [-0.194, -0.030]; for the two E5 models it makes no measurable
difference. "Turkish-only" is not enough either: the `emrecan` model was
trained for sentence similarity and truncates input at 75 tokens.

To submit a model, open a pull request adding `results/models/<org>__<name>/`
(`turkish-rag-eval run --model <org>/<name>`, then `leaderboard --check`).
CI re-derives every metric from the per-query relevance arrays committed
beside it, and checks the harness version and corpus fingerprint.

### Reranking

A cross-encoder reads the question and a passage together, so it ranks far
better than an embedding comparison and far too slowly to run over all 2 103
chunks; here it reorders the hybrid retriever's top 20.
`run --reranker BAAI/bge-reranker-v2-m3` on top of `multilingual-e5-small`,
58 questions, nDCG@10, one 16-thread CPU:

| Chunking | hybrid_rrf | + reranker | gain (95% paired bootstrap) | query P95 |
|---|---|---|---|---|
| fixed | 0.611 | 0.809 | +0.198 [+0.119, +0.282] | 35 ms → 12.8 s |
| sentence | 0.640 | 0.779 | +0.138 [+0.069, +0.209] | 31 ms → 13.3 s |
| hierarchical | 0.639 | 0.774 | +0.135 [+0.074, +0.201] | 32 ms → 13.0 s |

Every gain is clear of zero. On fixed chunks the reranked small model
(0.809) passes the best single model in the table above (Mursit,
0.781), and on the other two it comes within 0.01 of it — for roughly
360× the query latency on CPU, since every question now sends 20
passages through a 568 M parameter model. A GPU changes that arithmetic; it
has not been measured here. Per-query results are in `results/rerank/`.

## Stress test

People do not type the way a gold set is written. `turkish-rag-eval stress`
rewrites every question three ways and scores each one as typed and after
`normalize.QueryNormalizer`:

- **no Turkish letters**: ç ğ ı ö ş ü typed as c g i o s u, as on a keyboard without them;
- **caps**: caps lock, upper-cased the Turkish way;
- **naive lower**: the caps question run through Python's `str.lower()`, which
  turns "İ" into "i" plus a combining dot.

Dense retrieval on hierarchical chunks, nDCG@10, 58 questions; each cell is
*as typed → normalised*:

| Model | as written | no Turkish letters | caps | naive lower |
|---|---|---|---|---|
| MiniLM (default) | 0.501 | 0.259 → 0.481 | 0.051 → 0.496 | 0.278 → 0.454 |
| emrecan (Turkish) | 0.497 | 0.429 → 0.483 | 0.133 → 0.529 | 0.337 → 0.524 |
| e5-small | 0.642 | 0.471 → 0.631 | 0.389 → 0.664 | 0.440 → 0.623 |
| e5-base | 0.668 | 0.523 → 0.651 | 0.436 → 0.666 | 0.571 → 0.639 |
| Mursit (Turkish) | 0.781 | 0.717 → 0.757 | 0.639 → 0.747 | 0.636 → 0.745 |
| stemmed BM25 | 0.494 | 0.288 → 0.477 | 0.494 → 0.494 | 0.254 → 0.481 |

Stemmed BM25 shrugs off caps because its tokenizer already lowercases the
Turkish way; every embedding model loses ground, and the default one almost
everything. Summed over the five models and the three ways of mistyping,
the normaliser wins back 94% of the nDCG@10 that dense retrieval lost, and
93% of what BM25 lost.

The normaliser restores letters only from words that occur in the
corpus, never touches a word that already has a Turkish letter, an acronym
("OKB") or a two-letter word ("mi" and "mı" are both particles), and leaves
all 58 correctly typed questions exactly as they were — every "as written"
score is unchanged by it. An earlier version recased "OKB" to "Okb" and
"mi" to "mı"; the stress run caught it (the default model scored 0.490 on
the *unchanged* questions instead of 0.501) and both are now tests.

These are single runs without intervals; the large drops do not need them,
the small differences between normalised columns should not be read as a
ranking. `results/stress/` holds per-query relevance for every cell.

```bash
turkish-rag-eval stress --model intfloat/multilingual-e5-small
turkish-rag-eval stress --sparse-only        # BM25 only, no model download
turkish-rag-eval run --normalize             # the same fix in front of a normal run
```

## Why not an existing benchmark?

MTEB-style retrieval benchmarks, TR-MTEB included, score an embedding model on
passages that are already split. They answer "which model?", not "which
chunker, is Turkish stemming worth it, does a hybrid help, and what does each
cost on a CPU?". This harness keeps the articles whole, lets every chunker cut
them its own way, and judges each chunk by the answer span, so pipeline
choices are compared on the same labels. For a model-only comparison the same
data exports to the BEIR layout MTEB reads (`turkish-rag-eval export-hf`),
published as
[RizgarOzan/turkish-rag-eval](https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval)
(the 58 human questions, plus all 296 under the `full-*` configs, each marked
`human` or `llm-draft`); adding it to MTEB is proposed in
[embeddings-benchmark/mteb#5536](https://github.com/embeddings-benchmark/mteb/issues/5536).

## Your own corpus

The harness is not tied to its own articles. Point it at a folder of `.txt` or
`.md` files, a BEIR directory, or a JSON corpus:

```bash
turkish-rag-eval run --corpus ./belgelerim --gold ./sorular.json \
                     --retrievers bm25_stem5 bm25_nostem
```

A BM25-only run needs no model download and no torch — enough to answer "which
chunker, and is stemming worth it on my documents" from the base install.

**No labelled questions?** That is the real wall, and the reason most
benchmarks only ever measure themselves. `bootstrap` drafts a starting point (the name means drafting a gold set here,
not the statistical resampling above):

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

## Running it

Python 3.10+. CPU only — no GPU anywhere in this project.

```bash
pip install turkish-rag-eval          # metrics, BM25, gold-set tooling
pip install 'turkish-rag-eval[all]'   # + dense retrieval, charts, LLM commands
```

| Command | What it does |
|---|---|
| `run` | every chunking × retriever combination; writes `results/` |
| `ask` | one question, every retriever's top chunks side by side ([example](#one-question-at-a-time)) |
| `serve` | BM25 or hybrid search over HTTP ([load test](#serving-it)) |
| `report` | intervals, paired comparisons, and a recommendation |
| `charts` | the two figures above, light and dark |
| `leaderboard` | rebuild the model table; `--check` verifies every entry |
| `abstain` | coverage / selective-accuracy curve for the best configuration ([details](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#abstention)) |
| `groundedness` | score the generation half against the gold spans ([details](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#groundedness); [results](#generation-groundedness)) |
| `bootstrap` | draft a gold set for your own corpus |
| `agreement` | inter-annotator agreement over the gold set |
| `fetch-corpus` | `--verify` checks the committed snapshot against its lock; without it, refetch live Wikipedia and report drift; `--include-drafts` for [all 300 questions](#scoring-all-300) |
| `export-hf` | the BEIR layout MTEB reads |
| `validate` | every gold file's invariants |

From a checkout:

```bash
git clone https://github.com/RizgarOzan/turkish-rag-eval
cd turkish-rag-eval
pip install -e '.[all]'
turkish-rag-eval fetch-corpus --verify   # the pinned data/raw/corpus.json ships in the repo
turkish-rag-eval run              # results/summary.json + per-query files
python -m pytest -q
```

`run --model <name>` swaps the embedding model; any sentence-transformers
model works. Results for a non-default model go to
`results/models/<org>__<name>/`, so the main table is never overwritten.

`run --embedding-cache <dir>` keeps the chunk embeddings and reuses them while
model and chunk text stay the same. With multilingual-e5-small on CPU
the three dense indexes took 91, 75 and 81 s on the first run and 0.1 s or
less on the second, with identical scores; `dense_index_seconds` in the
results still reports the original encoding time. The large model is where it
pays off: a dense-only Mursit-Large-TR-Retrieval run took 48 min cold (1132,
882 and 797 s to encode the fixed, sentence and hierarchical indexes) and
76 s warm, with the same nDCG@10 as `results/models/` and a 23 MB cache.

`stress --embedding-cache <dir>` reads the same cache: it scores hierarchical
chunks, and the two commands share hierarchical cache entries either way. For
e5-small the stress run went from 139 s to 44 s (the 64 s index encode is
skipped; every query variant is still encoded), with every cell identical to
`results/stress/`.

Dynamic int8 quantization helps the large model and hurts the small one.
`docs/measure_int8.py <model>` runs the dense retriever with torch's int8 Linear
layers on CPU and writes to `results/quantize/`; the fp32 side is the
`results/models/` run (95% paired bootstrap on nDCG@10):

| Model | Chunking | nDCG@10 fp32 → int8 | change | encode | query P95 |
|---|---|---|---|---|---|
| e5-small | fixed | 0.654 → 0.539 | -0.115 [-0.175, -0.060] | 69 s → 36 s | 22 ms → 25 ms |
| e5-small | sentence | 0.644 → 0.522 | -0.121 [-0.193, -0.052] | 60 s → 30 s | 22 ms → 21 ms |
| e5-small | hierarchical | 0.642 → 0.571 | -0.071 [-0.133, -0.014] | 64 s → 33 s | 21 ms → 20 ms |
| Mursit-Large-TR-Retrieval | fixed | 0.746 → 0.700 | -0.045 [-0.096, +0.002] | 733 s → 280 s | 187 ms → 61 ms |
| Mursit-Large-TR-Retrieval | sentence | 0.740 → 0.722 | -0.018 [-0.066, +0.026] | 635 s → 245 s | 176 ms → 55 ms |
| Mursit-Large-TR-Retrieval | hierarchical | 0.781 → 0.771 | -0.010 [-0.063, +0.045] | 689 s → 277 s | 178 ms → 77 ms |

e5-small loses 0.07 to 0.12, every loss clear of zero, and its queries get no
faster. Mursit-Large-TR-Retrieval encodes about 2.5 times and answers 2.3 to 3.2
times faster with no loss the bootstrap can tell from zero, though fixed comes
close. The two sides were timed on different days, so read the speedups as
rough. torch marks this quantization API deprecated, which is why it is a
script and not a `run` flag.

### One question at a time

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

### Serving it

`serve` puts one retriever behind HTTP: stemmed BM25 by default, or with
`--model` the hybrid of that model and BM25. It is the standard library's
threaded server, so the base install needs nothing new.

```text
$ turkish-rag-eval serve
bm25_stem5 over 2103 hierarchical chunks, index built in 0.1 s
listening on http://127.0.0.1:8000

GET /search?q=Bacak damarında pıhtı kollarda da olur mu?&k=5
{"query": "...", "retriever": "bm25_stem5", "results": [{"rank": 1, "score": ..., "chunk_id": ..., "title": ..., "heading_path": ..., "url": ..., "text": ...}, ...]}
```

`docs/load_test.py` sends the 58 gold questions at a running server, 580
requests per level at k=5, and writes `results/serve/`. BM25 on one desktop
CPU, client on the same machine:

| Clients | P50 | P95 | P99 | req/s | errors |
|---|---|---|---|---|---|
| 1 | 4.5 ms | 30.3 ms | 32.5 ms | 97 | 0 |
| 4 | 12.5 ms | 33.6 ms | 38.2 ms | 256 | 0 |
| 16 | 59.3 ms | 80.3 ms | 92.4 ms | 264 | 0 |

Throughput stops growing past four clients, because BM25 scoring holds
Python's GIL. The listen queue is raised from the standard library's 5 to 128:
with 5, P99 at 16 clients was 570 ms, as connections past the queue waited out a TCP
retry.

The same run against `serve --model intfloat/multilingual-e5-small`, the hybrid
of that model and BM25 (index built in 152 s, no embedding cache), where every
request also encodes the query:

| Clients | P50 | P95 | P99 | req/s | errors |
|---|---|---|---|---|---|
| 1 | 40.5 ms | 57.7 ms | 89.4 ms | 23 | 0 |
| 4 | 111.9 ms | 144.1 ms | 200.4 ms | 35 | 0 |
| 16 | 498.1 ms | 629.8 ms | 723.7 ms | 33 | 0 |

Query encoding on the CPU is the bottleneck: it costs about 35 ms over BM25 per
request and caps throughput near 35 req/s, so sixteen clients just queue. Rows
come from `results/serve/hybrid_rrf_e5-small.json`. The Docker image is not
built yet.

CI also guards the speed of the BM25 path. `docs/speed_gate.py` times chunking,
the index build and the 58 searches against a fixed Python loop on the same
runner, and fails when a stage takes more than twice its ratio in
`results/speed/baseline.json`. The ratio moves up to ~15% between runs and
~1.6x between Python 3.13 and 3.14, so the gate is coarse on purpose:
re-packing each section once per chunk (chunking 3.5x slower) and rebuilding
the index per query (search 64x slower) both turned it red, while a tokeniser
made 1.4x slower passed.

## Gold set

| Files | Questions | Labelled by | In the results above |
|---|---|---|---|
| `data/eval/gold.json` (health) | 58 | one human | yes |
| `data/eval/contrib/llm-draft-*.json` (history, geography, astronomy, biology, computing) | 242 | two independent LLM passes | not yet |

The set reached its planned 300 questions on 2026-09-26. Questions a model
drafted are marked `"source": "llm-draft"`. A second model then picked its own
answer span for each one without seeing the first label
(`second_annotation`). Two spans agree when one contains the other or their
token F1 is at least 0.5; agreed items get `"review": "agreed"`, the rest get
`"needs-human"` and are never loaded. Batches 1 and 2 were 30 of 30 agreed,
batch 3 was 29 of 30: for "what are several ribosomes working on one mRNA
called?" the passes picked two different sentences that both name polysomes,
so that question waits for a person. Batch 4 was 27 of 30: the second pass
named the other claimant to the Hungarian throne, took the sentence beside the
lysozyme result instead of the result itself, and answered "cross compilers"
with the bare term where the first took its definition. Batch 5 was 30 of 30,
and every pair is a containment pair: 9 identical, 15 differing only by a
trailing full stop. One of its answers, Robert W. Holley's 1968 Nobel Prize,
was cut to its last clause on 2026-09-27 and labelled again by the second
pass: the sentence splitter breaks after "W.", so no hierarchical chunk held
the whole sentence ([#17](https://github.com/RizgarOzan/turkish-rag-eval/issues/17)). Batch 6 was 30 of 30 on reworded questions whose answers
often run to two sentences; six first-pass spans were cut to one sentence
before merging, because the two-sentence version fitted inside no chunk and
so could never be retrieved. Batch 7 was 30 of 30 again, with 25 identical
spans; one first-pass span was cut to its clause for the same reason. Like
batch 5 its questions mostly point at a single sentence, so its high
agreement says little about harder questions. Batch 8, the last 32, was
written to be harder: 17 first-pass answers ran to two or more sentences.
Seven of those crossed a sentence- or hierarchical-chunk boundary and were
cut, leaving 12 multi-sentence answers. It was still 32 of 32, 29 identical:
two LLMs reading the same article pick the same sentences even when the
answer is long, which is one more reason these drafts need a human pass.

Drafts stay out of every number above until a re-run says otherwise:
`load_gold()` skips them unless called with `include_drafts=True`. Synthetic
test questions are common practice as long as they are declared and their
agreement is measured. This section is that declaration.

The corpus is 54 Turkish Wikipedia articles, 1.09 M characters; 27 of them
answer at least one question and the other 27 are distractors from the same
domain.

| Batch | Questions | Identical | Mean IoU | Mean token F1 | Cohen's κ, fixed / sentence / hierarchical |
|---|---|---|---|---|---|
| 1 — 2026-09-18 (Malazgirt, Kapadokya, Mars, Mitokondri, Linux) | 30 | 16 | 0.816 | 0.878 | 1.00 / 1.00 / 1.00 |
| 2 — 2026-09-20 (İstanbul'un Fethi, Ağrı Dağı, Jüpiter, Fotosentez, İnternet)¹ | 30 | 4 | 0.427 | 0.549 | 0.94 / 0.99 / 0.99 |
| 3 — 2026-09-21 (Çaldıran Muharebesi, Tuz Gölü, Satürn, Ribozom, Unix) | 30 | 18 | 0.829 | 0.872 | 0.92 / 0.96 / 0.96 |
| 4 — 2026-09-24 (Mohaç Muharebesi, Kızılırmak, Venüs, Enzim, Derleyici) | 30 | 15 | 0.747 | 0.792 | 0.96 / 0.96 / 0.96 |
| 5 — 2026-09-25 (Kösedağ Muharebesi, Uludağ, Neptün, RNA, İşletim sistemi) | 30 | 24 | 0.922 | 0.943 | 1.00 / 1.00 / 1.00 |
| 6 — 2026-09-25 (Preveze Deniz Muharebesi, Erciyes, Uranüs, Hemoglobin, Veritabanı) | 30 | 20 | 0.869 | 0.901 | 0.93 / 0.90 / 0.87 |
| 7 — 2026-09-26 (Ankara Muharebesi, Van Gölü, Merkür, DNA, World Wide Web) | 30 | 25 | 0.944 | 0.960 | 0.93 / 0.93 / 0.97 |
| 8 — 2026-09-26 (Niğbolu Muharebesi, Fırat, Ay, Protein, Yapay zekâ) | 32 | 29 | 0.964 | 0.975 | 0.94 / 0.97 / 1.00 |
| **All drafts** | 242 | 151 | 0.816 | 0.860 | 0.95 / 0.97 / 0.97 |

To measure agreement over the drafts eligible for loading, run
`turkish-rag-eval agreement --include-drafts`. Embedded second labels are
expanded into ordinary second annotations before grouping. Without the flag,
the command still loads only the hand-labelled set. The table above measures
all draft files, including unresolved `needs-human` questions; the CLI keeps
`load_gold`'s review filters. With the current files it therefore reports
238 questions and mean token F1 0.872, versus 242 and 0.860 for all drafts.
Pass `--corpus data/raw/corpus-full.json` for the chunk-level kappa columns;
an article missing from the corpus given gives `n/a` there.

¹ Re-measured 2026-09-26: the Turkish Wikipedia article *Jüpiter* was rewritten that day to correct errors, and five of its spans no longer matched the live text. Three changed only in wording (a comma, "30,003" → "30" seconds, "dört uydu" → "uydular") and were edited in both labels; two changed in substance — the 40,000 km mantle thickness is gone and the Great Red Spot went from "at least 400 years" to "recorded since 1831" — so those two questions were rewritten and labelled again by both passes. The row was 0.419 / 0.540 / 0.96 / 1.00 / 1.00 before.

The passes disagree about how much of a sentence to take, rather than where
the answer is. Two LLMs tend to pick the same sentence, so read this as a
sanity check rather than human agreement. Full discussion in the
[design notes](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#agreement-between-the-two-passes).

### Scoring all 300

The drafted questions point at 40 articles outside the health snapshot, so
they get a corpus of their own: the 54 health articles plus those 40, 1.89 M
characters, pinned by `data/corpus-full.lock.json`. The published snapshot and
its lock stay as they are.

```bash
turkish-rag-eval fetch-corpus --verify --include-drafts   # data/raw/corpus-full.json, also committed
turkish-rag-eval run --include-drafts            # results/full/<model>/
```

Measured 2026-10-18 (re-run after the sentence splitter fix in #29) on 296 questions (the 4 `needs-human` drafts never load),
hierarchical chunks, nDCG@10:

| Retriever | 58 human questions | 296 questions (238 drafted) |
|---|---|---|
| bm25_stem5 | 0.494 | 0.559 |
| MiniLM (default), dense | 0.501 | 0.408 |
| MiniLM (default), hybrid_rrf | 0.613 | 0.576 |
| multilingual-e5-small, dense | 0.642 | 0.565 |
| multilingual-e5-small, hybrid_rrf | 0.639 | 0.644 |
| multilingual-e5-base, dense | 0.668 | 0.649 |
| multilingual-e5-base, hybrid_rrf | 0.642 | 0.667 |
| Mursit-Large-TR-Retrieval, dense | 0.781 | 0.611 |
| Mursit-Large-TR-Retrieval, hybrid_rrf | 0.672 | 0.662 |

On the wider set stemmed BM25 gets stronger and every dense model weaker, so
fusing the two now helps all four models, where on the 58 health questions it
cost the E5 models and Mursit. Mursit drops the most, from 0.781 to 0.611, and
falls below e5-base, so its lead on the human set does not carry over to the
drafted questions yet. Those questions were written by an LLM from one sentence
each; whether that style suits the smaller models better is open until people
have checked them. These rows are not in the tables above: most of the
questions have not been checked by a person yet. The run also names every
question no chunk can answer; a span split by a chunk boundary is the usual
reason. Here that is 17 questions with fixed-size chunks and none with
sentence or hierarchical chunks (the one hierarchical miss was fixed on
2026-09-27, see batch 5 above). Results for all three chunkers are in
[`results/full/`](https://github.com/RizgarOzan/turkish-rag-eval/tree/main/results/full).

## Status

v0.2.0, [on PyPI](https://pypi.org/project/turkish-rag-eval/).

- **Works:** the retrieval harness, [`groundedness`](#generation-groundedness) on the 58 human questions, `report`, [`serve`](#serving-it), `leaderboard --check`, `bootstrap`,
  `agreement`, the Hugging Face export, scoring [all 300 questions](#scoring-all-300)
  on a pinned corpus, and CI that validates every contributed question against
  live Wikipedia.
- **In progress:** human review of the gold set. All 300 planned questions are
  in (58 human, 242 LLM-drafted); the drafts join the results once people have
  reviewed them. Issues labelled
  [`human-review`](https://github.com/RizgarOzan/turkish-rag-eval/labels/human-review)
  hand out about ten drafts each; `turkish-rag-eval validate` counts how many
  are done.
- **Not yet:** EmbeddingGemma, and `groundedness` on the 242 drafted questions.

## Limits

- **58 queries is a small set.** The intervals above are the honest width of
  that; most of the table's ordering is not resolved by this much data. They
  replace the eyeballed rule of thumb this project used to carry — "treat
  differences under roughly 0.05 nDCG as noise" — which was both too strict
  for paired comparisons and too loose for unpaired ones.
- **One annotator for the human set.** The 58 health questions are
  single-annotated, so they have no agreement figure. The 242 drafted questions
  are double-labelled, but by two LLM passes rather than by two people.
- **The corpus is Wikipedia, and so is much of the training data.** Every
  embedding model ranked here was almost certainly trained on Turkish
  Wikipedia. Absolute scores are therefore optimistic; the *comparison*
  between pipeline choices on the same corpus is what this measures. A
  non-Wikipedia domain is the most valuable thing a contributor could add.
- **Six embedding models, one corpus.** `bge-m3` is a partial run and
  EmbeddingGemma is missing. `abstain` uses only the default model.
- **Encyclopaedic text, not clinical text.** Nothing here transfers to a
  clinical setting without re-measurement. No patient data is used anywhere,
  and nothing here is a medical device.
- **The generation half is one generator, one judge, 58 questions.**
  [Groundedness](#generation-groundedness) was measured once, through hosted
  APIs, on the human set only; the judge was checked by hand on 16 verdicts.
- **The embedding model is pinned by name, not by revision.**

## Contribute

The first two limits shrink with every contributor. Adding questions needs no
ML background — pick a Turkish Wikipedia article, write 5–10 paraphrased
questions, and open a pull request with one JSON file. A validator checks each
file against Wikipedia in CI. Smaller still: each
[`human-review`](https://github.com/RizgarOzan/turkish-rag-eval/labels/human-review)
issue lists ten LLM-drafted questions to check against their article, and
marking them `ok`, `fixed` or `rejected` is the whole job. See [CONTRIBUTING.md](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/CONTRIBUTING.md) (Türkçe
açıklama dahil) and the
[good first issues](https://github.com/RizgarOzan/turkish-rag-eval/issues?q=is%3Aopen+label%3A%22good+first+issue%22).

Submitting an embedding model is one command and a pull request — see
[Leaderboard](#leaderboard).

Contributors so far: [@KonstantinBurkin](https://github.com/KonstantinBurkin)
(the `agreement` script, [#14](https://github.com/RizgarOzan/turkish-rag-eval/pull/14))
and [@eyupcodes](https://github.com/eyupcodes) (sentence splitting after
initials, [#29](https://github.com/RizgarOzan/turkish-rag-eval/pull/29);
`agreement --include-drafts`, [#30](https://github.com/RizgarOzan/turkish-rag-eval/pull/30)).

## Data

Turkish Wikipedia, CC BY-SA 4.0. See [NOTICE.md](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/NOTICE.md).

## Licence

Code MIT ([LICENSE](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/LICENSE)); data under `data/` CC BY-SA 4.0
([NOTICE.md](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/NOTICE.md)).
