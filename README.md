# turkish-rag-eval

[![validate](https://github.com/RizgarOzan/turkish-rag-eval/actions/workflows/validate.yml/badge.svg)](https://github.com/RizgarOzan/turkish-rag-eval/actions/workflows/validate.yml)
[![licence: MIT + CC BY-SA 4.0](https://img.shields.io/badge/licence-MIT%20%2B%20CC%20BY--SA%204.0-blue)](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/NOTICE.md)
[![dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-RizgarOzan%2Fturkish--rag--eval-yellow)](https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval)

**Which parts of a RAG pipeline actually earn their cost on an agglutinative
language?** Three chunking strategies × four retrievers × six embedding
models, measured on a hand-labelled Turkish gold set, with confidence
intervals and a cost column — then pointed at your own documents.

```bash
pip install git+https://github.com/RizgarOzan/turkish-rag-eval   # not on PyPI yet
turkish-rag-eval run --corpus ./belgelerim --gold ./sorular.json
turkish-rag-eval report
```

## Four findings

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

Longer write-up of the first result:
[English](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/blog/2026-09-19-bm25-turkish-en.md) ·
[Türkçe](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/blog/2026-09-19-bm25-turkish-tr.md).

**Contents:** [Results](#results) · [Leaderboard](#leaderboard) ·
[Why not an existing benchmark?](#why-not-an-existing-benchmark) ·
[Your own corpus](#your-own-corpus) · [Running it](#running-it) ·
[Gold set](#gold-set) · [Status](#status) · [Limits](#limits) · [Contribute](#contribute) ·
[Design notes](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md)

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

## Leaderboard

Dense `nDCG@10` per chunking strategy, the hybrid (dense + stemmed BM25, RRF)
on hierarchical chunks, and the cost of each model on one 16-thread CPU,
measured 2026-09-18. Stemmed BM25 alone scores 0.476 / 0.510 / 0.494.
`turkish-rag-eval leaderboard` rebuilds this from `results/models/`.

| Model | Params | Turkish-only | Dense fixed | Dense sentence | Dense hierarchical | Hybrid hierarchical | Embed corpus | Query P95 |
|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 (default) | 118 M | no | 0.446 | 0.461 | 0.501 | 0.607 | 2 min | 23 ms |
| [emrecan/bert-base-turkish-cased-mean-nli-stsb-tr](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr) | 111 M | yes | 0.408 | 0.431 | 0.497 | 0.654 | 4 min | 43 ms |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 118 M | no | 0.654 | 0.644 | 0.642 | 0.639 | 3 min | 22 ms |
| [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 278 M | no | 0.631 | 0.677 | 0.668 | 0.648 | 10 min | 49 ms |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 568 M | no | 0.766 | 0.767 | — | — | > 45 min | — |
| **[newmindai/Mursit-Large-TR-Retrieval](https://huggingface.co/newmindai/Mursit-Large-TR-Retrieval)** | 404 M | yes | **0.746** | **0.740** | **0.781** | 0.673 | 35 min | 214 ms |

The two Turkish-only models are the most-downloaded Turkish entries in the
Hugging Face `sentence-similarity` category. `bge-m3` was stopped after 45
minutes, before the hierarchical chunks; its two numbers come from that
partial run. `google/embeddinggemma-300m` is gated behind a licence click and
was not run. Per-query results for every completed model are in
`results/models/`.

> These five rows were measured together on one machine before v0.1.0, which
> is why the timing columns are comparable with each other and not with the
> main table above. Their `dense` columns are unaffected by the stable
> tie-break, but each `Hybrid hierarchical` figure will move by roughly +0.006
> when the model is re-run — the default row's went 0.607 → 0.613.
> `turkish-rag-eval leaderboard --check` reports them as missing provenance
> until then.

**Hybrid fusion only pays for a weak dense model.** RRF lifts the small
default by +0.106 but pulls Mursit down from 0.781 to 0.673. "Turkish-only"
is not enough either: the `emrecan` model was trained for sentence similarity
and truncates input at 75 tokens. These leaderboard rows do not have
confidence intervals yet.

To submit a model, open a pull request adding `results/models/<org>__<name>/`
(`turkish-rag-eval run --model <org>/<name>`, then `leaderboard --check`).
CI re-derives every metric from the per-query relevance arrays committed
beside it, and checks the harness version and corpus fingerprint.

## Why not an existing benchmark?

MTEB-style retrieval benchmarks, TR-MTEB included, score an embedding model on
passages that are already split. They answer "which model?", not "which
chunker, is Turkish stemming worth it, does a hybrid help, and what does each
cost on a CPU?". This harness keeps the articles whole, lets every chunker cut
them its own way, and judges each chunk by the answer span, so pipeline
choices are compared on the same labels. For a model-only comparison the same
data exports to the BEIR layout MTEB reads (`turkish-rag-eval export-hf`),
published as
[RizgarOzan/turkish-rag-eval](https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval);
adding it to MTEB is proposed in
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
pip install git+https://github.com/RizgarOzan/turkish-rag-eval                       # metrics, BM25, gold-set tooling
pip install 'turkish-rag-eval[all] @ git+https://github.com/RizgarOzan/turkish-rag-eval'  # + dense retrieval, charts, LLM commands
```

| Command | What it does |
|---|---|
| `run` | every chunking × retriever combination; writes `results/` |
| `report` | intervals, paired comparisons, and a recommendation |
| `charts` | the two figures above, light and dark |
| `leaderboard` | rebuild the model table; `--check` verifies every entry |
| `abstain` | coverage / selective-accuracy curve for the best configuration ([details](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#abstention)) |
| `groundedness` | score the generation half against the gold spans ([details](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#groundedness); no results committed yet) |
| `bootstrap` | draft a gold set for your own corpus |
| `agreement` | inter-annotator agreement over the gold set |
| `fetch-corpus` | download the Wikipedia snapshot; `--verify` checks the lock |
| `export-hf` | the BEIR layout MTEB reads |
| `validate` | every gold file's invariants |

From a checkout:

```bash
git clone https://github.com/RizgarOzan/turkish-rag-eval
cd turkish-rag-eval
pip install -e '.[all]'
turkish-rag-eval fetch-corpus     # rebuilds data/raw/corpus.json
turkish-rag-eval run              # results/summary.json + per-query files
python -m pytest -q
```

`run --model <name>` swaps the embedding model; any sentence-transformers
model works. Results for a non-default model go to
`results/models/<org>__<name>/`, so the main table is never overwritten.

## Gold set

| Files | Questions | Labelled by | In the results above |
|---|---|---|---|
| `data/eval/gold.json` (health) | 58 | one human | yes |
| `data/eval/contrib/llm-draft-*.json` (history, geography, astronomy, biology, computing) | 120 | two independent LLM passes | not yet |

The set is growing toward 300 questions across more domains. Questions a model
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
with the bare term where the first took its definition.

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
| 2 — 2026-09-20 (İstanbul'un Fethi, Ağrı Dağı, Jüpiter, Fotosentez, İnternet) | 30 | 4 | 0.419 | 0.540 | 0.96 / 1.00 / 1.00 |
| 3 — 2026-09-21 (Çaldıran Muharebesi, Tuz Gölü, Satürn, Ribozom, Unix) | 30 | 18 | 0.829 | 0.872 | 0.92 / 0.96 / 0.96 |
| 4 — 2026-09-24 (Mohaç Muharebesi, Kızılırmak, Venüs, Enzim, Derleyici) | 30 | 15 | 0.747 | 0.792 | 0.96 / 0.96 / 0.96 |
| **All drafts** | 120 | 53 | 0.703 | 0.771 | 0.96 / 0.98 / 0.98 |

The passes disagree about how much of a sentence to take, rather than where
the answer is. Two LLMs tend to pick the same sentence, so read this as a
sanity check rather than human agreement. Full discussion in the
[design notes](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#agreement-between-the-two-passes).

## Status

v0.1.0, tagged but not on PyPI yet — install from GitHub as above.

- **Works:** the retrieval harness, `report`, `leaderboard --check`, `bootstrap`,
  `agreement`, the Hugging Face export, and CI that validates every
  contributed question against live Wikipedia.
- **In progress:** the gold set, 178 of a planned 300 questions (58 human,
  120 LLM-drafted); the drafts join the results once people have reviewed them.
- **Not yet:** committed results for `groundedness`, confidence intervals on the
  leaderboard rows, and EmbeddingGemma.

## Limits

- **58 queries is a small set.** The intervals above are the honest width of
  that; most of the table's ordering is not resolved by this much data. They
  replace the eyeballed rule of thumb this project used to carry — "treat
  differences under roughly 0.05 nDCG as noise" — which was both too strict
  for paired comparisons and too loose for unpaired ones.
- **One annotator for the human set.** The 58 health questions are
  single-annotated, so they have no agreement figure. The 120 drafted questions
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
- **The generation half has no committed results yet.** Only retrieval is
  measured in the tables above.
- **The embedding model is pinned by name, not by revision.**

## Contribute

The first two limits shrink with every contributor. Adding questions needs no
ML background — pick a Turkish Wikipedia article, write 5–10 paraphrased
questions, and open a pull request with one JSON file. A validator checks each
file against Wikipedia in CI. See [CONTRIBUTING.md](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/CONTRIBUTING.md) (Türkçe
açıklama dahil) and the
[good first issues](https://github.com/RizgarOzan/turkish-rag-eval/issues?q=is%3Aopen+label%3A%22good+first+issue%22).

Submitting an embedding model is one command and a pull request — see
[Leaderboard](#leaderboard).

## Data

Turkish Wikipedia, CC BY-SA 4.0. See [NOTICE.md](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/NOTICE.md).

## Licence

Code MIT ([LICENSE](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/LICENSE)); data under `data/` CC BY-SA 4.0
([NOTICE.md](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/NOTICE.md)).
