# turkish-rag-eval

[![validate](https://github.com/RizgarOzan/turkish-rag-eval/actions/workflows/validate.yml/badge.svg)](https://github.com/RizgarOzan/turkish-rag-eval/actions/workflows/validate.yml)
[![licence: MIT + CC BY-SA 4.0](https://img.shields.io/badge/licence-MIT%20%2B%20CC%20BY--SA%204.0-blue)](NOTICE.md)
[![dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-RizgarOzan%2Fturkish--rag--eval-yellow)](https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval)

A retrieval evaluation harness for Turkish, built around one question:
**which parts of a RAG pipeline actually earn their cost on an agglutinative
language?**

Three chunking strategies × four retrievers = 12 configurations, measured on a
hand-labelled gold set of 58 Turkish health questions over 54 Wikipedia
articles (1.09 M characters), then repeated for six embedding models. Every
number below is reproducible with `python src/run_eval.py [--model <name>]`.

**Short answer:** Turkish stemming is the cheapest win for BM25, and the choice
of embedding model matters more than anything else. A Turkish retrieval model
reaches nDCG@10 0.781, against 0.494 for stemmed BM25 and 0.501 for the small
multilingual default ([Embedding models](#embedding-models)). The story behind
both results is in a short write-up: [English](docs/blog/2026-09-19-bm25-turkish-en.md) ·
[Türkçe](docs/blog/2026-09-19-bm25-turkish-tr.md).

**Contents:** [Results](#results) · [Embedding models](#embedding-models) ·
[Why not an existing benchmark?](#why-not-an-existing-benchmark) ·
[Uncertainty handling](#uncertainty-handling) · [Running it](#running-it) ·
[Gold set](#gold-set) · [Limits](#limits) · [Contribute](#contribute)

## Results

`nDCG@10`, `Recall@5` and `MRR` over 58 queries, with the default embedding
model `paraphrase-multilingual-MiniLM-L12-v2`. Latency is per query on CPU (no
GPU anywhere in this project).

| Chunking | Retriever | nDCG@10 | R@5 | MRR | P95 |
|---|---|---|---|---|---|
| **hierarchical** | **hybrid_rrf** | **0.607** | **0.690** | **0.550** | 25 ms |
| sentence | hybrid_rrf | 0.552 | 0.621 | 0.498 | 25 ms |
| fixed | hybrid_rrf | 0.510 | 0.578 | 0.476 | 31 ms |
| sentence | bm25_stem5 | 0.510 | 0.638 | 0.462 | 4 ms |
| hierarchical | dense | 0.501 | 0.569 | 0.441 | 25 ms |
| hierarchical | bm25_stem5 | 0.494 | 0.569 | 0.444 | 5 ms |
| fixed | bm25_stem5 | 0.476 | 0.526 | 0.421 | 4 ms |
| sentence | dense | 0.461 | 0.552 | 0.410 | 19 ms |
| fixed | dense | 0.446 | 0.491 | 0.402 | 21 ms |
| sentence | bm25_nostem | 0.410 | 0.483 | 0.355 | 4 ms |
| fixed | bm25_nostem | 0.387 | 0.414 | 0.319 | 4 ms |
| hierarchical | bm25_nostem | 0.383 | 0.500 | 0.319 | 6 ms |

## What the numbers say

**1. Turkish stemming is the single cheapest win.** Truncating tokens to a
5-character prefix before BM25 lifts nDCG@10 by 23–29% across every chunking
strategy (0.387→0.476 fixed, 0.410→0.510 sentence, 0.383→0.494 hierarchical).
Turkish is agglutinative: *diyabet*, *diyabetin*, *diyabete* and *diyabetli*
are four surface forms of one concept, so an unstemmed index almost never
matches the query's form. No morphological analyser is needed to capture most
of that.

**2. Hierarchical chunking mostly helps the dense side.** Prepending the
heading path (`Alzheimer hastalığı > Nedenler > Genetik > Geç başlangıç`) to
the embedded text moves dense nDCG@10 from 0.446 → 0.461 → 0.501 as chunking
goes fixed → sentence → hierarchical. A bare paragraph loses the context that
told you what it was about; the heading path puts it back. BM25 benefits far
less, because the section words were often already in the body.

**3. With this model, hybrid beats both parts, everywhere.** RRF fusion wins in all three
chunking regimes, and the best single config (hierarchical + hybrid, 0.607) is
21% above the best non-hybrid one (hierarchical + dense, 0.501). Dense and
sparse fail on different queries, so fusing ranks recovers more than either.
This does not carry over to stronger embedding models (see below).

**4. Latency is not the constraint at this scale.** BM25 answers in 4–6 ms
P95, dense in 19–25 ms, hybrid in 25–31 ms — the hybrid's extra cost is the
dense leg, not the fusion. On a 2 000-chunk corpus, retrieval quality is worth
far more than the milliseconds.

## Embedding models

The same harness with five more models, measured 2026-09-18 on one 16-thread
CPU (8 torch threads). Dense `nDCG@10` per chunking strategy, the hybrid
(dense + stemmed BM25, RRF) on hierarchical chunks, and the cost: time to embed
all three chunkings (5 643 chunks) and per-query P95 on hierarchical chunks.
Stemmed BM25 alone scores 0.476 / 0.510 / 0.494.

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
minutes, before the hierarchical chunks; its two numbers come from that partial
run. `google/embeddinggemma-300m` is gated behind a licence click and was not
run. Per-query results for every completed model are in `results/models/`.

**1. The model was the problem, not dense retrieval.** Every model trained for
retrieval (E5, bge-m3, Mursit) beats stemmed BM25 on its own, on every chunking.
A small E5 model the same size as the default goes from 0.501 to 0.642 with no
other change. It is as fast (22 ms) and embeds the corpus in 3 minutes, so it is
the obvious replacement for the default on a CPU.

**2. Hybrid only pays for a weak dense model.** RRF gives BM25's ranking the same
weight as the dense one. That lifts MiniLM (+0.106), but it pulls a strong model
down: Mursit drops from 0.781 to 0.673 and e5-base from 0.668 to 0.648. Whether
to fuse depends on the dense model, so it should be measured and not assumed.

**3. "Turkish-only" is not enough.** The emrecan model was trained for sentence
similarity (NLI + STS-b) and truncates input at 75 tokens, which cuts 83–99 % of
chunks short. As a dense retriever it is no better than the default. Mursit
was trained for retrieval and is the best model here, at about 10× the query latency
of e5-small and 35 minutes to embed the corpus on a CPU.

E5 models expect `query: ` / `passage: ` in front of every input; `src/models.py`
adds these, and a run without them is not a fair E5 number.

## Why not an existing benchmark?

MTEB-style retrieval benchmarks score an embedding model on passages that
are already split. They answer "which model?", not "which chunker, is Turkish
stemming worth it, does a hybrid help, and what does each cost on a CPU?".
This harness keeps the articles whole, lets every chunker cut them its own way,
and judges each chunk by the answer span, so pipeline choices can be compared
on the same labels. For a model-only comparison, the same data exports to the
BEIR layout MTEB reads (`python src/export_hf.py`), published as
[RizgarOzan/turkish-rag-eval](https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval)
on Hugging Face.

## What did not work

**The multilingual embedding model underperformed plain BM25.** With
`paraphrase-multilingual-MiniLM-L12-v2`, keyword search with a 5-character
stemmer beat dense retrieval on two of three chunking strategies (fixed:
0.476 vs 0.446; sentence: 0.510 vs 0.461) while being ~5× faster. Dense only
edges ahead once hierarchical chunking gives it heading context, and even
then barely (0.501 vs 0.494).

This is worth stating plainly because the default assumption — *embeddings
beat keywords* — does not hold for this model. A small distilled multilingual model
carries limited Turkish capacity. A Turkish-specific or larger multilingual
embedding model is the obvious next experiment, and the harness is built to
swap it with `--model`. That experiment is now in
[Embedding models](#embedding-models): with a retrieval-trained model, dense
retrieval wins clearly.

**Fixed-size chunking lost on every retriever.** It is the most common default
and the worst performer in all four columns.

## Uncertainty handling

`src/abstain.py` treats "should this be answered automatically?" as a
measurement rather than a guess: it sweeps a confidence threshold and reports
the coverage / selective-accuracy trade-off, so an operator can pick the point
that meets an accuracy floor and escalate the rest to a human.

Two signals are measured, because the obvious one fails.

**Top-1 margin does not work.** The intuitive signal — how far ahead the top
hit is — carries no information on RRF output. Fused scores are `1/(60+rank)`,
so the top-two gap is ~2% on every query regardless of difficulty. Selective
accuracy *falls* as the threshold rises (0.500 at 0.05 → 0.250 at 0.10 → 0.000
at 0.20), which is what a useless signal looks like.

**Raw cosine similarity does.** The dense retriever's absolute top-1 score is
monotone in the right direction:

| threshold | coverage | selective accuracy | answered | escalated |
|---|---|---|---|---|
| 0.65 | 0.672 | 0.436 | 39 | 19 |
| 0.70 | 0.466 | 0.444 | 27 | 31 |
| 0.75 | 0.259 | 0.467 | 15 | 43 |
| **0.80** | **0.121** | **0.714** | 7 | 51 |

Unfiltered top-1 accuracy is 0.466. At a 70% accuracy floor the harness picks
threshold 0.80: 12% of queries answered automatically, 51 sent to a human. At
an 80% or 90% floor it reports that **no threshold qualifies** — a real answer,
not a failure. It means this configuration should not run unattended at that
requirement.

## Running it

Needs **Python 3.10+** (the code uses `int | None` and `list[str]`). CPU only —
no GPU anywhere.

```bash
git clone https://github.com/RizgarOzan/turkish-rag-eval
cd turkish-rag-eval
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt   # pulls CPU-only torch from PyTorch's index
python src/fetch_corpus.py        # rebuilds data/raw/corpus.json from Wikipedia
python src/run_eval.py            # writes results/summary.json + per-query files
python src/run_abstain.py         # coverage / accuracy curve for the best config
python src/export_hf.py           # corpus + queries + qrels as a Hugging Face dataset in data/hf/
```

`python src/run_eval.py --model <name>` swaps the embedding model; any
sentence-transformers model works. Results for a non-default model go to
`results/models/<org>__<name>/`, so the main table above is never overwritten. `run_abstain.py` always uses the default
model, so pass nothing there.

The metric implementations have their own tests:

```bash
pip install pytest && python -m pytest tests -q
```

## How relevance is defined

Each gold item names the source document and a short verbatim answer span from
it. A retrieved chunk counts as relevant when it comes from that document *and*
contains the span. That definition survives changing the chunker — otherwise
every strategy would need its own labels, and the comparison would be
meaningless.

Questions are deliberately **paraphrased**, not copied from the articles
("Şeker hastalığı teşhisi konan kişilerin ne kadarında ketoasidoz da
bulunuyor?" against text reading "yaklaşık %25'i, diyabet teşhisi
konulduğunda..."). Lexically copied questions would hand BM25 an unearned
advantage and make the dense/sparse comparison worthless.

## Gold set

| Files | Questions | Labelled by | In the results above |
|---|---|---|---|
| `data/eval/gold.json` (health) | 58 | one human | yes |
| `data/eval/contrib/llm-draft-*.json` (history, geography, astronomy, biology, computing) | 60 | two independent LLM passes | not yet |

The set is growing toward 300 questions across more domains. Questions a
model drafted are marked `"source": "llm-draft"`. A second model then picked
its own answer span for each one without seeing the first label
(`second_annotation`). Two spans agree when one contains the other or their
token F1 is at least 0.5; agreed items get `"review": "agreed"`, the rest get
`"needs-human"` and are never loaded. Every batch so far is 30 of 30 agreed.

### Agreement between the two passes

| Batch | Questions | Identical | Mean IoU | Mean token F1 | Cohen's κ, fixed / sentence / hierarchical |
|---|---|---|---|---|---|
| 1 — 2026-09-18 (Malazgirt, Kapadokya, Mars, Mitokondri, Linux) | 30 | 16 | 0.816 | 0.878 | 1.00 / 1.00 / 1.00 |
| 2 — 2026-09-20 (İstanbul'un Fethi, Ağrı Dağı, Jüpiter, Fotosentez, İnternet) | 30 | 4 | 0.419 | 0.540 | 0.96 / 1.00 / 1.00 |
| **All drafts** | 60 | 20 | 0.617 | 0.709 | 0.98 / 1.00 / 1.00 |

"Identical" means identical after tokenisation, so case and punctuation are
folded; by raw string the counts are 1 and 2. IoU and token F1 come from
`src/agreement.py`; κ is that file's chunk-level measure — for each chunking
strategy, the binary "does this chunk contain the answer" label each span
assigns to each chunk of its article, which is exactly how `run_eval.py`
decides what counts as a hit.

The two rows disagree about wording, not about the answer. In batch 2 the
second pass kept picking the shortest span that still answers the question
("7.4 büyüklüğünde" against the whole clause around it), which halves IoU -
yet the labels the benchmark actually scores are the same: of the 180
question × strategy runs, 3 differ, all with fixed-size chunks, where the
shorter span also fell inside one neighbouring overlapping window. Two LLMs
tend to pick the same sentence, so read this as a sanity check rather than as
human agreement.

The κ column is measured locally, because it needs the ten draft articles in
the corpus and `data/raw/` is fetched rather than committed; the other
columns are recomputed from the files in CI (`tests/test_readme_agreement.py`).
Wiring the embedded second labels into `src/agreement.py` itself is
[#15](https://github.com/RizgarOzan/turkish-rag-eval/issues/15).

Drafts stay out of every number above until a re-run says otherwise:
`load_gold()` skips them unless called with `include_drafts=True`. Synthetic
test questions are common practice as long as they are declared and their
agreement is measured. This section is that declaration.

## Limits

- **58 queries is a small set.** Differences under roughly 0.05 nDCG should be
  read as noise, not as a ranking.
- **One annotator for the human set.** The 58 health questions are
  single-annotated, so they have no agreement figure. The 60 drafted questions
  are double-labelled, but by two LLM passes rather than by two people.
- **Six embedding models, one corpus.** The model comparison uses the same 58
  health questions; `bge-m3` is a partial run and EmbeddingGemma is missing.
  `run_abstain.py` still uses only the default model.
- **Encyclopaedic text, not clinical text.** Wikipedia prose differs from
  clinical notes in vocabulary, structure and abbreviation density. Nothing
  here transfers to a clinical setting without re-measurement.
- **Retrieval only.** No generation, no answer-quality evaluation.
- **The corpus is a live snapshot, not a pinned one.** `fetch_corpus.py` pulls the
  current revision of each article, so re-running it months later gives slightly
  different chunk counts and therefore slightly different numbers. The gold set
  and the code are pinned; Wikipedia is not.
- **The embedding model is pinned by name, not by revision**, and ties in the
  sparse rankings are broken by `numpy.argsort`, which is not stable. Neither
  moves a result by more than rounding, but neither is bit-reproducible either.

## Contribute

The first two limits above shrink with every contributor. Adding questions
needs no ML background — pick a Turkish Wikipedia article, write 5–10
paraphrased questions, and open a pull request with one JSON file. A validator
checks each file against Wikipedia in CI. See [CONTRIBUTING.md](CONTRIBUTING.md)
(Türkçe açıklama dahil) and the [open issues](https://github.com/RizgarOzan/turkish-rag-eval/issues).

## Notes on Turkish

Two language-specific traps are handled in `src/turkish_text.py`:

- `"İLTİHAP".lower()` returns `i̇ltihap` in Python — an `i` plus a combining
  dot (U+0307) — and `"ISIRIK".lower()` returns `isirik` instead of `ısırık`.
  Turkish needs `I→ı` and `İ→i` applied before the generic lowercase.
- Fixed-prefix stemming at 5 characters is used instead of a morphological
  analyser. It is crude and will conflate unrelated words sharing a prefix; the
  table above is the argument that it still pays for itself.

## Data

Turkish Wikipedia, CC BY-SA 4.0. See [NOTICE.md](NOTICE.md). No patient data or
personal health information is used anywhere in this project, and nothing here
is a medical device.

## Licence

Code MIT ([LICENSE](LICENSE)); data under `data/` CC BY-SA 4.0 ([NOTICE.md](NOTICE.md)).
