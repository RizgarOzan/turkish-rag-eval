# turkish-rag-eval

A retrieval evaluation harness for Turkish, built around one question:
**which parts of a RAG pipeline actually earn their cost on an agglutinative
language?**

Three chunking strategies × four retrievers = 12 configurations, measured on a
hand-labelled gold set of 58 Turkish health questions over 54 Wikipedia
articles (1.09 M characters). Every number below is reproducible with
`python src/run_eval.py`.

## Results

`nDCG@10`, `Recall@5` and `MRR` over 58 queries. Latency is per query on CPU
(no GPU anywhere in this project).

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

**3. Hybrid beats both parts, everywhere.** RRF fusion wins in all three
chunking regimes, and the best single config (hierarchical + hybrid, 0.607) is
21% above the best non-hybrid one (hierarchical + dense, 0.501). Dense and
sparse fail on different queries, so fusing ranks recovers more than either.

**4. Latency is not the constraint at this scale.** BM25 answers in 4–6 ms
P95, dense in 19–25 ms, hybrid in 25–31 ms — the hybrid's extra cost is the
dense leg, not the fusion. On a 2 000-chunk corpus, retrieval quality is worth
far more than the milliseconds.

## What did not work

**The multilingual embedding model underperformed plain BM25.** With
`paraphrase-multilingual-MiniLM-L12-v2`, keyword search with a 5-character
stemmer beat dense retrieval on two of three chunking strategies (fixed:
0.476 vs 0.446; sentence: 0.510 vs 0.461) while being ~5× faster. Dense only
edges ahead once hierarchical chunking gives it heading context, and even
then barely (0.501 vs 0.494).

This is worth stating plainly because the default assumption — *embeddings
beat keywords* — does not hold here. A small distilled multilingual model
carries limited Turkish capacity. A Turkish-specific or larger multilingual
embedding model is the obvious next experiment, and the harness is built to
swap it with `--model`.

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

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python src/fetch_corpus.py    # rebuilds data/raw/corpus.json from Wikipedia
python src/run_eval.py        # writes results/summary.json + per-query files
python src/run_abstain.py     # coverage / accuracy curve for the best config
```

`--model <name>` swaps the embedding model. Any sentence-transformers model
works.

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

## Limits

- **58 queries is a small set.** Differences under roughly 0.05 nDCG should be
  read as noise, not as a ranking.
- **One annotator, no second pass.** Gold labels are single-annotated; there is
  no inter-annotator agreement figure.
- **One embedding model tested.** The dense results characterise that model,
  not dense retrieval in general.
- **Encyclopaedic text, not clinical text.** Wikipedia prose differs from
  clinical notes in vocabulary, structure and abbreviation density. Nothing
  here transfers to a clinical setting without re-measurement.
- **Retrieval only.** No generation, no answer-quality evaluation.

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
