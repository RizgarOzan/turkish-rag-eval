# Design notes

The reasoning and the detail behind the [README](../README.md): why the harness is built the way it is,
and the parts of the results that need more than a table.

## What else the numbers say

**Hierarchical chunking only helps the dense retriever.** Prepending the
heading path to the embedded text moves dense nDCG@10 from 0.446 → 0.461 →
0.501 as chunking goes fixed → sentence → hierarchical. BM25 does not care.

**Fixed-size chunking lost on every retriever.** It is the most common default
and the worst performer in all four columns.

**Hybrid fusion only pays for a weak dense model.** RRF gives BM25's ranking
the same weight as the dense one. That lifts the small default (+0.106), but
pulls a strong model down: Mursit drops from 0.781 to 0.673, e5-base from
0.668 to 0.648. Whether to fuse should be measured, not assumed.

## Design decisions

The parts worth arguing about, rather than the parts that were obvious.

**Relevance is defined at the answer span, not the chunk.** Each gold item
names the source document and a short verbatim span from it; a retrieved chunk
counts as relevant when it comes from that document *and* contains the span.
Chunk-level labels would have to be redone for every chunking strategy, which
would make the comparison between strategies meaningless — the thing the
harness exists to measure.

**Questions are paraphrased, never copied.** "Şeker hastalığı teşhisi konan
kişilerin ne kadarında ketoasidoz da bulunuyor?" is asked of text reading
"yaklaşık %25'i, diyabet teşhisi konulduğunda...". A lexically copied question
hands BM25 an unearned win and silently inflates every sparse row in the
table. CI rejects any contributed question with more than 60% word overlap
with its own answer span.

**The relevance gate folds case with `str.lower()`, not `turkish_lower()`.**
That looks like exactly the bug this project studies, and is not.
`turkish_lower` matters when a *query* is matched against a *document*: the
two are written independently, so surface form and casing diverge. The
relevance gate instead matches a verbatim span against the very text it was
copied from, so both operands are the same run of characters, and `lower()` is
context-free per character. Using a different fold here would desynchronise
the evaluator from `validate_gold.py`, which admits gold under the same
normaliser.

**RRF instead of score interpolation.** Cosine similarity and BM25 scores live
on different, corpus-dependent scales; fusing ranks needs no per-corpus
tuning. `k=60`, from Cormack et al. (2009).

**Two annotation passes agree by containment, not by similarity.** 235 of the
242 double-labelled questions here are containment pairs — one span inside the
other — yet 52 fall below a 0.6 Jaccard floor, 48 of them containment pairs.
The passes were almost never disagreeing about *where* the answer is, only
about how much of the sentence to sweep in, and containment is what the
harness itself tests. A similarity floor alone would have sent a reviewer to
arbitrate a third of an already-reviewed set. The four genuine disagreements —
two different sentences that both name polysomes, and three from batch 4 — are
the ones the rule holds back, and `passes_agree()` reproduces all 242 of the
committed labels exactly.

**Ties break stably.** `np.argsort` defaults to an unstable sort, and sparse
scores tie constantly — every chunk sharing no query term scores exactly 0.0.
An unstable tie at rank 1 moves Recall@1 and MRR between runs on identical
data.

## Abstention

`turkish-rag-eval abstain` treats "should this be answered automatically?" as
a measurement rather than a guess: it sweeps a confidence threshold and
reports the coverage / selective-accuracy trade-off, so an operator can pick
the point that meets an accuracy floor and escalate the rest.

Unfiltered top-1 accuracy is 0.466. Using the dense retriever's raw cosine:

| threshold | coverage | selective accuracy | answered | escalated |
|---|---|---|---|---|
| 0.65 | 0.672 | 0.436 | 39 | 19 |
| 0.70 | 0.466 | 0.444 | 27 | 31 |
| 0.75 | 0.259 | 0.467 | 15 | 43 |
| **0.80** | **0.121** | **0.714** | 7 | 51 |

At a 70% accuracy floor the harness picks threshold 0.80: 12% of queries
answered automatically, 51 sent to a human. At an 80% or 90% floor it reports
that **no threshold qualifies** — a real answer, not a failure. It means this
configuration should not run unattended at that requirement.

The top-1 margin, plotted in the README, is the signal most people reach for first and
is actively misleading on fused rankings.

## Groundedness

Retrieval is half of RAG, so the generation half is scored too — using a label
the gold set already carries, the verbatim answer span, rather than a second
round of annotation:

```bash
turkish-rag-eval groundedness --limit 20
```

The split is the point. The harness already knows whether the answer span was
retrieved, which divides every query into two populations that deserve
different questions:

- **Span retrieved** — the answer should rest on the passages and convey the
  span. Both go to a judge.
- **Span not retrieved** — nothing in the context answers the question, so the
  only correct behaviour is to decline. Answering anyway is a hallucination,
  and *that* rate is what decides whether a Turkish RAG system can face users.

A single pooled "accuracy" hides exactly that number. Abstention is detected
deterministically through a sentinel the generator is instructed to emit, so
"did it decline" never depends on a judge's mood; only groundedness and
correctness cost a model call.

**Status:** the command and its tests are in place, but no groundedness
results are committed yet. By default the generator and the judge are the same
model (`--answer-model` and `--judge-model` both default to `claude-opus-5`),
which invites self-preference; a published run should use a judge from a
different model family and spot-check its verdicts by hand.

## Reproducibility

A benchmark whose numbers cannot be reproduced is not comparable, so:

- **The corpus is pinned.** `data/corpus.lock.json` records a fingerprint over
  every article's text plus its Wikipedia revision id.
  `turkish-rag-eval fetch-corpus --verify` fails and names the articles that
  moved. Wikipedia still changes; the point is that it can no longer change
  silently.
- **Every result records its provenance** — harness version, corpus
  fingerprint, document count — in each summary row.
- **Ranking is deterministic**, ties included.
- **Dependencies are pinned**, and results are quoted against a release tag.

The README's main table was re-run on v0.1.0 and carries the pinned corpus
fingerprint. The re-run is also the clearest evidence the tie-break mattered:
**eight of the twelve rows came back bit-identical**, and the four that moved
are exactly the ones where ties are expected — the three `hybrid_rrf` rows,
whose RRF scores collide at `1/(60+rank)`, and one `bm25_nostem` row, where
every chunk sharing no query term scores exactly 0.0. No `dense` or
`bm25_stem5` row changed by a single digit.

## Agreement between the two passes

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

¹ Re-measured 2026-09-26: the Turkish Wikipedia article *Jüpiter* was rewritten that day to correct errors, and five of its spans no longer matched the live text. Three changed only in wording (a comma, "30,003" → "30" seconds, "dört uydu" → "uydular") and were edited in both labels; two changed in substance — the 40,000 km mantle thickness is gone and the Great Red Spot went from "at least 400 years" to "recorded since 1831" — so those two questions were rewritten and labelled again by both passes. The row was 0.419 / 0.540 / 0.96 / 1.00 / 1.00 before.

"Identical" means identical after tokenisation, so case and punctuation are
folded; by raw string the counts are 1, 2, 18, 6, 8, 20, 25 and 29. IoU and token F1 come from
`agreement.py`; κ is that file's chunk-level measure — for each chunking
strategy, the binary "does this chunk contain the answer" label each span
assigns to each chunk of its article, which is exactly how `run_eval.py`
decides what counts as a hit.

The two rows disagree about wording, not about the answer, and that gap is
what set the containment rule in [Design decisions](#design-decisions). In
batch 2 the second pass kept picking the shortest span that still answers the
question ("7.4 büyüklüğünde" against the whole clause around it), which halves
IoU — yet the labels the benchmark actually scores are the same: of the 180
question × strategy runs, 7 differ, all the same short-span effect: the
shorter span also falls in a second chunk. Three are fixed-size neighbours;
four come from the two *Jüpiter* questions relabelled after the article was
rewritten (see ¹), where the second pass's "1831'den beri" also appears
in an earlier sentence. Batch 3
has 5 differing runs of 90: three are the needs-human polysome question from the README's gold-set section, two are
the same short-span effect with fixed chunks. Batch 4 has 3 differing runs of
90, all one question: the first pass took the definition of cross compilers,
the second only the term, which sits in an earlier sentence. Its other two
needs-human questions still mark the same chunks, because both passes'
sentences fall in the same paragraph. Batch 5 has no differing runs. Batch 6 has 9 of 90, from four questions, and
all go the same way: the second pass kept a two-sentence answer that fits in
no chunk for that strategy, so it marks nothing relevant, while the first
pass's one-sentence span marks one chunk. The first pass had the same long
spans until they were measured against the chunkers and cut; a span that no
chunk contains is a question no retriever can get right. Batch 7 has 5 of
90, from three questions, the same way round. Batch 8 has 3 of 96: once the
second pass took the two sentences the first pass had cut to fit the
sentence chunker, and twice a span missed or caught a fixed window the
other did not. Two LLMs
tend to pick the same sentence, so read this as a sanity check rather than as
human agreement.

The κ column is measured locally, because it needs the forty draft articles
in the corpus and `data/raw/` is fetched rather than committed; the other
columns are recomputed from the files in CI (`tests/test_readme_agreement.py`).
Wiring the embedded second labels into `agreement.py` itself is
[#15](https://github.com/RizgarOzan/turkish-rag-eval/issues/15).

## Why not an existing benchmark?

MTEB-style retrieval benchmarks score an embedding model on passages that are
already split. They answer "which model?", not "which chunker, is Turkish
stemming worth it, does a hybrid help, and what does each cost on a CPU?".
This harness keeps articles whole, lets every chunker cut them its own way,
and judges each chunk by the answer span, so pipeline choices can be compared
on the same labels. For a model-only comparison, the same data exports to the
BEIR layout MTEB reads (`turkish-rag-eval export-hf`), published as
[RizgarOzan/turkish-rag-eval](https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval)
at two levels. Whole articles average ~20 000 characters, so a 512-token model
mostly sees each lead and scores crowd the top. The `passages-*` configs carry
this harness's hierarchical chunks with the same answer-span rule; scored
through MTEB's retrieval evaluator they give 0.501 / 0.642 / 0.668 / 0.779
nDCG@10 for MiniLM / e5-small / e5-base / Mursit, the leaderboard's
hierarchical dense column to within 0.003.

## Notes on Turkish

Two language-specific traps are handled in `turkish_text.py`:

- `"İLTİHAP".lower()` returns `i̇ltihap` in Python — an `i` plus a combining
  dot (U+0307) — and `"ISIRIK".lower()` returns `isirik` instead of `ısırık`.
  Turkish needs `I→ı` and `İ→i` applied before the generic lowercase.
- Fixed-prefix stemming at 5 characters is used instead of a morphological
  analyser. It is crude and will conflate unrelated words sharing a prefix;
  the README's results table is the argument that it still pays for itself. This is a
  known-strong Turkish IR baseline, not a new idea — what is measured here is
  what it is worth inside a modern chunked RAG pipeline.
