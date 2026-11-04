# Gold set

| Files | Questions | Labelled by | In the published results |
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

Drafts stay out of every published number until a re-run says otherwise:
`load_gold()` skips them unless called with `include_drafts=True`. Synthetic
test questions are common practice as long as they are declared and their
agreement is measured. This section is that declaration.

The corpus is 54 Turkish Wikipedia articles, 1.09 M characters; 27 of them
answer at least one question and the other 27 are distractors from the same
domain.

| Batch | Questions | Identical | Mean IoU | Mean token F1 | Cohen's κ, fixed / sentence / hierarchical |
|---|---|---|---|---|---|
| 1 — 2026-09-18 (Malazgirt, Kapadokya, Mars, Mitokondri, Linux) | 30 | 15 | 0.798 | 0.866 | 1.00 / 1.00 / 1.00 |
| 2 — 2026-09-20 (İstanbul'un Fethi, Ağrı Dağı, Jüpiter, Fotosentez, İnternet)¹ | 30 | 4 | 0.427 | 0.549 | 0.94 / 0.99 / 0.99 |
| 3 — 2026-09-21 (Çaldıran Muharebesi, Tuz Gölü, Satürn, Ribozom, Unix) | 30 | 18 | 0.829 | 0.872 | 0.92 / 0.96 / 0.96 |
| 4 — 2026-09-24 (Mohaç Muharebesi, Kızılırmak, Venüs, Enzim, Derleyici) | 30 | 15 | 0.747 | 0.792 | 0.96 / 0.96 / 0.96 |
| 5 — 2026-09-25 (Kösedağ Muharebesi, Uludağ, Neptün, RNA, İşletim sistemi) | 30 | 24 | 0.922 | 0.943 | 1.00 / 1.00 / 1.00 |
| 6 — 2026-09-25 (Preveze Deniz Muharebesi, Erciyes, Uranüs, Hemoglobin, Veritabanı) | 30 | 20 | 0.869 | 0.901 | 0.93 / 0.90 / 0.87 |
| 7 — 2026-09-26 (Ankara Muharebesi, Van Gölü, Merkür, DNA, World Wide Web) | 30 | 25 | 0.944 | 0.960 | 0.93 / 0.93 / 0.97 |
| 8 — 2026-09-26 (Niğbolu Muharebesi, Fırat, Ay, Protein, Yapay zekâ) | 32 | 29 | 0.964 | 0.975 | 0.94 / 0.97 / 1.00 |
| **All drafts** | 242 | 150 | 0.814 | 0.858 | 0.95 / 0.97 / 0.97 |

To measure agreement over the drafts eligible for loading, run
`turkish-rag-eval agreement --include-drafts`. Embedded second labels are
expanded into ordinary second annotations before grouping. Without the flag,
the command still loads only the hand-labelled set. The table above measures
all draft files, including unresolved `needs-human` questions; the CLI keeps
`load_gold`'s review filters. With the current files it therefore reports
237 questions and mean token F1 0.872, versus 242 and 0.858 for all drafts.
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
have checked them. These rows are not in the main tables: most of the
questions have not been checked by a person yet. The run also names every
question no chunk can answer; a span split by a chunk boundary is the usual
reason. Here that is 17 questions with fixed-size chunks and none with
sentence or hierarchical chunks (the one hierarchical miss was fixed on
2026-09-27, see batch 5 above). Results for all three chunkers are in
[`results/full/`](https://github.com/RizgarOzan/turkish-rag-eval/tree/main/results/full).
