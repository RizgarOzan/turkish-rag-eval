# Results

Everything measured so far, with the setup each number came from. The
`tests/test_readme_*.py` tests rebuild these tables from the committed files in
`results/`, so a re-run that moves a number turns CI red until this page
follows.

## Main table

The 12-row table (58 queries, default model
`paraphrase-multilingual-MiniLM-L12-v2`, CPU only) is on the
[README](../README.md#results); `turkish-rag-eval report` regenerates it.

`turkish-rag-eval report` also names the cheapest
configuration the data cannot separate from the best:

```
Recommended: sentence + hybrid_rrf
  0.558 ndcg@10 against 0.613 for hierarchical + hybrid_rrf, a gap of 0.056
  that a paired bootstrap over 58 shared queries cannot distinguish from zero.
  It answers in 27 ms at P95 against 37 ms.
```

Comparisons are **paired**: both systems answered the same queries, so
resampling per-query differences removes the variance of some queries being
harder than others. Here is where that matters:
`hierarchical + hybrid_rrf` beats `fixed + hybrid_rrf` by 0.097 and beats
`sentence + bm25_stem5` by 0.104, but only the **larger** gap clears zero,
because its per-query differences are so much steadier (sd 0.34 against 0.40).
Ranking by the gap alone gets this backwards.

These intervals replace the eyeballed rule of thumb this project used to carry
("treat differences under roughly 0.05 nDCG as noise"), which was both too
strict for paired comparisons and too loose for unpaired ones.

Hierarchical chunking only helps the dense retriever, and fixed-size chunking,
the most common default, lost on every retriever. More in the
[design notes](https://github.com/RizgarOzan/turkish-rag-eval/blob/main/docs/design.md#what-else-the-numbers-say).

### Generation: groundedness

The retrieval results feed a generation step: the 58 questions answered from the top 5
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

To rerun it with other models, see [usage](usage.md#groundedness-with-other-models).

### Is 5 the right prefix?

The harness fixed the BM25 prefix at 5 characters before measuring anything
else. `docs/measure_stem.py` sweeps it (BM25 only, no model needed) and writes
`results/stem/`; the interval is a 95% paired bootstrap on nDCG@10 against 5:

| Chunking | none | 3 | 4 | **5** | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| fixed | 0.387 | 0.415 | 0.421 | 0.476 | 0.416 | 0.448 | 0.429 |
| sentence | 0.411 | 0.489 | 0.499 | 0.510 | 0.440 | 0.478 | 0.456 |
| hierarchical | 0.383 | 0.472 | 0.482 | 0.494 | 0.433 | 0.444 | 0.415 |

5 has the highest score on all three chunkings. No stemming and a 6-character
prefix are clearly worse on every one (sentence 6: -0.069 [-0.130, -0.015]). On
sentence and hierarchical chunks a 3 or 4 character prefix is within noise of 5
(hierarchical 4: -0.012 [-0.059, +0.035]); only fixed chunks separate them. The
curve is not smooth (7 beats 6 on every chunking), which is what 58 questions do. So 5 holds
up, but it was picked first and then confirmed on the same questions, and the
data cannot separate it from 4.

### Is 700 the right chunk size?

Every chunker takes 700 characters, picked before anything was measured.
`docs/measure_chunk_size.py` sweeps it (BM25 only, stem 5) and writes
`results/chunk_size/`; the interval is a 95% paired bootstrap on nDCG@10
against 700. All 58 questions are scored at every size: no answer span ever
straddles a chunk boundary.

| Chunking | 300 | 500 | **700** | 1000 | 1500 |
|---|---|---|---|---|---|
| fixed | 0.396 | 0.458 | 0.476 | 0.520 | 0.586 |
| sentence | 0.409 | 0.423 | 0.510 | 0.525 | 0.575 |
| hierarchical | 0.427 | 0.434 | 0.494 | 0.469 | 0.496 |

300 is clearly worse on every chunking. Above 700 the three strategies part
ways: fixed keeps climbing (1500: +0.110 [+0.030, +0.193]) and sentence
follows it, while hierarchical is flat (1500: +0.002 [-0.055, +0.056]),
because a section boundary caps a chunk before the size does (1,266 chunks
at 1500 against 782 for fixed).

Read the climb with care. The corpus goes from 5,331 fixed chunks at 300 to
782 at 1500, so a bigger chunk is both a bigger target and one of fewer
candidates; nDCG@10 rewards that on its own, and the whole chunk is what the
generator would be handed. So this sweep does not say that 1500 retrieves
better, it says 700 is not the top of the BM25 curve for fixed and sentence
chunks, and that hierarchical chunking is the one whose score the size
parameter stops moving.

The same sweep with a dense model
(`python docs/measure_chunk_size.py intfloat/multilingual-e5-small`, written
to `results/chunk_size/multilingual-e5-small.json`) sees the same shrinking
pool of candidates and still does not move:

| e5-small | 300 | 500 | **700** | 1000 | 1500 |
|---|---|---|---|---|---|
| fixed | 0.617 | 0.644 | 0.654 | 0.654 | 0.707 |
| sentence | 0.583 | 0.612 | 0.644 | 0.651 | 0.641 |
| hierarchical | 0.613 | 0.643 | 0.642 | 0.652 | 0.669 |

None of the twelve changes against 700 clears zero; the largest is fixed at
1500, +0.054 [-0.039, +0.145]. No chunk passes the model's 512-token limit at
any size (the longest at 1500 is 471 tokens), so nothing is cut off. The
climb above is mostly BM25's: a longer chunk holds more of the query's
words. For the dense retriever 700 is as good a size as any in the range.

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
> are comparable with each other and not with the main table. The four
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

To submit a model, see [usage](usage.md#choosing-a-model).

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

Typed in capitals, the default model's nDCG@10 falls from 0.501 to 0.051.
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

Commands are in [usage](usage.md#stress-test).

## Speed and hardware

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

The GPU (RTX 5050 Laptop, 8 GB) changes how long encoding takes, not what it
finds. `run --retrievers dense` on a CUDA build of torch (sentence-transformers
picks the GPU by itself; no flag) wrote `results/gpu/`, and every nDCG@10 is
identical to the CPU run in `results/models/`. Only the times move:

| Model | Chunking | encode CPU → GPU | query P95 CPU → GPU |
|---|---|---|---|
| e5-small | fixed | 69 s → 22 s | 22 ms → 31 ms |
| e5-small | sentence | 60 s → 4 s | 22 ms → 19 ms |
| e5-small | hierarchical | 64 s → 5 s | 21 ms → 21 ms |
| Mursit-Large-TR-Retrieval | fixed | 733 s → 70 s | 187 ms → 34 ms |
| Mursit-Large-TR-Retrieval | sentence | 635 s → 49 s | 176 ms → 33 ms |
| Mursit-Large-TR-Retrieval | hierarchical | 689 s → 53 s | 178 ms → 33 ms |

Mursit-Large-TR-Retrieval goes from about 11 minutes of encoding to under one
(roughly 10 to 13 times) and its query P95 drops from around 180 ms to 33 ms.
e5-small encodes 3 to 14 times faster, but its queries do not get faster: one
short query does not fill a GPU, and P95 sits at 19 to 31 ms either way. The
first e5-small row (fixed) includes CUDA start-up, which is why it is the slowest
GPU encode. The CPU and GPU runs used different torch builds (2.13 CPU, 2.11
cu128) on different days, so read the ratios as rough.

ONNX Runtime on the CPU finds the same passages too. `docs/measure_onnx.py <model>`
loads the model through sentence-transformers' `backend="onnx"` (needs
`sentence-transformers[onnx]`; e5-small uses the model's own ONNX export from the
Hub, Mursit is exported on load) and writes `results/onnx/`; every nDCG@10 matches
the torch run in `results/models/` to four decimals:

| Model | Chunking | encode torch → ONNX | query P95 torch → ONNX |
|---|---|---|---|
| e5-small | fixed | 69 s → 61 s | 22 ms → 8 ms |
| e5-small | sentence | 60 s → 52 s | 22 ms → 8 ms |
| e5-small | hierarchical | 64 s → 56 s | 21 ms → 11 ms |
| Mursit-Large-TR-Retrieval | fixed | 733 s → 816 s | 187 ms → 57 ms |
| Mursit-Large-TR-Retrieval | sentence | 635 s → 732 s | 176 ms → 59 ms |
| Mursit-Large-TR-Retrieval | hierarchical | 689 s → 803 s | 178 ms → 60 ms |

The win is at query time: P95 falls roughly 2 to 3 times for both models, with no
loss in nDCG@10 (int8 gave Mursit a similar speedup but cost e5-small 0.07 to
0.12). Encoding the corpus is a wash, 12 to 13% faster for e5-small and 11 to 17%
slower for Mursit, so if you only index once ONNX buys little; if you serve
queries it is the cheaper win. The torch and ONNX runs were timed on different
days, so read single-digit-percent gaps as noise.

## Serving

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
