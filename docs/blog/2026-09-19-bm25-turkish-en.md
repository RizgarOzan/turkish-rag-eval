# The model that beats BM25 in Turkish is the one trained for retrieval, not the small multilingual default

*Rızgar Ozan · 2026-09-19 · [Türkçe sürüm](2026-09-19-bm25-turkish-tr.md)*

I built [turkish-rag-eval](https://github.com/RizgarOzan/turkish-rag-eval) around one
question: which parts of a RAG pipeline earn their cost on Turkish? After the first run I
wrote in the README that the multilingual embedding model could not beat stemmed BM25. The
sentence was true, and it pointed at the wrong culprit. Five more models later, the loser
turned out to be the model I had picked, not dense retrieval. This post is about those two
measurements.

## The setup

The gold set is 58 Turkish health questions that I wrote and labelled by hand over 54
Turkish Wikipedia articles (1.09 M characters). Each question paraphrases its source
instead of copying it: where the article says "diyabet teşhisi konulduğunda" (when diabetes
is diagnosed), the question asks about "şeker hastalığı teşhisi konan kişiler" (people
diagnosed with sugar disease). Copied questions would give BM25 an advantage it had not
earned.

A chunk counts as relevant when it comes from the right article *and* contains the answer
span. That definition does not depend on the chunker, so three chunking strategies (fixed
size, sentence, and hierarchical with the heading path prepended) can be compared across
four retrievers (plain BM25, BM25 with a 5-character prefix stemmer, dense, and an RRF
hybrid of the two) on the same labels. Twelve configurations, all on a CPU. There is no GPU
anywhere in this project.

## First run: "embeddings did not beat BM25"

The default model was `paraphrase-multilingual-MiniLM-L12-v2`: 118 M parameters, small and
fast, and the first model most multilingual search tutorials reach for. nDCG@10:

| Chunking | BM25 + stemming | Dense (MiniLM) |
|---|---|---|
| fixed | 0.476 | 0.446 |
| sentence | 0.510 | 0.461 |
| hierarchical | 0.494 | 0.501 |

Stemmed BM25 beat dense retrieval on two of the three chunkers while being roughly 5× faster.
Dense only edged ahead on hierarchical chunks, where the heading path gives back the context
a bare paragraph loses. Stemming alone was the cheapest win: cutting every token to its first
five characters lifted nDCG@10 by 23–29% on every chunker. Turkish is agglutinative;
*diyabet*, *diyabetin*, *diyabete* and *diyabetli* are four surface forms of one concept,
and an unstemmed index rarely matches the form the question uses. No morphological analyser
is needed for most of that gain.

The best config was the hybrid: hierarchical chunks with RRF at 0.607. So the first README
said: stem, fuse, and don't trust embeddings on their own.

## Second run: change the model

The harness had a `--model` flag from the start, and I did not use it before generalising.
On 18 September I measured five more models on the same 58 questions, the same corpus and
the same 16-thread CPU:

| Model | Params | Dense hierarchical | Hybrid hierarchical | Query P95 |
|---|---|---|---|---|
| MiniLM (default) | 118 M | 0.501 | 0.607 | 23 ms |
| emrecan (Turkish-only) | 111 M | 0.497 | 0.654 | 43 ms |
| multilingual-e5-small | 118 M | 0.642 | 0.639 | 22 ms |
| multilingual-e5-base | 278 M | 0.668 | 0.648 | 49 ms |
| Mursit-Large-TR-Retrieval (Turkish-only) | 404 M | **0.781** | 0.673 | 214 ms |

I stopped `bge-m3` after 45 minutes, before it reached the hierarchical chunks; on fixed and
sentence chunks it scored 0.766 and 0.767, where stemmed BM25 sits at 0.476 and 0.510.

## Three things the table says

**1. The model was the problem, not dense retrieval.** Every model trained for retrieval
(E5, bge-m3, Mursit) beats stemmed BM25 on its own, on every chunker. The telling row is
e5-small: the same size as the default, the same query latency (22 ms), three minutes to
embed the corpus, and it goes from 0.501 to 0.642 with nothing else changed. On a CPU it is
the first thing to swap in.

**2. Hybrid only pays for a weak dense model.** RRF gives BM25's ranking the same weight as
the dense one. That lifts MiniLM by 0.106, but with a strong model it mixes a good ranking
with a worse one: Mursit drops from 0.781 to 0.673 and e5-base from 0.668 to 0.648. The
first run's "hybrid wins everywhere" was a symptom of the weak model. Whether to fuse has
to be measured per model.

**3. "Turkish-only" is not enough.** I took the two most-downloaded Turkish models in the
Hugging Face `sentence-similarity` category. One (emrecan) was trained for sentence
similarity (NLI + STS-b) and truncates input at 75 tokens; counting with its tokenizer, that
cuts 83–99% of chunks short. As a dense retriever it is no better than the default (0.497 vs
0.501). The other (Mursit) was trained for retrieval and is the best model here. The
difference is the training objective, not the language.

## The cost

Quality has a price, and on a CPU you feel it. Mursit spends 214 ms per query, about 10× what
e5-small needs, and 35 minutes to embed all three chunkings. My picks:

- **Latency matters:** multilingual-e5-small. 0.642 at 22 ms, and no hybrid needed (0.639 is
  within noise of the dense score).
- **Quality matters and 214 ms per query is acceptable:** Mursit, dense only, 0.781.
- **No embeddings at all:** stemmed BM25. No analyser, 4–6 ms.

E5 models expect `query: ` / `passage: ` in front of every input. An E5 number without them
is not a fair E5 number; the harness adds them in `src/models.py`.

## Limits

58 questions is a small set; read nDCG differences under roughly 0.05 as noise, not as a
ranking. One person labelled them. They all come from one domain, Wikipedia's health
articles, which is not clinical text. `bge-m3` is a partial run, and
`google/embeddinggemma-300m` is gated behind a licence click and was not run. Every number is
in the README and reproducible with `python src/run_eval.py --model <name>`.

The set is growing toward 300 questions over five domains. A language model drafted the
first 30 new ones and a second, independent pass re-labelled them without seeing the first;
30 of 30 agreed. The human-verified share of those drafts is still 0%, and the README says
so. Adding questions needs no ML background: write 5–10 questions from a Turkish Wikipedia
article and open a pull request with one JSON file. [CONTRIBUTING.md](../../CONTRIBUTING.md)
walks through it.

## Takeaway

The first measurement was right, but the conclusion I drew from it was not. Before writing
"embeddings don't beat BM25 in Turkish" I only had to swap the model, and the harness was
already built for it. The README now carries both results: the old table stayed, and the new
one and this post sit next to it.
