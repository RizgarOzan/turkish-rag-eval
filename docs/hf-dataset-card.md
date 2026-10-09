---
language:
- tr
license: cc-by-sa-4.0
task_categories:
- text-retrieval
task_ids:
- document-retrieval
pretty_name: Turkish RAG Eval
size_categories:
- n<1K
tags:
- turkish
- retrieval
- rag
- wikipedia
configs:
- config_name: corpus
  data_files:
  - split: corpus
    path: corpus.jsonl
- config_name: queries
  data_files:
  - split: queries
    path: queries.jsonl
- config_name: default
  data_files:
  - split: test
    path: qrels/test.jsonl
- config_name: passages-corpus
  data_files:
  - split: corpus
    path: passages/corpus.jsonl
- config_name: passages-queries
  data_files:
  - split: queries
    path: passages/queries.jsonl
- config_name: passages-qrels
  data_files:
  - split: test
    path: passages/qrels/test.jsonl
- config_name: full-corpus
  data_files:
  - split: corpus
    path: full/corpus.jsonl
- config_name: full-queries
  data_files:
  - split: queries
    path: full/queries.jsonl
- config_name: full-qrels
  data_files:
  - split: test
    path: full/qrels/test.jsonl
- config_name: full-passages-corpus
  data_files:
  - split: corpus
    path: full/passages/corpus.jsonl
- config_name: full-passages-queries
  data_files:
  - split: queries
    path: full/passages/queries.jsonl
- config_name: full-passages-qrels
  data_files:
  - split: test
    path: full/passages/qrels/test.jsonl
---

# Turkish RAG Eval

58 hand-written Turkish questions over 54 Turkish Wikipedia health articles
(1.09 M characters), each labelled with the article that answers it and a
verbatim answer span. Built for
[turkish-rag-eval](https://github.com/RizgarOzan/turkish-rag-eval), a harness
that measures which parts of a RAG pipeline (chunking, stemming, embedding
model) pay off on Turkish.

```python
from datasets import load_dataset

corpus = load_dataset("RizgarOzan/turkish-rag-eval", "corpus", split="corpus")
queries = load_dataset("RizgarOzan/turkish-rag-eval", "queries", split="queries")
qrels = load_dataset("RizgarOzan/turkish-rag-eval", split="test")
```

## Files

BEIR layout, the one MTEB retrieval tasks read.

| File | Rows | Fields |
|---|---|---|
| `corpus.jsonl` | 54 | `_id` (MediaWiki pageid), `title`, `text`, `url` |
| `queries.jsonl` | 58 | `_id`, `text`, `answer_span` |
| `qrels/test.jsonl` | 58 | `query-id`, `corpus-id`, `score` (always 1) |

27 of the 54 articles answer at least one question; the other 27 are
distractors from the same domain.

`passages/` holds the same data at passage level (configs `passages-corpus`,
`passages-queries`, `passages-qrels`): the harness's hierarchical chunks
(≤700 characters, `title` is the section path), and a passage is relevant
when it comes from the answering article and contains `answer_span`.
Articles average ~20 000 characters, so a 512-token encoder only sees the
lead of each one at article level; passages are what a RAG pipeline
actually retrieves.

### Full set: 296 questions, six domains

`full/` (configs prefixed `full-`, same layout, article and passage level)
adds 238 LLM-drafted questions over 40 more articles in five domains:
history, geography, astronomy, biology and computing. 94 articles,
1.89 M characters, 296 questions. Each query has a `source` field, `human`
or `llm-draft`, so you can filter either way.

```python
queries = load_dataset("RizgarOzan/turkish-rag-eval", "full-queries", split="queries")
drafted = queries.filter(lambda q: q["source"] == "llm-draft")
```

The drafts were **double-labelled, not human-verified**: one model pass
wrote the question and marked the span, a second, separate pass marked the
span again without seeing the first. Only items where the two passes agree
are here (4 disagreements are left out). Agreement figures per batch are in
the [repo README](https://github.com/RizgarOzan/turkish-rag-eval#gold-set).
Treat the numbers you get on them as provisional.

## How it was made

- **Questions are paraphrased, not copied.** "Şeker hastalığı teşhisi konan
  kişilerin ne kadarında ketoasidoz da bulunuyor?" is asked of text reading
  "yaklaşık %25'i, diyabet teşhisi konulduğunda...". Copied wording would hand
  lexical retrievers an unearned advantage.
- **One human annotator, one pass.** No inter-annotator agreement figure yet.
- **Relevance here is article-level.** The harness is stricter: a retrieved
  chunk counts only if it comes from the right article *and* contains
  `answer_span`. Use the span if you chunk the corpus yourself.
- **Corpus snapshot.** Articles were fetched through the MediaWiki API as
  plain-text extracts, `== Section ==` markers kept. The GitHub repo refetches
  live revisions; this upload pins one snapshot.

## Results on this data

From the harness: hierarchical chunks, chunk-level relevance, CPU only. All
configurations and six models are in the
[repo README](https://github.com/RizgarOzan/turkish-rag-eval#embedding-models).

| Retriever | nDCG@10 |
|---|---|
| dense, `newmindai/Mursit-Large-TR-Retrieval` | 0.781 |
| dense, `intfloat/multilingual-e5-base` | 0.668 |
| dense, `paraphrase-multilingual-MiniLM-L12-v2` | 0.501 |
| BM25, 5-character prefix stemming | 0.494 |

On the full 296-question set the order changes, which is why the drafts are
kept apart: e5-base 0.646, Mursit 0.613, e5-small 0.568, MiniLM 0.405, and
stemmed BM25 0.558
([details](https://github.com/RizgarOzan/turkish-rag-eval#scoring-all-300)).

## Limits

- 58 queries is small: differences under about 0.05 nDCG are noise.
- The 58 human questions cover one domain (health). The other five domains
  exist only as LLM drafts in `full/` until people check them.
- Encyclopaedic text, not clinical text. No patient data is used anywhere, and
  nothing here is a medical device.

## Licence

Article text and answer spans come from Turkish Wikipedia, **CC BY-SA 4.0**;
each article keeps its `url`. The questions were written for this dataset and
are released under the same licence. The harness code is MIT.

## Citation

```bibtex
@misc{ozan2026turkishrageval,
  author       = {Rızgar Ozan},
  title        = {Turkish RAG Eval: a retrieval test set for Turkish Wikipedia},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval}},
  note         = {Harness: \url{https://github.com/RizgarOzan/turkish-rag-eval}}
}
```
