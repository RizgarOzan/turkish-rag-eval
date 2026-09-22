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

## Limits

- 58 queries is small: differences under about 0.05 nDCG are noise.
- One domain (health) for now. Questions from five more domains are being
  added in the repo and will join this dataset once verified.
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
