# Submitting this dataset to MTEB

Everything that can be prepared without network access to Hugging Face is
prepared: [`turkish_rag_eval.py`](turkish_rag_eval.py) is a complete MTEB task
definition, validated against `mteb` 2.21.5 — it imports, its `TaskMetadata`
passes validation, and `mteb.get_task("TurkishRagEvalRetrieval")` resolves it.

Two things are left, and both need credentials or a network this environment
does not have: **refreshing the Hugging Face dataset**, and **running the two
models MTEB asks for**.

## Why this fills a gap

Measured with MTEB's own API, not asserted:

```python
>>> len(mteb.get_tasks(languages=["tur"], task_types=["Retrieval"]))
6
>>> [t.metadata.name for t in ... if t.metadata.eval_langs == ["tur-Latn"]]
["TurHistQuadRetrieval"]
```

Six Turkish retrieval tasks, of which **one** is monolingual Turkish — Ottoman
history QA. Everything else reaches Turkish through a multilingual or
translated set. This adds a second native Turkish retrieval task in a
different domain, with questions written as paraphrases rather than copied
spans, so it does not hand lexical retrievers an unearned advantage.

## Step 1 — refresh the Hugging Face dataset

The dataset currently on the Hub predates v0.1.0: it was exported from the
earlier corpus snapshot. Pinning that revision would give MTEB a corpus whose
fingerprint does not match the one this repository's results were measured on.

```bash
turkish-rag-eval fetch-corpus --verify   # must report the pinned fingerprint
turkish-rag-eval export-hf --out data/hf
```

Upload `data/hf/` to `RizgarOzan/turkish-rag-eval`, then record the commit sha:

```python
from huggingface_hub import HfApi
print(HfApi().dataset_info("RizgarOzan/turkish-rag-eval").sha)
```

Put that sha into `turkish_rag_eval.py` in place of
`REPLACE_WITH_HF_COMMIT_SHA`. MTEB pins datasets by commit, which is the same
discipline this repository already applies to its corpus — a benchmark that
does not pin cannot be compared against itself later.

## Step 2 — put the task in a fork

```bash
git clone https://github.com/<you>/mteb && cd mteb
pip install -e .

cp <this repo>/docs/mteb/turkish_rag_eval.py mteb/tasks/retrieval/tur/
```

Then add it to `mteb/tasks/retrieval/tur/__init__.py`:

```python
from .tur_hist_quad import TurHistQuadRetrieval
from .turkish_rag_eval import TurkishRagEvalRetrieval

__all__ = ["TurHistQuadRetrieval", "TurkishRagEvalRetrieval"]
```

That is the whole registration mechanism — `mteb.get_task()` picks it up from
there.

No `load_data` override is needed. The export already writes the layout
`RetrievalDatasetLoader` expects: a `corpus` config (`_id`, `title`, `text`),
a `queries` config (`_id`, `text`), and a `default` config with a `test` split
(`query-id`, `corpus-id`, `score`). The loader renames `_id` to `id` itself and
selects exactly the three qrels columns, so the extra `url` and `answer_span`
columns are ignored rather than a problem.

## Step 3 — verify it loads and scores

```python
import mteb

task = mteb.get_task("TurkishRagEvalRetrieval")
task.load_data()

test = task.dataset["default"]["test"]
print(test["corpus"][0])
print(test["queries"][0])

task.calculate_descriptive_statistics()   # required before the PR
```

Then run the two models the checklist asks for:

```bash
mteb run -m mteb/baseline-random-encoder -t TurkishRagEvalRetrieval
mteb run -m intfloat/multilingual-e5-small -t TurkishRagEvalRetrieval
```

**Expected values, and what would mean something is wrong.** MTEB scores
retrieval over whole documents, while this repository chunks first, so its
numbers are the closest available reference rather than a prediction:
`intfloat/multilingual-e5-small` reaches nDCG@10 0.642–0.654 here depending on
the chunker. Article-level retrieval over 54 documents should land at or above
that. The random baseline should be near zero.

If e5-small comes back near 1.0, the task is trivial and the corpus needs more
distractors. If it comes back near the random baseline, something is wired
wrong — check that qrels ids match corpus ids.

## Step 4 — the pull request

Title:

> Add TurkishRagEvalRetrieval: a second monolingual Turkish retrieval task

Body — the checklist from MTEB's contributing guide, with the parts that are
already settled filled in:

```markdown
- [x] I have outlined why this dataset is filling an existing gap in `mteb`

`mteb.get_tasks(languages=["tur"], task_types=["Retrieval"])` returns six
tasks, of which one — TurHistQuadRetrieval — is monolingual Turkish; the rest
reach Turkish through multilingual or translated sets. This adds a second
native Turkish retrieval task, in a different domain (Wikipedia health
articles rather than Ottoman history).

Turkish is agglutinative, and retrieval behaves differently because of it: on
this data a 5-character prefix stemmer lifts BM25 nDCG@10 by 23-29%, with
every one of those gains clear of zero under a paired bootstrap. A retrieval
benchmark that reaches Turkish only through translated multilingual sets does
not surface that.

- [ ] I have tested that the dataset runs with the `mteb` package.
- [ ] I have run the following models on the task (adding the results to the pr).
  - [ ] `mteb/baseline-random-encoder`
  - [ ] `intfloat/multilingual-e5-small`
- [ ] I have checked that the performance is neither trivial nor random.
- [x] I have considered the size of the dataset and reduced it if it is too big

58 queries over 54 documents. Small, and deliberately so: every question is
hand-written and hand-checked against its source article. 27 of the 54
documents answer no question and exist as same-domain distractors.

- [x] I reproduced scores from the original paper (if applicable)

No paper. The harness that produced the dataset publishes its own numbers with
95% bootstrap intervals at https://github.com/RizgarOzan/turkish-rag-eval —
`intfloat/multilingual-e5-small` scores nDCG@10 0.642-0.654 there depending on
the chunking strategy. Those are chunk-level; MTEB scores whole documents, so
they are a reference point rather than a target.
```

Worth adding to the PR body, because reviewers ask:

- **Construction.** Questions are paraphrases of the passage that answers
  them, never copies — CI rejects any contributed question sharing more than
  60% of its words with its own answer span. Copied questions hand BM25 an
  unearned win and make a dense/sparse comparison meaningless.
- **Annotation.** The 58 questions in this export are single-annotated by a
  human. The repository also holds 90 LLM-drafted double-labelled questions,
  which are **not** in this export — they are excluded until reviewed.
- **Contamination.** The corpus is Turkish Wikipedia, and most embedding
  models have seen it. Absolute scores are optimistic. Say so in the PR rather
  than waiting to be asked; it is the first thing a careful reviewer checks.
- **Licence.** Article text is CC BY-SA 4.0 (Wikipedia); the questions were
  written for this dataset and released under the same licence.
