# Changelog

Results are only comparable when they name the version that produced them.
Every summary row records `harness_version`, so quote a number as "measured on
turkish-rag-eval v0.1.0" rather than "measured on turkish-rag-eval".

This project follows [semantic versioning](https://semver.org). For a
benchmark, the version's job is slightly unusual: **anything that can move a
published number is a breaking change**, even when no API changed.

## [Unreleased]

Nothing yet.

## [0.1.0]

First release. The harness existed before it; this is the point from which
results can be cited.

### Added

- **Installable package and one CLI.** `pip install turkish-rag-eval` and
  `turkish-rag-eval <command>` replace running scripts from a checkout with
  the right working directory.
- **Bring your own corpus.** `run --corpus` accepts a folder of `.txt`/`.md`,
  a BEIR directory, or a JSON corpus; `--gold` accepts your own questions.
  `--retrievers` makes the embedding model optional, so a BM25-only run needs
  no torch and no download.
- **Confidence intervals.** Bootstrap over queries for every metric, and a
  *paired* bootstrap for comparing two configurations on the queries both
  scored.
- **`report`** — the cheapest configuration the data cannot separate from the
  best, with what the top of the table would cost.
- **`charts`** — the interval and abstention figures, light and dark.
- **`leaderboard`** — the model table, and `--check`, which re-derives every
  reported metric from the per-query relevance arrays committed beside it. CI
  runs it on every push and holds new submissions to a provenance bar.
- **`bootstrap`** — drafts a gold set for a corpus that has none, in two
  independent passes, holding disagreements for human review.
- **`groundedness`** — scores the generation half against the gold answer
  spans, reporting the hallucination rate separately for queries whose
  evidence was never retrieved.
- **Corpus pinning.** `data/corpus.lock.json` records a fingerprint over every
  article's text plus its Wikipedia revision id; `fetch-corpus --verify` names
  the articles that moved.

### Changed

- **Ranking breaks ties stably.** `np.argsort` defaulted to an unstable sort
  and sparse scores tie constantly — every chunk sharing no query term scores
  exactly 0.0 — so a tie at rank 1 could move Recall@1 and MRR between runs on
  identical data. *This can move a published number: results produced before
  this release may differ in the last decimal.*
- Annotation passes agree by containment first, token F1 second. All 60
  double-labelled questions in the repository are containment pairs, yet 29
  fall below a 0.6 Jaccard floor; a similarity floor alone would have flagged
  half an already-reviewed gold set.
- Every summary row now carries `harness_version`, `corpus_fingerprint` and
  `corpus_documents`.
- The README leads with the findings and the charts, and states the
  reasoning behind the measurement design rather than leaving it in comments.
- Limits records the Wikipedia contamination caveat: every embedding model
  ranked here was almost certainly trained on Turkish Wikipedia, so absolute
  scores are optimistic and only the comparison between pipeline choices on
  the same corpus is sound.

### Fixed

- A relative `--corpus` directory crashed in `Path.as_uri()`.
- `slugify` used `str.lower()`, which turns "İ" into an `i` plus a combining
  dot, so "İstanbul" became "i-stanbul" — the casing trap this project exists
  to measure, in its own identifiers.

### Known

- The committed `results/` predate the stable tie-break and have no
  provenance fields. Re-run `turkish-rag-eval run` to bring them onto v0.1.0
  numbering; `leaderboard --check` reports this as a warning until then.

[Unreleased]: https://github.com/RizgarOzan/turkish-rag-eval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/RizgarOzan/turkish-rag-eval/releases/tag/v0.1.0
