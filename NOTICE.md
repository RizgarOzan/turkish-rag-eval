# Data attribution

The retrieval corpus in `data/raw/corpus.json` is built from Turkish Wikipedia
articles fetched through the MediaWiki API by `src/fetch_corpus.py`.

Wikipedia text is licensed **CC BY-SA 4.0**. Each document keeps its `title`
and permanent `url`, so every retrieved chunk can be traced to its source
article. The corpus file is not committed (see `.gitignore`); run the fetcher
to rebuild it.

## The gold set

`data/eval/gold.json` **is** committed, and each item carries a short verbatim
`answer_span` quoted from the Turkish Wikipedia article named by its `doc_id`.
Those spans are therefore also **CC BY-SA 4.0**, attributed to the article in the
same record (`doc_id` = MediaWiki pageid, plus the title and URL in the fetched
corpus). The questions themselves were written for this project - deliberately
paraphrased rather than copied - and are released under CC BY-SA 4.0 as well, so
the whole gold set can be reused under one licence. Single annotator (the repo
author), one pass, no inter-annotator agreement figure.

The **code** in this repository is MIT (see `LICENSE`). The licences do not
conflict: MIT covers the harness, CC BY-SA 4.0 covers the Wikipedia-derived text
in `data/`.

## No health data

No patient data, clinical records or any personal health information is used
anywhere in this project. The corpus is public encyclopaedic text only, and
nothing here is a medical device or clinical decision support tool.
