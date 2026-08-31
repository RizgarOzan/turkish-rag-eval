# Data attribution

The retrieval corpus in `data/raw/corpus.json` is built from Turkish Wikipedia
articles fetched through the MediaWiki API by `src/fetch_corpus.py`.

Wikipedia text is licensed **CC BY-SA 4.0**. Each document keeps its `title`
and permanent `url`, so every retrieved chunk can be traced to its source
article. The corpus file is not committed (see `.gitignore`); run the fetcher
to rebuild it.

No patient data, clinical records or any personal health information is used
anywhere in this project. The corpus is public encyclopaedic text only, and
nothing here is a medical device or clinical decision support tool.
