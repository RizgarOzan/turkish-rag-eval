"""Load a corpus from wherever it lives, and prove which one it was.

Two jobs that belong together.

**Loading.** The harness started out reading one hardcoded file, which meant it
could only ever re-measure its own bundled articles. A corpus now comes from
any of three shapes, picked by looking at the path:

``corpus.json``      the bundled shape - a list of {doc_id, title, text, url}
``<dir>/``           every .txt and .md under it, one document per file
``<dir>/`` (BEIR)    a directory holding corpus.jsonl, the layout MTEB reads

**Identity.** Wikipedia is a moving target: ``fetch_corpus`` pulls whatever
revision is current, so the same command a month later gives slightly
different text, slightly different chunks and therefore slightly different
numbers. For a benchmark that is disqualifying - if you run it and disagree
with the published table, neither of us can tell whether the pipeline changed
or the encyclopaedia did.

So every corpus carries a fingerprint: a hash over (doc_id, hash of text)
pairs, order-independent, that every result file records. ``corpus.lock.json``
pins the fingerprint together with each article's revision id, and
``fetch_corpus --verify`` says exactly which articles moved. That does not
freeze Wikipedia, but it makes drift loud instead of silent, which is the
property a comparison actually needs.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

LOCK_VERSION = 1
TEXT_SUFFIXES = (".txt", ".md")

#: A document shorter than this is almost always a stub, a redirect or an
#: accidentally-included README, and it only adds noise to the index.
MIN_DOCUMENT_CHARS = 200


class CorpusError(ValueError):
    """The corpus could not be loaded, with a message aimed at whoever ran it."""


@dataclass(frozen=True)
class Corpus:
    docs: list[dict]
    source: str

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def chars(self) -> int:
        return sum(len(d["text"]) for d in self.docs)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.docs)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(docs: list[dict]) -> str:
    """A hash identifying this exact set of documents.

    Built from sorted ``doc_id:sha256(text)`` lines, so it does not depend on
    the order documents arrive in, and titles or URLs changing without the text
    changing does not move it - the retrieval numbers only depend on the text.
    """
    lines = sorted(f"{d['doc_id']}:{text_digest(d['text'])}" for d in docs)
    return "sha256:" + hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _normalise(doc: dict, source: str) -> dict:
    """Accept both the bundled shape and the BEIR one, reject anything else."""
    doc_id = doc.get("doc_id") or doc.get("_id")
    text = doc.get("text")
    if not doc_id or not isinstance(text, str):
        raise CorpusError(
            f"{source}: every document needs an id (doc_id or _id) and a text "
            f"field; got keys {sorted(doc)}")
    normalised = {
        "doc_id": str(doc_id),
        "title": doc.get("title") or str(doc_id),
        "text": text,
        "url": doc.get("url", ""),
    }
    # Carried through when the source recorded it, so a corpus fetched from
    # Wikipedia can be locked to exact revisions without a second round trip.
    if doc.get("revid") is not None:
        normalised["revid"] = doc["revid"]
    return normalised


def _from_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise CorpusError(f"{path}: expected a JSON list of documents")
    return [_normalise(d, str(path)) for d in payload]


def _from_jsonl(path: Path) -> list[dict]:
    docs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            docs.append(_normalise(json.loads(line), f"{path}:{number}"))
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}:{number}: invalid JSON - {exc}") from exc
    return docs


def _from_directory(directory: Path) -> list[dict]:
    """One document per .txt/.md file, keyed by path relative to the directory.

    The relative path is the id rather than the bare filename, so two
    ``notes.md`` in different folders stay distinct documents.
    """
    docs = []
    # Resolved, because as_uri() refuses a relative path and --corpus is very
    # often given as one.
    directory = directory.resolve()
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < MIN_DOCUMENT_CHARS:
            continue
        relative = path.relative_to(directory)
        docs.append({
            # Posix separators so an id written on Linux still matches on
            # Windows: the gold set has to name documents portably.
            "doc_id": relative.as_posix(),
            "title": path.stem,
            "text": text,
            "url": path.as_uri(),
        })
    return docs


def load_corpus(path: Path | str) -> Corpus:
    """Load a corpus, choosing the reader from what is actually at ``path``."""
    path = Path(path)
    if not path.exists():
        raise CorpusError(
            f"{path} does not exist. Point --corpus at a JSON file, a BEIR "
            f"directory holding corpus.jsonl, or a folder of .txt/.md files.")

    if path.is_file():
        docs = _from_jsonl(path) if path.suffix == ".jsonl" else _from_json(path)
    elif (path / "corpus.jsonl").exists():
        docs = _from_jsonl(path / "corpus.jsonl")
    else:
        docs = _from_directory(path)
        if not docs:
            raise CorpusError(
                f"{path} holds no .txt or .md file of at least "
                f"{MIN_DOCUMENT_CHARS} characters.")

    duplicates = {d["doc_id"] for d in docs}
    if len(duplicates) != len(docs):
        raise CorpusError(f"{path}: document ids are not unique")
    if not docs:
        raise CorpusError(f"{path}: no documents found")

    # Sorted by id so chunk order - and therefore every tie-break downstream -
    # does not depend on filesystem or JSON ordering.
    docs.sort(key=lambda d: d["doc_id"])
    return Corpus(docs=docs, source=str(path))


def build_lock(docs: list[dict], revisions: dict[str, int] | None = None) -> dict:
    """The lockfile body: the fingerprint plus per-document identity."""
    revisions = revisions or {}
    return {
        "version": LOCK_VERSION,
        "fingerprint": fingerprint(docs),
        "documents": [
            {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "revid": revisions.get(d["doc_id"]),
                "sha256": text_digest(d["text"]),
                "chars": len(d["text"]),
            }
            for d in sorted(docs, key=lambda d: d["doc_id"])
        ],
    }


def compare_to_lock(lock: dict, docs: list[dict]) -> list[str]:
    """Human-readable drift between a lockfile and a corpus, empty when equal."""
    if lock.get("version") != LOCK_VERSION:
        return [f"lockfile version {lock.get('version')} is not {LOCK_VERSION}; "
                f"regenerate it with --update-lock"]
    if lock.get("fingerprint") == fingerprint(docs):
        return []

    locked = {d["doc_id"]: d for d in lock.get("documents", [])}
    current = {d["doc_id"]: d for d in docs}
    drift = []
    for doc_id in sorted(set(locked) - set(current)):
        drift.append(f"missing: {locked[doc_id]['title']} ({doc_id})")
    for doc_id in sorted(set(current) - set(locked)):
        drift.append(f"added: {current[doc_id]['title']} ({doc_id})")
    for doc_id in sorted(set(locked) & set(current)):
        was, now = locked[doc_id], current[doc_id]
        if was["sha256"] != text_digest(now["text"]):
            delta = len(now["text"]) - was["chars"]
            drift.append(f"changed: {now['title']} ({doc_id}), "
                         f"{delta:+d} characters")
    return drift
