"""Three chunking strategies, compared head to head by the eval harness.

fixed        - character window with overlap, ignores document structure
sentence     - packs whole sentences up to a size budget, never splits one
hierarchical - respects ``== Section ==`` boundaries and prepends the heading
               path ("Diyabet > Belirtiler") to every chunk, so the embedding
               carries the context the raw paragraph lost
"""

import re

# Turkish abbreviations that end in a period but do not end a sentence.
ABBREVIATIONS = {
    "dr", "doç", "prof", "op", "uzm", "yrd", "sn", "av", "bkz", "vb", "vs",
    "örn", "yy", "no", "sk", "mah", "cad", "tel", "bkm", "mg", "ml", "gr",
}

_SENT_END = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1$", re.MULTILINE)


def split_sentences(text: str) -> list[str]:
    """Split on sentence enders, then glue back false splits after abbreviations."""
    parts = _SENT_END.split(text)
    out: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        if out:
            tail = out[-1].rstrip()
            last_word = tail.split()[-1].rstrip(".").lower() if tail.split() else ""
            # "Dr." or a bare number ("1." in a list) does not end a sentence.
            if last_word in ABBREVIATIONS or last_word.isdigit():
                out[-1] = tail + " " + part
                continue
        out.append(part)
    return out


def chunk_fixed(doc: dict, size: int = 700, overlap: int = 100) -> list[dict]:
    text = _HEADING.sub(" ", doc["text"])
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start, step = [], 0, size - overlap
    while start < len(text):
        body = text[start:start + size].strip()
        if len(body) > 50:
            chunks.append(_make(doc, body, len(chunks), heading_path=doc["title"]))
        start += step
    return chunks


def chunk_sentence(doc: dict, size: int = 700) -> list[dict]:
    text = _HEADING.sub(" ", doc["text"])
    text = re.sub(r"[ \t]+", " ", text)
    chunks, buf = [], ""
    for sentence in split_sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if buf and len(buf) + len(sentence) + 1 > size:
            chunks.append(_make(doc, buf, len(chunks), heading_path=doc["title"]))
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if len(buf) > 50:
        chunks.append(_make(doc, buf, len(chunks), heading_path=doc["title"]))
    return chunks


def chunk_hierarchical(doc: dict, size: int = 700) -> list[dict]:
    """Split by section, then pack sentences within each section separately."""
    chunks = []
    for heading_path, section_text in _sections(doc):
        for sentence_chunk in _pack(section_text, size):
            chunks.append(_make(doc, sentence_chunk, len(chunks),
                                heading_path=heading_path))
    return chunks


def _sections(doc: dict) -> list[tuple[str, str]]:
    """Walk the article, tracking the current heading stack by ``=`` depth."""
    text, sections, stack = doc["text"], [], []
    matches = list(_HEADING.finditer(text))
    intro = text[:matches[0].start()] if matches else text
    if intro.strip():
        sections.append((doc["title"], intro.strip()))
    for i, match in enumerate(matches):
        depth = len(match.group(1)) - 1  # "==" is depth 1
        title = match.group(2)
        stack = stack[:depth - 1] + [title]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append((" > ".join([doc["title"]] + stack), body))
    return sections


def _pack(text: str, size: int) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    out, buf = [], ""
    for sentence in split_sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if buf and len(buf) + len(sentence) + 1 > size:
            out.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if len(buf) > 50:
        out.append(buf)
    return out


def _make(doc: dict, body: str, index: int, heading_path: str) -> dict:
    return {
        "chunk_id": f"{doc['doc_id']}::{index}",
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "url": doc["url"],
        "heading_path": heading_path,
        "body": body,
        # What actually gets embedded. Only the hierarchical strategy has a
        # heading path worth more than the bare article title.
        "embed_text": f"{heading_path}\n{body}",
    }


STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "hierarchical": chunk_hierarchical,
}
