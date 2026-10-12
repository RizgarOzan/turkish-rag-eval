"""Undo the ways people actually mistype a Turkish query, before retrieval.

``turkish-rag-eval stress`` measures three of them on the gold set: no Turkish
keyboard ("sigara icmek"), caps lock, and a pipeline that ran Python's
``str.lower()`` over capitals ("İ" becomes "i" plus a combining dot). Each one
costs the retrievers a large share of their nDCG; this module is the fix.

Turkish letters are restored from the corpus itself: every word in the
documents is indexed under its ASCII fold, and an ASCII-only query word is
replaced by the most frequent corpus word with the same fold. No dictionary,
no model, no dependency - and nothing the corpus does not contain can be
invented. Words that already carry a Turkish letter, two-letter words and
words with nothing to restore are never touched.
"""

import re
import unicodedata
from collections import Counter
from typing import Iterable

from .turkish_text import turkish_lower

_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")
_WORD = re.compile(r"[^\W\d_]+")
_COMBINING_DOT = "̇"


def fold_ascii(text: str) -> str:
    """Drop every Turkish mark: "Şeker" -> "Seker", "i̇" -> "i"."""
    text = unicodedata.normalize("NFC", text).translate(_ASCII)
    return text.replace(_COMBINING_DOT, "")


def _turkish_upper_first(word: str) -> str:
    head = {"i": "İ", "ı": "I"}.get(word[0], word[0].upper())
    return head + word[1:]


def _is_shouting(text: str) -> bool:
    """Caps lock, not an acronym: enough letters, and nearly all capitals."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 6 and sum(c.isupper() for c in letters) >= 0.8 * len(letters)


class QueryNormalizer:
    def __init__(self, texts: Iterable[str]):
        counts = Counter(
            word for text in texts
            for word in _WORD.findall(turkish_lower(text)))
        self.vocab: dict[str, str] = {}
        for word, _ in counts.most_common():
            self.vocab.setdefault(fold_ascii(word), word)

    def _restore(self, match: re.Match) -> str:
        word = match.group()
        # Two-letter words are too ambiguous ("mi" and "mı" are both
        # question particles), and a word with a Turkish letter is already typed.
        if len(word) < 3 or not word.isascii():
            return word
        form = self.vocab.get(fold_ascii(turkish_lower(word)))
        if form is None:
            return word
        restored = _turkish_upper_first(form) if word[0].isupper() else form
        # Only Turkish marks may change ("Istanbul" -> "İstanbul"), never the
        # case of other letters: that would turn an acronym like "OKB" into "Okb".
        return restored if fold_ascii(restored) == word else word

    def __call__(self, query: str) -> str:
        query = unicodedata.normalize("NFC", query).replace("i" + _COMBINING_DOT, "i")
        if _is_shouting(query):
            query = turkish_lower(query)
        return _WORD.sub(self._restore, query)
