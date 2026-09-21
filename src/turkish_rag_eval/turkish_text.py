"""Turkish-specific text handling for the sparse (BM25) retriever.

Two things break naive keyword search in Turkish:

1. Casing. Python's ``str.lower()`` maps "I" to "i" and "İ" to "i" plus a
   combining dot. Turkish needs I->ı and İ->i. Without this, "İLTİHAP" and
   "iltihap" are different tokens.
2. Agglutination. "diyabet", "diyabetin", "diyabete", "diyabetli" are four
   tokens for one concept, so a query term almost never matches the surface
   form in the document.

Fix 2 with fixed-prefix stemming: keep the first N characters. Crude, but it
is a known-strong baseline for Turkish IR and needs no morphological
analyser. The harness measures how much it is worth.
"""

import re
import unicodedata

_UPPER_MAP = str.maketrans("IİĞÜŞÖÇ", "ıiğüşöç")
_TOKEN = re.compile(r"[0-9a-zçğıöşü]+")

# Words carrying no retrieval signal. Deliberately short - an aggressive
# stoplist removes question words that matter ("nedir" is not in here).
STOPWORDS = {
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "da", "de", "ki", "mi",
    "için", "gibi", "daha", "çok", "en", "ise", "ama", "fakat", "ancak",
    "olan", "olarak", "olur", "her", "bazı", "hem", "ya", "the",
}


def turkish_lower(text: str) -> str:
    """Lowercase with the Turkish dotted/dotless i rule applied first.

    The text is normalised to NFC first: a decomposed "İ" (I + U+0307) would
    otherwise survive the translate table, and the combining dot is not in the
    token character class, which silently splits the word in two.
    """
    return unicodedata.normalize("NFC", text).translate(_UPPER_MAP).lower()


def tokenize(text: str, stem_length: int | None = 5) -> list[str]:
    """Lowercase, split on non-letters, drop stopwords, optionally stem.

    ``stem_length=None`` disables stemming, which is the baseline the
    harness compares against.
    """
    tokens = _TOKEN.findall(turkish_lower(text))
    out = []
    for token in tokens:
        if token in STOPWORDS or len(token) < 2:
            continue
        out.append(token[:stem_length] if stem_length else token)
    return out
