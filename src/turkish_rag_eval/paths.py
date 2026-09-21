"""Where the project's data and results live.

The harness runs three ways and each needs a different answer: from a git
checkout, from an install pointed at the bundled corpus, and from an install
pointed at *your* corpus. Only the first can assume the repository layout, so
the root is discovered rather than derived from ``__file__``.
"""

import os
from pathlib import Path

#: Presence of the bundled gold set is what identifies a checkout.
MARKER = Path("data") / "eval" / "gold.json"

ENV_VAR = "TURKISH_RAG_EVAL_ROOT"


def find_root(start: Path | None = None) -> Path:
    """The nearest directory holding the bundled gold set.

    Checked in order: the ``TURKISH_RAG_EVAL_ROOT`` override, ``start`` (or the
    working directory) and its parents, then the installed package's own
    location. The last one finds the checkout for an editable install and finds
    nothing for a wheel - where there is no bundled data to find, and every
    path the caller needs has to be passed explicitly anyway.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    origin = (start or Path.cwd()).resolve()
    installed = Path(__file__).resolve().parent.parent.parent
    for base in (origin, installed):
        for candidate in (base, *base.parents):
            if (candidate / MARKER).exists():
                return candidate
    return origin


ROOT = find_root()
