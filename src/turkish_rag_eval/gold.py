"""Load the gold set: the original hand-labelled file plus contributed files.

Contributed questions live one file per pull request under
``data/eval/contrib/``, so many people can add questions at once without
editing the same file and colliding.
"""

import json
import re
from pathlib import Path

from .paths import ROOT

GOLD = ROOT / "data" / "eval" / "gold.json"
CONTRIB = ROOT / "data" / "eval" / "contrib"


def gold_files() -> list[Path]:
    return [GOLD, *sorted(CONTRIB.glob("*.json"))]


def resolve_gold(path: Path | str | None) -> list[Path]:
    """Question files named by a ``--gold`` argument.

    A file is itself, a directory is every ``.json`` in it, and ``None`` is the
    bundled set - so pointing the harness at your own questions is one flag
    rather than a fork.
    """
    if path is None:
        return gold_files()

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"{path} holds no .json question files")
        return files
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return [path]


def load_gold(paths: list[Path] | None = None, include_drafts: bool = False) -> list[dict]:
    """LLM-drafted questions (``"source": "llm-draft"``) stay out unless asked
    for, so the published numbers keep coming from the hand-labelled set. A
    draft whose two annotators disagreed (``"review": "needs-human"``) never
    loads: its gold span is not settled yet."""
    items = []
    for path in paths or gold_files():
        for item in json.loads(path.read_text(encoding="utf-8")):
            if item.get("review") == "needs-human":
                continue
            if item.get("source") == "llm-draft" and not include_drafts:
                continue
            items.append(item)
    return items


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
