"""Every metric in a write-up under docs/blog/ must be one the README reports."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POSTS = sorted((ROOT / "docs" / "blog").glob("*.md"))


def decimals(text):
    # Turkish posts write 0,781; the README writes 0.781.
    return {m.replace(",", ".") for m in re.findall(r"\d+[.,]\d+", text)}


def latencies(text):
    return {f"{n} ms" for n in re.findall(r"(\d+) ms", text)}


def test_there_is_a_post_in_each_language():
    assert {p.stem.rsplit("-", 1)[-1] for p in POSTS} >= {"tr", "en"}


@pytest.mark.parametrize("post", POSTS, ids=lambda p: p.name)
def test_every_number_comes_from_the_readme(post):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    text = post.read_text(encoding="utf-8")
    assert decimals(text) - decimals(readme) == set()
    assert latencies(text) - latencies(readme) == set()
