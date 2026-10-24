"""The README's ask GIF is drawn from the README's own ask transcript.

The GIF is a picture of text, so it can go stale without anything noticing.
Each GIF carries a hash of the transcript it was drawn from; if the README
block changes and the GIF is not redrawn, the last test here goes red.
"""

import importlib.util

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from turkish_rag_eval.paths import ROOT  # noqa: E402

spec = importlib.util.spec_from_file_location("make_ask_gif", ROOT / "docs" / "make_ask_gif.py")
make_ask_gif = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_ask_gif)

README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_transcript_is_the_readme_block():
    command, lines = make_ask_gif.transcript(README)
    assert command == "turkish-rag-eval ask q034"
    assert lines[0].startswith("q034: ")
    assert lines[-1].startswith("missed in top 3:")
    assert sum(make_ask_gif.is_hit(line) for line in lines) == 1


def test_the_plan_types_the_command_then_shows_whole_blocks():
    command, lines = make_ask_gif.transcript(README)
    plan = make_ask_gif.plan(command, lines)
    assert plan[0][:2] == (0, 0)
    assert plan[-1][:2] == (len(command), len(lines))
    shown = [n for _, n, _ in plan]
    assert shown == sorted(shown)
    # A retriever block appears at once: output only ever stops at a blank
    # line or at the end, never with half a top-3 on screen.
    for n in set(shown) - {0, len(lines)}:
        assert lines[n] == "", f"frame stops inside a block at line {n}"
    assert plan[-1][2] >= 3000, "the last frame should stay up long enough to read"


def test_both_themes_write_every_frame(tmp_path):
    command, lines = make_ask_gif.transcript(README)
    frames = len(make_ask_gif.plan(command, lines))
    firsts = []
    for theme in ("light", "dark"):
        out = tmp_path / f"{theme}.gif"
        make_ask_gif.write(command, lines, out, theme)
        with Image.open(out) as gif:
            assert gif.n_frames == frames
            firsts.append(gif.convert("RGB").getpixel((2, 2)))
    assert firsts[0] != firsts[1]


@pytest.mark.parametrize("name", ["ask-q034.gif", "ask-q034-dark.gif"])
def test_the_committed_gif_matches_the_readme(name):
    with Image.open(ROOT / "docs" / "media" / name) as gif:
        drawn_from = gif.info.get("comment", b"").decode()
    assert drawn_from == make_ask_gif.fingerprint(README), \
        "README ask block changed: run python docs/make_ask_gif.py"
