"""Draws the README's `ask` transcript as an animated GIF, light and dark.

The text comes from the README's own ```text block that starts with
`$ turkish-rag-eval ask`, so the picture shows exactly what the README says
the command prints. Re-run this after that block changes; the test in
tests/test_ask_gif.py goes red until you do.

Usage:  python docs/make_ask_gif.py
"""

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "docs" / "media"

# Consolas has no ✓, so it is not a candidate.
FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
]
FONT_FILES = ["CascadiaMono.ttf", "DejaVuSansMono.ttf"]
FONT_SIZE = 15
COLUMNS = 104
PAD = 16

THEMES = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#59636e",
              "accent": "#0550ae", "hit": "#1a7f37"},
    "dark": {"bg": "#0d1117", "ink": "#e6edf3", "muted": "#9198a1",
             "accent": "#79c0ff", "hit": "#3fb950"},
}
RETRIEVERS = {"dense", "bm25_stem5", "bm25_nostem", "hybrid_rrf"}


def transcript(readme: str) -> tuple[str, list[str]]:
    """The command and the output lines of the README's ask example."""
    lines = readme.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("$ turkish-rag-eval ask "))
    end = lines.index("```", start)
    return lines[start][2:], lines[start + 1:end]


def fingerprint(readme: str) -> str:
    command, lines = transcript(readme)
    return hashlib.sha256("\n".join([command, *lines]).encode()).hexdigest()


def plan(command: str, lines: list[str]) -> list[tuple[int, int, int]]:
    """Frames as (characters of the command typed, output lines shown, ms)."""
    frames = [(typed, 0, 60) for typed in range(0, len(command), 3)]
    frames.append((len(command), 0, 700))
    blocks = [i for i, line in enumerate(lines) if line == ""] + [len(lines)]
    for shown in blocks:
        frames.append((len(command), shown, 900))
    frames[-1] = (len(command), len(lines), 5000)
    return frames


def _font() -> ImageFont.FreeTypeFont:
    for directory in FONT_DIRS:
        for name in FONT_FILES:
            if (directory / name).exists():
                return ImageFont.truetype(str(directory / name), FONT_SIZE)
    raise SystemExit("No monospace font with a ✓ found; tried %s in %s."
                     % (", ".join(FONT_FILES), ", ".join(map(str, FONT_DIRS))))


def is_hit(line: str) -> bool:
    """A ranked chunk marked as holding the answer (the header has a ✓ too)."""
    return line.startswith("  ") and line.split()[1:2] == ["✓"]


def _colour(line: str, theme: dict) -> str:
    if is_hit(line) or line.startswith("found by:"):
        return theme["hit"]
    if line in RETRIEVERS:
        return theme["accent"]
    if line.startswith("  ") or line.startswith("missed in"):
        return theme["muted"]
    return theme["ink"]


def _clip(line: str) -> str:
    return line if len(line) <= COLUMNS else line[:COLUMNS - 1] + "…"


def write(command: str, lines: list[str], out: Path, theme_name: str,
          comment: str = "") -> Path:
    theme = THEMES[theme_name]
    font = _font()
    step = round(font.size * 1.45)
    width = round(font.getlength("M") * COLUMNS) + 2 * PAD
    height = step * (len(lines) + 1) + 2 * PAD
    images, durations = [], []
    for typed, shown, ms in plan(command, lines):
        image = Image.new("RGB", (width, height), theme["bg"])
        draw = ImageDraw.Draw(image)
        draw.text((PAD, PAD), "$ ", font=font, fill=theme["muted"])
        draw.text((PAD + font.getlength("$ "), PAD), command[:typed],
                  font=font, fill=theme["ink"])
        for row, line in enumerate(lines[:shown], start=1):
            draw.text((PAD, PAD + row * step), _clip(line), font=font,
                      fill=_colour(line, theme))
        images.append(image.quantize(colors=32))
        durations.append(ms)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True,
                   comment=comment.encode())
    return out


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    command, lines = transcript(readme)
    for theme, name in [("light", "ask-q034.gif"), ("dark", "ask-q034-dark.gif")]:
        out = write(command, lines, MEDIA / name, theme, fingerprint(readme))
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
