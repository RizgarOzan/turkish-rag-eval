"""Every CLI command has a row in the README's command table, so a new
command cannot ship undocumented."""
from pathlib import Path

from turkish_rag_eval.cli import COMMANDS

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def test_every_command_has_a_row_in_the_readme_table():
    missing = [name for name in COMMANDS if f"| `{name}` |" not in README]
    assert not missing, missing
