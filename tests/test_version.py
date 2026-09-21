"""The version is what a citation points at, so the three copies must agree.

A result quoted as "measured on v0.1.0" is only meaningful if the package, the
build metadata and the changelog say the same thing. They live in three files
and drift silently otherwise.
"""

import re
from pathlib import Path

from turkish_rag_eval import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_matches_the_package():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    assert match, "pyproject.toml has no version"
    assert match.group(1) == __version__


def test_changelog_documents_this_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in changelog, (
        f"CHANGELOG.md has no section for {__version__}")


def test_version_is_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
