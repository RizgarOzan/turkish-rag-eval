"""The dispatcher is hand-rolled, so its argument handling is worth pinning.

The bug these guard against is real and was hit during development: an
``add_subparsers``-based dispatcher swallows any option it does not itself
declare, so ``run --model X`` died before ``run_eval`` was imported.
"""

import pytest

from turkish_rag_eval import __version__, cli


def test_every_command_names_an_importable_module():
    import importlib
    for command, (module_name, _) in cli.COMMANDS.items():
        try:
            importlib.import_module(f"turkish_rag_eval.{module_name}")
        except ImportError as exc:
            # An optional dependency that is simply not installed here is not a
            # broken command; a missing module file is.
            if f"turkish_rag_eval.{module_name}" in str(exc):
                pytest.fail(f"'{command}' points at a module that does not exist")


def test_help_lists_every_command():
    text = cli.format_help()
    for command in cli.COMMANDS:
        assert command in text


def test_no_arguments_is_an_error_but_explicit_help_is_not(capsys):
    assert cli.main([]) == 2
    assert cli.main(["--help"]) == 0
    assert "commands:" in capsys.readouterr().out


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_command_is_rejected(capsys):
    assert cli.main(["definitely-not-a-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_command_options_reach_the_module(capsys):
    # --online belongs to validate_gold, not to the dispatcher. Passing it
    # must not trip the top-level parser.
    assert cli.main(["validate", "--online"]) == 0
    assert "hata" in capsys.readouterr().out


def test_command_help_is_the_modules_own(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["validate", "--help"])
    assert exit_info.value.code == 0
    assert "--online" in capsys.readouterr().out


def test_a_lazily_imported_extra_gets_the_friendly_message(capsys, monkeypatch):
    # charts reaches for matplotlib inside the drawing function, not at module
    # load, so the import error escapes run() rather than the import. It still
    # has to name the extra instead of printing a traceback.
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    assert cli.main(["charts"]) == 2
    message = capsys.readouterr().err
    assert "turkish-rag-eval[charts]" in message
    assert "Traceback" not in message


def test_the_extras_map_only_names_real_extras():
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml")
        .read_text(encoding="utf-8"))
    declared = set(pyproject["project"]["optional-dependencies"])
    assert set(cli.EXTRAS.values()) <= declared
