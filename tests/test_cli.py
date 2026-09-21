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
