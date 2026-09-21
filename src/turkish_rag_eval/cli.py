"""One entry point for the whole harness: ``turkish-rag-eval <command>``.

Every command is a thin wrapper over a module that also runs standalone, so
``python -m turkish_rag_eval.run_eval`` and ``turkish-rag-eval run`` do the
same thing. Each module exposes ``add_arguments(parser)`` and ``run(args)``;
this file only picks the module and hands it the rest of the command line.

Two deliberate choices here:

Dispatch is done by hand rather than with ``add_subparsers``. A subparser
would have to claim its own options to build help, which means it also claims
``--help`` and anything it does not recognise - so ``... run --model X`` fails
before the module that defines ``--model`` is ever imported. Slicing ``argv``
keeps every option the property of the command that declares it.

Modules are imported inside the handler, because they pull in very different
dependencies: ``validate`` needs nothing but the standard library, ``run``
needs numpy and, for dense retrieval, torch. Someone who installed the base
package should still be able to validate a gold set.
"""

import argparse
import importlib
import sys

from . import __version__

#: command -> (module, one-line help). The module must expose ``run(args)``
#: and may expose ``add_arguments(parser)``.
COMMANDS = {
    "run": ("run_eval", "evaluate every chunking x retriever combination"),
    "report": ("report", "turn results into a recommendation"),
    "abstain": ("run_abstain", "produce abstain records and the coverage curve"),
    "abstain-report": ("abstain", "re-print the coverage curve from saved records"),
    "charts": ("charts", "render the result charts the README embeds"),
    "leaderboard": ("leaderboard", "rebuild the model leaderboard from results/"),
    "agreement": ("agreement", "inter-annotator agreement over the gold set"),
    "fetch-corpus": ("fetch_corpus", "download the Wikipedia corpus snapshot"),
    "export-hf": ("export_hf", "export the BEIR layout MTEB reads"),
    "validate": ("validate_gold", "check every gold file's invariants"),
}

USAGE = """usage: turkish-rag-eval <command> [options]

Measure which parts of a Turkish RAG pipeline pay off.

commands:
{commands}

Run 'turkish-rag-eval <command> --help' for a command's own options.
"""


def format_help() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"  {name:<{width}}  {help_text}"
             for name, (_, help_text) in COMMANDS.items()]
    return USAGE.format(commands="\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        print(format_help(), end="")
        return 2
    if argv[0] in ("-h", "--help"):
        print(format_help(), end="")
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"turkish-rag-eval {__version__}")
        return 0

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"turkish-rag-eval: unknown command '{command}'\n", file=sys.stderr)
        print(format_help(), end="", file=sys.stderr)
        return 2

    module_name, help_text = COMMANDS[command]
    try:
        module = importlib.import_module(f".{module_name}", __package__)
    except ImportError as exc:
        # A missing optional dependency should name the extra that provides it
        # rather than surface as a traceback from three imports down.
        print(f"turkish-rag-eval: '{command}' is unavailable: {exc}\n"
              f"Install the optional dependencies with: "
              f"pip install 'turkish-rag-eval[all]'", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(
        prog=f"turkish-rag-eval {command}",
        description=module.__doc__ or help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    if hasattr(module, "add_arguments"):
        module.add_arguments(parser)
    return module.run(parser.parse_args(rest)) or 0


if __name__ == "__main__":
    raise SystemExit(main())
