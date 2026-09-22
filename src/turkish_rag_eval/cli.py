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

from . import __version__, paths
from .corpus import CorpusError

#: command -> (module, one-line help). The module must expose ``run(args)``
#: and may expose ``add_arguments(parser)``.
COMMANDS = {
    "run": ("run_eval", "evaluate every chunking x retriever combination"),
    "report": ("report", "turn results into a recommendation"),
    "abstain": ("run_abstain", "produce abstain records and the coverage curve"),
    "abstain-report": ("abstain", "re-print the coverage curve from saved records"),
    "charts": ("charts", "render the result charts the README embeds"),
    "leaderboard": ("leaderboard", "rebuild the model leaderboard from results/"),
    "groundedness": ("groundedness", "score answers against the gold answer spans"),
    "bootstrap": ("bootstrap", "draft a gold set for your own corpus"),
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
        return _missing_dependency(command, exc)

    parser = argparse.ArgumentParser(
        prog=f"turkish-rag-eval {command}",
        description=module.__doc__ or help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    if hasattr(module, "add_arguments"):
        module.add_arguments(parser)

    try:
        return module.run(parser.parse_args(rest)) or 0
    except ImportError as exc:
        # Not every optional dependency is imported at module load. charts
        # reaches for matplotlib inside the function that draws, so a missing
        # extra surfaced as a traceback from four frames down until this
        # handler existed.
        return _missing_dependency(command, exc)
    except CorpusError as exc:
        return _missing_corpus(exc)


def _missing_corpus(exc: CorpusError) -> int:
    message = f"turkish-rag-eval: {exc}"
    # A wheel ships code only, so a bare `run` after `pip install` lands here.
    if not (paths.ROOT / paths.MARKER).exists():
        message += ("\nThe bundled Wikipedia corpus comes with a checkout: "
                    "git clone https://github.com/RizgarOzan/turkish-rag-eval")
    print(message, file=sys.stderr)
    return 2


#: Which extra provides which module, so the message names the fix.
EXTRAS = {
    "matplotlib": "charts",
    "anthropic": "llm",
    "torch": "dense",
    "sentence_transformers": "dense",
}


def _missing_dependency(command: str, exc: ImportError) -> int:
    # The top-level package, because a failure inside a submodule reports the
    # submodule: a missing matplotlib can surface as either "matplotlib" or
    # "matplotlib.pyplot" depending on where the import broke, and only the
    # distribution name maps to an extra.
    missing = (getattr(exc, "name", None) or "").split(".")[0]
    extra = EXTRAS.get(missing, "all")
    print(f"turkish-rag-eval: '{command}' needs a dependency that is not "
          f"installed: {exc}\n"
          f"Install it with: pip install 'turkish-rag-eval[{extra}]'",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
