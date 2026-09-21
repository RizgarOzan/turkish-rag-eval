"""Per-model settings for the dense retriever.

E5 models were trained with "query: " / "passage: " in front of every input
and their model cards require the same at inference. Models that ship their
own prompts, or need none, get empty prefixes.
"""

from .gold import ROOT

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

E5_PREFIXES = ("query: ", "passage: ")
MODEL_PREFIXES = {
    "intfloat/multilingual-e5-small": E5_PREFIXES,
    "intfloat/multilingual-e5-base": E5_PREFIXES,
}


def prefixes_for(model: str) -> tuple[str, str]:
    """(query prefix, document prefix) for ``model``."""
    return MODEL_PREFIXES.get(model, ("", ""))


def results_dir(model: str):
    """The default model writes to results/, which the README's main table is
    built from; every other model gets results/models/<org>__<name>/."""
    if model == DEFAULT_MODEL:
        return ROOT / "results"
    return ROOT / "results" / "models" / model.replace("/", "__")
