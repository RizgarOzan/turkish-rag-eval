"""The README's leaderboard intervals must be what the per-query files say.

Every interval and paired difference quoted under the model table is
recomputed here from results/ and results/models/ (hierarchical chunks,
nDCG@10), so a re-run that moves a number turns this red until the README
follows.
"""
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from turkish_rag_eval.metrics import paired_bootstrap_ci  # noqa: E402
from turkish_rag_eval.report import load_configurations, pair  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = {
    "MiniLM": RESULTS,
    "emrecan": RESULTS / "models" / "emrecan__bert-base-turkish-cased-mean-nli-stsb-tr",
    "e5-small": RESULTS / "models" / "intfloat__multilingual-e5-small",
    "e5-base": RESULTS / "models" / "intfloat__multilingual-e5-base",
    "Mursit": RESULTS / "models" / "newmindai__Mursit-Large-TR-Retrieval",
}
METRIC = "ndcg@10"


def configuration(model: str, retriever: str):
    return next(c for c in load_configurations(MODELS[model])
                if c.chunking == "hierarchical" and c.retriever == retriever)


def difference(a, b) -> str:
    x, y = pair(a, b, METRIC)
    low, high = paired_bootstrap_ci(x, y)
    mean = sum(x) / len(x) - sum(y) / len(y)
    return f"{mean:+.3f}\n[{low:+.3f}, {high:+.3f}]"


def section() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    body = text.split("## Leaderboard", 1)[1].split("\n## ", 1)[0]
    # Compare with line breaks normalised, so rewrapping the prose is free.
    return " ".join(body.split())


def test_every_interval_matches_its_per_query_file():
    text = section()
    for model in MODELS:
        dense = configuration(model, "dense")
        low, high = dense.interval(METRIC)
        quoted = f"{model} {dense.mean(METRIC):.3f} [{low:.3f}, {high:.3f}]"
        assert quoted in text, quoted


@pytest.mark.parametrize("a, b, retriever_a, retriever_b", [
    ("Mursit", "e5-base", "dense", "dense"),
    ("e5-base", "e5-small", "dense", "dense"),
    ("MiniLM", "emrecan", "dense", "dense"),
    ("Mursit", "Mursit", "hybrid_rrf", "dense"),
])
def test_every_paired_difference_matches(a, b, retriever_a, retriever_b):
    quoted = difference(configuration(a, retriever_a),
                        configuration(b, retriever_b))
    assert " ".join(quoted.split()) in section(), (a, b, quoted)
