"""The README's stress table must be what results/stress/ says.

Each row is rebuilt from the per-model JSON that `turkish-rag-eval stress`
writes, so a re-run that moves a number turns this red until the README
follows. The finding quoted above the table is checked the same way.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRESS = ROOT / "results" / "stress"
README = (ROOT / "README.md").read_text(encoding="utf-8")

ROWS = {
    "MiniLM (default)": "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2",
    "emrecan (Turkish)": "emrecan__bert-base-turkish-cased-mean-nli-stsb-tr",
    "e5-small": "intfloat__multilingual-e5-small",
    "e5-base": "intfloat__multilingual-e5-base",
    "Mursit (Turkish)": "newmindai__Mursit-Large-TR-Retrieval",
}


def scores(name: str) -> dict:
    data = json.loads((STRESS / f"{name}.json").read_text(encoding="utf-8"))
    return {key: value["ndcg@10"] for key, value in data["scores"].items()}


def row(label: str, s: dict, retriever: str) -> str:
    cells = [f"{s[f'{retriever}/original']:.3f}"]
    for variant in ("ascii", "caps", "naive_lower"):
        cells.append(f"{s[f'{retriever}/{variant}']:.3f} → "
                     f"{s[f'{retriever}/{variant}/normalized']:.3f}")
    return f"| {label} | " + " | ".join(cells) + " |"


def test_every_dense_row_matches_its_results_file():
    for label, name in ROWS.items():
        assert row(label, scores(name), "dense") in README, label


def test_the_bm25_row_matches():
    s = scores(ROWS["MiniLM (default)"])
    assert row("stemmed BM25", s, "bm25_stem5") in README


def test_the_normaliser_leaves_correct_questions_alone():
    # The README says so; every retriever of every model must agree.
    for name in ROWS.values():
        s = scores(name)
        for retriever in ("bm25_stem5", "dense", "hybrid_rrf"):
            assert s[f"{retriever}/original/normalized"] == s[f"{retriever}/original"]


def test_the_quoted_finding_matches():
    minilm = scores(ROWS["MiniLM (default)"])
    bm25 = minilm
    assert f"from {minilm['dense/original']:.3f} to {minilm['dense/caps']:.3f}" in README
    assert (f"{bm25['bm25_stem5/ascii']:.3f} → {bm25['bm25_stem5/ascii/normalized']:.3f}"
            in README)
