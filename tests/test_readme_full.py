"""The README's "Scoring all 300" table must be what the result files say.

Both columns are read back from committed summaries (hierarchical chunks,
nDCG@10): the 58-question column from results/ and results/models/, the
296-question column from results/full/. The corpus is fetched, not committed,
so the runs themselves are not repeated here.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = {
    "MiniLM (default)": "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual-e5-small": "intfloat__multilingual-e5-small",
    "multilingual-e5-base": "intfloat__multilingual-e5-base",
}


def ndcg(summary: Path, retriever: str) -> float:
    rows = json.loads(summary.read_text(encoding="utf-8"))
    row = next(r for r in rows
               if r["chunking"] == "hierarchical" and r["retriever"] == retriever)
    return round(row["ndcg@10"], 3)


def readme_rows() -> dict[str, tuple[float, float]]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("### Scoring all 300", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for label, human, full in re.findall(
            r"^\| ([^|]+?) \| (\d\.\d{3}) \| (\d\.\d{3}) \|$", section, re.M):
        rows[label] = (float(human), float(full))
    return rows


def test_every_row_matches_its_summary():
    rows = readme_rows()
    assert len(rows) == 1 + 2 * len(MODELS)

    human_default = RESULTS / "summary.json"
    full_default = RESULTS / "full" / MODELS["MiniLM (default)"] / "summary.json"
    assert rows["bm25_stem5"] == (ndcg(human_default, "bm25_stem5"),
                                  ndcg(full_default, "bm25_stem5"))

    for label, folder in MODELS.items():
        human = (human_default if label == "MiniLM (default)"
                 else RESULTS / "models" / folder / "summary.json")
        full = RESULTS / "full" / folder / "summary.json"
        for retriever in ("dense", "hybrid_rrf"):
            assert rows[f"{label}, {retriever}"] == (
                ndcg(human, retriever), ndcg(full, retriever)), (label, retriever)
