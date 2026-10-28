"""The README's ONNX table must be what results/onnx/ says.

Each row is rebuilt from the ONNX run and the torch one in results/models/.
The runtime changes timing, and the float arithmetic may differ in the last
digits, so nDCG@10 must match to four decimals.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"intfloat__multilingual-e5-small": "e5-small",
          "newmindai__Mursit-Large-TR-Retrieval": "Mursit-Large-TR-Retrieval"}
README = (ROOT / "README.md").read_text(encoding="utf-8")
CHUNKINGS = ("fixed", "sentence", "hierarchical")


def summary(folder: str, name: str, chunking: str) -> dict:
    rows = json.loads((ROOT / "results" / folder / name / "summary.json").read_text(encoding="utf-8"))
    return next(r for r in rows if r["chunking"] == chunking and r["retriever"] == "dense")


def row(name: str, chunking: str) -> str:
    torch, onnx = summary("models", name, chunking), summary("onnx", name, chunking)
    return (f"| {MODELS[name]} | {chunking} | "
            f"{torch['dense_index_seconds']:.0f} s → {onnx['dense_index_seconds']:.0f} s | "
            f"{torch['latency_ms_p95']:.0f} ms → {onnx['latency_ms_p95']:.0f} ms |")


def test_every_onnx_row_matches_the_results():
    for name in MODELS:
        for chunking in CHUNKINGS:
            assert row(name, chunking) in README, (name, chunking)


def test_onnx_finds_what_torch_finds():
    for name in MODELS:
        for chunking in CHUNKINGS:
            torch, onnx = summary("models", name, chunking), summary("onnx", name, chunking)
            assert round(torch["ndcg@10"], 4) == round(onnx["ndcg@10"], 4), (name, chunking)
            assert onnx["backend"] == "onnx", (name, chunking)
