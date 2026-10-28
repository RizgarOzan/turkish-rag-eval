"""The README's GPU table must be what results/gpu/ says.

Each row is rebuilt from the GPU run and the CPU one in results/models/. The
GPU changes timing only, so nDCG@10 must match the CPU run exactly.
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
    cpu, gpu = summary("models", name, chunking), summary("gpu", name, chunking)
    return (f"| {MODELS[name]} | {chunking} | "
            f"{cpu['dense_index_seconds']:.0f} s → {gpu['dense_index_seconds']:.0f} s | "
            f"{cpu['latency_ms_p95']:.0f} ms → {gpu['latency_ms_p95']:.0f} ms |")


def test_every_gpu_row_matches_the_results():
    for name in MODELS:
        for chunking in CHUNKINGS:
            assert row(name, chunking) in README, (name, chunking)


def test_the_gpu_finds_exactly_what_the_cpu_finds():
    for name in MODELS:
        for chunking in CHUNKINGS:
            cpu, gpu = summary("models", name, chunking), summary("gpu", name, chunking)
            assert round(cpu["ndcg@10"], 6) == round(gpu["ndcg@10"], 6), (name, chunking)
