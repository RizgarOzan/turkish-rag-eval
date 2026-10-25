"""The README's int8 table must be what results/quantize/ says.

Each row is rebuilt from the int8 run and the fp32 one in results/models/,
paired bootstrap, encode time and P95 included, so a re-run of
docs/measure_int8.py that moves a number turns this red.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from turkish_rag_eval.metrics import ndcg_at_k, paired_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"intfloat__multilingual-e5-small": "e5-small",
          "newmindai__Mursit-Large-TR-Retrieval": "Mursit-Large-TR-Retrieval"}
README = (ROOT / "README.md").read_text(encoding="utf-8")
CHUNKINGS = ("fixed", "sentence", "hierarchical")


def runs(name: str) -> tuple[Path, Path]:
    return ROOT / "results" / "models" / name, ROOT / "results" / "quantize" / name


def per_query(run: Path, chunking: str) -> list[float]:
    rows = json.loads((run / f"perquery_{chunking}_dense.json").read_text(encoding="utf-8"))
    return [ndcg_at_k(r["relevance"], r["total_relevant"], 10) for r in rows if r["total_relevant"] > 0]


def summary(run: Path, chunking: str) -> dict:
    rows = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return next(r for r in rows if r["chunking"] == chunking and r["retriever"] == "dense")


def row(name: str, chunking: str) -> str:
    fp32_run, int8_run = runs(name)
    fp32, int8 = per_query(fp32_run, chunking), per_query(int8_run, chunking)
    low, high = paired_bootstrap_ci(int8, fp32)
    a, b = sum(fp32) / len(fp32), sum(int8) / len(int8)
    s32, s8 = summary(fp32_run, chunking), summary(int8_run, chunking)
    return (f"| {MODELS[name]} | {chunking} | {a:.3f} → {b:.3f} | "
            f"{b - a:+.3f} [{low:+.3f}, {high:+.3f}] | "
            f"{s32['dense_index_seconds']:.0f} s → {s8['dense_index_seconds']:.0f} s | "
            f"{s32['latency_ms_p95']:.0f} ms → {s8['latency_ms_p95']:.0f} ms |")


def test_every_int8_row_matches_the_results():
    for name in MODELS:
        for chunking in CHUNKINGS:
            assert row(name, chunking) in README, (name, chunking)


def test_the_text_says_which_losses_are_clear_of_zero():
    # e5-small: every interval below zero. Mursit: every interval spans zero.
    for name, clear in (("intfloat__multilingual-e5-small", True),
                        ("newmindai__Mursit-Large-TR-Retrieval", False)):
        fp32_run, int8_run = runs(name)
        for chunking in CHUNKINGS:
            _, high = paired_bootstrap_ci(per_query(int8_run, chunking),
                                          per_query(fp32_run, chunking))
            assert (high < 0) == clear, (name, chunking)


def test_the_int8_rows_say_what_they_are():
    for name in MODELS:
        rows = json.loads((runs(name)[1] / "summary.json").read_text(encoding="utf-8"))
        assert all(r["quantization"] == "int8" and r["queries"] == 58 for r in rows)
