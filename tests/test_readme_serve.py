"""The README's serve load-test table must be what results/serve/ says, so a
re-run of docs/load_test.py that moves a number turns this red."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
RESULT = json.loads((ROOT / "results" / "serve" / "bm25_stem5.json").read_text(encoding="utf-8"))


def test_every_load_level_matches_the_results():
    for level in RESULT["levels"]:
        row = (f"| {level['concurrency']} | {level['latency_ms_p50']} ms | "
               f"{level['latency_ms_p95']} ms | {level['latency_ms_p99']} ms | "
               f"{level['requests_per_second']:.0f} | {level['errors']} |")
        assert row in README, row


def test_the_run_is_the_gold_set_against_the_shipped_corpus():
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    assert RESULT["retriever"] == "bm25_stem5" and RESULT["questions"] == 58
    assert f"over {RESULT['chunks']} hierarchical chunks" in README
    assert all(level["requests"] == 580 for level in RESULT["levels"])
    assert {r["queries"] for r in summary} == {RESULT["questions"]}
