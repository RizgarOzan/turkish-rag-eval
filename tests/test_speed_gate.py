"""The CI speed gate: what it compares, and that its baseline covers every stage.

The timing itself runs as its own CI step (``python docs/speed_gate.py``), not
here: a wall-clock assertion inside the unit tests would make them flaky.
"""

import importlib.util
import json

from turkish_rag_eval.paths import ROOT

spec = importlib.util.spec_from_file_location("speed_gate", ROOT / "docs" / "speed_gate.py")
speed_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed_gate)

BASELINE = ROOT / "results" / "speed" / "baseline.json"


def test_a_stage_within_the_factor_passes():
    assert speed_gate.check({"chunk": 3.9, "search": 1.0},
                            {"chunk": 2.0, "search": 1.0}, factor=2.0) == []


def test_a_stage_past_the_factor_fails_and_is_named():
    failures = speed_gate.check({"chunk": 4.1, "search": 1.0},
                                {"chunk": 2.0, "search": 1.0}, factor=2.0)
    assert len(failures) == 1 and failures[0].startswith("chunk")


def test_a_stage_missing_from_the_baseline_fails():
    # a new stage must come with a baseline, or it is never gated
    assert speed_gate.check({"chunk": 1.0, "rerank": 1.0}, {"chunk": 1.0}, factor=2.0)


def test_the_committed_baseline_covers_exactly_the_measured_stages():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert set(baseline["ratios"]) == set(speed_gate.STAGES)
    assert all(v > 0 for v in baseline["ratios"].values())
