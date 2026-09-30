"""The README's groundedness table must be what results/groundedness.json says.

The run itself calls two hosted models and cannot be repeated in CI, so this
only checks that the committed summary and the README agree, and that the
summary names the models, date and temperature the table is tied to.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "results" / "groundedness.json")
                     .read_text(encoding="utf-8"))["summary"]

ROWS = {
    "answered when the span was retrieved": "answered_when_retrieved",
    "answer rests on the passages": "grounded_when_answered",
    "answer conveys the gold span": "correct_when_answered",
    "declined when the span was not retrieved": "abstained_when_not_retrieved",
    "answered anyway": "answered_when_not_retrieved",
    "of those, answer is right": "correct_when_answered_without_span",
    "hallucinated (not in the passages)": "hallucination_rate",
}


def section() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return text.split("### Generation: groundedness", 1)[1].split("\n## ", 1)[0]


def test_every_rate_matches_the_summary():
    rows = dict(re.findall(r"^\| ([^|]+?) \| (\d+\.\d)% \|", section(), re.M))
    assert set(rows) == set(ROWS)
    for label, key in ROWS.items():
        assert float(rows[label]) == round(100 * SUMMARY[key], 1), label


def test_the_counts_and_provenance_are_stated():
    text = section()
    assert f"{SUMMARY['span_retrieved']} of {SUMMARY['queries']}" in text
    for key in ("answer_model", "judge_model", "date"):
        assert f"`{SUMMARY[key]}`" in text, key
    assert SUMMARY["temperature"] == 0 and "temperature 0" in text


def test_the_raw_model_output_is_not_committed():
    # The free API terms forbid redistributing it; only rates are published.
    assert "records" not in json.loads(
        (ROOT / "results" / "groundedness.json").read_text(encoding="utf-8"))
    assert not list((ROOT / "results").rglob("*.records.json")) or \
        "results/*.records.json" in (ROOT / ".gitignore").read_text()
