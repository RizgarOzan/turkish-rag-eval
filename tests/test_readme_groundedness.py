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
#: One summary per extra generator, same questions and judge as the baseline.
OTHERS = [json.loads(p.read_text(encoding="utf-8"))["summary"]
          for p in sorted((ROOT / "results" / "groundedness").glob("*.json"))
          if not p.name.endswith(".records.json")]

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
    rows = dict(re.findall(r"^\| ([^|]+?) \| (\d+\.\d)% \|$", section(), re.M))
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


def test_no_other_section_says_the_results_are_missing():
    # Limits and Status once said "no committed results yet"; once the summary
    # exists, no line of the README may still claim that.
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "no committed results" not in text
    assert "no results committed" not in text


def generator_rows() -> dict:
    """The comparison table: model name -> its cells, from the README."""
    rows = {}
    for line in section().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 6 and cells[0].startswith("`"):
            rows[cells[0].split("`")[1]] = cells[1:]
    return rows


def test_every_extra_generator_has_a_row_that_matches_its_summary():
    rows = generator_rows()
    assert OTHERS, "results/groundedness/ holds the other generators"
    for summary in OTHERS + [SUMMARY]:
        assert summary["judge_model"] == SUMMARY["judge_model"]
        assert (summary["queries"], summary["span_retrieved"]) == (58, 33)
        cells = rows[summary["answer_model"]]
        expected = [f"{100 * summary[k]:.1f}%" for k in (
            "answered_when_retrieved", "grounded_when_answered",
            "correct_when_answered", "abstained_when_not_retrieved")]
        assert cells[:4] == expected, summary["answer_model"]
        missing = summary["span_not_retrieved"]
        n = round(summary["hallucination_rate"] * missing)
        rate = f"{100 * summary['hallucination_rate']:.1f}%"
        assert cells[4] == f"{rate} ({n}/{missing})", summary["answer_model"]
    assert set(rows) == {s["answer_model"] for s in OTHERS + [SUMMARY]}


def test_the_comparison_states_its_date_and_does_not_rank():
    text = section()
    for summary in OTHERS:
        assert f"`{summary['date']}`" in text
    assert "does not rank" in text
