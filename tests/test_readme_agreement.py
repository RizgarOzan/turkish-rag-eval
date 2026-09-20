"""The README's agreement row must be what the draft files actually say.

Per-batch rows are history: each was measured when the batch landed. The
"All drafts" row is the one that moves every week, so it is recomputed here
from the files themselves. Chunk-level kappa is not checked: it needs the
corpus, which is fetched, not committed.
"""
import json
import re
from pathlib import Path

from agreement import span_f1, span_iou

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = sorted((ROOT / "data" / "eval" / "contrib").glob("llm-draft-*.json"))
ROW = re.compile(r"^\|\s*\*\*All drafts\*\*\s*\|(.+)\|\s*$", re.M)


def pairs():
    for path in DRAFTS:
        for item in json.loads(path.read_text(encoding="utf-8")):
            second = item.get("second_annotation")
            if second:
                yield item["answer_span"], second["answer_span"]


def readme_row():
    match = ROW.search((ROOT / "README.md").read_text(encoding="utf-8"))
    assert match, "README has no '| **All drafts** |' agreement row"
    return [cell.strip() for cell in match.group(1).split("|")]


def test_overall_agreement_row_matches_the_files():
    spans = list(pairs())
    questions, identical, iou, f1, _kappa = readme_row()
    assert int(questions) == len(spans)
    assert int(identical) == sum(span_f1(a, b) >= 0.9999 for a, b in spans)
    assert float(iou) == round(sum(span_iou(a, b) for a, b in spans) / len(spans), 3)
    assert float(f1) == round(sum(span_f1(a, b) for a, b in spans) / len(spans), 3)


def test_every_draft_carries_a_second_label():
    items = [i for p in DRAFTS for i in json.loads(p.read_text(encoding="utf-8"))]
    assert [i["qid"] for i in items if "second_annotation" not in i] == []
