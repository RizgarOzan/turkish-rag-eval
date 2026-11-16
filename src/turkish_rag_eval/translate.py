"""Translate the eval questions to English for the query-language experiment.

See docs/preregistration.md. Each question is translated once by
claude-sonnet-5-5 through the CLI; the passages are not touched. Writes the
same files under data/eval/en/ with ``question`` in English and the Turkish
original kept as ``question_tr``, plus docs/translation-review.md for checking
the translations by hand.
"""
import json
import random
from pathlib import Path

from .groundedness import ClaudeCliClient

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval"
OUT = EVAL / "en"
MODEL = "claude-sonnet-5-5"
SYSTEM = ("Translate the user's Turkish question into natural English. Keep "
          "names, numbers, dates and quoted terms as they are. Do not answer "
          "it, explain it or add anything. Reply with the English question only.")


def translate_file(src: Path, dst: Path, client: ClaudeCliClient) -> list[dict]:
    items = json.loads(src.read_text(encoding="utf-8"))
    out = []
    for i, item in enumerate(items, 1):
        english = client.complete(SYSTEM, item["question"], max_tokens=200)
        out.append({**item, "question": english, "question_tr": item["question"]})
        print(f"{src.name} {i}/{len(items)}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return out


def review_table(items: list[dict]) -> str:
    rows = ["| qid | Turkish | English |", "|---|---|---|"]
    for it in items:
        tr, en = (s.replace("|", "\\|") for s in (it["question_tr"], it["question"]))
        rows.append(f"| {it['qid']} | {tr} | {en} |")
    return "\n".join(rows)


def main() -> None:
    client = ClaudeCliClient(MODEL, cache=ROOT / ".cache" / "translate.cache.jsonl")
    (ROOT / ".cache").mkdir(exist_ok=True)
    gold = translate_file(EVAL / "gold.json", OUT / "gold.json", client)
    drafted = []
    for src in sorted((EVAL / "contrib").glob("*.json")):
        drafted += translate_file(src, OUT / "contrib" / src.name, client)
    sample = sorted(random.Random(0).sample(drafted, 25), key=lambda it: it["qid"])
    (ROOT / "docs" / "translation-review.md").write_text(
        "# Translation review\n\n"
        f"Made by `{MODEL}` (see docs/preregistration.md). Mark any row whose "
        "English changes the meaning, and fix it in data/eval/en/.\n\n"
        f"## The 58 labelled questions (all checked)\n\n{review_table(gold)}\n\n"
        f"## 25 of the {len(drafted)} drafted questions (random, seed 0)\n\n"
        f"{review_table(sample)}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
