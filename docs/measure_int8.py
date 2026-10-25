"""Runs the dense retriever with torch dynamic int8 Linear layers (CPU).

This is a measurement, not a `run` flag: torch.ao.quantization is deprecated
and slated for removal, so it stays out of the CLI.
The model is wrapped as it loads and everything else is the normal `run`,
dense only. Rows land in results/quantize/<org>__<name>/ tagged
"quantization": "int8"; tests/test_readme_quantize.py checks the README
against them.

Usage:  python docs/measure_int8.py <model>
"""

import argparse
import json
import sys
from pathlib import Path

import sentence_transformers
import torch

from turkish_rag_eval import run_eval

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    model = sys.argv[1]
    out = ROOT / "results" / "quantize" / model.replace("/", "__")

    load = sentence_transformers.SentenceTransformer

    def int8(name, *args, **kwargs):
        return torch.ao.quantization.quantize_dynamic(
            load(name, *args, **kwargs), {torch.nn.Linear}, dtype=torch.qint8)

    sentence_transformers.SentenceTransformer = int8
    parser = argparse.ArgumentParser()
    run_eval.add_arguments(parser)
    run_eval.run(parser.parse_args(
        ["--model", model, "--retrievers", "dense", "--out", str(out)]))

    summary = out / "summary.json"
    rows = json.loads(summary.read_text(encoding="utf-8"))
    for row in rows:
        row["quantization"] = "int8"
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
