"""Runs the dense retriever through the ONNX Runtime backend (CPU).

A measurement, not a `run` flag: it needs `sentence-transformers[onnx]`,
which the base install does not ask for. The model is exported to ONNX as it
loads and everything else is the normal `run`, dense only. Rows land in
results/onnx/<org>__<name>/ tagged "backend": "onnx"; tests/test_readme_onnx.py
checks the README against them.

Usage:  python docs/measure_onnx.py <model>
"""

import argparse
import json
import sys
from pathlib import Path

import sentence_transformers

from turkish_rag_eval import run_eval

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    model = sys.argv[1]
    out = ROOT / "results" / "onnx" / model.replace("/", "__")

    load = sentence_transformers.SentenceTransformer

    def onnx(name, *args, **kwargs):
        return load(name, *args, backend="onnx", device="cpu", **kwargs)

    sentence_transformers.SentenceTransformer = onnx
    parser = argparse.ArgumentParser()
    run_eval.add_arguments(parser)
    run_eval.run(parser.parse_args(
        ["--model", model, "--retrievers", "dense", "--out", str(out)]))

    summary = out / "summary.json"
    rows = json.loads(summary.read_text(encoding="utf-8"))
    for row in rows:
        row["backend"] = "onnx"
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
