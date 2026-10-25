"""Load-test a running ``turkish-rag-eval serve`` with the 58 gold questions.

    turkish-rag-eval serve &
    python docs/load_test.py --url http://127.0.0.1:8000 --out results/serve/bm25_stem5.json

For each concurrency level, that many client threads send the questions in
turn (k=5) until ``--requests`` answers are in, after one untimed warm-up pass.
Latency is wall time per request seen by the client, so it includes HTTP and
JSON; the client runs on the same machine and competes for the same CPU.
"""

import argparse
import json
import platform
import statistics
import threading
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from turkish_rag_eval.gold import load_gold, resolve_gold


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def fetch(url: str) -> None:
    with urlopen(url, timeout=120) as response:
        response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")


def level(base: str, questions: list[str], concurrency: int, requests: int) -> dict:
    urls = [f"{base}/search?k=5&q={quote(q)}" for q in questions]
    latencies: list[float] = []
    errors = 0
    lock = threading.Lock()
    counter = iter(range(requests))

    def worker():
        nonlocal errors
        while True:
            with lock:
                n = next(counter, None)
            if n is None:
                return
            started = time.perf_counter()
            try:
                fetch(urls[n % len(urls)])
            except Exception:
                with lock:
                    errors += 1
                continue
            elapsed = (time.perf_counter() - started) * 1000
            with lock:
                latencies.append(elapsed)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started
    return {"concurrency": concurrency, "requests": requests, "errors": errors,
            "latency_ms_p50": round(statistics.median(latencies), 1),
            "latency_ms_p95": round(percentile(latencies, 95), 1),
            "latency_ms_p99": round(percentile(latencies, 99), 1),
            "requests_per_second": round(len(latencies) / wall, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--requests", type=int, default=580)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    questions = [g["question"] for g in load_gold(resolve_gold(None))]
    with urlopen(f"{args.url}/health", timeout=10) as response:
        health = json.loads(response.read().decode("utf-8"))
    for q in questions:
        fetch(f"{args.url}/search?k=5&q={quote(q)}")

    levels = []
    for c in args.concurrency:
        row = level(args.url, questions, c, args.requests)
        print(json.dumps(row))
        levels.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "retriever": health["retriever"], "chunks": health["chunks"],
        "corpus": health["corpus"], "questions": len(questions),
        "date": date.today().isoformat(), "python": platform.python_version(),
        "machine": platform.processor() or platform.machine(),
        "levels": levels}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
