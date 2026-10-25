"""Answer retrieval queries over HTTP.

    turkish-rag-eval serve                                # BM25, port 8000
    turkish-rag-eval serve --model intfloat/multilingual-e5-small   # hybrid

    GET /health              {"status": "ok", "retriever": ..., "chunks": ..., "corpus": ...}
    GET /search?q=...&k=5    {"query": ..., "retriever": ..., "results": [...]}

The index is built once at start-up from the same chunks and retrievers that
``run`` scores, so a result here is a result the README measured. Without
``--model`` it serves stemmed BM25 and needs nothing beyond the base install;
with one it serves the RRF hybrid of that model and BM25.

It is the standard library's threaded HTTP server, not a framework: two
routes do not need one, and the base install stays at three dependencies.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .chunking import STRATEGIES
from .corpus import load_corpus
from .run_eval import CORPUS

MAX_K = 50


class Server(ThreadingHTTPServer):
    # the stdlib default of 5 pending connections makes the 6th concurrent
    # client wait out a TCP retry (~0.5 s on Windows) before it is even accepted
    request_queue_size = 128


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunking", default="hierarchical", choices=list(STRATEGIES))
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--model", default=None,
                        help="serve the hybrid of this embedding model and BM25 "
                             "(default: BM25 only, no torch)")
    parser.add_argument("--embedding-cache", type=Path, default=None, metavar="DIR",
                        help="reuse dense chunk embeddings cached by `run`")


def build(chunks: list[dict], model: str | None = None,
          embedding_cache: Path | None = None) -> tuple[str, object]:
    """The retriever ``serve`` answers with, and its name in the results."""
    from .retrieval import SparseRetriever

    sparse = SparseRetriever(chunks, stem_length=5)
    if model is None:
        return "bm25_stem5", sparse
    from sentence_transformers import SentenceTransformer

    from .models import prefixes_for
    from .retrieval import DenseRetriever, HybridRetriever
    query_prefix, doc_prefix = prefixes_for(model)
    dense = DenseRetriever(chunks, SentenceTransformer(model), query_prefix, doc_prefix,
                           cache_dir=embedding_cache, model_name=model)
    return "hybrid_rrf", HybridRetriever(dense, sparse)


def make_handler(chunks: list[dict], name: str, retriever, corpus: str):
    health = {"status": "ok", "retriever": name, "chunks": len(chunks), "corpus": corpus}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self._send(200, health)
            if url.path != "/search":
                return self._send(404, {"error": f"no route {url.path}"})
            params = parse_qs(url.query)
            query = " ".join(params.get("q", [""])).strip()
            if not query:
                return self._send(400, {"error": "missing q"})
            try:
                k = int(params.get("k", ["5"])[0])
            except ValueError:
                k = 0
            if not 1 <= k <= MAX_K:
                return self._send(400, {"error": f"k must be 1..{MAX_K}"})
            results = [{"rank": rank, "score": round(float(score), 6),
                        "chunk_id": chunks[idx]["chunk_id"],
                        "title": chunks[idx]["title"],
                        "heading_path": chunks[idx]["heading_path"],
                        "url": chunks[idx]["url"],
                        "text": chunks[idx]["body"]}
                       for rank, (idx, score) in enumerate(retriever.search(query, k), 1)]
            self._send(200, {"query": query, "retriever": name, "results": results})

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # one line per request is noise under load
            pass

    return Handler


def run(args) -> int:
    corpus = load_corpus(args.corpus or CORPUS)
    chunks = [c for doc in corpus.docs for c in STRATEGIES[args.chunking](doc)]
    started = time.perf_counter()
    name, retriever = build(chunks, args.model, args.embedding_cache)
    handler = make_handler(chunks, name, retriever, corpus.fingerprint)
    server = Server((args.host, args.port), handler)
    print(f"{name} over {len(chunks)} {args.chunking} chunks, index built in "
          f"{time.perf_counter() - started:.1f} s")
    print(f"listening on http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
