"""Dense, sparse and hybrid retrievers over a fixed chunk list.

Dense  - multilingual sentence embeddings, cosine similarity.
Sparse - BM25 over Turkish-normalised tokens (see turkish_text).
Hybrid - Reciprocal Rank Fusion of the two ranked lists.

RRF is used instead of score interpolation because cosine similarity and
BM25 scores live on different, corpus-dependent scales; fusing ranks needs
no per-corpus tuning. k=60 is the value from Cormack et al. (2009).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .turkish_text import tokenize

RRF_K = 60


def _rank(scores, k: int) -> list[tuple[int, float]]:
    """Top ``k`` (index, score) pairs, ties broken by the lower chunk index.

    ``np.argsort`` defaults to an introsort that is *not* stable, so two chunks
    with identical scores could swap places between runs on the same data -
    which is invisible in aggregate but moves a top-1 metric, and makes a
    published number impossible to reproduce exactly. Sparse scores tie often:
    every chunk sharing no query term scores exactly 0.0.
    """
    top = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    """Cosine search over chunk embeddings.

    With ``cache_dir`` and ``model_name`` the embedding matrix is stored as
    ``<sha256>.npz``, the hash taken over the model name and every prefixed
    chunk text, so any change to the chunks re-encodes. The model's weights are
    not hashed: a model re-published under the same name needs a fresh cache.
    ``encode_seconds`` is the time the encoding took when it actually ran, also
    on a cache hit, so a cached run does not report a 0 s index.
    """

    def __init__(self, chunks: list[dict], model, query_prefix: str = "",
                 doc_prefix: str = "", cache_dir: Path | None = None,
                 model_name: str | None = None):
        self.chunks = chunks
        self.model = model
        self.query_prefix = query_prefix
        texts = [doc_prefix + c["embed_text"] for c in chunks]
        path = None
        if cache_dir is not None and model_name is not None:
            key = hashlib.sha256(json.dumps(
                [model_name, texts], ensure_ascii=False).encode("utf-8"))
            path = Path(cache_dir) / f"{key.hexdigest()}.npz"
        self.cache_hit = path is not None and path.exists()
        if self.cache_hit:
            with np.load(path) as stored:
                self.matrix = stored["matrix"]
                self.encode_seconds = float(stored["encode_seconds"])
            return
        start = time.perf_counter()
        embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.encode_seconds = time.perf_counter() - start
        self.matrix = embeddings.astype(np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            with partial.open("wb") as f:
                np.savez(f, matrix=self.matrix, encode_seconds=self.encode_seconds)
            os.replace(partial, path)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        vector = self.model.encode(
            [self.query_prefix + query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        scores = self.matrix @ vector  # both normalised -> cosine
        return _rank(scores, k)


class SparseRetriever:
    def __init__(self, chunks: list[dict], stem_length: int | None = 5):
        self.chunks = chunks
        self.stem_length = stem_length
        corpus = [tokenize(c["embed_text"], stem_length) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query, self.stem_length))
        return _rank(scores, k)


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever,
                 depth: int = 50):
        self.dense = dense
        self.sparse = sparse
        self.depth = depth

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for retriever in (self.dense, self.sparse):
            for rank, (idx, _) in enumerate(retriever.search(query, self.depth)):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        # Tie-break on the index rather than relying on insertion order, which
        # depends on which retriever happened to surface a chunk first.
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(idx, score) for idx, score in ranked]


class RerankRetriever:
    """Reorder the base retriever's top ``depth`` with a cross-encoder.

    A cross-encoder reads query and passage together, so it is far more
    accurate than an embedding comparison and far too slow to run over the
    whole corpus: it only sees what the cheap first stage already found.
    Ties keep the base order, for the same reproducibility reason as ``_rank``.
    """

    def __init__(self, base, chunks: list[dict], cross_encoder, depth: int = 20):
        self.base = base
        self.chunks = chunks
        self.cross_encoder = cross_encoder
        self.depth = depth

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        hits = self.base.search(query, self.depth)
        if not hits:
            return []
        scores = self.cross_encoder.predict(
            [(query, self.chunks[idx]["embed_text"]) for idx, _ in hits],
            show_progress_bar=False)
        order = sorted(range(len(hits)), key=lambda j: (-float(scores[j]), j))[:k]
        return [(hits[j][0], float(scores[j])) for j in order]
