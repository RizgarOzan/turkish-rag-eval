"""Dense, sparse and hybrid retrievers over a fixed chunk list.

Dense  - multilingual sentence embeddings, cosine similarity.
Sparse - BM25 over Turkish-normalised tokens (see turkish_text).
Hybrid - Reciprocal Rank Fusion of the two ranked lists.

RRF is used instead of score interpolation because cosine similarity and
BM25 scores live on different, corpus-dependent scales; fusing ranks needs
no per-corpus tuning. k=60 is the value from Cormack et al. (2009).
"""

import numpy as np
from rank_bm25 import BM25Okapi

from turkish_text import tokenize

RRF_K = 60


class DenseRetriever:
    def __init__(self, chunks: list[dict], model, query_prefix: str = "",
                 doc_prefix: str = ""):
        self.chunks = chunks
        self.model = model
        self.query_prefix = query_prefix
        embeddings = model.encode(
            [doc_prefix + c["embed_text"] for c in chunks],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.matrix = embeddings.astype(np.float32)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        vector = self.model.encode(
            [self.query_prefix + query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        scores = self.matrix @ vector  # both normalised -> cosine
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class SparseRetriever:
    def __init__(self, chunks: list[dict], stem_length: int | None = 5):
        self.chunks = chunks
        self.stem_length = stem_length
        corpus = [tokenize(c["embed_text"], stem_length) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query, self.stem_length))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


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
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(idx, score) for idx, score in ranked]
