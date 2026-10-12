"""The reranker reorders the base retriever's top ``depth`` with a cross-encoder.
A fake cross-encoder keeps the test free of model downloads."""

import pytest

pytest.importorskip("numpy")

from turkish_rag_eval.retrieval import RerankRetriever  # noqa: E402

CHUNKS = [{"embed_text": text} for text in ("alfa", "beta", "gama", "delta")]


class FixedBase:
    def __init__(self, hits):
        self.hits = hits
        self.asked = None

    def search(self, query, k):
        self.asked = k
        return self.hits[:k]


class LengthScorer:
    """Scores a passage by its length, so the expected order is obvious."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, **_):
        self.pairs = pairs
        return [len(passage) for _, passage in pairs]


def test_reorders_the_base_hits_by_cross_encoder_score():
    base = FixedBase([(0, 0.9), (2, 0.8), (3, 0.7)])
    reranker = RerankRetriever(base, CHUNKS, LengthScorer(), depth=20)
    assert [idx for idx, _ in reranker.search("soru", 3)] == [3, 0, 2]


def test_asks_the_base_for_depth_and_returns_k():
    base = FixedBase([(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)])
    scorer = LengthScorer()
    hits = RerankRetriever(base, CHUNKS, scorer, depth=3).search("soru", 2)
    assert base.asked == 3
    assert len(scorer.pairs) == 3
    assert scorer.pairs[0] == ("soru", "alfa")
    assert len(hits) == 2


def test_ties_keep_the_base_order():
    # "alfa" and "beta" score the same; the base ranked beta first.
    base = FixedBase([(1, 0.9), (0, 0.8)])
    hits = RerankRetriever(base, CHUNKS, LengthScorer()).search("soru", 2)
    assert [idx for idx, _ in hits] == [1, 0]


def test_no_hits_means_no_cross_encoder_call():
    scorer = LengthScorer()
    assert RerankRetriever(FixedBase([]), CHUNKS, scorer).search("soru", 5) == []
    assert scorer.pairs == []
