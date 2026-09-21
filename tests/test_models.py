"""Tests for per-model settings: input prefixes and where results are written.

Run with: python -m pytest tests -q
"""
import pytest

from turkish_rag_eval.models import DEFAULT_MODEL, prefixes_for, results_dir
from turkish_rag_eval.gold import ROOT


def test_e5_models_get_their_query_and_passage_prefixes():
    # The E5 model cards require these; without them retrieval quality drops.
    assert prefixes_for("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert prefixes_for("intfloat/multilingual-e5-base") == ("query: ", "passage: ")


def test_other_models_get_no_prefix():
    assert prefixes_for(DEFAULT_MODEL) == ("", "")
    assert prefixes_for("BAAI/bge-m3") == ("", "")


def test_default_model_keeps_writing_to_results_root():
    # The README's main table is regenerated from results/ - that must not move.
    assert results_dir(DEFAULT_MODEL) == ROOT / "results"


def test_other_models_write_to_their_own_folder():
    assert results_dir("BAAI/bge-m3") == ROOT / "results" / "models" / "BAAI__bge-m3"


def test_dense_retriever_applies_prefixes():
    np = pytest.importorskip("numpy")
    pytest.importorskip("rank_bm25")
    from turkish_rag_eval.retrieval import DenseRetriever

    seen = []

    class FakeModel:
        def encode(self, texts, **_):
            seen.extend(texts)
            return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)

    chunks = [{"embed_text": "Diyabet"}, {"embed_text": "Astım"}]
    retriever = DenseRetriever(chunks, FakeModel(), "query: ", "passage: ")
    retriever.search("şeker hastalığı", 1)
    assert seen == ["passage: Diyabet", "passage: Astım", "query: şeker hastalığı"]
