"""The dense index can be cached on disk, keyed by model, prefix and chunk text.
A fake model counts encode calls, so the test needs no download."""

import pytest

np = pytest.importorskip("numpy")

from turkish_rag_eval.retrieval import DenseRetriever  # noqa: E402

CHUNKS = [{"embed_text": text} for text in ("alfa", "beta", "gama")]


class CountingModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **_):
        self.calls += 1
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def build(tmp_path, chunks=CHUNKS, model_name="m", doc_prefix=""):
    model = CountingModel()
    dense = DenseRetriever(chunks, model, doc_prefix=doc_prefix,
                           cache_dir=tmp_path, model_name=model_name)
    return dense, model


def test_second_build_reads_the_cache_instead_of_encoding(tmp_path):
    first, _ = build(tmp_path)
    second, model = build(tmp_path)
    assert model.calls == 0
    assert second.cache_hit and not first.cache_hit
    assert np.array_equal(first.matrix, second.matrix)


def test_a_hit_reports_the_original_encode_time(tmp_path):
    first, _ = build(tmp_path)
    second, _ = build(tmp_path)
    assert second.encode_seconds == first.encode_seconds


@pytest.mark.parametrize("change", [
    {"chunks": [{"embed_text": t} for t in ("alfa", "beta", "gamma")]},
    {"model_name": "other"},
    {"doc_prefix": "passage: "},
])
def test_any_change_to_the_input_misses(tmp_path, change):
    build(tmp_path)
    _, model = build(tmp_path, **change)
    assert model.calls == 1


def test_without_a_cache_dir_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = CountingModel()
    DenseRetriever(CHUNKS, model)
    DenseRetriever(CHUNKS, model)
    assert model.calls == 2
    assert list(tmp_path.iterdir()) == []
