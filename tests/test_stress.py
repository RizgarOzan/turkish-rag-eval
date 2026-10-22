"""The stress command rewrites every question the ways people mistype Turkish
and scores each rewrite with and without the normaliser."""

import argparse
import sys
import json

import pytest

from turkish_rag_eval import stress

pytest.importorskip("rank_bm25")

ARTICLE = """== Diyabet ==
Diyabet, kan şekerinin uzun süre yüksek seyrettiği bir metabolizma hastalığıdır.

== Belirtiler ==
Sık idrara çıkma ve aşırı susama en yaygın belirtilerdir.
Tedavi edilmediğinde ketoasidoz gelişebilir.
"""

QUESTION = {
    "qid": "t-1",
    "doc_id": "diyabet.md",
    "doc_title": "diyabet",
    "question": "Şeker hastalığında aşırı susama görülür mü?",
    "answer_span": "Sık idrara çıkma ve aşırı susama en yaygın belirtilerdir.",
}


def test_variants():
    q = "Işık için şişe"
    assert stress.VARIANTS["ascii"](q) == "Isik icin sise"
    assert stress.VARIANTS["caps"](q) == "IŞIK İÇİN ŞİŞE"
    assert "i\u0307" in stress.VARIANTS["naive_lower"](q)


def test_sparse_run_writes_every_variant_raw_and_normalized(tmp_path):
    corpus = tmp_path / "belgeler"
    corpus.mkdir()
    (corpus / "diyabet.md").write_text(ARTICLE, encoding="utf-8")
    gold = tmp_path / "sorular.json"
    gold.write_text(json.dumps([QUESTION], ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "stress.json"

    parser = argparse.ArgumentParser()
    stress.add_arguments(parser)
    args = parser.parse_args(["--corpus", str(corpus), "--gold", str(gold),
                              "--out", str(out), "--sparse-only"])
    assert stress.run(args) == 0

    scores = json.loads(out.read_text(encoding="utf-8"))["scores"]
    assert set(scores) == {f"bm25_stem5/{v}{n}" for v in stress.VARIANTS
                           for n in ("", "/normalized")}
    # "asiri susama" only matches after the normaliser puts the ı and ş back.
    assert scores["bm25_stem5/ascii/normalized"]["ndcg@10"] >= \
        scores["bm25_stem5/ascii"]["ndcg@10"]


def test_a_second_dense_run_reads_the_chunk_embeddings_from_the_cache(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    encoded = []

    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, texts, **_):
            encoded.append(len(texts))
            return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    fake = type(sys)("sentence_transformers")
    fake.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    corpus = tmp_path / "belgeler"
    corpus.mkdir()
    (corpus / "diyabet.md").write_text(ARTICLE, encoding="utf-8")
    gold = tmp_path / "sorular.json"
    gold.write_text(json.dumps([QUESTION], ensure_ascii=False), encoding="utf-8")
    cache = tmp_path / "cache"

    parser = argparse.ArgumentParser()
    stress.add_arguments(parser)
    argv = ["--corpus", str(corpus), "--gold", str(gold), "--model", "fake",
            "--embedding-cache", str(cache)]
    assert stress.run(parser.parse_args(argv + ["--out", str(tmp_path / "a.json")])) == 0
    first = list(encoded)
    encoded.clear()
    assert stress.run(parser.parse_args(argv + ["--out", str(tmp_path / "b.json")])) == 0

    assert len(list(cache.glob("*.npz"))) == 1
    # Only the chunk matrix is skipped; every query is still encoded.
    assert len(encoded) == len(first) - 1
    a = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["scores"]
    b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["scores"]
    assert a == b
