"""serve answers over HTTP with the same ranking the retriever gives in process,
and refuses requests it cannot answer instead of guessing."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import pytest

pytest.importorskip("rank_bm25")

from turkish_rag_eval import serve  # noqa: E402
from turkish_rag_eval.chunking import chunk_hierarchical  # noqa: E402
from turkish_rag_eval.corpus import fingerprint  # noqa: E402

DOCS = [
    {"doc_id": "diyabet", "title": "Diyabet", "url": "https://tr.wikipedia.org/wiki/Diyabet",
     "text": "== Belirtiler ==\nSık idrara çıkma ve aşırı susama en yaygın belirtilerdir.\n\n"
             "== Tedavi ==\nTedavi edilmediğinde ketoasidoz gelişebilir ve hastanede bakım gerekir.\n"},
    {"doc_id": "mars", "title": "Mars", "url": "https://tr.wikipedia.org/wiki/Mars",
     "text": "== Yüzey ==\nMars yüzeyindeki demir oksit gezegene kızıl rengini verir.\n\n"
             "== Uydular ==\nMarsın Phobos ve Deimos adlı iki küçük uydusu vardır.\n"},
]


@pytest.fixture()
def server():
    chunks = [c for doc in DOCS for c in chunk_hierarchical(doc)]
    name, retriever = serve.build(chunks)
    handler = serve.make_handler(chunks, name, retriever, fingerprint(DOCS))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", chunks, retriever
    httpd.shutdown()
    httpd.server_close()


def get(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_names_the_retriever_and_the_corpus(server):
    base, chunks, _ = server
    status, body = get(f"{base}/health")
    assert status == 200
    assert body == {"status": "ok", "retriever": "bm25_stem5",
                    "chunks": len(chunks), "corpus": fingerprint(DOCS)}


def test_search_returns_the_retrievers_ranking(server):
    base, chunks, retriever = server
    question = "Mars neden kızıl renkte görünür?"
    status, body = get(f"{base}/search?q={quote(question)}&k=3")
    assert status == 200
    assert body["query"] == question and body["retriever"] == "bm25_stem5"
    expected = retriever.search(question, 3)
    assert [r["chunk_id"] for r in body["results"]] == [chunks[i]["chunk_id"] for i, _ in expected]
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    top = body["results"][0]
    assert top["title"] == "Mars" and "kızıl" in top["text"]


def test_turkish_letters_survive_the_round_trip(server):
    base, _, _ = server
    status, body = get(f"{base}/search?q={quote('aşırı susama')}&k=1")
    assert status == 200
    assert body["query"] == "aşırı susama"
    assert "susama" in body["results"][0]["text"]


@pytest.mark.parametrize("path, status", [
    ("/search", 400),
    ("/search?q=%20", 400),
    ("/search?q=mars&k=0", 400),
    ("/search?q=mars&k=51", 400),
    ("/search?q=mars&k=bes", 400),
    ("/nowhere", 404),
])
def test_bad_requests_get_an_error_not_a_guess(server, path, status):
    base, _, _ = server
    code, body = get(f"{base}{path}")
    assert code == status and "error" in body
