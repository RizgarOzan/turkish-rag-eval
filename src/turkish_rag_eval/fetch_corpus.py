"""Fetch Turkish health articles from Wikipedia as the retrieval corpus.

Plain-text extracts keep the ``== Section ==`` markers, which is exactly what
the hierarchical chunker needs to rebuild the heading path of every chunk.

One title per request. Batching looks tempting, but the API answers a
multi-title whole-article extract with "exlimit was too large ... lowered
to 1" and returns a single extract, so a batch of twenty silently yields one
document. One second between calls keeps it under the 429 threshold.
Re-running only fetches what is missing from disk.

Content is CC BY-SA 4.0 (Wikipedia). See NOTICE.md for attribution.
"""

import argparse
import json
import time
from pathlib import Path

import requests

from .corpus import build_lock, compare_to_lock, load_corpus
from .gold import load_gold
from .paths import ROOT

API = "https://tr.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "turkish-rag-eval/0.1 (research; github.com/RizgarOzan)"}
DELAY_SECONDS = 1.0
MIN_EXTRACT_CHARS = 800
OUT = ROOT / "data" / "raw" / "corpus.json"
#: Committed, unlike the corpus itself: it is small, and it is the only thing
#: that lets a checkout tell whether its articles match the published numbers.
LOCK = ROOT / "data" / "corpus.lock.json"

TOPICS = [
    "Diyabet", "Hipertansiyon", "Astım", "Migren", "Anemi",
    "Hipotiroidi", "Hipertiroidi", "Gastrit", "Kronik obstrüktif akciğer hastalığı",
    "Depresyon (ruh hâli)", "Osteoporoz", "Grip", "Zatürre", "Böbrek taşı",
    "Safra kesesi taşı", "Peptik ülser", "Epilepsi", "Romatoid artrit",
    "Çölyak hastalığı", "Hepatit B", "Sedef hastalığı", "Egzama",
    "Ateroskleroz", "Kalp yetmezliği", "Miyokard enfarktüsü", "İnme",
    "Derin ven trombozu", "Anafilaksi", "Menenjit", "Tüberküloz",
    "Sıtma", "Hepatit C", "Siroz", "Pankreatit", "Apandisit",
    "İrritabl bağırsak sendromu", "Crohn hastalığı", "Ülseratif kolit",
    "Gut hastalığı", "Fibromiyalji", "Multipl skleroz", "Parkinson hastalığı",
    "Alzheimer hastalığı", "Şizofreni", "Bipolar bozukluk",
    "Anksiyete bozukluğu", "Uyku apnesi", "Obezite", "Metabolik sendrom",
    "Böbrek yetmezliği", "İdrar yolu enfeksiyonu", "Prostat büyümesi",
    "Endometriozis", "Polikistik over sendromu", "Osteoartrit",
    "Katarakt", "Glokom", "Otitis media", "Sinüzit", "Bronşit",
]


def fetch_batch(titles: list[str], attempt: int = 0) -> list[dict]:
    params = {
        "action": "query",
        # info comes along for lastrevid: the revision each extract was taken
        # from, which is what corpus.lock.json pins.
        "prop": "extracts|info",
        "explaintext": 1,
        "exlimit": "max",
        "redirects": 1,
        "format": "json",
        "titles": "|".join(titles),
    }
    try:
        response = requests.get(API, params=params, timeout=60, headers=HEADERS)
        response.raise_for_status()
    except Exception as exc:
        if attempt >= 4:
            print(f"  vazgecildi ({len(titles)} baslik): {exc}")
            return []
        wait = 5 * (attempt + 1)
        print(f"  {exc.__class__.__name__}, {wait}s bekleniyor "
              f"(deneme {attempt + 1}/4)")
        time.sleep(wait)
        return fetch_batch(titles, attempt + 1)

    out = []
    for page in response.json()["query"]["pages"].values():
        if "extract" not in page or len(page["extract"]) < MIN_EXTRACT_CHARS:
            continue
        out.append({
            "doc_id": str(page["pageid"]),
            "title": page["title"],
            "url": f"https://tr.wikipedia.org/?curid={page['pageid']}",
            "text": page["extract"],
            "revid": page.get("lastrevid"),
        })
    return out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verify", action="store_true",
        help="do not fetch; only check the corpus on disk against the lockfile")
    parser.add_argument(
        "--update-lock", action="store_true",
        help="accept the fetched corpus as the new pinned snapshot")


def report_drift(docs: list[dict], update: bool) -> int:
    """Compare against corpus.lock.json and say what moved.

    Wikipedia keeps editing underneath us, so a plain refetch is not
    reproducible. Rather than pretend otherwise, drift is made loud: the
    command fails, names the articles that changed, and waits for someone to
    decide whether the published numbers still describe this corpus.
    """
    if update or not LOCK.exists():
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        lock = build_lock(docs, {d["doc_id"]: d.get("revid") for d in docs})
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"kilit yazildi: {lock['fingerprint']} -> {LOCK}")
        return 0

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    drift = compare_to_lock(lock, docs)
    if not drift:
        print(f"kilit dogrulandi: {lock['fingerprint']}")
        return 0

    print(f"\nkorpus kilitten farkli ({len(drift)} degisiklik):")
    for line in drift:
        print(f"  {line}")
    print("\nYayinlanan sayilar bu korpusu tarif etmiyor olabilir. Kabul etmek "
          "icin: turkish-rag-eval fetch-corpus --update-lock")
    return 1


def run(args=None) -> int:
    if args is not None and getattr(args, "verify", False):
        if not OUT.exists():
            print(f"{OUT} yok - once fetch-corpus calistirin")
            return 1
        return report_drift(load_corpus(OUT).docs, update=False)

    docs = {}
    if OUT.exists():
        for doc in json.loads(OUT.read_text(encoding="utf-8")):
            docs[doc["title"]] = doc
    print(f"diskte {len(docs)} belge var\n")

    # Contributed questions may point at articles outside the original list.
    topics = TOPICS + sorted({i["doc_title"] for i in load_gold()} - set(TOPICS))

    wanted = {}  # requested title -> resolved title, so redirects are visible
    for i, topic in enumerate(topics, 1):
        fetched = fetch_batch([topic])
        if fetched:
            doc = fetched[0]
            docs[doc["title"]] = doc
            wanted[topic] = doc["title"]
            arrow = f" -> {doc['title']}" if doc["title"] != topic else ""
            print(f"  [{i}/{len(topics)}] {topic}{arrow}: "
                  f"{len(doc['text'])} karakter")
        else:
            print(f"  [{i}/{len(topics)}] {topic}: yok / cok kisa")
        time.sleep(DELAY_SECONDS)

    ordered = sorted(docs.values(), key=lambda d: d["title"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    chars = sum(len(d["text"]) for d in ordered)
    print(f"\n{len(ordered)} belge, {chars} karakter -> {OUT}")

    missing = [t for t in topics if t not in wanted]
    if missing:
        print(f"alinamayan {len(missing)}: {', '.join(missing)}")

    return report_drift(load_corpus(OUT).docs,
                        update=bool(args and getattr(args, "update_lock", False)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
