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

import json
import time
from pathlib import Path

import requests

API = "https://tr.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "turkish-rag-eval/0.1 (research; github.com/RizgarOzan)"}
DELAY_SECONDS = 1.0
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "corpus.json"

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
        "prop": "extracts",
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
        if "extract" not in page or len(page["extract"]) < 800:
            continue
        out.append({
            "doc_id": str(page["pageid"]),
            "title": page["title"],
            "url": f"https://tr.wikipedia.org/?curid={page['pageid']}",
            "text": page["extract"],
        })
    return out


def main() -> None:
    docs = {}
    if OUT.exists():
        for doc in json.loads(OUT.read_text(encoding="utf-8")):
            docs[doc["title"]] = doc
    print(f"diskte {len(docs)} belge var\n")

    wanted = {}  # requested title -> resolved title, so redirects are visible
    for i, topic in enumerate(TOPICS, 1):
        fetched = fetch_batch([topic])
        if fetched:
            doc = fetched[0]
            docs[doc["title"]] = doc
            wanted[topic] = doc["title"]
            arrow = f" -> {doc['title']}" if doc["title"] != topic else ""
            print(f"  [{i}/{len(TOPICS)}] {topic}{arrow}: "
                  f"{len(doc['text'])} karakter")
        else:
            print(f"  [{i}/{len(TOPICS)}] {topic}: yok / cok kisa")
        time.sleep(DELAY_SECONDS)

    ordered = sorted(docs.values(), key=lambda d: d["title"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    chars = sum(len(d["text"]) for d in ordered)
    print(f"\n{len(ordered)} belge, {chars} karakter -> {OUT}")

    missing = [t for t in TOPICS if t not in wanted]
    if missing:
        print(f"alinamayan {len(missing)}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
