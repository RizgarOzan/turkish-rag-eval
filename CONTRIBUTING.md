# Contributing questions

The benchmark's biggest weakness is written in its own README: **58 questions,
one annotator.** Every question you add makes the numbers less noisy, and every
new annotator makes the labels less personal. No machine-learning background
is needed — if you read Turkish well, you can contribute.

*Türkçe açıklama aşağıda.*

## Add questions (one file per pull request)

1. Pick an article on [Turkish Wikipedia](https://tr.wikipedia.org). Any topic
   is welcome — health, history, law, sport, science, geography. Open issues
   labelled `good first issue` suggest topics nobody has covered yet.
2. Find its **page id**: open the article, click *Sayfa bilgisi* (Page
   information) in the side menu — it is the "Sayfa kimliği" number.
3. Write **5–10 questions** whose answers are stated in that article.
4. Create `data/eval/contrib/<your-github-name>-<topic>.json`. This is the
   format, shown with the benchmark's first question (the article really says
   *"yaklaşık %25'i, diyabet teşhisi konulduğunda diyabetik ketoasidoz da
   gelişmiştir"*; the question asks for it in other words):

```json
[
  {
    "qid": "yourname-001",
    "question": "Şeker hastalığı teşhisi konan kişilerin ne kadarında ketoasidoz da bulunuyor?",
    "doc_id": "710430",
    "doc_title": "Diyabet",
    "answer_span": "yaklaşık %25'i, diyabet teşhisi konulduğunda diyabetik ketoasidoz da gelişmiştir",
    "annotator": "your-github-name"
  }
]
```

5. Check it locally, then open a pull request:

```bash
pip install requests pytest
turkish-rag-eval validate --online data/eval/contrib/yourname-topic.json
```

### The rules that make a question useful

- **Paraphrase, never copy.** Ask the way a person would ask, in different
  words from the article. A question that repeats the article's words gives
  keyword search an unfair win and makes the comparison meaningless. The
  validator rejects questions that share more than 60% of their words with the
  answer.
- **`answer_span` is copied verbatim** from the article — a short phrase
  (roughly 5–25 words) that contains the answer. Copy it exactly; the
  validator searches for it.
- **One clear answer.** If two different parts of the article could answer the
  question, make the question more specific.
- **`doc_title`** is the article title exactly as Wikipedia shows it.
- **`qid`** starts with your GitHub name and is unique: `yourname-001`,
  `yourname-002`, …
- Questions end with `?`.

CI runs the same validator on every pull request. When it is green, a
maintainer reviews the questions by hand and merges.

## Second annotations

Annotator agreement is the other missing number. Issues labelled
`second-annotation` list existing questions that need an independent second
answer span. Write yours *without* looking at the first one.

Reuse the original `question`, `doc_id` and `doc_title` unchanged, add your
own `answer_span`, and set `"second_of"` to the qid you are re-annotating:

```json
{
  "qid": "yourname-q001",
  "question": "Şeker hastalığı teşhisi konan kişilerin ne kadarında ketoasidoz da bulunuyor?",
  "doc_id": "710430",
  "doc_title": "Diyabet",
  "answer_span": "diyabetik ketoasidoz da gelişmiştir",
  "annotator": "yourname",
  "second_of": "q001"
}
```

`turkish-rag-eval validate` allows the repeated question only when
`second_of` resolves this way. Once a question has two (or more) independent
spans, `turkish-rag-eval agreement` reports IoU (Jaccard overlap of the spans'
token sets, averaged over every annotator pair if there are more than two) as
the headline number, token-level F1 alongside it, and, per chunking strategy,
Cohen's kappa over which chunks each span would mark relevant.

## Submit an embedding model

The leaderboard takes submissions. Run the harness and commit what it wrote:

```bash
pip install -e '.[all]'
turkish-rag-eval fetch-corpus        # or --verify, if you already have it
turkish-rag-eval run --model <org>/<name>
turkish-rag-eval leaderboard --check
```

That writes `results/models/<org>__<name>/` — a summary plus the per-query
relevance array behind every number. Commit the whole directory; the numbers
alone are not a submission.

CI re-derives each reported metric from those per-query files rather than
taking them on trust, and a new submission must additionally carry the harness
version and a corpus fingerprint matching `data/corpus.lock.json`. If the
fingerprint check fails, your corpus has drifted from the pinned snapshot and
the run is not comparable to the other rows — refetch before rerunning.

Say in the pull request what hardware you measured on. The timing columns are
not comparable across machines and the table says so.

## Code

Bug fixes and new retrievers or embedding models are welcome too. Run
`python -m pytest` before opening a pull request, keep changes small, and say
in the description what you measured.

Anything that can move a published number — a tie-break, a chunker, a
normaliser — is a breaking change for a benchmark even if no API changed. Note
it in `CHANGELOG.md` under Unreleased.

---

## Türkçe

Bu benchmark'ın en büyük zayıflığı README'de yazıyor: **58 soru, tek
etiketleyici.** Eklediğin her soru sonuçlardaki gürültüyü azaltır. Makine
öğrenmesi bilmen gerekmiyor — Türkçeyi iyi okuyorsan katkı verebilirsin.

1. Türkçe Vikipedi'den bir makale seç (sağlık, tarih, hukuk, spor, bilim…
   her konu olur). `good first issue` etiketli issue'lar henüz kimsenin
   yapmadığı konuları öneriyor.
2. Makalenin **sayfa kimliğini** bul: yan menüde *Sayfa bilgisi* → "Sayfa
   kimliği".
3. Cevabı makalede açıkça yazan **5–10 soru** yaz.
4. `data/eval/contrib/<github-adın>-<konu>.json` dosyasını yukarıdaki biçimde
   oluştur, `turkish-rag-eval validate --online <dosya>` ile kontrol et ve
   pull request aç.

**Kurallar:** Soruyu makalenin kelimeleriyle değil, bir insanın soracağı gibi
**kendi kelimelerinle** sor (makaleden kopyalanmış soru, kelime aramasına
haksız avantaj verir; doğrulayıcı reddeder). `answer_span` ise makaleden
**birebir kopyalanan** kısa bir ifade. Her sorunun tek net cevabı olsun.
`qid` GitHub adınla başlasın ve benzersiz olsun. Sorular `?` ile bitsin.
