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
python src/validate_gold.py --online data/eval/contrib/yourname-topic.json
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

`python src/validate_gold.py` allows the repeated question only when
`second_of` resolves this way. Once a question has two (or more) independent
spans, `python src/agreement.py` reports IoU (Jaccard overlap of the spans'
token sets, averaged over every annotator pair if there are more than two) as
the headline number, token-level F1 alongside it, and, per chunking strategy,
Cohen's kappa over which chunks each span would mark relevant.

## Code

Bug fixes and new retrievers or embedding models are welcome too. Run
`python -m pytest` before opening a pull request, keep changes small, and say
in the description what you measured.

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
   oluştur, `python src/validate_gold.py --online <dosya>` ile kontrol et ve
   pull request aç.

**Kurallar:** Soruyu makalenin kelimeleriyle değil, bir insanın soracağı gibi
**kendi kelimelerinle** sor (makaleden kopyalanmış soru, kelime aramasına
haksız avantaj verir; doğrulayıcı reddeder). `answer_span` ise makaleden
**birebir kopyalanan** kısa bir ifade. Her sorunun tek net cevabı olsun.
`qid` GitHub adınla başlasın ve benzersiz olsun. Sorular `?` ile bitsin.
