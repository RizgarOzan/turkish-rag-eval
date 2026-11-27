# Legislation domain (draft)

The first non-Wikipedia domain: 50 questions over Turkish public-procurement
regulations, with their own pinned corpus. Everything here is machine-generated
and has not been checked by a person yet, so it sits outside the published
numbers in the same way the `llm-draft-*` files do.

## What is in it

- `llm-draft-mevzuat.json`: 50 questions in the usual gold format. They come in
  four groups of 12 or 13, one for each of the four implementing regulations
  for public procurement: goods (Mal Alımı), services (Hizmet Alımı), works
  (Yapım İşleri) and consultancy (Danışmanlık Hizmet Alımı).
- `../../raw/mevzuat/corpus.json`: 374 articles from those four regulations,
  one document per article. `doc_id` is `<regulation number>:<article>`. Each
  document records its `url` on mevzuat.gov.tr, the access date and the SHA-256
  of the downloaded source file.
- `../../raw/mevzuat/corpus.lock.json`: the fingerprint, in the same format as
  `corpus.lock.json`.

The four regulations were written from one template. Many articles are almost
word for word the same across them, so most of the corpus acts as a
distractor. The same rule often shows up in all four with a different number,
form or authority. For almost every question, the only thing that picks the
right article is the kind of procurement it mentions: goods, services, works or
consultancy. A few questions can also be told apart by topic alone
(`mevzuat-003`, for example, by its R&D and domestic-goods wording). Every
`answer_span` occurs verbatim in exactly one article of the corpus, and a test
enforces this.

## Where the questions come from

The questions come from [TR-Mevzuat-SC](https://huggingface.co/datasets/mvdelibas/tr-mevzuat-sc)
(CC0-1.0). They were made in three steps, all by models:

1. One model wrote each question from an article, and the answer is a verbatim
   span of that article.
2. A second model answered it independently. Only questions where both models
   agreed are kept.
3. For this domain, an LLM rewrote each question. It removed the regulation
   name and the article number, and it put the question in everyday wording.

Each item keeps its source id, the original question and both model names
under `provenance`. The span is the dataset's `dayanak` field, unchanged.

Article texts are the codified texts on mevzuat.gov.tr as of 25 October 2026.
Laws and regulations are not protected by copyright under Turkish law
(FSEK art. 31), and the extracted texts are released as CC0.

## Running it

```bash
turkish-rag-eval run --corpus data/raw/mevzuat/corpus.json \
    --gold data/eval/mevzuat --include-drafts
```

`tests/test_mevzuat_domain.py` checks the questions offline against the pinned
corpus, since there is no live article to fetch:

- spans are verbatim and unique;
- questions are paraphrases under the 60% overlap rule;
- no digits, regulation names or article references appear in the questions;
- the corpus matches its lockfile.

## Not done yet

- No human review. A person who reads procurement law should check that each
  question has one answer and that the answer is right.
- No English translation in `data/eval/en/`.
- `validate --online` does not cover these files, because they are not under
  `contrib/` and have no Wikipedia page id.

---

**Türkçe özet.** Kamu ihale mevzuatının dört uygulama yönetmeliğinden (mal,
hizmet, yapım, danışmanlık) 50 soru ve 374 maddelik sabit bir derlem. Bütün
sorular makine üretimi; henüz hiçbirine insan bakmadı. Sorularda yönetmelik
adı ve madde numarası geçmez. Doğru maddeyi yalnızca işin türü (mal, hizmet,
yapım, danışmanlık) ayırır, çünkü dört yönetmelikte aynı kural çoğu zaman
neredeyse aynı cümleyle yazılmıştır.
