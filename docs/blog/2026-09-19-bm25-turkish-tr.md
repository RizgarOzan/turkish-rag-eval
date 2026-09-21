# Türkçe'de BM25'i yenen model: küçük multilingual değil, retrieval için eğitilmiş olan

*Rızgar Ozan · 2026-09-19 · [English version](2026-09-19-bm25-turkish-en.md)*

[turkish-rag-eval](https://github.com/RizgarOzan/turkish-rag-eval)'i tek bir soru için
kurdum: bir RAG hattının hangi parçası Türkçede maliyetini hak ediyor? İlk ölçümden sonra
README'ye şunu yazdım: çok dilli embedding modeli, köklemeli BM25'i geçemedi. Cümle doğruydu
ama yanlış şeyi anlatıyordu. Beş model daha ölçünce kaybedenin yoğun (dense) arama değil,
seçtiğim model olduğu ortaya çıktı. Bu yazı o iki ölçümün hikâyesi.

## Düzenek

Altın set 58 Türkçe sağlık sorusu. Soruları 54 Türkçe Vikipedi makalesinden (toplam 1,09
milyon karakter) kendim yazdım ve etiketledim. Her soru makaledeki cümlenin kopyası değil,
başka kelimelerle sorulmuş hâli: metin "diyabet teşhisi konulduğunda" diyorsa soru "şeker
hastalığı teşhisi konan kişiler" diye soruyor. Sorular metinden kopyalansaydı BM25 hak
etmediği bir avantaj kazanırdı.

Bir parça (chunk), doğru makaleden geliyorsa *ve* cevap cümlesini içeriyorsa ilgili sayılıyor.
Bu tanım parçalama yönteminden bağımsız. Böylece üç parçalama yöntemini (sabit boyut,
cümle, başlık yolunu ekleyen hiyerarşik) dört arayıcıyla (köklemesiz BM25, 5 harflik önek
köklemeli BM25, dense, ikisini RRF ile birleştiren hibrit) aynı etiketlerle
karşılaştırabiliyorum. 12 konfigürasyon, hepsi CPU'da; projede GPU yok.

## İlk ölçüm: "embedding BM25'i yenemedi"

Varsayılan model `paraphrase-multilingual-MiniLM-L12-v2` idi: 118 M parametre, küçük ve hızlı,
çok dilli arama örneklerinin çoğunda ilk karşılaşılan model. nDCG@10 sonuçları:

| Parçalama | BM25 + kökleme | Dense (MiniLM) |
|---|---|---|
| sabit | 0,476 | 0,446 |
| cümle | 0,510 | 0,461 |
| hiyerarşik | 0,494 | 0,501 |

Köklemeli BM25 üç yöntemin ikisinde dense aramayı geçti, üstelik yaklaşık 5 kat daha hızlıydı.
Dense ancak hiyerarşik parçalamada, başlık yolu bağlamı geri verince, kıl payı öne geçti.
Kökleme de tek başına en ucuz kazançtı: her kelimeyi ilk 5 harfine kırpmak nDCG@10'u her
parçalamada %23–29 artırdı. Türkçe eklemeli bir dil; *diyabet*, *diyabetin*, *diyabete*,
*diyabetli* aynı kavramın dört yüzeyi. Köklemesiz bir indeks sorudaki biçimi nadiren buluyor,
bunun için morfolojik analizöre de gerek yok.

En iyi sonuç hibritteydi: hiyerarşik parçalama + RRF 0,607. README'nin ilk hâlinin özeti
buydu: kökle, hibrit kullan, embedding'e tek başına güvenme.

## İkinci ölçüm: model değişince

Harness'e baştan `--model` seçeneği koymuştum, ama ilk sonucu genelleştirmeden önce onu
kullanmadım. 18 Eylül'de beş model daha ölçtüm. Hepsi aynı 58 soru, aynı korpus, aynı
16 iş parçacıklı CPU:

| Model | Parametre | Dense hiyerarşik | Hibrit hiyerarşik | Sorgu P95 |
|---|---|---|---|---|
| MiniLM (varsayılan) | 118 M | 0,501 | 0,607 | 23 ms |
| emrecan (yalnız Türkçe) | 111 M | 0,497 | 0,654 | 43 ms |
| multilingual-e5-small | 118 M | 0,642 | 0,639 | 22 ms |
| multilingual-e5-base | 278 M | 0,668 | 0,648 | 49 ms |
| Mursit-Large-TR-Retrieval (yalnız Türkçe) | 404 M | **0,781** | 0,673 | 214 ms |

`bge-m3`'ü 45 dakikada durdurdum. Hiyerarşik parçalara gelemedi ama sabit ve cümle
parçalamada 0,766 ve 0,767 aldı. Köklemeli BM25 aynı parçalamalarda 0,476 ve 0,510'da kalıyor.

## Tablodan çıkan üç şey

**1. Sorun dense arama değil, modeldi.** Retrieval için eğitilmiş her model (E5, bge-m3,
Mursit) her parçalamada köklemeli BM25'i tek başına geçiyor. En çarpıcı satır e5-small:
varsayılanla aynı boyutta, sorgu başına aynı hızda (22 ms), korpusu 3 dakikada gömüyor ve
başka hiçbir şey değişmeden 0,501'den 0,642'ye çıkıyor. CPU'da çalışan bir sistemde
varsayılanın yerine konacak ilk model bu.

**2. Hibrit yalnız zayıf bir modele yarıyor.** RRF, BM25'in sıralamasına dense sıralamayla
aynı ağırlığı veriyor. MiniLM'i 0,106 yukarı çekiyor. Güçlü bir modelde ise iyi sıralamayı
daha kötü bir sıralamayla karıştırmış oluyor: Mursit 0,781'den 0,673'e, e5-base 0,668'den
0,648'e düşüyor. İlk ölçümdeki "hibrit her yerde kazanıyor" sonucu modelin zayıflığından
geliyormuş. Birleştirip birleştirmemeye karar vermek için önce ölçmek gerekiyor.

**3. "Türkçe'ye özel" olmak yetmiyor.** Hugging Face'in `sentence-similarity` kategorisinde en
çok indirilen iki Türkçe modeli seçtim. Biri (emrecan) cümle benzerliği için eğitilmiş
(NLI + STS-b) ve girdiyi 75 token'da kesiyor. Tokenizer'la saydım: parçaların %83–99'u
kırpılıyor. Dense arayıcı olarak varsayılandan farkı yok (0,497'ye 0,501). Diğeri (Mursit)
doğrudan retrieval için eğitilmiş ve buradaki en iyi model. Aradaki fark dilde değil,
modelin ne için eğitildiğinde.

## Maliyet

Kalitenin bir bedeli var ve CPU'da bunu hissediyorsunuz. Mursit sorgu başına 214 ms
harcıyor, e5-small'ın yaklaşık 10 katı, üç parçalamanın hepsini gömmesi de 35 dakika sürüyor.
Benim seçimim şöyle olurdu:

- **Gecikme önemliyse:** multilingual-e5-small. 0,642, 22 ms, hibrite gerek yok (0,639 ile
  dense'ten farkı gürültü düzeyinde).
- **Kalite önemliyse ve sorgu başına 214 ms kabul edilebiliyorsa:** Mursit, hibritsiz, 0,781.
- **Embedding hiç kullanılamıyorsa:** köklemeli BM25. Morfolojik analizör yok, 4–6 ms.

E5 modelleri her girdinin başında `query: ` / `passage: ` öneki bekliyor. Önek olmadan
alınan E5 sonucu adil bir sonuç değil; harness bunları `src/models.py`'de ekliyor.

## Sınırlar

58 soru küçük bir set. Yaklaşık 0,05'ten küçük nDCG farklarını sıralama olarak değil, gürültü
olarak okuyun. Soruları tek kişi etiketledi. Hepsi tek alandan, Vikipedi'nin sağlık
makalelerinden geliyor ve klinik metne olduğu gibi aktarılmaz. `bge-m3` yarım bir koşu,
`google/embeddinggemma-300m` lisans onayı istediği için ölçülmedi. Sayıların hepsi
README'de ve `turkish-rag-eval run --model <ad>` ile yeniden üretilebiliyor.

Set büyüyor: hedef beş alanda 300 soru. İlk 30 yeni soruyu bir dil modeli taslak olarak
yazdı, ikinci bağımsız bir geçiş onu görmeden yeniden etiketledi; 30'un 30'u uyuştu. İnsan
tarafından doğrulanmış taslak oranı henüz %0 ve README bunu açıkça yazıyor. Türkçe bir
Vikipedi makalesinden 5–10 soru yazıp tek bir JSON dosyasıyla PR açmak için ML bilmek
gerekmiyor; [CONTRIBUTING.md](../../CONTRIBUTING.md) nasıl yapılacağını anlatıyor.

## Sonuç

İlk ölçümüm yanlış değildi, ama ondan çıkardığım genelleme yanlıştı. "Türkçede embedding
BM25'i yenemiyor" diye yazmadan önce yapmam gereken tek şey modeli değiştirmekti ve harness
buna zaten hazırdı. Artık README iki sonucu da taşıyor: eski tablo silinmedi, yanına
yenisi ve bu yazı eklendi.
