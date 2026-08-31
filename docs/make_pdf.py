"""Türkçe teknik açıklama dokümanını PDF olarak üretir.

Sonuç tabloları results/ altındaki JSON dosyalarından okunur; ölçüm
değiştiğinde doküman da değişir, elle güncellenmez.

Kullanım:  python docs/make_pdf.py [cikti_klasoru]
"""

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", FONTS / "arial.ttf"))
pdfmetrics.registerFont(TTFont("BodyBold", FONTS / "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FONTS / "consola.ttf"))

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#c8c8c8")
BAND = colors.HexColor("#eef2f6")
ACCENT = colors.HexColor("#14496b")

_base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontName="BodyBold",
                            fontSize=22, leading=27, textColor=INK,
                            spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", parent=_base["Normal"],
                               fontName="Body", fontSize=11, leading=15,
                               textColor=MUTED, spaceAfter=18),
    "h1": ParagraphStyle("h1", parent=_base["Heading1"], fontName="BodyBold",
                         fontSize=15, leading=19, textColor=ACCENT,
                         spaceBefore=18, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=_base["Heading2"], fontName="BodyBold",
                         fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=12, spaceAfter=4),
    "body": ParagraphStyle("body", parent=_base["Normal"], fontName="Body",
                           fontSize=9.7, leading=14.6, textColor=INK,
                           alignment=TA_JUSTIFY, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", parent=_base["Normal"], fontName="Body",
                             fontSize=9.7, leading=14.4, textColor=INK,
                             leftIndent=13, bulletIndent=3, spaceAfter=4),
    "code": ParagraphStyle("code", parent=_base["Normal"], fontName="Mono",
                           fontSize=8.3, leading=11.6, textColor=INK,
                           backColor=colors.HexColor("#f4f6f8"),
                           borderPadding=6, leftIndent=4, spaceAfter=8),
    "note": ParagraphStyle("note", parent=_base["Normal"], fontName="Body",
                           fontSize=9.4, leading=14, textColor=INK,
                           backColor=BAND, borderPadding=8, leftIndent=2,
                           rightIndent=2, spaceBefore=4, spaceAfter=9),
    "cell": ParagraphStyle("cell", parent=_base["Normal"], fontName="Body",
                           fontSize=8.5, leading=11, textColor=INK),
    "cellhead": ParagraphStyle("cellhead", parent=_base["Normal"],
                               fontName="BodyBold", fontSize=8.5, leading=11,
                               textColor=colors.white),
}


def para(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def code(text):
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;").replace("\n", "<br/>")
                   .replace(" ", "&nbsp;"))
    return Paragraph(escaped, S["code"])


def table(rows, widths, highlight_first_row=True):
    data = []
    for r, row in enumerate(rows):
        style = "cellhead" if (r == 0 and highlight_first_row) else "cell"
        data.append([Paragraph(str(c), S[style]) for c in row])
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]
    if highlight_first_row:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), ACCENT))
    t.setStyle(TableStyle(commands))
    return t


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def results_table():
    summary = load("summary.json")
    if not summary:
        return para("<i>results/summary.json bulunamadı — önce "
                    "run_eval.py çalıştırılmalı.</i>")
    rows = [["Parçalama", "Getirici", "nDCG@10", "R@5", "MRR", "P95"]]
    for row in summary:
        rows.append([
            row["chunking"], row["retriever"],
            f"{row['ndcg@10']:.3f}", f"{row['recall@5']:.3f}",
            f"{row['mrr']:.3f}", f"{row['latency_ms_p95']:.0f} ms",
        ])
    return table(rows, [2.9 * cm, 3.1 * cm, 2.3 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])


def stemming_table():
    summary = load("summary.json")
    if not summary:
        return para("<i>ölçüm dosyası yok.</i>")
    by_key = {(r["chunking"], r["retriever"]): r for r in summary}
    rows = [["Parçalama", "Gövdesiz", "Gövdeli (5 harf)", "Kazanç"]]
    for strategy in ("fixed", "sentence", "hierarchical"):
        raw = by_key.get((strategy, "bm25_nostem"))
        stemmed = by_key.get((strategy, "bm25_stem5"))
        if not raw or not stemmed:
            continue
        gain = (stemmed["ndcg@10"] - raw["ndcg@10"]) / raw["ndcg@10"] * 100
        rows.append([strategy, f"{raw['ndcg@10']:.3f}",
                     f"{stemmed['ndcg@10']:.3f}", f"+%{gain:.0f}"])
    return table(rows, [3.5 * cm, 3.2 * cm, 3.9 * cm, 2.6 * cm])


def abstain_table(signal):
    curve = load(f"abstain_curve_{signal}.json")
    if not curve:
        return para("<i>results/abstain_curve_%s.json bulunamadı.</i>" % signal)
    rows = [["Eşik", "Kapsam", "Doğruluk", "Cevaplanan", "İnsana"]]
    for point in curve:
        if point["answered"] == 0:
            continue
        rows.append([
            f"{point['threshold']:.2f}", f"{point['coverage']:.3f}",
            f"{point['selective_accuracy']:.3f}",
            str(point["answered"]), str(point["escalated"]),
        ])
    return table(rows, [2.2 * cm, 2.6 * cm, 2.6 * cm, 2.9 * cm, 2.4 * cm])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2.2 * cm, 1.3 * cm, "turkish-rag-eval — teknik açıklama")
    canvas.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(2.2 * cm, 1.7 * cm, A4[0] - 2.2 * cm, 1.7 * cm)
    canvas.restoreState()


def build(out_path: Path):
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm, bottomMargin=2.2 * cm,
        title="turkish-rag-eval - teknik açıklama", author="Rızgar Ozan",
    )
    story = []
    add = story.append

    # ---------------------------------------------------------------- kapak
    add(para("turkish-rag-eval", "title"))
    add(para("Türkçe bir RAG getirme hattının hangi parçalarının gerçekten "
             "işe yaradığını ölçen değerlendirme düzeneği — kodun tamamının "
             "sade Türkçe açıklaması", "subtitle"))

    add(para("Bu doküman ne işe yarar?", "h1"))
    add(para(
        "Bu projedeki her dosyanın <b>ne yaptığını, neden öyle yazıldığını ve "
        "ölçümün ne söylediğini</b> anlatır. Amaç, kodu okumadan da projeyi "
        "savunabilecek kadar anlamanı sağlamak: bir mülakatta "
        "&quot;burada neden BM25 kullandın?&quot; ya da &quot;RRF'te k neden "
        "60?&quot; diye sorulduğunda cevabın bu dokümanda var."))
    add(para(
        "Sondaki <b>Sık sorulacak sorular</b> bölümü, teknik bir görüşmede "
        "gelmesi muhtemel soruları ve kısa cevaplarını toplar."))

    # ------------------------------------------------------------ RAG nedir
    add(para("1. Önce temel: RAG nedir?", "h1"))
    add(para(
        "Bir dil modeline &quot;şeker hastalığında tarama hangi yaşta "
        "başlar?&quot; diye sorduğunda model cevabı ezberinden üretir. Ezber "
        "eksikse uydurur. <b>RAG (Retrieval-Augmented Generation)</b> bu "
        "sorunu şöyle çözer: soru gelince önce bir belge havuzundan konuyla "
        "ilgili metin parçalarını <i>bulur</i>, sonra o parçaları modele "
        "&quot;cevabı bunlara dayanarak yaz&quot; diye verir."))
    add(para(
        "Yani RAG iki aşamadır: <b>getirme (retrieval)</b> ve <b>üretme "
        "(generation)</b>. Bu proje bilinçli olarak <b>sadece birinci "
        "aşamayı</b> ele alır. Sebebi basit: getirme yanlış parçayı "
        "bulduysa, arkasındaki model ne kadar iyi olursa olsun cevap "
        "yanlış olur. Zincirin zayıf halkası ölçülmeden güçlendirilemez."))
    add(Paragraph(
        "<b>Bu projenin cevapladığı soru:</b> Türkçe gibi eklemeli bir dilde, "
        "getirme hattının hangi parçaları maliyetini hak ediyor?", S["note"]))

    add(para("Hattın parçaları", "h2"))
    add(para(
        "Belgeler doğrudan aranmaz; önce küçük parçalara bölünür "
        "(<b>chunking / parçalama</b>), her parça aranabilir hale getirilir "
        "(<b>indeksleme</b>), soru gelince en yakın parçalar sıralanır "
        "(<b>getirme</b>). Bu projede parçalamanın 3, getirmenin 4 farklı "
        "yolu var; 3 × 4 = 12 kombinasyonun hepsi aynı sorularla ölçülüyor."))

    # -------------------------------------------------------------- korpus
    add(para("2. Veri: korpus nasıl toplandı?", "h1"))
    add(para(
        "<b>Dosya:</b> <font name='Mono'>src/fetch_corpus.py</font>"))
    add(para(
        "Türkçe Vikipedi'den 54 sağlık maddesi çekiliyor (yaklaşık 1,09 "
        "milyon karakter). Vikipedi metni CC BY-SA 4.0 lisanslı, yani "
        "kaynak gösterilerek kullanılabilir. <b>Hiçbir hasta verisi, klinik "
        "kayıt veya kişisel sağlık bilgisi kullanılmıyor</b> — bu, sağlık "
        "alanında çalışan bir projede en baştan netleştirilmesi gereken bir "
        "sınır."))
    add(para(
        "Metin <font name='Mono'>explaintext</font> parametresiyle düz yazı "
        "olarak isteniyor. Bunun önemli bir yan faydası var: Vikipedi bölüm "
        "başlıklarını <font name='Mono'>== Başlık ==</font> biçiminde "
        "koruyor. Hiyerarşik parçalayıcı tam olarak bu işaretlere dayanıyor."))

    add(para("Yolda çıkan tuzak: batch isteği sessizce tek belge döndürüyor",
             "h2"))
    add(para(
        "İlk yazımda 60 başlık tek tek isteniyordu ve Vikipedi <b>429 Too "
        "Many Requests</b> döndürdü; sadece 11 belge geldi. Doğal çözüm "
        "istekleri gruplamaktı: API tek çağrıda birden fazla başlık kabul "
        "ediyor. 20'şerli gruplara geçtim — ve sonuç <b>13 belgede kaldı</b>."))
    add(para(
        "Burada tahmin etmek yerine ham cevaba baktım. API kendi uyarısında "
        "sebebi zaten söylüyordu:"))
    add(code('"exlimit" was too large for a whole article extracts\n'
             'request, lowered to 1.'))
    add(para(
        "Yani <b>tam makale metni istendiğinde batch çalışmıyor</b>; API "
        "sınırı zorla 1'e düşürüyor ve 20 başlık gönderip 1 metin alıyorsun. "
        "Bu, hata vermeden yanlış sonuç üreten türden bir davranış — sessiz "
        "olduğu için tehlikeli. Çözüm tek tek isteğe dönüp araya <b>1 saniye</b> "
        "koymak ve 429 durumunda artan bekleme ile (5s, 10s, 15s, 20s) yeniden "
        "denemek oldu. Sonuç: 54 belge."))
    add(Paragraph(
        "<b>Alınacak ders:</b> bir sistem beklenenden az veri döndürdüğünde "
        "sebebi tahmin etme; ham cevabı yazdır. Cevabın içinde çoğu zaman "
        "açıklaması yazılıdır.", S["note"]))

    # ------------------------------------------------------------ chunking
    add(PageBreak())
    add(para("3. Parçalama: aynı metni bölmenin üç yolu", "h1"))
    add(para("<b>Dosya:</b> <font name='Mono'>src/chunking.py</font>"))
    add(para(
        "Bir makale bütün olarak aranmaz, çünkü 20 bin karakterlik bir "
        "metnin tamamı tek bir &quot;anlam&quot;a sıkıştırılamaz. Parçalara "
        "bölmek şart — ama nasıl böldüğün sonucu doğrudan etkiliyor."))

    add(para("a) fixed — sabit pencere", "h2"))
    add(para(
        "En yaygın varsayılan: metni 700 karakterlik pencerelere böl, ardışık "
        "pencereler 100 karakter örtüşsün. Örtüşmenin sebebi, tam sınıra denk "
        "gelen bir cümlenin ikiye bölünüp iki parçada da anlamsız kalmasını "
        "engellemek. Basit, hızlı — ve ölçümde <b>en kötüsü</b>."))

    add(para("b) sentence — cümle sınırında bölme", "h2"))
    add(para(
        "Cümleleri asla ortadan kesmez; 700 karakterlik bütçeye sığdığı kadar "
        "tam cümle doldurur. Türkçede cümle bölmenin kendine has bir zorluğu "
        "var: nokta her zaman cümle bitirmez. &quot;Dr.&quot;, &quot;vb.&quot;, "
        "&quot;yy.&quot; gibi kısaltmalar ve &quot;1.&quot; gibi liste "
        "numaraları yanlış bölmeye yol açar. Kod önce noktadan böler, sonra "
        "kısaltma listesine bakıp yanlış bölünmeleri geri yapıştırır."))

    add(para("c) hierarchical — başlık yapısını koruyan bölme", "h2"))
    add(para(
        "Bu projenin asıl fikri. Önce makale <font name='Mono'>==</font> "
        "işaretlerinden bölümlere ayrılır, her bölüm kendi içinde cümlelere "
        "göre paketlenir — yani bir parça asla iki farklı bölüme yayılmaz. "
        "Ayrıca her parçanın önüne <b>başlık yolu</b> eklenir:"))
    add(code("Alzheimer hastalığı > Nedenler > Genetik > Geç başlangıç"))
    add(para(
        "Neden önemli? Bir paragrafı bağlamından kopardığında, o paragrafın "
        "<i>neyle ilgili olduğu</i> bilgisi kaybolur. &quot;Vakaların "
        "%1-5'inin sebebi genetik faktörlerdir&quot; cümlesi tek başına hangi "
        "hastalığı anlattığını söylemez. Başlık yolunu parçanın metnine "
        "eklediğinde bu bilgi geri gelir — ve gömme (embedding) hesaplanırken "
        "o bağlam da hesaba katılır."))
    add(para(
        "Kodda bu ayrım <font name='Mono'>body</font> (ham metin) ve "
        "<font name='Mono'>embed_text</font> (başlık yolu + ham metin) "
        "alanlarıyla yapılıyor. Aranan <font name='Mono'>embed_text</font>, "
        "kullanıcıya gösterilecek olan <font name='Mono'>body</font>."))

    # -------------------------------------------------------------- türkçe
    add(para("4. Türkçe: iki tuzak ve bir ucuz çözüm", "h1"))
    add(para("<b>Dosya:</b> <font name='Mono'>src/turkish_text.py</font>"))

    add(para("Tuzak 1: Python'un lower() fonksiyonu Türkçede bozuk", "h2"))
    add(para(
        "Türkçede noktalı ve noktasız i ayrı harflerdir: büyük <b>I</b>'nın "
        "küçüğü <b>ı</b>, büyük <b>İ</b>'nin küçüğü <b>i</b>. Python bunu "
        "bilmez ve iki ayrı hata yapar:"))
    add(code('"İLTİHAP".lower()  ->  i̇ltihap    (i + birleşen nokta U+0307)\n'
             '"ISIRIK".lower()   ->  isirik     (olması gereken: ısırık)'))
    add(para(
        "Birincisi görünüşte doğru ama aslında bir <b>i</b> harfi artı ayrı "
        "bir &quot;birleşen nokta&quot; karakteri; yani metindeki normal "
        "<b>i</b> ile eşleşmez. İkincisinde harf düpedüz yanlış. Her ikisi de "
        "arama sırasında sessizce eşleşme kaybettirir. Çözüm, genel "
        "küçültmeden <i>önce</i> Türkçeye özel bir harf haritası uygulamak."))
    add(Paragraph(
        "<b>Not:</b> bu haritayı ilk yazışımda <font name='Mono'>I→i</font> "
        "yazmıştım, yani hatayı tekrar üretmiştim. Testte "
        "<font name='Mono'>ISIRIK → isirik</font> çıkınca yakalandı ve "
        "<font name='Mono'>I→ı</font> olarak düzeltildi. Küçük bir birim testi "
        "olmasaydı bu hata tüm ölçümü sessizce bozacaktı.", S["note"]))

    add(para("Tuzak 2: Türkçe eklemeli bir dil", "h2"))
    add(para(
        "Türkçede kelimeler ek alarak uzar. Tek bir kavram metinde onlarca "
        "farklı yüzey biçiminde geçer:"))
    add(code("diyabet · diyabetin · diyabete · diyabetli · diyabetlilerin"))
    add(para(
        "Anahtar kelime araması (BM25) kelimeleri birebir eşleştirir. "
        "Kullanıcı &quot;diyabet&quot; yazar, belgede &quot;diyabetin&quot; "
        "geçer — eşleşme yok. İngilizcede bu sorun çok daha küçüktür, çünkü "
        "kelimeler bu kadar ek almaz. Türkçe arama sistemlerinde bu, göz ardı "
        "edilirse en büyük kayıp kaynağıdır."))

    add(para("Çözüm: 5 harflik gövde kesme", "h2"))
    add(para(
        "Tam çözüm bir <b>morfolojik çözümleyici</b> kullanmaktır (kelimeyi "
        "kök ve eklerine ayıran dilbilimsel araç) ama bu ağır bir bağımlılık. "
        "Bu proje çok daha kaba bir yöntem deniyor: <b>her kelimenin ilk 5 "
        "harfini al, gerisini at.</b>"))
    add(code("diyabetin -> diyab     diyabete -> diyab     diyabet -> diyab"))
    add(para(
        "Kaba olduğu açık: farklı kelimeler aynı 5 harfle başlıyorsa "
        "karışacaklar. Ama soru &quot;kusursuz mu&quot; değil, <b>&quot;maliyetini "
        "hak ediyor mu&quot;</b>. Ölçüm bunu net cevaplıyor:"))
    add(Spacer(1, 4))
    add(stemming_table())
    add(Spacer(1, 8))
    add(para(
        "Üç parçalama stratejisinde de <b>%23–%29 kazanç</b>. Tek bir dilim "
        "işleminden gelen bu kadar büyük bir fark, Türkçe getirme "
        "sistemlerinde gövdelemenin isteğe bağlı değil, zorunlu olduğunu "
        "gösteriyor."))

    # ----------------------------------------------------------- retrieval
    add(PageBreak())
    add(para("5. Getirme: dört yöntem", "h1"))
    add(para("<b>Dosya:</b> <font name='Mono'>src/retrieval.py</font>"))

    add(para("a) dense — anlam araması", "h2"))
    add(para(
        "Her metin parçası bir sinir ağı tarafından sayı listesine "
        "(<b>vektör / gömme</b>) çevrilir. Anlamca yakın metinler uzayda "
        "birbirine yakın düşer. Soru da aynı şekilde vektöre çevrilir ve en "
        "yakın parçalar bulunur. Yakınlık ölçüsü <b>kosinüs benzerliği</b>: "
        "iki vektör arasındaki açının kosinüsü, −1 ile 1 arasında."))
    add(para(
        "Kodda vektörler <font name='Mono'>normalize_embeddings=True</font> "
        "ile birim uzunluğa getiriliyor. Bunun faydası, kosinüs benzerliğinin "
        "basit bir <b>iç çarpıma</b> indirgenmesi — yani tüm korpusa karşı "
        "arama tek bir matris çarpımı oluyor. Kullanılan model "
        "<font name='Mono'>paraphrase-multilingual-MiniLM-L12-v2</font>: "
        "küçük, çok dilli, CPU'da çalışabilen bir model."))
    add(para(
        "Güçlü yanı: kelimeler tutmasa bile anlam tutarsa bulur. &quot;Nefes "
        "darlığı hastalığı&quot; sorusu &quot;astım&quot; belgesini "
        "getirebilir."))

    add(para("b) bm25 — anahtar kelime araması", "h2"))
    add(para(
        "Klasik ve hâlâ çok güçlü bir yöntem. Bir parçanın bir sorguya "
        "uygunluğunu, sorgudaki kelimelerin o parçada ne sıklıkta geçtiğine "
        "ve o kelimelerin tüm korpusta ne kadar <i>nadir</i> olduğuna bakarak "
        "puanlar. Nadir kelimeler daha ayırt edicidir: &quot;ve&quot; hiçbir "
        "şey söylemez, &quot;bradikardi&quot; çok şey söyler."))
    add(para(
        "Projede iki değişkeni var: <font name='Mono'>bm25_stem5</font> "
        "(5 harflik gövdeleme açık) ve <font name='Mono'>bm25_nostem</font> "
        "(kapalı). İkincisi bilerek kötü bir taban çizgisi olarak tutuluyor — "
        "gövdelemenin katkısını ölçebilmek için."))

    add(para("c) hybrid_rrf — iki listeyi birleştirme", "h2"))
    add(para(
        "Dense ve BM25 farklı sorularda başarısız olur. İkisini birleştirmek "
        "mantıklı, ama <b>puanları toplayamazsın</b>: kosinüs benzerliği 0–1 "
        "arasında, BM25 puanı ise korpusa göre değişen sınırsız bir sayı. "
        "Farklı ölçeklerdeki puanları toplamak anlamsızdır."))
    add(para(
        "<b>RRF (Reciprocal Rank Fusion)</b> bu sorunu puanları hiç "
        "kullanmayarak çözer — sadece <b>sıralamaya</b> bakar:"))
    add(code("puan(parça) = Σ  1 / (k + sıra)        k = 60"))
    add(para(
        "Her getiriciden gelen listede parçanın kaçıncı sırada olduğuna bakar, "
        "1/(60+sıra) ekler. İki listede de üstlerde çıkan bir parça toplamda "
        "öne geçer. <b>k = 60</b> değeri Cormack ve arkadaşlarının 2009 "
        "tarihli özgün RRF makalesinden gelir; k büyüdükçe ilk sıraların "
        "avantajı azalır, yani listeler daha eşit ağırlıkla harmanlanır. "
        "Ölçekten bağımsız olduğu için korpusa göre ayar gerektirmez."))

    # ------------------------------------------------------------ metrikler
    add(para("6. Metrikler: başarı nasıl ölçülüyor?", "h1"))
    add(para("<b>Dosya:</b> <font name='Mono'>src/metrics.py</font>"))
    add(para(
        "Dördü de aynı girdiyi alır: sıralı sonuç listesinde her sonucun "
        "doğru olup olmadığını gösteren bir doğru/yanlış dizisi."))
    add(Spacer(1, 2))
    add(table([
        ["Metrik", "Ne ölçer", "Ne zaman önemli"],
        ["Recall@k",
         "Doğru parçaların yüzde kaçı ilk k sonuca girdi",
         "Modele bağlam verirken: doğru parça listede yoksa cevap üretilemez"],
        ["Precision@k",
         "İlk k sonucun yüzde kaçı doğruydu",
         "Bağlam penceresi darsa: çöp sonuç yer kaplar"],
        ["MRR",
         "İlk doğru sonucun sırasının tersi (1. sıra = 1,00, 4. sıra = 0,25)",
         "Kullanıcıya tek cevap gösterilecekse"],
        ["nDCG@k",
         "Doğru sonuçlar ne kadar üst sıralarda; ideal sıralamaya oranlanır",
         "Genel sıralama kalitesi — bu projede ana metrik"],
    ], [2.4 * cm, 5.4 * cm, 6.5 * cm]))
    add(Spacer(1, 6))
    add(para(
        "<b>nDCG neden ana metrik?</b> Recall bir parçanın listede olup "
        "olmadığına bakar ama <i>nerede</i> olduğuna bakmaz; 1. sırada da "
        "10. sırada da aynı puanı verir. nDCG üst sıraları logaritmik olarak "
        "ödüllendirir, yani gerçek kullanım koşuluna daha yakındır."))

    add(para("Doğruluk nasıl tanımlandı? (en kritik tasarım kararı)", "h2"))
    add(para(
        "Her soru için altın veride <b>hangi belgede</b> ve o belgeden "
        "<b>birebir hangi metin parçasında</b> cevabın geçtiği yazılı. Bir "
        "sonuç, o belgeden geliyorsa <i>ve</i> o metin parçasını içeriyorsa "
        "doğru sayılır."))
    add(para(
        "Bu tanımın önemi şu: parçalama stratejisi değiştiğinde parça "
        "sınırları değişir, ama cevabın bulunduğu <i>metin</i> değişmez. "
        "Yani aynı 58 etiketle üç stratejiyi de ölçebiliyorsun. Eğer doğruluk "
        "&quot;şu numaralı parça doğrudur&quot; diye tanımlansaydı, her "
        "strateji için ayrı etiketleme gerekir ve karşılaştırma imkânsız "
        "olurdu."))

    add(para("Sorular neden parafraz?", "h2"))
    add(para(
        "58 sorunun hiçbiri belgedeki kelimelerle yazılmadı. Örneğin belgede "
        "&quot;yaklaşık %25'i, diyabet teşhisi konulduğunda diyabetik "
        "ketoasidoz da gelişmiştir&quot; yazarken soru şöyle: <i>&quot;Şeker "
        "hastalığı teşhisi konan kişilerin ne kadarında ketoasidoz da "
        "bulunuyor?&quot;</i>"))
    add(para(
        "Sebebi: sorular belgeden kopyalansaydı BM25 kelimeleri birebir "
        "bulacağı için haksız bir avantaj kazanırdı ve dense/sparse "
        "karşılaştırması anlamsızlaşırdı. Parafraz, gerçek kullanıcı "
        "davranışına da daha yakın — kimse belgedeki cümleyi ezberden yazmaz."))

    # ------------------------------------------------------------- sonuçlar
    add(PageBreak())
    add(para("7. Sonuçlar", "h1"))
    add(para("58 soru, 54 belge, 12 yapılandırma. Süreler CPU üzerinde, "
             "GPU kullanılmadı."))
    add(Spacer(1, 4))
    add(results_table())
    add(Spacer(1, 10))

    add(para("Dört bulgu", "h2"))
    add(KeepTogether(bullets([
        "<b>Türkçe gövdeleme en ucuz kazanç.</b> Tek bir dilim işlemi "
        "nDCG@10'u %23–29 artırıyor. Türkçe bir sistemde bu tartışmasız.",
        "<b>Hiyerarşik parçalama esas olarak dense tarafını iyileştiriyor.</b> "
        "Başlık yolu eklendiğinde dense nDCG@10 0,446 → 0,461 → 0,501 diye "
        "yükseliyor. BM25 daha az fayda görüyor, çünkü bölüm kelimeleri zaten "
        "metnin içinde geçiyordu.",
        "<b>Hibrit her koşulda kazanıyor.</b> En iyi yapılandırma (hiyerarşik "
        "+ hibrit, 0,607) en iyi tekil yöntemin (hiyerarşik + dense, 0,501) "
        "%21 üstünde. İki yöntem farklı sorularda hata yaptığı için "
        "birleştirmek gerçekten yeni bilgi katıyor.",
        "<b>Gecikme bu ölçekte sorun değil.</b> BM25 4–6 ms, dense 19–25 ms, "
        "hibrit 25–31 ms (P95). Hibridin ek maliyeti füzyondan değil dense "
        "ayağından geliyor.",
    ])))

    add(para("Ve bir de işe yaramayan şey", "h2"))
    add(Paragraph(
        "<b>Çok dilli gömme modeli, düz anahtar kelime aramasını "
        "yenemedi.</b> Üç parçalama stratejisinin ikisinde BM25 (gövdelemeli) "
        "dense'ten daha iyi sonuç verdi — üstelik yaklaşık 5 kat hızlı. Dense "
        "ancak hiyerarşik parçalama ile başlık bağlamı kazandığında öne "
        "geçebildi, o da kıl payı (0,501'e karşı 0,494).", S["note"]))
    add(para(
        "Bu, alandaki yaygın varsayımın (&quot;gömmeler anahtar kelimeyi "
        "yener&quot;) her durumda geçerli olmadığını gösteriyor. Kullanılan "
        "model küçük ve damıtılmış bir çok dilli model; Türkçe kapasitesi "
        "sınırlı. Bir sonraki adım Türkçeye özel ya da daha büyük bir gömme "
        "modeli denemek — düzenek bunu "
        "<font name='Mono'>--model</font> parametresiyle destekliyor."))
    add(para(
        "Negatif sonucu raporlamak bilerek yapılan bir tercih. Bir "
        "değerlendirme düzeneğinin değeri, beklentiyi doğruladığında değil, "
        "<b>beklentiyi yanlışladığında</b> ortaya çıkar."))

    # -------------------------------------------------------------- abstain
    add(para("8. Belirsizlik yönetimi: ne zaman insana devredilmeli?", "h1"))
    add(para("<b>Dosyalar:</b> <font name='Mono'>src/abstain.py</font>, "
             "<font name='Mono'>src/run_abstain.py</font>"))
    add(para(
        "Sağlık gibi düzenlemeye tabi bir alanda asıl soru &quot;sistem ne "
        "kadar doğru&quot; değil, <b>&quot;cevaplamayı seçtiği sorularda ne "
        "kadar doğru, ve kaç soruyu cevaplamayı seçti&quot;</b>. Bu ikisi "
        "birbirine terstir: eşiği yükselttikçe doğruluk artar ama daha az "
        "soru cevaplanır. Doğru yaklaşım tek bir sayı seçip eğriyi saklamak "
        "değil, eğrinin tamamını raporlamaktır."))

    add(para("Denenen birinci sinyal: sıralama farkı (margin)", "h2"))
    add(para(
        "İlk sezgi: birinci sonuç ikinciden ne kadar öndeyse sistem o kadar "
        "emindir. Ölçüm bunu <b>çürüttü</b>. Sebebi RRF'in kendisi: füzyon "
        "puanı 1/(60+sıra) olduğu için birinci ve ikinci arasındaki fark her "
        "sorguda yaklaşık aynı (~%2) çıkıyor — sorgu kolay da olsa zor da "
        "olsa. Yani margin, emin olmakla ilgili hiçbir bilgi taşımıyor."))
    add(abstain_table("margin"))
    add(Spacer(1, 6))
    add(para(
        "Tablo bunu açıkça gösteriyor: eşik yükseldikçe doğruluk artacağına "
        "<i>düşüyor</i>. İşe yaramayan bir sinyalin görüntüsü budur."))

    add(para("Denenen ikinci sinyal: ham kosinüs benzerliği (score)", "h2"))
    add(para(
        "Doğru sinyal <b>mutlak</b> bir yakınlık ölçüsü olmalı: korpustaki "
        "hiçbir parça soruya yakın değilse, sıralamanın şekli ne olursa olsun "
        "sistem emin olmamalı. Dense getiricinin ham kosinüs benzerliği tam "
        "olarak bunu verir."))
    add(abstain_table("score"))
    add(Spacer(1, 6))
    add(para(
        "Kod ayrıca bir doğruluk tabanı verildiğinde (%70, %80, %90) o tabanı "
        "tutturan en yüksek kapsamlı eşiği seçer. <b>Hiçbir eşik tabanı "
        "tutturamıyorsa bunu açıkça söyler</b> — ve bu bir hata değil, gerçek "
        "bir cevaptır: sistem o doğruluk şartıyla insansız çalıştırılmamalıdır."))

    # --------------------------------------------------------------- sınırlar
    add(para("9. Sınırlar — bu projenin söyleyemedikleri", "h1"))
    add(para(
        "Bir değerlendirme projesinin en kolay hatası, ölçtüğünden fazlasını "
        "iddia etmektir. Bu projenin sınırları:"))
    add(KeepTogether(bullets([
        "<b>58 soru küçük bir küme.</b> Yaklaşık 0,05'in altındaki nDCG "
        "farkları gürültü sayılmalı, sıralama olarak okunmamalı.",
        "<b>Tek etiketleyici, ikinci geçiş yok.</b> Altın veri tek kişi "
        "tarafından etiketlendi; etiketleyiciler arası uyum ölçüsü yok.",
        "<b>Tek gömme modeli denendi.</b> Dense sonuçları o modeli tarif "
        "eder, genel olarak dense getirmeyi değil.",
        "<b>Ansiklopedik metin, klinik metin değil.</b> Vikipedi dili klinik "
        "notlardan kelime dağarcığı, yapı ve kısaltma yoğunluğu bakımından "
        "farklıdır. Buradaki hiçbir sonuç klinik ortama yeniden ölçülmeden "
        "taşınamaz.",
        "<b>Sadece getirme.</b> Üretim aşaması ve cevap kalitesi hiç "
        "ölçülmedi.",
    ])))

    # ------------------------------------------------------------------ SSS
    add(PageBreak())
    add(para("10. Sık sorulacak sorular", "h1"))

    qa = [
        ("Neden morfolojik çözümleyici yerine 5 harflik kesme?",
         "Çünkü ölçtüm ve maliyetini fazlasıyla çıkardı: nDCG@10'da %23–29 "
         "kazanç, sıfır bağımlılık. Morfolojik çözümleyici muhtemelen daha "
         "iyisini yapar ama bu, ölçülmesi gereken bir sonraki deney — "
         "varsayılacak bir şey değil. Bilinen zayıflığı, aynı 5 harfle "
         "başlayan ilgisiz kelimeleri birleştirmesi."),
        ("RRF'te k neden 60?",
         "Cormack ve arkadaşlarının 2009 tarihli özgün RRF makalesindeki "
         "değer. k, ilk sıraların ne kadar baskın olacağını belirler: "
         "büyüdükçe sıralar arası fark azalır, listeler daha eşit harmanlanır. "
         "Bu korpusta ayrıca ayarlanmadı."),
        ("Neden puanları toplamak yerine RRF?",
         "Kosinüs benzerliği 0–1 arasında sınırlı, BM25 puanı ise korpusa "
         "bağlı sınırsız bir sayı. Farklı ölçeklerdeki puanları toplamak "
         "anlamsız olur ve her korpus için yeniden ağırlık ayarı gerektirir. "
         "RRF sadece sıralamaya baktığı için ölçekten bağımsızdır."),
        ("Dense neden BM25'i yenemedi? Bu bir hata mı?",
         "Hata değil, bulgu. Kullanılan model küçük, damıtılmış, çok dilli "
         "bir model ve Türkçe kapasitesi sınırlı. Türkçeye özel ya da daha "
         "büyük bir model muhtemelen tabloyu değiştirir — düzenek modeli "
         "parametreyle değiştirebilecek şekilde yazıldı, yani bu deney "
         "tek komutla tekrarlanabilir."),
        ("58 soru yeterli mi?",
         "Hayır, ve README'de böyle yazıyor. Bu büyüklükte bir kümede "
         "0,05'in altındaki farklar gürültüdür. Küme, üç stratejinin "
         "sıralamasını görmeye yeter; ince ayar kararları için yetmez."),
        ("Sağlık verisiyle çalışırken mahremiyet nasıl ele alındı?",
         "Hiçbir hasta verisi, klinik kayıt veya kişisel sağlık bilgisi "
         "kullanılmadı. Korpus tamamen kamuya açık ansiklopedik metin "
         "(Vikipedi, CC BY-SA 4.0) ve kaynak her parçada saklanıyor. Proje "
         "tıbbi cihaz değildir ve klinik karar desteği olarak sunulmuyor."),
        ("Bu sistemi üretime nasıl taşırsın?",
         "Üç eksik var. Birincisi, gömme matrisi bellekte tutuluyor — bu "
         "ölçekte doğru, ama büyük korpus için bir vektör veritabanı gerekir. "
         "İkincisi, üretim aşaması ve cevap kalitesi hiç ölçülmedi. Üçüncüsü, "
         "gecikme rakamları tek eşzamanlı sorgu içindir; yük altındaki "
         "davranış ayrıca ölçülmelidir."),
        ("En çok neyi yanlış yaptın?",
         "İki şeyi. Türkçe küçük harf haritasında I→i yazıp aynı hatayı "
         "tekrar ürettim; küçük bir testte yakalandı. Bir de Vikipedi'den az "
         "veri gelince sebebini tahmin edip batch'e geçtim — asıl sebep "
         "API'nin kendi uyarı mesajında yazılıydı ve ham cevaba bakınca "
         "hemen çıktı."),
    ]
    for question, answer in qa:
        add(KeepTogether([para(question, "h2"), para(answer)]))

    add(para("Çalıştırma", "h1"))
    add(code("python -m venv .venv\n"
             ".venv/Scripts/pip install -r requirements.txt\n"
             "python src/fetch_corpus.py     # korpusu Vikipedi'den kurar\n"
             "python src/run_eval.py         # 12 yapılandırmayı ölçer\n"
             "python src/run_abstain.py      # kapsam / doğruluk eğrisi\n"
             "python docs/make_pdf.py        # bu dokümanı üretir"))
    add(para(
        "Veri: Türkçe Vikipedi, CC BY-SA 4.0 — ayrıntı için "
        "<font name='Mono'>NOTICE.md</font>.", "subtitle"))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs"
    target.mkdir(parents=True, exist_ok=True)
    written = build(target / "turkish-rag-eval-teknik-aciklama.pdf")
    print(f"yazildi: {written}  ({written.stat().st_size / 1024:.0f} KB)")
