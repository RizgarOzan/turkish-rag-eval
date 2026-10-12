"""The query normaliser undoes the three ways people actually mistype Turkish:
no Turkish keyboard ("sigara icmek"), caps lock, and a pipeline that called
Python's str.lower() on capitals ("İ" -> "i" + combining dot)."""

from hypothesis import given, strategies as st

from turkish_rag_eval.normalize import QueryNormalizer, fold_ascii
from turkish_rag_eval.turkish_text import turkish_lower

CORPUS = [
    "Sigara içmek akciğeri doğrudan etkiler. Akciğer kanseri sık görülür.",
    "İstanbul, Türkiye'nin en kalabalık şehridir. İstanbul'da iki kıta var.",
    "Şeker hastalığı için düzenli ölçüm gerekir.",
    "OKB tedavisi uzun sürer mı diye sorulur; okb yaygındır.",
]
normalise = QueryNormalizer(CORPUS)


def test_fold_ascii_drops_every_turkish_mark():
    assert fold_ascii("çğıöşü ÇĞİÖŞÜ") == "cgiosu CGIOSU"
    assert fold_ascii("i̇stanbul") == "istanbul"


def test_restores_turkish_letters_from_the_corpus_vocabulary():
    assert normalise("sigara icmek akcigeri nasil etkiler") == \
        "sigara içmek akciğeri nasil etkiler"


def test_keeps_a_leading_capital_and_uses_dotted_capital_i():
    assert normalise("Istanbul kac kita var") == "İstanbul kac kıta var"


def test_caps_lock_is_lowered_the_turkish_way():
    assert normalise("İSTANBUL HANGİ KITADA") == "istanbul hangi kıtada"


def test_naive_python_lower_is_repaired():
    assert normalise("İSTANBUL".lower() + " nerede") == "istanbul nerede"


def test_a_short_acronym_is_not_treated_as_caps_lock():
    assert normalise("DNA nedir") == "DNA nedir"


def test_an_acronym_with_nothing_to_restore_keeps_its_case():
    # Found in the gold set: "OKB" came back as "Okb", "MS" as "Ms".
    assert normalise("OKB nasil tedavi edilir") == "OKB nasil tedavi edilir"


def test_two_letter_words_are_too_ambiguous_to_restore():
    # "mi" and "mı" are both question particles; vowel harmony, not the
    # corpus, decides. Found in the gold set: "mi?" came back as "mı?".
    assert normalise("uzun surer mi?") == "uzun sürer mi?"


def test_correct_text_is_left_alone():
    text = "Şeker hastalığı için ölçüm gerekir"
    assert normalise(text) == text


_letters = st.text(alphabet="abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ", max_size=40)


@given(st.one_of(_letters, st.sampled_from(CORPUS)))
def test_idempotent(text):
    once = normalise(text)
    assert normalise(once) == once


@given(st.one_of(_letters, st.sampled_from(CORPUS)))
def test_only_ever_adds_marks_and_lowers_case(text):
    """Strip marks and case from input and output: they must be the same
    letters. The normaliser may restore a cedilla or lower a shout, never
    swap one word for another."""
    assert fold_ascii(turkish_lower(normalise(text))) == fold_ascii(turkish_lower(text))


@given(st.sampled_from(CORPUS[:3]))
def test_folding_then_normalising_recovers_corpus_text(text):
    assert normalise(fold_ascii(text)) == text
