import pytest

from turkish_rag_eval.chunking import split_sentences


@pytest.mark.parametrize(
    "text",
    [
        "Robert W. Holley bir maya RNA'sının yapısını çözdü.",
        "Osmanlı Padişahı I. Bayezid Niğbolu'da kazandı.",
        "Fransa Kralı XIV. Louis uzun süre hüküm sürdü.",
        "Bunu J. R. R. Tolkien yazdı.",
        "Savaştan sonra II. Mehmed tahta çıktı.",
        "Sözleşmeyi Ömer Ş. Yılmaz imzaladı.",
    ],
)
def test_initials_and_roman_numerals_do_not_end_a_sentence(text: str) -> None:
    assert split_sentences(text) == [text]


def test_lone_capital_after_lowercase_word_still_ends_a_sentence() -> None:
    text = "Seçenekler A, B, C, D ve E. Sonraki cümle burada başlar."

    assert split_sentences(text) == [
        "Seçenekler A, B, C, D ve E.",
        "Sonraki cümle burada başlar.",
    ]


def test_ordinary_sentence_ends_still_split() -> None:
    text = "Ankara başkenttir. İstanbul en kalabalık şehirdir! Neden mi?"

    assert split_sentences(text) == [
        "Ankara başkenttir.",
        "İstanbul en kalabalık şehirdir!",
        "Neden mi?",
    ]


def test_existing_abbreviation_and_number_rules_are_kept() -> None:
    text = "Dr. Ayşe geldi. Liste 1. madde ile başlar."

    assert split_sentences(text) == ["Dr. Ayşe geldi.", "Liste 1. madde ile başlar."]
