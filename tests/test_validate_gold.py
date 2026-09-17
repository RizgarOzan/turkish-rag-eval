import json

from gold import load_gold
from validate_gold import check_items, check_online

GOOD = {
    "qid": "ali-001",
    "question": "Şeker hastalığı teşhisi konan kişilerin ne kadarında ketoasidoz da bulunuyor?",
    "doc_id": "710430",
    "doc_title": "Diyabet",
    "answer_span": "yaklaşık %25'i, diyabet teşhisi konulduğunda diyabetik ketoasidoz da gelişmiştir",
    "annotator": "ali",
}
ARTICLE = {"title": "Diyabet",
           "text": "Giriş. " + "x " * 500 + GOOD["answer_span"].replace(" ", "\n  ") + " son."}


def test_valid_item_passes():
    assert check_items({"data/eval/contrib/ali.json": [GOOD]}) == []


def test_missing_field_is_reported():
    item = {k: v for k, v in GOOD.items() if k != "answer_span"}
    errors = check_items({"contrib/ali.json": [item]})
    assert len(errors) == 1 and "answer_span" in errors[0]


def test_contributed_items_need_an_annotator_but_gold_does_not():
    item = {k: v for k, v in GOOD.items() if k != "annotator"}
    assert check_items({"data/eval/gold.json": [item]}) == []
    assert "annotator" in check_items({"contrib/x.json": [item]})[0]


def test_duplicate_qid_across_files():
    other = dict(GOOD, question="Tip 1 diyabet belirtileri ne kadar sürede çıkar?")
    errors = check_items({"a.json": [GOOD], "b.json": [other]})
    assert errors == ["b.json ali-001: qid already used in a.json"]


def test_duplicate_question_ignores_whitespace_and_turkish_case():
    shouted = GOOD["question"].replace("i", "İ").replace("ı", "I").upper()
    other = dict(GOOD, qid="ali-002", question="  " + shouted)
    errors = check_items({"a.json": [GOOD, other]})
    assert errors == ["a.json ali-002: same question as a.json ali-001"]


def test_copied_question_is_rejected():
    copied = dict(GOOD, question="Diyabet teşhisi konulduğunda diyabetik ketoasidoz gelişmiş mi?")
    errors = check_items({"a.json": [copied]})
    assert len(errors) == 1 and "paraphrase" in errors[0]


def test_non_numeric_doc_id_and_missing_question_mark():
    item = dict(GOOD, doc_id="Diyabet", question=GOOD["question"].rstrip("?"))
    errors = check_items({"a.json": [item]})
    assert len(errors) == 2


def test_second_annotation_reusing_the_question_is_allowed():
    second = dict(GOOD, qid="ayse-001", answer_span="yaklaşık %25'i",
                  annotator="ayse", second_of="ali-001")
    errors = check_items({"data/eval/gold.json": [GOOD], "b.json": [second]})
    assert errors == []


def test_second_annotation_must_point_at_the_same_document():
    second = dict(GOOD, qid="ayse-001", doc_id="999999",
                  annotator="ayse", second_of="ali-001")
    errors = check_items({"data/eval/gold.json": [GOOD], "b.json": [second]})
    assert "different document" in errors[0]


def test_second_annotation_must_reuse_the_question_unchanged():
    second = dict(GOOD, qid="ayse-001", question="Bambaşka bir soru mu bu?",
                  annotator="ayse", second_of="ali-001")
    errors = check_items({"data/eval/gold.json": [GOOD], "b.json": [second]})
    assert "different question" in errors[0]


def test_second_of_must_reference_a_known_qid():
    second = dict(GOOD, qid="ayse-001", annotator="ayse", second_of="does-not-exist")
    errors = check_items({"b.json": [second]})
    assert "not a known qid" in errors[0]


def test_repeating_a_question_without_second_of_is_still_a_duplicate():
    other = dict(GOOD, qid="ayse-001", annotator="ayse")
    errors = check_items({"data/eval/gold.json": [GOOD], "b.json": [other]})
    assert "same question as" in errors[0]


def test_top_level_must_be_a_list():
    assert "JSON list" in check_items({"a.json": GOOD})[0]


def test_online_accepts_span_with_different_whitespace():
    assert check_online([GOOD], fetch=lambda _: ARTICLE, delay=0) == []


def test_online_reports_missing_article_wrong_title_and_missing_span():
    assert "no Turkish Wikipedia article" in check_online([GOOD], fetch=lambda _: None, delay=0)[0]
    wrong = {"title": "Astım", "text": ARTICLE["text"]}
    assert "not 'Diyabet'" in check_online([GOOD], fetch=lambda _: wrong, delay=0)[0]
    no_span = {"title": "Diyabet", "text": "y " * 600}
    assert "not found verbatim" in check_online([GOOD], fetch=lambda _: no_span, delay=0)[0]


def test_online_fetches_each_article_once():
    calls = []
    second = dict(GOOD, qid="ali-002")
    check_online([GOOD, second], fetch=lambda d: calls.append(d) or ARTICLE, delay=0)
    assert calls == ["710430"]


def test_load_gold_concatenates_files(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps([GOOD]), encoding="utf-8")
    b.write_text(json.dumps([dict(GOOD, qid="x")]), encoding="utf-8")
    assert [i["qid"] for i in load_gold([a, b])] == ["ali-001", "x"]


def test_shipped_gold_set_passes():
    from validate_gold import main
    import sys
    sys.argv = ["validate_gold.py"]
    assert main() == 0
