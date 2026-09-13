"""Tests for the three metrics every number in the README rests on.

Run with: python -m pytest tests -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from turkish_text import tokenize, turkish_lower  # noqa: E402


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([True, True, False, False], total_relevant=2, k=4) == 1.0


def test_ndcg_ideal_is_capped_at_k():
    # Six relevant chunks but k=2: retrieving two of them is still perfect at k.
    assert ndcg_at_k([True, True], total_relevant=6, k=2) == 1.0


def test_ndcg_rewards_the_higher_rank():
    high = ndcg_at_k([True, False, False], total_relevant=1, k=3)
    low = ndcg_at_k([False, False, True], total_relevant=1, k=3)
    assert high == 1.0
    assert high > low > 0.0


def test_ndcg_is_zero_without_a_hit_and_safe_without_relevants():
    assert ndcg_at_k([False, False], total_relevant=1, k=2) == 0.0
    assert ndcg_at_k([False], total_relevant=0, k=1) == 0.0


def test_ndcg_ignores_hits_past_k():
    assert ndcg_at_k([False, False, True], total_relevant=1, k=2) == 0.0


def test_recall_denominator_is_the_number_of_relevants():
    assert recall_at_k([True, False, True, True], total_relevant=4, k=2) == 0.25
    assert recall_at_k([True, True], total_relevant=2, k=5) == 1.0
    assert recall_at_k([False], total_relevant=0, k=5) == 0.0


def test_precision_at_k():
    assert precision_at_k([True, False, True, True], 4) == 0.75
    assert precision_at_k([True], 0) == 0.0


def test_reciprocal_rank_is_the_reciprocal_of_the_first_hit():
    assert reciprocal_rank([False, True, True]) == 0.5
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_turkish_lower_handles_dotted_and_dotless_i():
    assert turkish_lower("\u0130LT\u0130HAP") == "iltihap"
    assert turkish_lower("ISIRIK") == "\u0131s\u0131r\u0131k"
    # Decomposed \u0130 (I + U+0307) must survive NFC normalisation.
    assert turkish_lower("I\u0307LT\u0130HAP") == "iltihap"


def test_tokenize_stems_to_a_fixed_prefix_and_drops_stopwords():
    assert tokenize("diyabetin ve diyabete", stem_length=5) == ["diyab", "diyab"]
    assert tokenize("diyabetin", stem_length=None) == ["diyabetin"]
