"""Properties of the confidence intervals the README's tables now carry.

These are the checks that would catch an interval that looks plausible and is
wrong: one that moves between runs, one that ignores the pairing, or one that
narrows when the evidence gets weaker.
"""

import pytest

from turkish_rag_eval.metrics import (
    bootstrap_ci,
    paired_bootstrap_ci,
    separates,
)

pytest.importorskip("numpy")


def test_interval_brackets_the_mean():
    values = [0.2, 0.4, 0.6, 0.8, 1.0] * 12
    low, high = bootstrap_ci(values)
    assert low < sum(values) / len(values) < high


def test_interval_is_reproducible():
    values = [0.0, 1.0] * 30
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_more_queries_narrow_the_interval():
    few = bootstrap_ci([0.0, 1.0] * 10)
    many = bootstrap_ci([0.0, 1.0] * 200)
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_no_spread_means_no_interval():
    assert bootstrap_ci([0.7] * 40) == (0.7, 0.7)


def test_degenerate_inputs():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.5]) == (0.5, 0.5)


def test_wider_confidence_gives_a_wider_interval():
    values = [i / 50 for i in range(50)]
    narrow = bootstrap_ci(values, confidence=0.80)
    wide = bootstrap_ci(values, confidence=0.99)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_paired_comparison_detects_a_consistent_small_win():
    # A wins on every query by a hair. Per-query spread is large, so comparing
    # the two systems independently would drown this; the pairing keeps it.
    a = [0.10, 0.50, 0.90, 0.30, 0.70] * 12
    b = [value - 0.02 for value in a]

    paired = paired_bootstrap_ci(a, b)
    assert separates(paired), "a consistent win should not straddle zero"
    assert paired[0] > 0

    independent_a, independent_b = bootstrap_ci(a), bootstrap_ci(b)
    assert independent_a[0] < independent_b[1], (
        "the independent intervals overlap, which is why pairing is used")


def test_paired_comparison_stays_undecided_on_noise():
    a = [0.1, 0.9, 0.2, 0.8, 0.5] * 12
    b = [0.9, 0.1, 0.8, 0.2, 0.5] * 12
    assert not separates(paired_bootstrap_ci(a, b))


def test_paired_comparison_requires_the_same_queries():
    with pytest.raises(ValueError, match="same queries"):
        paired_bootstrap_ci([0.1, 0.2], [0.1])


def test_separates_reads_both_signs():
    assert separates((0.01, 0.05))
    assert separates((-0.05, -0.01))
    assert not separates((-0.01, 0.05))
    assert not separates((0.0, 0.05))
