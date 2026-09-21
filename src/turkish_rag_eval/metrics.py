"""Retrieval metrics. All take a ranked list of booleans (is this hit relevant).

Recall@k    - share of the query's relevant chunks that made the top k.
Precision@k - share of the top k that is relevant.
MRR         - 1 / rank of the first relevant hit, 0 if none.
nDCG@k      - rank-discounted gain, normalised by the best possible ordering.

Plus the uncertainty around them. With 58 queries a difference of a few points
can easily be sampling noise, and "roughly 0.05 is noise" was an eyeballed
number. ``bootstrap_ci`` replaces it with a measured interval, and
``paired_bootstrap_ci`` answers the question that actually gets asked - is A
better than B - on the pairing the data already has: both systems answered the
same queries, so comparing them per query removes the variance from queries
being easy or hard, which is most of it.
"""

import math

import numpy as np

#: Enough resamples that the interval endpoints are stable to three decimals,
#: cheap enough to run for every configuration.
RESAMPLES = 10_000

#: Fixed so an interval is reproducible. A seed that moves between runs would
#: reintroduce exactly the irreproducibility the rest of the harness removes.
SEED = 20260921


def recall_at_k(relevance: list[bool], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def precision_at_k(relevance: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(relevance[:k]) / k


def reciprocal_rank(relevance: list[bool]) -> float:
    for i, hit in enumerate(relevance, 1):
        if hit:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevance: list[bool], total_relevant: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1)
              for i, hit in enumerate(relevance[:k], 1) if hit)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(total_relevant, k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def _resample_means(values: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """Means of ``resamples`` draws of len(values) items, with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    return values[draws].mean(axis=1)


def bootstrap_ci(per_query: list[float], confidence: float = 0.95,
                 resamples: int = RESAMPLES,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile interval for the mean of a per-query metric.

    Queries are the unit of resampling because they are the unit of sampling:
    the gold set is one draw from the questions someone might ask, and the
    interval says how much the reported mean would move had we drawn different
    ones. Resampling anything finer (chunks, documents) would answer a question
    nobody asked and give a falsely narrow interval.
    """
    values = np.asarray(per_query, dtype=float)
    if values.size == 0:
        return (0.0, 0.0)
    if values.size == 1:
        return (float(values[0]), float(values[0]))

    tail = (1.0 - confidence) / 2.0
    means = _resample_means(values, resamples, seed)
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return (float(low), float(high))


def paired_bootstrap_ci(a_per_query: list[float], b_per_query: list[float],
                        confidence: float = 0.95, resamples: int = RESAMPLES,
                        seed: int = SEED) -> tuple[float, float]:
    """Interval for mean(a) - mean(b), resampling the *pairs*.

    Both systems answered the same queries, so the per-query difference is what
    carries the signal. Bootstrapping the two systems independently would mix
    in the variance of which queries are hard - the same variance for both -
    and hide differences that are perfectly real.
    """
    if len(a_per_query) != len(b_per_query):
        raise ValueError(
            f"paired comparison needs the same queries on both sides; "
            f"got {len(a_per_query)} and {len(b_per_query)}")
    differences = [a - b for a, b in zip(a_per_query, b_per_query)]
    return bootstrap_ci(differences, confidence, resamples, seed)


def separates(interval: tuple[float, float]) -> bool:
    """True when a difference interval excludes zero, i.e. the sign is settled."""
    low, high = interval
    return low > 0.0 or high < 0.0
