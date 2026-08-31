"""Retrieval metrics. All take a ranked list of booleans (is this hit relevant).

Recall@k    - share of the query's relevant chunks that made the top k.
Precision@k - share of the top k that is relevant.
MRR         - 1 / rank of the first relevant hit, 0 if none.
nDCG@k      - rank-discounted gain, normalised by the best possible ordering.
"""

import math


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
