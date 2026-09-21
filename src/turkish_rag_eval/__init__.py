"""turkish-rag-eval: which parts of a Turkish RAG pipeline actually pay off.

The package is a retrieval measurement harness. Point it at a corpus and a set
of labelled questions and it reports, with confidence intervals, what each
pipeline choice - chunking strategy, stemming, dense model, fusion - is worth
and what it costs.

Every published number is tagged with this version, because a benchmark whose
results cannot be reproduced against a fixed release is not comparable.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
