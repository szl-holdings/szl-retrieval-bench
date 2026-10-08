"""Reciprocal Rank Fusion over multiple rankings (Cormack et al., 2009)."""

from .metrics import _unique_ranking


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for i, doc in enumerate(_unique_ranking(ranking)):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores, key=lambda d: -scores[d])
