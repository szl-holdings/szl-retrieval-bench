"""Ranking metrics: nDCG@k, Recall@k, MRR, MAP. Conventions match TREC/BEIR."""
import math


def _dcg(rels, k):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(run, qrels, k=10):
    rels = [qrels.get(d, 0) for d in run[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    denom = _dcg(ideal, k)
    return _dcg(rels, k) / denom if denom > 0 else 0.0


def recall_at(run, qrels, k):
    rel = {d for d, r in qrels.items() if r > 0}
    return len(set(run[:k]) & rel) / len(rel) if rel else 0.0


def mrr(run, qrels):
    for i, d in enumerate(run):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(run, qrels):
    rel = {d for d, r in qrels.items() if r > 0}
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(run):
        if d in rel:
            hits += 1
            total += hits / (i + 1)
    return total / len(rel)
