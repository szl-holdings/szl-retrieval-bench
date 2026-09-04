"""Ranking metrics: nDCG@k, Recall@k, P@k, R-precision, MRR, and MAP."""
import math


def _document_ids(values, name):
    """Return hashable document IDs or a fail-closed validation detail."""
    if values is None or isinstance(values, (str, bytes)):
        return None, f"{name} must be an iterable of document identifiers"
    try:
        documents = list(values)
        set(documents)
    except (TypeError, ValueError):
        return None, f"{name} must be an iterable of hashable document identifiers"
    return documents, None


def precision_at_k(ranked, relevant, k):
    """Return measured precision at exactly ``k`` ranking positions.

    The denominator remains ``k`` when a run is shorter than the requested
    cutoff, matching the conventional P@k definition. A ranking with no hits
    is a valid measurement and is reported as ``0.0``.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return {"state": "INVALID", "detail": "k must be >= 1"}
    ranked_docs, error = _document_ids(ranked, "ranked")
    if error:
        return {"state": "INVALID", "detail": error}
    relevant_docs, error = _document_ids(relevant, "relevant")
    if error:
        return {"state": "INVALID", "detail": error}
    relevant_set = set(relevant_docs)
    hits = len(set(ranked_docs[:k]) & relevant_set)
    return {"state": "MEASURED", f"P@{k}": round(hits / k, 4)}


def r_precision(ranked, relevant):
    """Return precision at R, where R is the number of relevant documents."""
    ranked_docs, error = _document_ids(ranked, "ranked")
    if error:
        return {"state": "INVALID", "detail": error}
    relevant_docs, error = _document_ids(relevant, "relevant")
    if error:
        return {"state": "INVALID", "detail": error}
    relevant_set = set(relevant_docs)
    total_relevant = len(relevant_set)
    if total_relevant == 0:
        return {
            "state": "INVALID",
            "detail": "no relevant docs - R-precision undefined",
        }
    hits = len(set(ranked_docs[:total_relevant]) & relevant_set)
    return {
        "state": "MEASURED",
        "R_precision": round(hits / total_relevant, 4),
        "R": total_relevant,
    }


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
