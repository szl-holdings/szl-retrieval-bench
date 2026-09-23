"""Benchmark harness. MEASURED results only come from real runs; anything else is BLOCKED.

Run states: MEASURED | BLOCKED | INVALID | FAILED. Never fabricate a metric.
"""
import json
import time

from .bm25 import BM25
from .dense import TfidfDense
from .fuse import rrf
from .metrics import (
    average_precision,
    mrr,
    ndcg,
    precision_at_k,
    r_precision,
    recall_at,
)
from .receipts import ReceiptChain

STATES = ("MEASURED", "BLOCKED", "INVALID", "FAILED")
HARNESS = "szl-retrieval-bench"


def evaluate_run(run, qrels, k=10):
    relevant = {document for document, grade in qrels.items() if grade > 0}
    precision = precision_at_k(run, relevant, k)
    if precision["state"] != "MEASURED":
        return precision
    r_value = r_precision(run, relevant)
    if r_value["state"] != "MEASURED":
        return r_value
    return {
        "state": "MEASURED",
        f"ndcg@{k}": round(ndcg(run, qrels, k), 4),
        f"recall@{k}": round(recall_at(run, qrels, k), 4),
        f"P@{k}": precision[f"P@{k}"],
        "R_precision": r_value["R_precision"],
        "mrr": round(mrr(run, qrels), 4),
        "map": round(average_precision(run, qrels), 4),
    }


def _aggregate(per_query):
    first = next(iter(per_query.values()))
    metrics = [name for name, value in first.items()
               if name != "state" and isinstance(value, (int, float))]
    return {name: round(sum(row[name] for row in per_query.values()) / len(per_query), 4)
            for name in metrics}


def _invalid_evaluation(per_query):
    for query_id, result in per_query.items():
        if result.get("state") != "MEASURED":
            return {
                "state": "INVALID",
                "reason": f"query {query_id}: {result.get('detail', 'metric evaluation failed')}",
            }
    return None


def run_bm25(corpus, queries, qrels, k=10):
    """corpus: {doc_id: text}; queries: {qid: text}; qrels: {qid: {doc_id: grade}}."""
    if not corpus or not queries:
        return {"state": "BLOCKED", "reason": "empty corpus or query set"}
    doc_ids = list(corpus)
    bm = BM25([corpus[d] for d in doc_ids])
    t0 = time.perf_counter()
    per_query = {}
    for qid, qtext in queries.items():
        run = bm.rank(qtext, doc_ids)
        per_query[qid] = evaluate_run(run, qrels.get(qid, {}), k)
    invalid = _invalid_evaluation(per_query)
    if invalid:
        return invalid
    elapsed = time.perf_counter() - t0
    return {"state": "MEASURED", "lane": "bm25", "k": k, "queries": len(per_query),
            "corpus_size": len(doc_ids), "elapsed_s": round(elapsed, 4),
            "aggregate": _aggregate(per_query), "per_query": per_query}


def run_dense(corpus, queries, qrels, k=10):
    """Classical TF-IDF dense lane (cosine over L2-normalized vectors)."""
    if not corpus or not queries:
        return {"state": "BLOCKED", "reason": "empty corpus or query set"}
    doc_ids = list(corpus)
    dense = TfidfDense([corpus[d] for d in doc_ids])
    per_query = {}
    for qid, qtext in queries.items():
        run = dense.rank(qtext, doc_ids)
        per_query[qid] = evaluate_run(run, qrels.get(qid, {}), k)
    invalid = _invalid_evaluation(per_query)
    if invalid:
        return invalid
    return {"state": "MEASURED", "lane": "tfidf-dense", "k": k, "queries": len(per_query),
            "corpus_size": len(doc_ids), "aggregate": _aggregate(per_query),
            "per_query": per_query}


def run_hybrid(corpus, queries, qrels, dense_rank_fn=None, k=10):
    """Hybrid = RRF(BM25, dense). Dense lane is pluggable; absent -> BLOCKED, not faked."""
    if dense_rank_fn is None:
        return {"state": "BLOCKED", "reason": "no dense ranker configured; refusing to fabricate hybrid results"}
    doc_ids = list(corpus)
    bm = BM25([corpus[d] for d in doc_ids])
    per_query = {}
    for qid, qtext in queries.items():
        fused = rrf([bm.rank(qtext, doc_ids), dense_rank_fn(qtext, doc_ids)])
        per_query[qid] = evaluate_run(fused, qrels.get(qid, {}), k)
    invalid = _invalid_evaluation(per_query)
    if invalid:
        return invalid
    return {"state": "MEASURED", "lane": "hybrid_rrf", "k": k, "queries": len(per_query),
            "aggregate": _aggregate(per_query), "per_query": per_query}


def compare(runs, chain=None):
    """Fairness gate: runs must cover identical query sets to be compared."""
    measured = [r for r in runs if r.get("state") == "MEASURED"]
    if len(measured) < 2:
        return {"state": "BLOCKED", "reason": "fewer than two MEASURED runs to compare"}
    qsets = [frozenset(r["per_query"]) for r in measured]
    if len(set(qsets)) != 1:
        return {"state": "INVALID", "reason": "query sets differ across runs; comparison is unfair"}
    cutoffs = {r["k"] for r in measured}
    if len(cutoffs) != 1:
        return {"state": "INVALID", "reason": "metric cutoffs differ across runs; comparison is unfair"}
    board = [{"lane": r["lane"], **r["aggregate"]} for r in measured]
    score_key = f"ndcg@{next(iter(cutoffs))}"
    board.sort(key=lambda row: -row[score_key])
    result = {"state": "MEASURED", "leaderboard": board, "winner": board[0]["lane"]}
    if chain is not None:
        snapshot = {"state": result["state"], "winner": result["winner"],
                    "leaderboard": [dict(row) for row in board]}
        result["receipt"] = chain.emit({"harness": HARNESS, "type": "comparison",
                                        "result": snapshot})
    return result


def main():
    corpus = {"d1": "the cat sat on the mat", "d2": "dogs bark at night",
              "d3": "the cat chased the dog", "d4": "neural networks learn representations"}
    queries = {"q1": "cat mat", "q2": "neural representations"}
    qrels = {"q1": {"d1": 2, "d3": 1}, "q2": {"d4": 2}}
    chain = ReceiptChain()
    sparse = run_bm25(corpus, queries, qrels)
    dense_run = run_dense(corpus, queries, qrels)
    doc_ids = list(corpus)
    classical = TfidfDense([corpus[d] for d in doc_ids])
    hybrid = run_hybrid(corpus, queries, qrels,
                        dense_rank_fn=lambda q, ids: classical.rank(q, ids))
    verdict = compare([sparse, dense_run, hybrid], chain)
    out = {"bm25": sparse, "tfidf_dense": dense_run, "hybrid_rrf": hybrid,
           "comparison": verdict, "chain_valid": chain.verify()}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
