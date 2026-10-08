"""Benchmark harness. MEASURED results only come from real runs; anything else is BLOCKED.

Run states: MEASURED | BLOCKED | INVALID | FAILED. Never fabricate a metric.
"""
import hashlib
import json
import math
import time
from collections.abc import Mapping
from copy import deepcopy
from uuid import uuid4

from .bm25 import BM25
from .dense import TfidfDense
from .fuse import rrf
from .metrics import (
    _relevance_grades,
    _unique_ranking,
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
INPUT_SCHEMA = "szl.retrieval-bench.inputs/v1"
_DIGEST_FIELDS = ("corpus_sha256", "queries_sha256", "qrels_sha256")


def _sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_snapshot(values, name):
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must map nonempty string IDs to text")
    snapshot = dict(values)
    if any(not isinstance(key, str) or not key or not isinstance(text, str)
           for key, text in snapshot.items()):
        raise ValueError(f"{name} must map nonempty string IDs to text")
    return snapshot


def _snapshot_inputs(corpus, queries, qrels):
    """Bind the same private snapshot that the run will evaluate.

    Corpus and query order are included: stable ranking ties depend on corpus
    insertion order. Qrels may name documents outside the current corpus;
    those remain relevant misses, as in the existing metric contract.
    """
    corpus = _text_snapshot(corpus, "corpus")
    queries = _text_snapshot(queries, "queries")
    if not isinstance(qrels, Mapping):
        raise ValueError("qrels must map query IDs to relevance judgments")
    judgments = {}
    for query_id, grades in qrels.items():
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("qrels query IDs must be nonempty strings")
        judgments[query_id] = _relevance_grades(grades)
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in grades):
            raise ValueError("qrels document IDs must be nonempty strings")
    fingerprints = {
        "schema": INPUT_SCHEMA,
        "corpus_sha256": _sha256(list(corpus.items())),
        "queries_sha256": _sha256(list(queries.items())),
        "qrels_sha256": _sha256(judgments),
    }
    return corpus, queries, judgments, fingerprints


def _valid_fingerprints(binding):
    return (isinstance(binding, dict)
            and set(binding) == {"schema", *_DIGEST_FIELDS}
            and binding.get("schema") == INPUT_SCHEMA
            and all(isinstance(binding[key], str) and len(binding[key]) == 64
                    and all(char in "0123456789abcdef" for char in binding[key])
                    for key in _DIGEST_FIELDS))


def evaluate_run(run, qrels, k=10):
    try:
        run = _unique_ranking(run)
        qrels = _relevance_grades(qrels)
    except ValueError as error:
        return {"state": "INVALID", "detail": str(error)}
    relevant = {document for document, grade in qrels.items() if grade > 0}
    precision = precision_at_k(run, relevant, k)
    if precision["state"] != "MEASURED":
        return precision
    r_value = r_precision(run, relevant)
    if r_value["state"] != "MEASURED":
        return r_value
    try:
        gain = ndcg(run, qrels, k)
    except OverflowError:
        return {"state": "INVALID", "detail": "relevance grades exceed numeric range"}
    if not math.isfinite(gain):
        return {"state": "INVALID", "detail": "relevance grades exceed numeric range"}
    return {
        "state": "MEASURED",
        f"ndcg@{k}": round(gain, 4),
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
    try:
        corpus, queries, qrels, binding = _snapshot_inputs(corpus, queries, qrels)
    except ValueError as error:
        return {"state": "INVALID", "reason": str(error)}
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
            "run_id": str(uuid4()), "input_fingerprints": binding,
            "corpus_size": len(doc_ids), "elapsed_s": round(elapsed, 4),
            "aggregate": _aggregate(per_query), "per_query": per_query}


def run_dense(corpus, queries, qrels, k=10):
    """Classical TF-IDF dense lane (cosine over L2-normalized vectors)."""
    if not corpus or not queries:
        return {"state": "BLOCKED", "reason": "empty corpus or query set"}
    try:
        corpus, queries, qrels, binding = _snapshot_inputs(corpus, queries, qrels)
    except ValueError as error:
        return {"state": "INVALID", "reason": str(error)}
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
            "run_id": str(uuid4()), "input_fingerprints": binding,
            "corpus_size": len(doc_ids), "aggregate": _aggregate(per_query),
            "per_query": per_query}


def run_hybrid(corpus, queries, qrels, dense_rank_fn=None, k=10):
    """Hybrid = RRF(BM25, dense). Dense lane is pluggable; absent -> BLOCKED, not faked."""
    if dense_rank_fn is None:
        return {"state": "BLOCKED", "reason": "no dense ranker configured; refusing to fabricate hybrid results"}
    if not corpus or not queries:
        return {"state": "BLOCKED", "reason": "empty corpus or query set"}
    try:
        corpus, queries, qrels, binding = _snapshot_inputs(corpus, queries, qrels)
    except ValueError as error:
        return {"state": "INVALID", "reason": str(error)}
    doc_ids = list(corpus)
    known_ids = set(doc_ids)
    bm = BM25([corpus[d] for d in doc_ids])
    per_query = {}
    for qid, qtext in queries.items():
        # An adapter may mutate its ID list; never expose our ranking order.
        dense_output = dense_rank_fn(qtext, list(doc_ids))
        try:
            dense_ranking = _unique_ranking(dense_output)
            if any(document not in known_ids for document in dense_ranking):
                raise ValueError("ranking contains unknown corpus document identifiers")
        except ValueError as error:
            return {"state": "INVALID", "reason": f"query {qid}: dense ranking: {error}"}
        fused = rrf([bm.rank(qtext, doc_ids), dense_ranking])
        per_query[qid] = evaluate_run(fused, qrels.get(qid, {}), k)
    invalid = _invalid_evaluation(per_query)
    if invalid:
        return invalid
    return {"state": "MEASURED", "lane": "hybrid_rrf", "k": k, "queries": len(per_query),
            "run_id": str(uuid4()), "input_fingerprints": binding,
            "corpus_size": len(doc_ids),
            "aggregate": _aggregate(per_query), "per_query": per_query}


def compare(runs, chain=None):
    """Compare distinct runs bound to identical ordered evaluation inputs."""
    measured = [r for r in runs if r.get("state") == "MEASURED"]
    if len(measured) < 2:
        return {"state": "BLOCKED", "reason": "fewer than two MEASURED runs to compare"}
    qsets = [frozenset(r["per_query"]) for r in measured]
    if len(set(qsets)) != 1:
        return {"state": "INVALID", "reason": "query sets differ across runs; comparison is unfair"}
    cutoffs = {r["k"] for r in measured}
    if len(cutoffs) != 1:
        return {"state": "INVALID", "reason": "metric cutoffs differ across runs; comparison is unfair"}
    run_ids = [r.get("run_id") for r in measured]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        return {"state": "INVALID", "reason": "missing or invalid run IDs; comparison is unbound"}
    if len(set(run_ids)) != len(run_ids):
        return {"state": "INVALID", "reason": "duplicate run IDs; comparison would replay a run"}
    bindings = [r.get("input_fingerprints") for r in measured]
    if not all(_valid_fingerprints(binding) for binding in bindings):
        return {"state": "INVALID", "reason": "missing or invalid input fingerprints; comparison is unbound"}
    if any(binding != bindings[0] for binding in bindings[1:]):
        return {"state": "INVALID", "reason": "input fingerprints differ across runs; comparison is unfair"}
    board = [{"lane": r["lane"], "run_id": r["run_id"], **r["aggregate"]} for r in measured]
    score_key = f"ndcg@{next(iter(cutoffs))}"
    board.sort(key=lambda row: -row[score_key])
    result = {"state": "MEASURED", "leaderboard": board, "winner": board[0]["lane"],
              "winner_run_id": board[0]["run_id"], "k": next(iter(cutoffs)),
              "input_fingerprints": deepcopy(bindings[0])}
    if chain is not None:
        snapshot = deepcopy(result)
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
