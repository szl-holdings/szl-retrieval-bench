"""Proof-carrying late-interaction research lane: token-level MaxSim scoring.

Research-only, offline, dependency-free (stdlib only) adapter implementing the
classical late-interaction idea (ColBERT-style MaxSim; Santhanam et al., NAACL
2022) as pure math over caller-supplied token vectors. A query and each
candidate document are represented as bags of real vectors (one per token);
the score is a weighted mean over query tokens of the maximum cosine
similarity against the document's tokens:

    score(Q, D) = ( sum_i w_i * max_j cos(q_i, d_j) ) / ( sum_i w_i )

with strictly positive finite weights w_i. The unweighted arm uses w_i = 1
(so the score is the plain mean of per-token maxima, the original MaxSim).
Ranking ties are broken deterministically by document ID. Only the ranking
inputs (query token vectors, candidate document token vectors, weights,
candidate ID order, cutoff) may influence the score; relevance judgments
(qrels) are used strictly after ranking, for metrics.

Model-capability statement: this module operates on SYNTHETIC CALLER-SUPPLIED
token vectors. It is NOT the Qwen3-Embedding, ColBERTv2, or any neural model,
and it uses no model weights, no network access, and no learned parameters.
It makes NO benchmark-superiority claim: its outputs are paired experiments
on the supplied fixtures only, with cost accounting and regression flags, and
should not be read as rights-cleared external evaluation evidence.

An optional third post-filter comparison against the four-axis
evidence-admission adapter is intentionally NOT implemented in this module
yet (kept separate so ranking math and admission semantics never silently
couple); see the notes field of the returned receipt.

Fail-closed doctrine: malformed or oversized inputs return ``{"state":
"INVALID", "reason": <short code>}`` without echoing vectors or text; the
module never fabricates zero metrics for inputs it could not evaluate.
"""
from __future__ import annotations

import hashlib
import json
import math

MODULE_VERSION = "0.1.0"
SOURCE_ID = "szl-retrieval-bench/proof-carrying_late_interaction"

# Explicit strict shape / denial-of-service caps. Every cap is enforced BEFORE
# any scoring work; exceeding any cap is INVALID, never silently truncated.
MAX_QUERY_TOKENS = 64
MAX_DOC_TOKENS = 48
MAX_DOCS = 256
MAX_DIM = 2048
MAX_K = 128
MAX_DOT_PRODUCTS = 700_000  # op budget; below the shape-cap ceiling (64*256*48), so the
                             # per-item caps do NOT silently bypass the tensor budget
MAX_ELEMENT_ABS = 1e6         # |component| bound for every input vector
WEIGHT_MIN = 1e-12            # strictly positive, bounded from below
WEIGHT_MAX = 1e12             # bounded from above so weighted sums cannot overflow


def _fail(reason: str) -> dict:
    # Fail closed: a short machine-readable code, never raw vectors/text.
    return {"state": "INVALID", "reason": reason, "source": SOURCE_ID, "version": MODULE_VERSION}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_vector(v, dim, label) -> str | None:
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        return label + ":vector_not_nonempty_list"
    if len(v) != dim:
        return label + ":vector_dim_mismatch"
    for i, x in enumerate(v):
        if not _is_number(x):
            return label + ":vector_element_not_number:index=" + str(i)
        if not math.isfinite(x):
            return label + ":vector_element_not_finite:index=" + str(i)
        if abs(x) > MAX_ELEMENT_ABS:
            return label + ":vector_element_magnitude_exceeds_cap:index=" + str(i)
    return None


def _normalize(v):
    # math.hypot cannot overflow; a zero vector was refused earlier.
    n = math.hypot(*v)
    return [x / n for x in v]


def _validate_inputs(query_vectors, doc_vectors, doc_ids, weights, k):
    """Validate everything up front. Returns None on success or a failure dict."""
    if not isinstance(query_vectors, (list, tuple)) or len(query_vectors) == 0:
        return _fail("query_tokens_empty")
    if len(query_vectors) > MAX_QUERY_TOKENS:
        return _fail("query_tokens_exceed_cap")
    if not isinstance(doc_vectors, (list, tuple)) or len(doc_vectors) == 0:
        return _fail("doc_vectors_empty")
    if not isinstance(doc_ids, (list, tuple)) or len(doc_ids) != len(doc_vectors):
        return _fail("doc_count_mismatch_with_ids")
    if len(doc_ids) > MAX_DOCS:
        return _fail("docs_exceed_cap")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > MAX_K:
        return _fail("cutoff_invalid")
    if len(set(doc_ids)) != len(doc_ids):
        return _fail("doc_ids_not_unique")
    for i, did in enumerate(doc_ids):
        if not isinstance(did, str) or not did:
            return _fail("doc_id_not_nonempty_string:index=" + str(i))
    if not isinstance(query_vectors[0], (list, tuple)) or len(query_vectors[0]) == 0:
        return _fail("query:token:0:vector_not_nonempty_list")
    dim = len(query_vectors[0])
    if dim < 1 or dim > MAX_DIM:
        return _fail("dim_out_of_range")
    for i, vec in enumerate(query_vectors):
        err = _check_vector(vec, dim, "query:token:" + str(i))
        if err:
            return _fail(err)
    zero_vector_positions = []
    for i, vec in enumerate(query_vectors):
        if math.hypot(*vec) == 0.0:
            zero_vector_positions.append(("query", i, 0))
    total_doc_tokens = 0
    for di, dv in enumerate(doc_vectors):
        if not isinstance(dv, (list, tuple)) or len(dv) == 0:
            return _fail("doc_tokens_empty:index=" + str(di))
        if len(dv) > MAX_DOC_TOKENS:
            return _fail("doc_tokens_exceed_cap:index=" + str(di))
        for ti, vec in enumerate(dv):
            err = _check_vector(vec, dim, "doc:" + str(di) + ":token:" + str(ti))
            if err:
                return _fail(err)
            if math.hypot(*vec) == 0.0:
                zero_vector_positions.append(("doc", di, ti))
        total_doc_tokens += len(dv)
    if zero_vector_positions:
        kind, di, ti = zero_vector_positions[0]
        if kind == "query":
            return _fail("query_zero_vector:index=" + str(di))
        return _fail("doc_zero_vector:doc:" + str(di) + ":token:" + str(ti))
    if weights is not None:
        if not isinstance(weights, (list, tuple)):
            return _fail("weights_not_list")
        if len(weights) != len(query_vectors):
            return _fail("weights_length_mismatch")
        for i, w in enumerate(weights):
            if not _is_number(w):
                return _fail("weight_not_number:index=" + str(i))
            if not math.isfinite(w) or w <= 0.0:
                return _fail("weight_not_strictly_positive_finite:index=" + str(i))
            if w < WEIGHT_MIN or w > WEIGHT_MAX:
                return _fail("weight_out_of_bounded_range:index=" + str(i))
    dot_products = len(query_vectors) * total_doc_tokens
    if dot_products > MAX_DOT_PRODUCTS:
        return _fail("estimated_dot_products_exceed_cap")
    return None


def _canonical_digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _maxsim_arm(query_norm, docs_norm, doc_ids, weights):
    """Score all candidates. Pre-normalized vectors; weights None => unweighted."""
    scores = []
    per_token_maxima_total = 0.0
    for dv in docs_norm:
        weighted_sum = 0.0
        weight_total = 0.0
        for qi, qv in enumerate(query_norm):
            best = -1.0
            for dtv in dv:
                s = math.fsum(a * b for a, b in zip(qv, dtv))
                if s > best:
                    best = s
            w = 1.0 if weights is None else weights[qi]
            weighted_sum += w * best
            weight_total += w
            per_token_maxima_total += best
        scores.append(weighted_sum / weight_total)
    order = sorted(range(len(doc_ids)), key=lambda i: (-scores[i], doc_ids[i]))
    ranking = [doc_ids[i] for i in order]
    return ranking, scores, order, per_token_maxima_total


def _ranked_metrics(ranking, relevant, k):
    """P@k (always /k), binary nDCG@k, MRR. Pure functions on the final ranking."""
    r = min(k, len(ranking))
    hits = sum(1 for i in range(r) if ranking[i] in relevant)
    p_at_k = hits / k
    dcg = math.fsum(1.0 / math.log2(i + 2) for i in range(r) if ranking[i] in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = math.fsum(1.0 / math.log2(i + 2) for i in range(ideal_hits)) if ideal_hits else 0.0
    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0
    mrr = 0.0
    for i in range(len(ranking)):
        if ranking[i] in relevant:
            mrr = 1.0 / (i + 1)
            break
    return {"P_at_k": round(p_at_k, 6), "nDCG_at_k": round(ndcg, 6), "MRR": round(mrr, 6)}


def evaluate_two_arm(query_vectors, doc_vectors, doc_ids, qrels, cutoff=10, weights=None):
    """Paired fixed-candidate/fixed-qrels experiment: unweighted vs weighted MaxSim.

    Pure function, deterministic, offline. Returns an ``EVALUATED_OFFLINE``
    record (this is a research adapter, not deployment authorization) or an
    ``INVALID`` record on malformed input. qrels never influence either
    ranking; they are bound to the receipt digests and used for metrics only.
    """
    invalid = _validate_inputs(query_vectors, doc_vectors, doc_ids, weights, cutoff)
    if invalid is not None:
        return invalid
    if not isinstance(qrels, (set, frozenset, list, tuple)):
        return _fail("qrels_not_a_collection")
    relevant = set(qrels)
    if not relevant:
        return _fail("qrels_empty")
    for did in relevant:
        if did not in set(doc_ids):
            return _fail("qrels_doc_not_a_candidate")

    query_norm = [_normalize(v) for v in query_vectors]
    docs_norm = [[_normalize(v) for v in dv] for dv in doc_vectors]
    total_doc_tokens = sum(len(dv) for dv in doc_vectors)
    dot_products = len(query_vectors) * total_doc_tokens

    ranking_u, scores_u, _, _ = _maxsim_arm(query_norm, docs_norm, doc_ids, None)
    ranking_w, scores_w, _, _ = _maxsim_arm(query_norm, docs_norm, doc_ids, weights)

    metrics_u = _ranked_metrics(ranking_u, relevant, cutoff)
    metrics_w = _ranked_metrics(ranking_w, relevant, cutoff)
    deltas = {m: round(metrics_w[m] - metrics_u[m], 6) for m in metrics_u}
    regressions = sorted(m for m, d in deltas.items() if d < 0.0)

    cost = {
        "cosine_products_per_arm": dot_products,
        "multiplications_per_arm": dot_products * len(query_vectors[0]),
        "arms": 2,
    }
    ranking_inputs_digest = _canonical_digest({
        "source": SOURCE_ID,
        "version": MODULE_VERSION,
        "query_tokens": [list(v) for v in query_vectors],
        "doc_ids": list(doc_ids),
        "doc_tokens": [[list(v) for v in dv] for dv in doc_vectors],
        "weights": list(weights) if weights is not None else None,
        "cutoff": cutoff,
    })
    qrels_digest = _canonical_digest({"qrels": sorted(relevant)})
    combined = hashlib.sha256(
        (ranking_inputs_digest + ":" + qrels_digest + ":" + MODULE_VERSION).encode("utf-8")
    ).hexdigest()

    return {
        "state": "EVALUATED_OFFLINE",
        "source": SOURCE_ID,
        "version": MODULE_VERSION,
        "model_capabilities": (
            "synthetic caller-supplied token vectors; NOT Qwen/ColBERT or any "
            "neural model weights; offline math only; no external evaluation claim"
        ),
        "arms": {
            "unweighted": {"state": "MEASURED", "ranking": ranking_u,
                            "metrics": metrics_u},
            "weighted": {"state": "MEASURED", "ranking": ranking_w,
                         "metrics": metrics_w},
        },
        "paired_deltas": deltas,
        "regressions": regressions,
        "cost": cost,
        "input_digests": {
            "ranking_inputs_sha256": ranking_inputs_digest,
            "qrels_sha256": qrels_digest,
            "combined_sha256": combined,
        },
        "notes": [
            "qrels are bound to the receipt and metrics only; they never influence ranking",
            "third post-filter evidence-admission comparison intentionally not implemented yet",
        ],
    }


def maxsim_score(query_vectors, doc_vectors, weights=None):
    """Research scoring helper: weighted MaxSim of one (query, document) pair.

    Returns ``{"state": "MEASURED", "score": s, "cost": {...}}`` or INVALID.
    Never called with qrels: a single scored pair is not a ranking claim.
    """
    single = _validate_inputs(query_vectors, [doc_vectors], ["d0"], weights, 1)
    if single is not None:
        return single
    query_norm = [_normalize(v) for v in query_vectors]
    doc_norm = [_normalize(v) for v in doc_vectors]
    score = 0.0
    wsum = 0.0
    for qi, qv in enumerate(query_norm):
        best = max(math.fsum(a * b for a, b in zip(qv, dtv)) for dtv in doc_norm)
        w = 1.0 if weights is None else weights[qi]
        score += w * best
        wsum += w
    return {
        "state": "MEASURED",
        "score": score / wsum,
        "cost": {"cosine_products": len(query_vectors) * len(doc_vectors)},
        "source": SOURCE_ID,
        "version": MODULE_VERSION,
    }