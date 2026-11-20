# SPDX-License-Identifier: Apache-2.0
"""Bounded, offline MaxSim/importance ablation with optional declared-evidence veto.

Research only: synthetic caller-supplied vectors, not a neural retriever,
licensed embedding, signed receipt, real-time inference, or production gate.
The ranking never uses relevance judgments or declared provenance as features.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from .admission import AXES, AXIS_STATES, evaluate_admission_ablation
from .harness import evaluate_run

SCHEMA = "szl.retrieval-bench.late-interaction-ablation/v1"
MAX_DOCS = 128
MAX_QUERY_TOKENS = 64
MAX_DOC_TOKENS = 128
MAX_DIM = 128
MAX_QRELS = 4096
MAX_K = 128
MAX_SCALAR_PRODUCTS = 4_000_000
MAX_VALUE = 1e9
MAX_IMPORTANCE = 1e4
_MAX_SOURCE_ID = re.compile("^[A-Za-z0-9._/@:+-]{1,128}$")
_METRICS = ("R_precision", "mrr", "map")


class _BadInput(ValueError):
    pass


def _id(value, *, limit=128):
    if type(value) is not str or not 1 <= len(value) <= limit or not value.strip():
        raise _BadInput("INVALID_IDENTIFIER")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise _BadInput("INVALID_IDENTIFIER") from exc
    if any(ord(ch) < 32 or 0x7F <= ord(ch) <= 0x9F for ch in value):
        raise _BadInput("INVALID_IDENTIFIER")
    return value


def _number(value, *, maximum=MAX_VALUE, positive=False):
    if type(value) not in (int, float):
        raise _BadInput("INVALID_FINITE_NUMBER")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise _BadInput("INVALID_FINITE_NUMBER") from exc
    if not math.isfinite(number) or abs(number) > maximum or (positive and number <= 0):
        raise _BadInput("INVALID_FINITE_NUMBER")
    return number


def _sequence(value, *, minimum=1, maximum):
    if type(value) is not in (list, tuple) or not minimum <= len(value) <= maximum:
        raise _BadInput("INVALID_SEQUENCE_BOUND")
    return value


def _token(token, *, weighted, dimension=None):
    if type(token) is not dict or set(token) != (
        {"token_id", "vector", "importance"} if weighted else {"token_id", "vector"}
    ):
        raise _BadInput("INVALID_TOKEN_SHAPE")
    name = _id(token["token_id"], limit=64)
    vector = tuple(
        _number(value) for value in _sequence(token["vector"], maximum=MAX_DIM)
    )
    if dimension is not None and len(vector) != dimension:
        raise _BadInput("VECTOR_DIMENSION_MISMATCH")
    norm = math.hypot(*vector)
    if norm <= 0 or not math.isfinite(norm):
        raise _BadInput("ZERO_OR_INVALID_VECTOR")
    importance = (
        _number(token["importance"], maximum=MAX_IMPORTANCE, positive=True)
        if weighted
        else None
    )
    return {
        "token_id": name,
        "vector": vector,
        "unit": tuple(v / norm for v in vector),
        "importance": importance,
    }


def _tokens(raw, *, weighted, dim=None, maximum):
    tokens = []
    names = set()
    for value in _sequence(raw, maximum=maximum):
        token = _token(value, weighted=weighted, dimension=dim)
        if token["token_id"] in names:
            raise _BadInput("DUPLICATE_TOKEN_ID")
        names.add(token["token_id"])
        dim = len(token["vector"])
        tokens.append(token)
    return tokens, dim


def _grades(qrels):
    if type(qrels) is not dict or not 1 <= len(qrels) <= MAX_QRELS:
        raise _BadInput("INVALID_RELEVANCE_JUDGMENTS")
    result = {}
    for key, grade in qrels.items():
        name = _id(key)
        value = _number(grade, maximum=20)
        if value < 0:
            raise _BadInput("INVALID_RELEVANCE_JUDGMENTS")
        result[name] = value
    if not any(v > 0 for v in result.values()):
        raise _BadInput("NO_POSITIVE_RELEVANCE")
    return result


def _flags(flags, docs):
    if flags is None:
        return None
    if type(flags) is not dict or set(flags) != set(docs):
        raise _BadInput("EVIDENCE_SET_MISMATCH")
    result = {}
    for doc in sorted(docs):
        row = flags[doc]
        if type(row) is not dict or set(row) != set(AXES):
            raise _BadInput("EVIDENCE_AXES_INVALID")
        if any(
            type(row[axis]) is not str or row[axis] not in AXIS_STATES for axis in AXES
        ):
            raise _BadInput("EVIDENCE_STATES_INVALID")
        result[doc] = {axis: row[axis] for axis in AXES}
    return result


def _digest(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _snapshot(
    query_tokens,
    documents,
    qrels,
    evidence_flags,
    source_revision,
    k,
    max_scalar_products,
):
    if type(source_revision) is not str or not _MAX_SOURCE_ID.fullmatch(
        source_revision
    ):
        raise _BadInput("INVALID_SOURCE_REVISION")
    if type(k) is not int or not 1 <= k <= MAX_K:
        raise _BadInput("INVALID_CUTOFF")
    if (
        type(max_scalar_products) is not int
        or not 1 <= max_scalar_products <= MAX_SCALAR_PRODUCTS
    ):
        raise _BadInput("INVALID_WORK_BUDGET")
    query, dim = _tokens(query_tokens, weighted=True, maximum=MAX_QUERY_TOKENS)
    docs = {}
    for item in _sequence(documents, maximum=MAX_DOCS):
        if type(item) is not dict or set(item) != {"doc_id", "tokens"}:
            raise _BadInput("INVALID_DOCUMENT_SHAPE")
        name = _id(item["doc_id"])
        if name in docs:
            raise _BadInput("DUPLICATE_DOCUMENT_ID")
        tokens, _ = _tokens(
            item["tokens"], weighted=False, dim=dim, maximum=MAX_DOC_TOKENS
        )
        docs[name] = tokens
    labels = _grades(qrels)
    evidence = _flags(evidence_flags, docs)
    scalar_ops = dim * len(query) * sum(len(t) for t in docs.values())
    if scalar_ops > max_scalar_products:
        raise _BadInput("OPERATION_BUDGET_EXCEEDED")
    return query, docs, labels, evidence, dim, scalar_ops


def _evaluate_late_interaction_impl(
    query_tokens,
    documents,
    qrels,
    *,
    source_revision,
    k=10,
    max_scalar_products=MAX_SCALAR_PRODUCTS,
    evidence_flags=None,
):
    """Compare unweighted vs importance-weighted cosine MaxSim on identical candidates.

    All evidence states and relevance grades are caller-supplied, not verified.
    Scores and metrics are research observations only; no release authorization.
    Invalid data yields no partial scores or fabricated numerical measures.
    """
    try:
        query, docs, labels, evidence, dim, scalar_ops = _snapshot(
            query_tokens,
            documents,
            qrels,
            evidence_flags,
            source_revision,
            k,
            max_scalar_products,
        )
    except _BadInput as exc:
        state = (
            "BLOCKED"
            if str(exc) in ("NO_POSITIVE_RELEVANCE", "OPERATION_BUDGET_EXCEEDED")
            else "INVALID"
        )
        return {"state": state, "reason_code": str(exc), "training_ready": False}

    similarities = {}
    for name, doc_tokens in docs.items():
        max_sim = []
        for q in query:
            matches = [
                math.fsum(a * b for a, b in zip(q["unit"], d["unit"]))
                for d in doc_tokens
            ]
            max_sim.append(max(matches))
        similarities[name] = {
            "unweighted": math.fsum(max_sim),
            "weighted": math.fsum(q["importance"] * s for q, s in zip(query, max_sim)),
        }
    if any(
        not math.isfinite(value)
        for row in similarities.values()
        for value in row.values()
    ):
        return {
            "state": "INVALID",
            "reason_code": "NONFINITE_SCORE",
            "training_ready": False,
        }

    unweighted = sorted(docs, key=lambda doc: (-similarities[doc]["unweighted"], doc))
    weighted = sorted(docs, key=lambda doc: (-similarities[doc]["weighted"], doc))
    base = evaluate_run(unweighted, labels, k=k)
    candidate = evaluate_run(weighted, labels, k=k)
    if base.get("state") != "MEASURED" or candidate.get("state") != "MEASURED":
        return {
            "state": "INVALID",
            "reason_code": "METRICS_NOT_MEASURED",
            "training_ready": False,
        }

    metric_keys = (f"ndcg@{k}", f"recall@{k}", f"P@{k}", *_METRICS)
    optional = None
    if evidence is not None:
        optional = {
            "unweighted": evaluate_admission_ablation(
                unweighted, labels, evidence, k=k
            ),
            "weighted": evaluate_admission_ablation(weighted, labels, evidence, k=k),
        }
        if any(row.get("state") != "EVALUATED_OFFLINE" for row in optional.values()):
            return {
                "state": "INVALID",
                "reason_code": "EVIDENCE_ABLATION_INVALID",
                "training_ready": False,
            }
    inputs = {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "query": [
            {
                "token_id": t["token_id"],
                "vector": t["vector"],
                "importance": t["importance"],
            }
            for t in query
        ],
        "documents": [
            {
                "doc_id": doc,
                "tokens": [
                    {"token_id": token["token_id"], "vector": token["vector"]}
                    for token in docs[doc]
                ],
            }
            for doc in sorted(docs)
        ],
        "k": k,
        "max_scalar_products": max_scalar_products,
    }
    bindings = {
        "inputs_sha256": _digest(inputs),
        "qrels_sha256": _digest(sorted(labels.items())),
        "importance_sha256": _digest([t["importance"] for t in query]),
        "evidence_sha256": _digest(evidence) if evidence is not None else None,
    }
    return {
        "schema": SCHEMA,
        "state": "EVALUATED_OFFLINE",
        "source_revision": source_revision,
        "provenance": "CALLER_SUPPLIED_UNVERIFIED",
        "signed_receipt_verified": False,
        "training_ready": False,
        "production_admitted": False,
        "candidate_count": len(docs),
        "query_token_count": len(query),
        "embedding_dimension": dim,
        "work": {
            "pairwise_cosine": scalar_ops // dim,
            "cosine_dot_scalar_multiplications": scalar_ops,
            "query_weight_multiplications": len(query) * len(docs),
            "token_normalizations": len(query) + sum(len(t) for t in docs.values()),
            "pairs_shared_between_arms": True,
        },
        "ranking": {"unweighted": unweighted, "weighted": weighted},
        "scores": {name: similarities[name] for name in sorted(docs)},
        "metrics": {
            "unweighted": base,
            "weighted": candidate,
            "weighted_minus_unweighted": {
                metric: round(candidate[metric] - base[metric], 4)
                for metric in metric_keys
            },
        },
        "evidence_admission": optional,
        "bindings": bindings,
        "limits": (
            "Synthetic caller-provided token vectors and importance weights; no neural "
            "model, source authentication, license validation, confidence calibration, "
            "training authorization or production retrieval is implemented."
        ),
    }


def evaluate_late_interaction(
    query_tokens,
    documents,
    qrels,
    *,
    source_revision,
    k=10,
    max_scalar_products=MAX_SCALAR_PRODUCTS,
    evidence_flags=None,
):
    """Fail closed on arithmetic errors; never expose partial rankings or vectors.

    This is still an offline mathematical ablation, not execution authority.
    """
    try:
        return _evaluate_late_interaction_impl(
            query_tokens,
            documents,
            qrels,
            source_revision=source_revision,
            k=k,
            max_scalar_products=max_scalar_products,
            evidence_flags=evidence_flags,
        )
    except (ArithmeticError, ValueError):
        return {
            "state": "INVALID",
            "reason_code": "INVALID_NUMERIC_ARITHMETIC",
            "training_ready": False,
        }
