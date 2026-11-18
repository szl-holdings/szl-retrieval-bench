# SPDX-License-Identifier: Apache-2.0
"""Source-only evidence-admission ablation for fixed retrieval rankings.

This is an offline experiment, NOT a provenance verifier or runtime gate.
Flags are caller-declared fixture states, even when labeled PASS. It computes
ranking metrics over unchanged relevance judgments before and after refusing
candidates whose declared evidence axes are not all PASS.

The aim is to measure both gains AND lost recall, without selecting policies
on held-out labels or claiming statistical guarantees.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from .harness import evaluate_run
from .metrics import _relevance_grades, _unique_ranking

SCHEMA = "szl.retrieval-bench.admission-ablation/v1"
AXES = ("integrity", "license", "freshness", "derivation")
AXIS_STATES = frozenset({"PASS", "FAIL", "UNKNOWN"})
MAX_DOCUMENTS = 10_000
MAX_K = 1_000


def _invalid(reason: str) -> dict:
    return {"state": "INVALID", "reason": reason}


def _fingerprint(ranked: list[str], grades: dict, flags: dict) -> str:
    """Content checksum only; not a signature or proof of independent data."""
    payload = {
        "schema": SCHEMA,
        "ranking": ranked,
        "qrels": sorted(grades.items()),
        "flags": [[doc, [flags[doc][axis] for axis in AXES]] for doc in sorted(flags)],
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def evaluate_admission_ablation(
    ranked,
    qrels,
    evidence_flags,
    *,
    k: int = 10,
) -> dict:
    """Compare a fixed ranking to four-axis zero-veto filtering offline.

    "PASS" means only that the caller asserted an axis passed. Neither this
    function nor the digest verifies a signer, license, observation clock,
    derivation, evaluator, training job, or real-world source.
    """
    if type(k) is not int or not 1 <= k <= MAX_K:
        return _invalid("k must be an integer between 1 and 1000")
    try:
        ranked_docs = _unique_ranking(ranked)
    except (TypeError, ValueError):
        return _invalid("ranked must be an ordered unique document-ID list")
    if not ranked_docs:
        return {"state": "BLOCKED", "reason": "empty ranking cannot establish admission"}
    if len(ranked_docs) > MAX_DOCUMENTS or any(
        type(doc) is not str or not doc for doc in ranked_docs
    ):
        return _invalid("ranking must contain 1..10000 nonempty string document IDs")
    try:
        grades = _relevance_grades(qrels)
    except (TypeError, ValueError):
        return _invalid("qrels require finite nonnegative relevance grades")
    if any(type(doc) is not str or not doc for doc in grades):
        return _invalid("qrels document IDs must be nonempty strings")
    if not grades or not any(grade > 0 for grade in grades.values()):
        return {
            "state": "BLOCKED",
            "reason": "no positive relevance judgments; R-precision undefined",
        }
    if not isinstance(evidence_flags, Mapping):
        return _invalid("evidence_flags must map every ranked document to four axis states")
    try:
        flags = dict(evidence_flags)
        if set(flags) != set(ranked_docs):
            return _invalid("evidence flags must match exactly the ranked document IDs")
    except TypeError:
        return _invalid("evidence flags must match exactly the ranked document IDs")
    for observation in flags.values():
        if (
            not isinstance(observation, Mapping)
            or set(observation) != set(AXES)
            or any(type(observation[axis]) is not str
                   or observation[axis] not in AXIS_STATES for axis in AXES)
        ):
            return _invalid("each document needs four PASS, FAIL, or UNKNOWN axis states")

    # All decisions are from declared evidence flags, never from qrels.
    admitted = []
    veto_counts = {axis: 0 for axis in AXES}
    unknown_counts = {axis: 0 for axis in AXES}
    refused_relevant = 0
    for doc in ranked_docs:
        states = flags[doc]
        veto = [axis for axis in AXES if states[axis] != "PASS"]
        if not veto:
            admitted.append(doc)
            continue
        refused_relevant += int(grades.get(doc, 0) > 0)
        for axis in veto:
            veto_counts[axis] += 1
            unknown_counts[axis] += int(states[axis] == "UNKNOWN")

    original = evaluate_run(ranked_docs, grades, k=k)
    gated = evaluate_run(admitted, grades, k=k)
    if original["state"] != "MEASURED" or gated["state"] != "MEASURED":
        # Do not return a partially-measured comparison.
        return _invalid("ranking metrics could not be evaluated")
    metric_keys = (f"ndcg@{k}", f"recall@{k}", f"P@{k}", "R_precision", "mrr", "map")
    deltas = {name: round(gated[name] - original[name], 4) for name in metric_keys}
    if any(not math.isfinite(value) for value in deltas.values()):
        return _invalid("metric delta is not finite")

    return {
        "schema": SCHEMA,
        "state": "EVALUATED_OFFLINE",
        "observation_authority": "CALLER_DECLARED_UNVERIFIED",
        "execution_authority": "NONE",
        "signed_receipt_verified": False,
        "training_ready": False,
        "fingerprint_sha256": _fingerprint(ranked_docs, grades, flags),
        "cutoff": k,
        "denominators": {
            "ranked_candidates": len(ranked_docs),
            "admitted_candidates": len(admitted),
            "refused_candidates": len(ranked_docs) - len(admitted),
            "relevant_judgments": sum(grade > 0 for grade in grades.values()),
            "refused_relevant_candidates": refused_relevant,
        },
        "candidate_admission_fraction": round(len(admitted) / len(ranked_docs), 4),
        "vetoes_by_axis": veto_counts,
        "unknowns_by_axis": unknown_counts,
        "baseline": original,
        "gated": gated,
        "gated_minus_baseline": deltas,
        "limits": (
            "Offline caller-declared flags; PASS does not prove cryptographic "
            "verification, license rights, source freshness, or derivation integrity. "
            "Relevance labels are supplied for evaluation only; no statistical, "
            "production, paid-use, or training-readiness guarantee."
        ),
    }
