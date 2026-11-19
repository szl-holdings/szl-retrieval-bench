# SPDX-License-Identifier: Apache-2.0
"""Paired offline fidelity checks against exact cosine on identical vectors.

This does NOT measure answer correctness or an upstream AtlasKV model. It measures
whether a pruned hierarchical selector retains exact-cosine top-k neighbors.
"""

from __future__ import annotations

from collections.abc import Mapping
from .hierarchical import HierarchicalIndex, IndexRefused, _sha_json, _unit

SCHEMA = "szl.retrieval-bench.hierarchical-fidelity/v1"
MAX_QUERIES = 1000


def compare_exact(index: HierarchicalIndex, queries: Mapping[str, tuple[float, ...]],
                  *, tenant_id: str, top_k: int = 10,
                  max_leaf_visits: int = 1) -> dict:
    if not isinstance(index, HierarchicalIndex):
        raise IndexRefused("validated HierarchicalIndex required")
    if not isinstance(queries, Mapping) or not 1 <= len(queries) <= MAX_QUERIES:
        raise IndexRefused("queries must be a nonempty bounded mapping")
    snapshot: list[tuple[str, tuple[float, ...]]] = []
    for qid, qvec in queries.items():
        if not isinstance(qid, str) or not qid:
            raise IndexRefused("unique nonempty query IDs required")
        snapshot.append((qid, _unit(qvec, index.dimensions)))
    results = []
    total_oracle = 0
    retained = 0
    top1_hits = 0
    total_scored = 0
    total_exact = 0
    for qid, vector in snapshot:
        approximate = index.select(vector, tenant_id=tenant_id, top_k=top_k,
                                   max_leaf_visits=max_leaf_visits)
        exact = index.select(vector, tenant_id=tenant_id, top_k=top_k,
                             max_leaf_visits=index.leaf_count)
        oracle_ids = [item["fact_id"] for item in exact["ranking"]]
        candidate_ids = [item["fact_id"] for item in approximate["ranking"]]
        overlap = len(set(candidate_ids) & set(oracle_ids))
        total_oracle += len(oracle_ids)
        retained += overlap
        top1_hits += int(bool(oracle_ids and candidate_ids and
                              oracle_ids[0] == candidate_ids[0]))
        total_scored += approximate["scored_facts"]
        total_exact += exact["scored_facts"]
        results.append({"query_id": qid, "exact_neighbor_count": len(oracle_ids),
                        "retained_neighbors": overlap,
                        "top1_equal": bool(oracle_ids and candidate_ids and
                                           oracle_ids[0] == candidate_ids[0]),
                        "approximate_scored_facts": approximate["scored_facts"],
                        "exact_scored_facts": exact["scored_facts"]})
    return {
        "schema": SCHEMA,
        "state": "MEASURED_OFFLINE_PROXY",
        "model_executed": False,
        "evaluation_of_answer_quality": False,
        "observation_authority": "CALLER_DECLARED_UNVERIFIED",
        "index_sha256": index.snapshot_sha256,
        "queries_sha256": _sha_json(snapshot),
        "tenant_id": tenant_id,
        "denominators": {"queries": len(snapshot), "exact_topk_neighbors": total_oracle,
                         "approximate_scored_vectors": total_scored,
                         "exact_scored_vectors": total_exact},
        "top1_agreement_fraction": round(top1_hits / len(snapshot), 4),
        "exact_neighbor_retention_fraction": (round(retained / total_oracle, 4)
                                              if total_oracle else None),
        "per_query": results,
        "limits": "Offline fidelity to exact cosine on supplied vectors; no held-out answer labels, model execution, validated rights or training qualification.",
    }
