# SPDX-License-Identifier: Apache-2.0
"""Offline adversarial tests for the Kimi-assisted, research-only MaxSim ablation."""

from copy import deepcopy

import pytest

from szl_retrieval_bench import late_interaction
from szl_retrieval_bench.admission import AXES

MAX_DIM = late_interaction.MAX_DIM
MAX_DOCS = late_interaction.MAX_DOCS
evaluate_late_interaction = late_interaction.evaluate_late_interaction

EX = (1.0, 0.0)
EY = (0.0, 1.0)


def query():
    return [
        {"token_id": "qa", "vector": list(EX), "importance": 1.0},
        {"token_id": "qb", "vector": list(EY), "importance": 10.0},
    ]


def docs():
    return [
        {"doc_id": "B", "tokens": [{"token_id": "b1", "vector": list(EY)}]},
        {"doc_id": "A", "tokens": [{"token_id": "a1", "vector": list(EX)}]},
    ]


def evaluate(qrels=None, **changes):
    args = {
        "query_tokens": query(),
        "documents": docs(),
        "qrels": {"B": 1} if qrels is None else qrels,
        "k": 1,
        "source_revision": "synthetic-public-v1",
    }
    args.update(changes)
    return evaluate_late_interaction(**args)


def flags():
    return {doc: {axis: "PASS" for axis in AXES} for doc in ("A", "B")}


def test_weighted_importance_improves_rank_on_a_public_synthetic_fixture():
    r = evaluate()
    assert r["state"] == "EVALUATED_OFFLINE"
    assert r["ranking"]["unweighted"] == ["A", "B"]
    assert r["ranking"]["weighted"] == ["B", "A"]
    assert r["scores"]["A"] == {"unweighted": 1.0, "weighted": 1.0}
    assert r["scores"]["B"] == {"unweighted": 1.0, "weighted": 10.0}
    assert r["metrics"]["unweighted"]["recall@1"] == 0.0
    assert r["metrics"]["weighted"]["recall@1"] == 1.0
    assert r["metrics"]["weighted_minus_unweighted"]["mrr"] == 0.5
    assert r["work"] == {
        "pairwise_cosine": 4,
        "cosine_dot_scalar_multiplications": 8,
        "query_weight_multiplications": 4,
        "token_normalizations": 4,
        "pairs_shared_between_arms": True,
    }
    assert r["signed_receipt_verified"] is False
    assert r["production_admitted"] is False
    assert r["training_ready"] is False


def test_weighted_importance_can_hurt_accuracy_without_hiding_regression():
    r = evaluate({"A": 1})
    assert r["metrics"]["unweighted"]["recall@1"] == 1.0
    assert r["metrics"]["weighted"]["recall@1"] == 0.0
    assert r["metrics"]["weighted_minus_unweighted"]["mrr"] == -0.5
    assert r["metrics"]["weighted_minus_unweighted"]["ndcg@1"] == -1.0


def test_relevance_labels_never_change_ranking_or_vector_score():
    a = evaluate({"A": 1})
    b = evaluate({"B": 1})
    assert a["ranking"] == b["ranking"]
    assert a["scores"] == b["scores"]
    assert a["bindings"]["inputs_sha256"] == b["bindings"]["inputs_sha256"]
    assert a["bindings"]["qrels_sha256"] != b["bindings"]["qrels_sha256"]


def test_shuffling_document_inputs_does_not_change_ranks_or_fingerprint():
    a = evaluate()
    b = evaluate(documents=list(reversed(docs())))
    assert a["ranking"] == b["ranking"]
    assert a["scores"] == b["scores"]
    assert a["bindings"] == b["bindings"]


def test_same_qrels_denominator_includes_unretrieved_positive():
    r = evaluate({"B": 1, "unretrieved": 1})
    assert r["metrics"]["weighted"]["recall@1"] == 0.5


def test_negative_cosine_is_not_clipped_into_bogus_positive_confidence():
    q = [{"token_id": "q", "vector": [2, 0], "importance": 2}]
    d = [{"doc_id": "negative", "tokens": [{"token_id": "d", "vector": [-5, 0]}]}]
    r = evaluate(query_tokens=q, documents=d, qrels={"negative": 1})
    assert r["state"] == "EVALUATED_OFFLINE"
    assert r["scores"]["negative"]["unweighted"] == -1.0
    assert r["scores"]["negative"]["weighted"] == -2.0


def test_work_budget_exact_boundary_and_one_less_fails_before_metrics():
    a = evaluate(max_scalar_products=8)
    b = evaluate(max_scalar_products=7)
    assert a["state"] == "EVALUATED_OFFLINE"
    assert b == {
        "state": "BLOCKED",
        "reason_code": "OPERATION_BUDGET_EXCEEDED",
        "training_ready": False,
    }


def test_evidence_veto_reuses_existing_adapter_and_refusal_can_hurt():
    ev = flags()
    ev["B"]["license"] = "UNKNOWN"
    r = evaluate(evidence_flags=ev)
    assert r["state"] == "EVALUATED_OFFLINE"
    gated = r["evidence_admission"]["weighted"]
    assert gated["observation_authority"] == "CALLER_DECLARED_UNVERIFIED"
    assert gated["denominators"]["refused_relevant_candidates"] == 1
    assert gated["gated_minus_baseline"]["recall@1"] == -1.0
    assert r["ranking"]["weighted"] == ["B", "A"]
    assert r["scores"]["B"]["weighted"] == 10.0


def test_evidence_pass_is_not_independent_verification():
    r = evaluate(evidence_flags=flags())
    assert r["evidence_admission"]["unweighted"]["signed_receipt_verified"] is False
    assert r["evidence_admission"]["weighted"]["training_ready"] is False
    assert len(r["bindings"]["evidence_sha256"]) == 64


def test_source_weight_vectors_and_evidence_affect_the_right_bindings():
    base = evaluate(evidence_flags=flags())
    q = query()
    q[1]["importance"] = 11.0
    w = evaluate(query_tokens=q, evidence_flags=flags())
    assert base["bindings"]["inputs_sha256"] != w["bindings"]["inputs_sha256"]
    assert base["bindings"]["importance_sha256"] != w["bindings"]["importance_sha256"]
    assert base["bindings"]["qrels_sha256"] == w["bindings"]["qrels_sha256"]
    ev = flags()
    ev["A"]["freshness"] = "FAIL"
    altered = evaluate(evidence_flags=ev)
    assert base["bindings"]["evidence_sha256"] != altered["bindings"]["evidence_sha256"]
    rev = evaluate(source_revision="synthetic-public-v2", evidence_flags=flags())
    assert base["bindings"]["inputs_sha256"] != rev["bindings"]["inputs_sha256"]


def test_input_arguments_are_not_mutated():
    inp = (query(), docs(), {"B": 1}, flags())
    before = deepcopy(inp)
    evaluate_late_interaction(
        inp[0],
        inp[1],
        inp[2],
        evidence_flags=inp[3],
        source_revision="synthetic-public-v1",
    )
    assert inp == before


@pytest.mark.parametrize(
    "broken",
    [
        {"query_tokens": []},
        {"documents": []},
        {"query_tokens": [{"token_id": "q", "vector": [0, 0], "importance": 2}]},
        {
            "query_tokens": [
                {"token_id": "q", "vector": [1, float("nan")], "importance": 2}
            ]
        },
        {
            "query_tokens": [
                {"token_id": "q", "vector": [1, float("inf")], "importance": 2}
            ]
        },
        {"query_tokens": [{"token_id": "q", "vector": [1, 0], "importance": 0}]},
        {"query_tokens": [{"token_id": "q", "vector": [1, 0], "importance": -1}]},
        {"query_tokens": [{"token_id": "q", "vector": [1, 0], "importance": True}]},
        {"query_tokens": [{"token_id": "q", "vector": [1, 0]}]},
        {
            "query_tokens": [
                {"token_id": "q", "vector": [1, 0], "importance": float("nan")}
            ]
        },
        {"query_tokens": [{"token_id": "q", "vector": ["1", 0], "importance": 1}]},
        {"query_tokens": [{"token_id": "q", "vector": [True, 0], "importance": 1}]},
        {
            "query_tokens": [
                {"token_id": "q", "vector": [1, 0], "importance": 1},
                {"token_id": "q", "vector": [0, 1], "importance": 1},
            ]
        },
        {
            "query_tokens": [
                {"token_id": "q", "vector": [1, 0], "importance": 1},
                {"token_id": "q2", "vector": [1], "importance": 1},
            ]
        },
        {"documents": docs() + [docs()[0]]},
        {
            "documents": [
                {"doc_id": "A", "tokens": [{"token_id": "a", "vector": [0, 1, 0]}]}
            ]
        },
        {"documents": [{"doc_id": "A", "tokens": []}]},
        {
            "documents": [
                {"doc_id": "\ud800", "tokens": [{"token_id": "a", "vector": [1, 0]}]}
            ]
        },
        {"qrels": {"B": -1}},
        {"qrels": {"B": True}},
        {"qrels": {"B": float("inf")}},
        {"k": 0},
        {"k": True},
        {"k": 129},
        {"source_revision": "../secrets\n"},
        {"max_scalar_products": 0},
        {"max_scalar_products": True},
        {"evidence_flags": {"A": flags()["A"]}},
        {"evidence_flags": {"A": flags()["A"], "B": {"integrity": "PASS"}},
    ],
)
def test_bad_input_fails_closed_without_partial_metrics_or_raw_vectors(broken):
    r = evaluate(**broken)
    assert r["state"] in {"BLOCKED", "INVALID"}
    assert "metrics" not in r and "scores" not in r and "ranking" not in r
    assert "bindings" not in r
    assert "raw" not in str(r).lower()


def test_max_dimension_and_max_document_bounds_are_enforced():
    too_many_dims = [{"token_id": "q", "vector": [1] * (MAX_DIM + 1), "importance": 1}]
    too_many_docs = [
        {"doc_id": f"doc{i}", "tokens": [{"token_id": "t", "vector": [1, 0]}]}
        for i in range(MAX_DOCS + 1)
    ]
    assert evaluate(query_tokens=too_many_dims)["state"] == "INVALID"
    assert evaluate(documents=too_many_docs)["state"] == "INVALID"


def test_no_positive_judgments_are_blocked_not_reported_as_zeros():
    r = evaluate(qrels={"A": 0, "B": 0})
    assert r["state"] == "BLOCKED"
    assert "metrics" not in r


def test_fault_injected_numeric_overflow_is_invalid_not_a_crash(monkeypatch):
    # The actual finite input bounds avoid ordinary overflow; inject the
    # numerical failure to prove this public API never escapes with partial data.
    def overflow(_values):
        raise OverflowError("synthetic numeric fault - do not echo values")

    monkeypatch.setattr(late_interaction.math, "fsum", overflow)
    result = evaluate()
    assert result == {
        "state": "INVALID",
        "reason_code": "INVALID_NUMERIC_ARITHMETIC",
        "training_ready": False,
    }
