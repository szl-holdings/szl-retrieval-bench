# SPDX-License-Identifier: Apache-2.0
"""Offline adversarial tests for HIRAG-style evidence-admission ablations."""

from copy import deepcopy

import pytest

from szl_retrieval_bench.admission import AXES, evaluate_admission_ablation

PASS = {axis: "PASS" for axis in AXES}


def flags(*docs):
    return {doc: dict(PASS) for doc in docs}


def test_refusing_unrelevant_source_can_improve_precision_and_recall():
    ranked = ["junk", "hit1", "hit2", "extra"]
    evidence = flags(*ranked)
    evidence["junk"]["integrity"] = "FAIL"
    result = evaluate_admission_ablation(
        ranked, {"hit1": 1, "hit2": 1}, evidence, k=2,
    )
    assert result["state"] == "EVALUATED_OFFLINE"
    assert result["baseline"]["P@2"] == 0.5
    assert result["gated"]["P@2"] == 1.0
    assert result["gated_minus_baseline"]["P@2"] == 0.5
    assert result["gated_minus_baseline"]["recall@2"] == 0.5
    assert result["denominators"] == {
        "ranked_candidates": 4,
        "admitted_candidates": 3,
        "refused_candidates": 1,
        "relevant_judgments": 2,
        "refused_relevant_candidates": 0,
    }
    assert result["vetoes_by_axis"]["integrity"] == 1
    assert result["candidate_admission_fraction"] == 0.75
    assert result["signed_receipt_verified"] is False
    assert result["training_ready"] is False
    assert result["execution_authority"] == "NONE"


def test_refusing_relevant_but_unverified_source_hurts_recall():
    ranked = ["hit1", "hit2", "junk"]
    evidence = flags(*ranked)
    evidence["hit1"]["license"] = "UNKNOWN"
    result = evaluate_admission_ablation(
        ranked, {"hit1": 1, "hit2": 1}, evidence, k=2,
    )
    assert result["state"] == "EVALUATED_OFFLINE"
    assert result["baseline"]["recall@2"] == 1.0
    assert result["gated"]["recall@2"] == 0.5
    assert result["gated_minus_baseline"]["recall@2"] == -0.5
    assert result["denominators"]["refused_relevant_candidates"] == 1
    assert result["unknowns_by_axis"]["license"] == 1
    assert result["observation_authority"] == "CALLER_DECLARED_UNVERIFIED"


def test_relevance_denominator_never_shrinks_when_relevant_doc_is_absent():
    result = evaluate_admission_ablation(
        ["hit1", "junk"], {"hit1": 1, "not_in_results": 1},
        flags("hit1", "junk"), k=2,
    )
    assert result["denominators"]["relevant_judgments"] == 2
    assert result["gated"]["recall@2"] == 0.5
    assert result["baseline"]["recall@2"] == 0.5
    assert result["gated_minus_baseline"]["recall@2"] == 0.0


def test_all_vetoes_remain_offline_evaluation_and_never_admit_training():
    ranked = ["d1", "d2"]
    observations = flags(*ranked)
    for observation in observations.values():
        observation["freshness"] = "UNKNOWN"
    result = evaluate_admission_ablation(
        ranked, {"d1": 1}, observations, k=3,
    )
    assert result["state"] == "EVALUATED_OFFLINE"
    assert result["candidate_admission_fraction"] == 0.0
    assert result["gated"]["P@3"] == 0.0
    assert result["gated_minus_baseline"]["recall@3"] == -1.0
    assert result["training_ready"] is False
    assert result["signed_receipt_verified"] is False
    assert result["denominators"]["refused_candidates"] == 2


def test_multiple_vetoes_are_counted_per_axis_not_as_distinct_candidates():
    evidence = flags("d1", "d2")
    evidence["d1"].update({"integrity": "FAIL", "freshness": "UNKNOWN"})
    result = evaluate_admission_ablation(["d1", "d2"], {"d2": 1}, evidence, k=2)
    assert result["denominators"]["refused_candidates"] == 1
    assert result["vetoes_by_axis"]["integrity"] == 1
    assert result["vetoes_by_axis"]["freshness"] == 1
    assert result["unknowns_by_axis"]["freshness"] == 1


def test_input_checksum_is_stable_under_mapping_order_but_changes_with_flags():
    ranked = ["d2", "d1"]
    first = flags("d1", "d2")
    second = {"d2": dict(PASS), "d1": dict(PASS)}
    a = evaluate_admission_ablation(ranked, {"d1": 1, "d2": 2}, first, k=2)
    b = evaluate_admission_ablation(ranked, {"d2": 2, "d1": 1}, second, k=2)
    assert a["fingerprint_sha256"] == b["fingerprint_sha256"]
    second["d1"]["derivation"] = "UNKNOWN"
    c = evaluate_admission_ablation(ranked, {"d2": 2, "d1": 1}, second, k=2)
    assert c["fingerprint_sha256"] != a["fingerprint_sha256"]
    assert c["signed_receipt_verified"] is False
    assert len(a["fingerprint_sha256"]) == 64


@pytest.mark.parametrize("ranked", [
    ["d1", "d1"], ["", "d2"], ["d1", 3], "d1", {"d1": 1},
])
def test_invalid_rankings_fail_closed(ranked):
    result = evaluate_admission_ablation(
        ranked, {"d1": 1}, flags("d1"), k=2,
    )
    assert result["state"] == "INVALID"
    assert "fingerprint_sha256" not in result


@pytest.mark.parametrize("cutoff", [0, -1, True, 1.0, "10", 1001])
def test_invalid_cutoff_fails_closed(cutoff):
    result = evaluate_admission_ablation(["d1"], {"d1": 1}, flags("d1"), k=cutoff)
    assert result["state"] == "INVALID"


@pytest.mark.parametrize("qrels", [
    {"d1": True}, {"d1": -1}, {"d1": float("nan")}, {"d1": float("inf")},
    {"": 1}, {"d1": "1"}, ["d1"],
])
def test_invalid_judgments_fail_closed(qrels):
    result = evaluate_admission_ablation(["d1"], qrels, flags("d1"))
    assert result["state"] == "INVALID"


@pytest.mark.parametrize("qrels", [{}, {"d1": 0}])
def test_no_relevant_judgments_block_instead_of_fabricating_quality(qrels):
    result = evaluate_admission_ablation(["d1"], qrels, flags("d1"))
    assert result["state"] == "BLOCKED"
    assert "baseline" not in result


@pytest.mark.parametrize("observation", [
    None, [], {"d1": dict(PASS), "extra": dict(PASS)}, {},
    {"d1": {"integrity": "PASS"}},
    {"d1": {**PASS, "freshness": None}},
    {"d1": {**PASS, "license": "VALID"}},
    {"d1": {**PASS, "derivation": True}},
    {"d1": {**PASS, "extra": "PASS"}},
])
def test_unbound_or_malformed_observations_fail_closed(observation):
    result = evaluate_admission_ablation(["d1"], {"d1": 1}, observation)
    assert result["state"] == "INVALID"
    assert "gated_minus_baseline" not in result


def test_over_bound_ranked_documents_fail_before_any_provenance_claim():
    ranked = [f"d{i}" for i in range(10_001)]
    result = evaluate_admission_ablation(ranked, {"d1": 1}, {})
    assert result["state"] == "INVALID"
    assert "fingerprint_sha256" not in result


def test_no_candidates_blocks_and_no_vacuous_success():
    result = evaluate_admission_ablation([], {"d1": 1}, {})
    assert result["state"] == "BLOCKED"


def test_evidence_flags_are_not_inferred_from_relevance_labels():
    ranked = ["d1", "d2"]
    evidence = flags(*ranked)
    evidence["d1"]["integrity"] = "UNKNOWN"
    one = evaluate_admission_ablation(ranked, {"d1": 1}, evidence)
    two = evaluate_admission_ablation(ranked, {"d2": 1}, evidence)
    assert one["denominators"]["admitted_candidates"] == 1
    assert two["denominators"]["admitted_candidates"] == 1
    assert one["denominators"]["refused_relevant_candidates"] == 1
    assert two["denominators"]["refused_relevant_candidates"] == 0
    assert one["fingerprint_sha256"] != two["fingerprint_sha256"]


def test_caller_inputs_are_not_modified():
    ranked = ["d1", "d2"]
    qrels = {"d1": 1}
    evidence = flags(*ranked)
    evidence["d2"]["derivation"] = "FAIL"
    inputs = deepcopy((ranked, qrels, evidence))
    evaluate_admission_ablation(ranked, qrels, evidence)
    assert (ranked, qrels, evidence) == inputs
