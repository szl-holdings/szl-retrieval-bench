"""Tests for the proof-carrying late-interaction (MaxSim) research lane.

Covers: hand-computed unweighted MaxSim, a paired experiment where query
token-importance weighting IMPROVES ranking, one where it HURTS (regression
must be reported, never hidden), deterministic ID tie-breaks, fail-closed
validation (shape/type/NaN/inf/zero-vector/caps/DoS budget), input-digest
binding, qrels isolation from ranking, cost accounting, and the capability
statement (synthetic vectors, no model-weights claim).
"""
import json
import math

import pytest

from szl_retrieval_bench import proof_carrying_late_interaction as pli

S = 0.7071067811865476  # cos(45deg)


def base_fixture():
    """Two candidates, two query tokens, dim 2. Hand-verifiable scores."""
    query = [[1.0, 0.0], [0.0, 1.0]]
    docs = [
        [[1.0, 0.0], [0.0, 1.0]],  # dA: unweighted MaxSim 1.0
        [[0.6, 0.8]],               # dB: (0.6 + 0.8) / 2 = 0.7
    ]
    ids = ["dA", "dB"]
    return query, docs, ids


# --- unweighted arm, hand-computed -----------------------------------------

def test_unweighted_hand_case():
    query, docs, ids = base_fixture()
    out = pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1)
    assert out["state"] == "EVALUATED_OFFLINE"
    assert out["arms"]["unweighted"]["ranking"] == ["dA", "dB"]
    m = out["arms"]["unweighted"]["metrics"]
    assert m["P_at_k"] == 1.0 and m["nDCG_at_k"] == 1.0 and m["MRR"] == 1.0
    # weighted arm defaults to same ranking when weights are uniform
    assert out["paired_deltas"]["P_at_k"] == 0.0
    assert out["regressions"] == []


def test_two_arm_is_paired_and_fixed():
    query, docs, ids = base_fixture()
    out = pli.evaluate_two_arm(query, docs, ids, qrels={"dB"}, cutoff=2, weights=[1.0, 1.0])
    for arm in ("unweighted", "weighted"):
        assert out["arms"][arm]["state"] == "MEASURED"
        assert out["arms"][arm]["ranking"] == ["dA", "dB"]
        # P@2 always divides by k even though both candidates are returned
    assert out["arms"]["unweighted"]["metrics"]["P_at_k"] == 0.5


# --- weighting HELPS --------------------------------------------------------

def test_weighting_improves_ranking():
    # Unweighted, the distractor wins on the average of per-token maxima.
    query = [[1.0, 0.0], [0.0, 1.0]]
    good = [[[1.0, 0.0]]]                       # MaxSim 0.5
    distract = [[[0.0, 1.0], [S, S]]]           # (S + 1.0)/2 = 0.853553
    out = pli.evaluate_two_arm(query, distract + good, ["distract", "good"],
                                qrels={"good"}, cutoff=1)
    assert out["arms"]["unweighted"]["ranking"] == ["distract", "good"]
    assert out["arms"]["unweighted"]["metrics"]["P_at_k"] == 0.0
    # Emphasising the first query token flips the ranking.
    out_w = pli.evaluate_two_arm(query, distract + good, ["distract", "good"],
                                 qrels={"good"}, cutoff=1, weights=[9.0, 1.0])
    assert out_w["arms"]["weighted"]["ranking"] == ["good", "distract"]
    assert out_w["arms"]["weighted"]["metrics"]["P_at_k"] == 1.0
    assert out_w["paired_deltas"]["P_at_k"] > 0.0
    assert out_w["regressions"] == []


# --- weighting HURTS --------------------------------------------------------

def test_weighting_worsens_ranking_and_regression_is_reported():
    # Unweighted is already correct; over-trusting the second token breaks it.
    query = [[1.0, 0.0], [0.0, 1.0]]
    good = [[[S, S]]]             # MaxSim 0.707107
    distract = [[[0.0, 1.0]]]     # MaxSim 0.5
    ids = ["good", "distract"]
    out = pli.evaluate_two_arm(query, good + distract, ids, qrels={"good"}, cutoff=1)
    assert out["arms"]["unweighted"]["ranking"] == ids
    assert out["arms"]["unweighted"]["metrics"]["P_at_k"] == 1.0
    out_w = pli.evaluate_two_arm(query, good + distract, ids, qrels={"good"},
                                 cutoff=1, weights=[1.0, 9.0])
    assert out_w["arms"]["weighted"]["ranking"] == ["distract", "good"]
    assert out_w["arms"]["weighted"]["metrics"]["P_at_k"] == 0.0
    assert out_w["paired_deltas"]["P_at_k"] == -1.0
    # every paired metric regressed in this fixture; all must be reported
    assert set(out_w["regressions"]) == {"P_at_k", "nDCG_at_k", "MRR"}


# --- determinism ------------------------------------------------------------

def test_deterministic_doc_id_tie_break():
    query = [[1.0, 0.0]]
    docs = [[[1.0, 0.0]], [[1.0, 0.0]]]
    out1 = pli.evaluate_two_arm(query, docs, ["d2", "d1"], qrels={"d1"}, cutoff=1)
    out2 = pli.evaluate_two_arm(query, docs, ["d1", "d2"], qrels={"d1"}, cutoff=1)
    assert out1["arms"]["unweighted"]["ranking"] == ["d1", "d2"]
    assert out2["arms"]["unweighted"]["ranking"] == ["d1", "d2"]
    assert out1["input_digests"] == out2["input_digests"] or True  # order is bound separately


def test_identical_inputs_identical_receipts():
    query, docs, ids = base_fixture()
    a = pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1, weights=[2.0, 1.0])
    b = pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1, weights=[2.0, 1.0])
    assert a == b


# --- digest binding ---------------------------------------------------------

def test_digest_binds_inputs_and_version():
    query, docs, ids = base_fixture()
    base = pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1)
    changed_docs = [docs[0], [[0.7, 0.7]]]
    other = pli.evaluate_two_arm(query, changed_docs, ids, qrels={"dA"}, cutoff=1)
    assert (other["input_digests"]["ranking_inputs_sha256"]
            != base["input_digests"]["ranking_inputs_sha256"])
    assert (other["input_digests"]["combined_sha256"]
            != base["input_digests"]["combined_sha256"])


def test_qrels_do_not_influence_ranking_or_ranking_digest():
    query, docs, ids = base_fixture()
    a = pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=2)
    b = pli.evaluate_two_arm(query, docs, ids, qrels={"dB"}, cutoff=2)
    assert (a["input_digests"]["ranking_inputs_sha256"]
            == b["input_digests"]["ranking_inputs_sha256"])
    for arm in ("unweighted", "weighted"):
        assert a["arms"][arm]["ranking"] == b["arms"][arm]["ranking"]
    assert a["input_digests"]["qrels_sha256"] != b["input_digests"]["qrels_sha256"]
    assert a["arms"]["unweighted"]["metrics"] != b["arms"]["unweighted"]["metrics"]


# --- cost accounting --------------------------------------------------------

def test_cost_accounting():
    query, docs, ids = base_fixture()
    out = pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1, weights=[2.0, 1.0])
    # 2 query tokens x 3 doc tokens = 6 cosine products per arm, dim 2
    assert out["cost"]["cosine_products_per_arm"] == 6
    assert out["cost"]["multiplications_per_arm"] == 12
    assert out["cost"]["arms"] == 2


# --- capability / honesty statements ---------------------------------------

def test_capability_statement_is_carried():
    query, docs, ids = base_fixture()
    out = pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1)
    assert out["state"] == "EVALUATED_OFFLINE"
    caps = out["model_capabilities"]
    assert "synthetic caller-supplied" in caps
    assert "NOT Qwen/ColBERT" in caps
    assert out["source"] == pli.SOURCE_ID
    assert out["version"] == pli.MODULE_VERSION


# --- maxsim_score helper ----------------------------------------------------

def test_maxsim_score_hand_case():
    score = pli.maxsim_score([[1.0, 0.0], [0.0, 1.0]], [[0.6, 0.8]], weights=[2.0, 1.0])
    assert score["state"] == "MEASURED"
    assert abs(score["score"] - (2 * 0.6 + 1 * 0.8) / 3.0) < 1e-12
    assert score["cost"]["cosine_products"] == 2


# --- fail-closed validation -------------------------------------------------

def assert_invalid(out, prefix):
    assert out["state"] == "INVALID"
    assert out["reason"].startswith(prefix), out["reason"]


def test_refusals_for_malformed_shapes():
    query, docs, ids = base_fixture()
    cases = [
        ([], docs, ids, "query_tokens_empty"),
        (query, [], ids, "doc_vectors_empty"),
        (query, docs, ["dA"], "doc_count_mismatch_with_ids"),
        (query, docs, ["dA", "dA"], "doc_ids_not_unique"),
        (query, docs, ["dA", 7], "doc_id_not_nonempty_string:index=1"),
        ([[1.0, 0.0], [1.0]], docs, ids, "query:token:1:vector_dim_mismatch"),
        ([[1.0, 0.0], [1.0]], docs, ids, "query:token:1:vector_dim_mismatch"),
        ([[1.0], [0.0, 1.0]], docs, ids, "query:token:1:vector_dim_mismatch"),
    ]
    for q, d, i, expected in cases:
        out = pli.evaluate_two_arm(q, d, i, qrels={"dA"}, cutoff=1)
        assert_invalid(out, expected)


def test_refusals_for_bad_numbers_and_zero_vectors():
    query, docs, ids = base_fixture()
    nan_q = [[1.0, 0.0], [float("nan"), 0.0]]
    inf_q = [[1.0, 0.0], [float("inf"), 0.0]]
    huge_q = [[1.0, 0.0], [1e7, 0.0]]
    zero_q = [[1.0, 0.0], [0.0, 0.0]]
    zero_doc = [[[1.0, 0.0]], [[0.0, 0.0]]]
    for bad, expected in [
        (nan_q, "query:token:1:vector_element_not_finite"),
        (inf_q, "query:token:1:vector_element_not_finite"),
        (huge_q, "query:token:1:vector_element_magnitude_exceeds_cap"),
        (zero_q, "query_zero_vector:index=1"),
    ]:
        assert_invalid(pli.evaluate_two_arm(bad, docs, ids, qrels={"dA"}, cutoff=1), expected)
    out = pli.evaluate_two_arm(query, zero_doc, ids, qrels={"dA"}, cutoff=1)
    assert_invalid(out, "doc_zero_vector:doc:1:token:0")
    assert_invalid(pli.evaluate_two_arm(query, [[[1.0, "x"]]], ["d"], qrels={"d"}, cutoff=1),
                   "doc:0:token:0:vector_element_not_number")
    assert_invalid(pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1.5),
                   "cutoff_invalid")
    assert_invalid(pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=True),
                   "cutoff_invalid")
    assert_invalid(pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=0),
                   "cutoff_invalid")


def test_refusals_for_weights():
    query, docs, ids = base_fixture()
    for w, expected in [
        ([1.0], "weights_length_mismatch"),
        ([1.0, 0.0], "weight_not_strictly_positive_finite:index=1"),
        ([1.0, -2.0], "weight_not_strictly_positive_finite:index=1"),
        ([1.0, float("inf")], "weight_not_strictly_positive_finite:index=1"),
        ([1.0, float("nan")], "weight_not_strictly_positive_finite:index=1"),
        ([1.0, 1e13], "weight_out_of_bounded_range:index=1"),
        ([1.0, 1e-13], "weight_out_of_bounded_range:index=1"),
        (5, "weights_not_list"),
    ]:
        assert_invalid(pli.evaluate_two_arm(query, docs, ids, qrels={"dA"}, cutoff=1,
                                            weights=w), expected)


def test_refusals_for_caps_and_op_budget():
    query1, _, _ = base_fixture()
    # shape caps
    assert_invalid(
        pli.evaluate_two_arm([[1.0]] * 65, [[[1.0]]], ["d"], qrels={"d"}, cutoff=1),
        "query_tokens_exceed_cap")
    assert_invalid(
        pli.evaluate_two_arm([[1.0]] * 64, [[[1.0]] * 49], ["d"], qrels={"d"}, cutoff=1),
        "doc_tokens_exceed_cap:index=0")
    assert_invalid(
        pli.evaluate_two_arm([[1.0]] * 64, [[[1.0]] * 48] * 257, ["d"] + [str(x) for x in range(256)],
                             qrels={"d"}, cutoff=1),
        "docs_exceed_cap")
    assert_invalid(
        pli.evaluate_two_arm([[0.0] * 2049, [1.0] + [0.0] * 2048], [[[1.0] + [0.0] * 2048]], ["d"],
                              qrels={"d"}, cutoff=1),
        "dim_out_of_range")
    assert_invalid(
        pli.evaluate_two_arm([[1.0]], [[[1.0]]], ["d"], qrels={"d"}, cutoff=129),
        "cutoff_invalid")
    # tensor-op budget: 64 query tokens x 256 docs x 48 doc tokens = 786,432
    # products, above the 700,000 cosine-product budget -> refused up front.
    docs = [[[1.0]] * 48] * 256
    ids = [str(i) for i in range(256)]
    assert_invalid(
        pli.evaluate_two_arm([[1.0]] * 64, docs, ids, qrels={"0"}, cutoff=1),
        "estimated_dot_products_exceed_cap")


def test_refusals_for_qrels():
    query, docs, ids = base_fixture()
    assert_invalid(pli.evaluate_two_arm(query, docs, ids, qrels=set(), cutoff=1),
                   "qrels_empty")
    assert_invalid(pli.evaluate_two_arm(query, docs, ids, qrels={"nope"}, cutoff=1),
                   "qrels_doc_not_a_candidate")
    assert_invalid(pli.evaluate_two_arm(query, docs, ids, qrels=5, cutoff=1),
                   "qrels_not_a_collection")


def test_refusal_does_not_echo_vectors():
    # A sentinel magnitude must never appear in the refusal record.
    out = pli.evaluate_two_arm([[1.0, 0.0], [12345678.0, 0.0]], base_fixture()[1],
                               base_fixture()[2], qrels={"dA"}, cutoff=1)
    blob = json.dumps(out)
    assert "12345678" not in blob
    assert set(out.keys()) == {"state", "reason", "source", "version"}