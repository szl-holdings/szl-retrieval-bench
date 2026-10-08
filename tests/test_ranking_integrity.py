"""Synthetic regression cases for metric bounds and fair comparisons."""

from copy import deepcopy

import pytest

from szl_retrieval_bench.fuse import rrf
from szl_retrieval_bench.harness import (
    compare,
    evaluate_run,
    run_bm25,
    run_dense,
    run_hybrid,
)
from szl_retrieval_bench.metrics import average_precision, ndcg
from szl_retrieval_bench.receipts import ReceiptChain


def inputs():
    return (
        {"d1": "cat mat", "d2": "dog field"},
        {"q1": "cat mat"},
        {"q1": {"d1": 1}},
    )


def test_duplicate_ranking_cannot_be_measured_above_one():
    result = evaluate_run(["d1", "d1"], {"d1": 1}, k=2)
    assert result["state"] == "INVALID"
    assert "duplicate" in result["detail"]


@pytest.mark.parametrize("metric", [ndcg, average_precision])
def test_direct_gain_metrics_reject_duplicate_documents(metric):
    with pytest.raises(ValueError, match="duplicate"):
        metric(["d1", "d1"], {"d1": 1})


def test_rrf_rejects_repeated_vote_from_one_lane():
    with pytest.raises(ValueError, match="duplicate"):
        rrf([["a", "a", "a"], ["b", "a"]])
    assert rrf([["a", "b"], ["b", "a"], ["b"]])[0] == "b"


@pytest.mark.parametrize("ranking", [["d1", "d1"], ["unknown"], "d1", None])
def test_hybrid_rejects_invalid_adapter_output_before_fusion(ranking):
    result = run_hybrid(*inputs(), dense_rank_fn=lambda query, ids: ranking)
    assert result["state"] == "INVALID"
    assert "dense ranking" in result["reason"]


@pytest.mark.parametrize("ranking", [[], ["d1"], ["d2", "d1"]])
def test_partial_unique_adapter_output_remains_measured(ranking):
    result = run_hybrid(*inputs(), dense_rank_fn=lambda query, ids: ranking)
    assert result["state"] == "MEASURED"
    assert all(0 <= value <= 1 for value in result["aggregate"].values())


def test_hybrid_empty_query_set_is_blocked():
    corpus, _, qrels = inputs()
    result = run_hybrid(corpus, {}, qrels, dense_rank_fn=lambda query, ids: ids)
    assert result["state"] == "BLOCKED"


@pytest.mark.parametrize("changed_component", ["corpus", "queries", "qrels", "corpus_order"])
def test_same_query_ids_cannot_hide_different_evaluation_inputs(changed_component):
    corpus, queries, qrels = inputs()
    baseline = run_bm25(corpus, queries, qrels)
    if changed_component == "corpus":
        corpus["d1"] = "changed corpus text"
    elif changed_component == "queries":
        queries["q1"] = "changed query text"
    elif changed_component == "qrels":
        qrels["q1"] = {"d2": 1}
    else:
        corpus = dict(reversed(list(corpus.items())))
    candidate = run_dense(corpus, queries, qrels)
    result = compare([baseline, candidate])
    assert result["state"] == "INVALID"
    assert "input fingerprints differ" in result["reason"]


def test_compare_rejects_replayed_run_id():
    run = run_bm25(*inputs())
    result = compare([run, deepcopy(run)])
    assert result["state"] == "INVALID"
    assert "duplicate run IDs" in result["reason"]


def test_compare_rejects_missing_input_binding():
    first = run_bm25(*inputs())
    second = run_dense(*inputs())
    second.pop("input_fingerprints", None)
    result = compare([first, second])
    assert result["state"] == "INVALID"
    assert "input fingerprints" in result["reason"]


def test_compare_rejects_malformed_input_binding():
    first = run_bm25(*inputs())
    second = run_dense(*inputs())
    # A shared malformed declaration is not evidence of common inputs.
    first["input_fingerprints"] = second["input_fingerprints"] = {"corpus_sha256": "same"}
    assert compare([first, second])["state"] == "INVALID"


def test_input_fingerprints_and_run_ids_are_bound_into_receipt():
    first = run_bm25(*inputs())
    second = run_dense(*inputs())
    chain = ReceiptChain()
    result = compare([first, second], chain)
    assert result["state"] == "MEASURED"
    assert first["run_id"] != second["run_id"]
    assert result["input_fingerprints"] == first["input_fingerprints"]
    recorded = result["receipt"]["run"]["result"]
    assert recorded["input_fingerprints"] == result["input_fingerprints"]
    assert {row["run_id"] for row in recorded["leaderboard"]} == {
        first["run_id"], second["run_id"]
    }
    assert chain.verify()
    # Mutating returned views does not rewrite the already-emitted receipt.
    first["input_fingerprints"]["corpus_sha256"] = "0" * 64
    result["input_fingerprints"]["qrels_sha256"] = "0" * 64
    result["leaderboard"][0]["map"] = 0.0
    assert chain.verify()
    recorded["input_fingerprints"]["qrels_sha256"] = "0" * 64
    assert not chain.verify()


def test_adapter_cannot_mutate_inputs_after_their_fingerprint_is_taken():
    corpus, queries, qrels = inputs()
    queries["q2"] = "dog field"
    qrels["q2"] = {"d2": 1}
    baseline = run_bm25(corpus, queries, qrels)
    seen = []

    def mutating_adapter(query, ids):
        seen.append((query, tuple(ids)))
        corpus["d1"] = "rewritten text"
        queries["q2"] = "rewritten query"
        qrels["q2"]["d2"] = 0
        ids.clear()
        return ["d1", "d2"]

    candidate = run_hybrid(corpus, queries, qrels, dense_rank_fn=mutating_adapter)
    assert candidate["state"] == "MEASURED"
    assert seen == [("cat mat", ("d1", "d2")), ("dog field", ("d1", "d2"))]
    assert candidate["input_fingerprints"] == baseline["input_fingerprints"]
    assert compare([baseline, candidate])["state"] == "MEASURED"


@pytest.mark.parametrize("grade", [float("nan"), float("inf"), -1, "1", True, 1024])
def test_invalid_relevance_grades_do_not_produce_measured_metrics(grade):
    corpus, queries, _ = inputs()
    result = run_bm25(corpus, queries, {"q1": {"d1": grade}})
    assert result["state"] == "INVALID"
