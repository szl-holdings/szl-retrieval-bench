import json

import pytest

from szl_retrieval_bench.bm25 import BM25
from szl_retrieval_bench.dense import TfidfDense
from szl_retrieval_bench.fuse import rrf
from szl_retrieval_bench.harness import HARNESS, compare, main, run_bm25, run_dense, run_hybrid
from szl_retrieval_bench.metrics import (
    average_precision,
    mrr,
    ndcg,
    precision_at_k,
    r_precision,
    recall_at,
)
from szl_retrieval_bench.receipts import ReceiptChain

CORPUS = {"d1": "the cat sat on the mat", "d2": "dogs bark at night",
          "d3": "the cat chased the dog", "d4": "neural networks learn representations"}
QUERIES = {"q1": "cat mat", "q2": "neural representations"}
QRELS = {"q1": {"d1": 2, "d3": 1}, "q2": {"d4": 2}}


def test_bm25_ranks_relevant_first():
    bm = BM25(list(CORPUS.values()))
    assert bm.rank("cat mat", list(CORPUS))[0] == "d1"


def test_dense_ranks_relevant_first():
    d = TfidfDense(list(CORPUS.values()))
    assert d.rank("cat mat", list(CORPUS))[0] == "d1"


def test_dense_vectors_are_l2_normalized():
    d = TfidfDense(list(CORPUS.values()))
    for v in d.vectors:
        assert sum(x * x for x in v.values()) == pytest.approx(1.0)


def test_metrics_perfect_run():
    run = ["d1", "d3", "d2", "d4"]
    assert ndcg(run, QRELS["q1"]) == pytest.approx(1.0)
    assert recall_at(run, QRELS["q1"], 2) == pytest.approx(1.0)
    assert mrr(run, QRELS["q1"]) == pytest.approx(1.0)
    assert average_precision(run, QRELS["q1"]) == pytest.approx(1.0)


def test_metrics_empty_qrels_zero():
    assert ndcg(["d1"], {}) == 0.0
    assert recall_at(["d1"], {}, 10) == 0.0
    assert mrr(["d1"], {}) == 0.0


def test_precision_and_r_precision_reference_fixture():
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d3", "d9"}
    assert precision_at_k(ranked, relevant, 2) == {
        "state": "MEASURED",
        "P@2": 0.5,
    }
    assert r_precision(ranked, relevant) == {
        "state": "MEASURED",
        "R_precision": 0.6667,
        "R": 3,
    }


@pytest.mark.parametrize("invalid_k", [0, -1, True, 1.0, "1", None])
def test_precision_rejects_non_positive_or_non_integer_k(invalid_k):
    assert precision_at_k(["d1"], {"d1"}, invalid_k) == {
        "state": "INVALID",
        "detail": "k must be >= 1",
    }


def test_r_precision_rejects_empty_relevance_set():
    assert r_precision(["d1"], set()) == {
        "state": "INVALID",
        "detail": "no relevant docs - R-precision undefined",
    }


def test_precision_rejects_malformed_document_collections():
    assert precision_at_k(None, {"d1"}, 1)["state"] == "INVALID"
    assert precision_at_k(["d1"], "d1", 1)["state"] == "INVALID"
    assert r_precision([["unhashable"]], {"d1"})["state"] == "INVALID"


def test_duplicate_ranked_ids_do_not_inflate_hits():
    ranked = ["d1", "d1"]
    relevant = {"d1", "d2"}
    assert precision_at_k(ranked, relevant, 2)["P@2"] == 0.5
    assert r_precision(ranked, relevant)["R_precision"] == 0.5


def test_zero_overlap_query_is_retained_as_measured_zero():
    result = run_bm25(
        CORPUS,
        {"q0": "term absent from every document"},
        {"q0": {"outside-corpus": 1}},
        k=2,
    )
    assert result["state"] == "MEASURED"
    assert result["per_query"]["q0"]["state"] == "MEASURED"
    assert result["per_query"]["q0"]["P@2"] == 0.0
    assert result["per_query"]["q0"]["R_precision"] == 0.0
    assert result["aggregate"]["P@2"] == 0.0
    assert result["aggregate"]["R_precision"] == 0.0


def test_run_rejects_invalid_metric_edges():
    assert run_bm25(CORPUS, QUERIES, QRELS, k=0)["state"] == "INVALID"
    assert run_bm25(CORPUS, {"q0": "cat"}, {"q0": {}}, k=10)["state"] == "INVALID"


def test_rrf_prefers_consensus():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[0] in ("a", "b") and "d" in fused


def test_bm25_run_measured():
    r = run_bm25(CORPUS, QUERIES, QRELS)
    assert r["state"] == "MEASURED" and r["aggregate"]["ndcg@10"] == 1.0


def test_dense_run_measured():
    r = run_dense(CORPUS, QUERIES, QRELS)
    assert r["state"] == "MEASURED" and r["lane"] == "tfidf-dense"
    assert r["aggregate"]["ndcg@10"] == 1.0


def test_hybrid_blocked_without_dense():
    r = run_hybrid(CORPUS, QUERIES, QRELS)
    assert r["state"] == "BLOCKED" and "refusing" in r["reason"]


def test_hybrid_measured_with_dense():
    doc_ids = list(CORPUS)
    d = TfidfDense([CORPUS[x] for x in doc_ids])
    r = run_hybrid(CORPUS, QUERIES, QRELS, dense_rank_fn=lambda q, ids: d.rank(q, ids))
    assert r["state"] == "MEASURED" and r["lane"] == "hybrid_rrf"
    assert r["aggregate"]["ndcg@10"] == 1.0


def test_compare_needs_two_measured():
    only = run_bm25(CORPUS, QUERIES, QRELS)
    assert compare([only])["state"] == "BLOCKED"


def test_compare_rejects_mismatched_queries():
    a = run_bm25(CORPUS, QUERIES, QRELS)
    b = run_bm25(CORPUS, {"q1": "cat mat"}, {"q1": QRELS["q1"]})
    assert compare([a, b])["state"] == "INVALID"


def test_compare_three_lane_leaderboard():
    a = run_bm25(CORPUS, QUERIES, QRELS)
    b = run_dense(CORPUS, QUERIES, QRELS)
    doc_ids = list(CORPUS)
    d = TfidfDense([CORPUS[x] for x in doc_ids])
    c = run_hybrid(CORPUS, QUERIES, QRELS, dense_rank_fn=lambda q, ids: d.rank(q, ids))
    chain = ReceiptChain()
    v = compare([a, b, c], chain)
    assert v["state"] == "MEASURED" and len(v["leaderboard"]) == 3
    assert "receipt" in v and chain.verify()
    assert v["receipt"]["run"]["harness"] == HARNESS
    receipt_board = v["receipt"]["run"]["result"]["leaderboard"]
    assert all("P@10" in row and "R_precision" in row for row in receipt_board)
    v["receipt"]["run"]["harness"] = "other-harness"
    assert not chain.verify()


def test_compare_rejects_mismatched_metric_cutoffs():
    a = run_bm25(CORPUS, QUERIES, QRELS, k=10)
    b = run_dense(CORPUS, QUERIES, QRELS, k=2)
    assert compare([a, b]) == {
        "state": "INVALID",
        "reason": "metric cutoffs differ across runs; comparison is unfair",
    }


def test_demo_cli_prints_new_metrics_and_verifiable_receipt(capsys):
    main()
    output = json.loads(capsys.readouterr().out)
    assert output["chain_valid"] is True
    for lane in ("bm25", "tfidf_dense", "hybrid_rrf"):
        assert "P@10" in output[lane]["aggregate"]
        assert "R_precision" in output[lane]["aggregate"]
    assert output["comparison"]["receipt"]["run"]["harness"] == HARNESS
    receipt_board = output["comparison"]["receipt"]["run"]["result"]["leaderboard"]
    assert all("P@10" in row and "R_precision" in row for row in receipt_board)


def test_receipt_chain_verifies_and_detects_tamper():
    chain = ReceiptChain()
    chain.emit({"a": 1})
    chain.emit({"b": 2})
    assert chain.verify()
    chain.chain[0]["run"]["a"] = 999
    assert not chain.verify()
