import pytest

from szl_retrieval_bench.bm25 import BM25
from szl_retrieval_bench.dense import TfidfDense
from szl_retrieval_bench.fuse import rrf
from szl_retrieval_bench.harness import compare, run_bm25, run_dense, run_hybrid
from szl_retrieval_bench.metrics import average_precision, mrr, ndcg, recall_at
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


def test_receipt_chain_verifies_and_detects_tamper():
    chain = ReceiptChain()
    chain.emit({"a": 1})
    chain.emit({"b": 2})
    assert chain.verify()
    chain.chain[0]["run"]["a"] = 999
    assert not chain.verify()
