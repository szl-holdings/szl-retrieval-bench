# szl-retrieval-bench

Honest retrieval benchmark lane for the SZL estate. Sparse BM25 baseline,
pluggable dense adapter, RRF hybrid fusion, and ranking metrics
(nDCG@k, Recall@k, MRR, MAP) with fairness gates and hash-chained receipts.

Doctrine: a benchmark that cannot run returns `BLOCKED` with a reason.
It never fabricates a metric. Run states: `MEASURED | BLOCKED | INVALID | FAILED`.

## Install and verify

```
pip install -e . pytest
python -m pytest tests/ -q
python -m szl_retrieval_bench.harness   # demo run: BM25 MEASURED, hybrid honestly BLOCKED
```

## Lanes

- `bm25` — stdlib BM25 (k1=1.5, b=0.75). No downloads, no network.
- `hybrid_rrf` — RRF(k=60) fusion of BM25 and a pluggable dense ranker.
  Calling it without a dense ranker returns `BLOCKED`, by design.
- `compare` — fairness gate: runs covering different query sets are `INVALID`.
- Receipts — every comparison can emit a SHA-256 hash-chained
  `UNSIGNED_HONEST` receipt (integrity + order, not identity).

Multi-vector / late-interaction (ColBERT-style) lives in a separate lane:
different memory profile, different fairness constraints. Not mixed here.

Apache-2.0 · Doctrine v11 · SZL Holdings
