# szl-retrieval-bench

Honest retrieval benchmark lane for the SZL estate. Sparse BM25 baseline,
classical TF-IDF dense lane, RRF hybrid fusion, and ranking metrics
(nDCG@k, Recall@k, MRR, MAP) with fairness gates and hash-chained receipts.

Doctrine: a benchmark that cannot run returns `BLOCKED` with a reason.
It never fabricates a metric. Run states: `MEASURED | BLOCKED | INVALID | FAILED`.

## Install and verify

```
pip install -e . pytest
python -m pytest tests/ -q                  # 14 tests
python -m szl_retrieval_bench.harness       # demo: bm25 + tfidf-dense + hybrid all MEASURED
```

## Lanes

- `bm25` — stdlib BM25 (k1=1.5, b=0.75). No downloads, no network.
- `tfidf-dense` — classical dense retrieval: L2-normalized TF-IDF vectors,
  cosine similarity. Real dense-vector math, deterministic, honestly labeled
  classical (not a neural embedding). A neural adapter plugs into the same
  `rank(query, doc_ids)` interface without touching the harness.
- `hybrid_rrf` — RRF(k=60) fusion of BM25 and a pluggable dense ranker.
  Calling it without a dense ranker returns `BLOCKED`, by design.
- `compare` — fairness gate: runs covering different query sets are `INVALID`.
- Receipts — every comparison can emit a SHA-256 hash-chained
  `UNSIGNED_HONEST` receipt (integrity + order, not identity).

Multi-vector / late-interaction (ColBERT-style) lives in a separate lane:
different memory profile, different fairness constraints. Not mixed here.

## Changelog highlights

- v0.2.0: dense lane added; fixed a latent circular-reference bug in
  receipted comparisons (found by the new three-lane test before push —
  receipts now embed a snapshot, never a self-reference).
- v0.1.0: BM25, RRF, metrics, fairness gates, receipts, CI.

Apache-2.0 · Doctrine v11 · SZL Holdings
