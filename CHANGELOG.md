# Changelog — szl-retrieval-bench

All entries reference commits visible on `main`. Dates are UTC.

## 0.2.0 — 2026-09-03

### Added
- `tfidf-dense` classical dense lane: L2-normalized TF-IDF vectors + cosine
  similarity (`dense.py`, `run_dense`). Real vector math, deterministic,
  honestly labeled classical — the neural adapter plugs into the same
  `rank(query, doc_ids)` interface later.
- Demo harness now runs three MEASURED lanes (bm25, tfidf-dense, hybrid_rrf)
  with a receipted three-lane comparison.
- Five new tests (14 total): dense ranking, L2-norm invariant, dense and
  hybrid MEASURED states, three-lane receipted comparison.

### Fixed
- `compare()` emitted receipts with a circular reference (the receipt embedded
  the result dict it was being attached to), crashing chain verification with
  `ValueError: Circular reference detected`. Latent since v0.1.0; caught by the
  new three-lane test before this push. Receipts now embed a snapshot copy.

## 0.1.0 — 2026-09-03

Initial payload (commit `f3f9fd6f`): BM25, RRF fusion, nDCG/Recall/MRR/MAP,
fairness gates, hash-chained receipts, 9-test suite, CI matrix 3.11/3.12,
canonical Apache-2.0 LICENSE.

---

Format: newest first. Every entry names its commit SHA where the commit
predates the changelog entry. Doctrine v11 · Apache-2.0 · SZL Holdings.
