# Retrieval metric integrity — RAG-AUDIT-20261008

## Plan recorded before implementation

Source: `szl-holdings/szl-retrieval-bench` at `20bcdec06182a990ee34108ec1af28d3197c8766`.

Two local, synthetic counterexamples establish the repair scope:

- `evaluate_run(['d1', 'd1'], {'d1': 1}, k=2)` returns `MEASURED`, nDCG@2 = 1.6309 and MAP = 2.0. A ranked list containing the same document twice cannot be treated as two independent relevant hits.
- `compare` labels runs on different corpus texts comparable when the query IDs and cutoff match. Query text and relevance judgments are also unbound.

Implement a narrow repair in `harness.py`, `metrics.py`, and `fuse.py`: reject invalid duplicate rankings instead of inflating nDCG/AP or RRF; validate adapter output before fusion; capture each lane's exact ordered corpus, query text, and qrels identity before evaluation; carry immutable input fingerprints and a unique run ID into comparisons/receipts; reject absent/mismatched fingerprints and repeated run IDs. Preserve documented P@k/R-precision set-count behavior and the existing `fixture.py` synthetic-answer contract.

Success criteria: valid complete or partial unique rankings retain their existing metric definitions; duplicate rankings are not `MEASURED`; duplicate RRF input cannot add repeated votes from one lane; query/corpus/qrels changes prevent comparison; an adapter cannot mutate evaluated inputs after their fingerprint was taken; a proper three-lane comparison still emits a verifiable receipt. The receipt establishes integrity of recorded inputs/results, not evaluator authorship, production quality, or official benchmark performance.

Validation: reproduce the counterexamples before repair, run focused regression checks after repair, then run the existing offline suite and demo when dependencies are available. No network/provider call is permitted during benchmark execution. No fixture scorer, CI permission, publishing, or production configuration changes are in scope.

## Results

Implemented the planned repair without changing metric formulas for valid
unique rankings or the synthetic answer fixture scorer. Each measured lane
now has a fresh run ID and a `szl.retrieval-bench.inputs/v1` binding containing
the SHA-256 digests of its actual ordered corpus/query snapshots and qrels.
The hybrid lane validates document uniqueness and membership before fusion,
and copies the ID list supplied to the adapter. Comparisons reject absent or
mismatched bindings and repeated run IDs; their receipt snapshots include
the binding, cutoff, and each leaderboard run ID.

The original 26 regression cases produced **21 failures and 5 passes** on
the unmodified implementation, then **26 passes** after the repair. One
additional numeric-overflow regression was added after reviewing the repair.
The full offline test suite then passed **110 tests and 94 subtests**. This
includes all existing fixture/canonical receipt tests; `fixture.py` and its
manifest/canonicalization contract were not changed. The demo produced
three `MEASURED` lanes, a `MEASURED` comparison, and `chain_valid: true`.
`git diff --check` passed.

Tests used Python 3.12.14 and pytest 9.1.1 installed in an isolated scratch
dependency directory. Commands (with that directory prepended to
`PYTHONPATH` for pytest) were:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ranking_integrity.py -q
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
PYTHONPATH=src python -m szl_retrieval_bench.harness
git diff --check
```

All evidence is from local synthetic execution, without a provider or
production corpus. Input fingerprints bind the harness's recorded snapshot;
they cannot independently establish what an external adapter indexed, the
evaluator's identity, or official benchmark performance. Snapshotting also
does not claim protection against concurrently hostile Python objects or
code in the same interpreter. Historical results without the new binding
must be rerun for a comparable result. Publishing, CI gates, provider setup,
and operational deployment remain outside this local repair.
