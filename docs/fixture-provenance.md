# PUSH XXIX synthetic fixture integration

## 1. Source ownership

Target: `szl-holdings/szl-retrieval-bench`; CODEOWNERS assigns
`@stephenlutar2-hash`. Queue owner retains merge authority. Memory component
integration is outside this change. No canonical repository move is asserted.

## 2. Baseline and duplication check

Based on `b42ab67` on main, inspected 2026-09-30. No open PRs were present.
Existing ranking harness and ReceiptChain already exist; this adds a separate
answer-fixture scorer, without replacing either API or changing CI workflows.

## 3. Source identity

Library inputs: report `libfile_c395cff98ba4819191708a5b6e5b4898`, patch
`libfile_90b0998f75c08191a8e8cb1d7e426b84`, hash manifest
`libfile_17d46230cb008191a8aca65019fd0ea9`, evidence archive
`libfile_400fb72a7cec81918b879c70f3bec67a`. Consumer-local downloads retained
Library identity/version metadata. Report and patch bytes matched manifest
hashes; all 48 benchmark/guard archive entries matched size and SHA-256.

Corrected scorer source: 19,898 bytes, SHA-256
`67fc0be0ebb6b39b7586240e70b556d582d2893abf069fb7750e4183c67fb713`.
Original reviewed derivative: 5,443 bytes, SHA-256
`596caaf745340d39ee2a111cc401de21aeba6d10d3298148602d251f9e9c70d2`.
Acceptance suite source: 23,710 bytes, SHA-256
`ec5eb39318959864b52c0267cac4594b059d6523bd49e18c57a3aa7be3eed239`.
These are normalized review derivatives, not asserted original payload bytes.
The original provider code was inspected without execution. The orchestrator was
not executed.

## 4. Adaptation

The corrected scorer is packaged as `szl_retrieval_bench.fixture`. Integration
adds bounded receipt/artifact reads, pre-parse input hashes, finite numeric
overflow and deep-JSON rejection, ancestor link/reparse checks, and a verified
`external_performance: UNMEASURED` field. Tests use package imports and private
temporary directories. The legacy assertion requiring substring credit is
intentionally not carried into the passing suite; strict equality is the contract.

## 5. Validation coverage

Python 3.12.14, synthetic data only: 45 fixture tests pass. Full repository
pytest run: 73 tests and 94 subtests pass. Coverage includes strict normalization,
unique and malformed IDs, missing/empty predictions, invalid input/settings,
finite values, deterministic metrics, hashing before parsing, input/source
mutation, receipt tampering, bounded reads, path checks, concurrent unique runs,
collision refusal, and incomplete writes. Existing 28 ranking tests pass.
All 9 guard self-tests pass again after the Windows pytest adaptations.

Child execution used a credential allowlist, 60-second timeout, disabled pytest
plugin autoload, blocked network/provider/process transport, restricted reads,
and evidence-only writes. These are reviewed-code Python guards, not an OS
security boundary. Earlier local pytest setup attempts stopped on guarded config
discovery, Windows terminal library loading, and null-device access; no benchmark
failure was hidden. Small test dependencies were installed only in the workspace.
An independent reviewer inspected both original/corrected code and the integrated
diff and found no blocker to scoped synthetic testing.

## 6. Known limits

All external, vendor, deployed-system, retrieval/generation and official benchmark
performance is UNMEASURED. No real datasets, training/evaluation data, model runs,
provider calls, keys, paid judges or official benchmark adapters were used.
Unsupported modes error explicitly. Caller metadata/claims remain unverified and
noncomparable. Receipts are unsigned and cannot authenticate authorship or prevent
coordinated rewriting. A trusted evaluator digest verifies source identity only.
Unicode normalization version must match for verification. Use private local
directories; concurrent malicious path replacement is outside the portable checks.

## 7. Delivery boundary

Draft PR only, normal push, remote SHA and hosted CI to be checked after push.
No merge, auto-merge, deployment, force-push, repository move/archive, PR closure,
security-setting change, SQL migration, or memory-service change is included.
