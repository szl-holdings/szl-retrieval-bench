# SZL extractive citation control — research-only source gate

**Owner:** `szl-holdings/szl-retrieval-bench`
**Initial protected source:** `b5b1ace35f84bdbfbccfeb4b4b7d157a9c69dd6e`
**Scope:** one optional offline Python entry point; no model call, training, persistent
database, dataset upload, HF release, Cloudflare rule, deployment, or approval mutation.

## Problem and independently falsifiable hypothesis

The model can supply plausible but nonexistent artifact IDs or move quotation
offsets while still producing readable citations. Text-generation quality gates
must distinguish **retrieval**, **citation resolution**, **semantic support**,
**upstream signer authenticity**, and **deployment authorization**.

**Hypothesis:** A *strict extractive-only* mode can reject citations with
invented IDs, missing evidence, changed content digests, invalid UTF-8 byte
offsets or claimed words not present in the exact cited source span.
The validator checks finite, caller-declared source snapshots, **not** external
truth. A malicious or factually wrong source that matches its digest still
passes the narrow byte check. This must never be labelled
"hallucinations eliminated" or generalized to free-form answers.

This is deliberately **not** a second edition of the existing
`szl_retrieval_bench.admission` four-axis declared-evidence ablation, the
`fixture_exact_match_v1` text accuracy scorer, or
`scientific-evidence-gate`'s external DOI metadata checks.
The output concerns one *proposed extractive citation*, not retrieval quality,
scientific validity, signer trust, source rights, or clinical truth.

## Exact data contract

Import `szl_retrieval_bench.extractive_gate.evaluate_packet` and pass a
closed-field, synthetic dictionary:

```json
{
  "schema": "szl.retrieval-bench.extractive-citation/v1",
  "sources": [{
    "source_id": "synthetic:doc1",
    "text": "Solar exposure: 8 units.",
    "sha256": "<64 lowercase hex SHA-256 of exact UTF-8 text bytes>"
  }],
  "claims": [{
    "claim_id": "synthetic-claim1",
    "text": "8 units",
    "citation": {"source_id": "synthetic:doc1", "start_byte": 16, "end_byte": 23}
  }]
}
```

The packet uses **UTF-8 byte offsets**; a Unicode character can occupy
multiple bytes. The example's actual digest must be calculated first; the
placeholder is illustrative and is *not* an admitted proof.
Only full exact substring matches can produce `EXTRACTIVE_BYTES_MATCH`.
This state explicitly returns
`source_identity_verified=false`,
`signed_receipt_verified=false`,
`semantic_entailment_verified=false`,
`production_authorized=false`,
`training_authorized=false`, and
`model_hallucination_eliminated=false`.

The parser (`parse_packet_bytes`) refuses duplicate JSON keys, nonfinite
values, wrong encoding and inputs above 96 KiB. At most 32 source snapshots,
48 claims, 8192 UTF-8 bytes per source, 2048 bytes per claim and 65536 total
source bytes are admitted. Identifiers must match a closed ASCII syntax;
all fields are closed; each claim gets one exact citation into a named,
hash-matched source. A source digest matches caller bytes, **not a signature**.

Missing evidence with no claim returns `ABSTAIN`, not a manufactured
accuracy of zero. If the model attempts a claim with no evidence, it is
`REFUSED`. Any malformed/unknown evidence, mismatched quote, or one bad claim
refuses the *whole* output, never silently admitting a subset.
The `claims_count` denominator is `null` unless *all* claims pass.

## Run and challenge the idea

From the source checkout with Python 3.11+ and pytest available:

```sh
PYTHONPATH=src python -m pytest -q tests/test_extractive_gate.py
PYTHONPATH=src python -m pytest -q tests
python -m ruff check src/szl_retrieval_bench/extractive_gate.py tests/test_extractive_gate.py
```

The 15+ regression checks use **synthetic text only** and include exact
positive span, fake source IDs, changed source content, fabricated number,
invalid/boolean and Unicode-byte offsets, empty evidence, evidence
present-vs-absent counterfactual, duplicate registry and claim IDs,
no partial admissions, extra authority fields, oversized input,
nonfinite and duplicate-key JSON and deterministic output.
These tests do not prove performance on a trained language model.

**Next externally authorized experiment:** pin an independent retriever
registry before invoking the model; collect same-query evidence-present and
evidence-absent outputs; freeze a development cohort and seal a
lineage-disjoint held-out evaluation; report unsupported-citation rate,
legitimate-answer refusal rate, abstention precision, citation recall and
full-denominator coverage, including signed source identity. Do not train
on or tune against the sealed set. Hold deployment until separate
owner-reviewed runtime wiring, provenance signer verification, negative
auth checks, canary/bake and rollback.

## Conceptual references; no third-party code copied

Google Cloud's official [Check grounding with RAG](https://docs.cloud.google.com/generative-ai-app-builder/docs/check-grounding)
distinguishes per-claim support and cited reference facts; this module is
**strictly more limited** than its semantic support checker and establishes
only extractive string provenance. The official Hugging Face
[RAG evaluation cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation)
motivates independent evaluation cohorts, negative/positive questions and
measured accuracy. Neither document endorses this specific gate.
