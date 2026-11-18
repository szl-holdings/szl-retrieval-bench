# Retrieval evidence-admission ablation — source-only experiment

**Status:** offline evaluation adapter, not a deployed retrieval, cryptographic
verification, model-training, or publication claim. Source of truth:
`szl-holdings/szl-retrieval-bench`. This complements the explicitly
**unimplemented** HIRAG specification in
[`szl-second-brain/docs/HIRAG.md`](https://github.com/szl-holdings/szl-second-brain/blob/main/docs/HIRAG.md).

## Field study: published methods and public implementations

| System | Public source / paper | What SZL uses | Why that is not enough |
| --- | --- | --- | --- |
| Microsoft GraphRAG | [source](https://github.com/microsoft/graphrag), [Edge et al. 2024](https://arxiv.org/abs/2404.16130) | Entity/community indexing and local/global question modes | Community summary is not authenticated source evidence; model indexing costs must be reported. |
| HippoRAG / HippoRAG 2 | [source](https://github.com/OSU-NLP-Group/HippoRAG), [original paper](https://arxiv.org/abs/2405.14831) | Query-seeded Personalized PageRank (already present in A11oy Brain) | Graph centrality does not establish source rights, freshness or truth. |
| RAPTOR | [source](https://github.com/latentsp/raptor-rag), [Sarthi et al. 2024](https://arxiv.org/abs/2401.18059) | Tree-organized coarse-to-fine retrieval (candidate layer) | Derived summaries need provenance to their source chunks. |
| LightRAG | [source](https://github.com/HKUDS/LightRAG), [Guo et al. EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.568/) | Low/high-level graph retrieval, incremental update ideas | New graph edges do not automatically imply evidence admissibility. |
| GraphRAG-Bench | [official benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark), [paper](https://arxiv.org/abs/2506.05690) | Multi-task evaluation and honest flat-RAG baselines | Graph methods sometimes underperform flat retrieval; improvements must be measured on identical inputs. |
| Palantir Ontology | [official architecture](https://www.palantir.com/docs/foundry/ontology/overview) | Typed objects, links and governed action boundaries | The internal ontology backend is not public GitHub code; a visualization is not an operational ontology. |
| Anduril Lattice | [developer API](https://developer.anduril.com/guides/concepts/overview), [Python SDK](https://github.com/anduril/lattice-sdk-python) | Typed entities, tasks and objects as interoperable data contracts | SDK availability does not imply public Lattice engine code or grant effect authority. |
| NVIDIA Omniverse | [official library organization](https://github.com/NVIDIA-Omniverse), [docs](https://docs.nvidia.com/omniverse/) | OpenUSD and validation for *future physical digital twins* | An OpenUSD scene or 3D chart does not prove underlying evidence or physical validity. |
| W3C PROV-O / SLSA | [PROV-O](https://www.w3.org/TR/prov-o/), [SLSA provenance](https://slsa.dev/spec/v1.2/build-provenance) | Derived-from lineage, immutable software artifact predicates | Caller-entered flags and an unsigned digest are neither PROV conformance nor signed SLSA attestation. |
| Conformal Risk Control | [Angelopoulos et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | Future calibrated selective retrieval risk studies | This offline ablation has no exchangeability assumptions, calibration samples or finite-sample risk bound. |

This reference table credits the field. No competitor benchmark result is
represented as an SZL measurement, and no proprietary internal system was
copied or reverse engineered.

## What is implemented

Keep an existing ranking `R=(d_1,...,d_m)`, fixed relevance judgment map
`G`, cutoff `k`, and a per-document declaration of four axis states:
`integrity`, `license`, `freshness`, `derivation`, each in
`{PASS, FAIL, UNKNOWN}`.

The **offline** zero-veto indicator and stable admitted ranking are

```text
A(d) = product over axes a of 1[declared_state(d,a) = PASS]
R_admitted = [d in R in original rank order where A(d) = 1]
```

The same original `G` and `k` are passed to the existing ranking metric
implementation for both arms. For example,

```text
Recall@k(R,G) = |set(R[:k]) intersect {d : G[d] > 0}| / |{d : G[d] > 0}|
Delta_Recall@k = Recall@k(R_admitted,G) - Recall@k(R,G)
CandidateAdmissionFraction = |R_admitted| / |R|
```

The denominator is the **entire** positive-relevance set, including relevant
documents that a gate refused or that were not returned by the initial ranking.
It is never recomputed using only admitted documents. Precision@k retains
denominator `k` even when filtering leaves fewer than `k` documents.
The same invariant applies to nDCG@k, MRR, MAP and R-precision. This is a
paired, fixed-ranking *ablation*, not a new retrieval algorithm.

The source exposes
`szl_retrieval_bench.admission.evaluate_admission_ablation`.
It returns numerical deltas **including negative changes**, counts refused
relevant candidates, per-axis veto counts and an input SHA-256. No raw document
texts or user credentials are placed in the result. Fingerprints bind only the
caller-supplied input values. The API refuses duplicate/invalid IDs, missing
or extra axis records, malformed axis state or metric cutoff, non-finite qrels
and undefined positive-relevance denominators.

A `PASS` observation is **not independently validated**: no signature,
license, timestamp, source receipt or derivation certificate is checked by
this offline adapter. Its `observation_authority` is always
`CALLER_DECLARED_UNVERIFIED`, `execution_authority` is `NONE`,
`training_ready` is false, and its state is `EVALUATED_OFFLINE`
(or `BLOCKED`/`INVALID`). The result is not a decision token.

## Reproducible synthetic counterexample: admission can hurt

```python
from szl_retrieval_bench.admission import AXES, evaluate_admission_ablation

ranked = ["hit1", "hit2", "junk"]
flags = {d: {a: "PASS" for a in AXES} for d in ranked}
flags["hit1"]["license"] = "UNKNOWN"
result = evaluate_admission_ablation(
    ranked, {"hit1": 1, "hit2": 1}, flags, k=2,
)
assert result["gated_minus_baseline"]["recall@2"] == -0.5
assert result["denominators"]["refused_relevant_candidates"] == 1
assert result["signed_receipt_verified"] is False
```

In another synthetic case, refusing a non-relevant first-ranked document
increases P@2 and Recall@2 by 0.5. Neither outcome demonstrates production
quality. Both are required regressions: an attractive admission policy can
lower recall when justified but unverified evidence is excluded.

## Release and science boundaries

1. **Current stage**: Python stdlib source + offline unit tests in an
   isolated signed branch; no new package version, HF Space, model training
   or live query integration. Existing BM25, TF-IDF and RRF functions unchanged.
2. **Next experiment**: pre-register two query cohorts (fact retrieval and
   multi-hop/summary) and compare matched baseline versus admission for
   independent, rights-cleared labeled data. Keep fixed `k`, labels,
   retriever outputs and cost budget; report unknowns, refusal coverage,
   Recall@k, P@k, nDCG@k, p95 and work units; include regressions.
3. **Real integration gate**: bind observations to independently verified
   signed receipts and a declared license policy, with immutable source
   revisions and timestamps. Do not treat `szl.lambda/v1` or its
   unconditional-uniqueness conjecture as an authorization theorem.
4. **Publication**: compare against the official GraphRAG-Bench baselines
   only after a compatible end-to-end run; publish both successful and
   detrimental cases, plus dataset rights and full reproducibility evidence.

This is original, narrow benchmark glue around the existing SZL evaluator;
it makes no claim of a new theoretical guarantee or copied source.
