# AtlasKV-inspired hierarchy: Python-first SZL execution plan

**Status:** PHASE-0 SOURCE-ONLY PROPOSAL / OFFLINE PROXY. **Owner:** `szl-retrieval-bench` for experimental ranking; `a11oy` for any future production admission integration. **No live training, GPU inference, deployment, model qualification, or provider publication is authorized by this document.**

## Research anchor and honest boundary

- AtlasKV: Haoyu Huang et al., *AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM*, ICLR 2026, https://arxiv.org/abs/2510.17934; upstream source https://github.com/HKUST-KnowComp/AtlasKV at `61e3abc1fb09fbfb426cfc847c4c26c5e94bcaeb` (MIT, upstream code and pretrained components require their own review).
- AtlasKV's KG2KV and hierarchical pruning operate alongside specialized LLM attention code. The new `hierarchical.py` is **not** that neural attention path: it uses caller-supplied normalized vectors and a deterministic CPU binary centroid tree. It trains nothing, claims no speedup, cannot independently validate external rights, and caps input at **2,048 vectors / 1,024 dimensions**.
- `hierarchical_eval.compare_exact` measures only *neighbor-retention fidelity against full cosine search on identical vectors*. It does not evaluate grounded answers, trained weights, model quality, semantic claims, or production behavior.
- Prior evidence: `szl-second-brain/evidence/bm25-retrieval-eval-20261104.md` measures a 30-query public projection and exposes a weak-hit/no-support gap; `a11oy/execution/brain/RETRIEVAL_BENCHMARK.md` marks its large comparative benchmark PLANNED. Do not promote those numbers to a new model card.

## Ownership / placement

1. **GitHub authoritative source:** `szl-holdings/szl-retrieval-bench`, additive `src/szl_retrieval_bench/hierarchical.py`, `hierarchical_eval.py`, tests, and this runbook. Do not alter existing BM25/TF-IDF/RRF ranking, release workflow, package version, training jobs, or branch protection. The owner review and CI remain mandatory.
2. **Graph provenance:** `szl-holdings/a11oy/execution/brain/KNOWLEDGE_GRAPH_SCHEMA.md` specifies source-backed `Claim`, `Evidence`, `Receipt`, `SUPERSEDES`, `DERIVED_FROM` links; it currently labels durability MODELED. The graph model is not proof of a deployed store.
3. **Second Brain:** `szl-holdings/szl-second-brain` owns local memory admission/FTS; experimental retrieval may consume an *explicitly exported, rights-cleared, revision-pinned* copy, not live or private unreviewed memory.
4. **Khipu / inference:** a later optional adapter may return source handles for verification; no changes to Qwen weights, attention, vLLM routes, trainer targets, controllers, or GPU jobs in Phase 0.

## Proposed contract

`VectorFact`: stable fact ID, exact tenant, source locator, source revision, SHA-256-shaped source digest, license ID, permission scopes, four caller-declared PASS axes, and finite vector. An entire index fails closed on mixed tenants, missing permissions, malformed vectors, duplicate IDs or unknown/failed axes. Input indexing never runs a model or remote code. Query must match the tenant; results carry index and query digests, visited nodes/leaves, scored-fact denominator, exhaustive flag, and honest `ABSTAINED_OFFLINE` semantics.

**Critical boundary:** caller-declared PASS is not verified provenance or authorization. No private or cross-tenant dataset is allowed into this prototype until an upstream independent validator supplies verified permissions, source bytes, revocation state, and a binding receipt. No negative claim of *global absence* may be made from an unsearched subtree. Hashes here detect changes; they are not signatures.

## Phased acceptance, with explicit holds

| Stage | Deliverable | Acceptance criterion | Status |
| --- | --- | --- | --- |
| 0 — source pilot | Python stdlib bounded selector, exact-cosine fidelity comparator, adversarial tests | Owner review; Python 3.11 and 3.12 CI tests green; deterministic fixture hashes; zero existing baseline behavior changes | PR proposed; CI not inferred |
| 1 — legitimate corpus | Source-pinned, rights-cleared synthetic/public graph export; independent tenant/provenance validator | Source licenses, rights, data lineage, deletion/retraction and per-tenant authorization verified byte-for-byte before indexing | HOLD |
| 2 — retrieval evaluation | ≥200 source-group-held-out, preregistered questions over 1k facts; expand to 10k and 100k only after raising caps with profiling | Same corpus/query/qrels/model revisions across BM25, dense, RRF, exhaustive cosine and hierarchy; report Recall@5/10, nDCG@10, MRR, citation precision, unsupported-answer/abstention, p50/p95, peak RSS, bytes transferred, query cost | HOLD |
| 3 — performance gate | Repeated experiments over independently held-out groups and nontrivial graph paths | Target for decision (not a result): ≥0.95 exact neighbor retention@10 and ≥30% fewer exact vector comparisons, with no tenant leaks or material answer/abstention regression; report confidence intervals and failures | HOLD |
| 4 — model-attention experiment | Isolated faithful AtlasKV Llama reproduction on an authorized remote GPU, then separate Qwen feasibility study | Fixed upstream/model/dataset SHA, safe checkpoint handling, immutable steps/receipts, reproducible baselines, budget and owner approval; no trust_remote_code on unreviewed weights | HOLD / no GPU job created |
| 5 — HF showcase | Existing `SZLHOLDINGS` dataset/model/Space path, not a duplicate | Public/synthetic rights proven, exact artifact hashes and Hub commit readback; TRAINING / ARTIFACT_VERIFIED / EVALUATED / DEPLOYED / QUALIFIED distinctly marked; benchmark version, denominators, limitations | HOLD / no Hub write |
| 6 — Cloudflare | Python Worker **gateway** for authorized metadata/handles and proof links, not an 8B model in an edge Worker | `python_workers` compatibility flag, pywrangler build, authn/authz, deny-by-default tenant tests, service binding to governed backend, rate/size limits, staging smoke and rollback; no DNS or Access changes until approved | HOLD / no Worker write |
| 7 — PyPI | Optional reviewed additive API in `szl-retrieval-bench` package | Preserve existing trusted-publisher OIDC workflow, package-name/version/tag binding, wheel isolation, CI tests and downstream compatibility; explicit separate release approval and PyPI artifact attestation/readback | HOLD / no release |

**Cloudflare Python-specific constraint:** Python Workers run under Pyodide/WebAssembly, not a general CUDA or arbitrary CPython GPU environment. The edge should validate and route small requests, not own training or model-serving processes. Python backend can run on an authorized VM or Hugging Face Space; browser UI need not be rewritten for language purity.

## Cross-plane handoff (no silent promotion)

`GitHub protected source SHA` → `signed/verified admission decision` → `immutable dataset/vector source SHA` → `bounded benchmark receipt` → `HF Hub exact commit and public readback` → `Cloudflare staged deployment revision + authenticated smoke` → `PyPI gated wheel, tag and provenance`.

Each link must name the upstream revision, artifact hash, owner, UTC observation, tested corpus, held-out split, coverage and unresolved blockers. States are independent. CI green ≠ model trained; Hub card ≠ deployed model; HTTP 200 ≠ trusted facts; a hashed fixture ≠ independently signed evidence.

## Operational constraints

- The owner workstation is a control room. Follow its local guard and ownership instructions; do not run local test suites, GPU work, model downloads, Docker, new watchers or background services as part of this proposal.
- Respect other operators' locks, Flagship54 guard ownership, source CODEOWNERS, and protected reviews. No spending, credential extraction, account-wide security changes, private publication, training restarts, or unapproved edge/DNS changes.
- PyPI `szl-retrieval-bench` source package declares **0.3.2** on GitHub main, while the publicly inspected PyPI page displayed **0.3.1**; treat release as not verified until metadata and wheel readback agree. No version bump in this proposal.
- No evidence is presented here that the active HF/Cloudflare/Salad job estate is healthy or that it should be modified. Existing training-state watches retain sole ownership.

## Exact next engineering step

After the PR's CI and owner review are clean: add an explicitly **separate** data-export adapter that reads pinned public/synthetic facts from the existing A11oy schema, uses independently verified rights and tenancy checks, emits the prescribed provenance manifest, and feeds only admitted vectors into this offline comparator. Do not merge the AtlasKV custom attention path or deploy a Cloudflare Worker until the evaluation and security gates above have reproducible evidence.
