# Proof-carrying late interaction: bounded research ablation (2026-11-19)

**Status:** tested offline mathematical adapter; NOT a neural model, ColBERT
implementation, independently validated evidence verifier, model-training
component, HF deployment, or production admission. Every vector, relevance
judgment and evidence flag is caller-supplied. A content SHA-256 is **not** a
signature, and caller-declared PASS is **not** cryptographic PASS.

This adds `src/szl_retrieval_bench/late_interaction.py` without replacing
existing BM25, classical TF-IDF, RRF, metric, receipt-chain or evidence
admission code. Kimi was consulted through `kimi-k3.1-pro:cloud` in the
BETTERWITHAGE Ollama terminal. Its research suggestions were reviewed and
implemented independently, with source-specific attribution below.

## Field research, what is adopted and not claimed

| Public team / location | Published baseline | SZL-scoped idea | Explicit non-claim |
| --- | --- | --- | --- |
| Qwen / China | [Qwen3 Embedding/Reranker](https://github.com/QwenLM/Qwen3-Embedding), [paper](https://arxiv.org/abs/2506.05176) | Separate ranking from reranking evaluation; future multilingual matched cohorts | This module loads no Qwen weights, produces no multilingual model measurement |
| LG AI Research / South Korea | [EXAONE Deep](https://github.com/LG-AI-EXAONE/EXAONE-Deep), [K-EXAONE](https://github.com/LG-AI-EXAONE/K-EXAONE-2.0) | Treat math/coding/reasoning benchmark claims as external baselines, not direct comparators | No EXAONE weights, no Korea-language benchmark run |
| Research Ireland ADAPT, TCD | [GrOWL-RAG ontology-guided document relation extraction](https://www.adaptcentre.ie/news-and-events/adapt-at-trinity-college-dublin-researchers-paper-accepted-at-semantics-2025/) | Type and lineage should constrain retrieval separately from relevance | No ontology reasoning engine built or claims about Re-DocRED |
| University of Galway / Insight | [Knowledge graph extraction, linking and maintenance](https://www.universityofgalway.ie/dsi/knowledge_graphs/) | Keep evidence source IDs and derivation status separate from semantic rank | No entity-linking model or proprietary Irish archive ingestion |
| Stanford Future Data / US | [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/), [source](https://github.com/stanford-futuredata/ColBERT) | Compute bounded token-level MaxSim from synthetic vectors | No residual compression, PLAID engine, ColBERT tokenization or neural weights |
| Independent retrieval study | [Token importance in multi-vector retrieval](https://arxiv.org/abs/2511.16106) | Paired unweighted versus positive query-token importance | No claimed recall improvement absent independent held-out data |
| Microsoft Research / US | [GraphRAG](https://github.com/microsoft/graphrag), [query methods](https://github.com/microsoft/graphrag/blob/main/docs/query/overview.md) | Evaluate graph-aware candidate generation separately in a later cohort | No expensive LLM indexing or inference launched |
| Community supply-chain standard | [SLSA v1.2](https://slsa.dev/spec/v1.2/) | Separate unsigned integrity digests from independently authenticated attestations | No SLSA level or signed model provenance established |

The published techniques inspire a small independent mathematical implementation;
no external source code, tokenizers, corpora or model weights are copied.
Before training on any third-party dataset, verify dataset rights and terms.

## Mathematical contract

Each nonzero finite query/document token vector is normalized
`v_hat = v / ||v||_2`. The score for query tokens `q_i`, document tokens
`d_j` and positive query-token importance `w_i` is:

```
m_i(D) = max_j cosine(q_i, d_j)
S_unweighted(Q,D) = sum_i m_i(D)
S_weighted(Q,D) = sum_i w_i * m_i(D)
```

The same cosine matrix is reused for both arms; there is no double inference.
Rank by descending unrounded score, then ascending document ID. Negative
cosine similarity remains negative. Scores are NOT probabilities or confidence.

With `n` query tokens, `d` vector dimensions and document lengths
`m_j`, the deterministic operation counter is:

```
pairwise_cosine = n * sum_j(m_j)
cosine_dot_scalar_multiplications = d * pairwise_cosine
query_weight_multiplications = n * document_count
token_normalizations = n + sum_j(m_j)
```

The model refuses a configured scalar-work budget before scoring. These are
arithmetic counts, not measured GPU utilization, inference latency or energy.

Only after ranking are fixed relevance labels used to evaluate existing
nDCG@k, recall@k, P@k, R-precision, MRR and MAP definitions. Relevance labels
for absent positive documents remain in the denominator. If the caller
supplies per-document `integrity`, `license`, `freshness`, and `derivation`
flags, the existing `admission.py` ablation is called for each arm:

```
A(doc) = product_axis 1[caller_state(doc, axis) == PASS]
```

This is a hard declaration-based veto; it is not a signature/rights validator.
No relevance judgment ever influences the original scores/ranking. The
unchanged admitted-ranking evaluation may show lost recall.

## Synthetic counterexamples (not a leaderboard)

With `q_1=(1,0), w_1=1`, `q_2=(0,1), w_2=10`, candidates
`A=(1,0)` and `B=(0,1)`, the baseline scores tie at 1 and rank
`A,B`. Importance ranking scores `B=10,A=1` and ranks `B,A`. If `B`
is the only relevant document, recall@1 changes 0 to 1 and
MRR 0.5 to 1; if `A` alone is relevant, the very same weighting changes
recall@1 from 1 to 0 and MRR from 1 to 0.5. **Both** cases must pass tests.
No statistical advantage can be inferred.

## Scope / adversarial acceptance

* Strict, bounded shape and types: at most 128 docs, 64 query tokens, 128
  document tokens/doc, 128 dimensions, 4096 judged IDs and 4 million scalar
  products; finite vectors and strictly positive finite weights required.
* Reject duplicate token/document IDs, ragged/zero vectors, nonfinite values,
  zero/negative importance, bad cutoffs, malformed evidence and over-budget
  work. On invalid inputs emit only a stable code, not any partial metrics.
* SHA-256 bindings cover source ID, inputs, weights, qrels and optional
  declared evidence. Change detection is reproducible; authenticity requires
  external trusted signatures and independent timestamps.
* All outputs are `EVALUATED_OFFLINE` or explicit `BLOCKED/INVALID`;
  `training_ready=false`, `signed_receipt_verified=false`, and
  `production_admitted=false` must never become inferred successes.
* Run existing package test suite on Python 3.11 and 3.12 under protected PR
  CI; don't add dependencies, workflows or side effects.

## Path to genuine SZL frontier evaluation

1. Pre-register rights-cleared matched English, Korean/Chinese and scientific
   question cohorts with fixed qrels, source IDs and hold-out partitions.
2. Compare existing BM25, TF-IDF, RRF, token MaxSim and later real multilingual
   embeddings under the same candidate and compute budgets. Record both wins
   and regressions, tokenization/model-specific licenses, latency p95 and
   hardware measurements only if actually observed.
3. Route source/citation signatures and graph-edge derivations through
   Khipu/Hatun's independently verified receipts; derive the four-axis states
   from authenticated readback rather than caller declarations. Align evidence
   epochs to both graph node IDs and model/card revisions.
4. Keep a hard fail-closed policy outside quantitative selection. The existing
   Lutar Λ weighted geometric invariant, if used as a future selection
   objective, must never compensate for a failed legal, identity or
   derivation gate.
5. Only deploy from reviewed GitHub SHA after signed artifact parity and
   runtime/production admission. No provider credentials are used here.

**Status boundary:** This module demonstrates a deterministic research
ablation. It does not establish an original mathematical theorem or that
SZL exceeds the cited systems.
