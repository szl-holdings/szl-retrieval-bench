# Atlas hierarchy — Python frontend/backend research service

**Status:** source-only synthetic WSGI preview, stacked on
[szl-retrieval-bench PR #31](https://github.com/szl-holdings/szl-retrieval-bench/pull/31).
**Not merged, deployed, authenticated for real data, or qualified.**
Do not attach this demo directly to a11oy production routing.

## Source ownership and objective

- Own new files in `szl-retrieval-bench`; do not modify the active parent PR,
  retriever baselines, training, model weights or existing provider services.
- Python stdlib WSGI serves a responsive, accessible, server-rendered HTML frontend
  and a JSON backend API over **five synthetic 2D vectors only**. No JavaScript,
  database, private data, external endpoints, browser automation, inference or GPU.
- Endpoint `GET /`: research form and actual selected vector IDs with retention
  against the same-predicate exhaustive reference; mobile and desktop CSS.
- Endpoint `GET /api/v1/hierarchy/evaluate?vector=%5B1%2C0%5D&top_k=3&leaves=2`:
  JSON state, SHA-256 of synthetic index and query, resource comparison denominators,
  explicit unverified permission/qualification flags.
- `GET /healthz` only proves this demo callable responds.
  `GET /readyz` returns **503**, intentionally refusing production readiness.

## Running a local developer preview (NOT on BETTERWITHAGE)

In a separate authorized development container / remote host, after checkout and
install under CPython >=3.11:

    python -m pip install -e . pytest
    python -m pytest tests/test_hierarchical.py tests/test_hierarchy_web.py -q
    python -m szl_retrieval_bench.hierarchy_web --host 127.0.0.1 --port 7860

Open `http://127.0.0.1:7860/`. This is a local demo only: the CLI refuses
non-loopback binding, keeps no state, never contacts model providers, and does not
read local user content. Do not run on BETTERWITHAGE: its v1 laptop rules forbid
test suites and new servers there. No controller, new watcher or scheduler is created
when this module is merely imported.

## Security and research limitations

- All `PASS` axes are **caller-declared synthetic flags**, not independently
  verified rights, identity, source provenance or signature. Public serving of
  any real source or tenant data requires a distinct upstream authorized validator.
- Query parameters are limited to three known fields, 512 bytes total, a JSON
  vector of exactly two dimensions, an index-bounded leaf budget, and 1..5 results.
- No tenant switching, mutable input corpus, file upload, user-supplied source
  content, secrets, state-changing methods, or outbound calls.
- CSP, no-store and HTML escaping protect a small **demo** attack surface; these
  tests are not penetration testing or security certification.
- This is **not AtlasKV model-attention integration**. No trained weights, neural
  embedding, KV cache, benchmark against the paper or quality claim.
- Existing selector uses strict positive cosine threshold for BOTH the pruned
  and exhaustive reference; therefore this is not unconstrained signed-cosine
  top-k for negative similarities. No global absence can be inferred from pruning.
  Record any changes to this inherited predicate in the parent research lane.
- No real-user telemetry, analytics, logging of private queries or production
  treatment. The source and index digests are checksums, not signatures.
- A successful API/healthz response does not qualify data rights or deployment.

## GitHub → Hugging Face → Cloudflare → PyPI admission plan

1. **GitHub:** Parent PR #31 must pass required review and Python 3.11/3.12
   CI, including tests and CodeQL. This stacked branch must separately pass
   CI and owner review. Protected source SHAs and clean wheel installation
   must be checked. No branch-protection bypass or automatic merge.
2. **Independent rights/provenance:** Export a separately authorized,
   pinned, public/synthetic graph manifest from A11oy/Second Brain. Every
   item needs actual source bytes, license scope, tenant authorization,
   retractions and hash, not self-attested `PASS` strings. Deny by default.
3. **Held-out benchmark:** Frozen >=200 source-group-held-out queries with
   correct denominators, lexical, vector, hybrid and exhaustive baselines,
   citations and abstention, p50/p95 latency and resource usage. This demo
   gives no evidence for those.
4. **Hugging Face:** Once authorized, use an existing SZLHOLDINGS Space owner
   rather than creating a duplicate. Build an immutable, safe Docker/Gradio
   wrapper with a **public-synthetic-only** data contract. Read back exact
   Hub commit, run state, rendered page, permissions and shutdown behavior.
   Never publish private checkpoints, secrets or raw training evidence.
   Do not assume free org-hosted runtime; current HF hosting eligibility must
   be checked independently.
5. **Cloudflare:** Python Workers support WSGI via `workers.wsgi.entrypoint`.
   Edge Workers should own authentication and authorization **before**
   forwarding to a governed backend for real retrieval. Pywrangler build
   must prove package/runtime compatibility; use separate staging Worker,
   explicitly configured CORS, Access policy, rate/size caps, security checks,
   smoke and rollback. Do not route a public anonymous research demo to
   tenant data. No DNS/Access account changes or Worker deploy here.
6. **PyPI:** Existing `publish-pypi.yml` is source of truth. Never publish
   from a branch or bump a package without owner review, new version,
   verified wheel/sdist, tag binding and OIDC Trusted Publishing; read back
   signatures/attestations, install and import. This branch changes no
   version or release workflow.

## Verification and exit criteria

Run `tests/test_hierarchy_web.py` under GitHub CI on Python 3.11 and 3.12.
Tests cover valid HTML and API responses, mobile viewport metadata, source
and tenant boundaries, disallowed methods, query overload, malformed vectors,
non-finite values, readiness denial, escaping, caching and declared evidence
limitations. Zero requests touch Hugging Face, Cloudflare, GPU hosts or PyPI.

The definition of production-ready is NOT satisfied here. It requires
independent tenant and source authorization, measured held-out quality,
a deployed governed backend, observable frontend, Cloudflare staged routing,
immutable HF/CI receipts, and separately approved PyPI release.
