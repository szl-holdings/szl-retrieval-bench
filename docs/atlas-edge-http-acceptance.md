# Python Knowledge Atlas — HTTP and Cloudflare adapter acceptance

**Authority:** source-only, synthetic-data research preview.
**Not deployed, merged, Cloudflare-bundled, authenticated for private data, model trained, or production qualified.**

The GitHub implementation lives in `szl-retrieval-bench`, not a duplicate service.
This additive change attaches two bounded entrypoints to the existing source:

- `tests/test_hierarchy_http_smoke.py` opens a temporary **loopback-only** HTTP
  listener on an ephemeral port inside the GitHub Actions job, tests the actual
  HTML/JSON/405/400/503 responses, then closes and joins its thread. No daemon,
  controller, scheduler, external network or model process remains.
- `src/atlas_worker.py` (outside the publishable Python package) adapts the same WSGI callable using
  Cloudflare's documented `workers.wsgi.entrypoint(application)`. The adapter
  intentionally relies on the Cloudflare `workers` runtime; it is not used by
  normal CPython imports. A stub unit test checks the delegation only.
- `wrangler.atlas.jsonc` selects a research-only Worker name, current
  `python_workers` flag and module entrypoint. There are **no configured custom
  routes, workers.dev exposure, account IDs, secrets or production bindings**.

The release owners must verify whether preview URLs are implicitly enabled
by account policy before staging. **A checked-in config does not prevent
deployment by another operator, prove secrecy, or constitute account authorization.**
There is no Cloudflare deployment or Pyodide runtime test in this change.

## Exact acceptance / source-to-runtime gates

1. Obtain independent source-owner review for PRs #31, #32 and the corrective
   PR #33, with the true signed-cosine oracle before any public metric.
2. Run full Python 3.11 and 3.12 CI; retain JUnit and source commit IDs.
   `pytest tests/test_hierarchy_http_smoke.py tests/test_hierarchy_edge_contract.py`
   is included in the existing `tests/` CI job.
3. On a separately authorized ephemeral non-workstation runner, install `uv`
   and the Cloudflare Python Workers toolchain following official docs. Use
   a copy of the reviewed source SHA to build `wrangler.atlas.jsonc` with
   `python_workers`. Validate Pyodide imports and the WSGI router through
   actual HTTP responses before any deployment. CPython stub tests are NOT this test.
4. Stage under a unique research name with verified exact source SHA, restricted
   access and no production route. Record returned Worker version, manifest,
   status, content digests, HTTP responses, controlled negative tests, and rollback.
   Verify authentication and tenant denial before using anything but synthetic data.
5. For Hugging Face, **do not** reuse this loopback CLI as a public Docker entrypoint.
   A separately reviewed Space wrapper requires an explicit scoped bind on port
   7860, pinned container image, no private data, source commit and actual Hub
   runtime/commit readbacks. Current GitHub source does not establish a Space.
6. For PyPI, preserve the existing Trusted Publishing workflow, release review,
   version/tag/canonical-wheel checks, and public artifact attestation. Build CI
   is NOT a publication. Version remains the existing unreleased `0.3.2`.
7. Before connecting A11oy, Second Brain, HIRAG or Khipu, require actual
   provenance/rights verification, one writer/owner, a locked >=200-source-group
   held-out study, zero unauthorized tenant observations, citation precision,
   abstention quality, latency/RSS, baseline comparison and rollback.
   Do not infer model quality from ranking-overlap tests.

The publicly documented WSGI adapter (Sep 2, 2026) is
https://developers.cloudflare.com/changelog/post/2026-09-02-python-workers-web-framework-support/
and Python Workers project setup is
https://developers.cloudflare.com/workers/languages/python/.
Hugging Face Docker Space setup:
https://huggingface.co/docs/hub/spaces-sdks-docker/.

Status vocabulary: `SOURCE_CHECKED` and `HTTP_TESTED_IN_CI` are NOT
`STAGED`, `DEPLOYED`, `EVALUATED_FOR_QUALITY`, or `QUALIFIED`.
