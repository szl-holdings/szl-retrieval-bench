# Bounded source-metadata admission in hierarchical retrieval

**Source owner:** `szl-holdings/szl-retrieval-bench`
**Base main:** `fc6149e070a7fe459715f7fe345cdc411f0735f7`
**State:** SOURCE_ONLY / UNREVIEWED. No releases, provider writes, training or consumer admission.

## One scoped improvement

The existing hierarchical evaluation index enforces at most 2,048 facts and
1,024 vector dimensions but allowed arbitrarily large caller-declared
`source_uri` and `permission_scopes` strings in each fact. The snapshot
digest serializer uses these strings unmodified; a single already-accepted
fact could amplify memory and serialization work without breaking the
fact-count or vector-dimension limits.

**Patch:** a total limit of 8,192 JSON-escaped UTF-8 bytes across
the metadata string values per fact (including their JSON string quotes),
and at most 16 permission-scope declarations. Raw UTF-8 byte counts
underestimate JSON control-character and quote/backslash escaping. Invalid
Unicode is explicitly refused before snapshot serialization. Limits are source-only research resource
bounds; they do not verify source content, right to use data, signature
identity, or whether the declared source revision is truly immutable.

The `szl.retrieval-bench.hierarchical-offline/v1` result schema remains
unchanged. Existing ordinary facts produce unchanged digest and ranking
semantics. Excess metadata causes `IndexRefused`, not an attempt to
automatically trim or change a license/scope declaration.

## Reproducer and test proof

Run with the checked-out source on the local Python path:

```shell
python -m pytest -q tests/test_hierarchical.py
```

Prior baseline on BETTERWITHAGE: `32 passed` with an explicitly set local src path.
The follow-up adds synthetic control-character/quote/backslash adversarial
tests for the serialized bound; the exact-head hosted suite remains the
release evidence authority.
The new tests reject: 9,000 ASCII source bytes, 2,200 four-byte Unicode
codepoints, an invalid surrogate, and 17 scope entries; a 7,900-byte
ordinary fixture remains admissible, with `deployment_qualified=false`.

The broader Python suite returned **228 passed, 94 subtests**, when
`tests/test_package_version.py` was excluded. The full suite without a
project installation failed precisely because the isolated checkout had
no installed `szl-retrieval-bench` distribution to satisfy that
environment-dependent version test. Do not claim full-suite green until
the exact-head CI installs the package and passes its own release checks.

## Independent critique and prior-art boundary

Microsoft's GraphRAG paper (Edge et al.; arXiv:2404.16130,
https://arxiv.org/abs/2404.16130) addresses global sensemaking over a
corpus and does not establish this resource budget or guarantee retrieval
accuracy. No upstream source code was copied; this is a bounded security
hygiene change, **not a novel RAG algorithm**.

Public source-bound patch reviews were requested via local Ollama from
Kimi (`kimi-k3.1-pro:cloud`, proposer) and Ministral-3:3b (skeptic).
Kimi's initial response did not meet the required output schema;
a later short request returned an empty answer. Ministral returned HTTP
500 on the first request. These calls establish *attempted model
feedback*, not independent approval; exact-head CI, branch-required
review and post-merge admission remain outstanding. Model agreement,
even if later obtained, cannot authorize release.

**No Cloudflare/HF/PyPI/Snowflake deployment, training or signing action
is requested in this branch.**
