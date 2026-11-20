# SPDX-License-Identifier: Apache-2.0
"""Read-only synthetic hierarchy demo: Python WSGI API and responsive HTML.

This is an offline research surface, NOT a governed retrieval API. It never
reads user documents, credentials, provider data, network resources or models.
All facts are constructed in memory from public synthetic literals. Admission
flags are caller-declared in the fixture and do not verify real-world rights.
No live inference, model training, new watcher or persistence occurs.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from urllib.parse import parse_qs

from .hierarchical import HierarchicalIndex, IndexRefused, VectorFact
from .hierarchical_eval import compare_exact

TENANT = "synthetic-demo-only"
WEB_SCHEMA = "szl.retrieval-bench.hierarchy-demo/v1"
MAX_QUERY_BYTES = 512
SOURCE_SHA256 = "0" * 64  # Clearly synthetic, never a verified source digest.

_FACTS = (
    ("north", (1.0, 0.0)),
    ("northeast", (0.8, 0.6)),
    ("east", (0.0, 1.0)),
    ("south", (-1.0, 0.0)),
    ("southwest", (-0.6, -0.8)),
)
INDEX = HierarchicalIndex(
    [
        VectorFact(
            fact_id=fact_id,
            tenant_id=TENANT,
            source_uri="urn:szl:synthetic:atlas-hierarchy-demo",
            source_revision="synthetic-fixture-v1",
            source_sha256=SOURCE_SHA256,
            license_id="SYNTHETIC_FIXTURE_NOT_A_LICENSE_GRANT",
            permission_scopes=("retrieval_evaluation",),
            integrity="PASS",
            freshness="PASS",
            derivation="PASS",
            rights="PASS",
            vector=vector,
        )
        for fact_id, vector in _FACTS
    ],
    tenant_id=TENANT,
    leaf_size=1,
)

_CSS = """
:root{color-scheme:dark;font-family:system-ui,-apple-system,BlinkMacSystemFont,
"Segoe UI",sans-serif;background:#0d1421;color:#f5f7ff}
*{box-sizing:border-box}
body{margin:0}
main{width:min(100% - 2rem,920px);margin:0 auto;padding:2.2rem 0 4rem}
h1{font-size:clamp(1.7rem,4vw,3rem);line-height:1.12;margin:.5rem 0}
h2{font-size:1.1rem;margin:0 0 .8rem}
p{line-height:1.55;color:#c7d1e4}
.kicker{color:#8bdad0;text-transform:uppercase;font-size:.78rem;letter-spacing:.13em;font-weight:700}
.banner{border:1px solid #a97736;background:#352a18;color:#ffe9bb;padding:.8rem 1rem;border-radius:.65rem}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin:1.2rem 0}
.panel{border:1px solid #34425b;background:#162237;border-radius:1rem;padding:1.3rem;min-width:0}
label{display:block;margin:.8rem 0 .25rem;color:#dfecff;font-weight:600}
input{width:100%;min-height:44px;padding:.7rem;background:#09111f;border:1px solid #768aa7;border-radius:.45rem;color:#fff;font:inherit}
button{min-height:44px;margin-top:1.1rem;padding:.65rem 1.2rem;background:#6bd9c4;color:#06231d;border:0;border-radius:.5rem;font:inherit;font-weight:700;cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid #f9d477;outline-offset:2px}
.badge{display:inline-block;padding:.22rem .5rem;border-radius:.3rem;background:#33425b;color:#fff;font-size:.85rem}
.metric{font-size:1.9rem;font-variant-numeric:tabular-nums;font-weight:750}
small,footer{color:#bec9db}
code{overflow-wrap:anywhere;color:#b4f0eb}
ol{margin:0;padding-left:1.5rem}
li{padding:.35rem 0}
.error{padding:.8rem;background:#42202a;border:1px solid #f096a3;border-radius:.5rem;color:#ffe4e9}
@media(max-width:640px){.grid{grid-template-columns:1fr}main{width:min(100% - 1rem,920px);padding-top:1.2rem}.panel{padding:1rem}}
"""


class RequestRefused(ValueError):
    """Invalid user query never reaches a model or other tenant."""


def _params(environ: dict) -> tuple[list[float], str, int, int]:
    raw = environ.get("QUERY_STRING", "")
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_QUERY_BYTES:
        raise RequestRefused("query parameters exceed the 512-byte cap")
    try:
        params = parse_qs(raw, keep_blank_values=True, strict_parsing=True,
                          encoding="utf-8", errors="strict", max_num_fields=3) if raw else {}
    except (ValueError, UnicodeError) as exc:
        raise RequestRefused("malformed query parameters") from exc
    if set(params) - {"vector", "top_k", "leaves"} or any(len(v) != 1 for v in params.values()):
        raise RequestRefused("unknown or duplicated query parameter")
    vector_text = params.get("vector", ["[1,0]"])[0]
    if len(vector_text) > 96:
        raise RequestRefused("query vector is too long")
    try:
        vector = json.loads(vector_text)
    except (ValueError, TypeError) as exc:
        raise RequestRefused("vector must be a JSON array such as [1,0]") from exc
    if not isinstance(vector, list) or len(vector) != 2:
        raise RequestRefused("this synthetic demo requires exactly two vector dimensions")

    def bounded_int(name: str, default: str, max_allowed: int) -> int:
        raw_value = params.get(name, [default])[0]
        if not re.fullmatch(r"[0-9]{1,2}", raw_value):
            raise RequestRefused(f"{name} must be an integer")
        number = int(raw_value)
        if not 1 <= number <= max_allowed:
            raise RequestRefused(f"{name} is outside the allowed range")
        return number

    return (
        vector,
        vector_text,
        bounded_int("top_k", "3", len(_FACTS)),
        bounded_int("leaves", "2", INDEX.leaf_count),
    )


def _evaluate(vector: list[float], top_k: int, leaves: int) -> dict:
    # Selection excludes zero/negative neighbors; the evaluation reference
    # deliberately retains ALL signed-cosine neighbors. Their difference is
    # a measured research limitation, not proof of missing global support.
    selection = INDEX.select(vector, tenant_id=TENANT, top_k=top_k,
                             max_leaf_visits=leaves, min_similarity=0.0)
    fidelity = compare_exact(
        INDEX, {"demo-query": tuple(vector)}, tenant_id=TENANT,
        top_k=top_k, max_leaf_visits=leaves,
    )
    return {
        "schema": WEB_SCHEMA,
        "status": "SYNTHETIC_RESEARCH_ONLY",
        "qualified": False,
        "source_verified": False,
        "permissions_independently_verified": False,
        "training_executed": False,
        "model_executed": False,
        "index_sha256": INDEX.snapshot_sha256,
        "fact_count": len(_FACTS),
        "selection": selection,
        "fidelity": {
            "queries": fidelity["denominators"]["queries"],
            "top1_agreement_fraction": fidelity["top1_agreement_fraction"],
            "neighbor_retention_fraction": fidelity["exact_neighbor_retention_fraction"],
            "approximate_scored_vectors": fidelity["denominators"]["approximate_scored_vectors"],
            "reference_scored_vectors": fidelity["denominators"]["exact_scored_vectors"],
        },
        "limits": "Synthetic, caller-declared facts only. Selection requires strict cosine > 0 while exact reference ranks signed cosine (including zero/negative neighbors). No external rights validation, model, full AtlasKV attention, speedup, or production security claim.",
    }


def _page(vector_text: str, top_k: int, leaves: int,
          report: dict | None, error: str | None) -> str:
    items = ""
    metrics = "<p>No result. Correct the bounded query parameters.</p>"
    if report is not None:
        selected = report["selection"]
        rows = selected["ranking"]
        if rows:
            items = "<ol>" + "".join(
                "<li><strong>" + html.escape(str(item["fact_id"])) +
                "</strong> — cosine " + str(item["cosine"]) + "</li>"
                for item in rows
            ) + "</ol>"
        else:
            items = "<p>No matching candidates within the selected budget. This does not prove absence.</p>"
        fidelity = report["fidelity"]
        metrics = (
            "<div class='metric'>" + str(fidelity["neighbor_retention_fraction"]) + "</div>"
            "<small>Fraction of selected exact-reference neighbors retained, synthetic query only.</small>"
            "<p>Vector comparisons: " + str(fidelity["approximate_scored_vectors"]) +
            " pruned / " + str(fidelity["reference_scored_vectors"]) + " reference.</p>"
        )
    safe_vector = html.escape(vector_text, quote=True)
    fail = "<p class='error' role='alert'>" + html.escape(error) + "</p>" if error else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SZL Knowledge Atlas — Synthetic Research Demo</title>
<style>{_CSS}</style></head><body><main>
<div class="kicker">SZL Holdings · Python research interface</div>
<h1>Knowledge Atlas</h1>
<p>Explore a deterministic, bounded hierarchical vector selector. No account,
model, dataset or production system is connected to this page.</p>
<div class="banner" role="status"><strong>Research only.</strong> Five fabricated 2D
facts. PASS labels are declarations, not independently verified source rights.
No model training, qualified evaluation or production serving.</div>
<div class="grid">
<section class="panel" aria-labelledby="settings-title"><h2 id="settings-title">Query selector</h2>
{fail}
<form action="/" method="get">
<label for="vector">Query vector (JSON)</label>
<input id="vector" name="vector" required value="{safe_vector}"
maxlength="96" autocomplete="off" spellcheck="false">
<label for="top_k">Candidates to display (1–{len(_FACTS)})</label>
<input id="top_k" name="top_k" type="number" min="1" max="{len(_FACTS)}"
value="{top_k}" required>
<label for="leaves">Leaf visit budget (1–{INDEX.leaf_count})</label>
<input id="leaves" name="leaves" type="number" min="1" max="{INDEX.leaf_count}"
value="{leaves}" required>
<button type="submit">Evaluate synthetic query</button>
</form></section>
<section class="panel" aria-labelledby="results-title"><h2 id="results-title">Selected facts</h2>
{items or "<p>Run a query to see selected facts.</p>"}
</section>
<section class="panel" aria-labelledby="fidelity-title"><h2 id="fidelity-title">Offline comparison</h2>
{metrics}</section>
<section class="panel"><h2>Evidence boundary</h2>
<p><span class="badge">NOT QUALIFIED</span></p>
<p>Index content digest (not a signature):</p><code>{INDEX.snapshot_sha256}</code>
<p>HTTP API: <code>/api/v1/hierarchy/evaluate?vector=%5B1%2C0%5D</code></p>
<p>Production readiness: <code>/readyz</code> returns HTTP 503.</p></section>
</div>
<footer>Python WSGI. No JavaScript, user accounts, private data,
database writes or provider credentials. Selection excludes zero/negative
scores; the exact-cosine reference includes them. Research proxy only, not full AtlasKV.</footer>
</main></body></html>"""


def _respond(start_response, status: str, payload: str, mime: str):
    body = payload.encode("utf-8")
    headers = [
        ("Content-Type", mime + "; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        ("Content-Security-Policy",
         "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
         "base-uri 'none'; frame-ancestors 'none'"),
    ]
    start_response(status, headers)
    return [body]


def _json(start_response, status: str, data: dict):
    return _respond(start_response, status,
                    json.dumps(data, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", "application/json")


def application(environ, start_response):
    """PEP 3333 WSGI callable; bounded, read-only and synthetic by construction."""
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    if method != "GET":
        return _json(start_response, "405 Method Not Allowed",
                     {"status": "REFUSED", "reason": "GET only"})
    if path == "/healthz":
        return _json(start_response, "200 OK", {
            "status": "SYNTHETIC_DEMO_PROCESS_RESPONDING", "qualified": False,
            "provider_connected": False, "index_sha256": INDEX.snapshot_sha256,
        })
    if path == "/readyz":
        return _json(start_response, "503 Service Unavailable", {
            "status": "NOT_PRODUCTION_READY",
            "reason": "independent rights, owner review, evaluation and deployment gates absent",
        })
    if path not in ("/", "/api/v1/hierarchy/evaluate"):
        return _json(start_response, "404 Not Found",
                     {"status": "NOT_FOUND"})
    try:
        vector, original, top_k, leaves = _params(environ)
        report = _evaluate(vector, top_k, leaves)
    except (RequestRefused, IndexRefused) as exc:
        if path.startswith("/api/"):
            return _json(start_response, "400 Bad Request",
                         {"status": "REFUSED", "reason": str(exc)})
        return _respond(start_response, "400 Bad Request",
                        _page("[1,0]", 3, 2, None, str(exc)), "text/html")
    if path.startswith("/api/"):
        return _json(start_response, "200 OK", report)
    return _respond(start_response, "200 OK",
                    _page(original, top_k, leaves, report, None), "text/html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic research-only Python WSGI browser on loopback.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host != "127.0.0.1" or not 1024 <= args.port <= 65535:
        parser.error("research preview binds loopback and ports 1024..65535 only")
    # Import socket-serving support only for explicit local CLI execution.
    # Cloudflare's Pyodide Worker adapter must not import a socket listener.
    from wsgiref.simple_server import make_server

    with make_server(args.host, args.port, application) as server:
        print(f"SYNTHETIC_RESEARCH_ONLY http://{args.host}:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
