# SPDX-License-Identifier: Apache-2.0
"""Pure Python WSGI integration tests; no server, network, model or credentials."""

import json
from urllib.parse import urlencode

from szl_retrieval_bench.hierarchy_web import INDEX, TENANT, application


def request(path="/", *, qs="", method="GET"):
    observed = {}
    def capture(status, headers):
        observed["status"] = status
        observed["headers"] = dict(headers)
    raw = b"".join(application({
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": qs,
        "wsgi.version": (1, 0),
        "wsgi.input": None,
        "wsgi.errors": None,
        "wsgi.url_scheme": "http",
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": True,
    }, capture))
    return observed["status"], observed["headers"], raw.decode("utf-8")


def api(**params):
    return request("/api/v1/hierarchy/evaluate", qs=urlencode(params))


def test_root_is_responsive_and_has_actual_result_rows():
    status, headers, html = request("/")
    assert status == "200 OK"
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "name=\"viewport\"" in html
    assert "Knowledge Atlas" in html
    assert "Selected facts" in html
    assert "north" in html
    assert "SYNTHETIC" not in headers.get("Set-Cookie", "")
    assert "no-store" == headers["Cache-Control"]


def test_api_evaluates_only_synthetic_vectors_and_retains_pins():
    status, headers, body = api(vector="[1,0]", top_k="3", leaves="2")
    assert status == "200 OK"
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    result = json.loads(body)
    assert result["status"] == "SYNTHETIC_RESEARCH_ONLY"
    assert result["fact_count"] == 5
    assert result["index_sha256"] == INDEX.snapshot_sha256
    assert result["selection"]["tenant_id"] == TENANT
    assert result["selection"]["ranking"][0]["fact_id"] == "north"
    assert result["qualified"] is False
    assert result["source_verified"] is False
    assert result["training_executed"] is False
    assert result["model_executed"] is False


def test_query_changes_rank_and_budget_controls_scored_count():
    first = json.loads(api(vector="[1,0]", top_k="3", leaves="1")[2])
    second = json.loads(api(vector="[0,1]", top_k="3", leaves="1")[2])
    full = json.loads(api(vector="[1,0]", top_k="3",
                          leaves=str(INDEX.leaf_count))[2])
    assert first["selection"]["query_sha256"] != second["selection"]["query_sha256"]
    assert first["selection"]["scored_facts"] == 1
    assert full["selection"]["scored_facts"] == 5
    assert full["selection"]["exhaustive"] is True
    assert full["fidelity"]["neighbor_retention_fraction"] == 1.0


def test_fidelity_reports_denominators_and_does_not_claim_production():
    result = json.loads(api(vector="[1,0]", top_k="2", leaves="2")[2])
    fidelity = result["fidelity"]
    assert fidelity["queries"] == 1
    assert fidelity["approximate_scored_vectors"] <= fidelity["reference_scored_vectors"]
    assert result["permissions_independently_verified"] is False
    assert "strict cosine > 0" in result["limits"]


def test_health_checks_distinguish_process_from_readiness():
    status, _, body = request("/healthz")
    assert status == "200 OK"
    assert json.loads(body)["qualified"] is False
    status, _, body = request("/readyz")
    assert status == "503 Service Unavailable"
    assert json.loads(body)["status"] == "NOT_PRODUCTION_READY"


def test_rejects_unknown_method_and_path():
    assert request("/", method="POST")[0] == "405 Method Not Allowed"
    assert request("/hidden")[0] == "404 Not Found"


def test_refuses_invalid_vectors_without_500_or_echoing_input():
    for vector in ('[0,0]', '[1]', '[1,2,3]', '"text"', 'null',
                   '[true,0]', '[NaN,0]', '[Infinity,0]', '<script>alert(1)</script>'):
        status, headers, body = api(vector=vector)
        assert status == "400 Bad Request"
        assert json.loads(body)["status"] == "REFUSED"
        assert headers["Cache-Control"] == "no-store"
        assert "<script>" not in body


def test_refuses_query_overflow_duplicates_and_unknown_fields():
    for q in (
        "vector=" + "1" * 513,
        "vector=%5B1%2C0%5D&vector=%5B0%2C1%5D",
        "unknown=1",
        "vector=%5B1%2C0%5D&extra=x",
        "vector",
    ):
        assert request("/api/v1/hierarchy/evaluate", qs=q)[0] == "400 Bad Request"


def test_refuses_invalid_budget_before_index_query():
    for field, bad in (("top_k", "0"), ("top_k", "6"),
                       ("top_k", "1.5"), ("leaves", "0"),
                       ("leaves", "6"), ("leaves", "999")):
        status, _, _ = api(vector="[1,0]", **{field: bad})
        assert status == "400 Bad Request"


def test_headers_prevent_embedding_and_active_remote_resources():
    _, headers, _ = request("/")
    policy = headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"


def test_browser_validation_escapes_untrusted_errors():
    status, _, html = request("/", qs=urlencode({"vector": "<svg/onload=alert(1)>"}))
    assert status == "400 Bad Request"
    assert "<svg/onload" not in html
    assert "role='alert'" in html


def test_demo_cannot_access_external_projects_or_load_models():
    # The only index is immutable synthetic literals; a tenant-switch parameter
    # is not implemented in either public endpoint.
    assert request("/api/v1/hierarchy/evaluate",
                   qs="tenant_id=another-tenant")[0] == "400 Bad Request"
    assert all(not path.startswith("https://") for path in (
        fact.source_uri for fact in INDEX._facts
    ))
