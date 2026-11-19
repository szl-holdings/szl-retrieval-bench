# SPDX-License-Identifier: Apache-2.0
"""Real loopback HTTP acceptance for the synthetic hierarchy demo.

Executed by GitHub CI, never by BETTERWITHAGE. No external network calls,
persistent listener, training, model loading, or user data.
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from wsgiref.simple_server import WSGIRequestHandler, make_server
import json

from szl_retrieval_bench.hierarchy_web import application


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):
        pass


@contextmanager
def _ephemeral_server():
    server = make_server("127.0.0.1", 0, application,
                         handler_class=_QuietHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive(), "synthetic test listener did not stop"


def _get(url, *, method="GET"):
    request = Request(url, method=method)
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_real_http_serves_accessible_html_and_json():
    with _ephemeral_server() as root:
        code, headers, body = _get(root + "/")
        assert code == 200
        assert headers["Content-Type"].startswith("text/html")
        assert headers["Cache-Control"] == "no-store"
        assert b'name="viewport"' in body
        assert b"Knowledge Atlas" in body
        assert b"NOT QUALIFIED" in body

        code, headers, payload = _get(
            root + "/api/v1/hierarchy/evaluate?vector=%5B1%2C0%5D&top_k=3&leaves=2"
        )
        report = json.loads(payload.decode("utf-8"))
        assert code == 200
        assert headers["Content-Type"].startswith("application/json")
        assert report["status"] == "SYNTHETIC_RESEARCH_ONLY"
        assert report["qualified"] is False
        assert report["training_executed"] is False
        assert report["selection"]["ranking"][0]["fact_id"] == "north"
        assert report["fidelity"]["reference_scored_vectors"] == 5


def test_real_http_refuses_unsafe_methods_and_false_readiness():
    with _ephemeral_server() as root:
        code, _, data = _get(root + "/healthz")
        assert code == 200
        assert json.loads(data)["qualified"] is False
        code, _, data = _get(root + "/readyz")
        assert code == 503
        assert json.loads(data)["status"] == "NOT_PRODUCTION_READY"
        code, _, data = _get(root + "/api/v1/hierarchy/evaluate?vector=%5BNaN%2C0%5D")
        assert code == 400
        assert json.loads(data)["status"] == "REFUSED"
        code, _, data = _get(root + "/", method="POST")
        assert code == 405
        assert json.loads(data)["status"] == "REFUSED"
