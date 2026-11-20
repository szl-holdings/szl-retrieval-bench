# SPDX-License-Identifier: Apache-2.0
"""Source-level edge contract only, NOT a deployed Pyodide compatibility test."""
import importlib.util
import json
from pathlib import Path
import sys
import types

from szl_retrieval_bench.hierarchy_web import application


def test_cloudflare_manifest_is_non_public_and_has_no_prod_routes():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "wrangler.atlas.jsonc").read_text("utf-8"))
    assert config["main"] == "src/atlas_worker.py"
    assert "python_workers" in config["compatibility_flags"]
    assert config["workers_dev"] is False
    assert config["routes"] == []
    assert "account_id" not in config
    assert "vars" not in config
    assert config["observability"]["enabled"] is False


def test_worker_entrypoint_delegates_to_exact_wsgi_application(monkeypatch):
    fake = types.ModuleType("workers")
    called = []

    class WsgiStub:
        @staticmethod
        def entrypoint(app):
            called.append(app)
            return object()

    fake.wsgi = WsgiStub
    monkeypatch.setitem(sys.modules, "workers", fake)
    root = Path(__file__).resolve().parents[1]
    entry = root / "src" / "atlas_worker.py"
    spec = importlib.util.spec_from_file_location("atlas_worker_fixture", entry)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert called == [application]
    assert module.Default is not None
    assert not (root / "src" / "szl_retrieval_bench" / "edge_worker.py").exists()
