"""Receipts remain valid JSON regardless of an optional package's presence."""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def load_receipts():
    path = Path(__file__).parents[1] / "src/szl_retrieval_bench/receipts.py"
    spec = importlib.util.spec_from_file_location("isolated_receipts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["local", "shared"])
def receipts(request, monkeypatch):
    if request.param == "local":
        original = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "szl_evidence_core.canonical":
                raise ModuleNotFoundError("optional package absent", name="szl_evidence_core")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", missing)
    else:
        package = types.ModuleType("szl_evidence_core")
        package.__path__ = []
        canonical = types.ModuleType("szl_evidence_core.canonical")
        canonical.CANON_UTF8 = "szl.lambda/v1"

        def shared(value, *, profile, check):
            assert profile == "szl.lambda/v1"
            assert check is False
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False)

        canonical.canonical_json = shared
        monkeypatch.setitem(sys.modules, "szl_evidence_core", package)
        monkeypatch.setitem(sys.modules, "szl_evidence_core.canonical", canonical)
    return load_receipts()


def test_finite_utf8_bytes_remain_stable(receipts):
    assert receipts._canonical({"z": [True, None, -0.0], "a": "Küllinchu ✓ 日本 🚀"}) == (
        '{"a":"Küllinchu ✓ 日本 🚀","z":[true,null,-0.0]}'
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_run_never_enters_receipt_chain(receipts, invalid):
    chain = receipts.ReceiptChain()
    first = chain.emit({"score": 0.5})
    with pytest.raises(ValueError):
        chain.emit({"nested": [{"score": invalid}]})
    assert chain.chain == [first]
    assert chain.verify() is True


def test_incompatible_installed_package_does_not_silently_fall_back(monkeypatch):
    package = types.ModuleType("szl_evidence_core")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "szl_evidence_core", package)
    monkeypatch.setitem(sys.modules, "szl_evidence_core.canonical", None)
    with pytest.raises(ModuleNotFoundError):
        load_receipts()


def test_runtime_import_failure_does_not_silently_fall_back(monkeypatch):
    original = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "szl_evidence_core.canonical":
            raise RuntimeError("broken optional package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(RuntimeError, match="broken optional package"):
        load_receipts()
