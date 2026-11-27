"""The NeMo Retriever Q7 baseline stays blocked and metric-free."""

import json
from pathlib import Path

import pytest

from szl_retrieval_bench import harness, nemo_retriever_baseline as baseline

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "nemo-retriever-q7-baseline.v0.json"


def test_record_matches_spec_and_blocks_without_metrics():
    record = baseline.baseline_record()
    assert json.loads(SPEC.read_text(encoding="utf-8")) == record
    assert record["state"] == "BLOCKED"
    assert record["reason"] == "Q7_CORPUS_UNKNOWN"
    assert record["q7Corpus"] == "UNKNOWN"
    assert record["metrics"] is None
    assert record["launchAuthorized"] is False
    assert record["embedder"]["hfApiLicense"] == "other"
    assert record["embedder"]["hfApiLicenseName"] == "nvidia-open-model-license"
    assert record["embedder"]["amaruLicenseCheck"] == "PENDING"
    assert record["embedder"]["selectedForScoring"] is False
    assert record["reranker"]["hfApiLicenseName"] == "openmdw-1.1"
    assert record["reranker"]["namedOpenLicense"] is False
    assert record["reranker"]["selectedForScoring"] is False
    assert record["excluded"][0]["hfApiLicense"] == "cc-by-nc-4.0"
    assert record["caps"]["perRunUsd"] == 100
    assert record["caps"]["perMonthUsd"] == 5000
    assert record["ngcContainer"] is False


def test_rank_refuses_and_returns_no_ids():
    with pytest.raises(baseline.BaselineBlocked) as raised:
        baseline.rank("query", ["d1", "d2"])
    assert raised.value.record["metrics"] is None
    assert "nDCG" not in raised.value.record
    assert raised.value.record["state"] == "BLOCKED"


def test_harness_does_not_call_the_baseline():
    text = Path(harness.__file__).read_text(encoding="utf-8")
    assert "nemo_retriever" not in text
    module = Path(baseline.__file__).read_text(encoding="utf-8")
    assert "import urllib" not in module
    assert "huggingface" not in module.lower()
