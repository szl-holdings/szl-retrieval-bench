"""Blocked NeMo Retriever baseline for the Q7 embedder eval.

This module does not download weights, embed text, or emit a metric.
The measured BM25, TF-IDF, and RRF lanes stay in the harness.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "szl.nemo-retriever-q7-baseline/v0"
LICENSE_OBSERVED_ON = "2026-11-27"
NAMED_OPEN_LICENSE_NAMES = frozenset(
    {
        "nvidia-nemotron-open-model-license",
        "nvidia-open-model-license",
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
    }
)
USD_TO_UUSD = 1_000_000


class BaselineBlocked(Exception):
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        super().__init__(record["reason"])


def baseline_record() -> dict[str, Any]:
    embedder_license = "nvidia-open-model-license"
    reranker_license = "openmdw-1.1"
    return {
        "schema": SCHEMA,
        "state": "BLOCKED",
        "reason": "Q7_CORPUS_UNKNOWN",
        "q7Corpus": "UNKNOWN",
        "metrics": None,
        "launch": False,
        "launchAuthorized": False,
        "launcher": "saywa",
        "launcherImplementation": "UNKNOWN",
        "providers": ["massed", "salad"],
        "caps": {
            "perRunUsd": 100,
            "perDayUsd": 500,
            "perMonthUsd": 5000,
            "perRunUusd": 100 * USD_TO_UUSD,
            "perDayUusd": 500 * USD_TO_UUSD,
            "perMonthUusd": 5000 * USD_TO_UUSD,
        },
        "embedder": {
            "role": "candidate",
            "selectedForScoring": False,
            "hfRepoId": "nvidia/llama-nemotron-embed-1b-v2",
            "hfRevision": "113abe4acafa848e77ead9c0623205e511932348",
            "hfApiLicense": "other",
            "hfApiLicenseName": embedder_license,
            "hfApiLicenseLink": (
                "https://www.nvidia.com/en-us/agreements/enterprise-software/"
                "nvidia-open-model-license/"
            ),
            "licenseObservedOn": LICENSE_OBSERVED_ON,
            "namedOpenLicense": embedder_license in NAMED_OPEN_LICENSE_NAMES,
            "amaruLicenseCheck": "PENDING",
        },
        "reranker": {
            "role": "recorded-not-selected",
            "selectedForScoring": False,
            "hfRepoId": "nvidia/llama-nemotron-rerank-1b-v2",
            "hfRevision": "828765652b05bd439c9789d2a6d093db1caa1443",
            "hfApiLicense": "other",
            "hfApiLicenseName": reranker_license,
            "hfApiLicenseLink": "https://openmdw.ai/license/1-1/",
            "licenseObservedOn": LICENSE_OBSERVED_ON,
            "namedOpenLicense": reranker_license in NAMED_OPEN_LICENSE_NAMES,
            "amaruLicenseCheck": "PENDING",
        },
        "excluded": [
            {
                "hfRepoId": "nvidia/NV-Embed-v2",
                "hfApiLicense": "cc-by-nc-4.0",
                "reason": "NONCOMMERCIAL_LICENSE",
            }
        ],
        "ngcContainer": False,
    }


def rank(query: str, doc_ids: list[str]) -> list[str]:
    """Refuse to rank. The signature matches the dense-lane adapter."""

    del query, doc_ids
    raise BaselineBlocked(baseline_record())
