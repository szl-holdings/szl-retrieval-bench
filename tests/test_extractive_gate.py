# SPDX-License-Identifier: Apache-2.0
"""Red-first tests: strict extractive-only citation guard; synthetic source text."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from szl_retrieval_bench.extractive_gate import evaluate_packet, parse_packet_bytes

SCHEMA = "szl.retrieval-bench.extractive-citation/v1"
SOURCE = "Synthetic ledger.\nFlux measured: 8 units.\n"
CLAIM = "Flux measured: 8 units."


def sample():
    start = len(SOURCE[: SOURCE.index(CLAIM)].encode("utf-8"))
    return {
        "schema": SCHEMA,
        "sources": [
            {
                "source_id": "synthetic:doc1",
                "text": SOURCE,
                "sha256": hashlib.sha256(SOURCE.encode("utf-8")).hexdigest(),
            }
        ],
        "claims": [
            {
                "claim_id": "claim1",
                "text": CLAIM,
                "citation": {
                    "source_id": "synthetic:doc1",
                    "start_byte": start,
                    "end_byte": start + len(CLAIM.encode("utf-8")),
                },
            }
        ],
    }


def state(packet):
    return evaluate_packet(packet)["state"]


def test_exact_extractive_citation_is_only_a_byte_provenance_match():
    result = evaluate_packet(sample())
    assert result["state"] == "EXTRACTIVE_BYTES_MATCH"
    assert result["claims_count"] == 1
    assert result["semantic_entailment_verified"] is False
    assert result["source_identity_verified"] is False
    assert result["production_authorized"] is False
    assert result["model_hallucination_eliminated"] is False
    assert len(result["packet_sha256"]) == 64


def test_unknown_artifact_id_is_refused_even_with_exact_text():
    p = sample()
    p["claims"][0]["citation"]["source_id"] = "invented:receipt42"
    assert evaluate_packet(p)["reason"] == "UNKNOWN_SOURCE_ID"


def test_unpinned_or_changed_source_is_refused():
    p = sample()
    p["sources"][0]["text"] += "injected content"
    assert evaluate_packet(p)["reason"] == "SOURCE_DIGEST_MISMATCH"


def test_paraphrase_or_invented_number_is_refused_even_with_valid_citation():
    p = sample()
    p["claims"][0]["text"] = "Flux measured: 80 units."
    assert evaluate_packet(p)["reason"] == "EXTRACTIVE_TEXT_MISMATCH"


def test_citation_offsets_do_not_accept_incorrect_ranges():
    p = sample()
    p["claims"][0]["citation"]["end_byte"] += 100
    assert evaluate_packet(p)["reason"] == "SPAN_BOUNDS_INVALID"
    p = sample()
    p["claims"][0]["citation"]["start_byte"] = True
    assert evaluate_packet(p)["reason"] == "SPAN_BOUNDS_INVALID"
    p = sample()
    p["claims"][0]["citation"]["end_byte"] = -1
    assert evaluate_packet(p)["reason"] == "SPAN_BOUNDS_INVALID"


def test_unicode_offsets_are_utf8_bytes_not_codepoints():
    text = "Ecuador: ☀ evidence."
    claim = "☀ evidence"
    p = sample()
    p["sources"][0]["text"] = text
    p["sources"][0]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    p["claims"][0]["text"] = claim
    start = len(b"Ecuador: ")
    p["claims"][0]["citation"]["start_byte"] = start
    p["claims"][0]["citation"]["end_byte"] = start + len(claim.encode())
    assert state(p) == "EXTRACTIVE_BYTES_MATCH"
    p["claims"][0]["citation"]["start_byte"] += 1
    assert evaluate_packet(p)["reason"] == "SPAN_NOT_UTF8_OR_TEXT_MISMATCH"


def test_no_evidence_is_abstain_not_zero_or_success():
    p = {"schema": SCHEMA, "sources": [], "claims": []}
    out = evaluate_packet(p)
    assert out["state"] == "ABSTAIN"
    assert out["claims_count"] is None
    assert out["reason"] == "NO_EVIDENCE"
    assert out["production_authorized"] is False


def test_absent_evidence_counterfactual_refuses_claim_from_present_branch():
    with_evidence = sample()
    without_evidence = deepcopy(with_evidence)
    without_evidence["sources"] = []
    assert state(with_evidence) == "EXTRACTIVE_BYTES_MATCH"
    out = evaluate_packet(without_evidence)
    assert out["state"] == "REFUSED"
    assert out["reason"] == "NO_EVIDENCE"
    assert out["claims_count"] is None


def test_duplicate_ids_are_invalid_and_can_not_be_last_writer_wins():
    p = sample()
    p["sources"].append(deepcopy(p["sources"][0]))
    assert evaluate_packet(p)["reason"] == "DUPLICATE_SOURCE_ID"
    p = sample()
    p["claims"].append(deepcopy(p["claims"][0]))
    assert evaluate_packet(p)["reason"] == "DUPLICATE_CLAIM_ID"


def test_partial_validity_never_returns_partial_positive_result():
    p = sample()
    p["claims"].append(
        {
            "claim_id": "claim2",
            "text": "Fake result",
            "citation": {"source_id": "not-existing", "start_byte": 0, "end_byte": 10},
        }
    )
    out = evaluate_packet(p)
    assert out["state"] == "REFUSED"
    assert out["claims_count"] is None


def test_extra_authority_fields_are_rejected():
    p = sample()
    p["production_ready"] = True
    assert evaluate_packet(p)["state"] == "INVALID"
    p = sample()
    p["claims"][0]["signed"] = True
    assert evaluate_packet(p)["state"] == "INVALID"


def test_empty_claims_are_abstain_with_known_evidence():
    p = sample()
    p["claims"] = []
    out = evaluate_packet(p)
    assert out["state"] == "ABSTAIN"
    assert out["reason"] == "NO_CLAIMS"
    assert out["claims_count"] is None


def test_limits_and_nonfinite_json_fail_closed():
    p = sample()
    p["sources"][0]["text"] = "x" * 100_000
    assert evaluate_packet(p)["state"] == "INVALID"
    raw = json.dumps(sample()).replace('"start_byte": ', '"start_byte": NaN, "extra": ')
    try:
        parse_packet_bytes(raw.encode())
        assert False, "nonfinite JSON should fail"
    except ValueError:
        pass


def test_duplicate_raw_json_keys_are_rejected():
    raw = b'{"schema":"one","schema":"two","sources":[],"claims":[]}'
    try:
        parse_packet_bytes(raw)
        assert False, "duplicate key accepted"
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()


def test_direct_python_caller_cannot_bypass_96k_packet_limit():
    p = sample()
    excerpt = "X" * 2000
    p["sources"][0]["text"] = excerpt
    p["sources"][0]["sha256"] = hashlib.sha256(excerpt.encode()).hexdigest()
    p["claims"] = []
    for index in range(48):
        p["claims"].append(
            {
                "claim_id": f"c{index}",
                "text": excerpt,
                "citation": {
                    "source_id": "synthetic:doc1",
                    "start_byte": 0,
                    "end_byte": 2000,
                },
            }
        )
    encoded = json.dumps(
        p, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert len(encoded) > 96 * 1024
    out = evaluate_packet(p)
    assert out["state"] == "INVALID"
    assert out["reason"] == "PACKET_BYTES_LIMIT_EXCEEDED"
    assert out["claims_count"] is None


def test_input_packet_is_not_mutated_and_deterministic_digest():
    p = sample()
    before = deepcopy(p)
    a = evaluate_packet(p)
    b = evaluate_packet(p)
    assert a == b
    assert p == before
