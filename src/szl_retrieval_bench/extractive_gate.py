# SPDX-License-Identifier: Apache-2.0
"""Offline extractive-byte citation gate for caller-declared synthetic snapshots.

This deliberately does NOT verify factual truth, source identity, signer trust,
model quality or semantic entailment. A matching quote only proves that a
proposed answer is byte-exact text of a caller-supplied, hash-consistent source.
No network access, LLM inference, retrieval, training or deployment occurs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

SCHEMA = "szl.retrieval-bench.extractive-citation/v1"
MAX_PACKET_BYTES = 96 * 1024
MAX_SOURCES = 32
MAX_CLAIMS = 48
MAX_SOURCE_BYTES = 8192
MAX_CLAIM_BYTES = 2048
MAX_TOTAL_SOURCE_BYTES = 65536
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_REQUIRED_PACKET = frozenset({"schema", "sources", "claims"})
_REQUIRED_SOURCE = frozenset({"source_id", "text", "sha256"})
_REQUIRED_CLAIM = frozenset({"claim_id", "text", "citation"})
_REQUIRED_CITATION = frozenset({"source_id", "start_byte", "end_byte"})


def _result(state: str, reason: str, **counts: Any) -> dict[str, Any]:
    """Return unavailable denominators as null, never as observed zero."""
    return {
        "schema": SCHEMA,
        "state": state,
        "reason": reason,
        "claims_count": counts.get("claims_count"),
        "source_count": counts.get("source_count"),
        "packet_sha256": counts.get("packet_sha256"),
        "authority": "CALLER_DECLARED_BYTES_ONLY",
        "source_identity_verified": False,
        "signed_receipt_verified": False,
        "semantic_entailment_verified": False,
        "production_authorized": False,
        "training_authorized": False,
        "model_hallucination_eliminated": False,
    }


def _valid_id(value: Any) -> bool:
    return type(value) is str and _IDENT.fullmatch(value) is not None


def _utf8_bytes(value: Any, limit: int) -> bytes | None:
    if type(value) is not str:
        return None
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return None
    if not raw or len(raw) > limit:
        return None
    return raw


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_packet_sha(packet: dict) -> str | None:
    """Stable Python-local JSON checksum, not JCS, DSSE, or source authentication."""
    data = json.dumps(
        packet,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    if len(data) > MAX_PACKET_BYTES:
        return None
    return _sha256(data)


def evaluate_packet(packet: Any) -> dict[str, Any]:
    """Check all extractive claim spans against hash-consistent declared texts.

    A positive verdict is deliberately narrow: a caller-declared exact byte
    match. This cannot certify a model's answer, a source's authority, or a
    source's semantic support of a surrounding question.
    """
    if type(packet) is not dict or set(packet) != _REQUIRED_PACKET:
        return _result("INVALID", "PACKET_FIELDS_INVALID")
    if packet["schema"] != SCHEMA:
        return _result("INVALID", "SCHEMA_UNSUPPORTED")
    sources, claims = packet["sources"], packet["claims"]
    if type(sources) is not list or type(claims) is not list:
        return _result("INVALID", "COLLECTION_TYPE_INVALID")
    if len(sources) > MAX_SOURCES or len(claims) > MAX_CLAIMS:
        return _result("INVALID", "COLLECTION_LIMIT_EXCEEDED")

    # No model or arbitrary source provider may smuggle extra authority fields.
    registry: dict[str, bytes] = {}
    total_size = 0
    for source in sources:
        if type(source) is not dict or set(source) != _REQUIRED_SOURCE:
            return _result("INVALID", "SOURCE_FIELDS_INVALID")
        source_id = source["source_id"]
        if not _valid_id(source_id):
            return _result("INVALID", "SOURCE_ID_INVALID")
        if source_id in registry:
            return _result("INVALID", "DUPLICATE_SOURCE_ID")
        raw = _utf8_bytes(source["text"], MAX_SOURCE_BYTES)
        if raw is None:
            return _result("INVALID", "SOURCE_TEXT_INVALID")
        total_size += len(raw)
        if total_size > MAX_TOTAL_SOURCE_BYTES:
            return _result("INVALID", "SOURCE_BYTES_LIMIT_EXCEEDED")
        digest = source["sha256"]
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            return _result("INVALID", "SOURCE_DIGEST_INVALID")
        if _sha256(raw) != digest:
            return _result("REFUSED", "SOURCE_DIGEST_MISMATCH")
        registry[source_id] = raw

    if not sources:
        return _result("REFUSED" if claims else "ABSTAIN", "NO_EVIDENCE")
    if not claims:
        return _result("ABSTAIN", "NO_CLAIMS")

    seen_claims: set[str] = set()
    for claim in claims:
        if type(claim) is not dict or set(claim) != _REQUIRED_CLAIM:
            return _result("INVALID", "CLAIM_FIELDS_INVALID")
        claim_id = claim["claim_id"]
        if not _valid_id(claim_id):
            return _result("INVALID", "CLAIM_ID_INVALID")
        if claim_id in seen_claims:
            return _result("INVALID", "DUPLICATE_CLAIM_ID")
        seen_claims.add(claim_id)
        claim_raw = _utf8_bytes(claim["text"], MAX_CLAIM_BYTES)
        if claim_raw is None:
            return _result("INVALID", "CLAIM_TEXT_INVALID")
        citation = claim["citation"]
        if type(citation) is not dict or set(citation) != _REQUIRED_CITATION:
            return _result("INVALID", "CITATION_FIELDS_INVALID")
        source_id = citation["source_id"]
        if not _valid_id(source_id):
            return _result("INVALID", "CITATION_ID_INVALID")
        if source_id not in registry:
            return _result("REFUSED", "UNKNOWN_SOURCE_ID")
        start, end = citation["start_byte"], citation["end_byte"]
        source_raw = registry[source_id]
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > len(source_raw)
        ):
            return _result("REFUSED", "SPAN_BOUNDS_INVALID")
        excerpt = source_raw[start:end]
        try:
            excerpt.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return _result("REFUSED", "SPAN_NOT_UTF8_OR_TEXT_MISMATCH")
        if excerpt != claim_raw:
            return _result("REFUSED", "EXTRACTIVE_TEXT_MISMATCH")

    # Direct Python callers must meet the same aggregate budget as JSON users.
    packet_digest = _canonical_packet_sha(packet)
    if packet_digest is None:
        return _result("INVALID", "PACKET_BYTES_LIMIT_EXCEEDED")
    # Neither source authentication nor evidence truth follows from this match.
    return _result(
        "EXTRACTIVE_BYTES_MATCH",
        "ALL_CITED_CLAIMS_EXACTLY_EXTRACTED_FROM_DECLARED_SNAPSHOTS",
        claims_count=len(claims),
        source_count=len(registry),
        packet_sha256=packet_digest,
    )


def parse_packet_bytes(data: bytes) -> dict:
    """Strict bounded wire parser; duplicate keys and nonfinite values refuse."""
    if type(data) is not bytes or not 1 <= len(data) <= MAX_PACKET_BYTES:
        raise ValueError("packet bytes missing or over limit")

    def pairs(values: list[tuple[str, Any]]) -> dict:
        obj: dict[str, Any] = {}
        for key, value in values:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    def refuse_constant(value: str) -> None:
        raise ValueError("nonfinite JSON constant " + value)

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result

    try:
        packet = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=refuse_constant,
            parse_float=finite_float,
        )
    except (UnicodeDecodeError, RecursionError) as exc:
        raise ValueError("invalid packet encoding or nesting") from exc
    if type(packet) is not dict:
        raise ValueError("packet must be an object")
    return packet
