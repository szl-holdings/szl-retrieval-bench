# SPDX-License-Identifier: Apache-2.0
"""Offline, pure-Python HiKVP-inspired hierarchical vector-selection experiment.

NOT AtlasKV attention, a trained encoder, verified rights, deployment, or proof
of sublinear worst-case runtime. Callers supply already-admitted vector facts;
the explicit scope/rights declarations are validated but not authenticated.
This module never loads checkpoints, performs network calls, or trains models.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import re
from typing import Sequence

SCHEMA = "szl.retrieval-bench.hierarchical-offline/v1"
MAX_FACTS = 2048
MAX_DIMS = 1024
MAX_NODES_VISITED = 8192
MAX_FACT_METADATA_BYTES = 8192
MAX_PERMISSION_SCOPES = 16
SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")


class IndexRefused(ValueError):
    """Malformed or out-of-scope information must not enter an index."""


@dataclass(frozen=True)
class VectorFact:
    fact_id: str
    tenant_id: str
    source_uri: str
    source_revision: str
    source_sha256: str
    license_id: str
    permission_scopes: tuple[str, ...]
    integrity: str
    freshness: str
    derivation: str
    rights: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class _Node:
    centroid: tuple[float, ...]
    children: tuple[int, ...]
    facts: tuple[int, ...]


def _unit(vector: Sequence[float], dims: int | None = None) -> tuple[float, ...]:
    if not isinstance(vector, (tuple, list)) or not 1 <= len(vector) <= MAX_DIMS:
        raise IndexRefused("vector must be a bounded nonempty list or tuple")
    if dims is not None and len(vector) != dims:
        raise IndexRefused("vector dimension mismatch")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
        raise IndexRefused("vector values must be finite real numbers")
    norm = math.hypot(*vector)
    if not norm or not math.isfinite(norm):
        raise IndexRefused("vector must have a finite positive norm")
    return tuple(float(v / norm) for v in vector)


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sha_json(payload: object) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class HierarchicalIndex:
    """Immutable single-tenant offline index; best-first centroid pruning.

    All facts must be scoped to the same tenant and caller-declared admitted.
    A runtime must validate actual rights and provenance upstream. Never feed
    a cross-tenant or unverified source mixture to this research component.
    """

    def __init__(self, facts: Sequence[VectorFact], *, tenant_id: str,
                 leaf_size: int = 8) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise IndexRefused("explicit tenant_id required")
        if type(leaf_size) is not int or not 1 <= leaf_size <= 128:
            raise IndexRefused("leaf_size must be 1..128")
        if not isinstance(facts, (tuple, list)) or not 1 <= len(facts) <= MAX_FACTS:
            raise IndexRefused("index must contain 1..2048 bounded facts")
        ids: set[str] = set()
        copy: list[VectorFact] = []
        vectors: list[tuple[float, ...]] = []
        dims: int | None = None
        for fact in facts:
            if not isinstance(fact, VectorFact):
                raise IndexRefused("all facts must be VectorFact objects")
            if not isinstance(fact.fact_id, str) or not fact.fact_id or fact.fact_id in ids:
                raise IndexRefused("fact IDs must be nonempty and unique")
            ids.add(fact.fact_id)
            if fact.tenant_id != tenant_id:
                raise IndexRefused("mixed-tenant or out-of-scope fact refused")
            if any(not isinstance(value, str) or not value.strip() for value in
                   (fact.source_uri, fact.source_revision, fact.license_id)):
                raise IndexRefused("source and license identity required")
            if not isinstance(fact.source_sha256, str) or not SHA256_RE.fullmatch(fact.source_sha256):
                raise IndexRefused("source SHA-256 must be lowercase hex")
            # Prevent caller-declared, untrusted metadata from bypassing the
            # index's vector/fact bounds during snapshot JSON serialization.
            # This is a resource bound, NOT a source-rights or hash check.
            if (not isinstance(fact.permission_scopes, tuple) or
                    len(fact.permission_scopes) > MAX_PERMISSION_SCOPES or
                    "retrieval_evaluation" not in fact.permission_scopes or
                    any(
                        not isinstance(scope, str)
                        for scope in fact.permission_scopes
                    )):
                raise IndexRefused("bounded retrieval_evaluation scopes required")
            # Count the actual JSON-escaped UTF-8 string representation.
            # Raw bytes underestimate NUL, newline, quote and backslash data
            # before _sha_json serializes the snapshot. Each field's encoded
            # length is a lower bound on its serialized JSON representation.
            metadata_values = (
                fact.fact_id, fact.tenant_id, fact.source_uri,
                fact.source_revision, fact.source_sha256, fact.license_id,
                *fact.permission_scopes,
            )
            declared_bytes = 0
            try:
                for value in metadata_values:
                    if len(value) > MAX_FACT_METADATA_BYTES:
                        raise IndexRefused("fact metadata JSON byte budget exceeded")
                    declared_bytes += len(json.dumps(
                        value, ensure_ascii=False, allow_nan=False
                    ).encode("utf-8"))
                    if declared_bytes > MAX_FACT_METADATA_BYTES:
                        raise IndexRefused("fact metadata JSON byte budget exceeded")
            except UnicodeEncodeError as exc:
                raise IndexRefused("metadata must be valid UTF-8") from exc
            if any(getattr(fact, axis) != "PASS" for axis in
                   ("integrity", "freshness", "derivation", "rights")):
                raise IndexRefused("four caller-declared PASS axes required")
            vector = _unit(fact.vector, dims)
            dims = len(vector)
            vectors.append(vector)
            copy.append(fact)
        self.tenant_id = tenant_id
        self._facts = tuple(copy)
        self._vectors = tuple(vectors)
        self.dimensions = dims
        self.leaf_size = leaf_size
        self.snapshot_sha256 = _sha_json({
            "schema": SCHEMA, "tenant_id": tenant_id, "leaf_size": leaf_size,
            "facts": [
                {"id": f.fact_id, "source": f.source_uri, "revision": f.source_revision,
                 "sha256": f.source_sha256, "license": f.license_id,
                 "scopes": f.permission_scopes, "axes": [f.integrity, f.freshness,
                 f.derivation, f.rights], "vector": self._vectors[i]}
                for i, f in enumerate(self._facts)
            ],
        })
        self._nodes: list[_Node] = []
        self._root = self._build(tuple(range(len(copy))))
        self.leaf_count = sum(not n.children for n in self._nodes)

    def _center(self, indices: tuple[int, ...]) -> tuple[float, ...]:
        accum = [0.0] * self.dimensions
        for idx in indices:
            for j, x in enumerate(self._vectors[idx]):
                accum[j] += x
        # Antipodal vectors have zero centroid; deterministic first-point fallback.
        if math.hypot(*accum) < 1e-12:
            return self._vectors[indices[0]]
        return _unit(accum)

    def _build(self, members: tuple[int, ...]) -> int:
        center = self._center(members)
        node_id = len(self._nodes)
        self._nodes.append(_Node(center, (), members))
        if len(members) <= self.leaf_size:
            return node_id
        a = members[0]
        b = min(members[1:], key=lambda i: (_dot(self._vectors[a], self._vectors[i]), i))
        left, right = [a], [b]
        for i in members:
            if i in (a, b):
                continue
            (left if _dot(self._vectors[i], self._vectors[a]) >=
             _dot(self._vectors[i], self._vectors[b]) else right).append(i)
        # Bound degenerate tree depth without learning labels or touching qrels.
        if max(len(left), len(right)) > math.ceil(0.80 * len(members)):
            left, right = list(members[::2]), list(members[1::2])
        children = (self._build(tuple(left)), self._build(tuple(right)))
        self._nodes[node_id] = _Node(center, children, ())
        return node_id

    def select(self, query: Sequence[float], *, tenant_id: str, top_k: int = 10,
               max_leaf_visits: int = 1, min_similarity: float = 0.0) -> dict:
        if tenant_id != self.tenant_id:
            raise IndexRefused("query tenant does not match index tenant")
        if (type(top_k) is not int or not 1 <= top_k <= 100 or
                type(max_leaf_visits) is not int or
                not 1 <= max_leaf_visits <= self.leaf_count or
                type(min_similarity) not in (float, int) or
                not math.isfinite(min_similarity) or
                not -1 <= min_similarity <= 1):
            raise IndexRefused("invalid or unbounded query selection limits")
        q = _unit(query, self.dimensions)
        frontier = [(-_dot(q, self._nodes[self._root].centroid), self._root)]
        leaves = 0
        visited_nodes = 0
        candidates: list[tuple[float, str]] = []
        scored_facts = 0
        while frontier and leaves < max_leaf_visits and visited_nodes < MAX_NODES_VISITED:
            _, node_id = heapq.heappop(frontier)
            visited_nodes += 1
            node = self._nodes[node_id]
            if node.children:
                for child_id in node.children:
                    child = self._nodes[child_id]
                    heapq.heappush(frontier, (-_dot(q, child.centroid), child_id))
            else:
                leaves += 1
                scored_facts += len(node.facts)
                for idx in node.facts:
                    score = _dot(q, self._vectors[idx])
                    if score > min_similarity:
                        candidates.append((score, self._facts[idx].fact_id))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        exhausted = not frontier
        return {
            "schema": SCHEMA,
            "state": "EVALUATED_OFFLINE" if candidates else "ABSTAINED_OFFLINE",
            "observation_authority": "CALLER_DECLARED_UNVERIFIED",
            "model_executed": False,
            "training_executed": False,
            "deployment_qualified": False,
            "tenant_id": tenant_id,
            "index_sha256": self.snapshot_sha256,
            "query_sha256": _sha_json(q),
            "ranking": [{"fact_id": fact_id, "cosine": round(score, 8)}
                        for score, fact_id in candidates[:top_k]],
            "searched_leaves": leaves,
            "total_leaves": self.leaf_count,
            "scored_facts": scored_facts,
            "total_facts": len(self._facts),
            "visited_nodes": visited_nodes,
            "exhaustive": exhausted,
            "no_support_proven": False,
            "limits": "Offline vector-selection proxy only; no rights verification, no GPU/LLM execution; absence within a pruned budget is not evidence of global absence.",
        }
