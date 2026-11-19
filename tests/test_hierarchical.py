"""Offline adversarial contract tests; no external model/code/network required."""
import pytest

from szl_retrieval_bench.hierarchical import HierarchicalIndex, IndexRefused, VectorFact


DIGEST = "a" * 64


def fact(name, vector, tenant="tenant-a", **overrides):
    fields = dict(fact_id=name, tenant_id=tenant, source_uri="https://example.org/source",
                  source_revision="immutable-001", source_sha256=DIGEST, license_id="MIT",
                  permission_scopes=("retrieval_evaluation",), integrity="PASS",
                  freshness="PASS", derivation="PASS", rights="PASS", vector=tuple(vector))
    fields.update(overrides)
    return VectorFact(**fields)


def sample():
    return [fact("a", [1, 0]), fact("b", [0.9, 0.1]),
            fact("c", [-1, 0]), fact("d", [0, 1])]


def test_offline_pinned_index_is_deterministic():
    left = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    right = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    assert left.snapshot_sha256 == right.snapshot_sha256
    assert left.select([1, 0], tenant_id="tenant-a", max_leaf_visits=4) == right.select(
        [1, 0], tenant_id="tenant-a", max_leaf_visits=4)


def test_full_scan_nearest_neighbor_and_audit_counts():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    result = idx.select([1, 0], tenant_id="tenant-a", max_leaf_visits=idx.leaf_count)
    assert result["state"] == "EVALUATED_OFFLINE"
    assert result["ranking"][0]["fact_id"] == "a"
    assert result["exhaustive"] is True
    assert result["scored_facts"] == result["total_facts"] == 4
    assert result["model_executed"] is False
    assert result["deployment_qualified"] is False
    assert result["no_support_proven"] is False


def test_pruning_avoids_some_exact_scoring_but_does_not_claim_accuracy():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    result = idx.select([1, 0], tenant_id="tenant-a", max_leaf_visits=1)
    assert result["searched_leaves"] == 1
    assert result["scored_facts"] == 1
    assert result["exhaustive"] is False


def test_empty_evidence_yields_abstain_without_claiming_global_absence():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    result = idx.select([1, 0], tenant_id="tenant-a", max_leaf_visits=1,
                        min_similarity=1.0)
    assert result["state"] == "ABSTAINED_OFFLINE"
    assert not result["no_support_proven"]
    assert result["ranking"] == []


def test_query_tenant_mismatch_rejected():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    with pytest.raises(IndexRefused, match="tenant"):
        idx.select([1, 0], tenant_id="tenant-b", max_leaf_visits=1)


def test_cross_tenant_index_fails_closed():
    with pytest.raises(IndexRefused, match="tenant"):
        HierarchicalIndex(sample() + [fact("x", [1, 0], tenant="tenant-b")],
                          tenant_id="tenant-a")


@pytest.mark.parametrize("overrides", [
    {"source_sha256": "unverified"}, {"permission_scopes": ()},
    {"integrity": "UNKNOWN"}, {"rights": "FAIL"}, {"freshness": "UNKNOWN"},
    {"derivation": "FAIL"}, {"license_id": ""}, {"source_revision": ""},
    {"source_uri": ""}, {"vector": (float("nan"), 0)},
    {"vector": (0, 0)}, {"vector": (True, 0)},
])
def test_invalid_sources_and_vectors_refused(overrides):
    with pytest.raises(IndexRefused):
        HierarchicalIndex([fact("a", overrides.get("vector", [1, 0]),
                                **{k: v for k, v in overrides.items() if k != "vector"})],
                          tenant_id="tenant-a")


def test_duplicate_source_ids_refused():
    with pytest.raises(IndexRefused, match="unique"):
        HierarchicalIndex([fact("a", [1, 0]), fact("a", [0, 1])],
                          tenant_id="tenant-a")


def test_query_dimension_and_nonfinite_refused():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    for query in ([1], [float("inf"), 0], [0, 0]):
        with pytest.raises(IndexRefused):
            idx.select(query, tenant_id="tenant-a", max_leaf_visits=1)


def test_invalid_query_caps_refused():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    for bad in (0, -1, True, 1000):
        with pytest.raises(IndexRefused):
            idx.select([1, 0], tenant_id="tenant-a", max_leaf_visits=bad)


def test_mutating_original_input_does_not_change_index():
    values = sample()
    idx = HierarchicalIndex(values, tenant_id="tenant-a", leaf_size=1)
    first = idx.select([1, 0], tenant_id="tenant-a", max_leaf_visits=4)
    values.clear()
    assert first == idx.select([1, 0], tenant_id="tenant-a", max_leaf_visits=4)


def test_antipodal_centroid_is_handled():
    idx = HierarchicalIndex([fact("north", [1, 0]), fact("south", [-1, 0])],
                            tenant_id="tenant-a", leaf_size=1)
    assert idx.select([1, 0], tenant_id="tenant-a", max_leaf_visits=idx.leaf_count)["ranking"][0]["fact_id"] == "north"

from szl_retrieval_bench.hierarchical_eval import compare_exact


def test_paired_fidelity_complete_budget_agrees_with_exact():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    r = compare_exact(idx, {"query-one": (1, 0), "query-two": (0, 1)},
                      tenant_id="tenant-a", top_k=2,
                      max_leaf_visits=idx.leaf_count)
    assert r["state"] == "MEASURED_OFFLINE_PROXY"
    assert r["top1_agreement_fraction"] == 1.0
    assert r["exact_neighbor_retention_fraction"] == 1.0
    assert r["denominators"]["queries"] == 2
    assert r["evaluation_of_answer_quality"] is False


def test_paired_fidelity_rejects_cross_tenant_query():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    with pytest.raises(IndexRefused):
        compare_exact(idx, {"query": (1, 0)}, tenant_id="tenant-b")


def test_paired_fidelity_rejects_empty_or_invalid_queries():
    idx = HierarchicalIndex(sample(), tenant_id="tenant-a", leaf_size=1)
    for queries in ({}, {"q": (1,)}, {"": (1, 0)}, {"q": (float("nan"), 0)}):
        with pytest.raises(IndexRefused):
            compare_exact(idx, queries, tenant_id="tenant-a")


def test_one_fact_index_works_with_default_query_budget():
    idx = HierarchicalIndex([fact("only", [1, 0])], tenant_id="tenant-a")
    result = idx.select([1, 0], tenant_id="tenant-a")
    assert result["ranking"][0]["fact_id"] == "only"
    assert result["exhaustive"] is True
