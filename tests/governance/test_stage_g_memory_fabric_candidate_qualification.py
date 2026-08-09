from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _json(relative: str) -> dict:
    return json.loads(_text(relative))


def test_stage_g_candidate_matrix_contains_every_ratified_role_and_candidate() -> None:
    doc = _text("docs/STAGE_G_MEMORY_FABRIC_CANDIDATE_QUALIFICATION_MATRIX.md")

    for required in (
        "Redis-class",
        "KurrentDB",
        "Kafka",
        "Pulsar",
        "PostgreSQL",
        "Datomic-like fact architecture",
        "FoundationDB-backed custom store",
        "Neo4j",
        "FalkorDB",
        "Memgraph",
        "Amazon Neptune",
        "Qdrant",
        "Milvus",
        "Vespa",
        "Weaviate",
        "pgvector",
        "FAISS",
        "DiskANN",
        "HNSW",
        "S3-compatible / MinIO / NAS",
        "Temporal",
        "Dagster",
        "Prefect",
        "Ray",
        "SQLite",
        "vLLM",
        "PyTorch Distributed / Ray",
        "DeepSpeed",
        "FSDP",
        "Slurm",
        "Kubernetes",
        "MLflow-class / object-backed registry",
    ):
        assert required in doc

    assert "32 required named role assignments representing 31 unique concrete product/library candidates" in doc
    assert "All concrete named candidates above are mandatory Stage G evaluation targets" in doc
    assert "A family placeholder never counts as passed without a concrete runnable implementation." in doc


def test_candidate_policy_requires_every_named_candidate_and_family_resolution() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    assert policy["policy_id"] == "stage_g_memory_fabric_candidate_qualification"
    assert policy["status"] == "owner_ratified"
    assert policy["capability_ceiling"] is False
    assert "capability_state_semantics" not in policy
    assert policy["research_allowed"] is True

    roles = {item["role_id"]: item for item in policy["candidate_roles"]}
    assert len(roles) == 11

    assignments = [
        candidate
        for role in roles.values()
        for candidate in role["required_named_candidates"]
    ]
    assert len(assignments) == 32
    assert len(set(assignments)) == 31

    assert {"kurrentdb", "kafka", "pulsar"} == set(
        roles["experience_event_fabric"]["required_named_candidates"]
    )
    assert {"neo4j", "falkordb", "memgraph", "amazon_neptune"} == set(
        roles["cognitive_graph"]["required_named_candidates"]
    )
    assert {"qdrant", "milvus", "vespa", "weaviate", "pgvector", "faiss", "diskann", "hnsw"} == set(
        roles["vector_multimodal"]["required_named_candidates"]
    )
    assert {"temporal", "dagster", "prefect", "ray"} == set(
        roles["durable_workflow"]["required_named_candidates"]
    )
    assert {"pytorch_distributed", "ray", "deepspeed", "fsdp", "slurm", "kubernetes"} == set(
        roles["training"]["required_named_candidates"]
    )

    family_count = sum(
        len(role["implementation_family_candidates"])
        for role in roles.values()
    )
    assert family_count == 17
    assert policy["implementation_family_rule"]["may_be_marked_passed_without_concrete_runnable_implementation"] is False
    assert policy["required_named_candidate_rule"]["all_listed_required_named_candidates_must_be_evaluated"] is True
    assert policy["required_named_candidate_rule"]["may_be_removed_without_separate_owner_ratified_amendment"] is False


def test_stage_g_requires_individual_same_role_cross_role_multiplane_and_full_fabric_tests() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    levels = policy["qualification_levels"]
    close = policy["stage_g_closure"]

    assert levels["q0_registration_reproducibility"] is True
    assert levels["q1_individual_candidate_stress"] is True
    assert levels["q2_same_role_pairwise"] == "all_concrete_same_role_pairs"
    assert levels["q3_cross_role_pairwise"] == "all_candidate_substitutions_on_registered_interaction_edges"
    assert levels["q4_multi_plane_combinations"] is True
    assert levels["q5_full_end_to_end_cognitive_memory"] is True
    assert levels["failure_deletion_restore_concurrency_rebuild_rollback_scale"] is True
    assert levels["readme_governance_promise_coverage"] is True

    assert len(policy["interaction_edges"]) == 23
    assert close["requires_all_applicable_candidate_qualification_levels"] is True
    assert close["requires_all_same_role_pairwise_coverage"] is True
    assert close["requires_all_registered_cross_role_interaction_edges"] is True
    assert close["requires_multi_plane_combinatorial_coverage"] is True
    assert close["requires_full_fabric_end_to_end_runs"] is True
    assert close["requires_zero_unresolved_zero_tolerance_failures"] is True
    assert close["requires_readme_and_governance_promise_matrix"] is True
    assert close["requires_owner_acceptance"] is True
    assert close["stage_h_eligible_before_this_gate_passes"] is False


def test_complex_rayan_life_scale_and_promise_coverage_are_mandatory() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    workload = policy["workload"]

    assert workload["requires_complex_synthetic_rayan_life"] is True
    assert workload["requires_authorized_real_rayan_seed_where_applicable"] is True
    assert workload["requires_elaina_source_person_rayan_host_alice_continuity_separation"] is True
    assert workload["requires_friday_product_isolation"] is True
    assert workload["scale_certification_points"] == [1000, 10000, 100000, 1000000]
    assert workload["scale_certification_points_are_architectural_ceilings"] is False
    assert workload["larger_scale_research_allowed"] is True

    assert "README.md" in policy["promise_sources"]
    assert len(policy["zero_tolerance_classes"]) >= 16


def test_candidate_matrix_is_bound_into_active_architecture_and_migration_gate() -> None:
    architecture = _text("docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md")
    migration = _text("docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md")

    assert "**Version:** 2.1.0" in architecture
    assert "STAGE_G_MEMORY_FABRIC_CANDIDATE_QUALIFICATION_MATRIX.md" in architecture
    assert "every concrete named candidate" in architecture
    assert "same-role all-pairs comparison" in architecture
    assert "candidate substitutions" in architecture
    assert "complex synthetic Rayan-life workload" in architecture
    assert "Stage G cannot close" in architecture

    assert "**Version:** 1.8.0" in migration
    assert "STAGE_G_MEMORY_FABRIC_CANDIDATE_QUALIFICATION_MATRIX.md" in migration
    assert "every concrete named candidate" in migration
    assert "same-role all-pairs" in migration
    assert "cross-role candidate-substitution" in migration
    assert "README/governance promise coverage" in migration
    assert "Stage H remains ineligible" in migration
