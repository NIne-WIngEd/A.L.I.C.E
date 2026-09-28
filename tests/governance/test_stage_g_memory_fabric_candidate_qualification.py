from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _json(relative: str) -> dict:
    return json.loads(_text(relative))


def test_stage_g_requires_full_fabric_not_reduced_profile() -> None:
    doc = _text("docs/STAGE_G_MEMORY_FABRIC_CANDIDATE_QUALIFICATION_MATRIX.md")

    assert "**Version:** 3.0.0" in doc
    assert "No-capability-reduction rule" in doc
    for required in (
        "Experience/Event history",
        "bitemporal Claim authority",
        "episodic/autobiographical memory",
        "graph/relational memory",
        "associative graph retrieval",
        "vector/multimodal retrieval",
        "source-native retrieval",
        "parametric personal memory",
        "working/activation memory",
        "multi-device/federation semantics",
    ):
        assert required in doc


def test_selected_physical_architecture_matches_full_memory_roles() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")

    assert policy["schema_version"] == "3.0.0"
    assert policy["status"] == "owner_ratified"
    assert policy["capability_ceiling"] is False

    product = policy["product_doctrine"]
    assert product["fable_v1_is_reduced_profile"] is False
    assert product["full_transferable_personal_architecture_required"] is True
    assert product["feature_models_may_be_external"] is True
    assert product["personal_identity_memory_judgment_learning_must_not_be_outsourced"] is True
    assert product["deployment_topology_may_adapt_to_hardware"] is True
    assert product["deployment_adaptation_may_remove_logical_planes"] is False

    stack = policy["selected_architecture"]
    assert stack["experience_event"] == "nats_jetstream_behind_project_owned_evidence_log_contract"
    assert stack["claim_authority"] == "xtdb_v2_bitemporal"
    assert stack["cognitive_graph_local"] == "ladybugdb"
    assert stack["cognitive_graph_scale_out"] == "janusgraph_with_qualified_distributed_storage"
    assert stack["vector_multimodal"] == "qdrant_edge_or_server_cluster"
    assert stack["durable_workflow"] == "temporal"
    assert stack["shared_ephemeral"] == "valkey_with_process_local_l1"


def test_all_required_logical_planes_are_registered() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    planes = set(policy["required_logical_planes"])

    assert {
        "raw_evidence_object",
        "experience_event",
        "claim_authority",
        "episodic_autobiographical",
        "cognitive_multi_graph",
        "associative_graph_compute",
        "vector_multimodal",
        "source_native_live",
        "perceptual_personal_memory",
        "personal_cognitive_models",
        "parametric_personal_memory",
        "working_activation_memory",
        "memory_resource_manager",
        "retrieval_orchestrator",
        "lifecycle_curator",
        "cross_layer_deletion_unlearning",
        "durable_workflows",
        "multi_device_federation",
        "model_dataset_lineage",
    } <= planes


def test_authority_does_not_leak_into_derived_memory() -> None:
    authority = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")["authority_rules"]

    assert authority["event_is_evidence_not_automatically_claim"] is True
    assert authority["claim_authority_is_explicit"] is True
    assert authority["episodes_are_derived"] is True
    assert authority["graph_is_authority"] is False
    assert authority["vector_is_authority"] is False
    assert authority["retrieval_scores_are_source_evidence"] is False
    assert authority["procedural_memory_is_factual_authority"] is False
    assert authority["cache_is_authority"] is False
    assert authority["parametric_memory_is_sole_history_authority"] is False


def test_formation_and_retrieval_are_open_not_fixed_taxonomies_or_routes() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")

    formation = policy["formation"]
    assert formation["fast_path_required"] is True
    assert formation["slow_path_required"] is True
    assert formation["dynamic_overlapping_scene_domains"] is True
    assert formation["fixed_life_scene_taxonomy_is_product_ceiling"] is False
    assert formation["learned_episode_boundaries"] is True

    retrieval = policy["retrieval"]
    for key in (
        "exact",
        "source_native",
        "episodic",
        "graph_multi_view",
        "associative_activation",
        "vector_dense_sparse_multivector",
        "multimodal",
        "procedural",
        "personal_state",
        "mission",
        "live_source",
        "parallel_and_iterative_planning",
        "evidence_consumption_receipts",
        "memory_use_calibration",
    ):
        assert retrieval[key] is True
    assert retrieval["fixed_global_retrieval_order_required"] is False


def test_frontier_derived_invariants_are_not_optional() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    required = set(policy["required_invariants"])

    assert {
        "evidence_consumption_citation_lock",
        "consolidation_path_dependence",
        "memory_use_calibration",
        "dynamic_retrieval_routing",
        "usage_accessibility_projection_only",
        "procedural_memory_not_current_fact",
        "execution_state_deletion",
        "parameter_memory_backflow_prevention",
        "source_host_relationship_self_identity_separation",
        "personal_development_causal_interventions",
        "multi_device_federation",
        "product_host_isolation",
        "restore_rebuild_rollback",
    } <= required


def test_scale_points_are_not_caps() -> None:
    scale = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")["scale_policy"]

    assert scale["certification_points"] == [
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
    ]
    assert scale["points_are_architectural_ceilings"] is False
    assert scale["continue_beyond_points_when_needed"] is True
    assert scale["hardware_profile_is_not_capability_profile"] is True


def test_challengers_are_decision_triggered_not_tournaments() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    doctrine = policy["execution_doctrine"]

    assert doctrine["infrastructure_tournament_required"] is False
    assert doctrine["all_pairs_required"] is False
    assert doctrine["cartesian_candidate_substitution_required"] is False
    assert doctrine["challengers_require_real_decision_trigger"] is True
    assert doctrine["validation_must_protect_or_change_real_decision"] is True
    assert doctrine["mc10_style_overvalidation_prohibited"] is True

    assert {
        "logical_contract_failure",
        "observed_capability_or_fidelity_gap",
        "measured_scale_latency_resource_blocker",
        "recovery_deletion_privacy_or_custody_blocker",
        "licensing_or_distribution_blocker",
        "strong_frontier_evidence_exposes_missing_mechanism",
    } <= set(policy["challenger_triggers"])


def test_stage_g_closure_requires_complete_personal_loop() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    close = policy["stage_g_closure"]

    for key in (
        "requires_all_logical_planes",
        "requires_selected_implementation_contracts",
        "requires_full_fabric_end_to_end",
        "requires_fast_and_slow_formation",
        "requires_multi_graph_associative_vector_source_native_retrieval",
        "requires_multimodal_personal_memory",
        "requires_cross_layer_deletion_unlearning",
        "requires_personal_development_causal_tests",
        "requires_multi_device_federation",
        "requires_realistic_scale_resource_evidence",
        "requires_restore_rebuild_rollback",
        "requires_zero_unresolved_zero_tolerance_failures",
        "requires_exact_versions_hashes_configs_receipts",
        "requires_owner_acceptance",
    ):
        assert close[key] is True

    assert close["stage_h_eligible_before_gate_passes"] is False
    assert len(policy["zero_tolerance_classes"]) >= 19


def test_execution_map_rejects_lighter_fable_and_backend_simplification() -> None:
    execution = _text("docs/ALICE_PHASE2_REPLACEMENT_AND_FABLE_V1_EXECUTION_PLAN_2026-09-27.md")

    assert "**Version:** 2.0.0" in execution
    assert "Fable v1 is **not** a reduced, lightweight" in execution
    assert "Only replaceable **feature-model work** may be delegated" in execution
    assert "NATS JetStream" in execution
    assert "XTDB v2" in execution
    assert "LadybugDB" in execution
    assert "JanusGraph" in execution
    assert "Qdrant" in execution
    assert "Memory Resource Manager / Scheduler" in execution
    assert "cross-layer influence operation" in execution
    assert "No \"small memory demo,\"" in execution
