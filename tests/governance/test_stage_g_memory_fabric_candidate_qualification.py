from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _json(relative: str) -> dict:
    return json.loads(_text(relative))


def test_stage_g_uses_selected_stack_not_backend_tournament() -> None:
    doc = _text("docs/STAGE_G_MEMORY_FABRIC_CANDIDATE_QUALIFICATION_MATRIX.md")

    assert "**Version:** 2.0.0" in doc
    assert "selected default physical path" in doc
    assert "PostgreSQL append-only event/experience schema" in doc
    assert "PostgreSQL bitemporal claim schema" in doc
    assert "Neo4j derived projection" in doc
    assert "Qdrant derived projection" in doc
    assert "content-addressed local filesystem/object store" in doc
    assert "Temporal" in doc
    assert "Valkey" in doc

    assert "Stage G no longer requires:" in doc
    assert "every known backend to run" in doc
    assert "all same-role candidate pairs" in doc
    assert "all cross-role candidate substitutions" in doc
    assert "full Cartesian or broad combinatorial backend coverage" in doc


def test_stage_g_policy_ratifies_selected_stack_and_bounded_validation() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")

    assert policy["policy_id"] == "stage_g_memory_fabric_candidate_qualification"
    assert policy["schema_version"] == "2.0.0"
    assert policy["status"] == "owner_ratified"
    assert policy["capability_ceiling"] is False
    assert policy["research_allowed"] is True

    doctrine = policy["execution_doctrine"]
    assert doctrine["selected_stack_first"] is True
    assert doctrine["infrastructure_tournament_required"] is False
    assert doctrine["challengers_are_triggered_by_decisions"] is True
    assert doctrine["validation_must_change_or_protect_a_real_decision"] is True
    assert doctrine["mc10_style_overvalidation_prohibited"] is True

    stack = policy["selected_stack"]
    assert stack["experience_event_fabric"] == "postgresql_append_only_event_schema"
    assert stack["claim_fabric"] == "postgresql_bitemporal_claim_schema"
    assert stack["cognitive_graph"] == "neo4j_derived_projection"
    assert stack["vector_multimodal"] == "qdrant_derived_projection"
    assert "filesystem_sql_fts_metadata_api_live_source" == stack["exact_source_native_retrieval"]


def test_stage_g_keeps_authority_and_cognitive_invariants() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")

    authority = policy["authority_rules"]
    assert authority["graph_is_authority"] is False
    assert authority["vector_is_authority"] is False
    assert authority["cache_is_authority"] is False
    assert authority["workflow_is_claim_authority"] is False
    assert authority["retrieval_scores_are_source_evidence"] is False
    assert authority["one_canonical_authority_per_generation"] is True

    required = set(policy["required_invariants"])
    assert {
        "evidence_consumption_citation_lock",
        "consolidation_path_dependence",
        "memory_use_calibration",
        "adaptive_retrieval_routing",
        "usage_aware_retrieval_projection_only",
        "failure_recovery_deletion_rollback",
        "source_host_relationship_self_identity_separation",
        "product_host_isolation",
    } <= required

    assert len(policy["zero_tolerance_classes"]) >= 18


def test_challengers_require_a_real_trigger_and_do_not_block_by_default() -> None:
    policy = _json("policies/stage_g_memory_fabric_candidate_qualification_policy.json")
    triggers = set(policy["challenger_admission_triggers"])

    assert {
        "selected_stack_contract_failure",
        "observed_capability_or_fidelity_gap",
        "measured_scale_latency_resource_blocker",
        "deletion_privacy_recovery_custody_or_distribution_blocker",
        "licensing_or_packaging_blocker",
        "strong_frontier_evidence_not_expressible_by_selected_path",
    } <= triggers

    levels = policy["qualification_levels"]
    assert levels["requires_same_role_all_pairs"] is False
    assert levels["requires_all_candidate_substitutions"] is False
    assert levels["requires_cartesian_or_broad_combinatorial_coverage"] is False

    close = policy["stage_g_closure"]
    assert close["requires_all_same_role_pairwise_coverage"] is False
    assert close["requires_all_registered_cross_role_candidate_substitutions"] is False
    assert close["requires_multi_plane_candidate_combinatorial_coverage"] is False
    assert close["requires_full_fabric_end_to_end_runs"] is True
    assert close["requires_owner_acceptance"] is True
    assert close["stage_h_eligible_before_this_gate_passes"] is False


def test_selected_stack_rule_is_bound_into_architecture_and_migration() -> None:
    architecture = _text("docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md")
    migration = _text("docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md")
    execution = _text("docs/ALICE_PHASE2_REPLACEMENT_AND_FABLE_V1_EXECUTION_PLAN_2026-09-27.md")

    version_match = re.search(
        r"\*\*Version:\*\* (\d+)\.(\d+)\.(\d+)",
        architecture,
    )
    assert version_match is not None
    assert tuple(map(int, version_match.groups())) >= (2, 1, 0)

    assert "Selected-stack qualification gate" in architecture
    assert "same-role all-pairs" not in architecture
    assert "candidate substitutions are required across every" not in architecture

    assert "**Version:** 1.9.0" in migration
    assert "selected destination path" in migration
    assert "Same-role all-pairs, all candidate substitutions" in migration
    assert "Stage H remains ineligible" in migration

    assert "MC10" in execution
    assert "No all-pairs backend tournament." in execution
    assert "Fable v1" in execution
