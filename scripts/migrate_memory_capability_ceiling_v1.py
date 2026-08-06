from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_DESTINATION_DOCS = (
    "docs/ALICE_CAPABILITY_EXPANSION_MANIFEST.md",
    "docs/MEMORY_CAPABILITY_EXPANSION_AND_RATIFICATION_PROGRAM.md",
    "docs/MEMORY_ARCHITECTURE_V4.md",
    "docs/MEMORY_PERFORMANCE_AND_RELIABILITY_STANDARD.md",
    "docs/MEMORY_RECORD_AND_PROVENANCE_STANDARD.md",
    "docs/MEMORY_RENOVATION_PLAN.md",
    "docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md",
    "docs/MEMORY_PUBLIC_CLAIM_RELEASE_STANDARD.md",
    "docs/MEMORY_M1_RATIFICATION_PLAN.md",
    "docs/MEMORY_M1_DECISION_REGISTER.md",
    "docs/MEMORY_CLAIM_IDENTITY_AND_VERSION_PROPOSAL.md",
    "docs/MEMORY_M1_RESEARCH_BASIS.md",
    "docs/ARCHITECTURE.md",
    "docs/CAPABILITY_CATALOG.md",
    "docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md",
    "docs/ROADMAP.md",
    "docs/CONSTRAINT_REGISTRY.md",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any], *, dry_run: bool) -> bool:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == rendered:
        return False
    if dry_run:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return True


def _write_text(path: Path, rendered: str, *, dry_run: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == rendered:
        return False
    if dry_run:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return True


def _next_version(current: object, target: str) -> str:
    return target if str(current or "") != target else target


def _profile(
    *,
    domain: str,
    scope_kind: str,
    description: str,
    state: str,
    capabilities: dict[str, Any],
    activation_condition: str,
    successor_profile: str | None = None,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "scope_kind": scope_kind,
        "state": state,
        "capability_ceiling": False,
        "research_allowed": True,
        "shadow_allowed": True,
        "successor_profile": successor_profile,
        "activation_condition": activation_condition,
        "review_at": "evidence_or_owner_directive",
        "removal_criterion": "superseded_by_a_more_capable_evaluated_profile",
        "description": description,
        "capabilities": capabilities,
    }


def migrate_capability_profiles(path: Path, *, dry_run: bool) -> bool:
    payload = _load_json(path)
    if str(payload.get("version")) != "1.5.0":
        raise ValueError("capability_profiles.json baseline must remain version 1.5.0 during M1 documentation ratification")
    payload["authority"] = "docs/MEMORY_CAPABILITY_EXPANSION_AND_RATIFICATION_PROGRAM.md"
    payload["capability_ceiling"] = False
    payload["capability_state_vocabulary"] = [
        "destination",
        "research_active",
        "prototype_operational",
        "shadow_evaluated",
        "canary_enabled",
        "production_profile_enabled",
        "degraded",
        "superseded",
        "retired",
        "compatibility_only",
        "implemented_disabled",
        "enabled",
    ]
    payload["global_semantics"] = {
        "owner_sovereign": True,
        "local_capable": True,
        "deployment_unbounded": True,
        "capability_ceiling": False,
        "research_bans_are_invalid": True,
        "temporary_limits_require_profile": True,
        "fixed_limits_are_profile_defaults": True,
        "historical_profiles_do_not_govern_successors": True,
        "production_activation_is_profile_governed": True,
        "private_state_requires_owner_authorized_custody": True,
        "product_and_host_isolation_required": True,
        "provenance_required": True,
        "deletion_and_rollback_required": True,
    }

    profiles = payload.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("capability_profiles.json profiles must be an object")

    successor_by_name = {
        "memory.phase2.compatibility": "memory.workstation",
        "vault.phase1.compatibility": "memory.edge",
        "conversation.phase3.compatibility": "conversation.integrated",
        "orchestration.phase3.compatibility": "orchestration.adaptive",
        "information.phase4.foundation": "information.live_read_only",
        "legacy.release.compatibility": "memory.frontier_research",
        "friday.local_core": "friday.learning_alpha",
    }
    for name, raw in profiles.items():
        if not isinstance(raw, dict):
            continue
        raw.setdefault("capability_ceiling", False)
        raw["research_allowed"] = True
        raw["shadow_allowed"] = True
        if "state" not in raw:
            if "compatibility" in name or raw.get("scope_kind") == "phase_local":
                raw["state"] = "compatibility_only"
            else:
                raw["state"] = "enabled"
        raw.setdefault("successor_profile", successor_by_name.get(name))
        raw.setdefault(
            "activation_condition",
            "exact_profile_selection_plus_required_authority_and_controls",
        )
        raw.setdefault("review_at", "evidence_or_owner_directive")
        raw.setdefault(
            "removal_criterion",
            "superseded_by_a_more_capable_evaluated_profile",
        )

    component_state_entries: dict[str, Any] = {}
    excluded_component_names = {
        "capability_profiles.json",
        "phase_scope_registry.json",
        "lifelong_learning_policy.json",
        "storage_lifecycle_policy.json",
        "authority_kernel_policy.json",
        "capability_parity_ledger.json",
        "product_lines.json",
    }
    for component_path in sorted(path.parent.glob("*.json")):
        if component_path.name in excluded_component_names:
            continue
        try:
            component_payload = _load_json(component_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        false_paths = _find_false_implemented(component_payload)
        if not false_paths:
            continue
        component_state_entries[component_path.name] = {
            "source_policy_version": str(component_payload.get("version", "")),
            "false_implemented_fields": false_paths,
            "destination_status": "defined_by_capability_catalog_and_successor_profiles",
            "research_status": "allowed",
            "successor_profile": "registered_in_capability_profiles_or_phase_scope_registry",
            "activation_evidence": "implementation_tests_evaluation_and_required_authority",
            "meaning": "truthful_current_implementation_state_not_permanent_denial",
            "released_contract_unchanged": True,
        }
    payload["component_policy_state_registry"] = {
        "schema_version": "1.0.0",
        "capability_ceiling": False,
        "released_component_contracts_mutated": False,
        "semantics": "false _implemented fields describe current implementation state, not destination prohibition",
        "entries": component_state_entries,
    }

    common_memory = {
        "evidence_log": True,
        "claim_authority": True,
        "current_claim_projection": True,
        "graph_projection": True,
        "vector_projection": True,
        "payload_store": True,
        "durable_workflows": True,
        "adaptive_context": True,
        "deletion_coordinator": True,
        "rollback": True,
        "backend_replacement": True,
    }
    profiles.update(
        {
            "memory.edge": _profile(
                domain="memory",
                scope_kind="mission_selectable",
                description="Embedded and edge-capable cognitive fabric with owner-controlled custody and export.",
                state="research_active",
                capabilities={
                    **common_memory,
                    "single_node": True,
                    "distributed_execution": False,
                    "remote_compute": False,
                },
                activation_condition="edge_profile_evaluation_and_owner_custody",
                successor_profile="memory.workstation",
            ),
            "memory.workstation": _profile(
                domain="memory",
                scope_kind="mission_selectable",
                description="Single or multi-GPU workstation profile with replaceable local services.",
                state="research_active",
                capabilities={
                    **common_memory,
                    "single_node": True,
                    "multi_process": True,
                    "distributed_execution": True,
                    "remote_compute": False,
                },
                activation_condition="workstation_benchmark_privacy_deletion_and_rollback_pass",
                successor_profile="memory.home_cluster",
            ),
            "memory.home_cluster": _profile(
                domain="memory",
                scope_kind="mission_selectable",
                description="Owner-controlled multi-node home or lab cluster.",
                state="research_active",
                capabilities={
                    **common_memory,
                    "replication": True,
                    "sharding": True,
                    "distributed_execution": True,
                    "multi_device_sync": True,
                },
                activation_condition="cluster_failure_recovery_and_namespace_isolation_pass",
                successor_profile="memory.private_cluster",
            ),
            "memory.private_cluster": _profile(
                domain="memory",
                scope_kind="mission_selectable",
                description="Owner-authorized private cluster with durable services and accelerator scheduling.",
                state="research_active",
                capabilities={
                    **common_memory,
                    "replication": True,
                    "sharding": True,
                    "distributed_training": True,
                    "distributed_inference": True,
                    "multi_region": True,
                },
                activation_condition="private_cluster_slo_security_deletion_and_failover_pass",
                successor_profile="memory.hybrid_cloud",
            ),
            "memory.hybrid_cloud": _profile(
                domain="memory",
                scope_kind="owner_authorized",
                description="Hybrid local and remote deployment preserving owner-controlled authority and replacement.",
                state="research_active",
                capabilities={
                    **common_memory,
                    "local_continuity": True,
                    "remote_compute": True,
                    "remote_storage": True,
                    "replication": True,
                    "distributed_training": True,
                    "distributed_inference": True,
                },
                activation_condition="egress_custody_export_deletion_and_cost_profile_approved",
                successor_profile="memory.distributed",
            ),
            "memory.distributed": _profile(
                domain="memory",
                scope_kind="owner_authorized",
                description="Distributed multi-device and multi-region cognitive fabric.",
                state="research_active",
                capabilities={
                    **common_memory,
                    "multi_region": True,
                    "horizontal_scaling": True,
                    "federation": True,
                    "distributed_training": True,
                    "distributed_inference": True,
                },
                activation_condition="distributed_consistency_partition_recovery_and_owner_namespace_pass",
                successor_profile="memory.frontier_research",
            ),
            "memory.frontier_research": _profile(
                domain="memory",
                scope_kind="research_only",
                description="Open-ended frontier profile for later architectures and capability experiments.",
                state="research_active",
                capabilities={
                    **common_memory,
                    "new_backends_allowed": True,
                    "new_models_allowed": True,
                    "new_topologies_allowed": True,
                    "new_training_methods_allowed": True,
                    "open_ended_research": True,
                },
                activation_condition="isolated_or_owner_authorized_research_with_lineage",
            ),
            "graph.shadow": _profile(
                domain="graph",
                scope_kind="research_only",
                description="Cognitive graph projection and algorithm evaluation without claim authority.",
                state="research_active",
                capabilities={
                    "read_claims": True,
                    "write_graph_projection": True,
                    "run_algorithms": True,
                    "graph_embeddings": True,
                    "propose_claim_candidates": True,
                    "production_claim_authority": False,
                },
                activation_condition="registered_graph_generation_and_no_production_authority",
                successor_profile="graph.production",
            ),
            "graph.production": _profile(
                domain="graph",
                scope_kind="owner_ratified_mission",
                description="Production graph reasoning with provenance, deletion, reconciliation, and rollback.",
                state="destination",
                capabilities={
                    "graph_retrieval": True,
                    "graph_algorithms": True,
                    "graph_embeddings": True,
                    "claim_candidate_generation": True,
                    "registered_write_back": True,
                    "graph_to_claim_reconciliation": True,
                },
                activation_condition="graph_quality_failure_deletion_and_reconciliation_evidence",
            ),
            "curator.shadow": _profile(
                domain="curation",
                scope_kind="research_only",
                description="Durable Curator computation with candidate output and no production promotion.",
                state="research_active",
                capabilities={
                    "candidate_extraction": True,
                    "episode_construction": True,
                    "belief_candidate_generation": True,
                    "automatic_production_promotion": False,
                },
                activation_condition="isolated_shadow_workflows_and_complete_lineage",
                successor_profile="curator.production",
            ),
            "curator.production": _profile(
                domain="curation",
                scope_kind="mission_selectable",
                description="Profile-governed automatic curation and adjudication.",
                state="destination",
                capabilities={
                    "automatic_memory_formation": True,
                    "automatic_adjudication": True,
                    "belief_revision": True,
                    "skill_formation": True,
                    "learned_retention": True,
                },
                activation_condition="class_specific_evidence_thresholds_deletion_rollback_and_authority",
            ),
            "training.challenger": _profile(
                domain="training",
                scope_kind="research_only",
                description="Automatic challenger dataset construction and model training.",
                state="research_active",
                capabilities={
                    "dataset_build": True,
                    "distributed_training": True,
                    "remote_accelerators": True,
                    "checkpointing": True,
                    "production_influence": False,
                },
                activation_condition="dataset_lineage_compute_budget_and_isolation",
                successor_profile="training.canary",
            ),
            "training.canary": _profile(
                domain="training",
                scope_kind="owner_ratified_mission",
                description="Bounded canary influence for evaluated challenger models.",
                state="destination",
                capabilities={
                    "shadow_serving": True,
                    "canary_influence": True,
                    "automatic_rollback": True,
                    "production_influence": False,
                },
                activation_condition="champion_challenger_pass_canary_scope_and_rollback",
                successor_profile="training.production",
            ),
            "training.production": _profile(
                domain="training",
                scope_kind="owner_ratified_mission",
                description="Production learned-model influence with lineage and rollback.",
                state="destination",
                capabilities={
                    "production_influence": True,
                    "automatic_low_impact_promotion": True,
                    "distributed_training": True,
                    "distributed_serving": True,
                    "model_retirement": True,
                },
                activation_condition="A5_or_owner_authority_plus_full_evaluation_deletion_and_rollback",
            ),
            "federation.owner_namespace": _profile(
                domain="federation",
                scope_kind="owner_authorized",
                description="Replication and synchronization inside one owner-authorized authority namespace.",
                state="research_active",
                capabilities={
                    "multi_device_replication": True,
                    "offline_operation": True,
                    "causal_metadata": True,
                    "conflict_receipts": True,
                    "deletion_watermarks": True,
                    "cross_owner_access": False,
                },
                activation_condition="same_owner_namespace_encryption_conflict_and_deletion_controls",
            ),
        }
    )
    return _write_json(path, payload, dry_run=dry_run)


def migrate_phase_scope(path: Path, *, dry_run: bool) -> bool:
    payload = _load_json(path)
    if str(payload.get("version")) != "1.4.0":
        raise ValueError("phase_scope_registry.json baseline must remain version 1.4.0 during M1 documentation ratification")
    payload["capability_expansion_rules"] = {
        "platform_lock_in_requires_profile_scope": True,
        "broad_hold_requires_sunset": True,
        "fixed_limits_require_profile_and_override": True,
        "research_bans_are_invalid": True,
        "destination_policy_requires_successor_path": True,
        "historical_compatibility_cannot_govern_successors": True,
        "capability_ceiling": False,
    }
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("phase_scope_registry entries must be an object")
    for rel in ACTIVE_DESTINATION_DOCS:
        entries[rel] = {
            "scope_kind": "destination_policy",
            "profile": "memory.capability_first.m1",
            "milestone": "M1 capability-first ratification",
            "capability_ceiling": False,
            "research_allowed": True,
            "shadow_allowed": True,
            "successor_path": "later owner-ratified capability and deployment profiles",
            "production_activation_condition": "explicit_profile_evidence_and_required_authority",
            "review_at": "M1_final_consistency_review",
            "removal_criterion": "superseded_by_a_ratified_successor_document",
            "sunset_condition": "entry_updates_when_document_is_superseded_or_retired",
        }
    return _write_json(path, payload, dry_run=dry_run)


def migrate_lifelong_learning(path: Path, *, dry_run: bool) -> bool:
    payload = _load_json(path)
    if str(payload.get("version")) != "1.1.0":
        raise ValueError("lifelong_learning_policy.json baseline must remain version 1.1.0 during M1 documentation ratification")
    payload["capability_ceiling"] = False
    payload["distributed_training"] = "mission_and_budget_authorized"
    payload["remote_accelerators"] = "owner_authorized_profiles"
    payload["shadow_model_serving"] = "enabled_by_evaluation_profile"
    payload["automatic_challenger_generation"] = True
    payload["production_influence"] = "profile_and_authority_governed"
    payload["deletion_and_rollback_required"] = True
    substrates = payload.setdefault("learning_substrates", [])
    if not isinstance(substrates, list):
        raise ValueError("learning_substrates must be a list")
    for item in (
        "temporal_graph",
        "causal_graph",
        "multimodal_episode_models",
        "distributed_event_streams",
        "learned_retrieval",
        "learned_routing",
        "preference_models",
        "world_models",
        "source_person_models",
        "personal_adapters",
        "continual_pretraining_candidates",
        "embodied_skills",
        "simulation_generated_experience",
    ):
        if item not in substrates:
            substrates.append(item)
    return _write_json(path, payload, dry_run=dry_run)


def migrate_storage(path: Path, *, dry_run: bool) -> bool:
    payload = _load_json(path)
    if str(payload.get("version")) != "1.0.0":
        raise ValueError("storage_lifecycle_policy.json baseline must remain version 1.0.0 during M1 documentation ratification")
    payload["capability_ceiling"] = False
    payload["backend_neutral"] = True
    payload["profile_defaults_are_adaptive"] = True
    payload["storage_classes"] = [
        "embedded_local",
        "encrypted_nas",
        "edge_replica",
        "s3_compatible_object_store",
        "encrypted_cloud_archive",
        "distributed_object_store",
        "erasure_coded_store",
        "event_store",
        "relational_claim_store",
        "graph_projection_store",
        "vector_projection_store",
        "model_registry",
        "dataset_registry",
        "offline_archive",
        "multi_region_owner_authorized_replica",
    ]
    content = payload.setdefault("content_addressing", {})
    if isinstance(content, dict):
        # Preserve the released host/encryption-domain default for compatibility.
        # Broader owner-namespace deduplication is a separate evaluated profile.
        content.setdefault(
            "deduplication_scope",
            "host_instance_and_encryption_domain",
        )
        content["privacy_preserving_broader_scope"] = (
            "owner_authorized_authority_namespace_and_key_domain"
        )
        content["privacy_preserving_broader_scope_state"] = (
            "evaluation_profile_only"
        )
        content["leakage_analysis_required"] = True
        content["logical_metadata_must_remain_distinct"] = True
    capacity = payload.setdefault("capacity", {})
    if isinstance(capacity, dict):
        capacity["numeric_values_are_profile_defaults"] = True
        capacity["adaptive_override"] = "mission_value_risk_cost_and_recoverability"
    backup = payload.setdefault("backup", {})
    if isinstance(backup, dict):
        backup["distributed_and_multi_region_profiles_allowed"] = True
        backup["restores_apply_deletion_lineage_before_use"] = True
    deletion = payload.setdefault("deletion", {})
    if isinstance(deletion, dict):
        deletion["cross_backend_influence_graph_required"] = True
        deletion["noncompliant_derivative_action"] = "rebuild_retire_or_quarantine"
    return _write_json(path, payload, dry_run=dry_run)


def _find_false_implemented(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("_implemented") and child is False:
                found.append(key_path)
            found.extend(_find_false_implemented(child, key_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_false_implemented(child, f"{prefix}[{index}]"))
    return found



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    policies = repo / "policies"
    required = {
        "capability_profiles": policies / "capability_profiles.json",
        "phase_scope_registry": policies / "phase_scope_registry.json",
        "lifelong_learning": policies / "lifelong_learning_policy.json",
        "storage_lifecycle": policies / "storage_lifecycle_policy.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise SystemExit(f"Missing required policy for {name}: {path}")

    changed: list[str] = []
    operations = (
        ("policies/capability_profiles.json", migrate_capability_profiles),
        ("policies/phase_scope_registry.json", migrate_phase_scope),
        ("policies/lifelong_learning_policy.json", migrate_lifelong_learning),
        ("policies/storage_lifecycle_policy.json", migrate_storage),
    )
    for relative, operation in operations:
        path = repo / relative
        if operation(path, dry_run=args.dry_run):
            changed.append(relative)


    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "dry_run": args.dry_run,
        "changed_file_count": len(changed),
        "changed_files": sorted(changed),
        "capability_ceiling": False,
        "private_content_read": False,
        "released_component_contracts_mutated": False,
        "validator_contracts_mutated": False,
    }
    report_path = args.report or (
        repo / ".alice-reports" / "memory-capability-policy-migration.json"
    )
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
