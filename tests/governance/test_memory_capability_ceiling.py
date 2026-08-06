from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[2]

ACTIVE_DOCS = (
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

REQUIRED_INVARIANTS = (
    "owner sovereignty",
    "privacy",
    "provenance",
    "truth",
    "product isolation",
    "deletion",
    "rollback",
)

PERMISSION_IDS = {
    "graph.read",
    "graph.write_projection",
    "graph.run_algorithm",
    "event_stream.append",
    "event_stream.subscribe",
    "claim.append",
    "claim.adjudicate",
    "vector.index_build",
    "vector.search",
    "workflow.launch",
    "workflow.signal",
    "workflow.cancel",
    "cluster.provision",
    "cluster.scale",
    "remote_compute.use",
    "model.train_distributed",
    "model.evaluate_challenger",
    "model.deploy_shadow",
    "model.deploy_canary",
    "dataset.build",
    "dataset.export_authorized",
    "memory.federate_owner_namespace",
    "memory.replicate",
    "memory.resolve_replication_conflict",
}


def _load_scanner() -> ModuleType:
    path = REPO / "scripts" / "audit_capability_barriers.py"
    spec = importlib.util.spec_from_file_location("audit_capability_barriers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _false_implemented(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).endswith("_implemented") and child is False:
                return True
            if _false_implemented(child):
                return True
    elif isinstance(value, list):
        return any(_false_implemented(child) for child in value)
    return False


def test_required_capability_first_files_exist() -> None:
    required = set(ACTIVE_DOCS) | {
        "docs/ALICE_CAPABILITY_EXPANSION_MANIFEST.md",
        "docs/MEMORY_ARCHITECTURE_HOLD.md",
        "docs/MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md",
        "scripts/migrate_memory_capability_ceiling_v1.py",
    }
    missing = sorted(path for path in required if not (REPO / path).is_file())
    assert not missing, missing


def test_destination_architecture_is_owner_sovereign_and_unbounded() -> None:
    architecture = (REPO / "docs/MEMORY_ARCHITECTURE_V4.md").read_text(
        encoding="utf-8"
    )
    lowered = architecture.lower()
    assert "owner-sovereign" in lowered
    assert "local-capable" in lowered
    assert "deployment-unbounded" in lowered
    assert "sqlite remains a valid reference" in lowered
    assert "does not define the destination topology" in lowered


def test_enabling_invariants_are_preserved() -> None:
    program = (
        REPO / "docs/MEMORY_CAPABILITY_EXPANSION_AND_RATIFICATION_PROGRAM.md"
    ).read_text(encoding="utf-8").lower()
    for marker in REQUIRED_INVARIANTS:
        assert marker in program, marker


def test_repository_has_no_unresolved_capability_ceiling() -> None:
    scanner = _load_scanner()
    findings = scanner.audit(REPO)
    unresolved = [
        finding
        for finding in findings
        if finding.disposition == "unresolved_active_barrier"
    ]
    assert not unresolved, [
        (item.path, item.line, item.code, item.excerpt) for item in unresolved
    ]


def test_capability_profiles_have_explicit_state_and_successor_semantics() -> None:
    payload = json.loads(
        (REPO / "policies/capability_profiles.json").read_text(encoding="utf-8")
    )
    semantics = payload["global_semantics"]
    assert semantics["capability_ceiling"] is False
    assert semantics["research_bans_are_invalid"] is True
    required_profiles = {
        "memory.edge",
        "memory.workstation",
        "memory.home_cluster",
        "memory.private_cluster",
        "memory.hybrid_cloud",
        "memory.distributed",
        "memory.frontier_research",
        "graph.shadow",
        "graph.production",
        "curator.shadow",
        "curator.production",
        "training.challenger",
        "training.canary",
        "training.production",
        "federation.owner_namespace",
    }
    profiles = payload["profiles"]
    assert required_profiles.issubset(profiles)
    for name, profile in profiles.items():
        assert profile["capability_ceiling"] is False, name
        assert profile["research_allowed"] is True, name
        assert "state" in profile, name
        assert "activation_condition" in profile, name
        assert "review_at" in profile, name
        assert "removal_criterion" in profile, name


def test_phase_scope_registry_requires_successor_paths() -> None:
    payload = json.loads(
        (REPO / "policies/phase_scope_registry.json").read_text(encoding="utf-8")
    )
    rules = payload["capability_expansion_rules"]
    assert rules["research_bans_are_invalid"] is True
    assert rules["destination_policy_requires_successor_path"] is True
    entries = payload["entries"]
    for rel in ACTIVE_DOCS:
        entry = entries[rel]
        assert entry["scope_kind"] == "destination_policy"
        assert entry["capability_ceiling"] is False
        assert entry["research_allowed"] is True
        assert entry["shadow_allowed"] is True
        assert entry["successor_path"]
        assert entry["removal_criterion"]


def test_lifelong_learning_expands_substrates() -> None:
    payload = json.loads(
        (REPO / "policies/lifelong_learning_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["capability_ceiling"] is False
    assert payload["automatic_challenger_generation"] is True
    required = {
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
    }
    assert required.issubset(payload["learning_substrates"])


def test_storage_policy_is_backend_neutral_and_deletion_aware() -> None:
    payload = json.loads(
        (REPO / "policies/storage_lifecycle_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["capability_ceiling"] is False
    assert payload["backend_neutral"] is True
    assert "distributed_object_store" in payload["storage_classes"]
    assert "multi_region_owner_authorized_replica" in payload["storage_classes"]
    assert (
        payload["deletion"]["cross_backend_influence_graph_required"] is True
    )
    assert (
        payload["backup"]["restores_apply_deletion_lineage_before_use"] is True
    )


def test_component_false_implementation_states_are_registered_without_contract_mutation() -> None:
    exempt = {
        "capability_profiles.json",
        "phase_scope_registry.json",
        "lifelong_learning_policy.json",
        "storage_lifecycle_policy.json",
        "authority_kernel_policy.json",
        "capability_parity_ledger.json",
        "product_lines.json",
    }
    profiles = json.loads(
        (REPO / "policies/capability_profiles.json").read_text(encoding="utf-8")
    )
    registry = profiles["component_policy_state_registry"]
    assert registry["released_component_contracts_mutated"] is False
    entries = registry["entries"]
    missing: list[str] = []
    for path in sorted((REPO / "policies").glob("*.json")):
        if path.name in exempt:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "capability_state_semantics" not in payload, path.name
        if _false_implemented(payload):
            entry = entries.get(path.name)
            if not isinstance(entry, dict):
                missing.append(path.name)
                continue
            assert entry["research_status"] == "allowed", path.name
            assert entry["meaning"].endswith("not_permanent_denial"), path.name
            assert entry["released_contract_unchanged"] is True, path.name
            assert entry["false_implemented_fields"], path.name
    assert not missing, missing


def test_released_policy_and_validator_versions_remain_compatibility_baselines() -> None:
    expected = {
        "capability_profiles.json": "1.5.0",
        "phase_scope_registry.json": "1.4.0",
        "lifelong_learning_policy.json": "1.1.0",
        "storage_lifecycle_policy.json": "1.0.0",
    }
    for name, version in expected.items():
        payload = json.loads((REPO / "policies" / name).read_text(encoding="utf-8"))
        assert payload["version"] == version, name
    validator = (REPO / "scripts/validate_phase5_parity_release.py").read_text(
        encoding="utf-8"
    )
    assert 'profiles.get("version") != "1.5.0"' in validator
    assert "capability_state_semantics" not in validator


def test_permissions_include_cognitive_fabric_actions() -> None:
    text = (REPO / "policies/permissions.yaml").read_text(encoding="utf-8")
    discovered = set(re.findall(r"^\s*-\s+id:\s+([^\s#]+)", text, re.MULTILINE))
    assert PERMISSION_IDS.issubset(discovered)


def test_m1_decisions_are_owner_ratified_without_runtime_activation() -> None:
    register = (REPO / "docs/MEMORY_M1_DECISION_REGISTER.md").read_text(
        encoding="utf-8"
    )
    record = (REPO / "docs/MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md").read_text(
        encoding="utf-8"
    )
    for decision in (
        "M1-DX0", "M1-DX1", "M1-DX2", "M1-DX3",
        "M1-D0", "M1-D1", "M1-D2", "M1-D3", "M1-D4",
        "M1-D5", "M1-D6", "M1-D7", "M1-D8", "M1-D9",
    ):
        assert f"| {decision} |" in register
    assert register.count("| `ratified` |") == 14
    assert "pending owner acceptance" not in register
    assert "distributed_global" not in register
    assert "distributed_multi_region" in register
    assert "**Status:** Owner-ratified" in record
    assert "does not, by itself, make an experimental capability production-active" in record
    assert "does not assert that Claim Authority runtime is implemented" in record
    assert "does not assert that Phase 2 migration has started" in record
    assert "constitutes a permanent capability ceiling" in record
