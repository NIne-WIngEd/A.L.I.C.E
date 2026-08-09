from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _json(relative: str) -> dict:
    return json.loads(_text(relative))


def test_named_identity_roles_are_explicit() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")
    roles = policy["named_roles"]

    assert roles["source_person"] == "Mehejabin Elaina"
    assert roles["system"] == "A.L.I.C.E."
    assert roles["owner_host"] == "Rayan"
    assert roles["separate_product"] == "Friday"

    identity = policy["identity"]
    assert identity["alice_is_elaina_derived_clone"] is True
    assert identity["alice_is_elaina"] is False
    assert identity["clone_awareness_required"] is True


def test_rayan_host_learning_cannot_change_elaina_core_identity() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")

    assert policy["identity"]["ordinary_rayan_host_learning_may_modify_core_identity_anchor"] is False

    host = policy["host_learning"]
    assert "rayan_understanding" in host["allowed_updates"]
    assert "alice_rayan_relationship_model" in host["allowed_updates"]
    assert "shared_history" in host["allowed_updates"]
    assert "interaction_strategy" in host["allowed_updates"]

    assert "elaina_canonical_source_history" in host["forbidden_ordinary_updates"]
    assert "core_elaina_derived_identity_personality_anchor" in host["forbidden_ordinary_updates"]
    assert "clone_awareness_removal" in host["forbidden_ordinary_updates"]


def test_identity_model_and_memory_formation_model_are_separate() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")
    learned = policy["learned_components"]

    identity = learned["elaina_identity_personality_model"]
    formation = learned["memory_formation_model"]
    host = learned["rayan_host_model"]

    assert identity["learned_weights_or_equivalent_artifacts"] is True
    assert identity["separate_from_memory_formation_model"] is True
    assert identity["sole_canonical_biographical_record"] is False

    assert formation["learned_weights_or_equivalent_artifacts"] is True
    assert formation["separate_from_elaina_identity_personality_model"] is True
    assert formation["host_neutral_capability"] is True
    assert formation["canonical_authority"] is False
    assert formation["physical_backend_authority_selection"] is False

    assert host["ordinary_learning_method"] == "governed_memory_first"
    assert host["immediate_core_gradient_update_per_interaction"] is False
    assert host["may_modify_elaina_core_identity"] is False


def test_runtime_flow_keeps_rayan_and_elaina_distinct() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")
    runtime = policy["runtime_data_flow"]

    assert runtime["ordinary_rayan_data_becomes_elaina_source_history"] is False
    assert runtime["later_real_elaina_update_supported"] is True
    assert runtime["later_real_elaina_update_requires_explicit_source_person_classification"] is True


def test_friday_does_not_receive_alice_private_identity_or_host_state() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")
    friday = policy["friday"]

    assert friday["receives_separate_elaina_source_person_corpus"] is False
    assert friday["receives_elaina_specific_identity_weights"] is False
    assert friday["receives_rayan_host_data"] is False
    assert friday["receives_alice_rayan_relationship_history"] is False
    assert friday["receives_alice_continuity"] is False
    assert friday["host_neutral_memory_formation_architecture_transfer_allowed"] is True
    assert friday["friday_personality_develops_from_its_own_host_data"] is True


def test_stage_g_requires_full_cognitive_memory_qualification() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")
    stage_g = policy["stage_g"]

    assert stage_g["backend_persistence_alone_sufficient"] is False
    assert stage_g["requires_gold_semantic_decomposition"] is True
    assert stage_g["requires_memory_formation_model_training_and_evaluation"] is True
    assert stage_g["requires_elaina_identity_personality_model_training_and_evaluation"] is True
    assert stage_g["requires_real_authorized_rayan_host_seed"] is True
    assert stage_g["requires_complex_synthetic_rayan_life_continuation"] is True
    assert stage_g["requires_per_layer_tests"] is True
    assert stage_g["requires_cross_layer_tests"] is True
    assert stage_g["requires_retrieval_context_fusion_tests"] is True
    assert stage_g["requires_failure_recovery_concurrency_rebuild_rollback_scale_tests"] is True
    assert stage_g["stage_h_eligible_before_integrated_stage_g_acceptance"] is False


def test_phase2_final_replacement_waits_for_stage_j() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")
    phase2 = policy["phase2"]

    assert phase2["remains_current_released_authority_while_stage_g_open"] is True
    assert phase2["stage_h_bounded_canary"] is True
    assert phase2["stage_i_cutover"] is True
    assert phase2["final_replacement_or_retirement_requires_stage_j_acceptance"] is True


def test_repository_lifecycle_is_recorded() -> None:
    policy = _json("policies/memory_identity_host_learning_policy.json")
    lifecycle = policy["repository_lifecycle"]

    assert lifecycle["current"]["mode"] == "public_active_construction"
    assert lifecycle["current"]["documentation_priority"] == "accuracy_and_specificity_over_unnecessary_anonymization"
    assert lifecycle["current"]["explicit_names_allowed_when_needed_for_architectural_continuity"] is True
    assert lifecycle["current"]["raw_credentials_or_secrets_allowed"] is False
    assert lifecycle["current"]["raw_private_corpora_required_in_git"] is False

    assert lifecycle["after_friday_well_established"]["mode"] == "private_repository"
    assert lifecycle["long_term"]["mode"] == "owner_controlled_remote_repository_independent"
    assert lifecycle["long_term"]["remote_repository_required_for_operation"] is False
    assert lifecycle["long_term"]["runtime_or_deployment_topology_ceiling"] is False
    assert lifecycle["long_term"]["remote_compute_and_distributed_infrastructure_allowed_when_justified"] is True
    assert lifecycle["long_term"]["remote_repository_may_be_reestablished_when_useful"] is True
    assert lifecycle["long_term"]["must_preserve_history_provenance_backups_rollback_and_recovery"] is True


def test_named_roles_are_allowed_without_relaxing_private_payload_custody() -> None:
    custody = _json("policies/private_companion_custody.json")
    public = custody["public_repository"]

    assert public["opaque_directive_codes_allowed"] is True
    assert public["named_identity_roles_allowed"] is True
    assert public["architecture_specific_context_allowed"] is True
    assert public["encrypted_private_payload_allowed"] is False
    assert public["real_private_manifest_allowed"] is False
    assert public["keys_or_codebooks_allowed"] is False

    clone_policy = _json("policies/alice_clone_identity_policy.json")
    assert clone_policy["source_identity_disclosure"] == "named_source_person_architecture_private_evidence_custody"


def test_docs_remove_obsolete_opaque_only_boundary() -> None:
    doc = _text("docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md")
    clone = _text("docs/ALICE_CLONE_AWARE_IDENTITY_STANDARD.md")
    custody = _text("docs/PRIVATE_COMPANION_DATA_CUSTODY_STANDARD.md")
    migration = _text("docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md")
    readme = _text("README.md")

    for required in ("Mehejabin Elaina", "Rayan", "A.L.I.C.E.", "Friday"):
        assert required in doc

    assert "Rayan's ordinary host data must **not change A.L.I.C.E.'s core Elaina-derived personality/identity anchor**." in doc
    assert "Memory Formation Model is a learned model with its own weights" in doc
    assert "Final Phase 2 replacement or retirement is complete only after Stage J" in doc
    assert "operate without depending on a remote Git host or remote repository service as a required control-plane dependency" in doc
    assert "not a runtime or deployment topology ceiling" in doc
    assert "remote compute, distributed services, network storage, federation" in doc

    assert "**Version:** 1.2.0" in clone
    assert "Named source-person and owner/host roles; private evidence remains separately controlled" in clone
    assert "Mehejabin Elaina is the source person" in clone
    assert "Rayan host-learning firewall" in clone
    assert "Ordinary Rayan/host data must not modify" in clone
    assert "Opaque source identity" not in clone
    assert "source person's identity, history, and private evidence remain outside public Git" not in clone

    assert "**Version:** 1.1.0" in custody
    assert "Named identity roles and architecture-specific context may be documented in Git" in custody
    assert "Public Git contains only opaque identifiers" not in custody

    assert "**Version:** 1.7.0" in migration
    assert "Backend durability alone is not sufficient Stage G exit evidence." in migration
    assert "final replacement or retirement is complete only after Stage J" in migration

    assert "Memory Identity, Formation, Host Learning, and Repository Lifecycle" in readme
    assert "A.L.I.C.E.'s source person is **Mehejabin Elaina**." in readme
    assert "**Rayan** is A.L.I.C.E.'s owner/host." in readme
    assert "represented publicly only through opaque directives" not in readme
    assert "Their meanings are not stored in this repository." not in readme
