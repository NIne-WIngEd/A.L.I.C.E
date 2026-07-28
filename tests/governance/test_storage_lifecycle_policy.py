from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_storage_doctrine_is_aggressive_capture_selective_retention() -> None:
    policy = load_json("policies/storage_lifecycle_policy.json")
    assert policy["doctrine"] == "aggressive_temporary_capture_selective_durable_retention"
    assert policy["permanent_compact_event_ledger"] is True
    assert policy["permanent_full_payload_retention_default"] is False


def test_storage_uses_host_scoped_content_addressing() -> None:
    policy = load_json("policies/storage_lifecycle_policy.json")
    content = policy["content_addressing"]
    assert content["algorithm"] == "sha256"
    assert content["deduplication_scope"] == "host_instance_and_encryption_domain"
    assert content["cross_host_deduplication_allowed"] is False
    assert content["logical_metadata_must_remain_distinct"] is True


def test_storage_policy_defines_all_lifecycle_tiers() -> None:
    policy = load_json("policies/storage_lifecycle_policy.json")
    assert {"ledger", "raw_buffer", "hot", "warm", "cold", "quarantine", "deleted"}.issubset(policy["tiers"])
    assert policy["tiers"]["quarantine"]["learning_allowed"] is False
    assert policy["tiers"]["deleted"]["learning_allowed"] is False


def test_replay_is_representative_and_not_keep_everything() -> None:
    policy = load_json("policies/storage_lifecycle_policy.json")
    replay = policy["replay"]
    assert replay["selection"] == "representative_budgeted_and_versioned"
    assert replay["keep_every_event_equally"] is False
    assert "distribution_coverage" in replay["criteria"]
    assert "rare_and_surprising_cases" in replay["criteria"]
    assert "corrections" in replay["criteria"]


def test_storage_pressure_preserves_protected_artifacts() -> None:
    policy = load_json("policies/storage_lifecycle_policy.json")
    assert policy["capacity"]["preflight_peak_storage_estimate_required"] is True
    assert policy["capacity"]["protected_artifact_silent_deletion_allowed"] is False
    assert policy["backup"]["restore_testing_required"] is True
    assert policy["backup"]["mirroring_alone_is_backup"] is False


def test_deletion_prevents_deliberate_relearning() -> None:
    policy = load_json("policies/storage_lifecycle_policy.json")
    deletion = policy["deletion"]
    assert deletion["propagate_to_indexes_derivatives_replay_training_and_future_exports"] is True
    assert deletion["archive_restore_must_replay_deletion_lineage"] is True
    assert deletion["deliberate_relearning_of_deleted_payload_allowed"] is False


def test_lifelong_learning_policy_links_storage_lifecycle() -> None:
    policy = load_json("policies/lifelong_learning_policy.json")
    assert policy["capture_mode"] == "aggressive_temporary_capture"
    assert policy["permanent_compact_event_ledger"] is True
    assert policy["raw_experience_buffer"] == "policy_bounded"
    assert policy["durable_retention"] == "utility_weighted"
    assert policy["replay_selection"] == "representative_budgeted_and_versioned"
    assert policy["storage_lifecycle_policy"] == "policies/storage_lifecycle_policy.json"


def test_shared_kernel_and_friday_carry_storage_scope() -> None:
    products = load_json("policies/product_lines.json")
    required = set(products["shared_kernel"]["required_scopes"])
    assert {"content_digest", "retention_class", "storage_tier", "deletion_lineage"}.issubset(required)
    friday = products["products"]["friday"]
    assert friday["local_storage_lifecycle_required"] is True
    assert friday["cross_host_deduplication_allowed"] is False
    assert products["separation_rules"]["cross_host_deduplication_allowed"] is False


def test_storage_capability_has_product_family_parity() -> None:
    parity = load_json("policies/capability_parity_ledger.json")
    capability = parity["capabilities"]["storage_lifecycle_and_replay"]
    assert capability == {
        "alice": "planned",
        "kernel": "planned",
        "consumer": "planned",
        "target_phase": "5-8",
    }


def test_storage_architecture_documents_are_ratified_and_linked() -> None:
    storage = (ROOT / "docs" / "STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    memory = (ROOT / "docs" / "MEMORY_POLICY.md").read_text(encoding="utf-8")
    lifelong = (ROOT / "docs" / "LIFELONG_LEARNING_POLICY.md").read_text(encoding="utf-8")
    assert "aggressive temporary capture with selective durable retention" in storage.lower()
    assert "permanent compact event ledger" in storage.lower()
    assert "representative replay" in roadmap.lower()
    assert "STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md" in memory
    assert "STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md" in lifelong
