from __future__ import annotations

import json
from pathlib import Path

from product_family import HostInstance, HostSelectedIdentity, load_product_family_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_products_are_independent_and_phase65_is_readiness_gate() -> None:
    manifest = load_product_family_manifest(ROOT / "policies" / "product_lines.json")
    assert manifest.product("alice").kind != manifest.product("friday").kind
    assert manifest.product("alice").repository_independent is True
    assert manifest.product("friday").repository_independent is True
    assert manifest.shared_kernel_starts_at_phase == "5.0"
    assert manifest.independent_product_readiness_gate == "6.5"
    assert manifest.formal_repository_split_gate == "6.5"


def test_friday_source_never_lives_in_alice() -> None:
    manifest = load_product_family_manifest(ROOT / "policies" / "product_lines.json")
    friday = manifest.product("friday")
    assert friday.repository_required_before_product_source is True
    assert friday.product_source_allowed_in_alice_repository is False


def test_friday_defaults_to_local_vendor_non_access() -> None:
    friday = load_product_family_manifest(ROOT / "policies" / "product_lines.json").product("friday")
    assert friday.raw_host_data_vendor_access is False
    assert friday.mandatory_vendor_account is False
    assert friday.offline_core_required is True


def test_host_storage_scopes_are_distinct() -> None:
    host_a = HostInstance("friday", "host-a", "1", "personal")
    host_b = HostInstance("friday", "host-b", "1", "personal")
    host_a.assert_isolated_from(host_b)
    assert host_a.storage_scope() != host_b.storage_scope()


def test_phase_one_to_four_files_remain_migratable() -> None:
    payload = json.loads((ROOT / "policies" / "product_lines.json").read_text(encoding="utf-8"))
    assert payload["separation_rules"]["phase_1_to_4_files_are_migratable"] is True


def test_friday_privacy_defaults_do_not_enable_vendor_content_access() -> None:
    payload = json.loads((ROOT / "policies" / "friday_privacy_defaults.json").read_text(encoding="utf-8"))
    defaults = payload["defaults"]
    assert defaults["vendor_can_decrypt_host_data"] is False
    assert defaults["telemetry_personal_content_allowed"] is False
    assert defaults["network_egress_ledger_enabled"] is True


def test_consumer_uses_codename_but_host_selects_identity() -> None:
    consumer = load_product_family_manifest(ROOT / "policies" / "product_lines.json").product("friday")
    assert consumer.product_codename == "friday"
    assert consumer.public_product_brand_status == "to_be_selected"
    assert consumer.host_selects_assistant_name is True
    identity = HostSelectedIdentity("host-a", "Nova")
    identity.validate()


def test_consumer_has_parity_and_dual_production_approval() -> None:
    manifest = load_product_family_manifest(ROOT / "policies" / "product_lines.json")
    consumer = manifest.product("friday")
    assert consumer.full_capability_parity_with_alice is True
    assert consumer.permanent_consumer_capability_ceiling is False
    assert consumer.production_dual_approval_required is True
    assert manifest.production_governance["team_unilateral_production_promotion_allowed"] is False
    assert manifest.production_governance["emergency_new_behavior_allowed"] is False
    parity = json.loads((ROOT / "policies" / "capability_parity_ledger.json").read_text(encoding="utf-8"))
    assert parity["destination_parity_required"] is True
    assert "mission_graph.v1" in parity["capabilities"]

def test_fable_v1_has_one_full_personal_foundation_release_gate() -> None:
    payload = json.loads((ROOT / "policies" / "product_lines.json").read_text(encoding="utf-8"))
    friday = payload["products"]["friday"]

    assert friday["legacy_closed_alpha_gate"] == "8"
    assert friday["legacy_closed_alpha_gate_semantics"] == "internal_qualification_only_not_consumer_release"
    assert "closed_alpha_gate" not in friday
    assert friday["first_consumer_release_gate"] == "full_personal_cognitive_foundation_after_f11"
    assert friday["partial_capability_consumer_release_allowed"] is False
    assert friday["fable_v1_full_personal_cognitive_foundation_required"] is True
    assert friday["feature_model_api_delegation_allowed_v1"] is True
    assert friday["personal_cognitive_core_api_delegation_allowed_v1"] is False
    assert friday["internal_qualification_profiles_are_release_tiers"] is False
    assert friday["f12_platform_required_before_fable_v1"] is False
    assert friday["fable_v1_requires_internal_milestones"] == [
        "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11"
    ]

    separation = payload["separation_rules"]
    assert separation["internal_qualification_profile_may_define_consumer_release_capability"] is False
    assert separation["hardware_constraints_may_remove_fable_v1_logical_cognitive_planes"] is False
    assert separation["general_feature_models_may_be_external_in_fable_v1"] is True
    assert separation["personal_cognitive_foundation_may_be_externalized_in_fable_v1"] is False


def test_legacy_friday_profiles_are_internal_qualification_not_release_tiers() -> None:
    profiles = json.loads((ROOT / "policies" / "capability_profiles.json").read_text(encoding="utf-8"))["profiles"]

    assert profiles["friday.local_core"]["release_semantics"] == "internal_qualification_only"
    assert profiles["friday.learning_alpha"]["release_semantics"] == "legacy_named_internal_qualification_only"
    assert profiles["friday.optional_connected"]["release_semantics"] == "optional_overlay_not_release_tier"

    for profile_id in ("friday.local_core", "friday.learning_alpha", "friday.optional_connected"):
        assert profiles[profile_id]["consumer_release_eligible"] is False
        assert profiles[profile_id]["may_define_fable_v1_capability_boundary"] is False


def test_friday_roadmap_has_no_partial_consumer_launch_gate() -> None:
    roadmap = (ROOT / "docs" / "FRIDAY_ROADMAP.md").read_text(encoding="utf-8")

    assert "Fable v1 Release Gate" in roadmap
    assert "F4 through F11 are **internal engineering and qualification milestones**" in roadmap
    assert "No F4–F11 milestone by itself satisfies this release gate." in roadmap
    assert "minimum credible Friday launch cohort" not in roadmap
    assert "## F5 — Learning closed alpha" not in roadmap
    assert "## F6 — Personal intelligence beta" not in roadmap

def test_release_channels_do_not_define_fable_capability_tiers() -> None:
    production = json.loads(
        (ROOT / "policies" / "friday_production_governance.json").read_text(encoding="utf-8")
    )
    semantics = production["release_channel_semantics"]
    assert semantics["channel_labels_describe_distribution_maturity_not_cognitive_capability"] is True
    assert semantics["partial_capability_consumer_release_allowed"] is False
    assert semantics["approved_closed_alpha_does_not_imply_fable_v1_release"] is True

    kernel = json.loads(
        (ROOT / "policies" / "cognitive_kernel_release_attestation_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert kernel["release_channel_semantics"]["distribution_channel_is_not_capability_tier"] is True
    assert kernel["invariants"]["partial_capability_consumer_fable_release_allowed"] is False

