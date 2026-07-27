from __future__ import annotations

import json
from pathlib import Path

from product_family import HostInstance, HostSelectedIdentity, load_product_family_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_alice_and_friday_are_separate_product_identities() -> None:
    manifest = load_product_family_manifest(ROOT / "policies" / "product_lines.json")
    assert manifest.product("alice").kind != manifest.product("friday").kind
    assert manifest.shared_kernel_starts_at_phase == "5.0"
    assert manifest.formal_repository_split_gate == "6.5"


def test_friday_defaults_to_local_vendor_non_access() -> None:
    manifest = load_product_family_manifest(ROOT / "policies" / "product_lines.json")
    friday = manifest.product("friday")
    assert friday.raw_host_data_vendor_access is False
    assert friday.mandatory_vendor_account is False
    assert friday.offline_core_required is True


def test_host_storage_scopes_are_distinct() -> None:
    host_a = HostInstance("friday", "host-a", "1", "personal")
    host_b = HostInstance("friday", "host-b", "1", "personal")
    host_a.assert_isolated_from(host_b)
    assert host_a.storage_scope() != host_b.storage_scope()


def test_phase_one_to_four_files_remain_migratable() -> None:
    payload = json.loads(
        (ROOT / "policies" / "product_lines.json").read_text(encoding="utf-8")
    )
    assert payload["separation_rules"]["phase_1_to_4_files_are_migratable"] is True


def test_friday_privacy_defaults_do_not_enable_vendor_content_access() -> None:
    payload = json.loads(
        (ROOT / "policies" / "friday_privacy_defaults.json").read_text(
            encoding="utf-8"
        )
    )
    defaults = payload["defaults"]
    assert defaults["vendor_can_decrypt_host_data"] is False
    assert defaults["telemetry_personal_content_allowed"] is False
    assert defaults["network_egress_ledger_enabled"] is True


def test_consumer_uses_codename_but_host_selects_identity() -> None:
    manifest = load_product_family_manifest(ROOT / "policies" / "product_lines.json")
    consumer = manifest.product("friday")
    assert consumer.product_codename == "friday"
    assert consumer.public_product_brand_status == "to_be_selected"
    assert consumer.host_selects_assistant_name is True
    identity = HostSelectedIdentity("host-a", "Nova")
    identity.validate()
    assert identity.assistant_name == "Nova"


def test_consumer_has_full_destination_capability_parity() -> None:
    manifest = load_product_family_manifest(ROOT / "policies" / "product_lines.json")
    consumer = manifest.product("friday")
    assert consumer.full_capability_parity_with_alice is True
    assert consumer.permanent_consumer_capability_ceiling is False
    payload = json.loads((ROOT / "policies" / "capability_parity_ledger.json").read_text(encoding="utf-8"))
    assert payload["destination_parity_required"] is True
    assert payload["capabilities"]
