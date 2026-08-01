from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from product_family import load_product_family_manifest  # noqa: E402

REQUIRED_DOCS = [
    "docs/FRIDAY_PRODUCT_VISION.md",
    "docs/FRIDAY_ROADMAP.md",
    "docs/ALICE_FRIDAY_SEPARATION_PLAN.md",
    "docs/FRIDAY_ARCHITECTURE.md",
    "docs/FRIDAY_PRIVACY_AND_TRUST_MODEL.md",
    "docs/HOST_SELECTED_IDENTITY_STANDARD.md",
    "docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md",
    "docs/SHARED_KERNEL_EXTRACTION_STANDARD.md",
    "docs/FRIDAY_COGNITIVE_WORKSPACE_AND_PRODUCTION_GOVERNANCE_PLAN.md",
    "docs/FRIDAY_PRODUCTION_GOVERNANCE.md",
    "docs/decisions/ADR-011-friday-independent-repository-dual-approval.md",
    "policies/product_lines.json",
    "policies/friday_privacy_defaults.json",
    "policies/capability_parity_ledger.json",
    "policies/friday_production_governance.json",
    "policies/friday_release_attestation_schema.json",
]


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED_DOCS:
        if not (ROOT / relative).exists():
            errors.append(f"missing: {relative}")

    try:
        manifest = load_product_family_manifest()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid product-family manifest: {exc}")
        manifest = None

    try:
        privacy = json.loads((ROOT / "policies/friday_privacy_defaults.json").read_text(encoding="utf-8"))
        expected = {
            "vendor_can_decrypt_host_data": False,
            "mandatory_cloud_account": False,
            "core_functionality_offline": True,
            "telemetry_enabled": False,
            "telemetry_personal_content_allowed": False,
            "remote_inference_enabled": False,
            "network_egress_ledger_enabled": True,
            "update_packages_must_be_signed": True,
        }
        for key, value in expected.items():
            if privacy["defaults"].get(key) is not value:
                errors.append(f"Friday privacy default {key!r} must be {value!r}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid Friday privacy policy: {exc}")

    roadmap = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")
    for marker in (
        "Phase 5",
        "Active after approved P4.10 closure and merge",
        "Independent Product Readiness Gate",
        "Mission Graph and Cognitive Workspace lane",
        "Clone-aware private identity phase lane",
    ):
        if marker not in roadmap:
            errors.append(f"roadmap missing product marker: {marker}")

    try:
        parity = json.loads((ROOT / "policies/capability_parity_ledger.json").read_text(encoding="utf-8"))
        if parity.get("destination_parity_required") is not True:
            errors.append("capability parity ledger must require destination parity")
        for capability in ("mission_graph.v1", "result_capsule.v1", "workspace_projection.v1", "guest_grant.v1"):
            if capability not in parity.get("capabilities", {}):
                errors.append(f"capability parity ledger missing {capability}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid capability parity ledger: {exc}")

    if manifest is not None:
        print(
            "Validated product family:",
            ", ".join(sorted(manifest.products)),
            f"with readiness gate {manifest.independent_product_readiness_gate}",
        )

    if errors:
        print("Product-family validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("A.L.I.C.E.–Friday product-family architecture validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
