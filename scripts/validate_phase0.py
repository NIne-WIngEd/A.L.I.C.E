"""Validate the current constitutional and authority foundation.

The filename remains for CI compatibility. The validator now treats Phase 0 as the
living governance kernel rather than freezing the July 2026 implementation profile.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "SECURITY.md",
    "docs/ALICE_CONSTITUTION.md",
    "docs/ARCHITECTURE.md",
    "docs/PERMISSION_MODEL.md",
    "docs/MEMORY_POLICY.md",
    "docs/DATA_CLASSIFICATION.md",
    "docs/THREAT_MODEL.md",
    "docs/SCOPE_AND_NON_GOALS.md",
    "docs/EVALUATION_CHARTER.md",
    "docs/ROADMAP.md",
    "docs/CAPABILITY_UNBLOCKING_POLICY.md",
    "docs/PHASE_SCOPE_POLICY.md",
    "docs/IMPLEMENTATION_EVOLVABILITY_STANDARD.md",
    "docs/CONSTRAINT_REGISTRY.md",
    "policies/permissions.yaml",
    "policies/data_classes.yaml",
    "policies/authority_kernel_policy.json",
    "policies/lifelong_learning_policy.json",
    "policies/capability_profiles.json",
    "policies/phase_scope_registry.json",
    "policies/product_lines.json",
    "policies/friday_privacy_defaults.json",
    "docs/FRIDAY_PRODUCT_VISION.md",
    "docs/FRIDAY_ROADMAP.md",
    "docs/ALICE_FRIDAY_SEPARATION_PLAN.md",
    "docs/SHARED_KERNEL_EXTRACTION_STANDARD.md",
    "docs/MEGA_ARCHITECTURE_MIGRATION.md",
    "docs/CHAT_ARCHITECTURE_DECISION_LEDGER.md",
]
VALID_LEVELS = {"P0", "P1", "P2", "P3", "P4", "P5"}
VALID_AUTONOMY = {f"A{index}" for index in range(7)}
VALID_CLASSES = {"PUBLIC", "INTERNAL", "PRIVATE", "HIGHLY_SENSITIVE", "SECRETS"}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def load_yaml(relative: str):
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(relative: str):
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        fail("Missing required files:\n - " + "\n - ".join(missing))


def validate_constitution() -> None:
    text = (ROOT / "docs/ALICE_CONSTITUTION.md").read_text(encoding="utf-8")
    required = [
        "**Version:** 1.1.0",
        "**Owner and final constitutional authority:** MK Rayan",
        "Capability-First Governance",
        "Continuous Learning and Memory",
        "Self-Modification",
        "No Legacy Veto and Implementation Supremacy",
        "minimal authority kernel",
    ]
    for marker in required:
        if marker not in text:
            fail(f"Constitution is missing marker: {marker}")


def validate_permissions() -> None:
    policy = load_yaml("policies/permissions.yaml")
    if policy.get("policy_model") != "mission_scoped_autonomy":
        fail("Permission policy must use mission_scoped_autonomy")
    if policy.get("activation_model") != "capability_profile_plus_mission":
        fail("Permission policy must activate through capability profiles plus missions")
    levels = set((policy.get("levels") or {}).keys())
    if levels != VALID_LEVELS:
        fail(f"Compatibility permission levels must be exactly {sorted(VALID_LEVELS)}")
    autonomy = set((policy.get("autonomy_classes") or {}).keys())
    if autonomy != VALID_AUTONOMY:
        fail(f"Autonomy classes must be exactly {sorted(VALID_AUTONOMY)}")
    permissions = policy.get("permissions") or []
    ids = [item.get("id") for item in permissions]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        fail("Every permission requires a unique non-empty id")
    for item in permissions:
        if item.get("level") not in VALID_LEVELS:
            fail(f"{item.get('id')}: invalid compatibility level")
        if item.get("autonomy_class") not in VALID_AUTONOMY:
            fail(f"{item.get('id')}: invalid autonomy class")
        for data_class in item.get("allowed_data_classes", []):
            if data_class not in VALID_CLASSES:
                fail(f"{item.get('id')}: invalid data class {data_class}")


def validate_data_classes() -> None:
    policy = load_yaml("policies/data_classes.yaml")
    classes = policy.get("classes") or {}
    if set(classes) != VALID_CLASSES:
        fail(f"Data classes must be exactly {sorted(VALID_CLASSES)}")
    ranks = [classes[name].get("rank") for name in VALID_CLASSES]
    if sorted(ranks) != [0, 1, 2, 3, 4]:
        fail("Data-class ranks must be unique values 0 through 4")
    secrets = classes["SECRETS"]
    if secrets.get("dedicated_secret_manager_required") is not True:
        fail("SECRETS must require a dedicated secret manager")


def validate_profiles() -> None:
    registry = load_json("policies/capability_profiles.json")
    profiles = registry.get("profiles") or {}
    required = {
        "conversation.phase3.compatibility",
        "conversation.integrated",
        "orchestration.adaptive",
        "information.phase4.foundation",
        "information.live_read_only",
        "information.authorized_browser",
        "evolution.a5",
        "friday.local_core",
        "friday.learning_alpha",
        "friday.optional_connected",
    }
    if not required.issubset(profiles):
        fail(f"Capability registry is missing profiles: {sorted(required - set(profiles))}")
    for profile_id, profile in profiles.items():
        if profile.get("capability_ceiling") is not False:
            fail(f"{profile_id}: capability_ceiling must be false")


def validate_kernel() -> None:
    policy = load_json("policies/authority_kernel_policy.json")
    invariants = policy.get("root_invariants") or []
    if len(invariants) != 6:
        fail("Authority kernel must contain the six ratified root invariants")
    if policy.get("constitutional_activation_requires") != "explicit_owner_ratification":
        fail("Authority-kernel activation must require explicit owner ratification")


def scan_for_high_risk_files() -> None:
    forbidden_names = {".env", "credentials.json", "token.json"}
    forbidden_suffixes = {".pem", ".p12", ".pfx", ".key"}
    ignored_parts = {".git", ".venv", "venv", "__pycache__"}
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        if path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes:
            findings.append(str(path.relative_to(ROOT)))
    if findings:
        fail("Unbrokered credential files detected:\n - " + "\n - ".join(findings))


def main() -> int:
    validate_required_files()
    validate_constitution()
    validate_permissions()
    validate_data_classes()
    validate_profiles()
    validate_kernel()
    scan_for_high_risk_files()
    print("A.L.I.C.E. constitutional and capability foundation validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
