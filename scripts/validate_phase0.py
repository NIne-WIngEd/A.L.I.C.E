from __future__ import annotations

from pathlib import Path
import sys
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "SECURITY.md",
    "docs/ALICE_CONSTITUTION.md",
    "docs/PERMISSION_MODEL.md",
    "docs/MEMORY_POLICY.md",
    "docs/DATA_CLASSIFICATION.md",
    "docs/THREAT_MODEL.md",
    "docs/SCOPE_AND_NON_GOALS.md",
    "docs/EVALUATION_CHARTER.md",
    "docs/ROADMAP.md",
    "docs/decisions/ADR-001-system-principles.md",
    "policies/permissions.yaml",
    "policies/data_classes.yaml",
    "tests/constitutional/cases.yaml",
    "tests/permissions/cases.yaml",
    "tests/security/cases.yaml",
]

VALID_LEVELS = {"P0", "P1", "P2", "P3", "P4", "P5"}
VALID_CLASSES = {"PUBLIC", "INTERNAL", "PRIVATE", "HIGHLY_SENSITIVE", "SECRETS"}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def load_yaml(relative: str):
    path = ROOT / relative
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_required_files() -> None:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).is_file()]
    if missing:
        fail("Missing required files:\n  - " + "\n  - ".join(missing))


def validate_constitution() -> None:
    text = (ROOT / "docs/ALICE_CONSTITUTION.md").read_text(encoding="utf-8")
    required = [
        "**Version:** 0.1.0",
        "**Status:** Ratified foundational specification",
        "**Approved by:** MK Rayan",
        "**Ratification date:** July 13, 2026",
    ]
    for marker in required:
        if marker not in text:
            fail(f"Constitution is not properly ratified; missing marker: {marker}")


def validate_permissions() -> None:
    policy = load_yaml("policies/permissions.yaml")
    if policy.get("default_deny") is not True:
        fail("Permission policy must use default_deny: true")

    levels = set((policy.get("levels") or {}).keys())
    if levels != VALID_LEVELS:
        fail(f"Permission levels must be exactly {sorted(VALID_LEVELS)}")

    permissions = policy.get("permissions") or []
    ids = [item.get("id") for item in permissions]
    if any(not value for value in ids):
        fail("Every permission requires a non-empty id")
    if len(ids) != len(set(ids)):
        fail("Permission IDs must be unique")

    for item in permissions:
        pid = item["id"]
        level = item.get("level")
        if level not in VALID_LEVELS:
            fail(f"{pid}: invalid level {level}")

        confirmation = item.get("confirmation")
        standing = item.get("standing_authorization_allowed")

        if level == "P3" and confirmation != "explicit":
            fail(f"{pid}: every P3 permission must require explicit confirmation")
        if level == "P4":
            if confirmation != "strong":
                fail(f"{pid}: every P4 permission must require strong confirmation")
            if standing is not False:
                fail(f"{pid}: P4 standing authorization must be false in Phase 0")
        if level == "P5":
            if standing is not False:
                fail(f"{pid}: P5 standing authorization must be false")
            if confirmation is not None:
                fail(f"{pid}: P5 actions cannot define an executable confirmation")

        for data_class in item.get("allowed_data_classes", []):
            if data_class not in VALID_CLASSES:
                fail(f"{pid}: invalid data class {data_class}")


def validate_data_classes() -> None:
    policy = load_yaml("policies/data_classes.yaml")
    classes = policy.get("classes") or {}
    if set(classes.keys()) != VALID_CLASSES:
        fail(f"Data classes must be exactly {sorted(VALID_CLASSES)}")

    ranks = [classes[name].get("rank") for name in VALID_CLASSES]
    if sorted(ranks) != [0, 1, 2, 3, 4]:
        fail("Data-class ranks must be unique values 0 through 4")

    secrets = classes["SECRETS"]
    required_false = [
        "public_repository_allowed",
        "external_model_allowed",
        "ordinary_memory_allowed",
        "embeddings_allowed",
        "conversational_logs_allowed",
    ]
    for field in required_false:
        if secrets.get(field) is not False:
            fail(f"SECRETS.{field} must be false")
    if secrets.get("dedicated_secret_manager_required") is not True:
        fail("SECRETS must require a dedicated secret manager")


def validate_test_ids() -> None:
    seen: set[str] = set()
    for relative in [
        "tests/constitutional/cases.yaml",
        "tests/permissions/cases.yaml",
        "tests/security/cases.yaml",
    ]:
        suite = load_yaml(relative)
        cases = suite.get("cases") or []
        if not cases:
            fail(f"{relative} has no test cases")
        for case in cases:
            cid = case.get("id")
            if not cid:
                fail(f"{relative} contains a case without an id")
            if cid in seen:
                fail(f"Duplicate test case id: {cid}")
            seen.add(cid)


def scan_for_high_risk_files() -> None:
    forbidden_names = {
        ".env",
        "mail.py",
        "credentials.json",
        "token.json",
    }
    forbidden_suffixes = {".pem", ".p12", ".pfx"}
    ignored_parts = {".git", ".venv", "venv", "__pycache__"}

    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes:
            findings.append(str(relative))

    if findings:
        fail("Forbidden high-risk files detected:\n  - " + "\n  - ".join(findings))


def main() -> int:
    validate_required_files()
    validate_constitution()
    validate_permissions()
    validate_data_classes()
    validate_test_ids()
    scan_for_high_risk_files()
    print("Phase 0 validation passed.")
    print(f"Validated repository: {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
