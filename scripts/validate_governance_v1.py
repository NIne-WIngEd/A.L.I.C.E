from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
POLICIES = ROOT / "policies"

required = [
    DOCS / "ALICE_CONSTITUTION.md",
    DOCS / "ROADMAP.md",
    DOCS / "ARCHITECTURE.md",
    DOCS / "PERMISSION_MODEL.md",
    DOCS / "MEMORY_POLICY.md",
    DOCS / "LIFELONG_LEARNING_POLICY.md",
    DOCS / "AUTONOMY_AND_JUDGMENT_POLICY.md",
    DOCS / "CONSTRAINT_REGISTRY.md",
    DOCS / "CAPABILITY_UNBLOCKING_POLICY.md",
    DOCS / "PHASE_SCOPE_POLICY.md",
    DOCS / "IMPLEMENTATION_EVOLVABILITY_STANDARD.md",
    DOCS / "PROJECT_WIDE_CAPABILITY_AUDIT.md",
    DOCS / "PUBLIC_REPOSITORY_AUDIT_MANIFEST.md",
    DOCS / "MEGA_ARCHITECTURE_MIGRATION.md",
    DOCS / "CHAT_ARCHITECTURE_DECISION_LEDGER.md",
    POLICIES / "authority_kernel_policy.json",
    POLICIES / "lifelong_learning_policy.json",
    POLICIES / "capability_profiles.json",
    POLICIES / "phase_scope_registry.json",
    POLICIES / "permissions.yaml",
    ROOT / "src" / "alice_capability_profiles.py",
    ROOT / "src" / "alice_evolution" / "capability_runtime.py",
    ROOT / "scripts" / "audit_capability_barriers.py",
    DOCS / "FRIDAY_PRODUCT_VISION.md",
    DOCS / "FRIDAY_ROADMAP.md",
    DOCS / "ALICE_FRIDAY_SEPARATION_PLAN.md",
    DOCS / "FRIDAY_ARCHITECTURE.md",
    DOCS / "FRIDAY_PRIVACY_AND_TRUST_MODEL.md",
    DOCS / "HOST_SELECTED_IDENTITY_STANDARD.md",
    DOCS / "PRODUCT_FAMILY_CAPABILITY_PARITY.md",
    DOCS / "FRIDAY_HANDOFF_AND_MAINTENANCE_PLAN.md",
    DOCS / "SHARED_KERNEL_EXTRACTION_STANDARD.md",
    DOCS / "PHASE_1_4_PRODUCT_MIGRATION_PLAN.md",
    POLICIES / "product_lines.json",
    POLICIES / "friday_privacy_defaults.json",
    POLICIES / "capability_parity_ledger.json",
    ROOT / "src" / "product_family" / "manifest.py",
    ROOT / "scripts" / "validate_product_family.py",
]

errors: list[str] = []
for path in required:
    if not path.exists():
        errors.append(f"missing: {path.relative_to(ROOT)}")

constitution = (DOCS / "ALICE_CONSTITUTION.md").read_text(encoding="utf-8")
for phrase in [
    "Version:** 1.1.0",
    "Capability-First Governance",
    "Continuous Learning and Memory",
    "Self-Modification",
    "minimal authority kernel",
    "No Legacy Veto and Implementation Supremacy",
]:
    if phrase not in constitution:
        errors.append(f"constitution missing marker: {phrase}")

roadmap = (DOCS / "ROADMAP.md").read_text(encoding="utf-8")
for phase in range(16):
    if f"Phase {phase} " not in roadmap and f"Phase {phase} —" not in roadmap:
        errors.append(f"roadmap missing Phase {phase}")

for policy in [
    POLICIES / "authority_kernel_policy.json",
    POLICIES / "lifelong_learning_policy.json",
    POLICIES / "capability_profiles.json",
    POLICIES / "phase_scope_registry.json",
]:
    try:
        payload = json.loads(policy.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid JSON {policy.name}: {exc}")
        continue
    if policy.name == "capability_profiles.json":
        profiles = payload.get("profiles", {})
        if not profiles:
            errors.append("capability profile registry is empty")
        for profile_id, profile in profiles.items():
            if profile.get("capability_ceiling") is not False:
                errors.append(f"profile {profile_id} does not declare capability_ceiling=false")

forbidden = {
    "docs/ALICE_CONSTITUTION.md": ["Current effective revision:", "Amendment I —"],
    "docs/PERMISSION_MODEL.md": [
        "A.L.I.C.E. follows **default deny**",
        "The narrowest sufficient permission is used",
    ],
}
for rel, phrases in forbidden.items():
    text = (ROOT / rel).read_text(encoding="utf-8")
    for phrase in phrases:
        if phrase in text:
            errors.append(f"legacy barrier remains in {rel}: {phrase}")


product_roadmap = (DOCS / "FRIDAY_ROADMAP.md").read_text(encoding="utf-8")
for phrase in ["Phase 5.0", "Phase 6.5", "Identity Capsule", "closed alpha"]:
    if phrase not in product_roadmap:
        errors.append(f"Friday roadmap missing marker: {phrase}")

product_lines = json.loads((POLICIES / "product_lines.json").read_text(encoding="utf-8"))
if product_lines.get("shared_kernel", {}).get("starts_at_phase") != "5.0":
    errors.append("shared-kernel extraction must start at Phase 5.0")
if product_lines.get("shared_kernel", {}).get("formal_repository_split_gate") != "6.5":
    errors.append("Friday repository split must occur at Phase 6.5")
if product_lines.get("products", {}).get("friday", {}).get("full_capability_parity_with_alice") is not True:
    errors.append("consumer product must have full capability parity with A.L.I.C.E.")
if product_lines.get("products", {}).get("friday", {}).get("host_selects_assistant_name") is not True:
    errors.append("consumer hosts must select assistant names")
if product_lines.get("separation_rules", {}).get("phase_1_to_4_files_are_migratable") is not True:
    errors.append("Phase 1–4 files must remain migratable")

if errors:
    print("Governance and evolvability validation failed:")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)

print("A.L.I.C.E. Governance 1.1 and evolvability validation passed.")
