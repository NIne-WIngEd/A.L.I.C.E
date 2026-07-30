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
    DOCS / "STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md",
    DOCS / "AUTONOMY_AND_JUDGMENT_POLICY.md",
    DOCS / "CONSTRAINT_REGISTRY.md",
    DOCS / "CAPABILITY_UNBLOCKING_POLICY.md",
    DOCS / "PHASE_SCOPE_POLICY.md",
    DOCS / "IMPLEMENTATION_EVOLVABILITY_STANDARD.md",
    DOCS / "PROJECT_WIDE_CAPABILITY_AUDIT.md",
    DOCS / "PUBLIC_REPOSITORY_AUDIT_MANIFEST.md",
    DOCS / "MEGA_ARCHITECTURE_MIGRATION.md",
    DOCS / "CHAT_ARCHITECTURE_DECISION_LEDGER.md",
    DOCS / "decisions" / "ADR-008-aggressive-capture-selective-retention.md",
    POLICIES / "authority_kernel_policy.json",
    POLICIES / "lifelong_learning_policy.json",
    POLICIES / "storage_lifecycle_policy.json",
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
for phrase in ["aggressive temporary capture", "content-addressed", "representative replay"]:
    if phrase not in roadmap.lower():
        errors.append(f"roadmap missing storage marker: {phrase}")
for policy in [
    POLICIES / "authority_kernel_policy.json",
    POLICIES / "lifelong_learning_policy.json",
    POLICIES / "storage_lifecycle_policy.json",
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

storage = json.loads((POLICIES / "storage_lifecycle_policy.json").read_text(encoding="utf-8"))
if storage.get("doctrine") != "aggressive_temporary_capture_selective_durable_retention":
    errors.append("storage doctrine must require aggressive temporary capture and selective durable retention")
if storage.get("permanent_compact_event_ledger") is not True:
    errors.append("storage policy must require a permanent compact event ledger")
if storage.get("permanent_full_payload_retention_default") is not False:
    errors.append("full-payload permanent retention must not be the default")
content_addressing = storage.get("content_addressing", {})
if content_addressing.get("algorithm") != "sha256":
    errors.append("storage policy must use sha256 content addressing")
if content_addressing.get("deduplication_scope") != "host_instance_and_encryption_domain":
    errors.append("deduplication must be scoped by host instance and encryption domain")
if content_addressing.get("cross_host_deduplication_allowed") is not False:
    errors.append("cross-host deduplication violates host-isolation policy")
required_tiers = {"ledger", "raw_buffer", "hot", "warm", "cold", "quarantine", "deleted"}
if not required_tiers.issubset(storage.get("tiers", {})):
    errors.append("storage policy is missing required lifecycle tiers")
if storage.get("replay", {}).get("selection") != "representative_budgeted_and_versioned":
    errors.append("storage policy must use representative, budgeted, versioned replay")
if storage.get("replay", {}).get("keep_every_event_equally") is not False:
    errors.append("replay policy must not retain every event equally")
if storage.get("capacity", {}).get("protected_artifact_silent_deletion_allowed") is not False:
    errors.append("storage pressure must not silently delete protected artifacts")
if storage.get("backup", {}).get("restore_testing_required") is not True:
    errors.append("backup restoration testing must be required")
if storage.get("deletion", {}).get("deliberate_relearning_of_deleted_payload_allowed") is not False:
    errors.append("deleted payloads must not be deliberately relearned")

learning = json.loads((POLICIES / "lifelong_learning_policy.json").read_text(encoding="utf-8"))
if learning.get("capture_mode") != "aggressive_temporary_capture":
    errors.append("lifelong learning policy must use aggressive temporary capture")
if learning.get("permanent_compact_event_ledger") is not True:
    errors.append("lifelong learning policy must require a compact event ledger")
if learning.get("replay_selection") != "representative_budgeted_and_versioned":
    errors.append("lifelong learning policy must use representative replay")

product_roadmap = (DOCS / "FRIDAY_ROADMAP.md").read_text(encoding="utf-8")
for phrase in ["Phase 5.0", "Phase 6.5", "Identity Capsule", "closed alpha"]:
    if phrase not in product_roadmap:
        errors.append(f"Friday roadmap missing marker: {phrase}")
product_lines = json.loads((POLICIES / "product_lines.json").read_text(encoding="utf-8"))
if product_lines.get("shared_kernel", {}).get("starts_at_phase") != "5.0":
    errors.append("shared-kernel extraction must start at Phase 5.0")
if product_lines.get("shared_kernel", {}).get("formal_repository_split_gate") != "6.5":
    errors.append("Friday repository split must occur at Phase 6.5")
required_scopes = set(product_lines.get("shared_kernel", {}).get("required_scopes", []))
for scope in ["content_digest", "retention_class", "storage_tier", "deletion_lineage"]:
    if scope not in required_scopes:
        errors.append(f"shared kernel storage scope missing: {scope}")
if product_lines.get("products", {}).get("friday", {}).get("full_capability_parity_with_alice") is not True:
    errors.append("consumer product must have full capability parity with A.L.I.C.E.")
if product_lines.get("products", {}).get("friday", {}).get("host_selects_assistant_name") is not True:
    errors.append("consumer hosts must select assistant names")
if product_lines.get("products", {}).get("friday", {}).get("local_storage_lifecycle_required") is not True:
    errors.append("consumer product must implement the local storage lifecycle")
if product_lines.get("products", {}).get("friday", {}).get("cross_host_deduplication_allowed") is not False:
    errors.append("consumer product must prohibit cross-host deduplication")
if product_lines.get("separation_rules", {}).get("phase_1_to_4_files_are_migratable") is not True:
    errors.append("Phase 1–4 files must remain migratable")
if product_lines.get("separation_rules", {}).get("cross_host_deduplication_allowed") is not False:
    errors.append("product separation rules must prohibit cross-host deduplication")

parity = json.loads((POLICIES / "capability_parity_ledger.json").read_text(encoding="utf-8"))
if "storage_lifecycle_and_replay" not in parity.get("capabilities", {}):
    errors.append("capability parity ledger must include storage lifecycle and replay")

# final-architecture-cleanup-v1
constitution_status_markers = [
    "**Status:** Ratified and effective",
    "**Ratification:** Ratified by explicit owner approval and repository merge.",
]
for phrase in constitution_status_markers:
    if phrase not in constitution:
        errors.append(f"constitution ratification marker missing: {phrase}")

permissions_text = (POLICIES / "permissions.yaml").read_text(encoding="utf-8")
for phrase in [
    'version: "2.0.0"',
    'policy_model: "mission_scoped_autonomy"',
    'activation_model: "capability_profile_plus_mission"',
    "A5: autonomous_production_and_self_evolution",
    "A6: constitutional_and_authority_kernel",
    "id: code.self_modify_candidate",
    "id: model.train_candidate",
    "id: production.deploy_or_merge",
    "id: model.promote_production",
]:
    if phrase not in permissions_text:
        errors.append(f"mission authority marker missing: {phrase}")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for phrase in [
    "P4.5a citation-bound grounding is merged",
    "Aggressive temporary capture",
    "Personal Cognitive Kernel extraction starts at Phase 5.0",
    "Friday is not the user-facing assistant name",
]:
    if phrase not in readme:
        errors.append(f"README architecture marker missing: {phrase}")

evaluation = (DOCS / "EVALUATION_CHARTER.md").read_text(encoding="utf-8")
for phrase in [
    "**Status:** Ratified cross-phase evaluation charter",
    "catastrophic forgetting",
    "retention-value prediction",
    "representative replay quality",
    "Product-family parity",
    "backup and restore drills",
]:
    if phrase not in evaluation:
        errors.append(f"evaluation charter marker missing: {phrase}")
for phase in range(5, 16):
    if f"Phase {phase}" not in evaluation:
        errors.append(f"evaluation charter missing Phase {phase}")

scope = (DOCS / "SCOPE_AND_NON_GOALS.md").read_text(encoding="utf-8")
for phrase in [
    "HISTORICAL RELEASE SCOPE",
    "**Scope kind:** Historical compatibility document",
    "**Capability ceiling:** false",
    "intended future directions or research programs",
]:
    if phrase not in scope:
        errors.append(f"historical scope marker missing: {phrase}")
# phase4-post-phase-integrity-v1
phase4_post_audit_markers = {
    ROOT / "README.md": [
        "P4.10 operational live-public-information closure",
        "Phase 5.0",
        "blocked until P4.10",
    ],
    DOCS / "ROADMAP.md": [
        "P4.10 operational live-public-information closure",
        "Blocked until P4.10",
    ],
    DOCS / "CAPABILITY_CATALOG.md": [
        "IN DEVELOPMENT / P4.10 LIVE ACCEPTANCE",
    ],
    DOCS / "PHASE_4_POST_PHASE_AUDIT.md": [
        "operational live-public-information closure remains required",
    ],
    DOCS / "PHASE_BOUNDARY_AUDIT_STANDARD.md": [
        "Every top-level phase ends with an adversarial audit",
    ],
    DOCS / "decisions" / "ADR-009-phase4-live-public-information-closure.md": [
        "Phase 5 is blocked until P4.10",
    ],
}
for path, phrases in phase4_post_audit_markers.items():
    if not path.is_file():
        errors.append(f"phase-boundary file missing: {path.relative_to(ROOT)}")
        continue
    body = path.read_text(encoding="utf-8")
    for phrase in phrases:
        if phrase not in body:
            errors.append(
                f"phase-boundary marker missing in {path.relative_to(ROOT)}: {phrase}"
            )

live_acceptance_policy = POLICIES / "information_live_provider_acceptance_policy.json"
if not live_acceptance_policy.is_file():
    errors.append("live-provider acceptance policy missing")
else:
    live_acceptance_body = live_acceptance_policy.read_text(encoding="utf-8")
    for phrase in [
        '"milestone": "P4.10"',
        '"initial_query_classifications": [',
        '"PUBLIC"',
        '"phase5_start_gate": "blocked_until_p4_10_approved"',
        '"capability_ceiling": false',
    ]:
        if phrase not in live_acceptance_body:
            errors.append(f"live-provider acceptance marker missing: {phrase}")
if errors:
    print("Governance and evolvability validation failed:")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)

print("A.L.I.C.E. Governance 1.1 and evolvability validation passed.")
