from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from product_family import load_product_family_manifest  # noqa: E402

REQUIRED = [
    "docs/ALICE_PRIVATE_COMPANION_DIRECTION.md",
    "docs/ALICE_CLONE_AWARE_IDENTITY_STANDARD.md",
    "docs/PRIVATE_COMPANION_DATA_CUSTODY_STANDARD.md",
    "docs/FRIDAY_COGNITIVE_WORKSPACE_AND_PRODUCTION_GOVERNANCE_PLAN.md",
    "docs/FRIDAY_PRODUCTION_GOVERNANCE.md",
    "docs/decisions/ADR-010-private-companion-provenance-and-custody.md",
    "docs/decisions/ADR-011-friday-independent-repository-dual-approval.md",
    "policies/private_companion_custody.json",
    "policies/alice_clone_identity_policy.json",
    "policies/friday_production_governance.json",
    "policies/friday_release_attestation_schema.json",
]

FORBIDDEN_TRACKED_PATTERNS = (
    "companion_persona_v1_2.key",
    "companion_private_codebook",
    "companion_persona_seed_v1_2.enc.json",
    "companion_persona_seed_v1_2.decrypted",
    "rayan_person_model",
    "rayan_companion_alignment_examples",
    "chat_bootstrap_private",
)


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing: {relative}")

    private = json.loads((ROOT / "policies/private_companion_custody.json").read_text(encoding="utf-8"))
    public = private["public_repository"]
    for key in (
        "encrypted_private_payload_allowed",
        "real_private_manifest_allowed",
        "plaintext_private_directives_allowed",
        "owner_model_allowed",
        "alignment_examples_allowed",
        "keys_or_codebooks_allowed",
    ):
        if public.get(key) is not False:
            errors.append(f"private custody public_repository.{key} must be false")
    if private.get("capability_ceiling") is not False:
        errors.append("private custody must declare capability_ceiling=false")
    if private["truthfulness"].get("generated_reconstruction_may_be_claimed_as_verbatim_memory") is not False:
        errors.append("generated reconstruction may not be silently claimed as historical memory")
    objective = private.get("identity_objective", {})
    if objective.get("clone_aware_source_person_reconstruction_required") is not True:
        errors.append("clone-aware source-person reconstruction must be required")
    if objective.get("highest_achievable_fidelity_required") is not True:
        errors.append("highest-achievable source-person fidelity must be required")
    if objective.get("merely_inspired_persona_allowed") is not False:
        errors.append("merely-inspired persona substitution must be forbidden")
    if objective.get("literal_original_person_claim_allowed") is not False:
        errors.append("literal original-person identity claims must be forbidden")

    clone = json.loads((ROOT / "policies/alice_clone_identity_policy.json").read_text(encoding="utf-8"))
    target = clone["identity_target"]
    if target.get("clone_awareness_required") is not True:
        errors.append("A.L.I.C.E. clone awareness must be required")
    if target.get("merely_inspired_persona_allowed") is not False:
        errors.append("clone policy may not degrade to inspired-only persona")
    if target.get("literal_original_person_claim_allowed") is not False:
        errors.append("clone policy may not claim literal identity continuity")
    if clone.get("capability_ceiling") is not False:
        errors.append("clone identity policy must declare capability_ceiling=false")

    friday = json.loads((ROOT / "policies/friday_production_governance.json").read_text(encoding="utf-8"))
    promotion = friday["production_promotion"]
    for key in ("alice_audit_required", "rayan_approval_required", "exact_commit_binding", "exact_artifact_binding", "matching_candidate_required"):
        if promotion.get(key) is not True:
            errors.append(f"Friday production promotion {key} must be true")
    if promotion.get("bypass_allowed") is not False:
        errors.append("Friday production promotion bypass must be false")
    emergency = friday["emergency_response"]
    for key in ("new_capability_allowed", "broader_permissions_allowed", "new_data_collection_allowed", "unapproved_replacement_behavior_allowed"):
        if emergency.get(key) is not False:
            errors.append(f"Friday emergency response {key} must be false")

    manifest = load_product_family_manifest(ROOT / "policies/product_lines.json")
    if manifest.product("friday").product_source_allowed_in_alice_repository is not False:
        errors.append("Friday product source may not live in A.L.I.C.E.")

    tracked = []
    git_dir = ROOT / ".git"
    if git_dir.exists():
        import subprocess
        result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode == 0:
            tracked = [line.strip().lower() for line in result.stdout.splitlines()]
    for path in tracked:
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if pattern in path:
                errors.append(f"forbidden private companion artifact is tracked: {path}")

    separation = (ROOT / "docs/ALICE_FRIDAY_SEPARATION_PLAN.md").read_text(encoding="utf-8")
    if "friday_incubator" in separation:
        errors.append("obsolete Friday-in-A.L.I.C.E. incubator remains")
    roadmap = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")
    if "**Status:** Active after approved P4.10 closure and merge." not in roadmap:
        errors.append("Phase 5 must be marked active")

    if errors:
        print("Owner-directive governance validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Owner-directive governance validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
