from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cognitive_kernel import (  # noqa: E402
    CognitiveKernelReleaseAttestationPolicy,
    load_cognitive_kernel_release_attestation_policy,
)

IMPLEMENTED_CAPABILITIES = (
    "mission_graph.v1",
    "semantic_router.v1",
    "result_capsule.v1",
    "traceback_engine.v1",
    "attention_policy.v1",
    "workspace_projection.v1",
    "adaptive_compositor.v1",
    "host_window_override.v1",
    "speaker_context.v1",
    "guest_session.v1",
    "guest_grant.v1",
    "release_attestation.v1",
)
REQUIRED_RELEASE_BINDINGS = (
    "source_commit",
    "kernel_version",
    "dependency_lock_digest",
    "artifact_hashes",
    "model_pack_versions",
    "schema_versions",
    "policy_versions",
    "migration_manifest",
    "evaluation_bundle_digest",
    "deployment_manifest",
    "rollback_manifest",
    "release_channel",
)


def load(relative: str) -> dict[str, object]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return value


def main() -> int:
    errors: list[str] = []
    required_files = (
        "docs/PHASE_5_PARITY_RELEASE_ATTESTATION_CONTRACTS.md",
        "policies/cognitive_kernel_release_attestation_policy.json",
        "policies/friday_release_attestation_schema.json",
        "src/cognitive_kernel/release.py",
        "src/cognitive_kernel/release_policy.py",
    )
    for relative in required_files:
        if not (ROOT / relative).is_file():
            errors.append(f"missing: {relative}")

    try:
        parity = load("policies/capability_parity_ledger.json")
        if parity.get("version") != "2.1.0":
            errors.append("capability parity ledger must be version 2.1.0")
        eligibility = parity.get("eligibility_model")
        if not isinstance(eligibility, dict):
            errors.append("capability parity ledger lacks eligibility_model")
        else:
            expected = {
                "kernel_contract_implementation_does_not_equal_product_capability": True,
                "alice_capability_gained_required_before_friday_eligibility": True,
                "friday_pre_phase_6_5_state": "foundation_only",
                "owner_override_schema": "policies/rayan_owner_override_schema.json",
            }
            for key, value in expected.items():
                if eligibility.get(key) != value:
                    errors.append(f"eligibility_model {key!r} must be {value!r}")
        capabilities = parity.get("capabilities")
        if not isinstance(capabilities, dict):
            errors.append("capability parity ledger capabilities must be an object")
            capabilities = {}
        for capability in IMPLEMENTED_CAPABILITIES:
            entry = capabilities.get(capability)
            if not isinstance(entry, dict):
                errors.append(f"missing parity entry: {capability}")
                continue
            expected = {
                "kernel_status": "implemented_phase5_contract",
                "kernel_contract_implemented": True,
                "alice_capability_gained": False,
                "friday_eligibility_status": (
                    "not_eligible_alice_capability_not_yet_gained"
                ),
                "pre_phase_6_5_status": "foundation_only",
            }
            for key, value in expected.items():
                if entry.get(key) != value:
                    errors.append(
                        f"parity {capability} {key!r} must be {value!r}"
                    )
            if not isinstance(entry.get("evidence_bundle"), dict):
                errors.append(f"parity {capability} lacks evidence_bundle")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid capability parity ledger: {exc}")

    try:
        profiles = load("policies/capability_profiles.json")
        if profiles.get("version") != "1.2.0":
            errors.append("capability profiles must be version 1.2.0")
        profile_map = profiles.get("profiles")
        if not isinstance(profile_map, dict):
            raise ValueError("profiles must be an object")
        kernel = profile_map.get("kernel.phase5.foundation")
        friday = profile_map.get("friday.production.dual_approval")
        if not isinstance(kernel, dict) or not isinstance(friday, dict):
            raise ValueError("required profiles are missing")
        kernel_caps = kernel.get("capabilities")
        friday_caps = friday.get("capabilities")
        if not isinstance(kernel_caps, dict) or not isinstance(friday_caps, dict):
            raise ValueError("profile capabilities must be objects")
        if kernel_caps.get("release_attestation_verification_implemented") is not True:
            errors.append("kernel profile must mark release verifier implemented")
        for key in (
            "release_signing_implemented",
            "release_deployment_implemented",
            "approval_generation_implemented",
        ):
            if kernel_caps.get(key) is not False:
                errors.append(f"kernel profile {key} must be false")
        for key in (
            "candidate_research_allowed",
            "candidate_implementation_allowed",
            "uneligible_capability_testing_allowed",
        ):
            if friday_caps.get(key) is not False:
                errors.append(f"Friday production profile {key} must be false")
        for key in (
            "maintenance_allowed",
            "product_experience_research_allowed",
            "upstream_proposal_research_allowed",
            "eligible_capability_productization_allowed",
            "alice_capability_precedent_required",
            "exact_artifact_audit_required",
            "exact_candidate_owner_approval_required",
            "phase_6_5_foundation_only",
        ):
            if friday_caps.get(key) is not True:
                errors.append(f"Friday production profile {key} must be true")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid capability profiles: {exc}")

    try:
        production = load("policies/friday_production_governance.json")
        if tuple(production.get("required_bindings", ())) != REQUIRED_RELEASE_BINDINGS:
            errors.append("Friday production required bindings changed")
        schema = load("policies/friday_release_attestation_schema.json")
        required = schema.get("required")
        if not isinstance(required, list):
            errors.append("Friday release schema required must be a list")
            required = []
        for binding in REQUIRED_RELEASE_BINDINGS:
            if binding not in required:
                errors.append(f"Friday release schema missing {binding}")
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            errors.append("Friday release schema properties must be an object")
        else:
            schema_version = properties.get("schema_version")
            if not isinstance(schema_version, dict) or schema_version.get("const") != "2.0.0":
                errors.append("Friday release schema must be version 2.0.0")
            if "alice_audit_attestation" not in properties:
                errors.append("Friday release schema lacks A.L.I.C.E. audit")
            if "rayan_approval" not in properties:
                errors.append("Friday release schema lacks owner approval")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid release schema or production policy: {exc}")

    try:
        policy: CognitiveKernelReleaseAttestationPolicy = (
            load_cognitive_kernel_release_attestation_policy(
                repository_root=ROOT
            )
        )
        if policy.version != "0.5.0" or policy.milestone != "P5.0f":
            errors.append("release-attestation policy version or milestone changed")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid release-attestation policy: {exc}")

    friday_source = ROOT / "src" / "friday"
    if friday_source.exists():
        errors.append("Friday product source entered the A.L.I.C.E. repository")

    if errors:
        print("P5.0f parity and release validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "P5.0f parity synchronization and release-attestation validation passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
