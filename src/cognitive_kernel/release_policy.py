"""Loader for P5.0f parity and exact-artifact release policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    require_identifier,
    require_schema_version,
    require_text,
)
from .release import (
    AUDIT_DECISIONS,
    AUTHORIZED_AUDIT_DECISIONS,
    AUTHORIZED_OWNER_DECISIONS,
    OWNER_APPROVAL_DECISIONS,
    RELEASE_CHANNELS,
)

_POLICY_KEYS = {
    "cognitive_kernel_release_attestation_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "required_capabilities",
    "allowed_audit_decisions",
    "allowed_owner_decisions",
    "authorized_audit_decisions",
    "authorized_owner_decisions",
    "allowed_release_channels",
    "required_bindings",
    "invariants",
    "capability_ceiling",
}
_INVARIANT_KEYS = {
    "exact_source_commit_binding",
    "exact_artifact_manifest_binding",
    "exact_evaluation_bundle_binding",
    "exact_deployment_manifest_binding",
    "alice_audit_required",
    "owner_approval_required",
    "matching_candidate_required",
    "parity_contract_status_does_not_imply_alice_capability_gained",
    "friday_eligibility_requires_alice_capability_gained",
    "friday_pre_phase_6_5_foundation_only",
    "release_signing_implemented",
    "deployment_implemented",
    "approval_generation_implemented",
    "friday_product_source_implemented",
    "private_payload_allowed",
}
_REQUIRED_CAPABILITIES = ("release_attestation.v1",)
_REQUIRED_BINDINGS = (
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
_IMPLEMENTED_CAPABILITIES = (
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


@dataclass(frozen=True)
class CognitiveKernelReleaseAttestationPolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_capabilities: tuple[str, ...]
    allowed_audit_decisions: tuple[str, ...]
    allowed_owner_decisions: tuple[str, ...]
    authorized_audit_decisions: tuple[str, ...]
    authorized_owner_decisions: tuple[str, ...]
    allowed_release_channels: tuple[str, ...]
    required_bindings: tuple[str, ...]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "release-attestation policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        require_schema_version(self.version, "version")
        if self.version != "0.5.0":
            raise CognitiveKernelContractError(
                "release-attestation contract version changed"
            )
        if self.phase != "5" or self.milestone != "P5.0f":
            raise CognitiveKernelContractError(
                "release-attestation policy milestone changed"
            )
        if self.status != "contract_foundation":
            raise CognitiveKernelContractError(
                "release-attestation policy status is invalid"
            )
        if self.required_capabilities != _REQUIRED_CAPABILITIES:
            raise CognitiveKernelContractError(
                "release-attestation capabilities changed"
            )
        comparisons = (
            (set(self.allowed_audit_decisions), AUDIT_DECISIONS, "audit decisions"),
            (
                set(self.allowed_owner_decisions),
                OWNER_APPROVAL_DECISIONS,
                "owner decisions",
            ),
            (
                set(self.authorized_audit_decisions),
                AUTHORIZED_AUDIT_DECISIONS,
                "authorized audit decisions",
            ),
            (
                set(self.authorized_owner_decisions),
                AUTHORIZED_OWNER_DECISIONS,
                "authorized owner decisions",
            ),
            (
                tuple(self.allowed_release_channels),
                RELEASE_CHANNELS,
                "release channel order",
            ),
            (
                tuple(self.required_bindings),
                _REQUIRED_BINDINGS,
                "required release bindings",
            ),
        )
        for actual, expected, label in comparisons:
            if actual != expected:
                raise CognitiveKernelContractError(f"{label} changed")
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError(
                "release-attestation invariants changed"
            )
        required_true = {
            "exact_source_commit_binding",
            "exact_artifact_manifest_binding",
            "exact_evaluation_bundle_binding",
            "exact_deployment_manifest_binding",
            "alice_audit_required",
            "owner_approval_required",
            "matching_candidate_required",
            "parity_contract_status_does_not_imply_alice_capability_gained",
            "friday_eligibility_requires_alice_capability_gained",
            "friday_pre_phase_6_5_foundation_only",
        }
        required_false = _INVARIANT_KEYS - required_true
        for key in required_true:
            if self.invariants.get(key) is not True:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be true"
                )
        for key in required_false:
            if self.invariants.get(key) is not False:
                raise CognitiveKernelContractError(
                    f"invariant {key} must be false"
                )
        if self.capability_ceiling is not False:
            raise CognitiveKernelContractError(
                "release-attestation policy may not be a capability ceiling"
            )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CognitiveKernelContractError(
            f"could not load JSON policy: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(
            f"JSON policy must be an object: {path}"
        )
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CognitiveKernelContractError(
            f"{field} keys changed; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CognitiveKernelContractError(
            f"{field} must be a non-empty list"
        )
    result = tuple(require_text(item, field, maximum=256) for item in value)
    if len(set(result)) != len(result):
        raise CognitiveKernelContractError(
            f"{field} may not contain duplicates"
        )
    return result


def _require_dict(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(f"{field} must be an object")
    return value


def load_cognitive_kernel_release_attestation_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelReleaseAttestationPolicy:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = (
        Path(path).resolve()
        if path is not None
        else root
        / "policies"
        / "cognitive_kernel_release_attestation_policy.json"
    )
    payload = _read_json(source)
    _exact_keys(payload, _POLICY_KEYS, "release-attestation policy")
    invariants = _require_dict(payload["invariants"], "invariants")
    _exact_keys(invariants, _INVARIANT_KEYS, "release-attestation invariants")
    if any(not isinstance(value, bool) for value in invariants.values()):
        raise CognitiveKernelContractError(
            "release-attestation invariants must be booleans"
        )
    policy = CognitiveKernelReleaseAttestationPolicy(
        schema_version=int(
            payload["cognitive_kernel_release_attestation_policy_schema_version"]
        ),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(payload["milestone"], "milestone", maximum=32),
        status=require_identifier(payload["status"], "status"),
        required_capabilities=_sequence(
            payload["required_capabilities"], "required_capabilities"
        ),
        allowed_audit_decisions=_sequence(
            payload["allowed_audit_decisions"], "allowed_audit_decisions"
        ),
        allowed_owner_decisions=_sequence(
            payload["allowed_owner_decisions"], "allowed_owner_decisions"
        ),
        authorized_audit_decisions=_sequence(
            payload["authorized_audit_decisions"],
            "authorized_audit_decisions",
        ),
        authorized_owner_decisions=_sequence(
            payload["authorized_owner_decisions"],
            "authorized_owner_decisions",
        ),
        allowed_release_channels=_sequence(
            payload["allowed_release_channels"], "allowed_release_channels"
        ),
        required_bindings=_sequence(
            payload["required_bindings"], "required_bindings"
        ),
        invariants={str(key): bool(value) for key, value in invariants.items()},
        capability_ceiling=bool(payload["capability_ceiling"]),
        digest=canonical_sha256(payload),
        source_path=source,
    )
    policy.validate()

    previous = _read_json(
        root
        / "policies"
        / "cognitive_kernel_interaction_authority_policy.json"
    )
    if previous.get("version") != "0.4.0" or previous.get("milestone") != "P5.0e":
        raise CognitiveKernelContractError(
            "unexpected interaction-authority contract baseline"
        )

    production = _read_json(
        root / "policies" / "friday_production_governance.json"
    )
    if tuple(production.get("required_bindings", ())) != _REQUIRED_BINDINGS:
        raise CognitiveKernelContractError(
            "Friday production required bindings changed"
        )
    promotion = _require_dict(
        production.get("production_promotion"), "production_promotion"
    )
    for key in (
        "alice_audit_required",
        "rayan_approval_required",
        "exact_commit_binding",
        "exact_artifact_binding",
        "matching_candidate_required",
    ):
        if promotion.get(key) is not True:
            raise CognitiveKernelContractError(
                f"Friday production promotion no longer requires {key}"
            )

    flagship = _read_json(
        root / "policies" / "alice_friday_flagship_governance.json"
    )
    gate = _require_dict(
        flagship.get("default_new_capability_gate"),
        "default_new_capability_gate",
    )
    if gate.get("friday_first_capability_allowed") is not False:
        raise CognitiveKernelContractError(
            "Friday-first capability work is not denied"
        )
    if gate.get("alice_must_gain_capability_before_friday_eligibility") is not True:
        raise CognitiveKernelContractError(
            "A.L.I.C.E. capability precedent is not required"
        )

    readiness = _read_json(
        root / "policies" / "friday_pre_phase_6_5_gate.json"
    )
    if readiness.get("state") != "foundation_only_waiting_for_alice_phase_6_5":
        raise CognitiveKernelContractError(
            "Friday is not foundation-only before Phase 6.5"
        )

    parity = _read_json(root / "policies" / "capability_parity_ledger.json")
    eligibility = _require_dict(parity.get("eligibility_model"), "eligibility_model")
    if eligibility.get(
        "kernel_contract_implementation_does_not_equal_product_capability"
    ) is not True:
        raise CognitiveKernelContractError(
            "parity ledger collapses kernel and product capability state"
        )
    if eligibility.get(
        "alice_capability_gained_required_before_friday_eligibility"
    ) is not True:
        raise CognitiveKernelContractError(
            "parity ledger does not require A.L.I.C.E. capability precedent"
        )
    capabilities = _require_dict(parity.get("capabilities"), "capabilities")
    for capability in _IMPLEMENTED_CAPABILITIES:
        entry = _require_dict(capabilities.get(capability), capability)
        if entry.get("kernel_status") != "implemented_phase5_contract":
            raise CognitiveKernelContractError(
                f"parity ledger has stale kernel status for {capability}"
            )
        if entry.get("kernel_contract_implemented") is not True:
            raise CognitiveKernelContractError(
                f"parity ledger does not mark {capability} implemented"
            )
        if entry.get("alice_capability_gained") is not False:
            raise CognitiveKernelContractError(
                f"parity ledger incorrectly marks {capability} gained"
            )
        if entry.get("friday_eligibility_status") != (
            "not_eligible_alice_capability_not_yet_gained"
        ):
            raise CognitiveKernelContractError(
                f"parity ledger incorrectly makes {capability} Friday-eligible"
            )
        if entry.get("pre_phase_6_5_status") != "foundation_only":
            raise CognitiveKernelContractError(
                f"parity ledger bypasses the Phase 6.5 gate for {capability}"
            )
        if not isinstance(entry.get("evidence_bundle"), dict):
            raise CognitiveKernelContractError(
                f"parity ledger lacks evidence for {capability}"
            )

    profiles = _read_json(root / "policies" / "capability_profiles.json")
    profile_map = _require_dict(profiles.get("profiles"), "profiles")
    friday_profile = _require_dict(
        profile_map.get("friday.production.dual_approval"),
        "friday.production.dual_approval",
    )
    friday_caps = _require_dict(
        friday_profile.get("capabilities"),
        "friday.production.dual_approval.capabilities",
    )
    for key in (
        "candidate_research_allowed",
        "candidate_implementation_allowed",
        "uneligible_capability_testing_allowed",
    ):
        if friday_caps.get(key) is not False:
            raise CognitiveKernelContractError(
                f"Friday capability profile incorrectly enables {key}"
            )
    for key in (
        "alice_capability_precedent_required",
        "exact_artifact_audit_required",
        "exact_candidate_owner_approval_required",
        "phase_6_5_foundation_only",
    ):
        if friday_caps.get(key) is not True:
            raise CognitiveKernelContractError(
                f"Friday capability profile does not require {key}"
            )

    schema = _read_json(
        root / "policies" / "friday_release_attestation_schema.json"
    )
    properties = _require_dict(schema.get("properties"), "schema.properties")
    if _require_dict(properties.get("schema_version"), "schema_version").get(
        "const"
    ) != "2.0.0":
        raise CognitiveKernelContractError(
            "Friday release-attestation schema version changed"
        )
    required = schema.get("required")
    if not isinstance(required, list):
        raise CognitiveKernelContractError(
            "Friday release-attestation required list is invalid"
        )
    for binding in _REQUIRED_BINDINGS:
        if binding not in required:
            raise CognitiveKernelContractError(
                f"Friday release schema is missing {binding}"
            )
    return policy
