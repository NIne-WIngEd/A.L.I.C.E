"""Loader for P5.0e speaker, guest, and authority policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .authority import (
    AUTHORITY_CAPABILITIES,
    AUTHORITY_DECISIONS,
    AUTHORITY_EVIDENCE_CLASSES,
    AUTHORITY_LEVELS,
    CONSEQUENCE_CLASSES,
    OWNER_ONLY_CAPABILITIES,
)
from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    require_identifier,
    require_schema_version,
    require_text,
)
from .guest import (
    GUEST_CAPABILITIES,
    GUEST_DENIED_DOMAINS,
    GUEST_GRANT_STATUSES,
    GUEST_SESSION_MODES,
    GUEST_SESSION_STATUSES,
)
from .speaker import (
    SPEAKER_AUTHORITY_CEILINGS,
    SPEAKER_EVIDENCE_CLASSES,
    SPEAKER_STATES,
    SPEAKER_TRUST_STATES,
)

_POLICY_KEYS = {
    "cognitive_kernel_interaction_authority_policy_schema_version",
    "policy_id",
    "version",
    "phase",
    "milestone",
    "status",
    "required_capabilities",
    "allowed_speaker_states",
    "allowed_speaker_trust_states",
    "allowed_speaker_evidence_classes",
    "allowed_speaker_authority_ceilings",
    "allowed_guest_session_modes",
    "allowed_guest_session_statuses",
    "allowed_guest_grant_statuses",
    "allowed_guest_capabilities",
    "mandatory_guest_denied_domains",
    "allowed_authority_levels",
    "allowed_authority_decisions",
    "allowed_authority_evidence_classes",
    "allowed_consequence_classes",
    "allowed_authority_capabilities",
    "owner_only_capabilities",
    "invariants",
    "capability_ceiling",
}
_INVARIANT_KEYS = {
    "product_host_scope_required",
    "speaker_context_metadata_only",
    "raw_audio_forbidden",
    "voice_profile_forbidden",
    "voice_alone_privileged_authority_forbidden",
    "stronger_authentication_required_for_high_consequence",
    "guest_authority_visible_scoped_expiring_revocable",
    "guest_grants_non_delegable",
    "guest_self_expansion_forbidden",
    "private_views_hidden_for_guest",
    "guest_actions_locally_logged",
    "cross_host_guest_authority_forbidden",
    "owner_only_capabilities_protected",
    "authority_decisions_explainable",
    "biometric_recognition_implemented",
    "microphone_capture_implemented",
    "persistent_guest_store_implemented",
    "autonomous_authority_escalation_implemented",
    "complete_ui_implemented",
    "friday_product_source_implemented",
    "private_payload_allowed",
}
_REQUIRED_CAPABILITIES = (
    "speaker_context.v1",
    "guest_session.v1",
    "guest_grant.v1",
)


@dataclass(frozen=True)
class CognitiveKernelInteractionAuthorityPolicy:
    schema_version: int
    policy_id: str
    version: str
    phase: str
    milestone: str
    status: str
    required_capabilities: tuple[str, ...]
    allowed_speaker_states: tuple[str, ...]
    allowed_speaker_trust_states: tuple[str, ...]
    allowed_speaker_evidence_classes: tuple[str, ...]
    allowed_speaker_authority_ceilings: tuple[str, ...]
    allowed_guest_session_modes: tuple[str, ...]
    allowed_guest_session_statuses: tuple[str, ...]
    allowed_guest_grant_statuses: tuple[str, ...]
    allowed_guest_capabilities: tuple[str, ...]
    mandatory_guest_denied_domains: tuple[str, ...]
    allowed_authority_levels: tuple[str, ...]
    allowed_authority_decisions: tuple[str, ...]
    allowed_authority_evidence_classes: tuple[str, ...]
    allowed_consequence_classes: tuple[str, ...]
    allowed_authority_capabilities: tuple[str, ...]
    owner_only_capabilities: tuple[str, ...]
    invariants: Mapping[str, bool]
    capability_ceiling: bool
    digest: str
    source_path: Path

    def validate(self) -> None:
        if self.schema_version != 1:
            raise CognitiveKernelContractError(
                "interaction-authority policy schema version must be 1"
            )
        require_identifier(self.policy_id, "policy_id")
        require_schema_version(self.version, "version")
        if self.version != "0.4.0":
            raise CognitiveKernelContractError(
                "interaction-authority contract version changed"
            )
        if self.phase != "5" or self.milestone != "P5.0e":
            raise CognitiveKernelContractError(
                "interaction-authority policy milestone changed"
            )
        if self.status != "contract_foundation":
            raise CognitiveKernelContractError(
                "interaction-authority policy status is invalid"
            )
        if self.required_capabilities != _REQUIRED_CAPABILITIES:
            raise CognitiveKernelContractError(
                "interaction-authority capabilities changed"
            )
        comparisons = (
            (set(self.allowed_speaker_states), SPEAKER_STATES, "speaker states"),
            (set(self.allowed_speaker_trust_states), SPEAKER_TRUST_STATES, "speaker trust states"),
            (set(self.allowed_speaker_evidence_classes), SPEAKER_EVIDENCE_CLASSES, "speaker evidence classes"),
            (tuple(self.allowed_speaker_authority_ceilings), SPEAKER_AUTHORITY_CEILINGS, "speaker authority ceiling order"),
            (set(self.allowed_guest_session_modes), GUEST_SESSION_MODES, "guest session modes"),
            (set(self.allowed_guest_session_statuses), GUEST_SESSION_STATUSES, "guest session statuses"),
            (set(self.allowed_guest_grant_statuses), GUEST_GRANT_STATUSES, "guest grant statuses"),
            (set(self.allowed_guest_capabilities), GUEST_CAPABILITIES, "guest capabilities"),
            (set(self.mandatory_guest_denied_domains), GUEST_DENIED_DOMAINS, "guest denied domains"),
            (tuple(self.allowed_authority_levels), AUTHORITY_LEVELS, "authority level order"),
            (set(self.allowed_authority_decisions), AUTHORITY_DECISIONS, "authority decisions"),
            (set(self.allowed_authority_evidence_classes), AUTHORITY_EVIDENCE_CLASSES, "authority evidence classes"),
            (set(self.allowed_consequence_classes), CONSEQUENCE_CLASSES, "consequence classes"),
            (set(self.allowed_authority_capabilities), AUTHORITY_CAPABILITIES, "authority capabilities"),
            (set(self.owner_only_capabilities), OWNER_ONLY_CAPABILITIES, "owner-only capabilities"),
        )
        for actual, expected, label in comparisons:
            if actual != expected:
                raise CognitiveKernelContractError(f"{label} changed")
        if set(self.invariants) != _INVARIANT_KEYS:
            raise CognitiveKernelContractError(
                "interaction-authority invariants changed"
            )
        required_true = {
            "product_host_scope_required",
            "speaker_context_metadata_only",
            "raw_audio_forbidden",
            "voice_profile_forbidden",
            "voice_alone_privileged_authority_forbidden",
            "stronger_authentication_required_for_high_consequence",
            "guest_authority_visible_scoped_expiring_revocable",
            "guest_grants_non_delegable",
            "guest_self_expansion_forbidden",
            "private_views_hidden_for_guest",
            "guest_actions_locally_logged",
            "cross_host_guest_authority_forbidden",
            "owner_only_capabilities_protected",
            "authority_decisions_explainable",
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
                "interaction-authority policy may not be a capability ceiling"
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
            f"{field} keys changed; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
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


def _contains_string(value: object, target: str) -> bool:
    if value == target:
        return True
    if isinstance(value, dict):
        return target in value or any(
            _contains_string(item, target) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_string(item, target) for item in value)
    return False


def load_cognitive_kernel_interaction_authority_policy(
    path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> CognitiveKernelInteractionAuthorityPolicy:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = (
        Path(path).resolve()
        if path is not None
        else root / "policies" / "cognitive_kernel_interaction_authority_policy.json"
    )
    payload = _read_json(source)
    _exact_keys(payload, _POLICY_KEYS, "interaction-authority policy")
    invariants = payload["invariants"]
    if not isinstance(invariants, dict):
        raise CognitiveKernelContractError(
            "interaction-authority invariants must be an object"
        )
    _exact_keys(invariants, _INVARIANT_KEYS, "interaction-authority invariants")
    if any(not isinstance(value, bool) for value in invariants.values()):
        raise CognitiveKernelContractError(
            "interaction-authority invariants must be booleans"
        )
    policy = CognitiveKernelInteractionAuthorityPolicy(
        schema_version=int(
            payload["cognitive_kernel_interaction_authority_policy_schema_version"]
        ),
        policy_id=require_identifier(payload["policy_id"], "policy_id"),
        version=require_schema_version(payload["version"], "version"),
        phase=require_text(payload["phase"], "phase", maximum=8),
        milestone=require_text(payload["milestone"], "milestone", maximum=32),
        status=require_identifier(payload["status"], "status"),
        required_capabilities=_sequence(payload["required_capabilities"], "required_capabilities"),
        allowed_speaker_states=_sequence(payload["allowed_speaker_states"], "allowed_speaker_states"),
        allowed_speaker_trust_states=_sequence(payload["allowed_speaker_trust_states"], "allowed_speaker_trust_states"),
        allowed_speaker_evidence_classes=_sequence(payload["allowed_speaker_evidence_classes"], "allowed_speaker_evidence_classes"),
        allowed_speaker_authority_ceilings=_sequence(payload["allowed_speaker_authority_ceilings"], "allowed_speaker_authority_ceilings"),
        allowed_guest_session_modes=_sequence(payload["allowed_guest_session_modes"], "allowed_guest_session_modes"),
        allowed_guest_session_statuses=_sequence(payload["allowed_guest_session_statuses"], "allowed_guest_session_statuses"),
        allowed_guest_grant_statuses=_sequence(payload["allowed_guest_grant_statuses"], "allowed_guest_grant_statuses"),
        allowed_guest_capabilities=_sequence(payload["allowed_guest_capabilities"], "allowed_guest_capabilities"),
        mandatory_guest_denied_domains=_sequence(payload["mandatory_guest_denied_domains"], "mandatory_guest_denied_domains"),
        allowed_authority_levels=_sequence(payload["allowed_authority_levels"], "allowed_authority_levels"),
        allowed_authority_decisions=_sequence(payload["allowed_authority_decisions"], "allowed_authority_decisions"),
        allowed_authority_evidence_classes=_sequence(payload["allowed_authority_evidence_classes"], "allowed_authority_evidence_classes"),
        allowed_consequence_classes=_sequence(payload["allowed_consequence_classes"], "allowed_consequence_classes"),
        allowed_authority_capabilities=_sequence(payload["allowed_authority_capabilities"], "allowed_authority_capabilities"),
        owner_only_capabilities=_sequence(payload["owner_only_capabilities"], "owner_only_capabilities"),
        invariants={str(key): bool(value) for key, value in invariants.items()},
        capability_ceiling=bool(payload["capability_ceiling"]),
        digest=canonical_sha256(payload),
        source_path=source,
    )
    policy.validate()

    previous = _read_json(
        root / "policies" / "cognitive_kernel_attention_workspace_policy.json"
    )
    if previous.get("version") != "0.3.0" or previous.get("milestone") != "P5.0d":
        raise CognitiveKernelContractError(
            "unexpected attention-workspace contract baseline"
        )
    parity = _read_json(root / "policies" / "capability_parity_ledger.json")
    product_lines = _read_json(root / "policies" / "product_lines.json")
    for capability in policy.required_capabilities:
        if not _contains_string(parity, capability):
            raise CognitiveKernelContractError(
                f"parity ledger is missing {capability}"
            )
    for contract in ("speaker_context", "guest_grant"):
        if not _contains_string(product_lines, contract):
            raise CognitiveKernelContractError(
                f"product-lines policy is missing {contract}"
            )
    return policy
