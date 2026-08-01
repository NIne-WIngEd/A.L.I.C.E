"""Authority requests and decisions bound to speaker and guest evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
)
from .contracts import ProductHostScope
from .guest import GUEST_CAPABILITIES, GuestGrant
from .speaker import SPEAKER_TRUST_STATES, SpeakerContext

AUTHORITY_LEVELS = (
    "none",
    "guest_scoped",
    "host_context",
    "host_verified",
    "owner_verified",
)
AUTHORITY_DECISIONS = frozenset(
    {"approve", "deny", "require_stronger_authentication"}
)
AUTHORITY_EVIDENCE_CLASSES = frozenset(
    {
        "none",
        "speaker_context_only",
        "local_voice_match",
        "host_session_context",
        "stronger_authentication",
        "owner_explicit_approval",
        "explicit_guest_grant",
    }
)
CONSEQUENCE_CLASSES = frozenset(
    {
        "ordinary",
        "private_data",
        "external_commitment",
        "financial",
        "system_change",
        "destructive",
        "credential",
        "production_governance",
    }
)
OWNER_ONLY_CAPABILITIES = frozenset(
    {
        "production_promotion",
        "authority_policy_amendment",
        "private_companion_governance",
        "constitutional_change",
    }
)
HOST_VERIFIED_CAPABILITIES = frozenset(
    {
        "private_memory_read",
        "private_file_access",
        "messaging",
        "purchases",
        "system_changes",
        "external_commitments",
        "destructive_action",
        "credential_access",
    }
)
HOST_CONTEXT_CAPABILITIES = frozenset(
    {"host_workspace_control", "mission_routing_control"}
)
AUTHORITY_CAPABILITIES = frozenset(
    set(GUEST_CAPABILITIES)
    | OWNER_ONLY_CAPABILITIES
    | HOST_VERIFIED_CAPABILITIES
    | HOST_CONTEXT_CAPABILITIES
)
_CAPABILITY_CONSEQUENCE = {
    **{capability: "ordinary" for capability in GUEST_CAPABILITIES},
    "host_workspace_control": "ordinary",
    "mission_routing_control": "ordinary",
    "private_memory_read": "private_data",
    "private_file_access": "private_data",
    "messaging": "external_commitment",
    "external_commitments": "external_commitment",
    "purchases": "financial",
    "system_changes": "system_change",
    "destructive_action": "destructive",
    "credential_access": "credential",
    **{
        capability: "production_governance"
        for capability in OWNER_ONLY_CAPABILITIES
    },
}
_AUTHORITY_RANK = {value: index for index, value in enumerate(AUTHORITY_LEVELS)}


def _enum(value: object, field: str, allowed: Iterable[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


def required_authority_for(capability: object) -> str:
    normalized = _enum(capability, "capability", AUTHORITY_CAPABILITIES)
    if normalized in OWNER_ONLY_CAPABILITIES:
        return "owner_verified"
    if normalized in HOST_VERIFIED_CAPABILITIES:
        return "host_verified"
    if normalized in HOST_CONTEXT_CAPABILITIES:
        return "host_context"
    return "guest_scoped"


def consequence_for(capability: object) -> str:
    normalized = _enum(capability, "capability", AUTHORITY_CAPABILITIES)
    return _CAPABILITY_CONSEQUENCE[normalized]


def stronger_authentication_required(capability: object) -> bool:
    return consequence_for(capability) in {
        "private_data",
        "external_commitment",
        "financial",
        "system_change",
        "destructive",
        "credential",
        "production_governance",
    }


@dataclass(frozen=True)
class AuthorityRequest:
    """One exact capability request with a derived minimum authority."""

    schema_version: str
    request_id: str
    request_key: str
    scope: ProductHostScope
    requested_at: str
    actor_context_id: str
    actor_trust_state: str
    capability: str
    mission_id: str | None
    guest_grant_id: str | None
    consequence_class: str
    required_authority: str
    stronger_authentication_required: bool
    owner_only: bool
    reason_codes: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    request_sha256: str

    @classmethod
    def create(
        cls,
        *,
        request_key: object,
        scope: ProductHostScope,
        requested_at: object,
        actor_context_id: object,
        actor_trust_state: object,
        capability: object,
        mission_id: object | None = None,
        guest_grant_id: object | None = None,
        reason_codes: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "AuthorityRequest":
        scope.validate()
        normalized_capability = _enum(
            capability, "capability", AUTHORITY_CAPABILITIES
        )
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "request_key": require_identifier(request_key, "request_key"),
            "requested_at": normalize_timestamp(
                requested_at, "requested_at"
            ),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            request_id=f"authority-request-{canonical_sha256(identity)[:32]}",
            request_key=identity["request_key"],
            scope=scope,
            requested_at=identity["requested_at"],
            actor_context_id=require_identifier(
                actor_context_id, "actor_context_id"
            ),
            actor_trust_state=_enum(
                actor_trust_state,
                "actor_trust_state",
                SPEAKER_TRUST_STATES,
            ),
            capability=normalized_capability,
            mission_id=(
                require_identifier(mission_id, "mission_id")
                if mission_id is not None
                else None
            ),
            guest_grant_id=(
                require_identifier(guest_grant_id, "guest_grant_id")
                if guest_grant_id is not None
                else None
            ),
            consequence_class=consequence_for(normalized_capability),
            required_authority=required_authority_for(
                normalized_capability
            ),
            stronger_authentication_required=stronger_authentication_required(
                normalized_capability
            ),
            owner_only=normalized_capability in OWNER_ONLY_CAPABILITIES,
            reason_codes=normalize_identifier_sequence(
                reason_codes, "reason_codes"
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            request_sha256="0" * 64,
        )
        provisional._validate_material()
        request = cls(
            **{
                **provisional.__dict__,
                "request_sha256": canonical_sha256(
                    provisional.material_record()
                ),
            }
        )
        request.validate()
        return request

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "request_key": self.request_key,
            "requested_at": self.requested_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.request_key, "request_key")
        normalize_timestamp(self.requested_at, "requested_at")
        require_identifier(self.actor_context_id, "actor_context_id")
        _enum(
            self.actor_trust_state,
            "actor_trust_state",
            SPEAKER_TRUST_STATES,
        )
        capability = _enum(
            self.capability, "capability", AUTHORITY_CAPABILITIES
        )
        if self.mission_id is not None:
            require_identifier(self.mission_id, "mission_id")
        if self.guest_grant_id is not None:
            require_identifier(self.guest_grant_id, "guest_grant_id")
        if self.consequence_class != consequence_for(capability):
            raise CognitiveKernelContractError(
                "authority request consequence class changed"
            )
        if self.required_authority != required_authority_for(capability):
            raise CognitiveKernelContractError(
                "authority request minimum authority changed"
            )
        if self.stronger_authentication_required is not stronger_authentication_required(
            capability
        ):
            raise CognitiveKernelContractError(
                "stronger-authentication requirement changed"
            )
        if self.owner_only is not (capability in OWNER_ONLY_CAPABILITIES):
            raise CognitiveKernelContractError("owner-only marker changed")
        if self.actor_trust_state in {
            "guest_session",
            "delegated_guest_session",
        } and self.guest_grant_id is None:
            raise CognitiveKernelContractError(
                "guest authority request requires guest_grant_id"
            )
        normalize_identifier_sequence(self.reason_codes, "reason_codes")
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "request_id": self.request_id,
            "actor_context_id": self.actor_context_id,
            "actor_trust_state": self.actor_trust_state,
            "capability": self.capability,
            "mission_id": self.mission_id,
            "guest_grant_id": self.guest_grant_id,
            "consequence_class": self.consequence_class,
            "required_authority": self.required_authority,
            "stronger_authentication_required": self.stronger_authentication_required,
            "owner_only": self.owner_only,
            "reason_codes": list(self.reason_codes),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        return {**self.material_record(), "request_sha256": self.request_sha256}

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"authority-request-{canonical_sha256(self.identity_record())[:32]}"
        if self.request_id != expected_id:
            raise CognitiveKernelContractError("authority request identity is invalid")
        if self.request_sha256 != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError("authority request digest mismatch")


@dataclass(frozen=True)
class AuthorityDecision:
    """Explainable approval, denial, or stronger-authentication receipt."""

    schema_version: str
    decision_id: str
    decision_key: str
    scope: ProductHostScope
    request_id: str
    actor_context_id: str
    decided_at: str
    decision: str
    evidence_class: str
    granted_authority: str
    guest_grant_id: str | None
    expires_at: str | None
    reason_codes: tuple[str, ...]
    policy_bindings: tuple[str, ...]
    decision_sha256: str

    @classmethod
    def create(
        cls,
        *,
        decision_key: object,
        request: AuthorityRequest,
        speaker_context: SpeakerContext,
        decided_at: object,
        decision: object,
        evidence_class: object,
        granted_authority: object,
        guest_grant: GuestGrant | None = None,
        expires_at: object | None = None,
        reason_codes: tuple[object, ...] | list[object] = (),
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "AuthorityDecision":
        request.validate()
        speaker_context.validate()
        request.scope.validate()
        if request.scope.metadata_record() != speaker_context.scope.metadata_record():
            raise CognitiveKernelContractError(
                "authority request and speaker context must share scope"
            )
        if request.actor_context_id != speaker_context.context_id:
            raise CognitiveKernelContractError(
                "authority request actor context changed"
            )
        normalized_decision = _enum(
            decision, "decision", AUTHORITY_DECISIONS
        )
        normalized_evidence = _enum(
            evidence_class,
            "evidence_class",
            AUTHORITY_EVIDENCE_CLASSES,
        )
        normalized_authority = _enum(
            granted_authority, "granted_authority", AUTHORITY_LEVELS
        )
        decided = normalize_timestamp(decided_at, "decided_at")
        normalized_guest_grant_id: str | None = None
        if guest_grant is not None:
            guest_grant.validate()
            if guest_grant.scope.metadata_record() != request.scope.metadata_record():
                raise CognitiveKernelContractError(
                    "guest grant and authority request must share scope"
                )
            normalized_guest_grant_id = guest_grant.grant_id
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": request.scope.metadata_record(),
            "decision_key": require_identifier(decision_key, "decision_key"),
            "request_id": request.request_id,
            "decided_at": decided,
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            decision_id=f"authority-decision-{canonical_sha256(identity)[:32]}",
            decision_key=identity["decision_key"],
            scope=request.scope,
            request_id=request.request_id,
            actor_context_id=request.actor_context_id,
            decided_at=decided,
            decision=normalized_decision,
            evidence_class=normalized_evidence,
            granted_authority=normalized_authority,
            guest_grant_id=normalized_guest_grant_id,
            expires_at=(
                normalize_timestamp(expires_at, "expires_at")
                if expires_at is not None
                else None
            ),
            reason_codes=normalize_identifier_sequence(
                reason_codes, "reason_codes"
            ),
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            decision_sha256="0" * 64,
        )
        provisional._validate_material(request, speaker_context, guest_grant)
        receipt = cls(
            **{
                **provisional.__dict__,
                "decision_sha256": canonical_sha256(
                    provisional.material_record()
                ),
            }
        )
        receipt.validate(request, speaker_context, guest_grant)
        return receipt

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "decision_key": self.decision_key,
            "request_id": self.request_id,
            "decided_at": self.decided_at,
        }

    def _validate_material(
        self,
        request: AuthorityRequest,
        speaker_context: SpeakerContext,
        guest_grant: GuestGrant | None,
    ) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        request.validate()
        speaker_context.validate()
        require_identifier(self.decision_key, "decision_key")
        require_identifier(self.request_id, "request_id")
        require_identifier(self.actor_context_id, "actor_context_id")
        normalize_timestamp(self.decided_at, "decided_at")
        decision = _enum(self.decision, "decision", AUTHORITY_DECISIONS)
        evidence = _enum(
            self.evidence_class,
            "evidence_class",
            AUTHORITY_EVIDENCE_CLASSES,
        )
        granted = _enum(
            self.granted_authority, "granted_authority", AUTHORITY_LEVELS
        )
        if self.request_id != request.request_id:
            raise CognitiveKernelContractError("authority decision request changed")
        if self.actor_context_id != speaker_context.context_id:
            raise CognitiveKernelContractError("authority decision actor changed")
        if self.scope.metadata_record() != request.scope.metadata_record():
            raise CognitiveKernelContractError("authority decision scope changed")
        if self.expires_at is not None:
            normalize_timestamp(self.expires_at, "expires_at")
        normalize_identifier_sequence(self.reason_codes, "reason_codes")
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )

        if decision != "approve":
            if granted != "none" or self.guest_grant_id is not None:
                raise CognitiveKernelContractError(
                    "non-approval decisions may not grant authority"
                )
            return

        if _AUTHORITY_RANK[granted] < _AUTHORITY_RANK[request.required_authority]:
            raise CognitiveKernelContractError(
                "approved authority is below the request minimum"
            )
        if evidence in {"none", "speaker_context_only", "local_voice_match"}:
            raise CognitiveKernelContractError(
                "voice or speaker context alone may not authorize an action"
            )
        if request.stronger_authentication_required and evidence not in {
            "stronger_authentication",
            "owner_explicit_approval",
        }:
            raise CognitiveKernelContractError(
                "high-consequence approval requires stronger authentication"
            )
        if request.owner_only:
            if granted != "owner_verified" or evidence != "owner_explicit_approval":
                raise CognitiveKernelContractError(
                    "owner-only capability requires explicit owner approval"
                )
        if request.required_authority == "guest_scoped":
            if guest_grant is None:
                raise CognitiveKernelContractError(
                    "guest-scoped approval requires a guest grant"
                )
            if self.guest_grant_id != guest_grant.grant_id:
                raise CognitiveKernelContractError(
                    "authority decision guest grant changed"
                )
            if evidence != "explicit_guest_grant":
                raise CognitiveKernelContractError(
                    "guest approval requires explicit grant evidence"
                )
            if not guest_grant.authorizes(
                request.capability,
                at=self.decided_at,
                mission_id=request.mission_id,
            ):
                raise CognitiveKernelContractError(
                    "guest grant does not authorize the requested action"
                )
        elif self.guest_grant_id is not None:
            raise CognitiveKernelContractError(
                "non-guest approval may not bind a guest grant"
            )

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "decision_id": self.decision_id,
            "actor_context_id": self.actor_context_id,
            "decision": self.decision,
            "evidence_class": self.evidence_class,
            "granted_authority": self.granted_authority,
            "guest_grant_id": self.guest_grant_id,
            "expires_at": self.expires_at,
            "reason_codes": list(self.reason_codes),
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        return {**self.material_record(), "decision_sha256": self.decision_sha256}

    def validate(
        self,
        request: AuthorityRequest,
        speaker_context: SpeakerContext,
        guest_grant: GuestGrant | None = None,
    ) -> None:
        self._validate_material(request, speaker_context, guest_grant)
        expected_id = f"authority-decision-{canonical_sha256(self.identity_record())[:32]}"
        if self.decision_id != expected_id:
            raise CognitiveKernelContractError("authority decision identity is invalid")
        if self.decision_sha256 != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError("authority decision digest mismatch")
