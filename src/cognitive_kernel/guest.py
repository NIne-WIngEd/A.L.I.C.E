"""Purpose-bound guest session and grant contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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

GUEST_SESSION_MODES = frozenset({"guest", "delegated_guest"})
GUEST_SESSION_STATUSES = frozenset({"active", "revoked", "expired", "closed"})
GUEST_GRANT_STATUSES = frozenset({"active", "revoked", "expired"})
GUEST_GRANTOR_AUTHORITIES = frozenset({"host_verified", "owner_verified"})
GUEST_CAPABILITIES = frozenset(
    {
        "general_questions",
        "media_control",
        "timers",
        "sanitized_workspace_view",
        "local_status",
    }
)
GUEST_DENIED_DOMAINS = frozenset(
    {
        "private_memory",
        "private_files",
        "messaging",
        "purchases",
        "system_changes",
        "credential_access",
        "external_commitments",
        "destructive_actions",
    }
)


def _enum(value: object, field: str, allowed: Iterable[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not approved")
    return normalized


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class GuestGrant:
    """Visible, scoped, expiring, revocable, and non-delegable authority."""

    schema_version: str
    grant_id: str
    grant_key: str
    scope: ProductHostScope
    guest_reference_id: str
    session_reference_id: str
    purpose_code: str
    capabilities: tuple[str, ...]
    mission_ids: tuple[str, ...]
    denied_domains: tuple[str, ...]
    grantor_authority: str
    issued_at: str
    expires_at: str
    status: str
    revoked_at: str | None
    non_delegable: bool
    self_expansion_allowed: bool
    private_payload_included: bool
    policy_bindings: tuple[str, ...]
    grant_sha256: str

    @classmethod
    def create(
        cls,
        *,
        grant_key: object,
        scope: ProductHostScope,
        guest_reference_id: object,
        session_reference_id: object,
        purpose_code: object,
        capabilities: tuple[object, ...] | list[object],
        mission_ids: tuple[object, ...] | list[object],
        grantor_authority: object,
        issued_at: object,
        expires_at: object,
        status: object = "active",
        revoked_at: object | None = None,
        denied_domains: tuple[object, ...] | list[object] = tuple(
            sorted(GUEST_DENIED_DOMAINS)
        ),
        non_delegable: object = True,
        self_expansion_allowed: object = False,
        private_payload_included: object = False,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "GuestGrant":
        scope.validate()
        for field, value in (
            ("non_delegable", non_delegable),
            ("self_expansion_allowed", self_expansion_allowed),
            ("private_payload_included", private_payload_included),
        ):
            if not isinstance(value, bool):
                raise CognitiveKernelContractError(f"{field} must be boolean")
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "grant_key": require_identifier(grant_key, "grant_key"),
            "guest_reference_id": require_identifier(
                guest_reference_id, "guest_reference_id"
            ),
            "session_reference_id": require_identifier(
                session_reference_id, "session_reference_id"
            ),
            "issued_at": normalize_timestamp(issued_at, "issued_at"),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            grant_id=f"guest-grant-{canonical_sha256(identity)[:32]}",
            grant_key=identity["grant_key"],
            scope=scope,
            guest_reference_id=identity["guest_reference_id"],
            session_reference_id=identity["session_reference_id"],
            purpose_code=require_identifier(purpose_code, "purpose_code"),
            capabilities=normalize_identifier_sequence(
                capabilities, "capabilities"
            ),
            mission_ids=normalize_identifier_sequence(
                mission_ids, "mission_ids"
            ),
            denied_domains=normalize_identifier_sequence(
                denied_domains, "denied_domains"
            ),
            grantor_authority=_enum(
                grantor_authority,
                "grantor_authority",
                GUEST_GRANTOR_AUTHORITIES,
            ),
            issued_at=identity["issued_at"],
            expires_at=normalize_timestamp(expires_at, "expires_at"),
            status=_enum(status, "status", GUEST_GRANT_STATUSES),
            revoked_at=(
                normalize_timestamp(revoked_at, "revoked_at")
                if revoked_at is not None
                else None
            ),
            non_delegable=non_delegable,
            self_expansion_allowed=self_expansion_allowed,
            private_payload_included=private_payload_included,
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            grant_sha256="0" * 64,
        )
        provisional._validate_material()
        grant = cls(
            **{
                **provisional.__dict__,
                "grant_sha256": canonical_sha256(
                    provisional.material_record()
                ),
            }
        )
        grant.validate()
        return grant

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "grant_key": self.grant_key,
            "guest_reference_id": self.guest_reference_id,
            "session_reference_id": self.session_reference_id,
            "issued_at": self.issued_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.grant_key, "grant_key")
        require_identifier(self.guest_reference_id, "guest_reference_id")
        require_identifier(self.session_reference_id, "session_reference_id")
        require_identifier(self.purpose_code, "purpose_code")
        capabilities = normalize_identifier_sequence(
            self.capabilities, "capabilities"
        )
        if not capabilities or not set(capabilities).issubset(GUEST_CAPABILITIES):
            raise CognitiveKernelContractError(
                "guest capabilities must be a non-empty approved subset"
            )
        normalize_identifier_sequence(self.mission_ids, "mission_ids")
        denied = normalize_identifier_sequence(
            self.denied_domains, "denied_domains"
        )
        if set(denied) != GUEST_DENIED_DOMAINS:
            raise CognitiveKernelContractError(
                "guest grant must preserve all denied authority domains"
            )
        _enum(
            self.grantor_authority,
            "grantor_authority",
            GUEST_GRANTOR_AUTHORITIES,
        )
        issued = normalize_timestamp(self.issued_at, "issued_at")
        expires = normalize_timestamp(self.expires_at, "expires_at")
        if _instant(expires) <= _instant(issued):
            raise CognitiveKernelContractError(
                "guest grant expiration must follow issuance"
            )
        status = _enum(self.status, "status", GUEST_GRANT_STATUSES)
        if status == "revoked":
            if self.revoked_at is None:
                raise CognitiveKernelContractError(
                    "revoked grant requires revoked_at"
                )
            revoked = normalize_timestamp(self.revoked_at, "revoked_at")
            if not (_instant(issued) <= _instant(revoked) <= _instant(expires)):
                raise CognitiveKernelContractError(
                    "revoked_at must fall within grant lifetime"
                )
        elif self.revoked_at is not None:
            raise CognitiveKernelContractError(
                "only revoked grants may include revoked_at"
            )
        if self.non_delegable is not True:
            raise CognitiveKernelContractError("guest grants must be non-delegable")
        if self.self_expansion_allowed is not False:
            raise CognitiveKernelContractError(
                "guest grants may not allow self-expansion"
            )
        if self.private_payload_included is not False:
            raise CognitiveKernelContractError(
                "guest grants must not contain private payloads"
            )
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "grant_id": self.grant_id,
            "purpose_code": self.purpose_code,
            "capabilities": list(self.capabilities),
            "mission_ids": list(self.mission_ids),
            "denied_domains": list(self.denied_domains),
            "grantor_authority": self.grantor_authority,
            "expires_at": self.expires_at,
            "status": self.status,
            "revoked_at": self.revoked_at,
            "non_delegable": self.non_delegable,
            "self_expansion_allowed": self.self_expansion_allowed,
            "private_payload_included": self.private_payload_included,
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        return {**self.material_record(), "grant_sha256": self.grant_sha256}

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"guest-grant-{canonical_sha256(self.identity_record())[:32]}"
        if self.grant_id != expected_id:
            raise CognitiveKernelContractError("guest grant identity is invalid")
        if self.grant_sha256 != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError("guest grant digest mismatch")

    def is_active_at(self, timestamp: object) -> bool:
        self.validate()
        checked = normalize_timestamp(timestamp, "timestamp")
        return (
            self.status == "active"
            and _instant(self.issued_at) <= _instant(checked) < _instant(self.expires_at)
        )

    def authorizes(
        self,
        capability: object,
        *,
        at: object,
        mission_id: object | None = None,
    ) -> bool:
        normalized_capability = require_identifier(capability, "capability")
        if not self.is_active_at(at) or normalized_capability not in self.capabilities:
            return False
        if self.mission_ids:
            if mission_id is None:
                return False
            return require_identifier(mission_id, "mission_id") in self.mission_ids
        return True


@dataclass(frozen=True)
class GuestSession:
    """Ephemeral guest-mode receipt bound to one product-host scope."""

    schema_version: str
    session_id: str
    session_key: str
    scope: ProductHostScope
    guest_reference_id: str
    mode: str
    status: str
    started_at: str
    expires_at: str
    ended_at: str | None
    grant_ids: tuple[str, ...]
    private_views_hidden: bool
    local_action_logging_required: bool
    private_state_included: bool
    persistent_authority: bool
    policy_bindings: tuple[str, ...]
    session_sha256: str

    @classmethod
    def create(
        cls,
        *,
        session_key: object,
        scope: ProductHostScope,
        guest_reference_id: object,
        mode: object,
        status: object,
        started_at: object,
        expires_at: object,
        grant_ids: tuple[object, ...] | list[object] = (),
        ended_at: object | None = None,
        private_views_hidden: object = True,
        local_action_logging_required: object = True,
        private_state_included: object = False,
        persistent_authority: object = False,
        policy_bindings: tuple[object, ...] | list[object] = (),
        schema_version: object = "1.0.0",
    ) -> "GuestSession":
        scope.validate()
        for field, value in (
            ("private_views_hidden", private_views_hidden),
            ("local_action_logging_required", local_action_logging_required),
            ("private_state_included", private_state_included),
            ("persistent_authority", persistent_authority),
        ):
            if not isinstance(value, bool):
                raise CognitiveKernelContractError(f"{field} must be boolean")
        identity = {
            "schema_version": require_schema_version(schema_version),
            "scope": scope.metadata_record(),
            "session_key": require_identifier(session_key, "session_key"),
            "guest_reference_id": require_identifier(
                guest_reference_id, "guest_reference_id"
            ),
            "started_at": normalize_timestamp(started_at, "started_at"),
        }
        provisional = cls(
            schema_version=identity["schema_version"],
            session_id=f"guest-session-{canonical_sha256(identity)[:32]}",
            session_key=identity["session_key"],
            scope=scope,
            guest_reference_id=identity["guest_reference_id"],
            mode=_enum(mode, "mode", GUEST_SESSION_MODES),
            status=_enum(status, "status", GUEST_SESSION_STATUSES),
            started_at=identity["started_at"],
            expires_at=normalize_timestamp(expires_at, "expires_at"),
            ended_at=(
                normalize_timestamp(ended_at, "ended_at")
                if ended_at is not None
                else None
            ),
            grant_ids=normalize_identifier_sequence(grant_ids, "grant_ids"),
            private_views_hidden=private_views_hidden,
            local_action_logging_required=local_action_logging_required,
            private_state_included=private_state_included,
            persistent_authority=persistent_authority,
            policy_bindings=normalize_identifier_sequence(
                policy_bindings, "policy_bindings"
            ),
            session_sha256="0" * 64,
        )
        provisional._validate_material()
        session = cls(
            **{
                **provisional.__dict__,
                "session_sha256": canonical_sha256(
                    provisional.material_record()
                ),
            }
        )
        session.validate()
        return session

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.metadata_record(),
            "session_key": self.session_key,
            "guest_reference_id": self.guest_reference_id,
            "started_at": self.started_at,
        }

    def _validate_material(self) -> None:
        require_schema_version(self.schema_version)
        self.scope.validate()
        require_identifier(self.session_key, "session_key")
        require_identifier(self.guest_reference_id, "guest_reference_id")
        mode = _enum(self.mode, "mode", GUEST_SESSION_MODES)
        status = _enum(self.status, "status", GUEST_SESSION_STATUSES)
        started = normalize_timestamp(self.started_at, "started_at")
        expires = normalize_timestamp(self.expires_at, "expires_at")
        if _instant(expires) <= _instant(started):
            raise CognitiveKernelContractError(
                "guest session expiration must follow start"
            )
        grants = normalize_identifier_sequence(self.grant_ids, "grant_ids")
        if mode == "delegated_guest" and not grants:
            raise CognitiveKernelContractError(
                "delegated guest session requires at least one grant"
            )
        if mode == "guest" and grants:
            raise CognitiveKernelContractError(
                "ordinary guest session may not carry delegated grants"
            )
        if status in {"revoked", "closed"}:
            if self.ended_at is None:
                raise CognitiveKernelContractError(
                    "ended guest session requires ended_at"
                )
            ended = normalize_timestamp(self.ended_at, "ended_at")
            if not (_instant(started) <= _instant(ended) <= _instant(expires)):
                raise CognitiveKernelContractError(
                    "ended_at must fall within session lifetime"
                )
        elif self.ended_at is not None:
            raise CognitiveKernelContractError(
                "active or expired session may not include ended_at"
            )
        if self.private_views_hidden is not True:
            raise CognitiveKernelContractError(
                "guest sessions must hide private views"
            )
        if self.local_action_logging_required is not True:
            raise CognitiveKernelContractError(
                "guest actions must be logged locally"
            )
        if self.private_state_included or self.persistent_authority:
            raise CognitiveKernelContractError(
                "guest sessions may not contain private state or persistent authority"
            )
        normalize_identifier_sequence(
            self.policy_bindings, "policy_bindings"
        )

    def material_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "session_id": self.session_id,
            "mode": self.mode,
            "status": self.status,
            "expires_at": self.expires_at,
            "ended_at": self.ended_at,
            "grant_ids": list(self.grant_ids),
            "private_views_hidden": self.private_views_hidden,
            "local_action_logging_required": self.local_action_logging_required,
            "private_state_included": self.private_state_included,
            "persistent_authority": self.persistent_authority,
            "policy_bindings": list(self.policy_bindings),
        }

    def metadata_record(self) -> dict[str, object]:
        return {**self.material_record(), "session_sha256": self.session_sha256}

    def validate(self) -> None:
        self._validate_material()
        expected_id = f"guest-session-{canonical_sha256(self.identity_record())[:32]}"
        if self.session_id != expected_id:
            raise CognitiveKernelContractError("guest session identity is invalid")
        if self.session_sha256 != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError("guest session digest mismatch")
