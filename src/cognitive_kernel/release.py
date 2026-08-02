"""Exact-artifact release-attestation contracts for governed promotion."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
    require_text,
)
from .contracts import PRODUCT_IDS

_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

AUDIT_DECISIONS = frozenset(
    {
        "approved_for_owner_review",
        "approved_with_conditions",
        "return_for_revision",
        "blocked_security",
        "blocked_privacy",
        "blocked_authority",
        "blocked_evaluation",
        "blocked_architecture",
        "insufficient_evidence",
    }
)
OWNER_APPROVAL_DECISIONS = frozenset(
    {
        "approved",
        "approved_canary_only",
        "approved_closed_alpha",
        "approved_with_conditions",
        "rejected",
        "revision_required",
        "deferred",
    }
)
AUTHORIZED_AUDIT_DECISIONS = frozenset(
    {"approved_for_owner_review", "approved_with_conditions"}
)
AUTHORIZED_OWNER_DECISIONS = frozenset(
    {
        "approved",
        "approved_canary_only",
        "approved_closed_alpha",
        "approved_with_conditions",
    }
)
RELEASE_CHANNELS = (
    "internal",
    "canary",
    "closed_alpha",
    "alpha",
    "beta",
    "stable",
)


def _require_git_sha(value: object, field: str) -> str:
    normalized = require_text(value, field, maximum=40).lower()
    if _GIT_SHA_PATTERN.fullmatch(normalized) is None:
        raise CognitiveKernelContractError(
            f"{field} must be a lowercase 40-character Git commit"
        )
    return normalized


def _normalize_digest_mapping(
    value: Mapping[object, object],
    field: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise CognitiveKernelContractError(f"{field} must be a non-empty mapping")
    normalized: list[tuple[str, str]] = []
    for key, digest in value.items():
        normalized.append(
            (
                require_identifier(key, f"{field}.key"),
                require_sha256(digest, f"{field}.{key}"),
            )
        )
    normalized.sort()
    keys = [key for key, _ in normalized]
    if len(keys) != len(set(keys)):
        raise CognitiveKernelContractError(f"{field} contains duplicate keys")
    return tuple(normalized)


def _normalize_version_mapping(
    value: Mapping[object, object],
    field: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise CognitiveKernelContractError(f"{field} must be a mapping")
    normalized: list[tuple[str, str]] = []
    for key, version in value.items():
        normalized.append(
            (
                require_identifier(key, f"{field}.key"),
                require_text(version, f"{field}.{key}", maximum=256),
            )
        )
    normalized.sort()
    keys = [key for key, _ in normalized]
    if len(keys) != len(set(keys)):
        raise CognitiveKernelContractError(f"{field} contains duplicate keys")
    return tuple(normalized)


def artifact_manifest_digest(
    artifact_hashes: Mapping[object, object] | tuple[tuple[str, str], ...],
) -> str:
    if isinstance(artifact_hashes, tuple):
        material = artifact_hashes
    else:
        material = _normalize_digest_mapping(artifact_hashes, "artifact_hashes")
    return canonical_sha256(
        {"artifact_hashes": {key: digest for key, digest in material}}
    )


@dataclass(frozen=True)
class ReleaseAuditAttestation:
    """A.L.I.C.E. audit determination bound to one exact release candidate."""

    attestation_id: str
    decision: str
    source_commit: str
    artifact_manifest_digest: str
    evaluation_bundle_digest: str
    deployment_manifest_digest: str
    issued_at: str
    conditions_digest: str | None

    @classmethod
    def create(
        cls,
        *,
        attestation_id: object,
        decision: object,
        source_commit: object,
        artifact_manifest_digest: object,
        evaluation_bundle_digest: object,
        deployment_manifest_digest: object,
        issued_at: object,
        conditions_digest: object | None = None,
    ) -> "ReleaseAuditAttestation":
        value = cls(
            attestation_id=require_identifier(attestation_id, "attestation_id"),
            decision=require_identifier(decision, "decision"),
            source_commit=_require_git_sha(source_commit, "source_commit"),
            artifact_manifest_digest=require_sha256(
                artifact_manifest_digest, "artifact_manifest_digest"
            ),
            evaluation_bundle_digest=require_sha256(
                evaluation_bundle_digest, "evaluation_bundle_digest"
            ),
            deployment_manifest_digest=require_sha256(
                deployment_manifest_digest, "deployment_manifest_digest"
            ),
            issued_at=normalize_timestamp(issued_at, "issued_at"),
            conditions_digest=(
                require_sha256(conditions_digest, "conditions_digest")
                if conditions_digest is not None
                else None
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.decision not in AUDIT_DECISIONS:
            raise CognitiveKernelContractError("audit decision is not approved")
        if self.decision == "approved_with_conditions" and self.conditions_digest is None:
            raise CognitiveKernelContractError(
                "approved_with_conditions requires conditions_digest"
            )
        if self.decision != "approved_with_conditions" and self.conditions_digest is not None:
            raise CognitiveKernelContractError(
                "conditions_digest is only allowed for approved_with_conditions"
            )

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "attestation_id": self.attestation_id,
            "decision": self.decision,
            "source_commit": self.source_commit,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "evaluation_bundle_digest": self.evaluation_bundle_digest,
            "deployment_manifest_digest": self.deployment_manifest_digest,
            "issued_at": self.issued_at,
            "conditions_digest": self.conditions_digest,
        }


@dataclass(frozen=True)
class ReleaseOwnerApproval:
    """Owner decision bound to the same exact release candidate."""

    approval_id: str
    decision: str
    source_commit: str
    artifact_manifest_digest: str
    evaluation_bundle_digest: str
    deployment_manifest_digest: str
    issued_at: str
    conditions_digest: str | None

    @classmethod
    def create(
        cls,
        *,
        approval_id: object,
        decision: object,
        source_commit: object,
        artifact_manifest_digest: object,
        evaluation_bundle_digest: object,
        deployment_manifest_digest: object,
        issued_at: object,
        conditions_digest: object | None = None,
    ) -> "ReleaseOwnerApproval":
        value = cls(
            approval_id=require_identifier(approval_id, "approval_id"),
            decision=require_identifier(decision, "decision"),
            source_commit=_require_git_sha(source_commit, "source_commit"),
            artifact_manifest_digest=require_sha256(
                artifact_manifest_digest, "artifact_manifest_digest"
            ),
            evaluation_bundle_digest=require_sha256(
                evaluation_bundle_digest, "evaluation_bundle_digest"
            ),
            deployment_manifest_digest=require_sha256(
                deployment_manifest_digest, "deployment_manifest_digest"
            ),
            issued_at=normalize_timestamp(issued_at, "issued_at"),
            conditions_digest=(
                require_sha256(conditions_digest, "conditions_digest")
                if conditions_digest is not None
                else None
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.decision not in OWNER_APPROVAL_DECISIONS:
            raise CognitiveKernelContractError("owner decision is not approved")
        if self.decision == "approved_with_conditions" and self.conditions_digest is None:
            raise CognitiveKernelContractError(
                "approved_with_conditions requires conditions_digest"
            )
        if self.decision != "approved_with_conditions" and self.conditions_digest is not None:
            raise CognitiveKernelContractError(
                "conditions_digest is only allowed for approved_with_conditions"
            )

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "approval_id": self.approval_id,
            "decision": self.decision,
            "source_commit": self.source_commit,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "evaluation_bundle_digest": self.evaluation_bundle_digest,
            "deployment_manifest_digest": self.deployment_manifest_digest,
            "issued_at": self.issued_at,
            "conditions_digest": self.conditions_digest,
        }


@dataclass(frozen=True)
class ReleaseAttestation:
    """Deterministic release candidate plus dual exact-artifact authorization."""

    schema_version: str
    release_id: str
    product_id: str
    version: str
    source_commit: str
    kernel_version: str
    dependency_lock_digest: str
    artifact_hashes: tuple[tuple[str, str], ...]
    artifact_manifest_digest: str
    model_pack_versions: tuple[tuple[str, str], ...]
    schema_versions: tuple[tuple[str, str], ...]
    policy_versions: tuple[tuple[str, str], ...]
    migration_manifest: str
    evaluation_bundle_digest: str
    deployment_manifest: str
    rollback_manifest: str
    release_channel: str
    alice_audit_attestation: ReleaseAuditAttestation
    rayan_approval: ReleaseOwnerApproval
    release_digest: str

    @classmethod
    def create(
        cls,
        *,
        release_id: object,
        product_id: object,
        version: object,
        source_commit: object,
        kernel_version: object,
        dependency_lock_digest: object,
        artifact_hashes: Mapping[object, object],
        model_pack_versions: Mapping[object, object],
        schema_versions: Mapping[object, object],
        policy_versions: Mapping[object, object],
        migration_manifest: object,
        evaluation_bundle_digest: object,
        deployment_manifest: object,
        rollback_manifest: object,
        release_channel: object,
        alice_audit_attestation: ReleaseAuditAttestation,
        rayan_approval: ReleaseOwnerApproval,
        schema_version: object = "2.0.0",
    ) -> "ReleaseAttestation":
        normalized_artifacts = _normalize_digest_mapping(
            artifact_hashes, "artifact_hashes"
        )
        manifest_digest = artifact_manifest_digest(normalized_artifacts)
        draft = cls(
            schema_version=require_schema_version(schema_version),
            release_id=require_identifier(release_id, "release_id"),
            product_id=require_identifier(product_id, "product_id"),
            version=require_text(version, "version", maximum=128),
            source_commit=_require_git_sha(source_commit, "source_commit"),
            kernel_version=require_schema_version(kernel_version, "kernel_version"),
            dependency_lock_digest=require_sha256(
                dependency_lock_digest, "dependency_lock_digest"
            ),
            artifact_hashes=normalized_artifacts,
            artifact_manifest_digest=manifest_digest,
            model_pack_versions=_normalize_version_mapping(
                model_pack_versions, "model_pack_versions"
            ),
            schema_versions=_normalize_version_mapping(
                schema_versions, "schema_versions"
            ),
            policy_versions=_normalize_version_mapping(
                policy_versions, "policy_versions"
            ),
            migration_manifest=require_sha256(
                migration_manifest, "migration_manifest"
            ),
            evaluation_bundle_digest=require_sha256(
                evaluation_bundle_digest, "evaluation_bundle_digest"
            ),
            deployment_manifest=require_sha256(
                deployment_manifest, "deployment_manifest"
            ),
            rollback_manifest=require_sha256(
                rollback_manifest, "rollback_manifest"
            ),
            release_channel=require_identifier(release_channel, "release_channel"),
            alice_audit_attestation=alice_audit_attestation,
            rayan_approval=rayan_approval,
            release_digest="0" * 64,
        )
        draft._validate_material()
        digest = canonical_sha256(draft.material_record())
        value = cls(**{**draft.__dict__, "release_digest": digest})
        value.validate()
        return value

    def _validate_material(self) -> None:
        if self.schema_version != "2.0.0":
            raise CognitiveKernelContractError(
                "release-attestation schema version changed"
            )
        if self.product_id not in PRODUCT_IDS:
            raise CognitiveKernelContractError("product_id is not approved")
        if self.release_channel not in RELEASE_CHANNELS:
            raise CognitiveKernelContractError("release_channel is not approved")
        if self.kernel_version != "0.5.0":
            raise CognitiveKernelContractError(
                "release attestation requires cognitive_kernel 0.5.0"
            )
        if self.artifact_manifest_digest != artifact_manifest_digest(
            self.artifact_hashes
        ):
            raise CognitiveKernelContractError(
                "artifact_manifest_digest does not match artifact_hashes"
            )
        self.alice_audit_attestation.validate()
        self.rayan_approval.validate()
        bindings = (
            (
                self.alice_audit_attestation.source_commit,
                self.source_commit,
                "A.L.I.C.E. audit source_commit",
            ),
            (
                self.rayan_approval.source_commit,
                self.source_commit,
                "owner approval source_commit",
            ),
            (
                self.alice_audit_attestation.artifact_manifest_digest,
                self.artifact_manifest_digest,
                "A.L.I.C.E. audit artifact manifest",
            ),
            (
                self.rayan_approval.artifact_manifest_digest,
                self.artifact_manifest_digest,
                "owner approval artifact manifest",
            ),
            (
                self.alice_audit_attestation.evaluation_bundle_digest,
                self.evaluation_bundle_digest,
                "A.L.I.C.E. audit evaluation bundle",
            ),
            (
                self.rayan_approval.evaluation_bundle_digest,
                self.evaluation_bundle_digest,
                "owner approval evaluation bundle",
            ),
            (
                self.alice_audit_attestation.deployment_manifest_digest,
                self.deployment_manifest,
                "A.L.I.C.E. audit deployment manifest",
            ),
            (
                self.rayan_approval.deployment_manifest_digest,
                self.deployment_manifest,
                "owner approval deployment manifest",
            ),
        )
        for actual, expected, label in bindings:
            if actual != expected:
                raise CognitiveKernelContractError(
                    f"{label} does not match the outer candidate"
                )
        if (
            self.rayan_approval.decision == "approved_canary_only"
            and self.release_channel != "canary"
        ):
            raise CognitiveKernelContractError(
                "approved_canary_only requires the canary channel"
            )
        if (
            self.rayan_approval.decision == "approved_closed_alpha"
            and self.release_channel != "closed_alpha"
        ):
            raise CognitiveKernelContractError(
                "approved_closed_alpha requires the closed_alpha channel"
            )

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.release_digest, "release_digest")
        if self.release_digest != canonical_sha256(self.material_record()):
            raise CognitiveKernelContractError("release_digest does not match")

    @property
    def is_authorized(self) -> bool:
        self.validate()
        return (
            self.alice_audit_attestation.decision
            in AUTHORIZED_AUDIT_DECISIONS
            and self.rayan_approval.decision in AUTHORIZED_OWNER_DECISIONS
        )

    def assert_authorized(self) -> None:
        if not self.is_authorized:
            raise CognitiveKernelContractError(
                "release candidate does not have dual authorization"
            )

    def material_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "product_id": self.product_id,
            "version": self.version,
            "source_commit": self.source_commit,
            "kernel_version": self.kernel_version,
            "dependency_lock_digest": self.dependency_lock_digest,
            "artifact_hashes": dict(self.artifact_hashes),
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "model_pack_versions": dict(self.model_pack_versions),
            "schema_versions": dict(self.schema_versions),
            "policy_versions": dict(self.policy_versions),
            "migration_manifest": self.migration_manifest,
            "evaluation_bundle_digest": self.evaluation_bundle_digest,
            "deployment_manifest": self.deployment_manifest,
            "rollback_manifest": self.rollback_manifest,
            "release_channel": self.release_channel,
            "alice_audit_attestation": self.alice_audit_attestation.record(),
            "rayan_approval": self.rayan_approval.record(),
        }

    def record(self) -> dict[str, object]:
        self.validate()
        return {**self.material_record(), "release_digest": self.release_digest}

    def assert_isolated_from(self, other: "ReleaseAttestation") -> None:
        self.validate()
        other.validate()
        if self.product_id == other.product_id and self.release_id == other.release_id:
            raise CognitiveKernelContractError(
                "release attestations share the same product and release identity"
            )
