from __future__ import annotations

from cognitive_kernel import (
    ReleaseAttestation,
    ReleaseAuditAttestation,
    ReleaseOwnerApproval,
    artifact_manifest_digest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SOURCE_COMMIT = "1" * 40


def make_audit(
    *,
    source_commit: str = SOURCE_COMMIT,
    artifacts: dict[str, str] | None = None,
    evaluation_bundle_digest: str = SHA_C,
    deployment_manifest_digest: str = SHA_D,
    decision: str = "approved_for_owner_review",
    conditions_digest: str | None = None,
) -> ReleaseAuditAttestation:
    artifact_values = artifacts or {"app.exe": SHA_A, "manifest.json": SHA_B}
    return ReleaseAuditAttestation.create(
        attestation_id="audit-1",
        decision=decision,
        source_commit=source_commit,
        artifact_manifest_digest=artifact_manifest_digest(artifact_values),
        evaluation_bundle_digest=evaluation_bundle_digest,
        deployment_manifest_digest=deployment_manifest_digest,
        issued_at="2026-08-01T20:00:00Z",
        conditions_digest=conditions_digest,
    )


def make_owner(
    *,
    source_commit: str = SOURCE_COMMIT,
    artifacts: dict[str, str] | None = None,
    evaluation_bundle_digest: str = SHA_C,
    deployment_manifest_digest: str = SHA_D,
    decision: str = "approved",
    conditions_digest: str | None = None,
) -> ReleaseOwnerApproval:
    artifact_values = artifacts or {"app.exe": SHA_A, "manifest.json": SHA_B}
    return ReleaseOwnerApproval.create(
        approval_id="owner-approval-1",
        decision=decision,
        source_commit=source_commit,
        artifact_manifest_digest=artifact_manifest_digest(artifact_values),
        evaluation_bundle_digest=evaluation_bundle_digest,
        deployment_manifest_digest=deployment_manifest_digest,
        issued_at="2026-08-01T20:01:00Z",
        conditions_digest=conditions_digest,
    )


def make_release(
    *,
    release_id: str = "friday-0.1.0-canary",
    product_id: str = "friday",
    source_commit: str = SOURCE_COMMIT,
    artifact_hashes: dict[str, str] | None = None,
    evaluation_bundle_digest: str = SHA_C,
    deployment_manifest: str = SHA_D,
    release_channel: str = "canary",
    audit: ReleaseAuditAttestation | None = None,
    owner: ReleaseOwnerApproval | None = None,
) -> ReleaseAttestation:
    artifacts = artifact_hashes or {"app.exe": SHA_A, "manifest.json": SHA_B}
    return ReleaseAttestation.create(
        release_id=release_id,
        product_id=product_id,
        version="0.1.0",
        source_commit=source_commit,
        kernel_version="0.5.0",
        dependency_lock_digest=SHA_E,
        artifact_hashes=artifacts,
        model_pack_versions={"local-model": "1.2.0"},
        schema_versions={"release": "2.0.0"},
        policy_versions={"friday-production": "1.1.0"},
        migration_manifest=SHA_A,
        evaluation_bundle_digest=evaluation_bundle_digest,
        deployment_manifest=deployment_manifest,
        rollback_manifest=SHA_B,
        release_channel=release_channel,
        alice_audit_attestation=audit or make_audit(artifacts=artifacts),
        rayan_approval=owner or make_owner(artifacts=artifacts),
    )
