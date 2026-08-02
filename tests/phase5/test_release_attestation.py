from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    ReleaseAttestation,
    ReleaseAuditAttestation,
    ReleaseOwnerApproval,
    artifact_manifest_digest,
)

from release_attestation_helpers import (
    SHA_A,
    SHA_B,
    SHA_C,
    SHA_D,
    SHA_E,
    SOURCE_COMMIT,
    make_audit,
    make_owner,
    make_release,
)


def test_authorized_release_is_deterministic() -> None:
    first = make_release()
    second = make_release()
    assert first.release_digest == second.release_digest
    assert first.is_authorized is True
    first.assert_authorized()
    assert first.record()["artifact_manifest_digest"] == artifact_manifest_digest(
        {"app.exe": SHA_A, "manifest.json": SHA_B}
    )


def test_artifact_order_does_not_change_digest() -> None:
    first = make_release(
        artifact_hashes={"app.exe": SHA_A, "manifest.json": SHA_B}
    )
    second = make_release(
        artifact_hashes={"manifest.json": SHA_B, "app.exe": SHA_A}
    )
    assert first.release_digest == second.release_digest


def test_audit_source_commit_must_match_outer_candidate() -> None:
    with pytest.raises(CognitiveKernelContractError, match="audit source_commit"):
        make_release(audit=make_audit(source_commit="2" * 40))


def test_owner_artifact_manifest_must_match_outer_candidate() -> None:
    owner = make_owner(artifacts={"different.exe": SHA_A})
    with pytest.raises(CognitiveKernelContractError, match="owner approval artifact"):
        make_release(owner=owner)


def test_evaluation_and_deployment_bindings_are_exact() -> None:
    with pytest.raises(CognitiveKernelContractError, match="evaluation bundle"):
        make_release(audit=make_audit(evaluation_bundle_digest=SHA_D))
    with pytest.raises(CognitiveKernelContractError, match="deployment manifest"):
        make_release(owner=make_owner(deployment_manifest_digest=SHA_C))


def test_non_approving_receipts_are_representable_but_not_authorized() -> None:
    release = make_release(
        audit=make_audit(decision="return_for_revision"),
        owner=make_owner(decision="deferred"),
    )
    assert release.is_authorized is False
    with pytest.raises(CognitiveKernelContractError, match="dual authorization"):
        release.assert_authorized()


def test_canary_only_approval_requires_canary_channel() -> None:
    with pytest.raises(CognitiveKernelContractError, match="canary channel"):
        make_release(
            release_channel="stable",
            owner=make_owner(decision="approved_canary_only"),
        )


def test_closed_alpha_approval_requires_closed_alpha_channel() -> None:
    with pytest.raises(CognitiveKernelContractError, match="closed_alpha"):
        make_release(
            release_channel="canary",
            owner=make_owner(decision="approved_closed_alpha"),
        )
    release = make_release(
        release_channel="closed_alpha",
        owner=make_owner(decision="approved_closed_alpha"),
    )
    assert release.is_authorized is True


def test_conditional_decisions_require_conditions_digest() -> None:
    with pytest.raises(CognitiveKernelContractError, match="conditions_digest"):
        make_audit(decision="approved_with_conditions")
    with pytest.raises(CognitiveKernelContractError, match="conditions_digest"):
        make_owner(decision="approved_with_conditions")
    audit = make_audit(
        decision="approved_with_conditions", conditions_digest=SHA_E
    )
    owner = make_owner(
        decision="approved_with_conditions", conditions_digest=SHA_E
    )
    assert make_release(audit=audit, owner=owner).is_authorized is True


def test_release_digest_detects_tampering() -> None:
    release = make_release()
    tampered = replace(release, rollback_manifest=SHA_D)
    with pytest.raises(CognitiveKernelContractError, match="release_digest"):
        tampered.validate()


def test_invalid_product_and_kernel_version_are_rejected() -> None:
    with pytest.raises(CognitiveKernelContractError, match="product_id"):
        make_release(product_id="other")
    with pytest.raises(CognitiveKernelContractError, match="0.5.0"):
        ReleaseAttestation.create(
            release_id="bad-kernel",
            product_id="friday",
            version="0.1.0",
            source_commit=SOURCE_COMMIT,
            kernel_version="0.4.0",
            dependency_lock_digest=SHA_E,
            artifact_hashes={"app.exe": SHA_A},
            model_pack_versions={},
            schema_versions={},
            policy_versions={},
            migration_manifest=SHA_A,
            evaluation_bundle_digest=SHA_C,
            deployment_manifest=SHA_D,
            rollback_manifest=SHA_B,
            release_channel="canary",
            alice_audit_attestation=make_audit(artifacts={"app.exe": SHA_A}),
            rayan_approval=make_owner(artifacts={"app.exe": SHA_A}),
        )


def test_receipt_decision_vocabulary_is_closed() -> None:
    with pytest.raises(CognitiveKernelContractError, match="audit decision"):
        ReleaseAuditAttestation.create(
            attestation_id="audit-invalid",
            decision="invented",
            source_commit=SOURCE_COMMIT,
            artifact_manifest_digest=SHA_A,
            evaluation_bundle_digest=SHA_B,
            deployment_manifest_digest=SHA_C,
            issued_at="2026-08-01T20:00:00Z",
        )
    with pytest.raises(CognitiveKernelContractError, match="owner decision"):
        ReleaseOwnerApproval.create(
            approval_id="owner-invalid",
            decision="invented",
            source_commit=SOURCE_COMMIT,
            artifact_manifest_digest=SHA_A,
            evaluation_bundle_digest=SHA_B,
            deployment_manifest_digest=SHA_C,
            issued_at="2026-08-01T20:00:00Z",
        )
