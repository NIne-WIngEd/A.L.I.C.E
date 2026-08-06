"""Memory M2.6 reversible deletion-propagation profile tests."""

from pathlib import Path
import sqlite3

import pytest

from cognitive_kernel.canonical import (
    CognitiveKernelContractError,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.deletion_prototype import (
    DeletionPropagationConflictError,
    DeletionPropagationIsolationError,
    DeletionPropagationProfile,
    DeletionPropagationPrototypeStore,
    UnsafeDeletionPropagationPathError,
    open_deletion_propagation_prototype,
)

STAMP = "2026-08-06T23:00:00Z"


def scope(
    host: str = "owner-workstation",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id=host,
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def profile(
    host: str = "owner-workstation",
) -> DeletionPropagationProfile:
    return DeletionPropagationProfile.create(
        scope=scope(host),
        authority_namespace_id="owner-memory",
        profile_id="memory-m2-6-deletion-rehearsal",
        required_plane_kinds=(
            "claim_authority",
            "episode_projection",
        ),
    )


def open_store(
    tmp_path: Path,
    *,
    selected_profile: DeletionPropagationProfile | None = None,
) -> DeletionPropagationPrototypeStore:
    return open_deletion_propagation_prototype(
        path=tmp_path / "deletion.sqlite3",
        profile=selected_profile or profile(),
        repository_root=tmp_path / "public-repo",
    )


def begin(
    store: DeletionPropagationPrototypeStore,
    *,
    key: str = "begin-1",
):
    return store.begin_request(
        request_id="request-1",
        target_record_ids=("claim-2", "claim-1"),
        deletion_mode="mixed",
        reason_code="owner_delete",
        authority_decision_id="authority-decision-1",
        requested_by="owner",
        requested_at=STAMP,
        full_request_content={
            "owner_instruction": "delete these records",
            "private_payload_reference": "vault-object-1",
        },
        idempotency_namespace="test",
        idempotency_key=key,
    )


def plane(
    store: DeletionPropagationPrototypeStore,
    *,
    plane_kind: str,
    component_id: str,
    expected_generation: int,
    key: str,
):
    return store.record_plane_result(
        request_id="request-1",
        plane_kind=plane_kind,
        component_id=component_id,
        deletion_mode="tombstone",
        state="completed",
        completed_at=STAMP,
        target_count=2,
        deleted_count=2,
        blocked_count=0,
        evidence_record_ids=(f"evidence-{plane_kind}",),
        error_code=None,
        full_result_content={
            "plane": plane_kind,
            "deleted": ["claim-1", "claim-2"],
        },
        expected_generation=expected_generation,
        idempotency_namespace="test",
        idempotency_key=key,
    )


def prepare_complete(
    store: DeletionPropagationPrototypeStore,
) -> None:
    begin(store)
    plane(
        store,
        plane_kind="claim_authority",
        component_id="claim-store",
        expected_generation=1,
        key="plane-claim",
    )
    plane(
        store,
        plane_kind="episode_projection",
        component_id="projection-store",
        expected_generation=2,
        key="plane-projection",
    )


def finalize(
    store: DeletionPropagationPrototypeStore,
    *,
    key: str = "finalize-1",
):
    return store.finalize(
        request_id="request-1",
        propagation_state="completed",
        effective_at=STAMP,
        rollback_state="rehearsed",
        retirement_state="not_applicable",
        full_receipt_content={
            "summary": "cross-plane deletion rehearsal complete",
            "private_notes": {"reviewed_by": "owner"},
        },
        expected_generation=3,
        completed_at=STAMP,
        idempotency_namespace="test",
        idempotency_key=key,
    )


def test_profile_rejects_production_influence() -> None:
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationProfile.create(
            scope=scope(),
            authority_namespace_id="owner-memory",
            profile_id="memory-m2-6",
            required_plane_kinds=("claim_authority",),
            production_influence=True,
        )


def test_profile_rejects_canonical_claim_authority() -> None:
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationProfile.create(
            scope=scope(),
            authority_namespace_id="owner-memory",
            profile_id="memory-m2-6",
            required_plane_kinds=("claim_authority",),
            canonical_claim_authority=True,
        )


def test_profile_rejects_destructive_live_deletion() -> None:
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationProfile.create(
            scope=scope(),
            authority_namespace_id="owner-memory",
            profile_id="memory-m2-6",
            required_plane_kinds=("claim_authority",),
            destructive_live_deletion=True,
        )


def test_database_must_remain_outside_repository(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(UnsafeDeletionPropagationPathError):
        open_deletion_propagation_prototype(
            path=repository / "deletion.sqlite3",
            profile=profile(),
            repository_root=repository,
        )


def test_begin_request_persists_full_private_content(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        receipt = begin(store)
        content = store.request_content("request-1")
        assert receipt.operation_kind == "begin_request"
        assert content["request"]["target_record_ids"] == [
            "claim-1",
            "claim-2",
        ]
        assert (
            content["full_content"]["private_payload_reference"]
            == "vault-object-1"
        )


def test_begin_request_is_idempotent(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        first = begin(store)
        second = begin(store)
        assert first.operation_sha256 == second.operation_sha256


def test_begin_request_detects_changed_idempotent_content(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        with pytest.raises(DeletionPropagationConflictError):
            store.begin_request(
                request_id="request-1",
                target_record_ids=("claim-1",),
                deletion_mode="mixed",
                reason_code="owner_delete",
                authority_decision_id="authority-decision-1",
                requested_by="owner",
                requested_at=STAMP,
                full_request_content={"changed": True},
                idempotency_namespace="test",
                idempotency_key="begin-1",
            )


def test_plane_results_are_persisted_in_order(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        prepare_complete(store)
        values = store.plane_receipts("request-1")
        assert [value.plane_kind for value in values] == [
            "claim_authority",
            "episode_projection",
        ]


def test_plane_result_is_idempotent_after_generation_advances(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        first = plane(
            store,
            plane_kind="claim_authority",
            component_id="claim-store",
            expected_generation=1,
            key="plane-claim",
        )
        second = plane(
            store,
            plane_kind="claim_authority",
            component_id="claim-store",
            expected_generation=1,
            key="plane-claim",
        )
        assert first.operation_sha256 == second.operation_sha256


def test_plane_result_checks_expected_generation(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        with pytest.raises(DeletionPropagationConflictError):
            plane(
                store,
                plane_kind="claim_authority",
                component_id="claim-store",
                expected_generation=2,
                key="plane-claim",
            )


def test_unselected_plane_is_rejected(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        with pytest.raises(CognitiveKernelContractError):
            plane(
                store,
                plane_kind="backup",
                component_id="backup-store",
                expected_generation=1,
                key="plane-backup",
            )


def test_restore_filter_excludes_deleted_record(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        result = store.evaluate_restore_filter(
            request_id="request-1",
            target_record_id="claim-1",
            source_snapshot_id="snapshot-1",
            action="exclude",
            reason_code="deleted_record",
            evaluated_at=STAMP,
            source_content={"claim": "private"},
            replacement_record_id=None,
            full_decision_content={
                "restore_job": "restore-1",
                "owner_instruction": "do not restore",
            },
            idempotency_namespace="test",
            idempotency_key="restore-1",
        )
        decisions = store.restore_filter_decisions("request-1")
        assert result.operation_kind == "evaluate_restore_filter"
        assert decisions[0].action == "exclude"


def test_restore_filter_is_idempotent(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        kwargs = dict(
            request_id="request-1",
            target_record_id="claim-1",
            source_snapshot_id="snapshot-1",
            action="exclude",
            reason_code="deleted_record",
            evaluated_at=STAMP,
            source_content={"claim": "private"},
            replacement_record_id=None,
            full_decision_content={"owner_instruction": "exclude"},
            idempotency_namespace="test",
            idempotency_key="restore-1",
        )
        first = store.evaluate_restore_filter(**kwargs)
        second = store.evaluate_restore_filter(**kwargs)
        assert first.operation_sha256 == second.operation_sha256


def test_rollback_rehearsal_persists_full_content(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        store.record_rehearsal(
            request_id="request-1",
            rehearsal_kind="rollback",
            outcome="rehearsed",
            evaluated_at=STAMP,
            affected_artifact_ids=("claim-1",),
            measurements={"restore_seconds": 0.4},
            full_rehearsal_content={
                "snapshot_reference": "vault-snapshot-1"
            },
            idempotency_namespace="test",
            idempotency_key="rollback-1",
        )
        values = store.rehearsals("request-1")
        assert values[0]["rehearsal"]["rehearsal_kind"] == "rollback"
        assert (
            values[0]["full_content"]["snapshot_reference"]
            == "vault-snapshot-1"
        )


def test_model_retirement_rehearsal_is_supported(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        store.record_rehearsal(
            request_id="request-1",
            rehearsal_kind="model_retirement",
            outcome="replacement_required",
            evaluated_at=STAMP,
            affected_artifact_ids=("model-1", "adapter-1"),
            measurements={
                "residual_influence": 0.22,
                "threshold": 0.05,
            },
            full_rehearsal_content={
                "evaluation_report": "vault-report-1"
            },
            idempotency_namespace="test",
            idempotency_key="retirement-1",
        )
        assert (
            store.rehearsals("request-1")[0]["rehearsal"][
                "outcome"
            ]
            == "replacement_required"
        )


def test_finalize_requires_every_profile_plane(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        begin(store)
        plane(
            store,
            plane_kind="claim_authority",
            component_id="claim-store",
            expected_generation=1,
            key="plane-claim",
        )
        with pytest.raises(DeletionPropagationConflictError):
            store.finalize(
                request_id="request-1",
                propagation_state="completed",
                effective_at=STAMP,
                rollback_state="rehearsed",
                retirement_state="not_applicable",
                full_receipt_content={"summary": "incomplete"},
                expected_generation=2,
                completed_at=STAMP,
                idempotency_namespace="test",
                idempotency_key="finalize-1",
            )


def test_finalize_materializes_current_receipt(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        prepare_complete(store)
        operation = finalize(store)
        current = store.current_receipt("request-1")
        assert current is not None
        assert operation.result_record_id == current.receipt_id
        assert current.propagation_state == "completed"
        assert current.generation == 1


def test_finalize_is_idempotent_after_generation_advances(
    tmp_path: Path,
) -> None:
    with open_store(tmp_path) as store:
        prepare_complete(store)
        first = finalize(store)
        second = finalize(store)
        assert first.operation_sha256 == second.operation_sha256


def test_receipt_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "deletion.sqlite3"
    selected = profile()
    store = open_deletion_propagation_prototype(
        path=database,
        profile=selected,
    )
    prepare_complete(store)
    finalize(store)
    store.close()
    reopened = open_deletion_propagation_prototype(
        path=database,
        profile=selected,
    )
    try:
        current = reopened.current_receipt("request-1")
        assert current is not None
        assert current.propagation_state == "completed"
    finally:
        reopened.close()


def test_database_isolated_by_profile(tmp_path: Path) -> None:
    database = tmp_path / "deletion.sqlite3"
    first = open_deletion_propagation_prototype(
        path=database,
        profile=profile(),
    )
    first.close()
    with pytest.raises(DeletionPropagationIsolationError):
        open_deletion_propagation_prototype(
            path=database,
            profile=profile("different-host"),
        )


def test_integrity_report_is_healthy(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        prepare_complete(store)
        store.evaluate_restore_filter(
            request_id="request-1",
            target_record_id="claim-1",
            source_snapshot_id="snapshot-1",
            action="exclude",
            reason_code="deleted_record",
            evaluated_at=STAMP,
            source_content={"claim": "private"},
            replacement_record_id=None,
            full_decision_content={"owner_instruction": "exclude"},
            idempotency_namespace="test",
            idempotency_key="restore-1",
        )
        store.record_rehearsal(
            request_id="request-1",
            rehearsal_kind="rollback",
            outcome="rehearsed",
            evaluated_at=STAMP,
            affected_artifact_ids=("claim-1",),
            measurements={"seconds": 1},
            full_rehearsal_content={"snapshot": "vault-1"},
            idempotency_namespace="test",
            idempotency_key="rollback-1",
        )
        finalize(store)
        report = store.verify_integrity()
        assert report.healthy
        assert report.checked_requests == 1
        assert report.checked_plane_receipts == 2
        assert report.checked_restore_filters == 1
        assert report.checked_rehearsals == 1


def test_integrity_detects_plane_content_tamper(
    tmp_path: Path,
) -> None:
    database = tmp_path / "deletion.sqlite3"
    store = open_deletion_propagation_prototype(
        path=database,
        profile=profile(),
    )
    prepare_complete(store)
    store.close()
    connection = sqlite3.connect(database)
    connection.execute(
        """
        UPDATE plane_receipts
        SET full_content_json = '{"tampered":true}'
        WHERE 1 = 1
        """
    )
    connection.commit()
    connection.close()
    reopened = open_deletion_propagation_prototype(
        path=database,
        profile=profile(),
    )
    try:
        assert not reopened.verify_integrity().healthy
    finally:
        reopened.close()


def test_remove_database_is_reversible_cleanup(
    tmp_path: Path,
) -> None:
    database = tmp_path / "deletion.sqlite3"
    store = open_deletion_propagation_prototype(
        path=database,
        profile=profile(),
    )
    begin(store)
    assert database.exists()
    store.remove_database()
    assert not database.exists()
