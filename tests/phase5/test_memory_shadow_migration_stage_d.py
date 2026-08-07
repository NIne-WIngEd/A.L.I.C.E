from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_kernel import (
    BackfillDestinationResult,
    CognitiveKernelContractError,
    HistoricalBackfillCheckpoint,
    HistoricalBackfillManifest,
    HistoricalBackfillRecord,
    InMemoryIdempotentBackfillSink,
    ProductHostScope,
    build_synthetic_stage_d_manifest,
    run_historical_backfill_batch,
)
from cognitive_kernel.canonical import canonical_sha256
from cognitive_kernel.shadow_migration_stage_d_evaluation import (
    _assert_report_outside_repo,
    build_synthetic_stage_d_report,
)

TS = "2026-08-07T06:15:00Z"


def scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="stage-d-test-host",
        schema_version="1.0.0",
        encryption_domain="stage-d-test-domain",
    )


def manifest(*, workload_class="synthetic", authorization=None, destination="candidate.test"):
    return HistoricalBackfillManifest.create(
        scope=scope(),
        manifest_id="manifest.1",
        source_registration_id="source.registration.1",
        destination_candidate_id=destination,
        source_snapshot_sha256=canonical_sha256({"snapshot": 1}),
        mapping_version="1.0.0",
        workload_class=workload_class,
        authorization_reference_id=authorization,
        preferred_batch_size=128,
        created_at=TS,
    )


def record(
    *,
    manifest_id="manifest.1",
    record_id="record.1",
    checkpoint="000001",
    mapping_version="1.0.0",
    provenance_state="complete",
    disposition="accepted",
    reason=None,
    evidence=("evidence.1",),
    deletion=(),
):
    return HistoricalBackfillRecord.create(
        manifest_id=manifest_id,
        source_record_id=record_id,
        source_checkpoint=checkpoint,
        source_record_sha256=canonical_sha256(
            {"record": record_id, "checkpoint": checkpoint}
        ),
        mapped_record_sha256=canonical_sha256(
            {"mapped": record_id, "mapping": mapping_version}
        ),
        mapping_version=mapping_version,
        provenance_state=provenance_state,
        evidence_lineage_ids=evidence,
        deletion_lineage_ids=deletion,
        disposition=disposition,
        disposition_reason=reason,
    )


def test_manifest_is_backend_neutral_and_digest_bound():
    first = manifest(destination="candidate.polyglot.1")
    second = manifest(destination="candidate.polyglot.1")
    assert first == second
    assert first.authority_effect == "shadow_candidate_only"
    assert first.serving_effect == "none"
    assert first.preferred_batch_size == 128
    assert len(first.manifest_sha256) == 64


def test_owner_authorized_manifest_requires_reference():
    with pytest.raises(CognitiveKernelContractError):
        manifest(workload_class="owner_authorized")
    item = manifest(
        workload_class="owner_authorized",
        authorization="owner.authorization.stage_d.1",
    )
    item.validate()


def test_synthetic_manifest_rejects_false_authorization_claim():
    with pytest.raises(CognitiveKernelContractError):
        manifest(authorization="owner.authorization.stage_d.1")


def test_record_idempotency_is_deterministic_and_preserves_deletion_lineage():
    first = record(deletion=("deletion.1",))
    second = record(deletion=("deletion.1",))
    assert first == second
    assert first.deletion_lineage_ids == ("deletion.1",)
    assert len(first.idempotency_key) == 64


def test_accepted_record_cannot_invent_missing_provenance():
    with pytest.raises(CognitiveKernelContractError):
        record(provenance_state="missing")


def test_nonaccepted_record_requires_explicit_reason():
    with pytest.raises(CognitiveKernelContractError):
        record(disposition="quarantined", provenance_state="missing")


def test_batch_counts_and_reconciliation_are_deterministic():
    m = manifest()
    sink = InMemoryIdempotentBackfillSink()
    records = (
        record(record_id="record.1", checkpoint="000001"),
        record(
            record_id="record.2",
            checkpoint="000002",
            deletion=("deletion.2",),
        ),
        record(
            record_id="record.3",
            checkpoint="000003",
            provenance_state="missing",
            disposition="quarantined",
            reason="missing_provenance",
            evidence=(),
        ),
        record(
            record_id="record.4",
            checkpoint="000004",
            provenance_state="partial",
            disposition="ambiguous",
            reason="ambiguous_relation",
        ),
        record(
            record_id="record.5",
            checkpoint="000005",
            provenance_state="partial",
            disposition="rejected",
            reason="unsupported_semantics",
        ),
    )
    receipt = run_historical_backfill_batch(
        manifest=m,
        records=records,
        write_accepted_record=sink.write,
        completed_at=TS,
    )
    assert receipt.record_count == 5
    assert receipt.accepted_count == 2
    assert receipt.rejected_count == 1
    assert receipt.quarantined_count == 1
    assert receipt.ambiguous_count == 1
    assert receipt.applied_count == 2
    assert receipt.duplicate_count == 0
    assert sink.record_count == 2
    assert len(receipt.reconciliation_sha256) == 64


def test_replay_is_idempotent():
    m = manifest()
    sink = InMemoryIdempotentBackfillSink()
    records = (
        record(record_id="record.1", checkpoint="000001"),
        record(record_id="record.2", checkpoint="000002"),
    )
    first = run_historical_backfill_batch(
        manifest=m,
        records=records,
        write_accepted_record=sink.write,
        completed_at=TS,
    )
    replay = run_historical_backfill_batch(
        manifest=m,
        records=records,
        write_accepted_record=sink.write,
        completed_at=TS,
    )
    assert first.applied_count == 2
    assert replay.applied_count == 0
    assert replay.duplicate_count == 2
    assert first.batch_id == replay.batch_id
    assert sink.record_count == 2


def test_batch_rejects_mapping_version_mismatch():
    m = manifest()
    sink = InMemoryIdempotentBackfillSink()
    with pytest.raises(CognitiveKernelContractError):
        run_historical_backfill_batch(
            manifest=m,
            records=(record(mapping_version="2.0.0"),),
            write_accepted_record=sink.write,
            completed_at=TS,
        )


def test_destination_result_must_match_source_idempotency_key():
    m = manifest()
    item = record()

    def wrong_writer(_):
        return BackfillDestinationResult.create(
            idempotency_key=canonical_sha256({"wrong": 1}),
            outcome="applied",
            destination_record_sha256=item.mapped_record_sha256,
            detail={"wrong": True},
        )

    with pytest.raises(CognitiveKernelContractError):
        run_historical_backfill_batch(
            manifest=m,
            records=(item,),
            write_accepted_record=wrong_writer,
            completed_at=TS,
        )


def test_checkpoint_is_digest_bound():
    m = manifest()
    sink = InMemoryIdempotentBackfillSink()
    receipt = run_historical_backfill_batch(
        manifest=m,
        records=(record(),),
        write_accepted_record=sink.write,
        completed_at=TS,
    )
    checkpoint = HistoricalBackfillCheckpoint.from_receipts(
        manifest=m,
        receipts=(receipt,),
    )
    assert checkpoint.last_source_checkpoint == "000001"
    assert checkpoint.accepted_total == 1
    assert checkpoint.applied_total == 1
    assert len(checkpoint.checkpoint_sha256) == 64


def test_synthetic_report_is_deterministic_and_truthful():
    first = build_synthetic_stage_d_report()
    second = build_synthetic_stage_d_report()
    assert first == second
    state = first["material_state"]
    assert state["stage_d_deterministic_backfill_prototype"] == "operational"
    assert (
        state["historical_private_payload_execution"]
        == "not_performed_by_synthetic_evaluation"
    )
    assert state["canonical_authority"] == "phase2_released_profile_remains_current"
    assert state["production_serving_effect"] == "none"
    assert first["first_batch"]["accepted_count"] == 2
    assert first["replay_batch"]["duplicate_count"] == 2


def test_report_cannot_be_written_inside_repository(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError):
        _assert_report_outside_repo(repo / "report.json", repo)
    _assert_report_outside_repo(tmp_path / "vault" / "report.json", repo)


def test_component_policy_remains_governance_compatible():
    policy_path = Path("policies/memory_shadow_migration_stage_d_policy.json")
    if not policy_path.exists():
        pytest.skip("policy is added by the repository patcher")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert "capability_state_semantics" not in policy
    assert policy["capability_ceiling"] is False
    assert policy["research_status"] == "allowed"
    assert policy["production_status"] == "not_activated_by_this_profile"
