from __future__ import annotations

from cognitive_kernel import (
    TierTransitionIntent,
    open_tier_transition_store,
)
from tier_transition_helpers import (
    COMMITTED_AT,
    EXECUTED_AT,
    PAYLOAD,
    capture_raw,
    make_decision,
    make_paths,
    make_scope,
    open_journal,
    open_raw_store,
    open_tier_store,
)


def test_prepared_published_object_recovers_after_reopen(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    raw = open_raw_store(vault, repository)
    captured = capture_raw(raw)
    with open_journal(vault, repository) as journal:
        approved = make_decision(
            content_digest=captured.reference.content_digest
        )
        journal.append_record(approved, committed_at=COMMITTED_AT)
        tiers = open_tier_store(vault, repository)
        payload = raw.load_opaque_payload(
            captured.reference.reference_id
        )
        intent = TierTransitionIntent.create(
            lifecycle_decision_id=approved.decision_id,
            lifecycle_decision_sha256=approved.decision_sha256,
            scope=make_scope(),
            subject_reference=approved.subject_reference,
            content_digest=approved.content_digest,
            source_tier="raw_buffer",
            target_tier="hot",
            source_reference_id=captured.reference.reference_id,
            byte_length=len(payload),
            media_type=captured.reference.media_type,
            sensitivity_class=captured.reference.sensitivity_class,
            retention_class=captured.reference.retention_class,
            prepared_at=EXECUTED_AT,
        )
        tiers._insert_intent(intent)
        assert tiers._publish_object(
            payload,
            target_tier="hot",
            content_digest=approved.content_digest,
        ) is True
        tiers.close()
        reopened = open_tier_transition_store(
            vault / "tier-store",
            scope=make_scope(),
            repository_root=repository,
        )
        try:
            assert reopened.verify_integrity().pending_count == 1
            receipts = reopened.recover_pending(
                lifecycle_journal=journal,
                recovered_at="2026-08-03T05:46:00Z",
                raw_buffer_store=raw,
            )
            assert len(receipts) == 1
            assert receipts[0].recovered_from_prepared_intent is True
            assert reopened.verify_integrity().pending_count == 0
            assert raw.load_opaque_payload(
                captured.reference.reference_id
            ) == PAYLOAD
        finally:
            reopened.close()
    raw.close()


def test_pending_intent_without_final_object_replays_copy(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    raw = open_raw_store(vault, repository)
    captured = capture_raw(raw)
    with open_journal(vault, repository) as journal:
        approved = make_decision(
            content_digest=captured.reference.content_digest
        )
        journal.append_record(approved, committed_at=COMMITTED_AT)
        tiers = open_tier_store(vault, repository)
        intent = TierTransitionIntent.create(
            lifecycle_decision_id=approved.decision_id,
            lifecycle_decision_sha256=approved.decision_sha256,
            scope=make_scope(),
            subject_reference=approved.subject_reference,
            content_digest=approved.content_digest,
            source_tier="raw_buffer",
            target_tier="hot",
            source_reference_id=captured.reference.reference_id,
            byte_length=captured.reference.byte_length,
            media_type=captured.reference.media_type,
            sensitivity_class=captured.reference.sensitivity_class,
            retention_class=captured.reference.retention_class,
            prepared_at=EXECUTED_AT,
        )
        tiers._insert_intent(intent)
        tiers.close()
        reopened = open_tier_transition_store(
            vault / "tier-store",
            scope=make_scope(),
            repository_root=repository,
        )
        try:
            receipts = reopened.recover_pending(
                lifecycle_journal=journal,
                recovered_at="2026-08-03T05:46:00Z",
                raw_buffer_store=raw,
            )
            assert len(receipts) == 1
            assert receipts[0].physical_object_created is True
            assert reopened.load_opaque_payload(
                receipts[0].target_reference.reference_id
            ) == PAYLOAD
        finally:
            reopened.close()
    raw.close()
