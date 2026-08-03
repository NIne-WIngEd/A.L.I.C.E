from __future__ import annotations

from tier_transition_helpers import (
    COMMITTED_AT,
    EXECUTED_AT,
    PAYLOAD,
    capture_raw,
    make_decision,
    make_paths,
    open_journal,
    open_raw_store,
    open_tier_store,
)


def test_raw_to_hot_preserves_source_and_is_idempotent(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            approved = make_decision(
                content_digest=captured.reference.content_digest
            )
            journal.append_record(approved, committed_at=COMMITTED_AT)
            with open_tier_store(vault, repository) as tiers:
                receipt = tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=approved.decision_id,
                    source_reference_id=captured.reference.reference_id,
                    executed_at=EXECUTED_AT,
                    raw_buffer_store=raw,
                )
                assert receipt.source_preserved is True
                assert receipt.target_tier == "hot"
                assert tiers.load_opaque_payload(
                    receipt.target_reference.reference_id
                ) == PAYLOAD
                assert raw.load_opaque_payload(
                    captured.reference.reference_id
                ) == PAYLOAD
                repeated = tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=approved.decision_id,
                    source_reference_id=captured.reference.reference_id,
                    executed_at=EXECUTED_AT,
                    raw_buffer_store=raw,
                )
                assert repeated == receipt
                report = tiers.verify_integrity()
                assert report.published_count == 1
                assert report.pending_count == 0


def test_managed_hot_to_warm_preserves_hot_source(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            first = make_decision(
                content_digest=captured.reference.content_digest
            )
            journal.append_record(first, committed_at=COMMITTED_AT)
            with open_tier_store(vault, repository) as tiers:
                hot = tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=first.decision_id,
                    source_reference_id=captured.reference.reference_id,
                    executed_at=EXECUTED_AT,
                    raw_buffer_store=raw,
                )
                second = make_decision(
                    content_digest=captured.reference.content_digest,
                    decision_key="decision-2",
                    current_tier="hot",
                    proposed_tier="warm",
                )
                journal.append_record(
                    second,
                    committed_at="2026-08-03T05:44:00Z",
                )
                warm = tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=second.decision_id,
                    source_reference_id=(
                        hot.target_reference.reference_id
                    ),
                    executed_at="2026-08-03T05:45:00Z",
                )
                assert tiers.load_opaque_payload(
                    hot.target_reference.reference_id
                ) == PAYLOAD
                assert tiers.load_opaque_payload(
                    warm.target_reference.reference_id
                ) == PAYLOAD
                assert warm.source_preserved is True
                assert tiers.verify_integrity().published_count == 2


def test_quarantine_decision_is_executable_without_deletion(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            quarantine = make_decision(
                content_digest=captured.reference.content_digest,
                decision_type="quarantine",
                proposed_tier="quarantine",
            )
            journal.append_record(quarantine, committed_at=COMMITTED_AT)
            with open_tier_store(vault, repository) as tiers:
                receipt = tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=quarantine.decision_id,
                    source_reference_id=captured.reference.reference_id,
                    executed_at=EXECUTED_AT,
                    raw_buffer_store=raw,
                )
                assert receipt.target_tier == "quarantine"
                assert receipt.source_preserved is True
                assert raw.load_opaque_payload(
                    captured.reference.reference_id
                ) == PAYLOAD


def test_inspection_is_sanitized(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            approved = make_decision(
                content_digest=captured.reference.content_digest
            )
            journal.append_record(approved, committed_at=COMMITTED_AT)
            with open_tier_store(vault, repository) as tiers:
                tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=approved.decision_id,
                    source_reference_id=captured.reference.reference_id,
                    executed_at=EXECUTED_AT,
                    raw_buffer_store=raw,
                )
                record = tiers.inspect()[0].record()
                assert record["state"] == "published"
                assert "payload" not in record
                assert "absolute_path" not in record
