from __future__ import annotations

from cognitive_kernel import open_tier_transition_store
from tier_transition_helpers import (
    COMMITTED_AT,
    EXECUTED_AT,
    capture_raw,
    make_decision,
    make_paths,
    make_scope,
    open_journal,
    open_raw_store,
    open_tier_store,
)


def test_reopen_preserves_published_transition(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    raw = open_raw_store(vault, repository)
    captured = capture_raw(raw)
    with open_journal(vault, repository) as journal:
        approved = make_decision(
            content_digest=captured.reference.content_digest
        )
        journal.append_record(approved, committed_at=COMMITTED_AT)
        tiers = open_tier_store(vault, repository)
        receipt = tiers.execute(
            lifecycle_journal=journal,
            decision_id=approved.decision_id,
            source_reference_id=captured.reference.reference_id,
            executed_at=EXECUTED_AT,
            raw_buffer_store=raw,
        )
        tiers.close()
        reopened = open_tier_transition_store(
            vault / "tier-store",
            scope=make_scope(),
            repository_root=repository,
        )
        try:
            assert reopened.get_reference(
                receipt.target_reference.reference_id
            ) == receipt.target_reference
            assert reopened.verify_integrity().published_count == 1
        finally:
            reopened.close()
    raw.close()
