from __future__ import annotations

import pytest

from cognitive_kernel import TierTransitionIsolationError
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


def test_raw_source_scope_mismatch_is_rejected(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        other_scope = make_scope(host_instance_id="other-host")
        with open_raw_store(
            vault / "other",
            repository,
            scope=other_scope,
        ) as other_raw:
            with open_journal(vault, repository) as journal:
                approved = make_decision(
                    content_digest=captured.reference.content_digest
                )
                journal.append_record(
                    approved,
                    committed_at=COMMITTED_AT,
                )
                with open_tier_store(vault, repository) as tiers:
                    with pytest.raises(TierTransitionIsolationError):
                        tiers.execute(
                            lifecycle_journal=journal,
                            decision_id=approved.decision_id,
                            source_reference_id=(
                                captured.reference.reference_id
                            ),
                            executed_at=EXECUTED_AT,
                            raw_buffer_store=other_raw,
                        )


def test_journal_scope_mismatch_is_rejected(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        other_scope = make_scope(host_instance_id="other-host")
        with open_journal(
            vault / "other-journal",
            repository,
            scope=other_scope,
        ) as other_journal:
            with open_tier_store(vault, repository) as tiers:
                with pytest.raises(TierTransitionIsolationError):
                    tiers.execute(
                        lifecycle_journal=other_journal,
                        decision_id="lifecycle-decision-missing",
                        source_reference_id=(
                            captured.reference.reference_id
                        ),
                        executed_at=EXECUTED_AT,
                        raw_buffer_store=raw,
                    )
