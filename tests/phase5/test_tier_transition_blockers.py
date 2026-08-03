from __future__ import annotations

import pytest

from cognitive_kernel import (
    TierTransitionAuthorizationError,
    TierTransitionBlockedError,
)
from tier_transition_helpers import (
    COMMITTED_AT,
    EXECUTED_AT,
    capture_raw,
    make_decision,
    make_open_blocker,
    make_paths,
    make_resolution,
    open_journal,
    open_raw_store,
    open_tier_store,
)


def test_open_blocker_prevents_execution(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            approved = make_decision(
                content_digest=captured.reference.content_digest
            )
            blocker = make_open_blocker(
                content_digest=captured.reference.content_digest
            )
            journal.append_records(
                (approved, blocker),
                committed_at=COMMITTED_AT,
            )
            with open_tier_store(vault, repository) as tiers:
                with pytest.raises(
                    TierTransitionBlockedError,
                    match="active_project",
                ):
                    tiers.execute(
                        lifecycle_journal=journal,
                        decision_id=approved.decision_id,
                        source_reference_id=(
                            captured.reference.reference_id
                        ),
                        executed_at=EXECUTED_AT,
                        raw_buffer_store=raw,
                    )


def test_resolved_blocker_allows_execution(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            approved = make_decision(
                content_digest=captured.reference.content_digest
            )
            blocker = make_open_blocker(
                content_digest=captured.reference.content_digest
            )
            resolution = make_resolution(blocker)
            journal.append_records(
                (approved, blocker, resolution),
                committed_at=COMMITTED_AT,
            )
            with open_tier_store(vault, repository) as tiers:
                receipt = tiers.execute(
                    lifecycle_journal=journal,
                    decision_id=approved.decision_id,
                    source_reference_id=captured.reference.reference_id,
                    executed_at=EXECUTED_AT,
                    raw_buffer_store=raw,
                )
                assert receipt.target_tier == "hot"


def test_denied_decision_is_not_executable(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            denied = make_decision(
                content_digest=captured.reference.content_digest,
                outcome="denied",
            )
            journal.append_record(denied, committed_at=COMMITTED_AT)
            with open_tier_store(vault, repository) as tiers:
                with pytest.raises(
                    TierTransitionAuthorizationError,
                    match="not approved",
                ):
                    tiers.execute(
                        lifecycle_journal=journal,
                        decision_id=denied.decision_id,
                        source_reference_id=(
                            captured.reference.reference_id
                        ),
                        executed_at=EXECUTED_AT,
                        raw_buffer_store=raw,
                    )


def test_superseded_decision_is_rejected(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            first = make_decision(
                content_digest=captured.reference.content_digest
            )
            override = make_decision(
                content_digest=captured.reference.content_digest,
                decision_key="decision-override",
                current_tier="hot",
                proposed_tier="cold",
                decision_type="override",
                authority_level="owner_verified",
                parent_decision_id=first.decision_id,
            )
            journal.append_records(
                (first, override),
                committed_at=COMMITTED_AT,
            )
            with open_tier_store(vault, repository) as tiers:
                with pytest.raises(
                    TierTransitionAuthorizationError,
                    match="superseded",
                ):
                    tiers.execute(
                        lifecycle_journal=journal,
                        decision_id=first.decision_id,
                        source_reference_id=(
                            captured.reference.reference_id
                        ),
                        executed_at=EXECUTED_AT,
                        raw_buffer_store=raw,
                    )


def test_delete_eligibility_never_executes_payload_deletion(tmp_path) -> None:
    repository, vault = make_paths(tmp_path)
    with open_raw_store(vault, repository) as raw:
        captured = capture_raw(raw)
        with open_journal(vault, repository) as journal:
            deletion = make_decision(
                content_digest=captured.reference.content_digest,
                decision_key="decision-delete-eligible",
                proposed_tier="deleted",
                decision_type="delete_eligible",
                outcome="recorded",
                authority_level="host_verified",
            )
            journal.append_record(deletion, committed_at=COMMITTED_AT)
            with open_tier_store(vault, repository) as tiers:
                with pytest.raises(
                    TierTransitionAuthorizationError,
                    match="not executable",
                ):
                    tiers.execute(
                        lifecycle_journal=journal,
                        decision_id=deletion.decision_id,
                        source_reference_id=(
                            captured.reference.reference_id
                        ),
                        executed_at=EXECUTED_AT,
                        raw_buffer_store=raw,
                    )
                assert raw.load_opaque_payload(
                    captured.reference.reference_id
                )
