from __future__ import annotations

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    LifecycleJournalTransactionError,
    open_lifecycle_journal,
)
from lifecycle_helpers import (
    COMMITTED_AT,
    CREATED_AT,
    make_decision,
    make_scope,
    paths,
)


def test_quarantine_and_delete_eligibility_are_dedicated_decisions() -> None:
    quarantine = make_decision(
        decision_type="quarantine",
        proposed_tier="quarantine",
    )
    assert quarantine.outcome == "approved"
    eligible = make_decision(
        decision_key="delete-eligibility-1",
        decision_type="delete_eligible",
        proposed_tier="deleted",
        authority_level="host_verified",
        authority_decision_id="authority-decision-delete",
        outcome="recorded",
    )
    assert eligible.proposed_tier == "deleted"
    with pytest.raises(CognitiveKernelContractError, match="dedicated"):
        make_decision(proposed_tier="deleted")


def test_leaving_quarantine_requires_host_verified_authority() -> None:
    with pytest.raises(CognitiveKernelContractError, match="authority"):
        make_decision(
            current_tier="quarantine",
            proposed_tier="hot",
            authority_level="host_context",
        )
    decision = make_decision(
        current_tier="quarantine",
        proposed_tier="hot",
        authority_level="host_verified",
    )
    assert decision.current_tier == "quarantine"


def test_override_requires_owner_lineage_and_preserves_parent(tmp_path) -> None:
    repository, database = paths(tmp_path)
    prior = make_decision()
    override = make_decision(
        decision_key="override-1",
        decision_type="override",
        current_tier="warm",
        proposed_tier="cold",
        authority_level="owner_verified",
        authority_decision_id="authority-decision-owner",
        parent_decision_id=prior.decision_id,
    )
    with open_lifecycle_journal(
        database,
        scope=make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    ) as journal:
        with pytest.raises(
            LifecycleJournalTransactionError,
            match="parent does not exist",
        ):
            journal.append_record(override, committed_at=COMMITTED_AT)
        journal.append_records((prior, override), committed_at=COMMITTED_AT)
        assert journal.load_record(prior.record_id) == prior
        assert journal.load_record(override.record_id) == override


def test_override_cannot_authorize_payload_deletion() -> None:
    prior = make_decision()
    with pytest.raises(CognitiveKernelContractError, match="payload deletion"):
        make_decision(
            decision_key="override-delete",
            decision_type="override",
            proposed_tier="deleted",
            authority_level="owner_verified",
            parent_decision_id=prior.decision_id,
        )
