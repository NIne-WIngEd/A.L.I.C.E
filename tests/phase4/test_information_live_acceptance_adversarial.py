from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from alice_information.live_acceptance import InformationLiveAcceptanceError
from _information_live_acceptance_helpers import BOUNDARIES, D, record


@pytest.mark.parametrize(
    "field,value",
    [
        ("package_version", "0.17.0"),
        ("repository_clean", False),
        ("repository_snapshot_after_sha256", "e" * 64),
        ("deterministic_test_skipped", 1),
        ("provider_policy_binding", "other@1.0.0:" + D),
        ("acceptance_domains", ("provider_availability",)),
        ("decision_reasons", ("unapproved_reason",)),
        ("repository_regression_passed", 2123),
        ("repository_regression_subtests_passed", 13),
    ],
)
def test_record_rejects_substituted_release_fields(field, value):
    original = record()
    with pytest.raises(InformationLiveAcceptanceError):
        replace(original, **{field: value}).validate()


def test_record_rejects_weakened_boundaries():
    weakened = dict(BOUNDARIES)
    weakened["retry_allowed"] = True
    with pytest.raises(InformationLiveAcceptanceError):
        replace(record(), boundaries=MappingProxyType(weakened)).validate()


def test_record_reconstructs_and_validates_embedded_live_receipt():
    original = record()
    mutations = (
        ("citation_validation_outcome", "rejected"),
        ("fetch_attempt_count", 0),
        ("fetch_attempt_sequence_sha256", "e" * 64),
        ("fetch_failure_sha256s", (D,)),
    )
    for field, value in mutations:
        live = dict(original.live_research_receipt)
        live[field] = value
        with pytest.raises(InformationLiveAcceptanceError):
            replace(
                original,
                live_research_receipt=MappingProxyType(live),
            ).validate()


def test_record_rejects_source_outcome_substitution():
    original = record()
    outcome = dict(original.source_outcomes[0])
    outcome["disposition"] = "eligible_not_selected"
    with pytest.raises(InformationLiveAcceptanceError):
        replace(original, source_outcomes=(MappingProxyType(outcome),)).validate()
