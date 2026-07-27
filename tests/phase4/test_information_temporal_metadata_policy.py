from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alice_information.freshness_policy import (
    InformationFreshnessPolicyError,
    load_information_freshness_policy,
)
from alice_information.policy import InformationPolicyError, load_information_policy
from alice_information.temporal_metadata_policy import (
    ALLOWED_TEMPORAL_METADATA_KINDS,
    ALLOWED_TEMPORAL_METADATA_ORIGINS,
    InformationTemporalMetadataPolicyError,
    load_information_temporal_metadata_policy,
    parse_information_temporal_metadata_policy,
)


def _payload() -> dict:
    return json.loads(
        open(
            "policies/information_temporal_metadata_policy.json",
            encoding="utf-8",
        ).read()
    )


def test_temporal_metadata_policy_loads_with_exact_identity() -> None:
    policy = load_information_temporal_metadata_policy(
        information_policy=load_information_policy(),
        freshness_policy=load_information_freshness_policy(),
    )
    assert policy.milestone == "P4.4b"
    assert policy.allowed_candidate_kinds == ALLOWED_TEMPORAL_METADATA_KINDS
    assert policy.allowed_candidate_origins == ALLOWED_TEMPORAL_METADATA_ORIGINS


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("policy_name", "changed"),
        ("version", "1.0.1"),
        ("phase", "5"),
        ("milestone", "P4.5"),
        ("status", "changed"),
        ("permission_id", "tool.execute"),
    ),
)
def test_policy_identity_changes_fail_closed(field: str, value: str) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(InformationTemporalMetadataPolicyError):
        parse_information_temporal_metadata_policy(payload)


def test_policy_rejects_unknown_root_keys() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(InformationTemporalMetadataPolicyError):
        parse_information_temporal_metadata_policy(payload)


def test_policy_rejects_changed_kind_or_origin_vocabularies() -> None:
    for field in ("allowed_candidate_kinds", "allowed_candidate_origins"):
        payload = _payload()
        payload[field] = list(reversed(payload[field]))
        with pytest.raises(InformationTemporalMetadataPolicyError):
            parse_information_temporal_metadata_policy(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_candidates", 31),
        ("max_raw_value_characters", 255),
        ("min_cross_source_observations", 1),
        ("max_cross_source_observations", 15),
    ),
)
def test_policy_limits_are_version_pinned(field: str, value: int) -> None:
    payload = _payload()
    payload["limits"][field] = value
    with pytest.raises(InformationTemporalMetadataPolicyError):
        parse_information_temporal_metadata_policy(payload)


@pytest.mark.parametrize(
    "field",
    (
        "deterministic_html_metadata_extraction_required",
        "http_last_modified_allowed",
        "invalid_candidates_fail_closed",
        "conflicting_candidates_fail_closed",
        "source_digest_binding_required",
        "explicit_subject_digest_required",
        "cross_source_conflicts_preserved",
    ),
)
def test_required_controls_cannot_be_disabled(field: str) -> None:
    payload = _payload()
    payload[field] = False
    with pytest.raises(InformationTemporalMetadataPolicyError):
        parse_information_temporal_metadata_policy(payload)


@pytest.mark.parametrize(
    "field",
    (
        "visible_text_date_inference_allowed",
        "model_date_extraction_allowed",
        "conflict_winner_selection_allowed",
        "raw_temporal_metadata_logging_allowed",
    ),
)
def test_prohibited_capabilities_cannot_be_enabled(field: str) -> None:
    payload = _payload()
    payload[field] = True
    with pytest.raises(InformationTemporalMetadataPolicyError):
        parse_information_temporal_metadata_policy(payload)


def test_policy_revalidates_projected_instances() -> None:
    policy = load_information_temporal_metadata_policy()
    with pytest.raises(InformationTemporalMetadataPolicyError):
        replace(policy, min_cross_source_observations=1).validate()


def test_policy_requires_compatible_base_and_freshness_policies() -> None:
    policy = load_information_temporal_metadata_policy()
    base = replace(load_information_policy(), raw_content_logging_allowed=True)
    with pytest.raises(InformationPolicyError):
        policy.validate(information_policy=base)
    freshness = replace(
        load_information_freshness_policy(),
        model_temporal_inference_allowed=True,
    )
    with pytest.raises(InformationFreshnessPolicyError):
        policy.validate(freshness_policy=freshness)
