from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_information.research_evidence_policy import (
    InformationResearchEvidencePolicyError,
    load_information_research_evidence_policy,
    parse_information_research_evidence_policy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "policies/information_research_evidence_policy.json"


def payload():
    return json.loads(POLICY.read_text(encoding="utf-8"))


def test_policy_loads():
    policy = load_information_research_evidence_policy(POLICY)
    assert policy.milestone == "P4.6b"
    assert policy.max_sources == 12


@pytest.mark.parametrize(
    "field",
    [
        "raw_source_logging_allowed",
        "source_body_persistence_allowed",
        "live_provider_registration_allowed",
        "phase3_runtime_registration_allowed",
        "memory_write_allowed",
        "phase5_storage_runtime_allowed",
        "external_action_allowed",
        "background_execution_allowed",
        "model_claim_generation_allowed",
        "semantic_entailment_inference_allowed",
    ],
)
def test_prohibited_capabilities_fail_closed(field):
    value = payload()
    value[field] = True
    with pytest.raises(InformationResearchEvidencePolicyError):
        parse_information_research_evidence_policy(value)


@pytest.mark.parametrize(
    "field",
    [
        "research_run_revalidation_required",
        "injection_inspection_required",
        "freshness_assessment_required",
        "grounding_revalidation_required",
        "partial_research_preserved",
        "rejected_source_metadata_preserved",
    ],
)
def test_required_controls_cannot_be_disabled(field):
    value = payload()
    value[field] = False
    with pytest.raises(InformationResearchEvidencePolicyError):
        parse_information_research_evidence_policy(value)


def test_unknown_key_rejected():
    value = payload()
    value["surprise"] = True
    with pytest.raises(InformationResearchEvidencePolicyError):
        parse_information_research_evidence_policy(value)


def test_limit_change_requires_version_change():
    value = payload()
    value["limits"]["max_sources"] = 13
    with pytest.raises(InformationResearchEvidencePolicyError):
        parse_information_research_evidence_policy(value)

@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_name", "wrong"),
        ("version", "2.0.0"),
        ("phase", "5"),
        ("milestone", "P4.7"),
        ("status", "wrong"),
        ("permission_id", "memory.write"),
    ],
)
def test_identity_bindings_are_exact(field, value):
    item = payload()
    item[field] = value
    with pytest.raises(InformationResearchEvidencePolicyError):
        parse_information_research_evidence_policy(item)


def test_duplicate_json_key_rejected(tmp_path):
    text = POLICY.read_text(encoding="utf-8")
    text = text.replace(
        '"version": "1.0.0",',
        '"version": "1.0.0",\n  "version": "1.0.0",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InformationResearchEvidencePolicyError):
        load_information_research_evidence_policy(path)


def test_vocabulary_reordering_rejected():
    item = payload()
    item["allowed_pipeline_outcomes"] = list(reversed(item["allowed_pipeline_outcomes"]))
    with pytest.raises(InformationResearchEvidencePolicyError):
        parse_information_research_evidence_policy(item)
