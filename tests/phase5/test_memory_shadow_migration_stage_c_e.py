from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    DestinationCandidateEvaluation,
    DestinationCandidateProfile,
    ProductHostScope,
    ShadowReadComparisonReceipt,
    ShadowReadObservation,
    ShadowReadWorkload,
    build_m2_destination_candidate_profile,
)
from cognitive_kernel.canonical import canonical_sha256
from cognitive_kernel.shadow_migration_stage_c_e_evaluation import (
    _assert_report_outside_repo,
    build_synthetic_stage_c_e_report,
)

TS = "2026-08-07T04:30:00Z"


def scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="stage-ce-test-host",
        schema_version="1.0.0",
        encryption_domain="stage-ce-test-domain",
    )


def workload(*, workload_class: str = "synthetic", authorization=None):
    return ShadowReadWorkload.create(
        scope=scope(),
        workload_id="workload.1",
        workload_class=workload_class,
        query_sha256=canonical_sha256({"q": 1}),
        expected_record_ids=("claim.1",),
        expected_conflict_record_ids=("conflict.1",),
        expected_correction_record_ids=("correction.1",),
        expected_deleted_record_ids=("claim.deleted",),
        authorization_reference_id=authorization,
        created_at=TS,
    )


def observation(candidate_id: str, *, deleted=(), private=False, isolated=True, latency=10):
    return ShadowReadObservation.create(
        scope=scope(),
        workload_id="workload.1",
        candidate_id=candidate_id,
        result_record_ids=("claim.1",),
        conflict_record_ids=("conflict.1",),
        correction_record_ids=("correction.1",),
        deleted_record_ids_returned=deleted,
        latency_ms=latency,
        staleness_ms=2,
        product_isolation_passed=isolated,
        private_payload_exposed=private,
        explanation_trace_sha256=canonical_sha256({"candidate": candidate_id}),
        observed_at=TS,
    )


def test_candidate_profile_is_nonproduction_and_digest_bound():
    profile = build_m2_destination_candidate_profile(scope=scope(), created_at=TS)
    profile.validate()
    assert profile.production_authority is False
    assert profile.production_influence is False
    assert "distributed_multi_region" in profile.deployment_profiles
    with pytest.raises(CognitiveKernelContractError):
        DestinationCandidateProfile.create(
            scope=scope(), candidate_id="bad", backend_types=("sqlite",),
            component_ids=("component",), contract_roles=("claim_authority",),
            deployment_profiles=("embedded_edge",), created_at=TS,
            production_authority=True,
        )


def test_owner_authorized_workload_requires_authorization_reference():
    with pytest.raises(CognitiveKernelContractError):
        workload(workload_class="owner_authorized")
    item = workload(workload_class="owner_authorized", authorization="owner.authorization.1")
    item.validate()


def test_synthetic_workload_rejects_false_authorization_claim():
    with pytest.raises(CognitiveKernelContractError):
        workload(authorization="owner.authorization.1")


def test_comparison_detects_improvement_without_production_influence():
    receipt = ShadowReadComparisonReceipt.create(
        workload=workload(),
        baseline=observation("phase2.baseline", latency=12),
        candidate=observation("candidate.1", latency=8),
        compared_at=TS,
    )
    assert receipt.state == "candidate_improved"
    assert receipt.production_influence is False


def test_comparison_detects_deleted_record_regression():
    receipt = ShadowReadComparisonReceipt.create(
        workload=workload(),
        baseline=observation("phase2.baseline"),
        candidate=observation("candidate.1", deleted=("claim.deleted",)),
        compared_at=TS,
    )
    assert receipt.state == "candidate_degraded"
    assert receipt.deletion_correct is False


def test_comparison_detects_privacy_or_isolation_regression():
    receipt = ShadowReadComparisonReceipt.create(
        workload=workload(),
        baseline=observation("phase2.baseline"),
        candidate=observation("candidate.1", private=True, isolated=False),
        compared_at=TS,
    )
    assert receipt.state == "candidate_degraded"
    assert receipt.privacy_correct is False
    assert receipt.product_isolation_correct is False


def test_evaluation_never_selects_production_authority():
    profile = build_m2_destination_candidate_profile(scope=scope(), created_at=TS)
    receipt = ShadowReadComparisonReceipt.create(
        workload=workload(), baseline=observation("phase2.baseline", latency=12),
        candidate=observation(profile.candidate_id, latency=8), compared_at=TS,
    )
    evaluation = DestinationCandidateEvaluation.create(
        profile=profile, comparisons=(receipt,), evaluated_at=TS
    )
    assert evaluation.recommendation == "eligible_for_next_research_gate"
    assert evaluation.production_selection is False


def test_synthetic_report_is_deterministic_and_truthful():
    first = build_synthetic_stage_c_e_report()
    second = build_synthetic_stage_c_e_report()
    assert first == second
    assert first["material_state"]["stage_c_e_prototype_operational"] is True
    assert first["material_state"]["production_influence"] is False
    assert first["material_state"]["p5_1e_unblocked"] is False


def test_report_cannot_be_written_inside_repository(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError):
        _assert_report_outside_repo(repo / "report.json", repo)
    _assert_report_outside_repo(tmp_path / "vault" / "report.json", repo)


def test_component_policy_remains_governance_compatible():
    policy_path = Path("policies/memory_shadow_migration_stage_c_e_policy.json")
    if not policy_path.exists():
        pytest.skip("policy is added by the repository patcher")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert "capability_state_semantics" not in policy
    assert policy["capability_ceiling"] is False
    assert policy["production_status"] == "not_activated_by_this_profile"
