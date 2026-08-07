"""M2-CLOSEOUT contract and admission-review tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.m2_closeout import (
    M2_CLOSEOUT_COMPONENTS,
    M2CloseoutReport,
    M2ComponentEvaluation,
    ShadowMigrationAdmissionDecision,
)
from cognitive_kernel.m2_closeout_evaluation import (
    ADMITTED_STAGES,
    EXCLUDED_STAGES,
    run_m2_closeout_evaluation,
)


def evaluated(tmp_path: Path):
    return run_m2_closeout_evaluation(
        workspace=tmp_path / "evaluation"
    )


def test_closeout_contains_every_m2_component(
    tmp_path: Path,
) -> None:
    report, _, _ = evaluated(tmp_path)
    assert {
        item.component_id for item in report.component_results
    } == M2_CLOSEOUT_COMPONENTS


def test_every_component_passes(tmp_path: Path) -> None:
    report, _, _ = evaluated(tmp_path)
    assert all(
        item.state == "passed"
        for item in report.component_results
    )


def test_report_crosses_no_activation_boundary(
    tmp_path: Path,
) -> None:
    report, _, _ = evaluated(tmp_path)
    assert report.production_influence is False
    assert report.canonical_authority_transfer is False
    assert report.private_payload_read is False
    assert report.phase2_migration_started is False


def test_report_rejects_production_influence(
    tmp_path: Path,
) -> None:
    report, _, _ = evaluated(tmp_path)
    tampered = replace(report, production_influence=True)
    with pytest.raises(CognitiveKernelContractError):
        tampered.validate()


def test_report_rejects_missing_component(
    tmp_path: Path,
) -> None:
    report, _, _ = evaluated(tmp_path)
    tampered = replace(
        report,
        component_results=report.component_results[:-1],
    )
    with pytest.raises(CognitiveKernelContractError):
        tampered.validate()


def test_component_digest_tamper_is_detected(
    tmp_path: Path,
) -> None:
    report, _, _ = evaluated(tmp_path)
    first = report.component_results[0]
    tampered = replace(first, result_sha256="f" * 64)
    with pytest.raises(CognitiveKernelContractError):
        tampered.validate()


def test_admission_scope_is_preparatory_and_read_only(
    tmp_path: Path,
) -> None:
    _, decision, _ = evaluated(tmp_path)
    assert decision.outcome == "admitted_preparatory_read_only"
    assert set(decision.admitted_stages) == set(ADMITTED_STAGES)
    assert set(decision.excluded_stages) == set(EXCLUDED_STAGES)


def test_admission_does_not_start_phase2_or_unblock_p5_1e(
    tmp_path: Path,
) -> None:
    _, decision, _ = evaluated(tmp_path)
    assert decision.phase2_migration_started is False
    assert decision.p5_1e_unblocked is False
    assert decision.production_write_mirroring is False
    assert decision.canonical_authority_transfer is False
    assert decision.private_payload_read is False


def test_admission_rejects_stage_overlap(
    tmp_path: Path,
) -> None:
    report, decision, _ = evaluated(tmp_path)
    with pytest.raises(CognitiveKernelContractError):
        ShadowMigrationAdmissionDecision.create(
            envelope=decision.envelope,
            decision_id=decision.decision_id,
            evaluation_id=report.evaluation_id,
            outcome=decision.outcome,
            admitted_stages=decision.admitted_stages,
            excluded_stages=(
                *decision.excluded_stages,
                decision.admitted_stages[0],
            ),
            reason_codes=decision.reason_codes,
            decided_at=decision.decided_at,
        )


def test_admission_requires_complete_preparatory_scope(
    tmp_path: Path,
) -> None:
    report, decision, _ = evaluated(tmp_path)
    with pytest.raises(CognitiveKernelContractError):
        ShadowMigrationAdmissionDecision.create(
            envelope=decision.envelope,
            decision_id=decision.decision_id,
            evaluation_id=report.evaluation_id,
            outcome=decision.outcome,
            admitted_stages=decision.admitted_stages[:-1],
            excluded_stages=decision.excluded_stages,
            reason_codes=decision.reason_codes,
            decided_at=decision.decided_at,
        )


def test_admission_is_bound_to_its_evaluation(
    tmp_path: Path,
) -> None:
    report, decision, _ = evaluated(tmp_path)
    tampered = replace(
        decision,
        evaluation_id="different-evaluation",
    )
    with pytest.raises(CognitiveKernelContractError):
        tampered.assert_supported_by(report)
