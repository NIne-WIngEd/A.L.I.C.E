"""M2-CLOSEOUT cross-prototype integration evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

from cognitive_kernel.m2_closeout_evaluation import (
    run_m2_closeout_evaluation,
)


def test_cross_prototype_evaluation_persists_report(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reports" / "closeout.json"
    report, decision, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
        report_path=destination,
    )
    report.validate()
    decision.assert_supported_by(report)
    assert destination.is_file()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["artifact_sha256"] == payload["artifact_sha256"]


def test_cross_prototype_lineage_is_complete(
    tmp_path: Path,
) -> None:
    _, _, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
    )
    lineage = payload["full_evaluation_content"]["lineage"]
    assert {
        "evidence_id",
        "claim_version_id",
        "candidate_id",
        "adjudication_id",
        "episode_id",
        "projection_version_id",
        "context_packet_id",
        "retrieval_trace_id",
        "workflow_id",
        "curation_receipt_id",
        "deletion_receipt_id",
    } == set(lineage)


def test_each_persistent_plane_reports_integrity(
    tmp_path: Path,
) -> None:
    _, _, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
    )
    details = payload[
        "full_evaluation_content"
    ]["component_details"]
    assert details["claim_authority"]["integrity_valid"] is True
    assert (
        details["shadow_adjudication"]["integrity_state"]
        == "verified"
    )
    assert details["projection_fabric"]["integrity_valid"] is True
    assert details["bounded_serving"]["integrity_healthy"] is True
    assert details["durable_workflow"]["integrity_healthy"] is True
    assert details["deletion_propagation"]["integrity_healthy"] is True


def test_shadow_adjudication_writes_no_canonical_claim(
    tmp_path: Path,
) -> None:
    _, _, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
    )
    shadow = payload[
        "full_evaluation_content"
    ]["component_details"]["shadow_adjudication"]
    assert shadow["canonical_claim_written"] is False
    assert shadow["candidate_state"] == "eligible"


def test_projection_graph_and_vector_paths_are_exercised(
    tmp_path: Path,
) -> None:
    _, _, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
    )
    projection = payload[
        "full_evaluation_content"
    ]["component_details"]["projection_fabric"]
    assert projection["graph_neighbor_count"] >= 1
    assert projection["vector_result_count"] >= 1


def test_serving_packet_is_healthy_and_not_degraded(
    tmp_path: Path,
) -> None:
    _, _, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
    )
    serving = payload[
        "full_evaluation_content"
    ]["component_details"]["bounded_serving"]
    assert serving["hydrated_item_count"] >= 1
    assert serving["fallback_used"] is False
    assert serving["stale_index_observed"] is False


def test_workflow_and_deletion_rehearsal_complete(
    tmp_path: Path,
) -> None:
    _, _, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
    )
    details = payload[
        "full_evaluation_content"
    ]["component_details"]
    assert details["durable_workflow"]["outcome"] == "completed"
    assert (
        details["deletion_propagation"]["required_plane_count"]
        == 4
    )


def test_evaluation_uses_external_private_store_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_m2_closeout_evaluation(workspace=workspace)
    private_root = workspace / "private-stores"
    public_root = workspace / "public-repository-sentinel"
    assert private_root.is_dir()
    assert public_root.is_dir()
    assert list(private_root.glob("*.sqlite3"))
    assert not list(public_root.rglob("*.sqlite3"))


def test_evaluation_is_deterministic_across_clean_workspaces(
    tmp_path: Path,
) -> None:
    first, first_decision, first_payload = (
        run_m2_closeout_evaluation(
            workspace=tmp_path / "first"
        )
    )
    second, second_decision, second_payload = (
        run_m2_closeout_evaluation(
            workspace=tmp_path / "second"
        )
    )
    assert first.report_sha256 == second.report_sha256
    assert (
        first_decision.decision_sha256
        == second_decision.decision_sha256
    )
    assert (
        first_payload["artifact_sha256"]
        == second_payload["artifact_sha256"]
    )


def test_admission_payload_names_next_gate(
    tmp_path: Path,
) -> None:
    _, decision, payload = run_m2_closeout_evaluation(
        workspace=tmp_path / "workspace",
    )
    assert decision.outcome == "admitted_preparatory_read_only"
    assert (
        payload["admission_scope"]["next_gate"]
        == "implement Stage A inventory and registration plus "
        "Stage B read-only contract adapters"
    )
