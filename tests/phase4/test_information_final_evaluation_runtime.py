from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from _information_final_evaluation_helpers import ROOT, benchmark, policy
from alice_information.final_evaluation_runtime import (
    CANONICAL_RUNTIME_MANIFEST_DIGEST,
    InformationFinalEvaluationRuntimeError,
    information_runtime_backed_report_digest,
    load_information_final_evaluation_runtime_manifest,
    run_runtime_backed_information_final_evaluation,
    runtime_backed_report_to_dict,
    verify_information_final_evaluation_runtime_evidence,
    verify_information_runtime_backed_evaluation_report,
)


def _runtime_manifest():
    return load_information_final_evaluation_runtime_manifest(
        ROOT
        / "benchmarks/phase4/information_final_evaluation_runtime_v1.json",
        benchmark=benchmark(),
    )


def _repository(tmp_path, manifest):
    for relative in manifest.target_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic runtime target: {relative}\n", encoding="utf-8")
    return tmp_path


def _runner(manifest, *, execution_returncode: int = 0, marker: bool = True):
    collection_lines = [f"{path}::test_runtime_binding" for path in manifest.target_files]
    collection_lines.append("640 tests collected in 0.01s")
    if marker:
        collection_lines.append("alice_p48_network_guard=active")
    execution_lines = ["640 passed in 0.02s"]
    if marker:
        execution_lines.append("alice_p48_network_guard=active")

    def run(command, **_kwargs):
        collect_only = "--collect-only" in command
        return subprocess.CompletedProcess(
            command,
            0 if collect_only else execution_returncode,
            stdout="\n".join(collection_lines if collect_only else execution_lines),
            stderr="",
        )

    return run


def test_runtime_manifest_is_canonical_and_covers_all_cases() -> None:
    manifest = _runtime_manifest()
    assert manifest.digest == CANONICAL_RUNTIME_MANIFEST_DIGEST
    assert len(manifest.target_files) == 28
    assert tuple(item.case_id for item in manifest.case_targets) == tuple(
        item.case_id for item in benchmark().cases
    )
    assert all("final_evaluation" not in path for path in manifest.target_files)


def test_runtime_manifest_substitution_is_rejected(tmp_path) -> None:
    source = json.loads(
        (
            ROOT
            / "benchmarks/phase4/information_final_evaluation_runtime_v1.json"
        ).read_text(encoding="utf-8")
    )
    source["case_targets"][0]["target_files"] = [
        "tests/phase4/test_information_policy.py"
    ]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(InformationFinalEvaluationRuntimeError):
        load_information_final_evaluation_runtime_manifest(
            path,
            benchmark=benchmark(),
        )


def test_runtime_backed_evaluation_binds_real_test_evidence(tmp_path) -> None:
    manifest = _runtime_manifest()
    repo = _repository(tmp_path, manifest)
    report = run_runtime_backed_information_final_evaluation(
        repository_root=repo,
        benchmark=benchmark(),
        policy=policy(),
        manifest=manifest,
        command_runner=_runner(manifest),
        snapshotter=lambda _root: "a" * 64,
    )
    assert report.passed
    assert report.runtime_evidence.passed
    assert report.runtime_evidence.collected_test_count == 640
    assert report.runtime_evidence.passed_test_count == 640
    assert report.runtime_evidence.skipped_test_count == 0
    assert report.runtime_evidence.network_guard_active
    assert len(report.runtime_evidence.case_evidence) == 24
    assert all(item.collected_test_count > 0 for item in report.runtime_evidence.case_evidence)
    assert all(item.passed for item in report.runtime_evidence.case_evidence)
    value = runtime_backed_report_to_dict(report)
    assert value["passed"] is True
    assert "submissions" not in json.dumps(value)


def test_runtime_failure_fails_all_benchmark_cases(tmp_path) -> None:
    manifest = _runtime_manifest()
    repo = _repository(tmp_path, manifest)
    report = run_runtime_backed_information_final_evaluation(
        repository_root=repo,
        benchmark=benchmark(),
        policy=policy(),
        manifest=manifest,
        command_runner=_runner(manifest, execution_returncode=1),
        snapshotter=lambda _root: "b" * 64,
    )
    assert not report.passed
    assert not report.runtime_evidence.passed
    assert report.evaluation_report.passed_case_count == 0
    assert report.evaluation_report.critical_case_failure_count == 24


def test_runtime_requires_network_guard_marker(tmp_path) -> None:
    manifest = _runtime_manifest()
    repo = _repository(tmp_path, manifest)
    with pytest.raises(InformationFinalEvaluationRuntimeError):
        run_runtime_backed_information_final_evaluation(
            repository_root=repo,
            benchmark=benchmark(),
            policy=policy(),
            manifest=manifest,
            command_runner=_runner(manifest, marker=False),
            snapshotter=lambda _root: "c" * 64,
        )


def test_runtime_rejects_repository_mutation(tmp_path) -> None:
    manifest = _runtime_manifest()
    repo = _repository(tmp_path, manifest)
    values = iter(("d" * 64, "e" * 64))
    with pytest.raises(InformationFinalEvaluationRuntimeError):
        run_runtime_backed_information_final_evaluation(
            repository_root=repo,
            benchmark=benchmark(),
            policy=policy(),
            manifest=manifest,
            command_runner=_runner(manifest),
            snapshotter=lambda _root: next(values),
        )


def test_runtime_evidence_and_outer_report_tamper_are_rejected(tmp_path) -> None:
    manifest = _runtime_manifest()
    repo = _repository(tmp_path, manifest)
    report = run_runtime_backed_information_final_evaluation(
        repository_root=repo,
        benchmark=benchmark(),
        policy=policy(),
        manifest=manifest,
        command_runner=_runner(manifest),
        snapshotter=lambda _root: "f" * 64,
    )
    with pytest.raises(InformationFinalEvaluationRuntimeError):
        verify_information_final_evaluation_runtime_evidence(
            replace(report.runtime_evidence, passed_test_count=639)
        )
    tampered = replace(report, passed=False)
    tampered = replace(
        tampered,
        report_digest=information_runtime_backed_report_digest(tampered),
    )
    with pytest.raises(InformationFinalEvaluationRuntimeError):
        verify_information_runtime_backed_evaluation_report(tampered)
