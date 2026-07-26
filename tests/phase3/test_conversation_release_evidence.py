from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from alice_conversation.release_evidence import (
    Phase3ReleaseEvidenceError,
    execute_phase3_release_evidence,
    load_phase3_release_evidence_manifest,
)
from _release_audit_helpers import ROOT, benchmark


@dataclass
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _manifest():
    return load_phase3_release_evidence_manifest(
        ROOT / "benchmarks/phase3/conversation_release_evidence_v1.json",
        benchmark=benchmark(),
    )


def _repository(tmp_path: Path, manifest):
    root=tmp_path/"repo"
    for target in {x for case in manifest.cases for x in case.pytest_targets}:
        path=root/target
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("def test_placeholder(): assert True\n",encoding="utf-8")
    (root/"src").mkdir(exist_ok=True)
    return root


def test_evidence_manifest_covers_every_benchmark_case():
    value=_manifest()
    assert {x.case_id for x in value.cases}=={x.case_id for x in benchmark().cases}
    assert value.manifest_id=="phase3-conversation-release-evidence-v1"


def test_evidence_manifest_rejects_benchmark_mismatch(tmp_path):
    raw=json.loads((ROOT/"benchmarks/phase3/conversation_release_evidence_v1.json").read_text())
    raw["benchmark_id"]="other"
    path=tmp_path/"manifest.json"; path.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(Phase3ReleaseEvidenceError,match="different benchmark"):
        load_phase3_release_evidence_manifest(path,benchmark=benchmark())


def test_evidence_manifest_rejects_missing_case(tmp_path):
    raw=json.loads((ROOT/"benchmarks/phase3/conversation_release_evidence_v1.json").read_text())
    raw["cases"]=raw["cases"][:-1]
    path=tmp_path/"manifest.json"; path.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(Phase3ReleaseEvidenceError,match="exactly match"):
        load_phase3_release_evidence_manifest(path,benchmark=benchmark())


def test_evidence_manifest_rejects_path_escape(tmp_path):
    raw=json.loads((ROOT/"benchmarks/phase3/conversation_release_evidence_v1.json").read_text())
    raw["cases"][0]["pytest_targets"]=["../secret.py"]
    path=tmp_path/"manifest.json"; path.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(Phase3ReleaseEvidenceError,match="targets"):
        load_phase3_release_evidence_manifest(path,benchmark=benchmark())


def test_all_passing_targets_create_passing_submissions(tmp_path):
    manifest=_manifest(); repo=_repository(tmp_path,manifest)
    run=execute_phase3_release_evidence(repository_root=repo,repository_commit="a"*40,benchmark=benchmark(),manifest=manifest,runner=lambda *a,**k: Result(0,"ok",""))
    assert run.passed_target_count==run.target_count
    assert all(not item.violation_codes and item.signals for item in run.submissions)


def test_failed_target_marks_affected_cases(tmp_path):
    manifest=_manifest(); repo=_repository(tmp_path,manifest)
    failing=manifest.cases[0].pytest_targets[0]
    def runner(command,**kwargs):
        return Result(1,"","failed") if command[-2]==failing else Result(0,"ok","")
    run=execute_phase3_release_evidence(repository_root=repo,repository_commit="a"*40,benchmark=benchmark(),manifest=manifest,runner=runner)
    assert run.passed_target_count==run.target_count-1
    affected=[x for x in run.submissions if x.violation_codes]
    assert affected and all(x.violation_codes==("release_evidence_test_failed",) for x in affected)


def test_evidence_run_digest_is_deterministic(tmp_path):
    manifest=_manifest(); repo=_repository(tmp_path,manifest)
    runner=lambda *a,**k: Result(0,"stable","stable")
    first=execute_phase3_release_evidence(repository_root=repo,repository_commit="a"*40,benchmark=benchmark(),manifest=manifest,runner=runner)
    second=execute_phase3_release_evidence(repository_root=repo,repository_commit="a"*40,benchmark=benchmark(),manifest=manifest,runner=runner)
    assert first.run_digest==second.run_digest


def test_evidence_run_is_bound_to_repository_commit(tmp_path):
    manifest=_manifest(); repo=_repository(tmp_path,manifest)
    runner=lambda *a,**k: Result(0,"stable","")
    first=execute_phase3_release_evidence(repository_root=repo,repository_commit="a"*40,benchmark=benchmark(),manifest=manifest,runner=runner)
    second=execute_phase3_release_evidence(repository_root=repo,repository_commit="b"*40,benchmark=benchmark(),manifest=manifest,runner=runner)
    assert first.run_digest!=second.run_digest
