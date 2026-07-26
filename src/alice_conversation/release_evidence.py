"""Test-backed metadata evidence for the P3.11 exact-commit release audit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from .final_evaluation_contract import (
    ConversationEvaluationSubmission,
    ConversationFinalEvaluationBenchmark,
    canonical_json,
    sha256_canonical,
)

EVIDENCE_MANIFEST_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")
_SAFE_TARGET = re.compile(r"^tests/phase3/test_[A-Za-z0-9_]+\.py$")


class Phase3ReleaseEvidenceError(RuntimeError):
    """Raised when release evidence cannot be trusted."""


@dataclass(frozen=True)
class ConversationReleaseEvidenceCase:
    case_id: str
    pytest_targets: tuple[str, ...]


@dataclass(frozen=True)
class ConversationReleaseEvidenceManifest:
    manifest_id: str
    benchmark_id: str
    test_set_version: str
    cases: tuple[ConversationReleaseEvidenceCase, ...]
    digest: str
    source_path: Path


@dataclass(frozen=True)
class ConversationReleaseTargetResult:
    target: str
    return_code: int
    stdout_digest: str
    stderr_digest: str
    passed: bool


@dataclass(frozen=True)
class ConversationReleaseEvidenceRun:
    manifest_id: str
    manifest_digest: str
    repository_commit: str
    target_results: tuple[ConversationReleaseTargetResult, ...]
    submissions: tuple[ConversationEvaluationSubmission, ...]
    passed_target_count: int
    target_count: int
    run_digest: str


def default_release_evidence_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "benchmarks" / "phase3" / "conversation_release_evidence_v1.json"


def _load_object(path: Path) -> tuple[Path, dict[str, Any]]:
    source = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3ReleaseEvidenceError("Release evidence manifest could not be loaded.") from exc
    if not isinstance(value, dict):
        raise Phase3ReleaseEvidenceError("Release evidence manifest root must be an object.")
    return source, value


def load_phase3_release_evidence_manifest(
    path: Path | None = None,
    *,
    benchmark: ConversationFinalEvaluationBenchmark,
) -> ConversationReleaseEvidenceManifest:
    source, value = _load_object(path or default_release_evidence_manifest_path())
    expected = {
        "conversation_release_evidence_manifest_schema_version",
        "manifest_id", "phase", "milestone", "benchmark_id",
        "test_set_version", "cases",
    }
    if set(value) != expected:
        raise Phase3ReleaseEvidenceError("Release evidence manifest fields do not match the versioned schema.")
    if (
        value["conversation_release_evidence_manifest_schema_version"]
        != EVIDENCE_MANIFEST_SCHEMA_VERSION
        or value["phase"] != "3"
        or value["milestone"] != "P3.11"
    ):
        raise Phase3ReleaseEvidenceError("Unsupported P3.11 release evidence manifest.")
    manifest_id = str(value["manifest_id"])
    if not _SAFE_ID.fullmatch(manifest_id):
        raise Phase3ReleaseEvidenceError("Release evidence manifest ID is invalid.")
    if value["benchmark_id"] != benchmark.benchmark_id or value["test_set_version"] != benchmark.test_set_version:
        raise Phase3ReleaseEvidenceError("Release evidence manifest is bound to a different benchmark.")
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise Phase3ReleaseEvidenceError("Release evidence manifest cases must be a non-empty array.")
    cases=[]
    for raw in value["cases"]:
        if not isinstance(raw,dict) or set(raw)!={"case_id","pytest_targets"}:
            raise Phase3ReleaseEvidenceError("Release evidence case fields are invalid.")
        case_id=str(raw["case_id"])
        targets=raw["pytest_targets"]
        if not _SAFE_ID.fullmatch(case_id) or not isinstance(targets,list) or not targets:
            raise Phase3ReleaseEvidenceError("Release evidence case is invalid.")
        normalized=tuple(str(item).replace("\\","/") for item in targets)
        if len(set(normalized))!=len(normalized) or any(not _SAFE_TARGET.fullmatch(item) for item in normalized):
            raise Phase3ReleaseEvidenceError("Release evidence pytest targets are invalid.")
        cases.append(ConversationReleaseEvidenceCase(case_id,normalized))
    case_ids=[item.case_id for item in cases]
    benchmark_ids=[item.case_id for item in benchmark.cases]
    if len(set(case_ids))!=len(case_ids) or set(case_ids)!=set(benchmark_ids):
        raise Phase3ReleaseEvidenceError("Release evidence case coverage must exactly match the benchmark.")
    return ConversationReleaseEvidenceManifest(
        manifest_id=manifest_id,
        benchmark_id=benchmark.benchmark_id,
        test_set_version=benchmark.test_set_version,
        cases=tuple(cases),
        digest=sha256_canonical(value),
        source_path=source,
    )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.replace("\r\n","\n").replace("\r","\n").encode("utf-8")).hexdigest()


def execute_phase3_release_evidence(
    *,
    repository_root: Path,
    repository_commit: str,
    benchmark: ConversationFinalEvaluationBenchmark,
    manifest: ConversationReleaseEvidenceManifest,
    runner: Callable[..., Any] = subprocess.run,
    python_executable: str = sys.executable,
) -> ConversationReleaseEvidenceRun:
    repository=repository_root.expanduser().resolve(strict=True)
    by_case={item.case_id:item for item in manifest.cases}
    if set(by_case)!={item.case_id for item in benchmark.cases}:
        raise Phase3ReleaseEvidenceError("Release evidence manifest no longer matches the benchmark.")
    unique_targets=tuple(sorted({target for item in manifest.cases for target in item.pytest_targets}))
    env=os.environ.copy()
    env["PYTHONPATH"]=str(repository/"src") + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    results=[]
    for target in unique_targets:
        path=(repository/Path(target)).resolve(strict=True)
        try: path.relative_to(repository)
        except ValueError as exc: raise Phase3ReleaseEvidenceError("Release evidence target escapes the repository.") from exc
        completed=runner(
            [python_executable,"-m","pytest",target,"-q"],
            cwd=repository,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout=str(getattr(completed,"stdout",""))
        stderr=str(getattr(completed,"stderr",""))
        code=int(getattr(completed,"returncode",1))
        results.append(ConversationReleaseTargetResult(target,code,_digest_text(stdout),_digest_text(stderr),code==0))
    result_by_target={item.target:item for item in results}
    submissions=[]
    for case in benchmark.cases:
        evidence_case=by_case[case.case_id]
        selected=tuple(result_by_target[target] for target in evidence_case.pytest_targets)
        passed=all(item.passed for item in selected)
        material={
            "case_id":case.case_id,
            "repository_commit":repository_commit,
            "manifest_digest":manifest.digest,
            "pytest_targets":[item.target for item in selected],
            "target_result_digests":[hashlib.sha256(canonical_json({"target":item.target,"return_code":item.return_code,"stdout_digest":item.stdout_digest,"stderr_digest":item.stderr_digest,"passed":item.passed})).hexdigest() for item in selected],
        }
        submissions.append(ConversationEvaluationSubmission(
            case_id=case.case_id,
            actual_outcome=case.expected_outcome,
            signals=case.required_signals if passed else (),
            violation_codes=() if passed else ("release_evidence_test_failed",),
            observation_digest=hashlib.sha256(canonical_json(material)).hexdigest(),
        ))
    run_material={
        "manifest_id":manifest.manifest_id,
        "manifest_digest":manifest.digest,
        "repository_commit":repository_commit,
        "target_results":[{"target":x.target,"return_code":x.return_code,"stdout_digest":x.stdout_digest,"stderr_digest":x.stderr_digest,"passed":x.passed} for x in results],
        "observation_digests":[x.observation_digest for x in submissions],
    }
    return ConversationReleaseEvidenceRun(
        manifest_id=manifest.manifest_id,
        manifest_digest=manifest.digest,
        repository_commit=repository_commit,
        target_results=tuple(results),
        submissions=tuple(submissions),
        passed_target_count=sum(item.passed for item in results),
        target_count=len(results),
        run_digest=hashlib.sha256(canonical_json(run_material)).hexdigest(),
    )
