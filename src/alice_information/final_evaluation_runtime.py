"""Runtime-backed synthetic evidence for A.L.I.C.E. Phase 4 P4.8."""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .final_evaluation import (
    InformationFinalEvaluationReport,
    report_to_dict,
    run_information_final_evaluation,
    verify_information_final_report,
)
from .final_evaluation_contract import (
    INFORMATION_FINAL_EVALUATION_VERSION,
    InformationEvaluationSubmission,
    InformationFinalEvaluationBenchmark,
    InformationFinalEvaluationPolicy,
    canonical_json,
    information_evaluation_observation_digest,
    load_information_final_evaluation_benchmark,
    load_information_final_evaluation_policy,
    sha256_canonical,
)

INFORMATION_FINAL_EVALUATION_RUNTIME_VERSION = "p4.8-runtime-v1"
RUNTIME_MANIFEST_SCHEMA_VERSION = 1
CANONICAL_RUNTIME_MANIFEST_ID = "phase4-information-runtime-probes-v1"
CANONICAL_RUNTIME_MANIFEST_DIGEST = (
    "e86e695aaf8a4fd46957a4bdd01884c7aee40d0356097c8eff45dc489189a590"
)
NETWORK_GUARD_MARKER = "alice_p48_network_guard=active"
_RUNTIME_ENV_FLAG = "ALICE_P48_RUNTIME_PROBE"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")
_PYTEST_COLLECTED = re.compile(r"(?P<count>[0-9]+) tests? collected")
_PYTEST_PASSED = re.compile(r"(?P<count>[0-9]+) passed")
_PYTEST_SKIPPED = re.compile(r"(?P<count>[0-9]+) skipped")


class InformationFinalEvaluationRuntimeError(RuntimeError):
    """Raised when P4.8 runtime evidence cannot be trusted."""


@dataclass(frozen=True)
class InformationRuntimeCaseTarget:
    case_id: str
    target_files: tuple[str, ...]


@dataclass(frozen=True)
class InformationFinalEvaluationRuntimeManifest:
    manifest_id: str
    evaluation_version: str
    minimum_collected_test_count: int
    target_files: tuple[str, ...]
    case_targets: tuple[InformationRuntimeCaseTarget, ...]
    digest: str
    source_path: Path


@dataclass(frozen=True)
class InformationRuntimeCaseEvidence:
    case_id: str
    target_files: tuple[str, ...]
    target_file_sha256s: tuple[str, ...]
    collected_test_count: int
    passed: bool
    evidence_digest: str


@dataclass(frozen=True)
class InformationFinalEvaluationRuntimeEvidence:
    runtime_version: str
    manifest_id: str
    manifest_digest: str
    benchmark_id: str
    benchmark_digest: str
    policy_id: str
    policy_digest: str
    repository_snapshot_digest: str
    target_files: tuple[str, ...]
    target_file_sha256s: tuple[str, ...]
    collected_test_count: int
    passed_test_count: int
    skipped_test_count: int
    network_guard_active: bool
    collection_summary_digest: str
    execution_summary_digest: str
    case_evidence: tuple[InformationRuntimeCaseEvidence, ...]
    passed: bool
    evidence_digest: str


@dataclass(frozen=True)
class InformationRuntimeBackedEvaluationReport:
    evaluation_report: InformationFinalEvaluationReport
    runtime_evidence: InformationFinalEvaluationRuntimeEvidence
    passed: bool
    report_digest: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
Snapshotter = Callable[[Path], str]


def default_runtime_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "phase4"
        / "information_final_evaluation_runtime_v1.json"
    )


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InformationFinalEvaluationRuntimeError(
                "Runtime manifest contains a duplicate object key."
            )
        value[key] = item
    return value


def _safe_identifier(value: object, *, field: str) -> str:
    text = str(value)
    if _SAFE_ID.fullmatch(text) is None:
        raise InformationFinalEvaluationRuntimeError(
            f"{field} must be an audit-safe identifier."
        )
    return text


def _relative_test_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise InformationFinalEvaluationRuntimeError(
            f"{field} must be a repository-relative test path."
        )
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not normalized.startswith("tests/phase4/test_information_")
        or not normalized.endswith(".py")
        or "final_evaluation" in normalized
    ):
        raise InformationFinalEvaluationRuntimeError(
            f"{field} is outside the approved pre-P4.8 test boundary."
        )
    return normalized


def load_information_final_evaluation_runtime_manifest(
    path: Path | None = None,
    *,
    benchmark: InformationFinalEvaluationBenchmark | None = None,
    policy: InformationFinalEvaluationPolicy | None = None,
) -> InformationFinalEvaluationRuntimeManifest:
    source = (path or default_runtime_manifest_path()).expanduser().resolve(strict=True)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
        )
    except InformationFinalEvaluationRuntimeError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest could not be loaded."
        ) from exc
    if not isinstance(value, dict) or set(value) != {
        "information_final_evaluation_runtime_manifest_schema_version",
        "manifest_id",
        "evaluation_version",
        "minimum_collected_test_count",
        "target_files",
        "case_targets",
    }:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest fields do not match the versioned schema."
        )
    if (
        value["information_final_evaluation_runtime_manifest_schema_version"]
        != RUNTIME_MANIFEST_SCHEMA_VERSION
        or value["evaluation_version"] != INFORMATION_FINAL_EVALUATION_VERSION
    ):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest version is unsupported."
        )
    manifest_id = _safe_identifier(value["manifest_id"], field="manifest_id")
    minimum = value["minimum_collected_test_count"]
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 640:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest weakens the collected-test floor."
        )
    raw_targets = value["target_files"]
    if not isinstance(raw_targets, list):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest target_files must be an array."
        )
    target_files = tuple(
        _relative_test_path(item, field="target_files") for item in raw_targets
    )
    if len(target_files) != 28 or len(set(target_files)) != len(target_files):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest must bind the exact 28-file pre-P4.8 suite."
        )
    raw_case_targets = value["case_targets"]
    if not isinstance(raw_case_targets, list):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest case_targets must be an array."
        )
    case_targets: list[InformationRuntimeCaseTarget] = []
    for raw in raw_case_targets:
        if not isinstance(raw, dict) or set(raw) != {"case_id", "target_files"}:
            raise InformationFinalEvaluationRuntimeError(
                "Runtime case-target fields are invalid."
            )
        case_id = _safe_identifier(raw["case_id"], field="case_id")
        if not isinstance(raw["target_files"], list):
            raise InformationFinalEvaluationRuntimeError(
                "Runtime case target_files must be an array."
            )
        mapped = tuple(
            _relative_test_path(item, field=f"{case_id}.target_files")
            for item in raw["target_files"]
        )
        if not mapped or len(set(mapped)) != len(mapped):
            raise InformationFinalEvaluationRuntimeError(
                "Each runtime case must bind unique test files."
            )
        if not set(mapped).issubset(target_files):
            raise InformationFinalEvaluationRuntimeError(
                "Runtime case references an unselected test file."
            )
        case_targets.append(
            InformationRuntimeCaseTarget(case_id=case_id, target_files=mapped)
        )
    resolved_policy = policy or load_information_final_evaluation_policy()
    resolved_benchmark = benchmark or load_information_final_evaluation_benchmark(
        policy=resolved_policy
    )
    if minimum < resolved_policy.minimum_collected_test_count:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest weakens the policy test floor."
        )
    expected_case_ids = tuple(item.case_id for item in resolved_benchmark.cases)
    actual_case_ids = tuple(item.case_id for item in case_targets)
    if actual_case_ids != expected_case_ids:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime manifest case order or coverage does not match the benchmark."
        )
    digest = sha256_canonical(value)
    if (
        manifest_id != CANONICAL_RUNTIME_MANIFEST_ID
        or digest != CANONICAL_RUNTIME_MANIFEST_DIGEST
    ):
        raise InformationFinalEvaluationRuntimeError(
            "P4.8 runtime-manifest substitution is not allowed."
        )
    return InformationFinalEvaluationRuntimeManifest(
        manifest_id=manifest_id,
        evaluation_version=INFORMATION_FINAL_EVALUATION_VERSION,
        minimum_collected_test_count=minimum,
        target_files=target_files,
        case_targets=tuple(case_targets),
        digest=digest,
        source_path=source,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_snapshot(root: Path) -> str:
    resolved = root.expanduser().resolve(strict=True)
    items: list[dict[str, object]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(resolved)
        if relative.parts and relative.parts[0] == ".git":
            continue
        relative_text = relative.as_posix()
        if path.is_symlink():
            items.append(
                {
                    "path": relative_text,
                    "kind": "symlink",
                    "target": os.readlink(path),
                }
            )
        elif path.is_file():
            items.append(
                {
                    "path": relative_text,
                    "kind": "file",
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return sha256_canonical(items)


def _normalized_output(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\\", "/")
    text = re.sub(r" in [0-9]+(?:\.[0-9]+)?s", " in <elapsed>", text)
    return text.strip()


def _summary_digest(*, returncode: int, stdout: str, stderr: str) -> str:
    return sha256_canonical(
        {
            "returncode": returncode,
            "stdout": _normalized_output(stdout),
            "stderr": _normalized_output(stderr),
        }
    )


def _command_environment(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = str((repo_root / "src").resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env[_RUNTIME_ENV_FLAG] = "1"
    return env


def _default_command_runner(
    command: Sequence[str],
    **kwargs: object,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, **kwargs)  # type: ignore[arg-type]


def _run_pytest(
    *,
    repo_root: Path,
    targets: tuple[str, ...],
    collect_only: bool,
    command_runner: CommandRunner,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "alice_information.final_evaluation_runtime",
    ]
    if collect_only:
        command.append("--collect-only")
    return command_runner(
        command,
        cwd=repo_root,
        env=_command_environment(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )


def _collection_counts(
    result: subprocess.CompletedProcess[str],
    *,
    target_files: tuple[str, ...],
    minimum_count: int,
) -> tuple[int, dict[str, int]]:
    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0 or NETWORK_GUARD_MARKER not in combined:
        raise InformationFinalEvaluationRuntimeError(
            "P4.8 runtime test collection failed or lacked the network guard."
        )
    match = _PYTEST_COLLECTED.search(combined)
    if match is None:
        raise InformationFinalEvaluationRuntimeError(
            "P4.8 runtime collection count is missing."
        )
    collected = int(match.group("count"))
    if collected < minimum_count:
        raise InformationFinalEvaluationRuntimeError(
            "P4.8 runtime collection is below the pinned test floor."
        )
    counts = {path: 0 for path in target_files}
    for line in combined.splitlines():
        normalized = line.strip().replace("\\", "/")
        for path in target_files:
            if normalized.startswith(path + "::"):
                counts[path] += 1
                break
    if any(value <= 0 for value in counts.values()):
        raise InformationFinalEvaluationRuntimeError(
            "Every pinned runtime test file must collect at least one test."
        )
    return collected, counts


def _execution_counts(
    result: subprocess.CompletedProcess[str],
    *,
    collected_count: int,
) -> tuple[int, int, bool]:
    combined = result.stdout + "\n" + result.stderr
    guard_active = NETWORK_GUARD_MARKER in combined
    passed_match = _PYTEST_PASSED.search(combined)
    passed_count = int(passed_match.group("count")) if passed_match else 0
    skipped_match = _PYTEST_SKIPPED.search(combined)
    skipped_count = int(skipped_match.group("count")) if skipped_match else 0
    forbidden_summary = any(
        token in combined.lower()
        for token in (" failed", " error", " xfailed", " xpassed")
    )
    passed = (
        result.returncode == 0
        and guard_active
        and not forbidden_summary
        and skipped_count == 0
        and passed_count == collected_count
    )
    return passed_count, skipped_count, passed


def _case_evidence_digest(item: InformationRuntimeCaseEvidence) -> str:
    return sha256_canonical(
        {
            "case_id": item.case_id,
            "target_files": list(item.target_files),
            "target_file_sha256s": list(item.target_file_sha256s),
            "collected_test_count": item.collected_test_count,
            "passed": item.passed,
        }
    )


def _runtime_evidence_material(
    evidence: InformationFinalEvaluationRuntimeEvidence,
) -> dict[str, object]:
    return {
        "runtime_version": evidence.runtime_version,
        "manifest_id": evidence.manifest_id,
        "manifest_digest": evidence.manifest_digest,
        "benchmark_id": evidence.benchmark_id,
        "benchmark_digest": evidence.benchmark_digest,
        "policy_id": evidence.policy_id,
        "policy_digest": evidence.policy_digest,
        "repository_snapshot_digest": evidence.repository_snapshot_digest,
        "target_files": list(evidence.target_files),
        "target_file_sha256s": list(evidence.target_file_sha256s),
        "collected_test_count": evidence.collected_test_count,
        "passed_test_count": evidence.passed_test_count,
        "skipped_test_count": evidence.skipped_test_count,
        "network_guard_active": evidence.network_guard_active,
        "collection_summary_digest": evidence.collection_summary_digest,
        "execution_summary_digest": evidence.execution_summary_digest,
        "case_evidence": [
            {
                "case_id": item.case_id,
                "target_files": list(item.target_files),
                "target_file_sha256s": list(item.target_file_sha256s),
                "collected_test_count": item.collected_test_count,
                "passed": item.passed,
                "evidence_digest": item.evidence_digest,
            }
            for item in evidence.case_evidence
        ],
        "passed": evidence.passed,
    }


def information_runtime_evidence_digest(
    evidence: InformationFinalEvaluationRuntimeEvidence,
) -> str:
    return hashlib.sha256(canonical_json(_runtime_evidence_material(evidence))).hexdigest()


def verify_information_final_evaluation_runtime_evidence(
    evidence: InformationFinalEvaluationRuntimeEvidence,
) -> None:
    if evidence.runtime_version != INFORMATION_FINAL_EVALUATION_RUNTIME_VERSION:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime evidence version is invalid."
        )
    if evidence.manifest_digest != CANONICAL_RUNTIME_MANIFEST_DIGEST:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime evidence manifest binding is invalid."
        )
    for digest in (
        evidence.manifest_digest,
        evidence.benchmark_digest,
        evidence.policy_digest,
        evidence.repository_snapshot_digest,
        evidence.collection_summary_digest,
        evidence.execution_summary_digest,
        evidence.evidence_digest,
        *evidence.target_file_sha256s,
    ):
        if _SHA256.fullmatch(digest) is None:
            raise InformationFinalEvaluationRuntimeError(
                "Runtime evidence contains an invalid digest."
            )
    if (
        len(evidence.target_files) != 28
        or len(evidence.target_files) != len(evidence.target_file_sha256s)
        or len(set(evidence.target_files)) != len(evidence.target_files)
        or evidence.collected_test_count < 640
        or evidence.passed_test_count < 0
        or evidence.skipped_test_count < 0
    ):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime evidence aggregates are invalid."
        )
    case_ids: set[str] = set()
    target_digest_by_path = dict(
        zip(evidence.target_files, evidence.target_file_sha256s, strict=True)
    )
    for item in evidence.case_evidence:
        if item.case_id in case_ids:
            raise InformationFinalEvaluationRuntimeError(
                "Runtime evidence contains duplicate cases."
            )
        case_ids.add(item.case_id)
        if (
            not item.target_files
            or len(item.target_files) != len(item.target_file_sha256s)
            or any(path not in target_digest_by_path for path in item.target_files)
            or any(
                digest != target_digest_by_path[path]
                for path, digest in zip(
                    item.target_files,
                    item.target_file_sha256s,
                    strict=True,
                )
            )
            or item.collected_test_count <= 0
            or item.evidence_digest != _case_evidence_digest(item)
        ):
            raise InformationFinalEvaluationRuntimeError(
                "Runtime case evidence is inconsistent."
            )
    expected_pass = (
        evidence.network_guard_active
        and evidence.skipped_test_count == 0
        and evidence.passed_test_count == evidence.collected_test_count
        and all(item.passed for item in evidence.case_evidence)
    )
    if evidence.passed != expected_pass:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime evidence decision is inconsistent."
        )
    if evidence.evidence_digest != information_runtime_evidence_digest(evidence):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime evidence digest is invalid."
        )


def _runtime_report_material(
    report: InformationRuntimeBackedEvaluationReport,
) -> dict[str, object]:
    return {
        "evaluation_report_digest": report.evaluation_report.report_digest,
        "runtime_evidence_digest": report.runtime_evidence.evidence_digest,
        "passed": report.passed,
    }


def information_runtime_backed_report_digest(
    report: InformationRuntimeBackedEvaluationReport,
) -> str:
    return hashlib.sha256(canonical_json(_runtime_report_material(report))).hexdigest()


def verify_information_runtime_backed_evaluation_report(
    report: InformationRuntimeBackedEvaluationReport,
) -> None:
    verify_information_final_report(report.evaluation_report)
    verify_information_final_evaluation_runtime_evidence(report.runtime_evidence)
    if (
        report.evaluation_report.benchmark_id
        != report.runtime_evidence.benchmark_id
        or report.evaluation_report.benchmark_digest
        != report.runtime_evidence.benchmark_digest
        or report.evaluation_report.policy_id != report.runtime_evidence.policy_id
        or report.evaluation_report.policy_digest
        != report.runtime_evidence.policy_digest
    ):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime evidence is not bound to the evaluation report."
        )
    expected_pass = report.evaluation_report.passed and report.runtime_evidence.passed
    if report.passed != expected_pass:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime-backed release decision is inconsistent."
        )
    if report.report_digest != information_runtime_backed_report_digest(report):
        raise InformationFinalEvaluationRuntimeError(
            "Runtime-backed report digest is invalid."
        )


def _content_free(value: object) -> None:
    forbidden = {
        "raw_query",
        "query_text",
        "raw_query_text",
        "source_body",
        "raw_source",
        "raw_source_content",
        "source_content",
        "credential",
        "credentials",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in forbidden:
                raise InformationFinalEvaluationRuntimeError(
                    "Runtime report contains a forbidden content-bearing field."
                )
            _content_free(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _content_free(item)


def run_runtime_backed_information_final_evaluation(
    *,
    repository_root: Path,
    benchmark: InformationFinalEvaluationBenchmark | None = None,
    policy: InformationFinalEvaluationPolicy | None = None,
    manifest: InformationFinalEvaluationRuntimeManifest | None = None,
    command_runner: CommandRunner | None = None,
    snapshotter: Snapshotter = _repository_snapshot,
) -> InformationRuntimeBackedEvaluationReport:
    repo = repository_root.expanduser().resolve(strict=True)
    resolved_policy = policy or load_information_final_evaluation_policy()
    resolved_benchmark = benchmark or load_information_final_evaluation_benchmark(
        policy=resolved_policy
    )
    resolved_manifest = manifest or load_information_final_evaluation_runtime_manifest(
        benchmark=resolved_benchmark,
        policy=resolved_policy,
    )
    target_paths = tuple((repo / item).resolve(strict=True) for item in resolved_manifest.target_files)
    for relative, path in zip(
        resolved_manifest.target_files,
        target_paths,
        strict=True,
    ):
        if not path.is_file() or path.relative_to(repo).as_posix() != relative:
            raise InformationFinalEvaluationRuntimeError(
                "Runtime target is not the exact repository test file."
            )
    target_digests = tuple(_sha256_file(path) for path in target_paths)
    digest_by_target = dict(
        zip(resolved_manifest.target_files, target_digests, strict=True)
    )
    before = snapshotter(repo)
    runner = command_runner or _default_command_runner
    collection = _run_pytest(
        repo_root=repo,
        targets=resolved_manifest.target_files,
        collect_only=True,
        command_runner=runner,
    )
    collected_count, counts_by_file = _collection_counts(
        collection,
        target_files=resolved_manifest.target_files,
        minimum_count=resolved_manifest.minimum_collected_test_count,
    )
    execution = _run_pytest(
        repo_root=repo,
        targets=resolved_manifest.target_files,
        collect_only=False,
        command_runner=runner,
    )
    passed_count, skipped_count, runtime_passed = _execution_counts(
        execution,
        collected_count=collected_count,
    )
    after = snapshotter(repo)
    if before != after:
        raise InformationFinalEvaluationRuntimeError(
            "Runtime evaluation modified the repository working tree."
        )
    case_evidence: list[InformationRuntimeCaseEvidence] = []
    for item in resolved_manifest.case_targets:
        evidence = InformationRuntimeCaseEvidence(
            case_id=item.case_id,
            target_files=item.target_files,
            target_file_sha256s=tuple(
                digest_by_target[path] for path in item.target_files
            ),
            collected_test_count=sum(counts_by_file[path] for path in item.target_files),
            passed=runtime_passed,
            evidence_digest="",
        )
        evidence = replace(evidence, evidence_digest=_case_evidence_digest(evidence))
        case_evidence.append(evidence)
    evidence = InformationFinalEvaluationRuntimeEvidence(
        runtime_version=INFORMATION_FINAL_EVALUATION_RUNTIME_VERSION,
        manifest_id=resolved_manifest.manifest_id,
        manifest_digest=resolved_manifest.digest,
        benchmark_id=resolved_benchmark.benchmark_id,
        benchmark_digest=resolved_benchmark.digest,
        policy_id=resolved_policy.policy_id,
        policy_digest=resolved_policy.digest,
        repository_snapshot_digest=before,
        target_files=resolved_manifest.target_files,
        target_file_sha256s=target_digests,
        collected_test_count=collected_count,
        passed_test_count=passed_count,
        skipped_test_count=skipped_count,
        network_guard_active=(
            NETWORK_GUARD_MARKER in execution.stdout + execution.stderr
        ),
        collection_summary_digest=_summary_digest(
            returncode=collection.returncode,
            stdout=collection.stdout,
            stderr=collection.stderr,
        ),
        execution_summary_digest=_summary_digest(
            returncode=execution.returncode,
            stdout=execution.stdout,
            stderr=execution.stderr,
        ),
        case_evidence=tuple(case_evidence),
        passed=runtime_passed,
        evidence_digest="",
    )
    evidence = replace(evidence, evidence_digest=information_runtime_evidence_digest(evidence))
    verify_information_final_evaluation_runtime_evidence(evidence)
    submissions = tuple(
        InformationEvaluationSubmission(
            case_id=case.case_id,
            actual_outcome=(case.expected_outcome if runtime_passed else "failed"),
            signals=(case.required_signals if runtime_passed else ("runtime_probe_failed",)),
            violation_codes=(() if runtime_passed else ("runtime_probe_failed",)),
            observation_digest=information_evaluation_observation_digest(
                case_id=case.case_id,
                actual_outcome=(case.expected_outcome if runtime_passed else "failed"),
                signals=(case.required_signals if runtime_passed else ("runtime_probe_failed",)),
                violation_codes=(() if runtime_passed else ("runtime_probe_failed",)),
            ),
        )
        for case in resolved_benchmark.cases
    )
    evaluation = run_information_final_evaluation(
        submissions=submissions,
        benchmark=resolved_benchmark,
        policy=resolved_policy,
    )
    report = InformationRuntimeBackedEvaluationReport(
        evaluation_report=evaluation,
        runtime_evidence=evidence,
        passed=evaluation.passed and evidence.passed,
        report_digest="",
    )
    report = replace(report, report_digest=information_runtime_backed_report_digest(report))
    verify_information_runtime_backed_evaluation_report(report)
    _content_free(runtime_backed_report_to_dict(report))
    return report


def runtime_backed_report_to_dict(
    report: InformationRuntimeBackedEvaluationReport,
) -> dict[str, object]:
    verify_information_runtime_backed_evaluation_report(report)
    value = {
        "evaluation_report": report_to_dict(report.evaluation_report),
        "runtime_evidence": {
            **_runtime_evidence_material(report.runtime_evidence),
            "evidence_digest": report.runtime_evidence.evidence_digest,
        },
        "passed": report.passed,
        "report_digest": report.report_digest,
    }
    _content_free(value)
    return value


# Pytest plugin hooks used only inside the isolated runtime-probe subprocess.
def _deny_network(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("P4.8 runtime probe blocked an outbound network operation.")


def pytest_configure(config: object) -> None:
    if os.environ.get(_RUNTIME_ENV_FLAG) != "1":
        return
    socket.socket.connect = _deny_network  # type: ignore[method-assign]
    socket.socket.connect_ex = _deny_network  # type: ignore[method-assign]
    socket.create_connection = _deny_network  # type: ignore[assignment]


def pytest_terminal_summary(
    terminalreporter: object, exitstatus: object, config: object
) -> None:
    if os.environ.get(_RUNTIME_ENV_FLAG) != "1":
        return
    writer = getattr(terminalreporter, "write_line", None)
    if callable(writer):
        writer(NETWORK_GUARD_MARKER)
