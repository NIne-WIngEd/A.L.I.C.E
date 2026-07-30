from __future__ import annotations

import subprocess
from pathlib import Path

from alice_information.final_evaluation_contract import (
    load_information_final_evaluation_benchmark,
    load_information_final_evaluation_policy,
)
from alice_information.final_evaluation_runtime import (
    NETWORK_GUARD_MARKER,
    load_information_final_evaluation_runtime_manifest,
    run_runtime_backed_information_final_evaluation,
)
from alice_information.release_audit import (
    Phase4ReleaseMetadata,
    audit_phase4_release,
    load_phase4_release_audit_policy,
)

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
ROLLBACK = "b" * 40


def evaluation_policy():
    return load_information_final_evaluation_policy(
        ROOT / "policies" / "information_final_evaluation_policy.json"
    )


def benchmark():
    policy = evaluation_policy()
    return load_information_final_evaluation_benchmark(
        ROOT / "benchmarks" / "phase4" / "information_final_evaluation_v1.json",
        policy=policy,
    )


def runtime_manifest():
    policy = evaluation_policy()
    value = benchmark()
    return load_information_final_evaluation_runtime_manifest(
        ROOT
        / "benchmarks"
        / "phase4"
        / "information_final_evaluation_runtime_v1.json",
        benchmark=value,
        policy=policy,
    )


def _runtime_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "runtime-repo"
    repository.mkdir(parents=True)
    for target in runtime_manifest().target_files:
        path = repository / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
    return repository


def _runner(*, passed: bool):
    def run(command, **_kwargs):
        if "--collect-only" in command:
            targets = [
                item
                for item in command
                if isinstance(item, str) and item.startswith("tests/phase4/")
            ]
            nodes = "\n".join(f"{item}::test_probe" for item in targets)
            output = nodes + f"\n640 tests collected\n{NETWORK_GUARD_MARKER}\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        if passed:
            output = f"640 passed\n{NETWORK_GUARD_MARKER}\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        output = f"639 passed, 1 failed\n{NETWORK_GUARD_MARKER}\n"
        return subprocess.CompletedProcess(command, 1, output, "")

    return run


def runtime_report(tmp_path: Path, *, passed: bool = True):
    policy = evaluation_policy()
    value = benchmark()
    manifest = runtime_manifest()
    return run_runtime_backed_information_final_evaluation(
        repository_root=_runtime_repository(tmp_path),
        benchmark=value,
        policy=policy,
        manifest=manifest,
        command_runner=_runner(passed=passed),
        snapshotter=lambda _root: "e" * 64,
    )


def release_policy():
    return load_phase4_release_audit_policy(
        ROOT / "policies" / "information_release_audit_policy.json"
    )


def metadata(**changes):
    policy = release_policy()
    values = {
        "repository_commit": COMMIT,
        "repository_head_commit": COMMIT,
        "repository_clean": True,
        "evaluated_at": "2026-07-30T05:30:00Z",
        "policy_versions": (
            evaluation_policy().policy_id,
            policy.policy_id,
        ),
        "package_version": "0.15.0",
        "known_limitations": ("Synthetic benchmark only.",),
        "rollback_commit": ROLLBACK,
    }
    values.update(changes)
    return Phase4ReleaseMetadata(**values)


def passing_decision(tmp_path: Path):
    return audit_phase4_release(
        runtime_report(tmp_path),
        metadata=metadata(),
        release_policy=release_policy(),
    )
