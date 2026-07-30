"""Run the exact-commit Phase 4 release audit in a private workspace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_information import __version__ as information_package_version  # noqa: E402
from alice_information.final_evaluation_contract import (  # noqa: E402
    load_information_final_evaluation_benchmark,
    load_information_final_evaluation_policy,
)
from alice_information.final_evaluation_runtime import (  # noqa: E402
    load_information_final_evaluation_runtime_manifest,
    run_runtime_backed_information_final_evaluation,
)
from alice_information.release_audit import (  # noqa: E402
    Phase4ReleaseMetadata,
    audit_phase4_release,
    load_phase4_release_audit_policy,
    write_phase4_release_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the private P4.9 exact-commit information release audit."
    )
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--rollback-commit", required=True)
    parser.add_argument("--evaluation-policy")
    parser.add_argument("--benchmark")
    parser.add_argument("--runtime-manifest")
    parser.add_argument("--release-policy")
    return parser


def _git(
    repository: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        capture_output=True,
        check=check,
    )


def _verify_repository_state(
    repository: Path,
    *,
    expected_commit: str,
    rollback_ref: str,
) -> tuple[str, str]:
    head = _git(repository, "rev-parse", "HEAD").stdout.strip().lower()
    expected = expected_commit.strip().lower()
    if head != expected:
        raise RuntimeError("The supplied repository commit does not match HEAD.")
    status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=normal",
    ).stdout
    if status.strip():
        raise RuntimeError(
            "The repository working tree must be clean for release audit."
        )
    rollback = _git(repository, "rev-parse", rollback_ref).stdout.strip().lower()
    if rollback == head:
        raise RuntimeError("The rollback commit must differ from the release commit.")
    ancestry = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        rollback,
        head,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            "The rollback commit must be an ancestor of the release commit."
        )
    return head, rollback


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    vault = Path(args.vault_root).expanduser().resolve(strict=True)
    repository = Path(args.repository_root).expanduser().resolve(strict=True)
    head, rollback = _verify_repository_state(
        repository,
        expected_commit=args.repository_commit,
        rollback_ref=args.rollback_commit,
    )
    evaluation_policy = load_information_final_evaluation_policy(
        None if args.evaluation_policy is None else Path(args.evaluation_policy)
    )
    benchmark = load_information_final_evaluation_benchmark(
        None if args.benchmark is None else Path(args.benchmark),
        policy=evaluation_policy,
    )
    runtime_manifest = load_information_final_evaluation_runtime_manifest(
        None if args.runtime_manifest is None else Path(args.runtime_manifest),
        benchmark=benchmark,
        policy=evaluation_policy,
    )
    report = run_runtime_backed_information_final_evaluation(
        repository_root=repository,
        benchmark=benchmark,
        policy=evaluation_policy,
        manifest=runtime_manifest,
    )
    release_policy = load_phase4_release_audit_policy(
        None if args.release_policy is None else Path(args.release_policy)
    )
    decision = audit_phase4_release(
        report,
        metadata=Phase4ReleaseMetadata(
            repository_commit=head,
            repository_head_commit=head,
            repository_clean=True,
            evaluated_at=args.evaluated_at,
            policy_versions=(evaluation_policy.policy_id, release_policy.policy_id),
            package_version=information_package_version,
            known_limitations=(
                "The public benchmark and runtime probes are synthetic and metadata-only.",
                "The audit blocks live network access and does not exercise real provider credentials or private query text.",
                "Phase 4 is a read-only compatibility profile with no memory writes, external actions, or background execution.",
            ),
            rollback_commit=rollback,
        ),
        release_policy=release_policy,
    )
    path = write_phase4_release_record(
        decision,
        args.output,
        private_root=vault,
        repository_root=repository,
    )
    print(path)
    print("approved=" + str(decision.approved).lower())
    print("release_id=" + decision.release_id)
    print("repository_commit=" + decision.repository_commit)
    print("evaluation_report_digest=" + decision.evaluation_report_digest)
    print("runtime_evidence_digest=" + decision.runtime_evidence_digest)
    print("runtime_backed_report_digest=" + decision.runtime_backed_report_digest)
    print(
        "runtime_tests="
        + str(decision.runtime_passed_test_count)
        + "/"
        + str(decision.runtime_collected_test_count)
    )
    print(
        "evaluation_cases="
        + str(decision.passed_case_count)
        + "/"
        + str(decision.case_count)
    )
    return 0 if decision.approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
