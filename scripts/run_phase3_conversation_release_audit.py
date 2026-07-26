"""Run the exact-commit Phase 3 release audit in a private workspace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_conversation import __version__ as conversation_package_version
from alice_conversation.final_evaluation import run_conversation_final_evaluation
from alice_conversation.final_evaluation_contract import (
    load_conversation_final_evaluation_benchmark,
    load_conversation_final_evaluation_policy,
)
from alice_conversation.release_evidence import (
    execute_phase3_release_evidence,
    load_phase3_release_evidence_manifest,
)
from alice_conversation.release_audit import (
    Phase3ReleaseMetadata,
    audit_phase3_release,
    load_phase3_release_audit_policy,
    write_phase3_release_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private P3.11 exact-commit release audit.")
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--evidence-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--rollback-commit", required=True)
    parser.add_argument("--evaluation-policy")
    parser.add_argument("--benchmark")
    parser.add_argument("--release-policy")
    return parser


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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
    status = _git(repository, "status", "--porcelain", "--untracked-files=normal").stdout
    if status.strip():
        raise RuntimeError("The repository working tree must be clean for release audit.")
    rollback = _git(repository, "rev-parse", rollback_ref).stdout.strip().lower()
    if rollback == head:
        raise RuntimeError("The rollback commit must differ from the release commit.")
    ancestry = _git(repository, "merge-base", "--is-ancestor", rollback, head, check=False)
    if ancestry.returncode != 0:
        raise RuntimeError("The rollback commit must be an ancestor of the release commit.")
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
    evaluation_policy = load_conversation_final_evaluation_policy(
        None if args.evaluation_policy is None else Path(args.evaluation_policy)
    )
    benchmark = load_conversation_final_evaluation_benchmark(
        None if args.benchmark is None else Path(args.benchmark),
        policy=evaluation_policy,
    )
    evidence_manifest = load_phase3_release_evidence_manifest(
        None if args.evidence_manifest is None else Path(args.evidence_manifest),
        benchmark=benchmark,
    )
    evidence = execute_phase3_release_evidence(
        repository_root=repository,
        repository_commit=head,
        benchmark=benchmark,
        manifest=evidence_manifest,
    )
    report = run_conversation_final_evaluation(
        submissions=evidence.submissions,
        benchmark=benchmark,
        policy=evaluation_policy,
    )
    release_policy = load_phase3_release_audit_policy(
        None if args.release_policy is None else Path(args.release_policy)
    )
    decision = audit_phase3_release(
        report,
        metadata=Phase3ReleaseMetadata(
            repository_commit=head,
            repository_head_commit=head,
            repository_clean=True,
            evaluated_at=args.evaluated_at,
            policy_versions=(evaluation_policy.policy_id, release_policy.policy_id),
            package_version=conversation_package_version,
            evidence_manifest_id=evidence.manifest_id,
            evidence_manifest_digest=evidence.manifest_digest,
            evidence_run_digest=evidence.run_digest,
            evidence_target_count=evidence.target_count,
            evidence_passed_target_count=evidence.passed_target_count,
            known_limitations=(
                "The public benchmark is synthetic and metadata-only.",
                "Real-model conversational quality remains a separate private acceptance concern.",
                "Phase 3 enables no web access, tool calling, external actions, or conversational memory writes.",
            ),
            rollback_commit=rollback,
        ),
        release_policy=release_policy,
    )
    path = write_phase3_release_record(
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
    print("evidence_run_digest=" + decision.evidence_run_digest)
    print("evidence_targets=" + str(decision.evidence_passed_target_count) + "/" + str(decision.evidence_target_count))
    return 0 if decision.approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
