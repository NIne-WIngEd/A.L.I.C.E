"""Run the synthetic Phase 2 Memory Core release audit in a private workspace."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from alice_memory.cited_answer import MemoryAnswerAuthorization
from alice_memory.evaluation_contract import (
    load_memory_evaluation_benchmark,
    load_memory_evaluation_policy,
)
from alice_memory.evaluation_fixtures import build_memory_evaluation_fixture
from alice_memory.final_evaluation import run_memory_core_final_evaluation
from alice_memory.release_audit import (
    Phase2ReleaseMetadata,
    audit_phase2_release,
    write_phase2_release_record,
)
from alice_memory.sensitive_crypto import InMemoryTestKeyProtector
from alice_memory.store import open_memory_store


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--rollback-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    vault = Path(args.vault_root).expanduser().resolve(strict=True)
    repository = Path(args.repository_root).expanduser().resolve(strict=True)
    benchmark = load_memory_evaluation_benchmark()
    policy = load_memory_evaluation_policy()

    with tempfile.TemporaryDirectory(prefix="phase2-release-", dir=vault) as temp:
        workspace = Path(temp)
        with open_memory_store(
            workspace,
            repository_root=repository,
        ) as connection:
            fixture = build_memory_evaluation_fixture(
                connection,
                workspace,
                repository_root=repository,
                benchmark=benchmark,
                key_protector=InMemoryTestKeyProtector(),
            )
            report = run_memory_core_final_evaluation(
                connection,
                fixture=fixture,
                benchmark=benchmark,
                policy=policy,
                authorization=MemoryAnswerAuthorization(
                    actor="phase2-release-audit",
                    purpose="offline synthetic final Memory Core evaluation",
                    allowed=True,
                    max_classification="PRIVATE",
                ),
            )

    decision = audit_phase2_release(
        report,
        metadata=Phase2ReleaseMetadata(
            repository_commit=args.repository_commit,
            evaluated_at=args.evaluated_at,
            policy_versions=(policy.policy_id,),
            known_limitations=(
                "Synthetic benchmark only; no conversational model is included in Phase 2.",
                "Backup-expiry enforcement remains an operational deployment responsibility.",
            ),
            rollback_commit=args.rollback_commit,
        ),
    )
    path = write_phase2_release_record(
        decision,
        args.output,
        private_root=vault,
        repository_root=repository,
    )
    print(path)
    print("approved=" + str(decision.approved).lower())
    print("release_id=" + decision.release_id)
    return 0 if decision.approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
