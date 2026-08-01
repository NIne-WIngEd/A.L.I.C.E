#!/usr/bin/env python3
"""Run the private exact-commit P4.10c live information acceptance audit."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from alice_information.live_acceptance import (
    InformationLiveAcceptancePolicy,
    Phase4LiveAcceptanceRuntime,
    build_live_acceptance_record,
    repository_snapshot_sha256,
    run_deterministic_acceptance_tests,
    run_repository_regression_tests,
    validate_repository_release_state,
    write_live_acceptance_record,
)

_P410_TESTS = (
    "tests/phase4/test_information_live_provider_policy.py",
    "tests/phase4/test_information_live_provider_config.py",
    "tests/phase4/test_information_live_provider_contracts.py",
    "tests/phase4/test_information_brave_search.py",
    "tests/phase4/test_information_brave_search_live.py",
    "tests/phase4/test_information_live_fetch_provider.py",
    "tests/phase4/test_information_live_provider_registry.py",
    "tests/phase4/test_information_live_provider_preflight_script.py",
    "tests/phase4/test_information_live_research_policy.py",
    "tests/phase4/test_information_live_claims.py",
    "tests/phase4/test_information_live_research.py",
    "tests/phase4/test_information_live_research_adversarial.py",
    "tests/phase4/test_information_live_research_script.py",
    "tests/phase4/test_information_live_acceptance.py",
    "tests/phase4/test_information_live_acceptance_adversarial.py",
    "tests/phase4/test_information_live_acceptance_inspection.py",
    "tests/phase4/test_information_live_acceptance_script.py",
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("alice_private_p410c_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Private P4.10c runtime factory could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--runtime-factory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--rollback-commit", required=True)
    parser.add_argument("--evaluated-at", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path(args.repository_root).resolve(strict=True)
    vault = Path(args.vault_root).resolve(strict=True)
    factory_path = Path(args.runtime_factory).resolve(strict=True)
    output = Path(args.output).resolve()
    if _inside(vault, repository) or _inside(factory_path, repository) or _inside(output, repository):
        raise RuntimeError("Vault, private runtime, and acceptance record must remain outside Git.")
    validate_repository_release_state(
        repository,
        repository_commit=args.repository_commit,
        rollback_commit=args.rollback_commit,
    )
    policy = InformationLiveAcceptancePolicy.load(
        repository / "policies/information_live_acceptance_release_policy.json"
    )
    policy.validate()
    benchmark = repository / "benchmarks/phase4/information_live_acceptance_v1.json"
    snapshot_before = repository_snapshot_sha256(repository)
    tests = run_deterministic_acceptance_tests(
        repository, target_files=_P410_TESTS
    )
    regression = run_repository_regression_tests(repository)
    module = _load_module(factory_path)
    builder = getattr(module, "build_phase4_live_acceptance_runtime", None)
    if not callable(builder):
        raise RuntimeError(
            "Private runtime must export build_phase4_live_acceptance_runtime."
        )
    runtime = builder(repository_root=repository, evaluated_at=args.evaluated_at)
    if type(runtime) is not Phase4LiveAcceptanceRuntime:
        raise RuntimeError("Private factory returned a substituted P4.10c runtime.")
    runtime.validate()
    # Build executes the one exact live turn and captures the post-run snapshot.
    record = build_live_acceptance_record(
        repository=repository,
        repository_commit=args.repository_commit,
        rollback_commit=args.rollback_commit,
        evaluated_at=args.evaluated_at,
        package_version="0.18.0",
        policy=policy,
        benchmark_path=benchmark,
        runtime=runtime,
        deterministic_test_result=tests,
        repository_regression_result=regression,
        snapshot_before_sha256=snapshot_before,
    )
    final_after = repository_snapshot_sha256(repository)
    if final_after != snapshot_before:
        raise RuntimeError("Repository changed during P4.10c acceptance.")
    written = write_live_acceptance_record(
        record,
        output,
        repository_root=repository,
        private_root=vault,
    )
    print(str(written))
    print(f"approved={str(record.approved).lower()}")
    print(f"release_id={record.release_id}")
    print(f"repository_commit={record.repository_commit}")
    print(
        "deterministic_tests="
        f"{record.deterministic_test_passed}/{record.deterministic_test_collected}"
    )
    print(
        "repository_regression="
        f"{record.repository_regression_passed}/{record.repository_regression_collected}"
    )
    print(f"repository_subtests_passed={record.repository_regression_subtests_passed}")
    live = dict(record.live_research_receipt)
    print(f"live_outcome={live['outcome']}")
    print(f"live_fetch_attempts={live['fetch_attempt_count']}")
    print(f"live_fetches={len(live['fetch_receipt_sha256s'])}")
    print(f"live_fetch_failures={len(live['fetch_failure_sha256s'])}")
    print(f"grounded_sources={len(live['grounded_source_sha256s'])}")
    print(f"p36_precommit_validations={live['pre_commit_validation_count']}")
    print(f"p45b_validation_sha256={live['validation_sha256']}")
    return 0 if record.approved else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"phase4-live-acceptance failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
