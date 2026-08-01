#!/usr/bin/env python3
"""Run one private, foreground-only P4.10b live PUBLIC research turn.

The runtime factory must remain outside the repository. It constructs the exact
provider registry, Phase 3 orchestrator/model stack, policies, command, and
PUBLIC research request. This script writes metadata only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from alice_information.live_research import LiveInformationResearchExecutor


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("alice_private_p410b_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Private P4.10b runtime factory could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--runtime-factory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluated-at", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path(args.repository_root).resolve(strict=True)
    factory_path = Path(args.runtime_factory).resolve(strict=True)
    output = Path(args.output).resolve()
    if _inside(factory_path, repository) or _inside(output, repository):
        raise RuntimeError("Private runtime and receipt output must remain outside Git.")
    module = _load_module(factory_path)
    builder = getattr(module, "build_phase4_live_research_runtime", None)
    if not callable(builder):
        raise RuntimeError(
            "Private runtime must export build_phase4_live_research_runtime."
        )
    runtime = builder(repository_root=repository, evaluated_at=args.evaluated_at)
    validator = getattr(runtime, "validate", None)
    if callable(validator):
        validator()
    executor = getattr(runtime, "executor", None)
    command = getattr(runtime, "command", None)
    request = getattr(runtime, "request", None)
    if executor is None or command is None or request is None:
        raise RuntimeError("Private P4.10b runtime is incomplete.")
    if type(executor) is not LiveInformationResearchExecutor:
        raise RuntimeError("Private P4.10b runtime substituted the live executor.")
    executor.validate_operational_boundary()
    result = executor.run_turn(
        command,
        mode="research",
        availability="available",
        request=request,
        reference_time=getattr(runtime, "reference_time"),
        created_at=getattr(runtime, "created_at"),
        window_start=getattr(runtime, "window_start", None),
        window_end=getattr(runtime, "window_end", None),
        cancellation=getattr(runtime, "cancellation", None),
    )
    result.validate(executor=executor)
    record = {
        "record_type": "phase4_live_research_receipt",
        "evaluated_at": args.evaluated_at,
        "outcome": result.receipt.outcome,
        "live_research": result.receipt.to_metadata_record(),
        "fetch_failures": [
            item.metadata_record() for item in result.evidence.fetch_failures
        ],
        "source_outcomes": [
            item.metadata_record() for item in result.evidence.source_outcomes
        ],
    }
    serialized = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    raw_query = request.query.text
    credential = getattr(
        getattr(executor.registry.search_provider, "configuration", None),
        "credential",
        None,
    )
    secret = ""
    if credential is not None:
        reveal = getattr(credential, "reveal_for_exact_header", None)
        if callable(reveal):
            secret = reveal()
    if raw_query and raw_query in serialized:
        raise RuntimeError("Raw query text entered the metadata-only record.")
    if secret and secret in serialized:
        raise RuntimeError("Provider credential entered the metadata-only record.")
    source_bodies = [
        item.source_document.normalized_text
        for item in result.evidence.fetch_responses
    ]
    if any(body and body in serialized for body in source_bodies):
        raise RuntimeError("Source content entered the metadata-only record.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8", newline="\n")
    print(str(output))
    print(f"outcome={result.receipt.outcome}")
    print(f"receipt_id={result.receipt.receipt_id}")
    print(f"fetch_attempts={result.receipt.fetch_attempt_count}")
    print(f"fetches={len(result.evidence.fetch_responses)}")
    print(f"fetch_rejections={len(result.evidence.fetch_failures)}")
    print(f"grounded_sources={len(result.evidence.grounded_sources)}")
    print(f"p36_precommit_validations={result.receipt.pre_commit_validation_count}")
    print(f"p45b_validation_outcome={result.receipt.citation_validation_outcome}")
    print(f"p45b_validation_sha256={result.receipt.validation_sha256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"phase4-live-research failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
