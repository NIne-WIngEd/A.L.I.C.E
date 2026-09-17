#!/usr/bin/env python3
"""Execute only a previously preflighted frozen downstream arbitration manifest."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent


class ExecutionFreezeError(RuntimeError):
    pass


def load_sibling(module_name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ExecutionFreezeError(f"unable to load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionFreezeError(f"unable to load JSON {path}: {exc}") from exc


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionFreezeError(f"{label} must be an object")
    return value


def verify_repository_source_freeze(
    *, repo_root: Path, expected_revision: str
) -> dict[str, Any]:
    """Fail closed if any tracked repository content differs from frozen HEAD.

    Untracked run outputs are intentionally allowed. Every tracked source/config
    file, including transitive Python imports not listed in the arbitration
    manifest, must still match the immutable source revision.
    """
    repo_root = repo_root.resolve()
    try:
        head_result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ExecutionFreezeError(f"unable to resolve Git HEAD: {exc}") from exc
    if head_result.returncode != 0:
        raise ExecutionFreezeError(
            f"git rev-parse failed with status {head_result.returncode}"
        )
    observed = head_result.stdout.strip().lower()
    if observed != expected_revision.lower():
        raise ExecutionFreezeError(
            f"repository HEAD drift: expected {expected_revision.lower()}, got {observed}"
        )

    try:
        diff_result = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--quiet", "HEAD", "--"],
            check=False,
        )
    except OSError as exc:
        raise ExecutionFreezeError(
            f"unable to verify tracked worktree freeze: {exc}"
        ) from exc
    if diff_result.returncode == 1:
        raise ExecutionFreezeError(
            "tracked repository content changed after source freeze; refusing execution"
        )
    if diff_result.returncode != 0:
        raise ExecutionFreezeError(
            f"git diff freeze check failed with status {diff_result.returncode}"
        )
    return {
        "repo_root": str(repo_root),
        "source_revision": observed,
        "tracked_worktree_clean": True,
    }


def verify_preflight_freeze(
    *,
    manifest_path: Path,
    preflight_receipt_path: Path,
    observed_source_revision: str,
    arb: Any,
    full_eval: Any,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    preflight_receipt_path = preflight_receipt_path.resolve()
    manifest = require_mapping(load_json(manifest_path), "manifest")
    receipt = require_mapping(load_json(preflight_receipt_path), "preflight_receipt")

    if receipt.get("protocol_version") != arb.PROTOCOL_VERSION:
        raise ExecutionFreezeError("preflight receipt protocol mismatch")
    if receipt.get("status") != "PREFLIGHT_OK":
        raise ExecutionFreezeError("preflight receipt status must be PREFLIGHT_OK")

    manifest_sha = full_eval.sha256_file(manifest_path)
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ExecutionFreezeError(
            "arbitration manifest changed after preflight; refusing execution"
        )
    if manifest.get("source_revision") != receipt.get("source_revision"):
        raise ExecutionFreezeError("manifest/receipt source revision mismatch")
    if observed_source_revision != receipt.get("source_revision"):
        raise ExecutionFreezeError(
            "checked-out source revision changed after preflight; refusing execution"
        )

    try:
        verified = arb.validate_manifest(manifest, manifest_path)
    except arb.ProtocolError as exc:
        raise ExecutionFreezeError(f"frozen manifest no longer validates: {exc}") from exc

    if verified["common_fingerprint"] != receipt.get("common_fingerprint"):
        raise ExecutionFreezeError("common input fingerprint changed after preflight")
    if verified["common_inputs"] != receipt.get("common_inputs"):
        raise ExecutionFreezeError("common input bindings changed after preflight")
    if verified["arms"] != receipt.get("arms"):
        raise ExecutionFreezeError("arm graph bindings changed after preflight")
    if verified["metrics"] != receipt.get("metrics"):
        raise ExecutionFreezeError("metric comparison policy changed after preflight")

    policy_meta = require_mapping(manifest.get("metric_policy"), "metric_policy")
    policy_path = Path(str(policy_meta.get("path", ""))).expanduser().resolve()
    if not policy_path.is_file():
        raise ExecutionFreezeError(f"frozen metric policy file missing: {policy_path}")
    if full_eval.sha256_file(policy_path) != policy_meta.get("sha256"):
        raise ExecutionFreezeError("metric policy changed after preflight")

    return {
        "manifest_sha256": manifest_sha,
        "preflight_receipt_sha256": full_eval.sha256_file(preflight_receipt_path),
        "source_revision": observed_source_revision,
        "common_fingerprint": verified["common_fingerprint"],
    }


def execute(
    *,
    manifest_path: Path,
    preflight_receipt_path: Path,
    output_path: Path,
    observed_source_revision: str | None = None,
    enforce_repository_source_freeze: bool = True,
) -> dict[str, Any]:
    arb = load_sibling(
        "downstream_causal_arbitration_execute_v02",
        "downstream_causal_arbitration_v0_2.py",
    )
    full_eval = load_sibling(
        "downstream_full_stack_graph_evaluator_execute_v01",
        "downstream_full_stack_graph_evaluator_v0_1.py",
    )
    repo_root = Path(__file__).resolve().parents[3]
    revision = observed_source_revision or full_eval.git_revision(repo_root)
    freeze = verify_preflight_freeze(
        manifest_path=manifest_path,
        preflight_receipt_path=preflight_receipt_path,
        observed_source_revision=revision,
        arb=arb,
        full_eval=full_eval,
    )
    if enforce_repository_source_freeze:
        verify_repository_source_freeze(
            repo_root=repo_root,
            expected_revision=freeze["source_revision"],
        )
    if output_path.resolve().exists():
        raise ExecutionFreezeError(f"refusing to overwrite output: {output_path.resolve()}")
    try:
        result = arb.run_protocol(
            manifest_path.resolve(),
            output_path.resolve(),
            preflight_only=False,
        )
    except arb.ProtocolError as exc:
        raise ExecutionFreezeError(f"arbitration execution failed: {exc}") from exc
    if result.get("source_revision") != freeze["source_revision"]:
        raise ExecutionFreezeError("completed receipt source revision drift")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preflight-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        result = execute(
            manifest_path=args.manifest,
            preflight_receipt_path=args.preflight_receipt,
            output_path=args.output,
            enforce_repository_source_freeze=True,
        )
    except ExecutionFreezeError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
