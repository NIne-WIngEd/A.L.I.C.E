#!/usr/bin/env python3
"""Full-stack N0 graph evaluator for frozen downstream causal arbitration.

This entrypoint preserves the production N0 semantic -> structured -> evidence
adapter -> dual-endpoint evidence graph -> ratified fusion -> adaptive latent
pool path. The evidence-graph checkpoint is the only arm-varying model
artifact. All other model/config inputs are bound by a frozen stack manifest.

Diagnostic only: no training, gradient, mutation, ratification, promotion, or
scaling path exists here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "n0_downstream_full_stack_graph_evaluator_v0.1"
SCHEMA = "alice.eipm.n0.downstream-full-stack-graph-evaluator-result.v0.1"
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_STACK_ARTIFACTS = frozenset(
    {
        "latent_config",
        "semantic_config",
        "semantic_checkpoint",
        "tokenizer_dir",
        "structured_config",
        "structured_checkpoint",
        "evidence_adapter",
        "fusion_checkpoint",
        "fusion_config",
        "fusion_ratification",
    }
)
DIRECTORY_ARTIFACTS = frozenset(
    {
        "semantic_checkpoint",
        "tokenizer_dir",
        "structured_checkpoint",
        "fusion_checkpoint",
    }
)
EXTERNAL_INPUTS = frozenset({"latent_checkpoint", "evaluation_set"})
FORBIDDEN_GRAPH_ARTIFACT_KEYS = frozenset(
    {"graph", "graph_checkpoint", "evidence_graph", "evidence_graph_checkpoint"}
)


class BindingError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory_tree(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_dir():
        raise BindingError(f"directory artifact is not a directory: {path}")
    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise BindingError(f"directory artifact contains symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise BindingError(f"directory artifact contains non-regular entry: {item}")
        relative = item.relative_to(path).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(item),
                "size": item.stat().st_size,
            }
        )
    if not rows:
        raise BindingError(f"directory artifact is empty: {path}")
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest(), rows


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BindingError(f"unable to load JSON {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BindingError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BindingError(f"{label} must be a non-empty string")
    return value


def resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def verify_artifact(
    *,
    role: str,
    spec: Any,
    base: Path,
    expected_type: str,
) -> dict[str, Any]:
    obj = require_mapping(spec, f"artifacts.{role}")
    artifact_type = require_string(
        obj.get("artifact_type"), f"artifacts.{role}.artifact_type"
    )
    if artifact_type != expected_type:
        raise BindingError(
            f"artifacts.{role}.artifact_type must be {expected_type}, got {artifact_type}"
        )
    expected = require_string(obj.get("sha256"), f"artifacts.{role}.sha256").lower()
    if not SHA256_RE.fullmatch(expected):
        raise BindingError(f"artifacts.{role}.sha256 must be lowercase 64-hex")
    path = resolve(base, require_string(obj.get("path"), f"artifacts.{role}.path"))

    if artifact_type == "file":
        if not path.is_file():
            raise BindingError(f"artifacts.{role} is not a regular file: {path}")
        actual = sha256_file(path)
        detail: dict[str, Any] = {
            "path": str(path),
            "artifact_type": "file",
            "sha256": actual,
            "size": path.stat().st_size,
        }
    elif artifact_type == "directory_tree":
        actual, rows = sha256_directory_tree(path)
        detail = {
            "path": str(path),
            "artifact_type": "directory_tree",
            "sha256": actual,
            "files": len(rows),
        }
    else:
        raise BindingError(f"unsupported artifact_type for {role}: {artifact_type}")

    if actual != expected:
        raise BindingError(
            f"artifacts.{role} sha256 mismatch: expected {expected}, got {actual}"
        )
    return detail


def git_revision(repo_root: Path) -> str:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BindingError("unable to resolve immutable Git revision") from exc
    if not SOURCE_REVISION_RE.fullmatch(revision):
        raise BindingError(f"invalid Git revision returned by repository: {revision!r}")
    return revision


def validate_stack_manifest(
    manifest_path: Path,
    *,
    observed_source_revision: str,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = require_mapping(load_json(manifest_path), "manifest")
    if root.get("protocol_version") != PROTOCOL_VERSION:
        raise BindingError(f"protocol_version must be {PROTOCOL_VERSION}")

    revision = require_string(root.get("source_revision"), "source_revision").lower()
    if not SOURCE_REVISION_RE.fullmatch(revision):
        raise BindingError("source_revision must be immutable lowercase 40-hex")
    if observed_source_revision.lower() != revision:
        raise BindingError(
            "checked-out source revision does not match frozen stack manifest"
        )

    invariants = require_mapping(root.get("invariants"), "invariants")
    required_invariants = {
        "diagnostic_only": True,
        "training_enabled": False,
        "ratifies_repair": False,
        "promotion_authorized": False,
        "scale_authorized": False,
        "graph_checkpoint_is_only_arm_variant": True,
        "proxy_graph_transform_used": False,
    }
    for key, expected in required_invariants.items():
        if invariants.get(key) is not expected:
            raise BindingError(f"invariants.{key} must be {str(expected).lower()}")

    external_raw = require_mapping(root.get("external_inputs"), "external_inputs")
    if set(external_raw) != EXTERNAL_INPUTS:
        raise BindingError(
            f"external_inputs must contain exactly {sorted(EXTERNAL_INPUTS)}"
        )
    external: dict[str, dict[str, str]] = {}
    for role in sorted(EXTERNAL_INPUTS):
        obj = require_mapping(external_raw[role], f"external_inputs.{role}")
        expected = require_string(
            obj.get("sha256"), f"external_inputs.{role}.sha256"
        ).lower()
        if not SHA256_RE.fullmatch(expected):
            raise BindingError(
                f"external_inputs.{role}.sha256 must be lowercase 64-hex"
            )
        external[role] = {"sha256": expected}

    artifacts_raw = require_mapping(root.get("artifacts"), "artifacts")
    artifact_keys = set(artifacts_raw)
    forbidden = artifact_keys & FORBIDDEN_GRAPH_ARTIFACT_KEYS
    if forbidden:
        raise BindingError(
            "graph checkpoint must not be frozen as a common stack artifact: "
            f"{sorted(forbidden)}"
        )
    if artifact_keys != REQUIRED_STACK_ARTIFACTS:
        missing = REQUIRED_STACK_ARTIFACTS - artifact_keys
        extra = artifact_keys - REQUIRED_STACK_ARTIFACTS
        raise BindingError(
            "artifacts must contain exactly the full common stack; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    base = manifest_path.parent
    artifacts: dict[str, dict[str, Any]] = {}
    for role in sorted(REQUIRED_STACK_ARTIFACTS):
        expected_type = "directory_tree" if role in DIRECTORY_ARTIFACTS else "file"
        artifacts[role] = verify_artifact(
            role=role,
            spec=artifacts_raw[role],
            base=base,
            expected_type=expected_type,
        )

    binding_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "source_revision": revision,
        "invariants": dict(invariants),
        "external_inputs": external,
        "artifacts": {
            role: {
                "artifact_type": artifacts[role]["artifact_type"],
                "sha256": artifacts[role]["sha256"],
            }
            for role in sorted(artifacts)
        },
    }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_revision": revision,
        "invariants": dict(invariants),
        "external_inputs": external,
        "artifacts": artifacts,
        "stack_binding_fingerprint": hashlib.sha256(
            canonical_json_bytes(binding_payload)
        ).hexdigest(),
    }


def verify_external_file(
    *,
    role: str,
    path: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise BindingError(f"{role} is not a regular file: {path}")
    actual = sha256_file(path)
    expected = str(binding["external_inputs"][role]["sha256"])
    if actual != expected:
        raise BindingError(
            f"{role} sha256 mismatch against stack manifest: "
            f"expected {expected}, got {actual}"
        )
    return {"path": str(path), "sha256": actual, "size": path.stat().st_size}


def run_evaluation(
    *,
    graph_path: Path,
    latent_checkpoint_path: Path,
    stack_manifest_path: Path,
    evaluation_set_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    if output_path.exists():
        raise BindingError(f"refusing to overwrite output: {output_path}")

    graph_path = graph_path.resolve()
    if not graph_path.is_file():
        raise BindingError(f"graph checkpoint is not a regular file: {graph_path}")

    repo_root = Path(__file__).resolve().parents[3]
    observed_revision = git_revision(repo_root)
    binding = validate_stack_manifest(
        stack_manifest_path,
        observed_source_revision=observed_revision,
    )
    latent_binding = verify_external_file(
        role="latent_checkpoint",
        path=latent_checkpoint_path,
        binding=binding,
    )
    evaluation_binding = verify_external_file(
        role="evaluation_set",
        path=evaluation_set_path,
        binding=binding,
    )
    graph_binding = {
        "path": str(graph_path),
        "sha256": sha256_file(graph_path),
        "size": graph_path.stat().st_size,
    }

    # Heavy ML imports intentionally occur only after all immutable binding checks.
    import torch
    from safetensors.torch import load_file

    from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
        AdaptiveMultiViewLatentPoolV02,
    )
    from eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1 import (
        counterfactual_metrics,
        load_public_parents,
    )
    from train_n0_v02_adaptive_multi_view_latent_pool_full_scale import (
        LatentPoolDataset,
        load_ratified_fusion,
        precompute_fusion_cache,
    )
    from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale import (
        load_config as load_latent_config,
    )
    from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_2_full_scale import (
        corrected_evaluate,
    )
    from train_n0_v02_cross_context_fusion_full_scale import (
        build_parent_cache,
        read_jsonl,
    )

    if not torch.cuda.is_available():
        raise BindingError("full-stack downstream arbitration evaluator requires CUDA")
    device = torch.device("cuda")

    artifacts = binding["artifacts"]
    rows = read_jsonl(Path(evaluation_binding["path"]))
    if not rows:
        raise BindingError("evaluation set is empty")

    semantic_model, tokenizer, structured, adapter, graph = load_public_parents(
        semantic_config_path=Path(artifacts["semantic_config"]["path"]),
        semantic_checkpoint=Path(artifacts["semantic_checkpoint"]["path"]),
        tokenizer_dir=Path(artifacts["tokenizer_dir"]["path"]),
        structured_config_path=Path(artifacts["structured_config"]["path"]),
        structured_checkpoint=Path(artifacts["structured_checkpoint"]["path"]),
        evidence_adapter_path=Path(artifacts["evidence_adapter"]["path"]),
        evidence_graph_path=graph_path,
        device=device,
    )
    graph_report = graph.parameter_report()
    parent_cache = build_parent_cache(
        rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured=structured,
        evidence_adapter=adapter,
        evidence_graph=graph,
        device=device,
        encode_batch_size=32,
        raw_max_length=96,
        field_max_length=96,
    )
    del semantic_model, structured, adapter, graph
    torch.cuda.empty_cache()

    fusion = load_ratified_fusion(
        checkpoint_dir=Path(artifacts["fusion_checkpoint"]["path"]),
        fusion_config_path=Path(artifacts["fusion_config"]["path"]),
        ratification_path=Path(artifacts["fusion_ratification"]["path"]),
        device=device,
    )
    fusion_cache = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=parent_cache,
        device=device,
        batch_size=16,
    )
    del fusion
    torch.cuda.empty_cache()

    latent_model = AdaptiveMultiViewLatentPoolV02(
        load_latent_config(Path(artifacts["latent_config"]["path"]))
    )
    latent_model.load_state_dict(
        load_file(str(Path(latent_binding["path"])), device="cpu"),
        strict=True,
    )
    for parameter in latent_model.parameters():
        parameter.requires_grad = False
    latent_model = latent_model.to(device).eval()

    dataset = LatentPoolDataset(parent_cache, fusion_cache, list(range(len(rows))))
    metrics = corrected_evaluate(
        model=latent_model,
        dataset=dataset,
        parent_cache=parent_cache,
        device=device,
        batch_size=16,
    )
    cf = counterfactual_metrics(
        model=latent_model,
        dataset=dataset,
        rows=rows,
        device=device,
        batch_size=16,
    )

    result = {
        "schema": SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "source_revision": binding["source_revision"],
        "stack_manifest": {
            "path": binding["manifest_path"],
            "sha256": binding["manifest_sha256"],
            "stack_binding_fingerprint": binding["stack_binding_fingerprint"],
        },
        "graph_condition": {
            **graph_binding,
            "encoder": "DualEndpointEvidenceGraphEncoder",
            "parameter_report": graph_report,
        },
        "common_external_inputs": {
            "latent_checkpoint": latent_binding,
            "evaluation_set": evaluation_binding,
        },
        "common_stack_artifacts": artifacts,
        "rows_evaluated": len(rows),
        "metrics": metrics,
        "counterfactual_metrics": cf,
        "source_anchor_diagnostics": {
            "summary_max_abs_error": fusion_cache[
                "source_anchor_summary_max_abs_error"
            ],
            "token_max_abs_error": fusion_cache[
                "source_anchor_token_max_abs_error"
            ],
        },
        "invariants": {
            "diagnostic_only": True,
            "training_enabled": False,
            "gradient_performed": False,
            "parents_mutated": False,
            "ratifies_repair": False,
            "promotion_authorized": False,
            "scale_authorized": False,
            "graph_checkpoint_is_only_arm_variant": True,
            "proxy_graph_transform_used": False,
        },
        "n0_complete": False,
    }
    atomic_write_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--latent-checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evaluation-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        result = run_evaluation(
            graph_path=args.graph,
            latent_checkpoint_path=args.latent_checkpoint,
            stack_manifest_path=args.config,
            evaluation_set_path=args.evaluation_set,
            output_path=args.output,
        )
    except BindingError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
