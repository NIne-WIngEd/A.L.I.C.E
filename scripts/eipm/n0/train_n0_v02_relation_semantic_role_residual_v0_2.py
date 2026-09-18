#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_graph_semantic_role_residual import (
    SemanticRoleResidualEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_evidence_selector_repair_v0_1 as sr
import train_n0_v02_relation_endpoint_repair_v0_2 as endpoint

PREP_SCHEMA = "alice.eipm.n0.v02-relation-semantic-role-residual-preparation.v0.2"
MANIFEST_SCHEMA = "alice.eipm.n0.v02-relation-semantic-role-residual-curriculum.v0.2"
CHECKPOINT_SCHEMA = "alice.eipm.n0.v02-relation-semantic-role-residual-checkpoint.v0.2"
RESULT_SCHEMA = "alice.eipm.n0.v02-relation-semantic-role-residual-result.v0.2"

RESIDUAL_PREFIXES = (
    "semantic_role_query_adapter.",
    "semantic_role_delta_mlp.",
)

SOURCE_FAMILIES = {
    "corrects_current",
    "supersedes_current",
    "temporal_current",
    "causes_cause",
    "supports_supporter",
    "derived_item",
}
TARGET_FAMILIES = {
    "corrects_previous",
    "supersedes_previous",
    "temporal_previous",
    "causes_effect",
    "supports_supported",
    "derived_basis",
}


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def role_summary(metrics: dict[str, Any]) -> dict[str, float]:
    families = metrics.get("family_pair_accuracy", {})

    def mean(names: set[str]) -> float:
        values = [float(families[name]) for name in sorted(names) if name in families]
        return sum(values) / max(len(values), 1)

    return {
        "source_semantic_pair_accuracy": mean(SOURCE_FAMILIES),
        "target_semantic_pair_accuracy": mean(TARGET_FAMILIES),
    }


def is_residual_parameter(name: str) -> bool:
    return name.startswith(RESIDUAL_PREFIXES)


def configure_residual_only_trainable(
    graph: SemanticRoleResidualEvidenceGraphEncoder,
) -> list[torch.nn.Parameter]:
    for parameter in graph.parameters():
        parameter.requires_grad = False
    for name, parameter in graph.named_parameters():
        if is_residual_parameter(name):
            parameter.requires_grad = True

    trainable_names = [
        name for name, parameter in graph.named_parameters() if parameter.requires_grad
    ]
    expected = sorted(graph.semantic_role_parameter_names())
    if sorted(trainable_names) != expected:
        raise SystemExit(
            "semantic-role residual trainable scope drift: "
            f"expected={expected} observed={sorted(trainable_names)}"
        )
    return [parameter for parameter in graph.parameters() if parameter.requires_grad]


def load_parent_into_residual(
    *,
    graph: SemanticRoleResidualEvidenceGraphEncoder,
    parent_state: dict[str, torch.Tensor],
) -> None:
    missing, unexpected = graph.load_state_dict(parent_state, strict=False)
    expected_missing = sorted(graph.semantic_role_parameter_names())
    if sorted(missing) != expected_missing:
        raise SystemExit(
            "semantic-role residual parent-load missing-key drift: "
            f"expected={expected_missing} observed={sorted(missing)}"
        )
    if unexpected:
        raise SystemExit(f"semantic-role residual parent-load unexpected keys: {unexpected}")


def assert_parent_parameters_unchanged(
    *,
    graph: SemanticRoleResidualEvidenceGraphEncoder,
    parent_state: dict[str, torch.Tensor],
) -> None:
    state = graph.state_dict()
    changed: list[str] = []
    for name, parent_tensor in parent_state.items():
        current = state.get(name)
        if current is None:
            changed.append(name + ":missing")
            continue
        if not torch.equal(current.detach().cpu(), parent_tensor.detach().cpu()):
            changed.append(name)
    if changed:
        raise SystemExit(
            "pre-existing endpoint graph parameters changed during residual repair: "
            + repr(changed[:20])
        )


def ordinary_replay_ok(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    macro_tolerance: float,
    min_tolerance: float,
    top1_tolerance: float,
) -> bool:
    return (
        float(current["family_macro_target_support_mass"])
        >= float(baseline["family_macro_target_support_mass"]) - macro_tolerance
        and float(current["family_min_target_support_mass"])
        >= float(baseline["family_min_target_support_mass"]) - min_tolerance
        and float(current["family_macro_top1_support_accuracy"])
        >= float(baseline["family_macro_top1_support_accuracy"]) - top1_tolerance
    )


def endpoint_preservation_ok(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    pair_tolerance: float,
    family_tolerance: float,
    row_tolerance: float,
) -> bool:
    return (
        float(current["pair_accuracy"])
        >= float(baseline["pair_accuracy"]) - pair_tolerance
        and float(current["family_min_pair_accuracy"])
        >= float(baseline["family_min_pair_accuracy"]) - family_tolerance
        and float(current["row_accuracy"])
        >= float(baseline["row_accuracy"]) - row_tolerance
    )


def semantic_dev_ready(
    metrics: dict[str, Any],
    *,
    pair_min: float,
    family_min: float,
    row_min: float,
) -> bool:
    return (
        float(metrics["pair_accuracy"]) >= pair_min
        and float(metrics["family_min_pair_accuracy"]) >= family_min
        and float(metrics["row_accuracy"]) >= row_min
        and float(metrics["mean_graph_target_margin"]) > 0.0
    )


def save_checkpoint(
    *,
    graph: SemanticRoleResidualEvidenceGraphEncoder,
    parent_state: dict[str, torch.Tensor],
    output: Path,
    step: int,
    semantic_dev: dict[str, Any],
    endpoint_dev: dict[str, Any],
    ordinary_replay: dict[str, Any],
    parent_adapter_hash: str,
    parent_graph_hash: str,
    curriculum_hash: str,
    prep_hash: str,
    root: Path,
) -> dict[str, Any]:
    assert_parent_parameters_unchanged(graph=graph, parent_state=parent_state)

    cp = output / f"step-{step:08d}"
    cp.mkdir(parents=True, exist_ok=True)
    weights = cp / "evidence_graph_semantic_role_residual.safetensors"
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in graph.state_dict().items()
        },
        str(weights),
    )

    report = graph.parameter_report()
    trainable_names = [
        name for name, parameter in graph.named_parameters() if parameter.requires_grad
    ]
    receipt = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "TRAINED_NOT_RATIFIED",
        "step": step,
        "git_revision": git_revision(root),
        "graph_sha256": sha(weights),
        "parent_adapter_sha256": parent_adapter_hash,
        "parent_graph_sha256": parent_graph_hash,
        "curriculum_sha256": curriculum_hash,
        "preparation_receipt_sha256": prep_hash,
        "graph_total_parameters": report["total_parameters"],
        "semantic_role_residual_parameters": report["semantic_role_residual_parameters"],
        "trainable_semantic_role_parameters": sum(
            parameter.numel()
            for parameter in graph.parameters()
            if parameter.requires_grad
        ),
        "trainable_parameter_names": trainable_names,
        "trainable_scope": "new_zero_initialized_relation_semantic_role_residual_only",
        "trainable_scope_is_permanent_architecture_limit": False,
        "parent_endpoint_graph_parameters_exactly_unchanged": True,
        "semantic_grounding_dev_metrics": semantic_dev,
        "semantic_grounding_dev_role_summary": role_summary(semantic_dev),
        "endpoint_role_preservation_dev_metrics": endpoint_dev,
        "ordinary_replay_metrics": ordinary_replay,
        "message_passing_mutated": False,
        "parent_source_target_relation_read_mutated": False,
        "conflict_pool_mutated": False,
        "evidence_view_adapter_mutated": False,
        "semantic_parent_mutated": False,
        "structured_parent_mutated": False,
        "graph_pooled_cosine_optimized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "hard_parameter_ceiling": None,
        "production_promotion_authorized": False,
    }
    (cp / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--prep-receipt", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--structured-config", required=True)
    p.add_argument("--structured-checkpoint", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--ordinary-replay-cache", required=True)
    p.add_argument("--endpoint-replay-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--pair-batch-size", type=int, default=16)
    p.add_argument("--replay-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=12)
    p.add_argument("--semantic-ce-weight", type=float, default=0.60)
    p.add_argument("--pair-flip-weight", type=float, default=0.20)
    p.add_argument("--ordinary-replay-weight", type=float, default=0.10)
    p.add_argument("--endpoint-replay-weight", type=float, default=0.10)
    p.add_argument("--ordinary-replay-macro-tolerance", type=float, default=0.015)
    p.add_argument("--ordinary-replay-min-tolerance", type=float, default=0.025)
    p.add_argument("--ordinary-replay-top1-tolerance", type=float, default=0.03)
    p.add_argument("--endpoint-dev-pair-tolerance", type=float, default=0.02)
    p.add_argument("--endpoint-dev-family-tolerance", type=float, default=0.05)
    p.add_argument("--endpoint-dev-row-tolerance", type=float, default=0.02)
    p.add_argument("--dev-pair-accuracy", type=float, default=0.85)
    p.add_argument("--dev-family-min-pair-accuracy", type=float, default=0.66)
    p.add_argument("--dev-row-accuracy", type=float, default=0.90)
    p.add_argument("--test-pair-accuracy", type=float, default=0.90)
    p.add_argument("--test-family-min-pair-accuracy", type=float, default=0.80)
    p.add_argument("--test-row-accuracy", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=20260918)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("relation-semantic role-residual repair requires one CUDA device")
    device = torch.device("cuda")
    seed_all(args.seed)

    root = Path(args.repo_root).resolve()
    prep_path = Path(args.prep_receipt).resolve()
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if (
        prep.get("schema") != PREP_SCHEMA
        or prep.get("status")
        != "PASS_RELATION_SEMANTIC_ROLE_RESIDUAL_READY_FOR_GPU"
    ):
        raise SystemExit("relation-semantic role-residual preparation contract mismatch")
    if prep.get("git_revision") != git_revision(root):
        raise SystemExit("relation-semantic role-residual preparation git drift")
    if prep.get("hard_parameter_ceiling") is not None:
        raise SystemExit("hard parameter ceiling drift")
    if prep.get("scale_authorized") is not False:
        raise SystemExit("premature scale authorization")

    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("role-residual curriculum schema mismatch")
    if manifest.get("compiled_sha256") != sha(curriculum_path):
        raise SystemExit("role-residual curriculum hash mismatch")
    for key in (
        "frozen_latent_challenge_rows_used_for_training",
        "missing_evidence_localization_rows_used_for_training",
        "relation_semantic_v0_1_rows_reused",
        "endpoint_repair_v0_2_heldout_rows_reused",
    ):
        if manifest.get(key) is not False:
            raise SystemExit(f"forbidden row reuse: {key}")
    if manifest.get("query_names_endpoint_role_explicitly") is not False:
        raise SystemExit("role-residual curriculum leaked explicit endpoint-role wording")
    if manifest.get("graph_pooled_cosine_is_optimization_target") is not False:
        raise SystemExit("role-residual objective drift")

    hash_inputs = prep.get("artifact_sha256", {})
    check_paths = {
        "curriculum": curriculum_path,
        "manifest": manifest_path,
        "semantic_config": Path(args.semantic_config).resolve(),
        "semantic_checkpoint": Path(args.semantic_checkpoint).resolve()
        / "alice_n0_v02.safetensors",
        "tokenizer": Path(args.tokenizer_dir).resolve() / "tokenizer.json",
        "structured_config": Path(args.structured_config).resolve(),
        "structured_checkpoint": Path(args.structured_checkpoint).resolve()
        / "structured_state.safetensors",
        "parent_adapter": Path(args.parent_adapter).resolve(),
        "parent_graph": Path(args.parent_graph).resolve(),
        "ordinary_replay_cache": Path(args.ordinary_replay_cache).resolve(),
        "endpoint_replay_cache": Path(args.endpoint_replay_cache).resolve(),
        "trainer": Path(__file__).resolve(),
    }
    for key, path in check_paths.items():
        if not path.is_file():
            raise SystemExit(f"missing role-residual artifact: {key}={path}")
        if hash_inputs.get(key) != sha(path):
            raise SystemExit(f"role-residual artifact drift: {key}")

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite role-residual output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    semantic_cfg = load_n0_config(Path(args.semantic_config).resolve())
    semantic = AliceN0V02Model(semantic_cfg)
    semantic_state = load_file(
        str(
            Path(args.semantic_checkpoint).resolve()
            / "alice_n0_v02.safetensors"
        ),
        device="cpu",
    )
    missing, unexpected = semantic.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    structured_cfg = rr.load_structured_config(Path(args.structured_config).resolve())
    structured = StructuredStateEncoder(structured_cfg)
    structured.load_state_dict(
        load_file(
            str(
                Path(args.structured_checkpoint).resolve()
                / "structured_state.safetensors"
            ),
            device="cpu",
        ),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    rows = read_jsonl(curriculum_path)
    print("role_residual_semantic_cache_start=true", flush=True)
    payload = rr.build_repair_cache(
        rows,
        semantic_model=semantic,
        tokenizer=tokenizer,
        structured_parent=structured,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    semantic_cache = output / "role_residual_semantic_cache.pt"
    torch.save(payload, semantic_cache)
    print(
        f"role_residual_semantic_cache_complete=true path={semantic_cache}",
        flush=True,
    )
    del semantic, semantic_state, structured
    torch.cuda.empty_cache()

    ordinary_payload = torch.load(
        Path(args.ordinary_replay_cache).resolve(), map_location="cpu"
    )
    endpoint_payload = torch.load(
        Path(args.endpoint_replay_cache).resolve(), map_location="cpu"
    )

    parent_adapter_path = Path(args.parent_adapter).resolve()
    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(
        load_file(str(parent_adapter_path), device="cpu"), strict=True
    )
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()

    parent_graph_path = Path(args.parent_graph).resolve()
    parent_graph_state = load_file(str(parent_graph_path), device="cpu")

    graph = SemanticRoleResidualEvidenceGraphEncoder(
        rr.expanded_graph_config()
    ).to(device)
    load_parent_into_residual(graph=graph, parent_state=parent_graph_state)

    baseline_graph = DualEndpointEvidenceGraphEncoder(
        rr.expanded_graph_config()
    ).to(device)
    baseline_graph.load_state_dict(parent_graph_state, strict=True)
    for parameter in baseline_graph.parameters():
        parameter.requires_grad = False
    baseline_graph.eval()

    trainable = configure_residual_only_trainable(graph)
    if not trainable:
        raise SystemExit("role-residual repair has no trainable parameters")

    semantic_train_pairs = rr.PairDataset(
        payload, rr.pair_indices(payload, "train")
    )
    semantic_dev_pairs = rr.PairDataset(
        payload, rr.pair_indices(payload, "dev")
    )
    semantic_test_pairs = rr.PairDataset(
        payload, rr.pair_indices(payload, "test")
    )

    ordinary_train_indices = [
        i for i, split in enumerate(ordinary_payload["splits"]) if split == "train"
    ]
    ordinary_dev_indices = [
        i for i, split in enumerate(ordinary_payload["splits"]) if split == "dev"
    ]
    ordinary_train = rr.RowDataset(ordinary_payload, ordinary_train_indices)
    ordinary_dev = rr.RowDataset(ordinary_payload, ordinary_dev_indices)

    endpoint_train_indices = [
        i for i, split in enumerate(endpoint_payload["splits"]) if split == "train"
    ]
    endpoint_train = rr.RowDataset(endpoint_payload, endpoint_train_indices)
    endpoint_dev_pairs = rr.PairDataset(
        endpoint_payload, rr.pair_indices(endpoint_payload, "dev")
    )

    initial_semantic_dev = sr.evaluate_pairs(
        graph,
        adapter,
        semantic_dev_pairs,
        payload,
        device,
        args.eval_batch_size,
    )
    baseline_endpoint_dev = sr.evaluate_pairs(
        baseline_graph,
        adapter,
        endpoint_dev_pairs,
        endpoint_payload,
        device,
        args.eval_batch_size,
    )
    initial_endpoint_dev = sr.evaluate_pairs(
        graph,
        adapter,
        endpoint_dev_pairs,
        endpoint_payload,
        device,
        args.eval_batch_size,
    )
    baseline_ordinary = sr.evaluate_replay(
        baseline_graph,
        adapter,
        ordinary_dev,
        ordinary_payload,
        device,
        args.eval_batch_size,
    )
    initial_ordinary = sr.evaluate_replay(
        graph,
        adapter,
        ordinary_dev,
        ordinary_payload,
        device,
        args.eval_batch_size,
    )

    if initial_endpoint_dev != baseline_endpoint_dev:
        raise SystemExit("zero-init residual failed exact endpoint metric parity")
    if initial_ordinary != baseline_ordinary:
        raise SystemExit("zero-init residual failed exact ordinary replay metric parity")
    assert_parent_parameters_unchanged(
        graph=graph, parent_state=parent_graph_state
    )

    print(
        "initial_role_residual_semantic_dev="
        + json.dumps(initial_semantic_dev, sort_keys=True),
        flush=True,
    )
    print(
        "zero_init_endpoint_parity="
        + json.dumps(initial_endpoint_dev, sort_keys=True),
        flush=True,
    )
    print(
        "zero_init_ordinary_parity="
        + json.dumps(initial_ordinary, sort_keys=True),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-6)
        progress = (step - args.warmup_steps) / max(
            args.max_steps - args.warmup_steps, 1
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    semantic_loader = DataLoader(
        semantic_train_pairs,
        batch_size=args.pair_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    ordinary_loader = DataLoader(
        ordinary_train,
        batch_size=args.replay_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=0,
    )
    endpoint_loader = DataLoader(
        endpoint_train,
        batch_size=args.replay_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
        num_workers=0,
    )

    semantic_iter = iter(semantic_loader)
    ordinary_iter = iter(ordinary_loader)
    endpoint_iter = iter(endpoint_loader)
    receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, args.max_steps + 1):
        try:
            pair_batch = next(semantic_iter)
        except StopIteration:
            semantic_iter = iter(semantic_loader)
            pair_batch = next(semantic_iter)
        try:
            ordinary_batch = next(ordinary_iter)
        except StopIteration:
            ordinary_iter = iter(ordinary_loader)
            ordinary_batch = next(ordinary_iter)
        try:
            endpoint_batch = next(endpoint_iter)
        except StopIteration:
            endpoint_iter = iter(endpoint_loader)
            endpoint_batch = next(endpoint_iter)

        pair_batch = rr.to_device(pair_batch, device)
        ordinary_batch = rr.to_device(ordinary_batch, device)
        endpoint_batch = rr.to_device(endpoint_batch, device)
        a = rr.unprefix(pair_batch, "a")
        b = rr.unprefix(pair_batch, "b")

        graph.train()
        adapter.eval()
        optimizer.zero_grad(set_to_none=True)

        a_out = rr.graph_forward(graph, adapter, a)
        b_out = rr.graph_forward(graph, adapter, b)
        semantic_ce = 0.5 * (
            endpoint.distribution_ce(
                a_out["field_weights"], a["target_distribution"]
            )
            + endpoint.distribution_ce(
                b_out["field_weights"], b["target_distribution"]
            )
        )
        pair_flip = rr.pair_flip_loss(
            a_out, b_out, a, b, margin=0.25
        )

        current_ordinary = rr.graph_forward(
            graph, adapter, ordinary_batch
        )
        current_endpoint = rr.graph_forward(
            graph, adapter, endpoint_batch
        )
        with torch.inference_mode():
            baseline_ordinary_batch = rr.graph_forward(
                baseline_graph, adapter, ordinary_batch
            )
            baseline_endpoint_batch = rr.graph_forward(
                baseline_graph, adapter, endpoint_batch
            )

        ordinary_kl = endpoint.kl_to_baseline(
            current_ordinary["field_weights"],
            baseline_ordinary_batch["field_weights"],
            ordinary_batch["valid_mask"],
        )
        endpoint_kl = endpoint.kl_to_baseline(
            current_endpoint["field_weights"],
            baseline_endpoint_batch["field_weights"],
            endpoint_batch["valid_mask"],
        )

        total = (
            args.semantic_ce_weight * semantic_ce
            + args.pair_flip_weight * pair_flip
            + args.ordinary_replay_weight * ordinary_kl
            + args.endpoint_replay_weight * endpoint_kl
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        assert_parent_parameters_unchanged(
            graph=graph, parent_state=parent_graph_state
        )

        if step == 1 or step % 20 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "lr": scheduler.get_last_lr()[0],
                        "loss": float(total.detach().cpu()),
                        "semantic_ce": float(semantic_ce.detach().cpu()),
                        "pair_flip_loss": float(pair_flip.detach().cpu()),
                        "ordinary_replay_kl": float(
                            ordinary_kl.detach().cpu()
                        ),
                        "endpoint_replay_kl": float(
                            endpoint_kl.detach().cpu()
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if step % args.save_every == 0 or step == args.max_steps:
            graph.eval()
            semantic_dev = sr.evaluate_pairs(
                graph,
                adapter,
                semantic_dev_pairs,
                payload,
                device,
                args.eval_batch_size,
            )
            endpoint_dev = sr.evaluate_pairs(
                graph,
                adapter,
                endpoint_dev_pairs,
                endpoint_payload,
                device,
                args.eval_batch_size,
            )
            ordinary_replay = sr.evaluate_replay(
                graph,
                adapter,
                ordinary_dev,
                ordinary_payload,
                device,
                args.eval_batch_size,
            )
            receipt = save_checkpoint(
                graph=graph,
                parent_state=parent_graph_state,
                output=output,
                step=step,
                semantic_dev=semantic_dev,
                endpoint_dev=endpoint_dev,
                ordinary_replay=ordinary_replay,
                parent_adapter_hash=sha(parent_adapter_path),
                parent_graph_hash=sha(parent_graph_path),
                curriculum_hash=sha(curriculum_path),
                prep_hash=sha(prep_path),
                root=root,
            )
            receipts[f"step-{step:08d}"] = receipt
            print(
                "role_residual_checkpoint="
                + json.dumps(receipt, sort_keys=True),
                flush=True,
            )

    def preservation_ok(receipt: dict[str, Any]) -> bool:
        return (
            ordinary_replay_ok(
                receipt["ordinary_replay_metrics"],
                baseline_ordinary,
                macro_tolerance=args.ordinary_replay_macro_tolerance,
                min_tolerance=args.ordinary_replay_min_tolerance,
                top1_tolerance=args.ordinary_replay_top1_tolerance,
            )
            and endpoint_preservation_ok(
                receipt["endpoint_role_preservation_dev_metrics"],
                baseline_endpoint_dev,
                pair_tolerance=args.endpoint_dev_pair_tolerance,
                family_tolerance=args.endpoint_dev_family_tolerance,
                row_tolerance=args.endpoint_dev_row_tolerance,
            )
        )

    def dev_ready(receipt: dict[str, Any]) -> bool:
        return semantic_dev_ready(
            receipt["semantic_grounding_dev_metrics"],
            pair_min=args.dev_pair_accuracy,
            family_min=args.dev_family_min_pair_accuracy,
            row_min=args.dev_row_accuracy,
        )

    preservation_candidates = {
        key: receipt
        for key, receipt in receipts.items()
        if preservation_ok(receipt)
    }
    eligible = {
        key: receipt
        for key, receipt in preservation_candidates.items()
        if dev_ready(receipt)
    }

    def score(receipt: dict[str, Any]) -> tuple[float, ...]:
        dev = receipt["semantic_grounding_dev_metrics"]
        roles = receipt["semantic_grounding_dev_role_summary"]
        return (
            float(dev["family_min_pair_accuracy"]),
            min(
                float(roles["source_semantic_pair_accuracy"]),
                float(roles["target_semantic_pair_accuracy"]),
            ),
            float(dev["pair_accuracy"]),
            float(dev["row_accuracy"]),
            float(dev["mean_graph_target_margin"]),
            -int(receipt["step"]),
        )

    if not eligible:
        best_key = max(
            preservation_candidates or receipts,
            key=lambda key: score((preservation_candidates or receipts)[key]),
        )
        best = (preservation_candidates or receipts)[best_key]
        result = {
            "schema": RESULT_SCHEMA,
            "status": (
                "FAIL_RELATION_SEMANTIC_ROLE_RESIDUAL_DEV_STOP_NO_HELDOUT_EXPOSURE"
            ),
            "git_revision": git_revision(root),
            "parent_adapter_sha256": sha(parent_adapter_path),
            "parent_graph_sha256": sha(parent_graph_path),
            "preparation_receipt_sha256": sha(prep_path),
            "curriculum_sha256": sha(curriculum_path),
            "initial_semantic_dev": initial_semantic_dev,
            "baseline_endpoint_role_dev": baseline_endpoint_dev,
            "baseline_ordinary_replay": baseline_ordinary,
            "checkpoints": receipts,
            "preservation_candidate_keys": sorted(preservation_candidates),
            "eligible_candidate_keys": [],
            "best_dev_checkpoint": best_key,
            "best_dev_checkpoint_receipt": best,
            "semantic_heldout_test_evaluated": False,
            "semantic_heldout_rows_opened_after_training": False,
            "selection_policy": (
                "preserve_parent_behavior_then_require_semantic_dev_readiness_"
                "before_any_heldout_exposure"
            ),
            "trainable_scope": (
                "new_zero_initialized_relation_semantic_role_residual_only"
            ),
            "parent_endpoint_graph_parameters_exactly_unchanged": True,
            "relation_semantic_v0_1_failed_checkpoint_used_as_parent": False,
            "relation_semantic_v0_1_rows_reused": False,
            "frozen_latent_challenge_rows_used_for_training": False,
            "missing_evidence_localization_rows_used_for_training": False,
            "endpoint_repair_v0_2_heldout_test_reused": False,
            "private_identity_data": False,
            "private_identity_gradient": False,
            "hard_parameter_ceiling": None,
            "scale_authorized": False,
            "automatic_rerun_or_hotfix_authorized": False,
            "production_promotion_authorized": False,
            "n0_complete": False,
            "next_action": (
                "localize_dev_failure_without_opening_v0_2_heldout_or_automatic_hotfix"
            ),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    winner_key = max(eligible, key=lambda key: score(eligible[key]))
    winner = eligible[winner_key]
    graph.load_state_dict(
        load_file(
            str(
                output
                / winner_key
                / "evidence_graph_semantic_role_residual.safetensors"
            ),
            device="cpu",
        ),
        strict=True,
    )
    graph.to(device).eval()
    assert_parent_parameters_unchanged(
        graph=graph, parent_state=parent_graph_state
    )

    semantic_test = sr.evaluate_pairs(
        graph,
        adapter,
        semantic_test_pairs,
        payload,
        device,
        args.eval_batch_size,
    )
    endpoint_dev_final = sr.evaluate_pairs(
        graph,
        adapter,
        endpoint_dev_pairs,
        endpoint_payload,
        device,
        args.eval_batch_size,
    )
    ordinary_final = sr.evaluate_replay(
        graph,
        adapter,
        ordinary_dev,
        ordinary_payload,
        device,
        args.eval_batch_size,
    )

    preserved = (
        ordinary_replay_ok(
            ordinary_final,
            baseline_ordinary,
            macro_tolerance=args.ordinary_replay_macro_tolerance,
            min_tolerance=args.ordinary_replay_min_tolerance,
            top1_tolerance=args.ordinary_replay_top1_tolerance,
        )
        and endpoint_preservation_ok(
            endpoint_dev_final,
            baseline_endpoint_dev,
            pair_tolerance=args.endpoint_dev_pair_tolerance,
            family_tolerance=args.endpoint_dev_family_tolerance,
            row_tolerance=args.endpoint_dev_row_tolerance,
        )
    )
    passed = (
        preserved
        and float(semantic_test["pair_accuracy"]) >= args.test_pair_accuracy
        and float(semantic_test["family_min_pair_accuracy"])
        >= args.test_family_min_pair_accuracy
        and float(semantic_test["row_accuracy"]) >= args.test_row_accuracy
        and float(semantic_test["mean_graph_target_margin"]) > 0.0
    )

    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_RELATION_SEMANTIC_ROLE_RESIDUAL_HELDOUT_READY_FOR_ONE_FRESH_DOWNSTREAM_CAUSAL_CHECK"
            if passed
            else "FAIL_RELATION_SEMANTIC_ROLE_RESIDUAL_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX"
        ),
        "git_revision": git_revision(root),
        "parent_adapter_sha256": sha(parent_adapter_path),
        "parent_graph_sha256": sha(parent_graph_path),
        "preparation_receipt_sha256": sha(prep_path),
        "curriculum_sha256": sha(curriculum_path),
        "initial_semantic_dev": initial_semantic_dev,
        "baseline_endpoint_role_dev": baseline_endpoint_dev,
        "baseline_ordinary_replay": baseline_ordinary,
        "checkpoints": receipts,
        "preservation_candidate_keys": sorted(preservation_candidates),
        "eligible_candidate_keys": sorted(eligible),
        "selected_checkpoint": winner_key,
        "selected_graph_sha256": winner["graph_sha256"],
        "semantic_heldout_test_evaluated": True,
        "semantic_heldout_rows_opened_after_training": True,
        "semantic_heldout_test_metrics": semantic_test,
        "semantic_heldout_role_summary": role_summary(semantic_test),
        "endpoint_role_dev_preservation_metrics": endpoint_dev_final,
        "ordinary_replay_preservation_metrics": ordinary_final,
        "selection_policy": (
            "preserve_parent_behavior_then_semantic_dev_readiness_then_"
            "worst_family_then_balanced_roles_then_pair_then_row_then_margin_"
            "then_earlier_checkpoint"
        ),
        "heldout_gate": {
            "pass": passed,
            "test_pair_accuracy_min": args.test_pair_accuracy,
            "test_family_min_pair_accuracy_min": args.test_family_min_pair_accuracy,
            "test_row_accuracy_min": args.test_row_accuracy,
            "mean_graph_target_margin_must_be_positive": True,
            "thresholds_are_decision_gates_not_architecture_limits": True,
        },
        "trainable_scope": (
            "new_zero_initialized_relation_semantic_role_residual_only"
        ),
        "parent_endpoint_graph_parameters_exactly_unchanged": True,
        "relation_semantic_v0_1_failed_checkpoint_used_as_parent": False,
        "relation_semantic_v0_1_rows_reused": False,
        "frozen_latent_challenge_rows_used_for_training": False,
        "missing_evidence_localization_rows_used_for_training": False,
        "endpoint_repair_v0_2_heldout_test_reused": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "hard_parameter_ceiling": None,
        "scale_authorized": False,
        "automatic_rerun_or_hotfix_authorized": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "run_one_fresh_downstream_causal_check_on_independent_semantic_analogues"
            if passed
            else "inspect_single_role_residual_heldout_failure_and_choose_one_causal_next_change"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
