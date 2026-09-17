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

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_evidence_selector_repair_v0_1 as sr


PREP_SCHEMA = "alice.eipm.n0.v02-relation-endpoint-repair-preparation.v0.2"
CHECKPOINT_SCHEMA = "alice.eipm.n0.v02-relation-endpoint-repair-checkpoint.v0.2"
RESULT_SCHEMA = "alice.eipm.n0.v02-relation-endpoint-repair-result.v0.2"
MANIFEST_SCHEMA = "alice.eipm.n0.v02-relation-endpoint-repair-curriculum.v0.2"


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_endpoint_read_trainable(graph: DualEndpointEvidenceGraphEncoder) -> list[torch.nn.Parameter]:
    for parameter in graph.parameters():
        parameter.requires_grad = False
    graph.pool_query.requires_grad = True
    modules = [
        graph.source_relation_pool_mlp,
        graph.directed_relation_pool_mlp,
        graph.pool_relation_embedding,
        graph.query_projection,
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    allowed_prefixes = (
        "source_relation_pool_mlp.",
        "directed_relation_pool_mlp.",
        "pool_relation_embedding.",
        "query_projection.",
    )
    trainable_names = [name for name, p in graph.named_parameters() if p.requires_grad]
    unexpected = [
        name for name in trainable_names
        if name != "pool_query" and not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise SystemExit(f"endpoint-read trainable scope drift: {unexpected}")
    required = ["source_relation_pool_mlp.", "directed_relation_pool_mlp.", "pool_relation_embedding.", "query_projection."]
    if not all(any(name.startswith(prefix) for name in trainable_names) for prefix in required):
        raise SystemExit("endpoint-read trainable scope is incomplete")
    if "pool_query" not in trainable_names:
        raise SystemExit("endpoint-read pool_query is not trainable")
    return [p for p in graph.parameters() if p.requires_grad]


def distribution_ce(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(target * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()


def kl_to_baseline(current: torch.Tensor, baseline: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    c = current.clamp_min(1e-8)
    b = baseline.clamp_min(1e-8)
    term = b * (b.log() - c.log())
    term = torch.where(valid, term, torch.zeros_like(term))
    return term.sum(dim=-1).mean()


def role_summary(metrics: dict[str, Any]) -> dict[str, float]:
    families = metrics.get("family_pair_accuracy", {})
    source_names = ["corrects_source", "supersedes_source", "temporal_successor_source", "mixed_correct_support_source"]
    target_names = ["corrects_target", "supersedes_target", "supports_target", "conflict_support_target"]
    def mean(names: list[str]) -> float:
        values = [float(families[name]) for name in names if name in families]
        return sum(values) / max(len(values), 1)
    return {"source_role_pair_accuracy": mean(source_names), "target_role_pair_accuracy": mean(target_names)}


def save_checkpoint(
    *,
    graph: DualEndpointEvidenceGraphEncoder,
    output: Path,
    step: int,
    dev: dict[str, Any],
    replay: dict[str, Any],
    parent_adapter_hash: str,
    parent_graph_hash: str,
    curriculum_hash: str,
    prep_hash: str,
    root: Path,
) -> dict[str, Any]:
    from safetensors.torch import save_file
    cp = output / f"step-{step:08d}"
    cp.mkdir(parents=True, exist_ok=True)
    weights = cp / "evidence_graph_dual_endpoint.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in graph.state_dict().items()}, str(weights))
    trainable_names = [name for name, p in graph.named_parameters() if p.requires_grad]
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
        "graph_total_parameters": graph.parameter_report()["total_parameters"],
        "trainable_relation_read_parameters": sum(p.numel() for p in graph.parameters() if p.requires_grad),
        "trainable_parameter_names": trainable_names,
        "trainable_scope": "query_conditioned_source_target_relation_read_only",
        "trainable_scope_is_permanent_architecture_limit": False,
        "conflict_pool_mutated": False,
        "message_passing_mutated": False,
        "evidence_view_adapter_mutated": False,
        "semantic_parent_mutated": False,
        "structured_parent_mutated": False,
        "graph_pooled_cosine_optimized": False,
        "hard_parameter_ceiling": None,
        "hard_parameter_floor": None,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "dev_endpoint_metrics": dev,
        "dev_role_summary": role_summary(dev),
        "ordinary_replay_metrics": replay,
        "production_promotion_authorized": False,
    }
    (cp / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    p.add_argument("--replay-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--pair-batch-size", type=int, default=16)
    p.add_argument("--replay-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--endpoint-ce-weight", type=float, default=0.70)
    p.add_argument("--pair-flip-weight", type=float, default=0.15)
    p.add_argument("--replay-weight", type=float, default=0.15)
    p.add_argument("--replay-macro-mass-tolerance", type=float, default=0.015)
    p.add_argument("--replay-min-mass-tolerance", type=float, default=0.025)
    p.add_argument("--replay-top1-tolerance", type=float, default=0.03)
    p.add_argument("--test-pair-accuracy", type=float, default=0.90)
    p.add_argument("--test-family-min-pair-accuracy", type=float, default=0.80)
    p.add_argument("--test-row-accuracy", type=float, default=0.95)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("endpoint-role relation repair requires one CUDA device")
    device = torch.device("cuda")
    seed_all(args.seed)

    root = Path(args.repo_root).resolve()
    prep_path = Path(args.prep_receipt).resolve()
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if prep.get("schema") != PREP_SCHEMA or prep.get("status") != "PASS_RELATION_ENDPOINT_REPAIR_READY_FOR_GPU":
        raise SystemExit("endpoint-repair preparation contract mismatch")
    if prep.get("git_revision") != git_revision(root):
        raise SystemExit("endpoint-repair preparation git drift")
    if prep.get("hard_parameter_ceiling") is not None or prep.get("hard_parameter_floor") is not None:
        raise SystemExit("endpoint-repair preparation capacity contract drift")
    if prep.get("scale_authorized") is not False:
        raise SystemExit("endpoint-repair preparation scaling contract drift")

    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("endpoint-repair manifest schema mismatch")
    if manifest.get("compiled_sha256") != sha(curriculum_path):
        raise SystemExit("endpoint-repair curriculum hash mismatch")
    if manifest.get("prior_selector_repair_v0_1_test_rows_reused") is not False:
        raise SystemExit("selector v0.1 held-out leakage")
    if manifest.get("frozen_latent_challenge_rows_used_for_training") is not False:
        raise SystemExit("frozen latent challenge leakage")
    if manifest.get("parent_value_path_diagnostic_rows_used_for_training") is not False:
        raise SystemExit("parent diagnostic leakage")
    if manifest.get("graph_pooled_cosine_is_optimization_target") is not False:
        raise SystemExit("endpoint-repair objective drift")

    hash_inputs = prep.get("artifact_sha256", {})
    check_paths = {
        "curriculum": curriculum_path,
        "manifest": manifest_path,
        "semantic_config": Path(args.semantic_config).resolve(),
        "semantic_checkpoint": Path(args.semantic_checkpoint).resolve() / "alice_n0_v02.safetensors",
        "tokenizer": Path(args.tokenizer_dir).resolve() / "tokenizer.json",
        "structured_config": Path(args.structured_config).resolve(),
        "structured_checkpoint": Path(args.structured_checkpoint).resolve() / "structured_state.safetensors",
        "parent_adapter": Path(args.parent_adapter).resolve(),
        "parent_graph": Path(args.parent_graph).resolve(),
        "replay_cache": Path(args.replay_cache).resolve(),
        "trainer": Path(__file__).resolve(),
    }
    for key, path in check_paths.items():
        if not path.is_file():
            raise SystemExit(f"missing endpoint-repair artifact: {key}={path}")
        if hash_inputs.get(key) != sha(path):
            raise SystemExit(f"endpoint-repair artifact drift: {key}")

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite endpoint-repair output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file

    semantic_cfg = load_n0_config(Path(args.semantic_config).resolve())
    semantic = AliceN0V02Model(semantic_cfg)
    semantic_state = load_file(str(Path(args.semantic_checkpoint).resolve() / "alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = semantic.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    structured_cfg = rr.load_structured_config(Path(args.structured_config).resolve())
    structured = StructuredStateEncoder(structured_cfg)
    structured_state = load_file(str(Path(args.structured_checkpoint).resolve() / "structured_state.safetensors"), device="cpu")
    structured.load_state_dict(structured_state, strict=True)
    for parameter in structured.parameters():
        parameter.requires_grad = False

    rows = read_jsonl(curriculum_path)
    print("endpoint_repair_semantic_cache_start=true", flush=True)
    payload = rr.build_repair_cache(
        rows,
        semantic_model=semantic,
        tokenizer=tokenizer,
        structured_parent=structured,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    cache = output / "endpoint_repair_semantic_cache.pt"
    torch.save(payload, cache)
    print(f"endpoint_repair_semantic_cache_complete=true path={cache}", flush=True)
    del semantic, semantic_state, structured, structured_state
    torch.cuda.empty_cache()

    replay_payload = torch.load(Path(args.replay_cache).resolve(), map_location="cpu")

    parent_adapter_path = Path(args.parent_adapter).resolve()
    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(parent_adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()

    parent_graph_path = Path(args.parent_graph).resolve()
    parent_graph_state = load_file(str(parent_graph_path), device="cpu")
    graph = DualEndpointEvidenceGraphEncoder(rr.expanded_graph_config()).to(device)
    graph.load_state_dict(parent_graph_state, strict=True)
    baseline_graph = DualEndpointEvidenceGraphEncoder(rr.expanded_graph_config()).to(device)
    baseline_graph.load_state_dict(parent_graph_state, strict=True)
    for parameter in baseline_graph.parameters():
        parameter.requires_grad = False
    baseline_graph.eval()
    trainable = set_endpoint_read_trainable(graph)
    if not trainable:
        raise SystemExit("endpoint-repair has no trainable relation-read parameters")

    train_pairs = rr.PairDataset(payload, rr.pair_indices(payload, "train"))
    dev_pairs = rr.PairDataset(payload, rr.pair_indices(payload, "dev"))
    test_pairs = rr.PairDataset(payload, rr.pair_indices(payload, "test"))
    replay_train_indices = [i for i, split in enumerate(replay_payload["splits"]) if split == "train"]
    replay_dev_indices = [i for i, split in enumerate(replay_payload["splits"]) if split == "dev"]
    replay_train = rr.RowDataset(replay_payload, replay_train_indices)
    replay_dev = rr.RowDataset(replay_payload, replay_dev_indices)

    initial_dev = sr.evaluate_pairs(graph, adapter, dev_pairs, payload, device, args.eval_batch_size)
    baseline_replay = sr.evaluate_replay(baseline_graph, adapter, replay_dev, replay_payload, device, args.eval_batch_size)
    print("initial_endpoint_dev=" + json.dumps(initial_dev, sort_keys=True), flush=True)
    print("initial_endpoint_role_summary=" + json.dumps(role_summary(initial_dev), sort_keys=True), flush=True)
    print("baseline_replay=" + json.dumps(baseline_replay, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-6)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pair_loader = DataLoader(
        train_pairs,
        batch_size=args.pair_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    replay_loader = DataLoader(
        replay_train,
        batch_size=args.replay_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=0,
    )
    pair_iter = iter(pair_loader)
    replay_iter = iter(replay_loader)
    receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, args.max_steps + 1):
        try:
            pair_batch = next(pair_iter)
        except StopIteration:
            pair_iter = iter(pair_loader)
            pair_batch = next(pair_iter)
        try:
            replay_batch = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            replay_batch = next(replay_iter)

        pair_batch = rr.to_device(pair_batch, device)
        replay_batch = rr.to_device(replay_batch, device)
        a = rr.unprefix(pair_batch, "a")
        b = rr.unprefix(pair_batch, "b")

        graph.train()
        adapter.eval()
        optimizer.zero_grad(set_to_none=True)
        a_out = rr.graph_forward(graph, adapter, a)
        b_out = rr.graph_forward(graph, adapter, b)
        endpoint_ce = 0.5 * (
            distribution_ce(a_out["field_weights"], a["target_distribution"])
            + distribution_ce(b_out["field_weights"], b["target_distribution"])
        )
        pair_flip = rr.pair_flip_loss(a_out, b_out, a, b, margin=0.25)
        current_replay = rr.graph_forward(graph, adapter, replay_batch)
        with torch.inference_mode():
            baseline_replay_batch = rr.graph_forward(baseline_graph, adapter, replay_batch)
        replay_loss = kl_to_baseline(
            current_replay["field_weights"],
            baseline_replay_batch["field_weights"],
            replay_batch["valid_mask"],
        )
        total = (
            args.endpoint_ce_weight * endpoint_ce
            + args.pair_flip_weight * pair_flip
            + args.replay_weight * replay_loss
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            print(json.dumps({
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "loss": float(total.detach().cpu()),
                "endpoint_ce": float(endpoint_ce.detach().cpu()),
                "pair_flip_loss": float(pair_flip.detach().cpu()),
                "replay_kl_loss": float(replay_loss.detach().cpu()),
            }, sort_keys=True), flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            dev = sr.evaluate_pairs(graph, adapter, dev_pairs, payload, device, args.eval_batch_size)
            replay = sr.evaluate_replay(graph, adapter, replay_dev, replay_payload, device, args.eval_batch_size)
            receipt = save_checkpoint(
                graph=graph,
                output=output,
                step=step,
                dev=dev,
                replay=replay,
                parent_adapter_hash=sha(parent_adapter_path),
                parent_graph_hash=sha(parent_graph_path),
                curriculum_hash=sha(curriculum_path),
                prep_hash=sha(prep_path),
                root=root,
            )
            receipts[f"step-{step:08d}"] = receipt
            print("endpoint_checkpoint=" + json.dumps(receipt, sort_keys=True), flush=True)

    baseline_macro = float(baseline_replay["family_macro_target_support_mass"])
    baseline_min = float(baseline_replay["family_min_target_support_mass"])
    baseline_top1 = float(baseline_replay["family_macro_top1_support_accuracy"])
    def replay_ok(metrics: dict[str, Any]) -> bool:
        return (
            float(metrics["family_macro_target_support_mass"]) >= baseline_macro - args.replay_macro_mass_tolerance
            and float(metrics["family_min_target_support_mass"]) >= baseline_min - args.replay_min_mass_tolerance
            and float(metrics["family_macro_top1_support_accuracy"]) >= baseline_top1 - args.replay_top1_tolerance
        )

    def score(receipt: dict[str, Any]) -> tuple[float, ...]:
        dev = receipt["dev_endpoint_metrics"]
        replay = receipt["ordinary_replay_metrics"]
        roles = receipt["dev_role_summary"]
        return (
            1.0 if replay_ok(replay) else 0.0,
            float(dev["family_min_pair_accuracy"]),
            min(float(roles["source_role_pair_accuracy"]), float(roles["target_role_pair_accuracy"])),
            float(dev["pair_accuracy"]),
            float(dev["row_accuracy"]),
            float(dev["mean_graph_target_margin"]),
            -int(receipt["step"]),
        )

    winner_key = max(receipts, key=lambda key: score(receipts[key]))
    winner = receipts[winner_key]
    selected_graph_state = load_file(str(output / winner_key / "evidence_graph_dual_endpoint.safetensors"), device="cpu")
    graph.load_state_dict(selected_graph_state, strict=True)
    graph.to(device).eval()

    test = sr.evaluate_pairs(graph, adapter, test_pairs, payload, device, args.eval_batch_size)
    replay = sr.evaluate_replay(graph, adapter, replay_dev, replay_payload, device, args.eval_batch_size)
    passed = (
        replay_ok(replay)
        and float(test["pair_accuracy"]) >= args.test_pair_accuracy
        and float(test["family_min_pair_accuracy"]) >= args.test_family_min_pair_accuracy
        and float(test["row_accuracy"]) >= args.test_row_accuracy
        and float(test["mean_graph_target_margin"]) > 0.0
    )

    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_ENDPOINT_ROLE_REPAIR_HELDOUT_READY_FOR_EXISTING_LATENT_FRONTIER_CHALLENGE"
            if passed else
            "FAIL_ENDPOINT_ROLE_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX"
        ),
        "git_revision": git_revision(root),
        "parent_adapter_sha256": sha(parent_adapter_path),
        "parent_graph_sha256": sha(parent_graph_path),
        "preparation_receipt_sha256": sha(prep_path),
        "curriculum_sha256": sha(curriculum_path),
        "initial_dev": initial_dev,
        "initial_role_summary": role_summary(initial_dev),
        "baseline_ordinary_replay": baseline_replay,
        "checkpoints": receipts,
        "selected_checkpoint": winner_key,
        "selected_graph_sha256": winner["graph_sha256"],
        "selection_policy": "replay_floor_then_worst_family_dev_pair_accuracy_then_balanced_endpoint_roles_then_dev_pair_accuracy_then_row_accuracy_then_graph_margin_then_earlier_checkpoint",
        "heldout_test_evaluated_once": True,
        "heldout_test_metrics": test,
        "heldout_role_summary": role_summary(test),
        "selected_ordinary_replay": replay,
        "heldout_gate": {
            "pass": passed,
            "test_pair_accuracy_min": args.test_pair_accuracy,
            "test_family_min_pair_accuracy_min": args.test_family_min_pair_accuracy,
            "test_row_accuracy_min": args.test_row_accuracy,
            "mean_graph_target_margin_must_be_positive": True,
            "replay_macro_mass_tolerance": args.replay_macro_mass_tolerance,
            "replay_min_mass_tolerance": args.replay_min_mass_tolerance,
            "replay_top1_tolerance": args.replay_top1_tolerance,
            "thresholds_are_decision_gates_not_architecture_limits": True,
        },
        "trainable_scope": "query_conditioned_source_target_relation_read_only",
        "trainable_scope_is_permanent_architecture_limit": False,
        "conflict_pool_mutated": False,
        "message_passing_mutated": False,
        "evidence_view_adapter_mutated": False,
        "failed_selector_repair_adapter_used_as_parent": False,
        "semantic_parent_mutated": False,
        "structured_parent_mutated": False,
        "graph_pooled_cosine_optimized": False,
        "prior_selector_repair_v0_1_test_rows_reused": False,
        "frozen_latent_challenge_rows_used_for_training": False,
        "parent_value_path_diagnostic_rows_used_for_training": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "hard_parameter_ceiling": None,
        "hard_parameter_floor": None,
        "scale_authorized": False,
        "scaling_interpretation": {
            "no_scale_from_this_result_alone": True,
            "if_pass": "run the existing latent frontier challenge once with the original evidence adapter and selected repaired graph; only then can a latent bottleneck be localized",
            "if_fail": "inspect the single heldout endpoint-role failure and choose one causal next change; do not automatically broaden, rerun, or scale",
            "if_future_parent_trace_shows_target_signal_reaches_latent_input_but_latent_loses_it": "latent objective or capacity becomes eligible for a bounded scale study around the current operating point with no preset minimum or maximum",
        },
        "automatic_rerun_or_hotfix_authorized": False,
        "next_action": (
            "run_existing_latent_frontier_challenge_once_with_original_adapter_and_selected_repaired_graph"
            if passed else
            "inspect_heldout_endpoint_role_failure_once_and_choose_one_causal_next_change"
        ),
        "production_promotion_authorized": False,
        "n0_complete": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
