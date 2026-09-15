#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.curriculum import sha256_file
from alice_personality.n0.evidence_graph import EvidenceGraphConfig, EvidenceGraphEncoder
from alice_personality.n0.evidence_graph_objectives import evidence_graph_objective
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter, EvidenceViewAdapterConfig


STRUCTURED_PARENT_STEP = 80


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class GraphCacheDataset(Dataset):
    TENSOR_KEYS = (
        "field_semantic",
        "valid_mask",
        "target_distribution",
        "edge_index",
        "edge_type_ids",
        "edge_confidence",
        "edge_valid_mask",
        "counterfactual_active",
        "query_semantic",
        "summary_semantic",
        "parent_field_states",
        "parent_field_weights",
    )

    def __init__(self, payload: dict[str, Any], indices: list[int]) -> None:
        self.payload = payload
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        out = {key: self.payload[key][index] for key in self.TENSOR_KEYS}
        out["global_index"] = index
        return out


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def adapt_forward(
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return adapter(
        parent_field_states=batch["parent_field_states"],
        valid_mask=batch["valid_mask"],
        query_semantic=batch["query_semantic"],
        parent_field_weights=batch["parent_field_weights"],
    )


def graph_forward(
    graph: EvidenceGraphEncoder,
    adapted: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    corrupted: bool = False,
) -> dict[str, torch.Tensor]:
    edge_valid = batch["edge_valid_mask"]
    if corrupted:
        edge_valid = edge_valid.clone()
        for row in range(edge_valid.size(0)):
            active = torch.nonzero(edge_valid[row], as_tuple=False).flatten()
            if active.numel() and bool(batch["counterfactual_active"][row]):
                edge_valid[row, int(active[0])] = False
    return graph(
        field_states=adapted["field_states"],
        valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"],
        edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"],
        edge_valid_mask=edge_valid,
        query_semantic=batch["query_semantic"],
        base_field_weights=adapted["field_weights"],
    )


def permuted_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fields = batch["valid_mask"].size(1)
    order = torch.arange(fields - 1, -1, -1, device=batch["valid_mask"].device)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(fields, device=order.device)
    out = dict(batch)
    for key in (
        "field_semantic",
        "valid_mask",
        "target_distribution",
        "parent_field_states",
        "parent_field_weights",
    ):
        out[key] = batch[key].index_select(1, order)
    out["edge_index"] = inverse[batch["edge_index"]]
    return out


def evaluate(
    *,
    adapter: EvidenceViewAdapter,
    graph: EvidenceGraphEncoder,
    dataset: GraphCacheDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    adapter.eval()
    graph.eval()

    selection_losses: list[float] = []
    target_masses: list[float] = []
    distribution_l1: list[float] = []
    summary_cosines: list[float] = []
    counterfactual_deltas: list[float] = []
    permutation_cosines: list[float] = []
    family_values: dict[str, list[float]] = defaultdict(list)
    residual_scales: list[float] = []

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            adapted = adapt_forward(adapter, batch)
            output = graph_forward(graph, adapted, batch)
            corrupted = graph_forward(graph, adapted, batch, corrupted=True)
            pbatch = permuted_batch(batch)
            padapted = adapt_forward(adapter, pbatch)
            permuted = graph_forward(graph, padapted, pbatch)

            target = batch["target_distribution"]
            logp = output["field_weights"].clamp_min(1e-8).log()
            per_selection = -(target * logp).sum(dim=-1)
            support = target > 0
            mass = (
                output["field_weights"]
                * support.to(output["field_weights"].dtype)
            ).sum(dim=-1)
            l1 = (output["field_weights"] - target).abs().sum(dim=-1)
            summary = F.cosine_similarity(
                output["pooled_state"], batch["summary_semantic"], dim=-1
            )
            corrupt_summary = F.cosine_similarity(
                corrupted["pooled_state"], batch["summary_semantic"], dim=-1
            )
            delta = summary - corrupt_summary
            perm_cos = F.cosine_similarity(
                output["pooled_state"], permuted["pooled_state"], dim=-1
            )

            global_indices = batch["global_index"].detach().cpu().tolist()
            for local, global_index in enumerate(global_indices):
                family = payload["families"][int(global_index)]
                value = float(mass[local].detach().cpu())
                family_values[family].append(value)
                selection_losses.append(float(per_selection[local].detach().cpu()))
                target_masses.append(value)
                distribution_l1.append(float(l1[local].detach().cpu()))
                summary_cosines.append(float(summary[local].detach().cpu()))
                counterfactual_deltas.append(float(delta[local].detach().cpu()))
                permutation_cosines.append(float(perm_cos[local].detach().cpu()))
            residual_scales.append(float(adapted["residual_scale"].detach().cpu()))

    family_means = {
        family: sum(values) / len(values)
        for family, values in sorted(family_values.items())
    }
    return {
        "examples": len(target_masses),
        "family_target_support_mass": family_means,
        "family_macro_target_support_mass": sum(family_means.values()) / max(len(family_means), 1),
        "family_min_target_support_mass": min(family_means.values()),
        "mean_target_support_mass": sum(target_masses) / max(len(target_masses), 1),
        "mean_selection_cross_entropy": sum(selection_losses) / max(len(selection_losses), 1),
        "mean_distribution_l1": sum(distribution_l1) / max(len(distribution_l1), 1),
        "mean_summary_cosine": sum(summary_cosines) / max(len(summary_cosines), 1),
        "mean_relation_counterfactual_delta": sum(counterfactual_deltas) / max(len(counterfactual_deltas), 1),
        "counterfactual_positive_rate": sum(value > 0.0 for value in counterfactual_deltas) / max(len(counterfactual_deltas), 1),
        "mean_permutation_cosine": sum(permutation_cosines) / max(len(permutation_cosines), 1),
        "adapter_residual_scale": sum(residual_scales) / max(len(residual_scales), 1),
    }


def save_checkpoint(
    *,
    output_root: Path,
    variant_name: str,
    step: int,
    adapter: EvidenceViewAdapter,
    graph: EvidenceGraphEncoder,
    metrics: dict[str, Any],
    curriculum_hash: str,
    graph_cache_hash: str,
    structured_parent_hash: str,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    root = output_root / variant_name / f"step-{step:08d}"
    root.mkdir(parents=True, exist_ok=True)
    adapter_path = root / "evidence_view_adapter.safetensors"
    graph_path = root / "evidence_graph.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in adapter.state_dict().items()},
        str(adapter_path),
    )
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in graph.state_dict().items()},
        str(graph_path),
    )
    receipt = {
        "schema": "alice.eipm.n0.v02-evidence-specialist-checkpoint.v0.1",
        "status": "TRAINED_NOT_RATIFIED",
        "variant": variant_name,
        "step": step,
        "git_revision": git_revision(),
        "curriculum_sha256": curriculum_hash,
        "graph_cache_sha256": graph_cache_hash,
        "structured_parent_sha256": structured_parent_hash,
        "shared_structured_parent_mutated": False,
        "adapter_sha256": sha256_file(adapter_path),
        "graph_sha256": sha256_file(graph_path),
        "adapter_parameter_count": adapter.parameter_report()["total_parameters"],
        "graph_parameter_count": graph.parameter_report()["total_parameters"],
        "specialist_parameter_count": (
            adapter.parameter_report()["total_parameters"]
            + graph.parameter_report()["total_parameters"]
        ),
        "hard_parameter_ceiling": None,
        "metrics": metrics,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    (root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def train_variant(
    *,
    name: str,
    adapter_config: EvidenceViewAdapterConfig,
    graph_config: EvidenceGraphConfig,
    train_dataset: GraphCacheDataset,
    dev_dataset: GraphCacheDataset,
    payload: dict[str, Any],
    output_root: Path,
    curriculum_hash: str,
    graph_cache_hash: str,
    structured_parent_hash: str,
    device: torch.device,
    max_steps: int,
    save_every: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    adapter = EvidenceViewAdapter(adapter_config).to(device)
    graph = EvidenceGraphEncoder(graph_config).to(device)
    parameters = list(adapter.parameters()) + list(graph.parameters())

    initial = evaluate(
        adapter=adapter,
        graph=graph,
        dataset=dev_dataset,
        payload=payload,
        device=device,
        batch_size=eval_batch_size,
    )

    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    iterator = iter(loader)
    receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        adapter.train()
        graph.train()
        optimizer.zero_grad(set_to_none=True)

        adapted = adapt_forward(adapter, batch)
        output = graph_forward(graph, adapted, batch)
        corrupted = graph_forward(graph, adapted, batch, corrupted=True)
        pbatch = permuted_batch(batch)
        padapted = adapt_forward(adapter, pbatch)
        permuted = graph_forward(graph, padapted, pbatch)

        result = evidence_graph_objective(
            field_weights=output["field_weights"],
            target_distribution=batch["target_distribution"],
            valid_mask=batch["valid_mask"],
            pooled_state=output["pooled_state"],
            semantic_target=batch["summary_semantic"],
            corrupted_pooled_state=corrupted["pooled_state"],
            counterfactual_active_mask=batch["counterfactual_active"],
            field_states=output["field_states"],
            field_semantic=batch["field_semantic"],
            permuted_pooled_state=permuted["pooled_state"],
        )
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 20 == 0 or step == 1:
            print(
                json.dumps(
                    {
                        "variant": name,
                        "step": step,
                        "lr": scheduler.get_last_lr()[0],
                        "loss": float(result["loss"].detach().cpu()),
                        "adapter_residual_scale": float(
                            torch.tanh(adapter.residual_scale).detach().cpu()
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if step % save_every == 0 or step == max_steps:
            metrics = evaluate(
                adapter=adapter,
                graph=graph,
                dataset=dev_dataset,
                payload=payload,
                device=device,
                batch_size=eval_batch_size,
            )
            receipt = save_checkpoint(
                output_root=output_root,
                variant_name=name,
                step=step,
                adapter=adapter,
                graph=graph,
                metrics=metrics,
                curriculum_hash=curriculum_hash,
                graph_cache_hash=graph_cache_hash,
                structured_parent_hash=structured_parent_hash,
            )
            receipts[f"step-{step:08d}"] = receipt
            print("checkpoint=" + json.dumps(receipt, sort_keys=True), flush=True)

    return {
        "variant": name,
        "adapter_config": {
            "adapter_size": adapter_config.adapter_size,
            "num_layers": adapter_config.num_layers,
            "num_heads": adapter_config.num_heads,
            "feedforward_size": adapter_config.feedforward_size,
        },
        "graph_config": {
            "graph_size": graph_config.graph_size,
            "graph_layers": graph_config.graph_layers,
        },
        "initial_metrics": initial,
        "checkpoints": receipts,
    }


def capability_tuple(receipt: dict[str, Any]) -> tuple[float, ...]:
    metrics = receipt["metrics"]
    family = metrics["family_target_support_mass"]
    return (
        float(metrics["family_min_target_support_mass"]),
        float(metrics["family_macro_target_support_mass"]),
        float(family.get("supersession_historical", 0.0)),
        float(metrics["mean_summary_cosine"]),
        float(metrics["counterfactual_positive_rate"]),
        float(metrics["mean_relation_counterfactual_delta"]),
        -float(metrics["mean_distribution_l1"]),
        -int(receipt["step"]),
    )


def best_prior_joint(prior: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for name in ("compact_joint", "expanded_joint_512x2"):
        for receipt in prior["variants"][name]["checkpoints"].values():
            candidates.append(receipt)
    return max(
        candidates,
        key=lambda receipt: (
            float(receipt["metrics"]["family_min_target_support_mass"]),
            float(receipt["metrics"]["family_macro_target_support_mass"]),
            float(receipt["metrics"]["mean_summary_cosine"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-cache", required=True)
    parser.add_argument("--prior-comparison", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--save-every", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("evidence specialist pilot requires one CUDA device")
    device = torch.device("cuda")
    seed_everything(args.seed)

    graph_cache_path = Path(args.graph_cache).resolve()
    prior_comparison_path = Path(args.prior_comparison).resolve()
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    structured_config_path = Path(args.structured_config).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-evidence-graph-curriculum.v0.1":
        raise SystemExit("evidence graph curriculum schema mismatch")
    curriculum_hash = sha256_file(curriculum_path)
    if curriculum_hash != str(manifest.get("compiled_sha256")):
        raise SystemExit("evidence graph curriculum hash mismatch")
    if manifest.get("hard_parameter_ceiling") is not None:
        raise SystemExit("unexpected parameter ceiling")

    prior = json.loads(prior_comparison_path.read_text(encoding="utf-8"))
    if prior.get("schema") != "alice.eipm.n0.v02-evidence-graph-pilot-comparison.v0.1":
        raise SystemExit("prior graph comparison schema mismatch")
    if prior.get("semantic_parent") != "targeted-repair-v0.1/step-00000080":
        raise SystemExit("prior comparison semantic lineage mismatch")
    if prior.get("structured_parent") != "structured-state-pilot-v0.1/step-00000080":
        raise SystemExit("prior comparison structured lineage mismatch")

    structured_receipt = json.loads(
        (structured_checkpoint / "receipt.json").read_text(encoding="utf-8")
    )
    if int(structured_receipt.get("step", -1)) != STRUCTURED_PARENT_STEP:
        raise SystemExit("specialist pilot must preserve structured step80 parent")
    structured_parent_hash = sha256_file(
        structured_checkpoint / "structured_state.safetensors"
    )
    structured_config = json.loads(structured_config_path.read_text(encoding="utf-8"))
    expected_hash = str(structured_config["pilot"]["selected_checkpoint_sha256"])
    if structured_parent_hash != expected_hash:
        raise SystemExit("structured step80 hash does not match ratified config")

    rows = read_jsonl(curriculum_path)
    payload = torch.load(graph_cache_path, map_location="cpu")
    if [str(row["id"]) for row in rows] != list(payload["ids"]):
        raise SystemExit("graph cache ids do not match current curriculum")
    if [str(row["family"]) for row in rows] != list(payload["families"]):
        raise SystemExit("graph cache families do not match current curriculum")
    if [str(row["split"]) for row in rows] != list(payload["splits"]):
        raise SystemExit("graph cache split metadata does not match current curriculum")
    graph_cache_hash = sha256_file(graph_cache_path)

    train_indices = [i for i, split in enumerate(payload["splits"]) if split == "train"]
    dev_indices = [i for i, split in enumerate(payload["splits"]) if split == "dev"]
    if len(train_indices) != 180 or len(dev_indices) != 60:
        raise SystemExit("graph cache split counts drifted")
    train_dataset = GraphCacheDataset(payload, train_indices)
    dev_dataset = GraphCacheDataset(payload, dev_indices)

    variants = [
        (
            "specialized_compact_384x2_graph256x1",
            EvidenceViewAdapterConfig(
                semantic_size=640,
                adapter_size=384,
                num_layers=2,
                num_heads=6,
                feedforward_size=1536,
                dropout=0.0,
                max_fields=64,
                base_prior_scale=0.5,
            ),
            EvidenceGraphConfig(
                semantic_size=640,
                graph_size=256,
                graph_layers=1,
                max_fields=64,
                max_edges=256,
            ),
        ),
        (
            "specialized_expanded_640x3_graph512x2",
            EvidenceViewAdapterConfig(
                semantic_size=640,
                adapter_size=640,
                num_layers=3,
                num_heads=10,
                feedforward_size=2560,
                dropout=0.0,
                max_fields=64,
                base_prior_scale=0.5,
            ),
            EvidenceGraphConfig(
                semantic_size=640,
                graph_size=512,
                graph_layers=2,
                max_fields=64,
                max_edges=256,
            ),
        ),
    ]

    results: dict[str, Any] = {}
    for offset, (name, adapter_config, graph_config) in enumerate(variants):
        results[name] = train_variant(
            name=name,
            adapter_config=adapter_config,
            graph_config=graph_config,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            payload=payload,
            output_root=output_root,
            curriculum_hash=curriculum_hash,
            graph_cache_hash=graph_cache_hash,
            structured_parent_hash=structured_parent_hash,
            device=device,
            max_steps=args.max_steps,
            save_every=args.save_every,
            batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            seed=args.seed + offset * 101,
        )

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for variant_name, result in results.items():
        for checkpoint_name, receipt in result["checkpoints"].items():
            candidates.append((variant_name, checkpoint_name, receipt))
    winner_variant, winner_checkpoint, winner_receipt = max(
        candidates, key=lambda item: capability_tuple(item[2])
    )

    prior_best = best_prior_joint(prior)
    prior_metrics = prior_best["metrics"]
    winner_metrics = winner_receipt["metrics"]
    prior_hist = float(
        prior_metrics["family_target_support_mass"].get("supersession_historical", 0.0)
    )
    winner_hist = float(
        winner_metrics["family_target_support_mass"].get("supersession_historical", 0.0)
    )

    compact_best = max(
        results["specialized_compact_384x2_graph256x1"]["checkpoints"].values(),
        key=capability_tuple,
    )
    expanded_best = max(
        results["specialized_expanded_640x3_graph512x2"]["checkpoints"].values(),
        key=capability_tuple,
    )
    capacity_min_gain = (
        float(expanded_best["metrics"]["family_min_target_support_mass"])
        - float(compact_best["metrics"]["family_min_target_support_mass"])
    )
    capacity_macro_gain = (
        float(expanded_best["metrics"]["family_macro_target_support_mass"])
        - float(compact_best["metrics"]["family_macro_target_support_mass"])
    )

    comparison = {
        "schema": "alice.eipm.n0.v02-evidence-specialist-comparison.v0.1",
        "status": "EVIDENCE_READY_NOT_RATIFIED",
        "semantic_parent": "targeted-repair-v0.1/step-00000080",
        "structured_parent": "structured-state-pilot-v0.1/step-00000080",
        "structured_parent_sha256": structured_parent_hash,
        "shared_structured_parent_mutated": False,
        "shared_structured_capability_preserved_by_architecture": True,
        "hard_parameter_ceiling": None,
        "capacity_policy": "capability_and_personality_fidelity_first_efficiency_secondary",
        "prior_joint_best": prior_best,
        "variants": results,
        "provisional_winner": {
            "variant": winner_variant,
            "checkpoint": winner_checkpoint,
            "receipt": winner_receipt,
        },
        "diagnostics": {
            "winner_minus_prior_joint_family_min_target_mass": (
                float(winner_metrics["family_min_target_support_mass"])
                - float(prior_metrics["family_min_target_support_mass"])
            ),
            "winner_minus_prior_joint_family_macro_target_mass": (
                float(winner_metrics["family_macro_target_support_mass"])
                - float(prior_metrics["family_macro_target_support_mass"])
            ),
            "winner_minus_prior_joint_supersession_historical_mass": winner_hist - prior_hist,
            "expanded_minus_compact_family_min_target_mass": capacity_min_gain,
            "expanded_minus_compact_family_macro_target_mass": capacity_macro_gain,
            "additional_capacity_material_signal": (
                capacity_min_gain >= 0.02 or capacity_macro_gain >= 0.01
            ),
            "historical_relation_architecture_fix_material_signal": winner_hist - prior_hist >= 0.05,
            "further_capability_work_required": (
                float(winner_metrics["family_min_target_support_mass"]) < 0.80
                or float(winner_metrics["counterfactual_positive_rate"]) < 0.90
                or float(winner_metrics["mean_summary_cosine"]) < 0.95
            ),
        },
        "selection_policy": "worst-family evidence then macro evidence then historical supersession then summary fidelity then relation dependence; no parameter-count preference before capability",
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": "ratify specialist view if capability is strong; otherwise use diagnostics to scale data objectives or architecture without mutating shared structured parent",
    }
    comparison_path = output_root / "evidence_specialist_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
