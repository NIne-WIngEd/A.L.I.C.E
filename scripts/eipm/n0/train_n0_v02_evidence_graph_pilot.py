#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum import sha256_file
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph import EvidenceGraphConfig, EvidenceGraphEncoder
from alice_personality.n0.evidence_graph_objectives import evidence_graph_objective
from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model


EXPECTED_SEMANTIC_PARAMETERS = 136_594_435
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


def load_structured_config(path: Path) -> StructuredStateConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    architecture = raw["architecture"]
    return StructuredStateConfig(
        semantic_size=int(architecture["semantic_size"]),
        state_size=int(architecture["state_size"]),
        num_layers=int(architecture["num_layers"]),
        num_heads=int(architecture["num_heads"]),
        feedforward_size=int(architecture["feedforward_size"]),
        dropout=float(architecture["dropout"]),
        max_fields=int(architecture["max_fields"]),
    )


def encode_texts(
    *,
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            pooled = model.encode(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
        chunks.append(pooled.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def build_graph_cache(
    *,
    rows: list[dict[str, Any]],
    semantic_model: AliceN0V02Model,
    tokenizer: Any,
    structured_parent: StructuredStateEncoder,
    device: torch.device,
    encode_batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    max_fields = max(len(row["fields"]) for row in rows)
    max_edges = max(len(row["relations"]) for row in rows)
    if max_fields < 2 or max_edges < 1:
        raise SystemExit("evidence graph curriculum must contain multi-field relation examples")

    flat_field_texts = [str(field["text"]) for row in rows for field in row["fields"]]
    field_vectors_flat = encode_texts(
        model=semantic_model,
        tokenizer=tokenizer,
        texts=flat_field_texts,
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    query_vectors = encode_texts(
        model=semantic_model,
        tokenizer=tokenizer,
        texts=[str(row["query_text"]) for row in rows],
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    summary_vectors = encode_texts(
        model=semantic_model,
        tokenizer=tokenizer,
        texts=[str(row["target_summary_text"]) for row in rows],
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )

    n = len(rows)
    semantic_size = int(query_vectors.size(-1))
    field_semantic = torch.zeros(n, max_fields, semantic_size)
    field_type_ids = torch.zeros(n, max_fields, dtype=torch.long)
    provenance_ids = torch.zeros(n, max_fields, dtype=torch.long)
    relation_role_ids = torch.zeros(n, max_fields, dtype=torch.long)
    temporal_scope_ids = torch.zeros(n, max_fields, dtype=torch.long)
    confidence = torch.zeros(n, max_fields, 1)
    missing_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    valid_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    target_distribution = torch.zeros(n, max_fields)
    edge_index = torch.zeros(n, max_edges, 2, dtype=torch.long)
    edge_type_ids = torch.zeros(n, max_edges, dtype=torch.long)
    edge_confidence = torch.zeros(n, max_edges, 1)
    edge_valid_mask = torch.zeros(n, max_edges, dtype=torch.bool)
    counterfactual_active = torch.zeros(n, dtype=torch.bool)

    cursor = 0
    for row_index, row in enumerate(rows):
        fields = row["fields"]
        f = len(fields)
        field_semantic[row_index, :f] = field_vectors_flat[cursor : cursor + f]
        cursor += f
        valid_mask[row_index, :f] = True
        target_distribution[row_index, :f] = torch.tensor(
            row["target_evidence_distribution"], dtype=torch.float32
        )
        for j, item in enumerate(fields):
            field_type_ids[row_index, j] = int(item["field_type_id"])
            provenance_ids[row_index, j] = int(item["provenance_id"])
            relation_role_ids[row_index, j] = int(item["relation_role_id"])
            temporal_scope_ids[row_index, j] = int(item["temporal_scope_id"])
            confidence[row_index, j, 0] = float(item["confidence"])
            missing_mask[row_index, j] = bool(item["missing"])
        for edge_number, relation in enumerate(row["relations"]):
            edge_index[row_index, edge_number, 0] = int(relation["source"])
            edge_index[row_index, edge_number, 1] = int(relation["target"])
            edge_type_ids[row_index, edge_number] = int(relation["relation_type_id"])
            edge_confidence[row_index, edge_number, 0] = float(relation["confidence"])
            edge_valid_mask[row_index, edge_number] = True
        counterfactual_active[row_index] = bool(row.get("counterfactual_required", False))

    structured_parent = structured_parent.to(device).eval()
    parent_field_states: list[torch.Tensor] = []
    parent_pooled_states: list[torch.Tensor] = []
    parent_field_weights: list[torch.Tensor] = []
    chunk = 128
    with torch.inference_mode():
        for start in range(0, n, chunk):
            end = start + chunk
            output = structured_parent(
                semantic_values=field_semantic[start:end].to(device),
                field_type_ids=field_type_ids[start:end].to(device),
                provenance_ids=provenance_ids[start:end].to(device),
                relation_role_ids=relation_role_ids[start:end].to(device),
                temporal_scope_ids=temporal_scope_ids[start:end].to(device),
                confidence=confidence[start:end].to(device),
                missing_mask=missing_mask[start:end].to(device),
                valid_mask=valid_mask[start:end].to(device),
            )
            parent_field_states.append(output["field_states"].detach().float().cpu())
            parent_pooled_states.append(output["pooled_state"].detach().float().cpu())
            parent_field_weights.append(output["field_weights"].detach().float().cpu())

    return {
        "field_semantic": field_semantic,
        "field_type_ids": field_type_ids,
        "provenance_ids": provenance_ids,
        "relation_role_ids": relation_role_ids,
        "temporal_scope_ids": temporal_scope_ids,
        "confidence": confidence,
        "missing_mask": missing_mask,
        "valid_mask": valid_mask,
        "target_distribution": target_distribution,
        "edge_index": edge_index,
        "edge_type_ids": edge_type_ids,
        "edge_confidence": edge_confidence,
        "edge_valid_mask": edge_valid_mask,
        "counterfactual_active": counterfactual_active,
        "query_semantic": query_vectors,
        "summary_semantic": summary_vectors,
        "parent_field_states": torch.cat(parent_field_states, dim=0),
        "parent_pooled_state": torch.cat(parent_pooled_states, dim=0),
        "parent_field_weights": torch.cat(parent_field_weights, dim=0),
        "ids": [str(row["id"]) for row in rows],
        "families": [str(row["family"]) for row in rows],
        "splits": [str(row["split"]) for row in rows],
    }


class GraphDataset(Dataset):
    TENSOR_KEYS = (
        "field_semantic",
        "field_type_ids",
        "provenance_ids",
        "relation_role_ids",
        "temporal_scope_ids",
        "confidence",
        "missing_mask",
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
        "parent_pooled_state",
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
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def structured_forward(branch: StructuredStateEncoder, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return branch(
        semantic_values=batch["field_semantic"],
        field_type_ids=batch["field_type_ids"],
        provenance_ids=batch["provenance_ids"],
        relation_role_ids=batch["relation_role_ids"],
        temporal_scope_ids=batch["temporal_scope_ids"],
        confidence=batch["confidence"],
        missing_mask=batch["missing_mask"],
        valid_mask=batch["valid_mask"],
    )


def graph_forward(
    graph: EvidenceGraphEncoder,
    structured: dict[str, torch.Tensor],
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
        field_states=structured["field_states"],
        valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"],
        edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"],
        edge_valid_mask=edge_valid,
        query_semantic=batch["query_semantic"],
        base_field_weights=structured["field_weights"],
    )


def permuted_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fields = batch["valid_mask"].size(1)
    order = torch.arange(fields - 1, -1, -1, device=batch["valid_mask"].device)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(fields, device=order.device)
    out = dict(batch)
    for key in (
        "field_semantic",
        "field_type_ids",
        "provenance_ids",
        "relation_role_ids",
        "temporal_scope_ids",
        "confidence",
        "missing_mask",
        "valid_mask",
        "target_distribution",
        "parent_field_states",
        "parent_field_weights",
    ):
        out[key] = batch[key].index_select(1, order)
    out["edge_index"] = inverse[batch["edge_index"]]
    return out


def anchor_loss(structured: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    field_cos = F.cosine_similarity(structured["field_states"], batch["parent_field_states"], dim=-1)
    field = (1.0 - field_cos)[batch["valid_mask"]].mean()
    pooled = (1.0 - F.cosine_similarity(
        structured["pooled_state"], batch["parent_pooled_state"], dim=-1
    )).mean()
    return 0.5 * (field + pooled)


def evaluate_graph(
    *,
    structured_branch: StructuredStateEncoder,
    graph_branch: EvidenceGraphEncoder,
    dataset: GraphDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    structured_branch.eval()
    graph_branch.eval()
    selection_losses: list[float] = []
    target_masses: list[float] = []
    distribution_l1: list[float] = []
    summary_cosines: list[float] = []
    counterfactual_deltas: list[float] = []
    permutation_cosines: list[float] = []
    family_values: dict[str, list[float]] = defaultdict(list)
    examples = 0

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            structured = structured_forward(structured_branch, batch)
            output = graph_forward(graph_branch, structured, batch)
            corrupted = graph_forward(graph_branch, structured, batch, corrupted=True)
            pbatch = permuted_batch(batch)
            pstructured = structured_forward(structured_branch, pbatch)
            permuted = graph_forward(graph_branch, pstructured, pbatch)

            target = batch["target_distribution"]
            logp = output["field_weights"].clamp_min(1e-8).log()
            per_selection = -(target * logp).sum(dim=-1)
            support = target > 0
            mass = (output["field_weights"] * support.to(output["field_weights"].dtype)).sum(dim=-1)
            l1 = (output["field_weights"] - target).abs().sum(dim=-1)
            summary = F.cosine_similarity(output["pooled_state"], batch["summary_semantic"], dim=-1)
            corrupt_summary = F.cosine_similarity(corrupted["pooled_state"], batch["summary_semantic"], dim=-1)
            delta = summary - corrupt_summary
            perm_cos = F.cosine_similarity(output["pooled_state"], permuted["pooled_state"], dim=-1)

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
                examples += 1

    family_target_mass = {
        family: sum(values) / len(values) for family, values in sorted(family_values.items())
    }
    return {
        "examples": examples,
        "mean_selection_cross_entropy": sum(selection_losses) / max(examples, 1),
        "mean_target_support_mass": sum(target_masses) / max(examples, 1),
        "mean_distribution_l1": sum(distribution_l1) / max(examples, 1),
        "mean_summary_cosine": sum(summary_cosines) / max(examples, 1),
        "mean_relation_counterfactual_delta": sum(counterfactual_deltas) / max(examples, 1),
        "counterfactual_positive_rate": sum(value > 0.0 for value in counterfactual_deltas) / max(examples, 1),
        "mean_permutation_cosine": sum(permutation_cosines) / max(examples, 1),
        "family_target_support_mass": family_target_mass,
        "family_macro_target_support_mass": sum(family_target_mass.values()) / max(len(family_target_mass), 1),
        "family_min_target_support_mass": min(family_target_mass.values()) if family_target_mass else 0.0,
    }


def structured_regression_metrics(
    *,
    branch: StructuredStateEncoder,
    cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    dev = cache["dev"]
    size = int(dev["labels"].numel())
    logits: list[float] = []
    semantic_cos: list[float] = []
    branch.eval()
    with torch.inference_mode():
        for start in range(0, size, batch_size):
            end = start + batch_size
            batch = {
                "field_semantic": dev["field_semantic"][start:end].to(device),
                "field_type_ids": dev["field_type_ids"][start:end].to(device),
                "provenance_ids": dev["provenance_ids"][start:end].to(device),
                "relation_role_ids": dev["relation_role_ids"][start:end].to(device),
                "temporal_scope_ids": dev["temporal_scope_ids"][start:end].to(device),
                "confidence": dev["confidence"][start:end].to(device),
                "missing_mask": dev["missing_mask"][start:end].to(device),
                "valid_mask": dev["valid_mask"][start:end].to(device),
            }
            output = structured_forward(branch, batch)
            rationale = dev["rationale_target"][start:end].to(device)
            semantic = dev["semantic_target"][start:end].to(device)
            logits.extend(
                float(x)
                for x in (5.0 * F.cosine_similarity(output["pooled_state"], rationale, dim=-1)).cpu().tolist()
            )
            semantic_cos.extend(
                float(x)
                for x in F.cosine_similarity(output["pooled_state"], semantic, dim=-1).cpu().tolist()
            )

    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    labels = [int(x) for x in dev["labels"].tolist()]
    for teacher_id, logit, label in zip(dev["source_teacher_ids"], logits, labels):
        grouped[str(teacher_id)].append((float(logit), int(label)))
    top1 = sum(int(max(group, key=lambda item: item[0])[1] == 1) for group in grouped.values()) / max(len(grouped), 1)
    return {
        "grouped_preferred_top1_accuracy": top1,
        "mean_semantic_alignment_cosine": sum(semantic_cos) / max(len(semantic_cos), 1),
    }


def save_variant_checkpoint(
    *,
    output_root: Path,
    variant_name: str,
    step: int,
    structured_branch: StructuredStateEncoder,
    graph_branch: EvidenceGraphEncoder,
    structured_trainable: bool,
    metrics: dict[str, Any],
    regression: dict[str, float],
    curriculum_hash: str,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    root = output_root / variant_name / f"step-{step:08d}"
    root.mkdir(parents=True, exist_ok=True)
    graph_path = root / "evidence_graph.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in graph_branch.state_dict().items()},
        str(graph_path),
    )
    structured_hash = None
    if structured_trainable:
        structured_path = root / "structured_state.safetensors"
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in structured_branch.state_dict().items()},
            str(structured_path),
        )
        structured_hash = sha256_file(structured_path)
    receipt = {
        "schema": "alice.eipm.n0.v02-evidence-graph-pilot-checkpoint.v0.1",
        "status": "TRAINED_NOT_RATIFIED",
        "variant": variant_name,
        "step": step,
        "git_revision": git_revision(),
        "curriculum_sha256": curriculum_hash,
        "graph_sha256": sha256_file(graph_path),
        "structured_state_sha256": structured_hash,
        "structured_trainable": structured_trainable,
        "graph_parameter_count": graph_branch.parameter_report()["total_parameters"],
        "structured_parameter_count": structured_branch.parameter_report()["total_parameters"],
        "metrics": metrics,
        "structured_regression": regression,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def train_variant(
    *,
    name: str,
    graph_config: EvidenceGraphConfig,
    structured_parent_state: dict[str, torch.Tensor],
    structured_config: StructuredStateConfig,
    structured_trainable: bool,
    train_dataset: GraphDataset,
    dev_dataset: GraphDataset,
    payload: dict[str, Any],
    structured_regression_cache: dict[str, Any],
    output_root: Path,
    curriculum_hash: str,
    device: torch.device,
    max_steps: int,
    save_every: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    anchor_weight: float,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    structured = StructuredStateEncoder(structured_config)
    structured.load_state_dict(structured_parent_state, strict=True)
    for parameter in structured.parameters():
        parameter.requires_grad = structured_trainable
    structured.to(device)
    graph = EvidenceGraphEncoder(graph_config).to(device)

    initial = evaluate_graph(
        structured_branch=structured,
        graph_branch=graph,
        dataset=dev_dataset,
        payload=payload,
        device=device,
        batch_size=eval_batch_size,
    )
    initial_regression = structured_regression_metrics(
        branch=structured,
        cache=structured_regression_cache,
        device=device,
        batch_size=eval_batch_size,
    )

    parameters = list(graph.parameters())
    if structured_trainable:
        parameters += [parameter for parameter in structured.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

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
        structured.train(structured_trainable)
        graph.train()
        optimizer.zero_grad(set_to_none=True)
        structured_output = structured_forward(structured, batch)
        output = graph_forward(graph, structured_output, batch)
        corrupted = graph_forward(graph, structured_output, batch, corrupted=True)
        pbatch = permuted_batch(batch)
        pstructured = structured_forward(structured, pbatch)
        permuted = graph_forward(graph, pstructured, pbatch)
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
        total = result["loss"]
        anchor = total.new_zeros(())
        if structured_trainable and anchor_weight > 0.0:
            anchor = anchor_loss(structured_output, batch)
            total = total + anchor_weight * anchor
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 20 == 0 or step == 1:
            print(json.dumps({
                "variant": name,
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "loss": float(total.detach().cpu()),
                "anchor": float(anchor.detach().cpu()),
            }, sort_keys=True), flush=True)

        if step % save_every == 0 or step == max_steps:
            metrics = evaluate_graph(
                structured_branch=structured,
                graph_branch=graph,
                dataset=dev_dataset,
                payload=payload,
                device=device,
                batch_size=eval_batch_size,
            )
            regression = structured_regression_metrics(
                branch=structured,
                cache=structured_regression_cache,
                device=device,
                batch_size=eval_batch_size,
            )
            receipt = save_variant_checkpoint(
                output_root=output_root,
                variant_name=name,
                step=step,
                structured_branch=structured,
                graph_branch=graph,
                structured_trainable=structured_trainable,
                metrics=metrics,
                regression=regression,
                curriculum_hash=curriculum_hash,
            )
            receipts[f"step-{step:08d}"] = receipt
            print("checkpoint=" + json.dumps(receipt, sort_keys=True), flush=True)

    return {
        "variant": name,
        "structured_trainable": structured_trainable,
        "graph_config": {
            "graph_size": graph_config.graph_size,
            "graph_layers": graph_config.graph_layers,
        },
        "initial_metrics": initial,
        "initial_structured_regression": initial_regression,
        "checkpoints": receipts,
    }


def checkpoint_capability_tuple(receipt: dict[str, Any], parent_structured_top1: float) -> tuple[float, ...]:
    metrics = receipt["metrics"]
    regression = receipt["structured_regression"]
    regression_ok = float(regression["grouped_preferred_top1_accuracy"]) >= parent_structured_top1 - 0.02
    return (
        1.0 if regression_ok else 0.0,
        float(metrics["family_min_target_support_mass"]),
        float(metrics["family_macro_target_support_mass"]),
        float(metrics["mean_summary_cosine"]),
        float(metrics["counterfactual_positive_rate"]),
        -float(metrics["mean_distribution_l1"]),
        -int(receipt["step"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--structured-regression-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--save-every", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--joint-anchor-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("evidence graph pilot requires one CUDA device")
    device = torch.device("cuda")
    seed_everything(args.seed)

    config_path = Path(args.config).resolve()
    structured_config_path = Path(args.structured_config).resolve()
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    regression_cache_path = Path(args.structured_regression_cache).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-evidence-graph-curriculum.v0.1":
        raise SystemExit("evidence graph curriculum schema mismatch")
    if manifest.get("status") != "COMPILED_NOT_ACTIVATED":
        raise SystemExit("evidence graph curriculum has unexpected status")
    if manifest.get("hard_parameter_ceiling") is not None:
        raise SystemExit("evidence graph curriculum unexpectedly imposes a parameter ceiling")
    if manifest.get("private_identity_content") is not False:
        raise SystemExit("evidence graph curriculum violates the public identity-neutral boundary")
    if sha256_file(curriculum_path) != str(manifest.get("compiled_sha256")):
        raise SystemExit("evidence graph curriculum hash mismatch")

    from safetensors.torch import load_file

    structured_config = load_structured_config(structured_config_path)
    structured_parent = StructuredStateEncoder(structured_config)
    structured_state_path = structured_checkpoint / "structured_state.safetensors"
    structured_receipt_path = structured_checkpoint / "receipt.json"
    if not structured_state_path.is_file() or not structured_receipt_path.is_file():
        raise SystemExit("selected structured-state checkpoint is incomplete")
    structured_receipt = json.loads(structured_receipt_path.read_text(encoding="utf-8"))
    if int(structured_receipt.get("step", -1)) != STRUCTURED_PARENT_STEP:
        raise SystemExit("graph pilot must initialize from structured step80")
    structured_parent_state = load_file(str(structured_state_path), device="cpu")
    missing, unexpected = structured_parent.load_state_dict(structured_parent_state, strict=True)
    if missing or unexpected:
        raise SystemExit("structured parent state mismatch")

    n0_config = load_n0_config(config_path)
    semantic_model = AliceN0V02Model(n0_config)
    semantic_state_path = semantic_checkpoint / "alice_n0_v02.safetensors"
    semantic_state = load_file(str(semantic_state_path), device="cpu")
    missing, unexpected = semantic_model.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    rows = read_jsonl(curriculum_path)
    print("evidence_graph_semantic_cache_start=true", flush=True)
    payload = build_graph_cache(
        rows=rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured_parent=structured_parent,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    cache_path = output_root / "graph_semantic_cache.pt"
    torch.save(payload, cache_path)
    print(f"evidence_graph_semantic_cache_complete=true path={cache_path}", flush=True)
    del semantic_model, semantic_state
    torch.cuda.empty_cache()

    regression_cache = torch.load(regression_cache_path, map_location="cpu")
    parent_regression = structured_regression_metrics(
        branch=structured_parent.to(device),
        cache=regression_cache,
        device=device,
        batch_size=args.eval_batch_size,
    )
    structured_parent.to("cpu")
    torch.cuda.empty_cache()
    parent_structured_top1 = float(parent_regression["grouped_preferred_top1_accuracy"])

    train_indices = [i for i, split in enumerate(payload["splits"]) if split == "train"]
    dev_indices = [i for i, split in enumerate(payload["splits"]) if split == "dev"]
    train_dataset = GraphDataset(payload, train_indices)
    dev_dataset = GraphDataset(payload, dev_indices)
    if len(train_dataset) != 180 or len(dev_dataset) != 60:
        raise SystemExit("evidence graph train/dev split drift")

    variants = [
        ("compact_graph_only", EvidenceGraphConfig(semantic_size=640, graph_size=256, graph_layers=1), False),
        ("compact_joint", EvidenceGraphConfig(semantic_size=640, graph_size=256, graph_layers=1), True),
        ("expanded_joint_512x2", EvidenceGraphConfig(semantic_size=640, graph_size=512, graph_layers=2), True),
    ]

    results: dict[str, Any] = {}
    for offset, (name, graph_config, structured_trainable) in enumerate(variants):
        print(f"variant_start={name}", flush=True)
        results[name] = train_variant(
            name=name,
            graph_config=graph_config,
            structured_parent_state=copy.deepcopy(structured_parent_state),
            structured_config=structured_config,
            structured_trainable=structured_trainable,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            payload=payload,
            structured_regression_cache=regression_cache,
            output_root=output_root,
            curriculum_hash=sha256_file(curriculum_path),
            device=device,
            max_steps=args.max_steps,
            save_every=args.save_every,
            batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            anchor_weight=args.joint_anchor_weight,
            seed=args.seed + offset,
        )
        torch.cuda.empty_cache()

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for variant_name, variant in results.items():
        for checkpoint_name, receipt in variant["checkpoints"].items():
            candidates.append((variant_name, checkpoint_name, receipt))
    winner_variant, winner_checkpoint, winner_receipt = max(
        candidates,
        key=lambda item: checkpoint_capability_tuple(item[2], parent_structured_top1),
    )

    compact_joint_best = max(
        results["compact_joint"]["checkpoints"].values(),
        key=lambda receipt: checkpoint_capability_tuple(receipt, parent_structured_top1),
    )
    expanded_best = max(
        results["expanded_joint_512x2"]["checkpoints"].values(),
        key=lambda receipt: checkpoint_capability_tuple(receipt, parent_structured_top1),
    )
    compact_graph_best = max(
        results["compact_graph_only"]["checkpoints"].values(),
        key=lambda receipt: checkpoint_capability_tuple(receipt, parent_structured_top1),
    )

    joint_gain = (
        float(compact_joint_best["metrics"]["family_macro_target_support_mass"])
        - float(compact_graph_best["metrics"]["family_macro_target_support_mass"])
    )
    capacity_gain = (
        float(expanded_best["metrics"]["family_macro_target_support_mass"])
        - float(compact_joint_best["metrics"]["family_macro_target_support_mass"])
    )
    min_family_capacity_gain = (
        float(expanded_best["metrics"]["family_min_target_support_mass"])
        - float(compact_joint_best["metrics"]["family_min_target_support_mass"])
    )

    comparison = {
        "schema": "alice.eipm.n0.v02-evidence-graph-pilot-comparison.v0.1",
        "status": "EVIDENCE_READY_NOT_RATIFIED",
        "semantic_parent": "targeted-repair-v0.1/step-00000080",
        "structured_parent": "structured-state-pilot-v0.1/step-00000080",
        "structured_parent_regression": parent_regression,
        "hard_parameter_ceiling": None,
        "capacity_policy": "capability_first_efficiency_secondary",
        "variants": results,
        "provisional_winner": {
            "variant": winner_variant,
            "checkpoint": winner_checkpoint,
            "receipt": winner_receipt,
        },
        "diagnostics": {
            "compact_joint_minus_graph_only_family_macro_target_mass": joint_gain,
            "expanded_minus_compact_joint_family_macro_target_mass": capacity_gain,
            "expanded_minus_compact_joint_family_min_target_mass": min_family_capacity_gain,
            "joint_adaptation_material_signal": joint_gain >= 0.01,
            "additional_capacity_material_signal": capacity_gain >= 0.01 or min_family_capacity_gain >= 0.02,
            "further_capacity_escalation_required": (
                float(winner_receipt["metrics"]["family_min_target_support_mass"]) < 0.70
                or float(winner_receipt["metrics"]["counterfactual_positive_rate"]) < 0.75
            ),
        },
        "selection_policy": "structured-regression-floor_then_worst-family_evidence_then_macro-evidence_then-semantic-summary_then-relation-counterfactual; parameter count is only a tie-break consideration after capability",
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": "inspect capability diagnostics; ratify, prefer joint adaptation, or scale architecture based on heldout evidence",
    }
    comparison_path = output_root / "evidence_graph_pilot_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
