#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from alice_personality.n0.cross_context_fusion import (
    CrossContextFusion,
    CrossContextFusionConfig,
)
from alice_personality.n0.cross_context_fusion_objectives import (
    FusionObjectiveWeights,
    cross_context_fusion_objective,
)
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph import EvidenceGraphConfig
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import (
    EvidenceViewAdapter,
    EvidenceViewAdapterConfig,
)
from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model


EXPECTED_SEMANTIC_PARAMETERS = 136_594_435
EXPECTED_STRUCTURED_PARENT_STEP = 80
EXPECTED_RELATION_GRAPH_STEP = 80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_structured_config(path: Path) -> StructuredStateConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arch = raw["architecture"]
    return StructuredStateConfig(
        semantic_size=int(arch["semantic_size"]),
        state_size=int(arch["state_size"]),
        num_layers=int(arch["num_layers"]),
        num_heads=int(arch["num_heads"]),
        feedforward_size=int(arch["feedforward_size"]),
        dropout=float(arch["dropout"]),
        max_fields=int(arch.get("max_fields", 64)),
    )


def load_fusion_config(path: Path) -> CrossContextFusionConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arch = raw["architecture"]
    return CrossContextFusionConfig(
        semantic_size=int(arch["semantic_size"]),
        fusion_size=int(arch["fusion_size"]),
        num_layers=int(arch["interaction_stages"]),
        num_heads=int(arch["num_heads"]),
        feedforward_size=int(arch["feedforward_size"]),
        dropout=float(arch["dropout"]),
        num_views=int(arch["current_instantiated_view_count"]),
    )


def expanded_adapter_config() -> EvidenceViewAdapterConfig:
    return EvidenceViewAdapterConfig(
        semantic_size=640,
        adapter_size=640,
        num_layers=3,
        num_heads=10,
        feedforward_size=2560,
        dropout=0.0,
        max_fields=64,
        base_prior_scale=0.5,
    )


def expanded_graph_config() -> EvidenceGraphConfig:
    return EvidenceGraphConfig(
        semantic_size=640,
        graph_size=512,
        graph_layers=2,
        max_fields=64,
        max_edges=256,
    )


def encode_token_states(
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            token_states = model.encode_tokens(
                encoded["input_ids"].to(device),
                attention,
            )
        states.append(token_states.detach().float().cpu())
        masks.append(encoded["attention_mask"].bool().cpu())
    return torch.cat(states, dim=0), torch.cat(masks, dim=0)


def encode_pooled(
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
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


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def build_parent_cache(
    rows: list[dict[str, Any]],
    *,
    semantic_model: AliceN0V02Model,
    tokenizer: Any,
    structured: StructuredStateEncoder,
    evidence_adapter: EvidenceViewAdapter,
    evidence_graph: DualEndpointEvidenceGraphEncoder,
    device: torch.device,
    encode_batch_size: int,
    raw_max_length: int,
    field_max_length: int,
) -> dict[str, Any]:
    raw_token_states, raw_token_mask = encode_token_states(
        semantic_model,
        tokenizer,
        [str(row["raw_text"]) for row in rows],
        device=device,
        batch_size=encode_batch_size,
        max_length=raw_max_length,
    )
    query_semantic = encode_pooled(
        semantic_model,
        tokenizer,
        [str(row["query_text"]) for row in rows],
        device=device,
        batch_size=encode_batch_size,
        max_length=field_max_length,
    )
    summary_semantic = encode_pooled(
        semantic_model,
        tokenizer,
        [str(row["target_summary_text"]) for row in rows],
        device=device,
        batch_size=encode_batch_size,
        max_length=field_max_length,
    )

    flat_field_texts = [
        str(field["text"])
        for row in rows
        for field in row["fields"]
    ]
    flat_field_semantic = encode_pooled(
        semantic_model,
        tokenizer,
        flat_field_texts,
        device=device,
        batch_size=encode_batch_size,
        max_length=field_max_length,
    )

    n = len(rows)
    semantic_size = int(query_semantic.shape[-1])
    max_fields = max(len(row["fields"]) for row in rows)
    max_edges = max(1, max(len(row["relations"]) for row in rows))

    field_semantic = torch.zeros(n, max_fields, semantic_size)
    field_type_ids = torch.zeros(n, max_fields, dtype=torch.long)
    provenance_ids = torch.zeros(n, max_fields, dtype=torch.long)
    relation_role_ids = torch.zeros(n, max_fields, dtype=torch.long)
    temporal_scope_ids = torch.zeros(n, max_fields, dtype=torch.long)
    confidence = torch.zeros(n, max_fields, 1)
    missing_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    valid_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    edge_index = torch.zeros(n, max_edges, 2, dtype=torch.long)
    edge_type_ids = torch.zeros(n, max_edges, dtype=torch.long)
    edge_confidence = torch.zeros(n, max_edges, 1)
    edge_valid_mask = torch.zeros(n, max_edges, dtype=torch.bool)
    target_view_distribution = torch.zeros(n, 3)
    view_reliability = torch.zeros(n, 3)
    view_available = torch.zeros(n, 3, dtype=torch.bool)

    cursor = 0
    for i, row in enumerate(rows):
        fields = row["fields"]
        count = len(fields)
        field_semantic[i, :count] = flat_field_semantic[cursor : cursor + count]
        cursor += count
        valid_mask[i, :count] = True
        for j, item in enumerate(fields):
            field_type_ids[i, j] = int(item["field_type_id"])
            provenance_ids[i, j] = int(item["provenance_id"])
            relation_role_ids[i, j] = int(item["relation_role_id"])
            temporal_scope_ids[i, j] = int(item["temporal_scope_id"])
            confidence[i, j, 0] = float(item["confidence"])
            missing_mask[i, j] = bool(item["missing"])
        for edge_number, relation in enumerate(row["relations"]):
            edge_index[i, edge_number, 0] = int(relation["source"])
            edge_index[i, edge_number, 1] = int(relation["target"])
            edge_type_ids[i, edge_number] = int(relation["relation_type_id"])
            edge_confidence[i, edge_number, 0] = float(relation["confidence"])
            edge_valid_mask[i, edge_number] = True
        target_view_distribution[i] = torch.tensor(
            row["target_view_distribution"], dtype=torch.float32
        )
        view_reliability[i] = torch.tensor(row["view_reliability"], dtype=torch.float32)
        view_available[i] = torch.tensor(row["view_available"], dtype=torch.bool)

    structured = structured.to(device).eval()
    evidence_adapter = evidence_adapter.to(device).eval()
    evidence_graph = evidence_graph.to(device).eval()

    structured_states: list[torch.Tensor] = []
    structured_pooled: list[torch.Tensor] = []
    evidence_states: list[torch.Tensor] = []
    evidence_pooled: list[torch.Tensor] = []
    parent_batch = 64
    with torch.inference_mode():
        for start in range(0, n, parent_batch):
            end = min(start + parent_batch, n)
            structured_out = structured(
                semantic_values=field_semantic[start:end].to(device),
                field_type_ids=field_type_ids[start:end].to(device),
                provenance_ids=provenance_ids[start:end].to(device),
                relation_role_ids=relation_role_ids[start:end].to(device),
                temporal_scope_ids=temporal_scope_ids[start:end].to(device),
                confidence=confidence[start:end].to(device),
                missing_mask=missing_mask[start:end].to(device),
                valid_mask=valid_mask[start:end].to(device),
            )
            adapted = evidence_adapter(
                parent_field_states=structured_out["field_states"],
                valid_mask=valid_mask[start:end].to(device),
                query_semantic=query_semantic[start:end].to(device),
                parent_field_weights=structured_out["field_weights"],
            )
            graph_out = evidence_graph(
                field_states=adapted["field_states"],
                valid_mask=valid_mask[start:end].to(device),
                edge_index=edge_index[start:end].to(device),
                edge_type_ids=edge_type_ids[start:end].to(device),
                edge_confidence=edge_confidence[start:end].to(device),
                edge_valid_mask=edge_valid_mask[start:end].to(device),
                query_semantic=query_semantic[start:end].to(device),
                base_field_weights=adapted["field_weights"],
            )
            structured_states.append(structured_out["field_states"].detach().float().cpu())
            structured_pooled.append(structured_out["pooled_state"].detach().float().cpu())
            evidence_states.append(graph_out["field_states"].detach().float().cpu())
            evidence_pooled.append(graph_out["pooled_state"].detach().float().cpu())

    structured_tokens = torch.cat(structured_states, dim=0)
    structured_summary = torch.cat(structured_pooled, dim=0)
    evidence_tokens = torch.cat(evidence_states, dim=0)
    evidence_summary = torch.cat(evidence_pooled, dim=0)
    semantic_summary = masked_mean(raw_token_states, raw_token_mask)

    semantic_valid_mask = raw_token_mask & view_available[:, 0].unsqueeze(1)
    structured_valid_mask = valid_mask & view_available[:, 1].unsqueeze(1)
    evidence_valid_mask = valid_mask & view_available[:, 2].unsqueeze(1)
    source_view_summaries = torch.stack(
        [semantic_summary, structured_summary, evidence_summary], dim=1
    )

    return {
        "semantic_tokens": raw_token_states,
        "semantic_valid_mask": semantic_valid_mask,
        "structured_tokens": structured_tokens,
        "structured_valid_mask": structured_valid_mask,
        "evidence_tokens": evidence_tokens,
        "evidence_valid_mask": evidence_valid_mask,
        "query_semantic": query_semantic,
        "semantic_target": summary_semantic,
        "source_view_summaries": source_view_summaries,
        "target_view_distribution": target_view_distribution,
        "view_reliability": view_reliability,
        "view_available": view_available,
        "ids": [str(row["id"]) for row in rows],
        "families": [str(row["family"]) for row in rows],
        "splits": [str(row["split"]) for row in rows],
    }


TENSOR_KEYS = (
    "semantic_tokens",
    "semantic_valid_mask",
    "structured_tokens",
    "structured_valid_mask",
    "evidence_tokens",
    "evidence_valid_mask",
    "query_semantic",
    "semantic_target",
    "source_view_summaries",
    "target_view_distribution",
    "view_reliability",
    "view_available",
)


class FusionDataset(Dataset):
    def __init__(self, cache: dict[str, Any], indices: list[int]) -> None:
        self.cache = cache
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        out = {key: self.cache[key][index] for key in TENSOR_KEYS}
        out["global_index"] = index
        return out


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_fusion(
    model: CrossContextFusion,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return model(
        semantic_tokens=batch["semantic_tokens"],
        semantic_valid_mask=batch["semantic_valid_mask"],
        structured_tokens=batch["structured_tokens"],
        structured_valid_mask=batch["structured_valid_mask"],
        evidence_tokens=batch["evidence_tokens"],
        evidence_valid_mask=batch["evidence_valid_mask"],
        query_semantic=batch["query_semantic"],
        view_reliability=batch["view_reliability"],
    )


def objective(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    weights: FusionObjectiveWeights,
) -> dict[str, torch.Tensor]:
    return cross_context_fusion_objective(
        fused_state=output["fused_state"],
        semantic_target=batch["semantic_target"],
        view_weights=output["view_weights"],
        target_view_distribution=batch["target_view_distribution"],
        contextualized_view_summaries=output["view_summaries"],
        source_view_targets=batch["source_view_summaries"],
        view_available=batch["view_available"],
        weights=weights,
    )


def evaluate(
    *,
    model: CrossContextFusion,
    dataset: FusionDataset,
    cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    family_routing: dict[str, list[float]] = defaultdict(list)
    semantic_cosines: list[float] = []
    preservation_cosines: list[float] = []
    disagreement_errors: list[float] = []
    decisive_correct = 0
    decisive_total = 0
    missing_leak = 0.0
    gate_values: list[float] = []

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = forward_fusion(model, batch)

            routing_similarity = 1.0 - 0.5 * (
                out["view_weights"] - batch["target_view_distribution"]
            ).abs().sum(dim=-1)
            semantic_cos = F.cosine_similarity(
                out["fused_state"], batch["semantic_target"], dim=-1
            )
            view_cos = F.cosine_similarity(
                out["view_summaries"], batch["source_view_summaries"], dim=-1
            )
            active = batch["view_available"]
            preservation_cosines.extend(
                float(x)
                for x in view_cos[active].detach().float().cpu().tolist()
            )

            out_norm = F.normalize(out["view_summaries"], dim=-1)
            src_norm = F.normalize(batch["source_view_summaries"], dim=-1)
            out_sim = torch.einsum("bvd,bwd->bvw", out_norm, out_norm)
            src_sim = torch.einsum("bvd,bwd->bvw", src_norm, src_norm)
            pair_mask = active.unsqueeze(2) & active.unsqueeze(1)
            eye = torch.eye(active.size(1), device=device, dtype=torch.bool).unsqueeze(0)
            pair_mask = pair_mask & ~eye
            for row in range(active.size(0)):
                row_mask = pair_mask[row]
                if row_mask.any():
                    disagreement_errors.append(
                        float((out_sim[row] - src_sim[row]).abs()[row_mask].mean().float().cpu())
                    )

            unavailable = ~active
            if unavailable.any():
                missing_leak = max(
                    missing_leak,
                    float(out["view_weights"][unavailable].max().float().cpu()),
                )

            target_sorted, _ = batch["target_view_distribution"].sort(dim=-1, descending=True)
            decisive = (target_sorted[:, 0] - target_sorted[:, 1]) >= 0.25
            if decisive.any():
                pred = out["view_weights"].argmax(dim=-1)
                target = batch["target_view_distribution"].argmax(dim=-1)
                decisive_correct += int((pred[decisive] == target[decisive]).sum().item())
                decisive_total += int(decisive.sum().item())

            semantic_cosines.extend(float(x) for x in semantic_cos.float().cpu().tolist())
            gate_values.extend(
                float(x) for x in out["cross_gate_means"].float().cpu().reshape(-1).tolist()
            )
            global_indices = batch["global_index"].cpu().tolist()
            routing_cpu = routing_similarity.float().cpu().tolist()
            for local, global_index in enumerate(global_indices):
                family = str(cache["families"][int(global_index)])
                family_routing[family].append(float(routing_cpu[local]))

    family_means = {
        family: sum(values) / len(values)
        for family, values in sorted(family_routing.items())
    }
    return {
        "rows": len(dataset),
        "family_routing_similarity": family_means,
        "family_macro_routing_similarity": sum(family_means.values()) / len(family_means),
        "family_min_routing_similarity": min(family_means.values()),
        "decisive_view_accuracy": decisive_correct / max(decisive_total, 1),
        "decisive_rows": decisive_total,
        "fused_semantic_cosine": sum(semantic_cosines) / max(len(semantic_cosines), 1),
        "view_preservation_cosine": sum(preservation_cosines) / max(len(preservation_cosines), 1),
        "disagreement_geometry_mae": sum(disagreement_errors) / max(len(disagreement_errors), 1),
        "missing_view_max_weight": missing_leak,
        "mean_cross_gate_activation": sum(gate_values) / max(len(gate_values), 1),
    }


def capability_score(metrics: dict[str, Any], step: int) -> tuple[float, ...]:
    return (
        float(metrics["family_min_routing_similarity"]),
        float(metrics["family_macro_routing_similarity"]),
        float(metrics["decisive_view_accuracy"]),
        float(metrics["fused_semantic_cosine"]),
        float(metrics["view_preservation_cosine"]),
        -float(metrics["disagreement_geometry_mae"]),
        -float(step),
    )


def gate_pass(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["family_min_routing_similarity"]) >= 0.90
        and float(metrics["family_macro_routing_similarity"]) >= 0.95
        and float(metrics["decisive_view_accuracy"]) >= 0.95
        and float(metrics["fused_semantic_cosine"]) >= 0.90
        and float(metrics["view_preservation_cosine"]) >= 0.90
        and float(metrics["disagreement_geometry_mae"]) <= 0.10
        and float(metrics["missing_view_max_weight"]) <= 1e-6
    )


def save_checkpoint(
    *,
    model: CrossContextFusion,
    root: Path,
    step: int,
    metrics: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    checkpoint = root / f"step-{step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint / "cross_context_fusion.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(model_path),
    )
    receipt = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-checkpoint.v0.1",
        "status": "TRAINED_FULL_SCALE_NOT_RATIFIED",
        "step": step,
        "git_revision": git_revision(),
        "fusion_sha256": sha256_file(model_path),
        "fusion_parameter_report": model.parameter_report(),
        "dev_metrics": metrics,
        "gate_pass": gate_pass(metrics),
        "lineage": lineage,
        "full_scale_model": True,
        "reduced_capability_pilot": False,
        "hard_parameter_ceiling": None,
        "view_count_ceiling": None,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    (checkpoint / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-config", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--evidence-adapter", required=True)
    parser.add_argument("--evidence-graph", required=True)
    parser.add_argument("--fusion-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--prep-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-max-length", type=int, default=96)
    parser.add_argument("--field-max-length", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("full-scale cross-context fusion training requires CUDA")
    device = torch.device("cuda")

    training_config_path = Path(args.training_config).resolve()
    training_cfg = json.loads(training_config_path.read_text(encoding="utf-8"))
    if training_cfg.get("status") != "CONDITIONALLY_AUTHORIZED_AFTER_SAME_REVISION_PREP_PASS":
        raise SystemExit("fusion training authorization config status mismatch")
    auth = training_cfg.get("authorization", {})
    if auth.get("public_fusion_gradient_authorized") is not True:
        raise SystemExit("public fusion gradient is not authorized")
    if auth.get("private_identity_gradient") is not False:
        raise SystemExit("fusion training crossed private gradient boundary")

    prep_receipt = json.loads(Path(args.prep_receipt).read_text(encoding="utf-8"))
    current_revision = git_revision()
    if prep_receipt.get("status") != "PASS":
        raise SystemExit("fusion preparation receipt did not pass")
    if prep_receipt.get("git_revision") != current_revision:
        raise SystemExit(
            f"fusion preparation receipt revision mismatch: receipt={prep_receipt.get('git_revision')} current={current_revision}"
        )
    if prep_receipt.get("no_accidental_capability_ceilings_gate") is not True:
        raise SystemExit("no-accidental-capability-ceilings gate missing from preparation receipt")

    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-cross-context-fusion-curriculum.v0.2":
        raise SystemExit("fusion curriculum schema mismatch")
    if manifest.get("compiled_sha256") != sha256_file(curriculum_path):
        raise SystemExit("fusion curriculum hash mismatch")
    if manifest.get("private_identity_content") is not False:
        raise SystemExit("fusion curriculum crossed private identity boundary")
    if manifest.get("identity_authority") is not False:
        raise SystemExit("public synthetic fusion curriculum cannot be identity authority")

    rows = read_jsonl(curriculum_path)
    if len(rows) != int(training_cfg["curriculum"]["rows"]):
        raise SystemExit("fusion curriculum row count mismatch")
    if any(len(row["target_view_distribution"]) != 3 for row in rows):
        raise SystemExit("current N0 fusion curriculum must supply the three public views")

    opt_cfg = training_cfg["optimization"]
    seed = int(opt_cfg["seed"])
    seed_everything(seed)
    max_steps = int(opt_cfg["max_steps"])
    save_steps = [int(step) for step in opt_cfg["save_steps"]]
    train_batch_size = int(opt_cfg["train_batch_size"])
    eval_batch_size = int(opt_cfg["eval_batch_size"])
    learning_rate = float(opt_cfg["learning_rate"])
    warmup_steps = int(opt_cfg["warmup_steps"])
    weight_decay = float(opt_cfg["weight_decay"])

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file

    semantic_config = load_n0_config(Path(args.semantic_config).resolve())
    semantic_model = AliceN0V02Model(semantic_config)
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    semantic_state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"), device="cpu"
    )
    missing, unexpected = semantic_model.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    if semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    structured = StructuredStateEncoder(
        load_structured_config(Path(args.structured_config).resolve())
    )
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    structured_receipt = json.loads(
        (structured_checkpoint / "receipt.json").read_text(encoding="utf-8")
    )
    if int(structured_receipt.get("step", -1)) != EXPECTED_STRUCTURED_PARENT_STEP:
        raise SystemExit("fusion training requires ratified structured step80")
    structured.load_state_dict(
        load_file(str(structured_checkpoint / "structured_state.safetensors"), device="cpu"),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    evidence_adapter_path = Path(args.evidence_adapter).resolve()
    evidence_adapter = EvidenceViewAdapter(expanded_adapter_config())
    evidence_adapter.load_state_dict(
        load_file(str(evidence_adapter_path), device="cpu"), strict=True
    )
    for parameter in evidence_adapter.parameters():
        parameter.requires_grad = False

    evidence_graph_path = Path(args.evidence_graph).resolve()
    evidence_graph = DualEndpointEvidenceGraphEncoder(expanded_graph_config())
    evidence_graph.load_state_dict(
        load_file(str(evidence_graph_path), device="cpu"), strict=True
    )
    for parameter in evidence_graph.parameters():
        parameter.requires_grad = False

    print("fusion_parent_cache_start=true", flush=True)
    cache = build_parent_cache(
        rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured=structured,
        evidence_adapter=evidence_adapter,
        evidence_graph=evidence_graph,
        device=device,
        encode_batch_size=args.encode_batch_size,
        raw_max_length=args.raw_max_length,
        field_max_length=args.field_max_length,
    )
    cache_path = output_root / "fusion_parent_cache.pt"
    torch.save(cache, cache_path)
    print(f"fusion_parent_cache_sha256={sha256_file(cache_path)}", flush=True)

    lineage = {
        "semantic_sha256": sha256_file(semantic_checkpoint / "alice_n0_v02.safetensors"),
        "structured_sha256": sha256_file(structured_checkpoint / "structured_state.safetensors"),
        "evidence_adapter_sha256": sha256_file(evidence_adapter_path),
        "evidence_graph_sha256": sha256_file(evidence_graph_path),
        "curriculum_sha256": sha256_file(curriculum_path),
        "curriculum_manifest_sha256": sha256_file(manifest_path),
        "prep_receipt_sha256": sha256_file(Path(args.prep_receipt).resolve()),
        "parent_cache_sha256": sha256_file(cache_path),
    }

    del semantic_model, semantic_state, structured, evidence_adapter, evidence_graph
    torch.cuda.empty_cache()

    train_indices = [i for i, split in enumerate(cache["splits"]) if split == "train"]
    dev_indices = [i for i, split in enumerate(cache["splits"]) if split == "dev"]
    if len(train_indices) != int(training_cfg["curriculum"]["train_rows"]):
        raise SystemExit("fusion train split count mismatch")
    if len(dev_indices) != int(training_cfg["curriculum"]["dev_rows"]):
        raise SystemExit("fusion dev split count mismatch")

    train_dataset = FusionDataset(cache, train_indices)
    dev_dataset = FusionDataset(cache, dev_indices)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    fusion_config = load_fusion_config(Path(args.fusion_config).resolve())
    if fusion_config.num_views != 3:
        raise SystemExit("current public fusion training expects three instantiated N0 views")
    model = CrossContextFusion(fusion_config).to(device)
    report = model.parameter_report()
    print("fusion_parameter_report=" + json.dumps(report, sort_keys=True), flush=True)
    if report.get("hard_parameter_ceiling") is not None or report.get("view_count_ceiling") is not None:
        raise SystemExit("fusion architecture unexpectedly contains a hard capability ceiling")
    if report.get("full_scale_n0_candidate") is not True:
        raise SystemExit("fusion architecture is not marked full scale")
    if report.get("reduced_pilot_model") is not False:
        raise SystemExit("reduced fusion model is forbidden")

    weight_cfg = training_cfg["objective"]
    loss_weights = FusionObjectiveWeights(
        fused_semantic=float(weight_cfg["fused_semantic"]),
        view_routing=float(weight_cfg["view_routing"]),
        view_preservation=float(weight_cfg["view_preservation"]),
        disagreement_preservation=float(weight_cfg["disagreement_preservation"]),
    )

    baseline = evaluate(
        model=model,
        dataset=dev_dataset,
        cache=cache,
        device=device,
        batch_size=eval_batch_size,
    )
    print("fusion_random_baseline=" + json.dumps(baseline, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    loader_iter = iter(train_loader)
    saved: list[dict[str, Any]] = []
    running: defaultdict[str, float] = defaultdict(float)

    model.train()
    for step in range(1, max_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = forward_fusion(model, batch)
            losses = objective(output, batch, loss_weights)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for key, value in losses.items():
            running[key] += float(value.detach().float().cpu())

        if step % 20 == 0:
            averaged = {key: value / 20.0 for key, value in running.items()}
            print(
                "fusion_train_step="
                + json.dumps(
                    {
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        **averaged,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            running.clear()

        if step in save_steps:
            metrics = evaluate(
                model=model,
                dataset=dev_dataset,
                cache=cache,
                device=device,
                batch_size=eval_batch_size,
            )
            receipt = save_checkpoint(
                model=model,
                root=output_root,
                step=step,
                metrics=metrics,
                lineage=lineage,
            )
            saved.append(receipt)
            print(
                "fusion_checkpoint_eval=" + json.dumps(receipt, sort_keys=True),
                flush=True,
            )
            model.train()

    if not saved:
        raise SystemExit("fusion training produced no checkpoints")
    winner = max(
        saved,
        key=lambda item: capability_score(item["dev_metrics"], int(item["step"])),
    )
    comparison = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-training-comparison.v0.1",
        "status": "PASS_FULL_SCALE_TRAINED_NOT_RATIFIED" if winner["gate_pass"] else "FULL_SCALE_TRAINED_NEEDS_FAILURE_DRIVEN_WORK",
        "architecture": "full_scale_frontier_multi_stream_gated_bidirectional_cross_attention",
        "baseline": baseline,
        "checkpoints": saved,
        "provisional_winner": winner["step"],
        "winner": winner,
        "gate_pass": bool(winner["gate_pass"]),
        "selection_policy": "worst_family_routing_then_macro_routing_then_decisive_view_then_semantic_then_preservation_then_disagreement_then_earlier_checkpoint",
        "hard_parameter_ceiling": None,
        "view_count_ceiling": None,
        "run_step_bound_is_compute_control_not_model_limit": True,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": (
            "build_untouched_cross_context_fusion_challenge_before_ratification"
            if winner["gate_pass"]
            else "perform_failure_driven_data_or_architecture_work_without_shrinking_model_for_efficiency"
        ),
    }
    comparison_path = output_root / "cross_context_fusion_training_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
