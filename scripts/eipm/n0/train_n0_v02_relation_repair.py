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

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum import sha256_file
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph import EvidenceGraphConfig
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_graph_objectives import evidence_graph_objective
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter, EvidenceViewAdapterConfig
from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model


EXPECTED_SEMANTIC_PARAMETERS = 136_594_435
STRUCTURED_PARENT_STEP = 80
SPECIALIST_PARENT_STEP = 80


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
    arch = raw["architecture"]
    return StructuredStateConfig(
        semantic_size=int(arch["semantic_size"]),
        state_size=int(arch["state_size"]),
        num_layers=int(arch["num_layers"]),
        num_heads=int(arch["num_heads"]),
        feedforward_size=int(arch["feedforward_size"]),
        dropout=float(arch["dropout"]),
        max_fields=int(arch["max_fields"]),
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


def encode_texts(
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


def build_repair_cache(
    rows: list[dict[str, Any]],
    *,
    semantic_model: AliceN0V02Model,
    tokenizer: Any,
    structured_parent: StructuredStateEncoder,
    device: torch.device,
    encode_batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    max_fields = max(len(row["fields"]) for row in rows)
    max_edges = max(len(row["relations"]) for row in rows)
    flat_field_texts = [str(field["text"]) for row in rows for field in row["fields"]]
    field_vectors_flat = encode_texts(
        semantic_model, tokenizer, flat_field_texts,
        device=device, batch_size=encode_batch_size, max_length=max_length,
    )
    query_vectors = encode_texts(
        semantic_model, tokenizer, [str(row["query_text"]) for row in rows],
        device=device, batch_size=encode_batch_size, max_length=max_length,
    )
    summary_vectors = encode_texts(
        semantic_model, tokenizer, [str(row["target_summary_text"]) for row in rows],
        device=device, batch_size=encode_batch_size, max_length=max_length,
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

    parent_field_states: list[torch.Tensor] = []
    parent_field_weights: list[torch.Tensor] = []
    structured_parent = structured_parent.to(device).eval()
    with torch.inference_mode():
        for start in range(0, n, 128):
            end = start + 128
            out = structured_parent(
                semantic_values=field_semantic[start:end].to(device),
                field_type_ids=field_type_ids[start:end].to(device),
                provenance_ids=provenance_ids[start:end].to(device),
                relation_role_ids=relation_role_ids[start:end].to(device),
                temporal_scope_ids=temporal_scope_ids[start:end].to(device),
                confidence=confidence[start:end].to(device),
                missing_mask=missing_mask[start:end].to(device),
                valid_mask=valid_mask[start:end].to(device),
            )
            parent_field_states.append(out["field_states"].detach().float().cpu())
            parent_field_weights.append(out["field_weights"].detach().float().cpu())

    return {
        "field_semantic": field_semantic,
        "valid_mask": valid_mask,
        "target_distribution": target_distribution,
        "edge_index": edge_index,
        "edge_type_ids": edge_type_ids,
        "edge_confidence": edge_confidence,
        "edge_valid_mask": edge_valid_mask,
        "query_semantic": query_vectors,
        "summary_semantic": summary_vectors,
        "parent_field_states": torch.cat(parent_field_states, dim=0),
        "parent_field_weights": torch.cat(parent_field_weights, dim=0),
        "ids": [str(row["id"]) for row in rows],
        "pair_ids": [str(row["pair_id"]) for row in rows],
        "variants": [str(row["variant"]) for row in rows],
        "families": [str(row["family"]) for row in rows],
        "splits": [str(row["split"]) for row in rows],
    }


TENSOR_KEYS = (
    "field_semantic", "valid_mask", "target_distribution", "edge_index",
    "edge_type_ids", "edge_confidence", "edge_valid_mask", "query_semantic",
    "summary_semantic", "parent_field_states", "parent_field_weights",
)


class RowDataset(Dataset):
    def __init__(self, payload: dict[str, Any], indices: list[int]) -> None:
        self.payload = payload
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        out = {key: self.payload[key][index] for key in TENSOR_KEYS}
        out["global_index"] = index
        return out


class PairDataset(Dataset):
    def __init__(self, payload: dict[str, Any], pair_indices: list[tuple[int, int]]) -> None:
        self.payload = payload
        self.pair_indices = pair_indices

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        a_index, b_index = self.pair_indices[item]
        out: dict[str, Any] = {}
        for prefix, index in (("a", a_index), ("b", b_index)):
            for key in TENSOR_KEYS:
                out[f"{prefix}_{key}"] = self.payload[key][index]
            out[f"{prefix}_global_index"] = index
        return out


def pair_indices(payload: dict[str, Any], split: str) -> list[tuple[int, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for index, (pair_id, variant, row_split) in enumerate(
        zip(payload["pair_ids"], payload["variants"], payload["splits"])
    ):
        if row_split == split:
            grouped[str(pair_id)][str(variant)] = index
    pairs: list[tuple[int, int]] = []
    for pair_id in sorted(grouped):
        members = grouped[pair_id]
        if set(members) != {"A", "B"}:
            raise SystemExit(f"repair pair {pair_id} is incomplete")
        pairs.append((members["A"], members["B"]))
    return pairs


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def unprefix(batch: dict[str, Any], prefix: str) -> dict[str, torch.Tensor]:
    token = prefix + "_"
    return {
        key[len(token):]: value
        for key, value in batch.items()
        if key.startswith(token) and torch.is_tensor(value) and key != f"{prefix}_global_index"
    }


def adapter_forward(adapter: EvidenceViewAdapter, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return adapter(
        parent_field_states=batch["parent_field_states"],
        valid_mask=batch["valid_mask"],
        query_semantic=batch["query_semantic"],
        parent_field_weights=batch["parent_field_weights"],
    )


def graph_forward(
    graph: DualEndpointEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapted = adapter_forward(adapter, batch)
    return graph(
        field_states=adapted["field_states"],
        valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"],
        edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"],
        edge_valid_mask=batch["edge_valid_mask"],
        query_semantic=batch["query_semantic"],
        base_field_weights=adapted["field_weights"],
    )


def permute_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fields = batch["valid_mask"].size(1)
    order = torch.arange(fields - 1, -1, -1, device=batch["valid_mask"].device)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(fields, device=order.device)
    out = dict(batch)
    for key in (
        "field_semantic", "valid_mask", "target_distribution",
        "parent_field_states", "parent_field_weights",
    ):
        out[key] = batch[key].index_select(1, order)
    out["edge_index"] = inverse[batch["edge_index"]]
    return out


def row_objective(
    *,
    graph: DualEndpointEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
    paired_counterfactual: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = graph_forward(graph, adapter, batch)
    permuted_batch = permute_batch(batch)
    permuted = graph_forward(graph, adapter, permuted_batch)
    corrupted = paired_counterfactual if paired_counterfactual is not None else output["pooled_state"].detach()
    active = torch.ones(batch["valid_mask"].size(0), device=batch["valid_mask"].device, dtype=torch.bool)
    result = evidence_graph_objective(
        field_weights=output["field_weights"],
        target_distribution=batch["target_distribution"],
        valid_mask=batch["valid_mask"],
        pooled_state=output["pooled_state"],
        semantic_target=batch["summary_semantic"],
        corrupted_pooled_state=corrupted,
        counterfactual_active_mask=active,
        field_states=output["field_states"],
        field_semantic=batch["field_semantic"],
        permuted_pooled_state=permuted["pooled_state"],
        selection_weight=0.40,
        semantic_weight=0.25,
        counterfactual_weight=0.20 if paired_counterfactual is not None else 0.0,
        field_preservation_weight=0.10,
        permutation_weight=0.05 if paired_counterfactual is not None else 0.25,
        counterfactual_margin=0.10,
    )
    return result, output


def pair_flip_loss(a_out: dict[str, torch.Tensor], b_out: dict[str, torch.Tensor], a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], margin: float = 0.25) -> torch.Tensor:
    target_a = a["target_distribution"].argmax(dim=-1)
    target_b = b["target_distribution"].argmax(dim=-1)
    row = torch.arange(target_a.size(0), device=target_a.device)
    a_right = a_out["field_weights"][row, target_a]
    a_wrong = a_out["field_weights"][row, target_b]
    b_right = b_out["field_weights"][row, target_b]
    b_wrong = b_out["field_weights"][row, target_a]
    within = F.relu(margin - a_right + a_wrong) + F.relu(margin - b_right + b_wrong)
    cross = F.relu(0.5 * margin - a_right + b_out["field_weights"][row, target_a])
    cross = cross + F.relu(0.5 * margin - b_right + a_out["field_weights"][row, target_b])
    return (within + 0.5 * cross).mean()


def evaluate_pairs(
    *,
    graph: DualEndpointEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: PairDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    correct = 0
    examples = 0
    margins: list[float] = []
    target_mass: list[float] = []
    family_correct: dict[str, list[int]] = defaultdict(list)
    family_margin: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            a = unprefix(batch, "a")
            b = unprefix(batch, "b")
            a_out = graph_forward(graph, adapter, a)
            b_out = graph_forward(graph, adapter, b)
            a_target = a["target_distribution"].argmax(dim=-1)
            b_target = b["target_distribution"].argmax(dim=-1)
            row = torch.arange(a_target.size(0), device=device)
            a_pred = a_out["field_weights"].argmax(dim=-1)
            b_pred = b_out["field_weights"].argmax(dim=-1)
            pair_ok = (a_pred == a_target) & (b_pred == b_target)
            a_margin = a_out["field_weights"][row, a_target] - a_out["field_weights"][row, b_target]
            b_margin = b_out["field_weights"][row, b_target] - b_out["field_weights"][row, a_target]
            pair_margin = torch.minimum(a_margin, b_margin)
            target_mass.extend(float(x) for x in a_out["field_weights"][row, a_target].cpu().tolist())
            target_mass.extend(float(x) for x in b_out["field_weights"][row, b_target].cpu().tolist())
            correct += int(pair_ok.sum().item())
            examples += int(pair_ok.numel())
            margins.extend(float(x) for x in pair_margin.cpu().tolist())
            global_indices = batch["a_global_index"].cpu().tolist()
            for local, global_index in enumerate(global_indices):
                family = str(payload["families"][int(global_index)])
                family_correct[family].append(int(pair_ok[local].item()))
                family_margin[family].append(float(pair_margin[local].item()))

    family_accuracy = {family: sum(values) / len(values) for family, values in sorted(family_correct.items())}
    family_margins = {family: sum(values) / len(values) for family, values in sorted(family_margin.items())}
    return {
        "pairs": examples,
        "pair_flip_accuracy": correct / max(examples, 1),
        "family_pair_flip_accuracy": family_accuracy,
        "family_min_pair_flip_accuracy": min(family_accuracy.values()),
        "family_pair_flip_margin": family_margins,
        "mean_pair_flip_margin": sum(margins) / max(len(margins), 1),
        "mean_target_mass": sum(target_mass) / max(len(target_mass), 1),
    }


def evaluate_replay(
    *,
    graph: DualEndpointEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: RowDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    family_mass: dict[str, list[float]] = defaultdict(list)
    summary: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            out = graph_forward(graph, adapter, batch)
            target = batch["target_distribution"]
            support = target > 0
            mass = (out["field_weights"] * support.to(out["field_weights"].dtype)).sum(dim=-1)
            summary_cos = F.cosine_similarity(out["pooled_state"], batch["summary_semantic"], dim=-1)
            global_indices = batch["global_index"].cpu().tolist()
            for local, global_index in enumerate(global_indices):
                family = str(payload["families"][int(global_index)])
                family_mass[family].append(float(mass[local].item()))
                summary.append(float(summary_cos[local].item()))
    means = {family: sum(values) / len(values) for family, values in sorted(family_mass.items())}
    return {
        "family_macro_target_support_mass": sum(means.values()) / max(len(means), 1),
        "family_min_target_support_mass": min(means.values()),
        "family_target_support_mass": means,
        "mean_summary_cosine": sum(summary) / max(len(summary), 1),
    }


def set_relation_read_trainable(graph: DualEndpointEvidenceGraphEncoder) -> list[torch.nn.Parameter]:
    for parameter in graph.parameters():
        parameter.requires_grad = False
    modules = [
        graph.source_relation_pool_mlp,
        graph.directed_relation_pool_mlp,
        graph.conflict_pool_mlp,
        graph.pool_relation_embedding,
        graph.query_projection,
    ]
    graph.pool_query.requires_grad = True
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in graph.parameters() if parameter.requires_grad]


def save_checkpoint(
    *,
    graph: DualEndpointEvidenceGraphEncoder,
    output_root: Path,
    step: int,
    repair_metrics: dict[str, Any],
    replay_metrics: dict[str, Any],
    parent_adapter_hash: str,
    parent_graph_hash: str,
    repair_curriculum_hash: str,
    replay_cache_hash: str,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    root = output_root / f"step-{step:08d}"
    root.mkdir(parents=True, exist_ok=True)
    graph_path = root / "evidence_graph_dual_endpoint.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in graph.state_dict().items()},
        str(graph_path),
    )
    receipt = {
        "schema": "alice.eipm.n0.v02-relation-repair-checkpoint.v0.1",
        "status": "TRAINED_NOT_RATIFIED",
        "step": step,
        "git_revision": git_revision(),
        "parent_specialist_variant": "specialized_expanded_640x3_graph512x2",
        "parent_specialist_step": SPECIALIST_PARENT_STEP,
        "parent_adapter_sha256": parent_adapter_hash,
        "parent_graph_sha256": parent_graph_hash,
        "repair_curriculum_sha256": repair_curriculum_hash,
        "replay_cache_sha256": replay_cache_hash,
        "graph_sha256": sha256_file(graph_path),
        "graph_parameter_count": graph.parameter_report()["total_parameters"],
        "trainable_relation_read_parameters": sum(p.numel() for p in graph.parameters() if p.requires_grad),
        "repair_dev_metrics": repair_metrics,
        "ordinary_replay_dev_metrics": replay_metrics,
        "frozen_relation_essential_challenge_used_for_training": False,
        "shared_structured_parent_mutated": False,
        "evidence_view_adapter_mutated": False,
        "semantic_core_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "hard_parameter_ceiling": None,
        "production_promotion_authorized": False,
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-config", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--parent-adapter", required=True)
    parser.add_argument("--parent-graph", required=True)
    parser.add_argument("--repair-curriculum", required=True)
    parser.add_argument("--repair-manifest", required=True)
    parser.add_argument("--replay-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--train-pair-batch-size", type=int, default=16)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("relation repair requires one CUDA device")
    device = torch.device("cuda")
    seed_everything(args.seed)

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    repair_path = Path(args.repair_curriculum).resolve()
    repair_manifest_path = Path(args.repair_manifest).resolve()
    manifest = json.loads(repair_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-relation-repair-curriculum.v0.1":
        raise SystemExit("relation repair manifest schema mismatch")
    if manifest.get("training_authorized") is not True:
        raise SystemExit("relation repair curriculum is not training-authorized")
    if manifest.get("frozen_relation_essential_challenge_rows_used_for_training") is not False:
        raise SystemExit("frozen relation-essential challenge leaked into repair training")
    if manifest.get("private_identity_content") is not False:
        raise SystemExit("repair curriculum crossed private identity boundary")
    repair_hash = sha256_file(repair_path)
    if repair_hash != str(manifest.get("compiled_sha256")):
        raise SystemExit("relation repair curriculum hash mismatch")

    from safetensors.torch import load_file

    semantic_config = load_n0_config(Path(args.semantic_config).resolve())
    semantic_model = AliceN0V02Model(semantic_config)
    semantic_state = load_file(str(Path(args.semantic_checkpoint).resolve() / "alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = semantic_model.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    structured_config = load_structured_config(Path(args.structured_config).resolve())
    structured_parent = StructuredStateEncoder(structured_config)
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    structured_receipt = json.loads((structured_checkpoint / "receipt.json").read_text(encoding="utf-8"))
    if int(structured_receipt.get("step", -1)) != STRUCTURED_PARENT_STEP:
        raise SystemExit("relation repair must use ratified structured step80")
    structured_state = load_file(str(structured_checkpoint / "structured_state.safetensors"), device="cpu")
    structured_parent.load_state_dict(structured_state, strict=True)
    for parameter in structured_parent.parameters():
        parameter.requires_grad = False

    repair_rows = read_jsonl(repair_path)
    print("relation_repair_semantic_cache_start=true", flush=True)
    repair_payload = build_repair_cache(
        repair_rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured_parent=structured_parent,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    repair_cache_path = output_root / "repair_semantic_cache.pt"
    torch.save(repair_payload, repair_cache_path)
    print(f"relation_repair_semantic_cache_complete=true path={repair_cache_path}", flush=True)
    del semantic_model, semantic_state, structured_parent, structured_state
    torch.cuda.empty_cache()

    replay_cache_path = Path(args.replay_cache).resolve()
    replay_payload = torch.load(replay_cache_path, map_location="cpu")
    replay_cache_hash = sha256_file(replay_cache_path)

    adapter = EvidenceViewAdapter(expanded_adapter_config()).to(device)
    parent_adapter_path = Path(args.parent_adapter).resolve()
    parent_adapter_state = load_file(str(parent_adapter_path), device="cpu")
    adapter.load_state_dict(parent_adapter_state, strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()
    parent_adapter_hash = sha256_file(parent_adapter_path)

    graph = DualEndpointEvidenceGraphEncoder(expanded_graph_config()).to(device)
    parent_graph_path = Path(args.parent_graph).resolve()
    parent_graph_state = load_file(str(parent_graph_path), device="cpu")
    incompat = graph.load_state_dict(parent_graph_state, strict=False)
    if incompat.unexpected_keys:
        raise SystemExit(f"unexpected parent graph keys: {incompat.unexpected_keys}")
    if not incompat.missing_keys or not all(
        key.startswith("source_relation_pool_mlp.") for key in incompat.missing_keys
    ):
        raise SystemExit(f"unexpected missing graph keys: {incompat.missing_keys}")
    parent_graph_hash = sha256_file(parent_graph_path)
    trainable = set_relation_read_trainable(graph)
    if not trainable:
        raise SystemExit("relation repair has no trainable graph parameters")

    train_pairs = PairDataset(repair_payload, pair_indices(repair_payload, "train"))
    dev_pairs = PairDataset(repair_payload, pair_indices(repair_payload, "dev"))
    replay_train_indices = [i for i, split in enumerate(replay_payload["splits"]) if split == "train"]
    replay_dev_indices = [i for i, split in enumerate(replay_payload["splits"]) if split == "dev"]
    replay_train = RowDataset(replay_payload, replay_train_indices)
    replay_dev = RowDataset(replay_payload, replay_dev_indices)

    initial_repair = evaluate_pairs(
        graph=graph, adapter=adapter, dataset=dev_pairs, payload=repair_payload,
        device=device, batch_size=args.eval_batch_size,
    )
    initial_replay = evaluate_replay(
        graph=graph, adapter=adapter, dataset=replay_dev, payload=replay_payload,
        device=device, batch_size=args.eval_batch_size,
    )
    print("initial_repair=" + json.dumps(initial_repair, sort_keys=True), flush=True)
    print("initial_replay=" + json.dumps(initial_replay, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-6)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pair_loader = DataLoader(
        train_pairs, batch_size=args.train_pair_batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0, drop_last=False,
    )
    replay_loader = DataLoader(
        replay_train, batch_size=args.replay_batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1), num_workers=0, drop_last=False,
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

        pair_batch = to_device(pair_batch, device)
        replay_batch = to_device(replay_batch, device)
        a = unprefix(pair_batch, "a")
        b = unprefix(pair_batch, "b")

        graph.train()
        adapter.eval()
        optimizer.zero_grad(set_to_none=True)
        a_out = graph_forward(graph, adapter, a)
        b_out = graph_forward(graph, adapter, b)
        a_result, _ = row_objective(
            graph=graph, adapter=adapter, batch=a,
            paired_counterfactual=b_out["pooled_state"],
        )
        b_result, _ = row_objective(
            graph=graph, adapter=adapter, batch=b,
            paired_counterfactual=a_out["pooled_state"],
        )
        flip = pair_flip_loss(a_out, b_out, a, b)
        replay_result, _ = row_objective(
            graph=graph, adapter=adapter, batch=replay_batch,
            paired_counterfactual=None,
        )
        repair_loss = 0.5 * (a_result["loss"] + b_result["loss"])
        total = 0.65 * repair_loss + 0.20 * flip + 0.15 * replay_result["loss"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            print(json.dumps({
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "loss": float(total.detach().cpu()),
                "repair_loss": float(repair_loss.detach().cpu()),
                "pair_flip_loss": float(flip.detach().cpu()),
                "replay_loss": float(replay_result["loss"].detach().cpu()),
            }, sort_keys=True), flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            repair_metrics = evaluate_pairs(
                graph=graph, adapter=adapter, dataset=dev_pairs, payload=repair_payload,
                device=device, batch_size=args.eval_batch_size,
            )
            replay_metrics = evaluate_replay(
                graph=graph, adapter=adapter, dataset=replay_dev, payload=replay_payload,
                device=device, batch_size=args.eval_batch_size,
            )
            receipt = save_checkpoint(
                graph=graph, output_root=output_root, step=step,
                repair_metrics=repair_metrics, replay_metrics=replay_metrics,
                parent_adapter_hash=parent_adapter_hash, parent_graph_hash=parent_graph_hash,
                repair_curriculum_hash=repair_hash, replay_cache_hash=replay_cache_hash,
            )
            receipts[f"step-{step:08d}"] = receipt
            print("checkpoint=" + json.dumps(receipt, sort_keys=True), flush=True)

    initial_replay_macro = float(initial_replay["family_macro_target_support_mass"])
    initial_replay_min = float(initial_replay["family_min_target_support_mass"])
    def score(receipt: dict[str, Any]) -> tuple[float, ...]:
        repair = receipt["repair_dev_metrics"]
        replay = receipt["ordinary_replay_dev_metrics"]
        replay_ok = (
            float(replay["family_macro_target_support_mass"]) >= initial_replay_macro - 0.02
            and float(replay["family_min_target_support_mass"]) >= initial_replay_min - 0.03
        )
        return (
            1.0 if replay_ok else 0.0,
            float(repair["family_min_pair_flip_accuracy"]),
            float(repair["pair_flip_accuracy"]),
            float(repair["mean_pair_flip_margin"]),
            float(replay["family_macro_target_support_mass"]),
            -int(receipt["step"]),
        )

    winner_key = max(receipts, key=lambda key: score(receipts[key]))
    winner = receipts[winner_key]
    comparison = {
        "schema": "alice.eipm.n0.v02-relation-repair-comparison.v0.1",
        "status": "REPAIR_EVIDENCE_READY_NOT_RATIFIED",
        "parent_specialist": "specialized_expanded_640x3_graph512x2/step-00000080",
        "architecture_change": "dual_endpoint_query_conditioned_relation_read",
        "adapter_frozen": True,
        "shared_structured_parent_mutated": False,
        "semantic_core_mutated": False,
        "frozen_relation_essential_challenge_used_for_training": False,
        "hard_parameter_ceiling": None,
        "initial_repair_dev": initial_repair,
        "initial_ordinary_replay_dev": initial_replay,
        "checkpoints": receipts,
        "provisional_winner": winner_key,
        "selection_policy": "ordinary_replay_floor_then_worst_family_pair_flip_then_pair_flip_then_margin_then_replay_then_earlier_checkpoint",
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": "evaluate provisional winner on untouched frozen relation-essential challenge before ratification",
    }
    (output_root / "relation_repair_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
