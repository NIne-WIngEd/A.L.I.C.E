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
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph import relation_type_id
from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_graph_multilayer_interface import (
    RelationConditionedMultiLayerEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.relation_conditioned_multilayer_query import (
    RelationConditionedMultiLayerQueryInterface,
)
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr


PREP_SCHEMA = (
    "alice.eipm.n0.v02-relation-conditioned-multilayer-causal-study-preparation.v0.1"
)
CAL_SCHEMA = "alice.eipm.n0.v02-multilayer-preservation-calibration.v0.1"
MANIFEST_SCHEMA = (
    "alice.eipm.n0.v02-relation-conditioned-multilayer-interface-curriculum.v0.1"
)
CHECKPOINT_SCHEMA = (
    "alice.eipm.n0.v02-relation-conditioned-multilayer-interface-checkpoint.v0.1"
)
RESULT_SCHEMA = (
    "alice.eipm.n0.v02-relation-conditioned-multilayer-interface-result.v0.1"
)

EXPECTED_HIDDEN_STATES = 17
TARGET_RELATIONS = ("causes", "supports")

BASE_TENSOR_KEYS = (
    "field_semantic",
    "valid_mask",
    "target_distribution",
    "edge_index",
    "edge_type_ids",
    "edge_confidence",
    "edge_valid_mask",
    "query_semantic",
    "parent_field_states",
    "parent_field_weights",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def tensor_state_sha256(
    module: torch.nn.Module,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    h = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        tensor = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


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


def pooled(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def encode_queries(
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    pooled_chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        outputs = model.backbone(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        states = outputs.hidden_states
        if states is None or len(states) != EXPECTED_HIDDEN_STATES:
            raise SystemExit(
                f"semantic hidden-state depth drift: "
                f"{0 if states is None else len(states)}"
            )
        stack = torch.stack(states, dim=1)
        hidden_chunks.append(stack.detach().to(torch.float16).cpu())
        mask_chunks.append(mask.detach().bool().cpu())
        pooled_chunks.append(
            pooled(states[-1], mask).detach().float().cpu()
        )
    return (
        torch.cat(hidden_chunks, dim=0),
        torch.cat(mask_chunks, dim=0),
        torch.cat(pooled_chunks, dim=0),
    )


@torch.inference_mode()
def encode_pooled_texts(
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        values = model.encode(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )
        out.append(values.detach().float().cpu())
    return torch.cat(out, dim=0)


def compile_causal_payload(
    rows: list[dict[str, Any]],
    *,
    semantic: AliceN0V02Model,
    tokenizer: Any,
    structured: StructuredStateEncoder,
    device: torch.device,
    encode_batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if str(row["split"]) in {"train", "dev"}
    ]
    if len(selected) != 576:
        raise SystemExit(f"causal train/dev row count drift: {len(selected)}")

    query_hidden, query_mask, query_semantic = encode_queries(
        semantic,
        tokenizer,
        [str(row["query_text"]) for row in selected],
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )

    flat_fields = [
        str(field["text"])
        for row in selected
        for field in row["fields"]
    ]
    field_vectors = encode_pooled_texts(
        semantic,
        tokenizer,
        flat_fields,
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )

    n = len(selected)
    max_fields = max(len(row["fields"]) for row in selected)
    max_edges = max(len(row["relations"]) for row in selected)
    semantic_size = int(field_vectors.size(-1))

    field_semantic = torch.zeros(n, max_fields, semantic_size)
    field_type_ids = torch.ones(n, max_fields, dtype=torch.long)
    provenance_ids = torch.ones(n, max_fields, dtype=torch.long)
    relation_role_ids = torch.ones(n, max_fields, dtype=torch.long)
    temporal_scope_ids = torch.ones(n, max_fields, dtype=torch.long)
    confidence = torch.zeros(n, max_fields, 1)
    missing_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    valid_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    target_distribution = torch.zeros(n, max_fields)
    edge_index = torch.zeros(n, max_edges, 2, dtype=torch.long)
    edge_type_ids = torch.zeros(n, max_edges, dtype=torch.long)
    edge_confidence = torch.zeros(n, max_edges, 1)
    edge_valid_mask = torch.zeros(n, max_edges, dtype=torch.bool)

    cursor = 0
    for i, row in enumerate(selected):
        fields = row["fields"]
        count = len(fields)
        field_semantic[i, :count] = field_vectors[cursor : cursor + count]
        cursor += count
        valid_mask[i, :count] = True
        confidence[i, :count, 0] = torch.tensor(
            [float(item.get("confidence", 1.0)) for item in fields]
        )
        target_distribution[i, :count] = torch.tensor(
            row["target_evidence_distribution"],
            dtype=torch.float32,
        )
        for j, relation in enumerate(row["relations"]):
            edge_index[i, j, 0] = int(relation["source"])
            edge_index[i, j, 1] = int(relation["target"])
            edge_type_ids[i, j] = relation_type_id(str(relation["relation"]))
            edge_confidence[i, j, 0] = float(relation.get("confidence", 1.0))
            edge_valid_mask[i, j] = True

    structured = structured.to(device).eval()
    parent_field_states: list[torch.Tensor] = []
    parent_field_weights: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, n, 128):
            end = start + 128
            out = structured(
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
        "query_semantic": query_semantic,
        "query_hidden_states": query_hidden,
        "query_attention_mask": query_mask,
        "parent_field_states": torch.cat(parent_field_states, dim=0),
        "parent_field_weights": torch.cat(parent_field_weights, dim=0),
        "ids": [str(row["id"]) for row in selected],
        "quad_ids": [str(row["quad_id"]) for row in selected],
        "edge_direction_variants": [
            str(row["edge_direction_variant"]) for row in selected
        ],
        "query_roles": [str(row["query_role"]) for row in selected],
        "relations": [str(row["relation"]) for row in selected],
        "splits": [str(row["split"]) for row in selected],
    }


def attach_query_hidden_states(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    indices: list[int],
    semantic: AliceN0V02Model,
    tokenizer: Any,
    device: torch.device,
    encode_batch_size: int,
    max_length: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    by_id = {str(row["id"]): row for row in rows}
    texts: list[str] = []
    ordered_indices: list[int] = []
    for index in indices:
        row_id = str(payload["ids"][index])
        row = by_id.get(row_id)
        if row is None:
            raise SystemExit(f"preservation source row missing: {row_id}")
        if str(row["split"]) != str(payload["splits"][index]):
            raise SystemExit(f"preservation source split drift: {row_id}")
        texts.append(str(row["query_text"]))
        ordered_indices.append(index)

    hidden, mask, query_semantic = encode_queries(
        semantic,
        tokenizer,
        texts,
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    cached = payload["query_semantic"][ordered_indices].float()
    max_delta = float((query_semantic - cached).abs().max().item())
    if max_delta > 1e-4:
        raise SystemExit(
            f"preservation query text/cache semantic mismatch: max_delta={max_delta}"
        )
    return {
        index: (hidden[pos], mask[pos])
        for pos, index in enumerate(ordered_indices)
    }


class RowHiddenDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, Any],
        indices: list[int],
        hidden_lookup: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.payload = payload
        self.indices = indices
        self.hidden_lookup = hidden_lookup

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        out = {key: self.payload[key][index] for key in BASE_TENSOR_KEYS}
        if "query_hidden_states" in self.payload:
            out["query_hidden_states"] = self.payload["query_hidden_states"][index]
            out["query_attention_mask"] = self.payload["query_attention_mask"][index]
        elif self.hidden_lookup is not None:
            hidden, mask = self.hidden_lookup[index]
            out["query_hidden_states"] = hidden
            out["query_attention_mask"] = mask
        else:
            raise RuntimeError("hidden query state unavailable")
        out["global_index"] = index
        return out


class QuadDataset(Dataset):
    ORDER = (("A", "source"), ("A", "target"), ("B", "source"), ("B", "target"))

    def __init__(self, payload: dict[str, Any], split: str) -> None:
        grouped: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
        for index, row_split in enumerate(payload["splits"]):
            if str(row_split) != split:
                continue
            key = (
                str(payload["edge_direction_variants"][index]),
                str(payload["query_roles"][index]),
            )
            grouped[str(payload["quad_ids"][index])][key] = index
        self.payload = payload
        self.quads: list[list[int]] = []
        for quad_id in sorted(grouped):
            members = grouped[quad_id]
            if set(members) != set(self.ORDER):
                raise SystemExit(f"incomplete causal quad in payload: {quad_id}")
            self.quads.append([members[key] for key in self.ORDER])

    def __len__(self) -> int:
        return len(self.quads)

    def __getitem__(self, item: int) -> dict[str, Any]:
        indices = self.quads[item]
        out = {
            key: torch.stack([self.payload[key][i] for i in indices], dim=0)
            for key in BASE_TENSOR_KEYS
        }
        out["query_hidden_states"] = torch.stack(
            [self.payload["query_hidden_states"][i] for i in indices], dim=0
        )
        out["query_attention_mask"] = torch.stack(
            [self.payload["query_attention_mask"][i] for i in indices], dim=0
        )
        out["global_indices"] = torch.tensor(indices, dtype=torch.long)
        return out


def flatten_quad_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "global_indices":
            continue
        if not torch.is_tensor(value):
            continue
        out[key] = value.flatten(0, 1)
    return out


def to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def adapter_forward(
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return adapter(
        parent_field_states=batch["parent_field_states"],
        valid_mask=batch["valid_mask"],
        query_semantic=batch["query_semantic"],
        parent_field_weights=batch["parent_field_weights"],
    )


def candidate_forward(
    graph: RelationConditionedMultiLayerEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapted = adapter_forward(adapter, batch)
    hidden = batch["query_hidden_states"].float()
    hidden_tuple = tuple(hidden[:, layer] for layer in range(hidden.size(1)))
    return graph(
        field_states=adapted["field_states"],
        valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"],
        edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"],
        edge_valid_mask=batch["edge_valid_mask"],
        query_semantic=batch["query_semantic"],
        base_field_weights=adapted["field_weights"],
        query_hidden_states=hidden_tuple,
        query_attention_mask=batch["query_attention_mask"],
    )


def parent_forward(
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


def dist_ce(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(target * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()


def target_margin(
    weights: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    target_index = target.argmax(dim=-1)
    row = torch.arange(target_index.size(0), device=weights.device)
    right = weights[row, target_index]
    masked = weights.masked_fill(~valid, -1.0)
    masked[row, target_index] = -1.0
    wrong = masked.max(dim=-1).values.clamp_min(0.0)
    return right - wrong


def evaluate_causal(
    graph: RelationConditionedMultiLayerEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: QuadDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    rows = row_ok = quads = quad_ok = 0
    margins: list[float] = []
    relation_rows: dict[str, list[int]] = defaultdict(list)
    relation_quads: dict[str, list[int]] = defaultdict(list)
    relation_margins: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
            global_indices = batch["global_indices"]
            flat = to_device(flatten_quad_batch(batch), device)
            out = candidate_forward(graph, adapter, flat)
            target = flat["target_distribution"].argmax(dim=-1)
            pred = out["field_weights"].argmax(dim=-1)
            ok = pred.eq(target)
            margin = target_margin(
                out["field_weights"],
                flat["target_distribution"],
                flat["valid_mask"],
            )
            batch_quads = global_indices.size(0)
            ok4 = ok.reshape(batch_quads, 4)
            margin4 = margin.reshape(batch_quads, 4)
            qok = ok4.all(dim=1)

            rows += int(ok.numel())
            row_ok += int(ok.sum().item())
            quads += int(qok.numel())
            quad_ok += int(qok.sum().item())
            margins.extend(float(x) for x in margin.cpu().tolist())

            for local in range(batch_quads):
                first_index = int(global_indices[local, 0].item())
                relation = str(payload["relations"][first_index])
                relation_quads[relation].append(int(qok[local].item()))
                relation_rows[relation].extend(
                    int(x) for x in ok4[local].cpu().tolist()
                )
                relation_margins[relation].extend(
                    float(x) for x in margin4[local].cpu().tolist()
                )

    family_row = {
        key: sum(values) / len(values)
        for key, values in sorted(relation_rows.items())
    }
    family_quad = {
        key: sum(values) / len(values)
        for key, values in sorted(relation_quads.items())
    }
    family_margin = {
        key: sum(values) / len(values)
        for key, values in sorted(relation_margins.items())
    }
    return {
        "rows": rows,
        "row_accuracy": row_ok / max(rows, 1),
        "quads": quads,
        "quad_accuracy": quad_ok / max(quads, 1),
        "family_row_accuracy": family_row,
        "family_min_row_accuracy": min(family_row.values()),
        "family_quad_accuracy": family_quad,
        "family_min_quad_accuracy": min(family_quad.values()),
        "mean_target_margin": sum(margins) / max(len(margins), 1),
        "family_mean_target_margin": family_margin,
    }


def evaluate_ordinary(
    graph: RelationConditionedMultiLayerEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: RowHiddenDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    mass_by_family: dict[str, list[float]] = defaultdict(list)
    top1_by_family: dict[str, list[int]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            global_index = batch["global_index"]
            batch = to_device(batch, device)
            out = candidate_forward(graph, adapter, batch)
            support = batch["target_distribution"] > 0
            mass = (
                out["field_weights"] * support.to(out["field_weights"].dtype)
            ).sum(dim=-1)
            pred = out["field_weights"].argmax(dim=-1)
            row = torch.arange(pred.size(0), device=device)
            top1 = support[row, pred]
            for local, index in enumerate(global_index.tolist()):
                family = str(payload["families"][int(index)])
                mass_by_family[family].append(float(mass[local].item()))
                top1_by_family[family].append(int(top1[local].item()))
    mass = {
        key: sum(values) / len(values)
        for key, values in sorted(mass_by_family.items())
    }
    top1 = {
        key: sum(values) / len(values)
        for key, values in sorted(top1_by_family.items())
    }
    return {
        "family_macro_target_support_mass": sum(mass.values()) / len(mass),
        "family_min_target_support_mass": min(mass.values()),
        "family_target_support_mass": mass,
        "family_macro_top1_support_accuracy": sum(top1.values()) / len(top1),
        "family_min_top1_support_accuracy": min(top1.values()),
        "family_top1_support_accuracy": top1,
    }


def evaluate_endpoint(
    graph: RelationConditionedMultiLayerEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: Dataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    pair_total = pair_ok_total = row_total = row_ok_total = 0
    margins: list[float] = []
    family_ok: dict[str, list[int]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
            a_global = batch["a_global_index"]
            a = {
                key[2:]: value
                for key, value in batch.items()
                if key.startswith("a_") and key != "a_global_index"
            }
            b = {
                key[2:]: value
                for key, value in batch.items()
                if key.startswith("b_") and key != "b_global_index"
            }
            a = to_device(a, device)
            b = to_device(b, device)
            ao = candidate_forward(graph, adapter, a)
            bo = candidate_forward(graph, adapter, b)
            at = a["target_distribution"].argmax(dim=-1)
            bt = b["target_distribution"].argmax(dim=-1)
            aok = ao["field_weights"].argmax(dim=-1).eq(at)
            bok = bo["field_weights"].argmax(dim=-1).eq(bt)
            pok = aok & bok
            am = target_margin(
                ao["field_weights"], a["target_distribution"], a["valid_mask"]
            )
            bm = target_margin(
                bo["field_weights"], b["target_distribution"], b["valid_mask"]
            )
            pm = torch.minimum(am, bm)

            pair_total += int(pok.numel())
            pair_ok_total += int(pok.sum().item())
            row_total += int(aok.numel() + bok.numel())
            row_ok_total += int(aok.sum().item() + bok.sum().item())
            margins.extend(float(x) for x in pm.cpu().tolist())

            for local, index in enumerate(a_global.tolist()):
                family = str(payload["families"][int(index)])
                family_ok[family].append(int(pok[local].item()))

    fam = {
        key: sum(values) / len(values)
        for key, values in sorted(family_ok.items())
    }
    return {
        "pairs": pair_total,
        "pair_accuracy": pair_ok_total / max(pair_total, 1),
        "family_pair_accuracy": fam,
        "family_min_pair_accuracy": min(fam.values()),
        "row_accuracy": row_ok_total / max(row_total, 1),
        "mean_graph_target_margin": sum(margins) / max(len(margins), 1),
    }


class PairHiddenDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, Any],
        pairs: list[tuple[int, int]],
        hidden_lookup: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.payload = payload
        self.pairs = pairs
        self.hidden_lookup = hidden_lookup

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        a_index, b_index = self.pairs[item]
        out: dict[str, Any] = {}
        for prefix, index in (("a", a_index), ("b", b_index)):
            for key in BASE_TENSOR_KEYS:
                out[f"{prefix}_{key}"] = self.payload[key][index]
            hidden, mask = self.hidden_lookup[index]
            out[f"{prefix}_query_hidden_states"] = hidden
            out[f"{prefix}_query_attention_mask"] = mask
            out[f"{prefix}_global_index"] = index
        return out


def preservation_pass(
    ordinary: dict[str, Any],
    endpoint: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    candidate = {"ordinary": ordinary, "endpoint": endpoint}
    checks: list[dict[str, Any]] = []
    passed = True
    for item in policy["metrics"]:
        path = str(item["path"])
        current: Any = candidate
        for token in path.split("."):
            current = current[token]
        value = float(current)
        floor = float(item["baseline_mean"]) - float(item["atol"])
        ok = value >= floor
        checks.append(
            {
                "path": path,
                "candidate": value,
                "required_minimum": floor,
                "pass": ok,
            }
        )
        passed = passed and ok
    return passed, checks


def causal_ready(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    no_family_regression = all(
        float(current["family_row_accuracy"][relation])
        + 1e-12
        >= float(baseline["family_row_accuracy"][relation])
        for relation in baseline["family_row_accuracy"]
    )
    target_improvement = all(
        float(current["family_row_accuracy"][relation])
        > float(baseline["family_row_accuracy"][relation]) + 1e-12
        for relation in TARGET_RELATIONS
    )
    quad_improvement = (
        float(current["quad_accuracy"])
        > float(baseline["quad_accuracy"]) + 1e-12
    )
    worst_family_nonregression = (
        float(current["family_min_quad_accuracy"])
        + 1e-12
        >= float(baseline["family_min_quad_accuracy"])
    )
    margin_improvement = (
        float(current["mean_target_margin"])
        > float(baseline["mean_target_margin"]) + 1e-12
    )
    detail = {
        "no_relation_family_row_accuracy_regression": no_family_regression,
        "causes_and_supports_both_strictly_improve": target_improvement,
        "overall_quad_accuracy_strictly_improves": quad_improvement,
        "worst_family_quad_accuracy_not_worse": worst_family_nonregression,
        "mean_target_margin_strictly_improves": margin_improvement,
    }
    return all(detail.values()), detail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--preparation-receipt", required=True)
    p.add_argument("--calibration-receipt", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--curriculum-manifest", required=True)
    p.add_argument("--ordinary-source-curriculum", required=True)
    p.add_argument("--endpoint-source-curriculum", required=True)
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
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=32)
    p.add_argument("--quad-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--margin-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260919)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("multi-layer interface experiment requires one CUDA device")
    device = torch.device("cuda")
    seed_all(args.seed)

    root = Path(args.repo_root).resolve()
    prep_path = Path(args.preparation_receipt).resolve()
    cal_path = Path(args.calibration_receipt).resolve()
    map_path = Path(args.layer_map).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    ordinary_source = Path(args.ordinary_source_curriculum).resolve()
    endpoint_source = Path(args.endpoint_source_curriculum).resolve()
    semantic_config = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    structured_config = Path(args.structured_config).resolve()
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    ordinary_replay_path = Path(args.ordinary_replay_cache).resolve()
    endpoint_replay_path = Path(args.endpoint_replay_cache).resolve()
    output = Path(args.output_dir).resolve()

    for path in (
        prep_path, cal_path, map_path, curriculum_path, manifest_path,
        ordinary_source, endpoint_source, semantic_config,
        semantic_checkpoint / "alice_n0_v02.safetensors",
        tokenizer_dir / "tokenizer.json",
        structured_config,
        structured_checkpoint / "structured_state.safetensors",
        adapter_path, graph_path, ordinary_replay_path, endpoint_replay_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing multilayer experiment input: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite multilayer output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if prep.get("schema") != PREP_SCHEMA or prep.get("status") != (
        "PASS_CAUSAL_STUDY_INPUTS_FROZEN_TRAINING_DECISION_STILL_REQUIRED"
    ):
        raise SystemExit("causal preparation contract mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("causal curriculum manifest mismatch")
    if manifest.get("compiled_sha256") != sha(curriculum_path):
        raise SystemExit("causal curriculum hash drift")
    if manifest.get("test_split_opening_authorized") is not False:
        raise SystemExit("test split authorization drift")

    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal.get("schema") != CAL_SCHEMA or cal.get("status") != (
        "PASS_PARENT_ONLY_PRESERVATION_CALIBRATION"
    ):
        raise SystemExit("preservation calibration did not pass")
    if cal.get("candidate_result_observed") is not False:
        raise SystemExit("candidate contaminated preservation calibration")
    if cal.get("interface_training_gate_satisfied") is not True:
        raise SystemExit("interface training gate is not satisfied")
    if cal.get("authorization_scope") != (
        "one bounded relation-conditioned multilayer interface experiment only"
    ):
        raise SystemExit("training authorization scope drift")
    if cal.get("scale_authorized") is not False:
        raise SystemExit("scale authorization drift")
    if cal.get("heldout_opening_authorized") is not False:
        raise SystemExit("heldout authorization drift")

    expected = prep["artifact_sha256"]
    for key, path in {
        "curriculum": curriculum_path,
        "curriculum_manifest": manifest_path,
        "layer_map": map_path,
        "semantic_config": semantic_config,
        "semantic_checkpoint": semantic_checkpoint / "alice_n0_v02.safetensors",
        "structured_config": structured_config,
        "structured_checkpoint": structured_checkpoint / "structured_state.safetensors",
        "parent_adapter": adapter_path,
        "parent_graph": graph_path,
        "ordinary_replay_cache": ordinary_replay_path,
        "endpoint_replay_cache": endpoint_replay_path,
    }.items():
        if expected.get(key) != sha(path):
            raise SystemExit(f"frozen preparation artifact drift: {key}")

    semantic_cfg = load_n0_config(semantic_config)
    semantic = AliceN0V02Model(semantic_cfg)
    state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"),
        device="cpu",
    )
    missing, unexpected = semantic.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint mismatch missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    structured_cfg = rr.load_structured_config(structured_config)
    structured = StructuredStateEncoder(structured_cfg)
    structured.load_state_dict(
        load_file(
            str(structured_checkpoint / "structured_state.safetensors"),
            device="cpu",
        ),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    rows = read_jsonl(curriculum_path)
    print("multilayer_causal_cache_start=true", flush=True)
    causal = compile_causal_payload(
        rows,
        semantic=semantic,
        tokenizer=tokenizer,
        structured=structured,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    cache_path = output / "causal_train_dev_hidden_cache.pt"
    torch.save(causal, cache_path)
    print(f"multilayer_causal_cache_complete=true path={cache_path}", flush=True)

    ordinary_payload = torch.load(ordinary_replay_path, map_location="cpu")
    endpoint_payload = torch.load(endpoint_replay_path, map_location="cpu")
    ordinary_dev_indices = [
        i for i, split in enumerate(ordinary_payload["splits"])
        if str(split) == "dev"
    ]
    endpoint_dev_pairs = rr.pair_indices(endpoint_payload, "dev")
    endpoint_dev_indices = sorted(
        {i for pair in endpoint_dev_pairs for i in pair}
    )

    ordinary_hidden = attach_query_hidden_states(
        ordinary_payload,
        read_jsonl(ordinary_source),
        indices=ordinary_dev_indices,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    endpoint_hidden = attach_query_hidden_states(
        endpoint_payload,
        read_jsonl(endpoint_source),
        indices=endpoint_dev_indices,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )

    del semantic, state, structured
    torch.cuda.empty_cache()

    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()

    parent_state = load_file(str(graph_path), device="cpu")
    baseline_graph = DualEndpointEvidenceGraphEncoder(
        rr.expanded_graph_config()
    ).to(device)
    baseline_graph.load_state_dict(parent_state, strict=True)
    for parameter in baseline_graph.parameters():
        parameter.requires_grad = False
    baseline_graph.eval()

    interface = RelationConditionedMultiLayerQueryInterface.from_layer_map_file(
        layer_map_path=map_path,
        semantic_size=rr.expanded_graph_config().semantic_size,
        graph_size=rr.expanded_graph_config().graph_size,
        num_relation_types=rr.expanded_graph_config().num_relation_types,
        num_hidden_states=EXPECTED_HIDDEN_STATES,
        interface_size=rr.expanded_graph_config().graph_size,
        dropout=0.0,
    )
    graph = RelationConditionedMultiLayerEvidenceGraphEncoder(
        config=rr.expanded_graph_config(),
        query_interface=interface,
    ).to(device)
    missing, unexpected = graph.load_state_dict(parent_state, strict=False)
    allowed_missing = [
        name for name in missing
        if name.startswith(("query_interface.", "interface_endpoint_read."))
    ]
    if len(allowed_missing) != len(missing) or unexpected:
        raise SystemExit(
            f"candidate parent load drift missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )
    trainable = graph.freeze_parent_for_interface_training()
    parent_hash_before = tensor_state_sha256(
        graph,
        exclude_prefixes=("query_interface.", "interface_endpoint_read."),
    )

    train_quads = QuadDataset(causal, "train")
    dev_quads = QuadDataset(causal, "dev")
    if len(train_quads) != 108 or len(dev_quads) != 36:
        raise SystemExit(
            f"causal quad split drift train={len(train_quads)} dev={len(dev_quads)}"
        )

    ordinary_dev = RowHiddenDataset(
        ordinary_payload, ordinary_dev_indices, ordinary_hidden
    )
    endpoint_dev = PairHiddenDataset(
        endpoint_payload, endpoint_dev_pairs, endpoint_hidden
    )

    baseline_dev = evaluate_causal(
        graph, adapter, dev_quads, causal, device, args.eval_batch_size
    )

    # Exact-parent initialization is a model invariant, not a tolerance.
    first_dev = next(iter(DataLoader(dev_quads, batch_size=1, shuffle=False)))
    flat = to_device(flatten_quad_batch(first_dev), device)
    with torch.inference_mode():
        parent_out = parent_forward(baseline_graph, adapter, flat)
        candidate_out = candidate_forward(graph, adapter, flat)
    if not torch.equal(parent_out["field_weights"], candidate_out["field_weights"]):
        diff = float(
            (parent_out["field_weights"] - candidate_out["field_weights"])
            .abs().max().item()
        )
        raise SystemExit(f"zero-init candidate is not exact parent: diff={diff}")
    if not torch.equal(parent_out["pooled_state"], candidate_out["pooled_state"]):
        diff = float(
            (parent_out["pooled_state"] - candidate_out["pooled_state"])
            .abs().max().item()
        )
        raise SystemExit(f"zero-init pooled state is not exact parent: diff={diff}")

    initial_ordinary = evaluate_ordinary(
        graph, adapter, ordinary_dev, ordinary_payload, device, args.eval_batch_size
    )
    initial_endpoint = evaluate_endpoint(
        graph, adapter, endpoint_dev, endpoint_payload, device, args.eval_batch_size
    )
    initial_preserved, initial_checks = preservation_pass(
        initial_ordinary, initial_endpoint, cal["policy"]
    )
    if not initial_preserved:
        raise SystemExit(
            "zero-init candidate failed frozen parent preservation policy: "
            + json.dumps(initial_checks, sort_keys=True)
        )

    print("baseline_causal_dev=" + json.dumps(baseline_dev, sort_keys=True), flush=True)
    print("initial_preservation=" + json.dumps(
        {"ordinary": initial_ordinary, "endpoint": initial_endpoint},
        sort_keys=True,
    ), flush=True)

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

    loader = DataLoader(
        train_quads,
        batch_size=args.quad_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    iterator = iter(loader)
    receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, args.max_steps + 1):
        try:
            quad_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            quad_batch = next(iterator)

        flat = to_device(flatten_quad_batch(quad_batch), device)
        graph.train()
        adapter.eval()
        optimizer.zero_grad(set_to_none=True)

        out = candidate_forward(graph, adapter, flat)
        ce = dist_ce(out["field_weights"], flat["target_distribution"])
        margins = target_margin(
            out["field_weights"],
            flat["target_distribution"],
            flat["valid_mask"],
        )
        margin_loss = F.relu(args.margin - margins).mean()
        total = ce + args.margin_weight * margin_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            print(json.dumps({
                "step": step,
                "loss": float(total.detach().cpu()),
                "selection_ce": float(ce.detach().cpu()),
                "margin_loss": float(margin_loss.detach().cpu()),
                "lr": scheduler.get_last_lr()[0],
            }, sort_keys=True), flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            dev = evaluate_causal(
                graph, adapter, dev_quads, causal, device, args.eval_batch_size
            )
            ordinary = evaluate_ordinary(
                graph, adapter, ordinary_dev, ordinary_payload,
                device, args.eval_batch_size
            )
            endpoint = evaluate_endpoint(
                graph, adapter, endpoint_dev, endpoint_payload,
                device, args.eval_batch_size
            )
            preserved, preservation_checks = preservation_pass(
                ordinary, endpoint, cal["policy"]
            )
            causal_ok, causal_checks = causal_ready(dev, baseline_dev)

            cp = output / f"step-{step:08d}"
            cp.mkdir(parents=True, exist_ok=True)
            candidate_path = cp / "multilayer_interface.safetensors"
            interface_state = {
                name: value.detach().cpu().contiguous()
                for name, value in graph.state_dict().items()
                if name.startswith(("query_interface.", "interface_endpoint_read."))
            }
            save_file(interface_state, str(candidate_path))
            receipt = {
                "schema": CHECKPOINT_SCHEMA,
                "status": "TRAINED_NOT_RATIFIED",
                "step": step,
                "git_revision": git_revision(root),
                "candidate_sha256": sha(candidate_path),
                "parent_graph_sha256": sha(graph_path),
                "parent_adapter_sha256": sha(adapter_path),
                "preparation_receipt_sha256": sha(prep_path),
                "calibration_receipt_sha256": sha(cal_path),
                "curriculum_sha256": sha(curriculum_path),
                "layer_map_sha256": sha(map_path),
                "trainable_scope": (
                    "relation_conditioned_multilayer_query_interface_and_"
                    "zero_init_signed_endpoint_read_only"
                ),
                "trainable_parameters": sum(p.numel() for p in trainable),
                "parent_parameters_exactly_unchanged": (
                    tensor_state_sha256(
                        graph,
                        exclude_prefixes=(
                            "query_interface.",
                            "interface_endpoint_read.",
                        ),
                    )
                    == parent_hash_before
                ),
                "causal_dev_metrics": dev,
                "causal_dev_checks": causal_checks,
                "causal_dev_ready": causal_ok,
                "ordinary_preservation_metrics": ordinary,
                "endpoint_preservation_metrics": endpoint,
                "preservation_checks": preservation_checks,
                "preservation_pass": preserved,
                "eligible_for_heldout_decision": preserved and causal_ok,
                "test_split_evaluated": False,
                "frozen_challenge_evaluated": False,
                "preservation_rows_used_for_gradient": False,
                "semantic_parent_mutated": False,
                "structured_parent_mutated": False,
                "adapter_mutated": False,
                "parent_graph_mutated": False,
                "scale_authorized": False,
                "private_identity_gradient": False,
                "production_promotion_authorized": False,
                "n0_complete": False,
            }
            if receipt["parent_parameters_exactly_unchanged"] is not True:
                raise SystemExit("frozen parent parameters changed during interface training")
            (cp / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipts[cp.name] = receipt
            print(
                "multilayer_checkpoint="
                + json.dumps(receipt, sort_keys=True),
                flush=True,
            )

    eligible = {
        key: receipt
        for key, receipt in receipts.items()
        if receipt["eligible_for_heldout_decision"]
    }

    def score(receipt: dict[str, Any]) -> tuple[float, ...]:
        dev = receipt["causal_dev_metrics"]
        targeted = min(
            float(dev["family_row_accuracy"][relation])
            for relation in TARGET_RELATIONS
        )
        return (
            float(dev["family_min_quad_accuracy"]),
            float(dev["quad_accuracy"]),
            targeted,
            float(dev["row_accuracy"]),
            float(dev["mean_target_margin"]),
            -int(receipt["step"]),
        )

    selected = max(eligible, key=lambda key: score(eligible[key])) if eligible else None
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_CAUSAL_DEV_AND_PRESERVATION_READY_FOR_SEPARATE_HELDOUT_DECISION"
            if selected is not None
            else
            "FAIL_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
        ),
        "git_revision": git_revision(root),
        "preparation_receipt_sha256": sha(prep_path),
        "calibration_receipt_sha256": sha(cal_path),
        "curriculum_sha256": sha(curriculum_path),
        "layer_map_sha256": sha(map_path),
        "parent_graph_sha256": sha(graph_path),
        "parent_adapter_sha256": sha(adapter_path),
        "ordinary_source_curriculum_sha256": sha(ordinary_source),
        "endpoint_source_curriculum_sha256": sha(endpoint_source),
        "causal_train_dev_hidden_cache_sha256": sha(cache_path),
        "baseline_causal_dev": baseline_dev,
        "initial_parent_preservation": {
            "ordinary": initial_ordinary,
            "endpoint": initial_endpoint,
            "checks": initial_checks,
        },
        "checkpoints": receipts,
        "eligible_checkpoint_keys": sorted(eligible),
        "selected_checkpoint": selected,
        "selected_candidate_sha256": (
            eligible[selected]["candidate_sha256"] if selected else None
        ),
        "selection_policy": (
            "preservation_first_then_worst_relation_quad_accuracy_then_overall_"
            "quad_accuracy_then_min_causes_supports_row_accuracy_then_row_accuracy_"
            "then_target_margin_then_earlier_checkpoint"
        ),
        "causal_readiness_policy": {
            "no_relation_family_row_accuracy_regression_vs_parent": True,
            "causes_and_supports_each_must_strictly_improve_vs_parent": True,
            "overall_quad_accuracy_must_strictly_improve_vs_parent": True,
            "worst_family_quad_accuracy_may_not_regress_vs_parent": True,
            "mean_target_margin_must_strictly_improve_vs_parent": True,
        },
        "preservation_policy_source": "canonical_parent_repeatability_only",
        "test_split_evaluated": False,
        "test_split_opened_after_training": False,
        "frozen_challenge_evaluated": False,
        "preservation_rows_used_for_gradient": False,
        "parent_graph_parameters_exactly_unchanged": (
            tensor_state_sha256(
                graph,
                exclude_prefixes=("query_interface.", "interface_endpoint_read."),
            )
            == parent_hash_before
        ),
        "semantic_parent_mutated": False,
        "structured_parent_mutated": False,
        "adapter_mutated": False,
        "scale_authorized": False,
        "automatic_rerun_or_hotfix_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "make_one_explicit_heldout_opening_decision_without_changing_candidate"
            if selected is not None
            else
            "reopen_language_graph_boundary_from_single_failed_causal_experiment"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
