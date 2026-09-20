#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_graph_query_edge_dual_view import (
    DualViewLateInteractionEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.query_edge_dual_view_late_interaction import (
    DualViewLateInteractionBridge,
)
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import qualify_n0_v02_dual_view_late_interaction_v0_1 as dq
import train_n0_v02_relation_repair as rr
import train_n0_v02_relation_conditioned_multilayer_interface_v0_1 as ml
import train_n0_v02_query_edge_cross_attention_bridge_v0_1 as old


QUAL_SCHEMA = "alice.eipm.n0.dual-view-late-interaction-qualification.v0.1"
ANCHOR_SCHEMA = "alice.eipm.n0.query-edge-preservation-train-anchors.v0.1"
CHECKPOINT_SCHEMA = "alice.eipm.n0.dual-view-specialist-training-checkpoint.v0.1"
RESULT_SCHEMA = "alice.eipm.n0.dual-view-specialist-training-result.v0.1"

EXPECTED_QUAL_SHA256 = (
    "714f06d4939abad2b816a7e3ffb5c5fa10112129ef0b520fd838f2d182286278"
)
EXPECTED_CAUSAL_CACHE_SHA256 = (
    "5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
)
EXPECTED_CURRICULUM_SHA256 = (
    "c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
)
EXPECTED_MANIFEST_SHA256 = (
    "0d05f1dcd7d11cc1defdd0a4112c10e3b1d7febfaabf97fe611a854d6905b99a"
)
EXPECTED_ANCHORS_SHA256 = (
    "f5b3f438f776216f6dbd557a8c785aac70af38d256e3f7cf94f0f71a43f53e94"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
)
EXPECTED_CALIBRATION_SHA256 = (
    "d485fce0e4304d9cbd8af1658ed7cfedfd1bd4d44b0f40cf56a2618d5ad2390b"
)
EXPECTED_HIDDEN_STATES = 17
SPECIAL_IDS = {0, 1, 2, 3, 4}


def content_mask(ids: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    mask = attention.bool().clone()
    for token_id in SPECIAL_IDS:
        mask &= ids.ne(token_id)
    return mask


@torch.inference_mode()
def encode_unique_final_field_tokens(
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    unique = list(dict.fromkeys(texts))
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    model.eval()
    for start in range(0, len(unique), batch_size):
        chunk = unique[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        result = model.backbone(
            input_ids=ids,
            attention_mask=attention,
            return_dict=True,
        )
        final = result.last_hidden_state.detach().to(torch.float16).cpu()
        masks = content_mask(ids.cpu(), attention.cpu())
        for pos, text in enumerate(chunk):
            out[text] = (final[pos], masks[pos])
    return out


def build_query_content_lookup(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    tokenizer: Any,
    max_length: int,
    attention_lookup: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[int, torch.Tensor]:
    by_id = {str(row["id"]): row for row in rows}
    out: dict[int, torch.Tensor] = {}
    for index in indices:
        row_id = str(payload["ids"][index])
        row = by_id.get(row_id)
        if row is None:
            raise SystemExit(f"query-content source row missing: {row_id}")
        encoded = tokenizer(
            str(row["query_text"]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"][0]
        attention = encoded["attention_mask"][0].bool()
        if "query_attention_mask" in payload:
            cached = payload["query_attention_mask"][index].bool()
            if not torch.equal(attention, cached):
                raise SystemExit(f"query attention/cache drift: {row_id}")
        elif attention_lookup is not None:
            _hidden, cached = attention_lookup[index]
            if not torch.equal(attention, cached.bool()):
                raise SystemExit(f"preservation query attention drift: {row_id}")
        out[index] = content_mask(ids, attention)
    return out


def build_field_lookup(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    semantic: AliceN0V02Model,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    by_id = {str(row["id"]): row for row in rows}
    selected: dict[int, list[str]] = {}
    texts: list[str] = []
    for index in indices:
        row_id = str(payload["ids"][index])
        row = by_id.get(row_id)
        if row is None:
            raise SystemExit(f"field-token source row missing: {row_id}")
        field_texts = [str(field["text"]) for field in row["fields"]]
        selected[index] = field_texts
        texts.extend(field_texts)

    encoded = encode_unique_final_field_tokens(
        semantic,
        tokenizer,
        texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    max_fields = int(payload["valid_mask"].size(1))
    semantic_size = int(next(iter(encoded.values()))[0].size(-1))
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index in indices:
        field_texts = selected[index]
        if len(field_texts) > max_fields:
            raise SystemExit("field-token padding width drift")
        states = torch.zeros(
            max_fields,
            max_length,
            semantic_size,
            dtype=torch.float16,
        )
        masks = torch.zeros(max_fields, max_length, dtype=torch.bool)
        for slot, text in enumerate(field_texts):
            state, mask = encoded[text]
            states[slot] = state
            masks[slot] = mask
        valid = payload["valid_mask"][index].bool()
        if int(valid.sum().item()) != len(field_texts):
            raise SystemExit(f"field valid-mask/source count drift: {payload['ids'][index]}")
        out[index] = (states, masks)
    return out


class DualViewRowDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, Any],
        indices: list[int],
        *,
        hidden_lookup: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
        query_content_lookup: dict[int, torch.Tensor],
        field_lookup: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.payload = payload
        self.indices = indices
        self.hidden_lookup = hidden_lookup
        self.query_content_lookup = query_content_lookup
        self.field_lookup = field_lookup

    def __len__(self) -> int:
        return len(self.indices)

    def row(self, index: int) -> dict[str, Any]:
        out = {key: self.payload[key][index] for key in ml.BASE_TENSOR_KEYS}
        if "query_hidden_states" in self.payload:
            out["query_hidden_states"] = self.payload["query_hidden_states"][index]
            out["query_attention_mask"] = self.payload["query_attention_mask"][index]
        elif self.hidden_lookup is not None:
            hidden, mask = self.hidden_lookup[index]
            out["query_hidden_states"] = hidden
            out["query_attention_mask"] = mask
        else:
            raise RuntimeError("query hidden state unavailable")
        out["query_content_mask"] = self.query_content_lookup[index]
        fields, field_mask = self.field_lookup[index]
        out["field_token_states"] = fields
        out["field_content_mask"] = field_mask
        out["global_index"] = index
        return out

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.row(self.indices[item])


class DualViewPairDataset(Dataset):
    def __init__(self, rows: DualViewRowDataset, pairs: list[tuple[int, int]]) -> None:
        self.rows = rows
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        a_index, b_index = self.pairs[item]
        out: dict[str, Any] = {}
        for prefix, index in (("a", a_index), ("b", b_index)):
            row = self.rows.row(index)
            for key, value in row.items():
                out[f"{prefix}_{key}"] = value
        return out


class DualViewQuadDataset(Dataset):
    ORDER = (("A", "source"), ("A", "target"), ("B", "source"), ("B", "target"))

    def __init__(
        self,
        payload: dict[str, Any],
        split: str,
        *,
        query_content_lookup: dict[int, torch.Tensor],
        field_lookup: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
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
        self.query_content_lookup = query_content_lookup
        self.field_lookup = field_lookup
        self.quads: list[list[int]] = []
        for quad_id in sorted(grouped):
            members = grouped[quad_id]
            if set(members) != set(self.ORDER):
                raise SystemExit(f"incomplete dual-view quad: {quad_id}")
            self.quads.append([members[key] for key in self.ORDER])

    def __len__(self) -> int:
        return len(self.quads)

    def __getitem__(self, item: int) -> dict[str, Any]:
        indices = self.quads[item]
        out = {
            key: torch.stack([self.payload[key][i] for i in indices], dim=0)
            for key in ml.BASE_TENSOR_KEYS
        }
        for key in (
            "query_hidden_states",
            "query_attention_mask",
            "relevant_edge_index",
            "same_relation_edge_mask",
        ):
            out[key] = torch.stack([self.payload[key][i] for i in indices], dim=0)
        out["query_content_mask"] = torch.stack(
            [self.query_content_lookup[i] for i in indices], dim=0
        )
        out["field_token_states"] = torch.stack(
            [self.field_lookup[i][0] for i in indices], dim=0
        )
        out["field_content_mask"] = torch.stack(
            [self.field_lookup[i][1] for i in indices], dim=0
        )
        out["global_indices"] = torch.tensor(indices, dtype=torch.long)
        return out


def flatten_quad_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "global_indices" or not torch.is_tensor(value):
            continue
        out[key] = value.flatten(0, 1)
    return out


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def candidate_forward(
    graph: DualViewLateInteractionEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapted = ml.adapter_forward(adapter, batch)
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
        query_content_mask=batch["query_content_mask"],
        field_token_states=batch["field_token_states"].float(),
        field_content_mask=batch["field_content_mask"],
    )


def specialist_bce(out: dict[str, torch.Tensor], target: float) -> torch.Tensor:
    targets = torch.full_like(out["specialist_logit"], float(target))
    return F.binary_cross_entropy_with_logits(out["specialist_logit"], targets)


def evaluate_causal(
    graph: DualViewLateInteractionEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: DualViewQuadDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    rows = row_ok = quads = quad_ok = 0
    specialist_ok = specialist_total = 0
    conditional_all_ok = conditional_same_ok = 0
    margins: list[float] = []
    specialist_probs: list[float] = []
    target_conditional_probs: list[float] = []
    family_rows: dict[str, list[int]] = defaultdict(list)
    family_quads: dict[str, list[int]] = defaultdict(list)
    family_specialist: dict[str, list[int]] = defaultdict(list)
    family_margins: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
            global_indices = batch["global_indices"]
            flat = to_device(flatten_quad_batch(batch), device)
            out = candidate_forward(graph, adapter, flat)
            target = flat["target_distribution"].argmax(dim=-1)
            pred = out["field_weights"].argmax(dim=-1)
            ok = pred.eq(target)
            margin = ml.target_margin(
                out["field_weights"],
                flat["target_distribution"],
                flat["valid_mask"],
            )
            specialist = out["specialist_probability"]
            specialist_row_ok = specialist.gt(0.5)
            conditional = out["conditional_edge_probability"]
            relevant = flat["relevant_edge_index"].long()
            all_edge_ok = conditional.argmax(dim=-1).eq(relevant)
            same_mask = flat["same_relation_edge_mask"].bool()
            same_scores = conditional.masked_fill(~same_mask, -1.0)
            same_edge_ok = same_scores.argmax(dim=-1).eq(relevant)
            row_index = torch.arange(relevant.size(0), device=device)
            target_conditional = conditional[row_index, relevant]

            batch_quads = global_indices.size(0)
            ok4 = ok.reshape(batch_quads, 4)
            spec4 = specialist_row_ok.reshape(batch_quads, 4)
            margin4 = margin.reshape(batch_quads, 4)
            qok = ok4.all(dim=1)

            rows += int(ok.numel())
            row_ok += int(ok.sum().item())
            quads += int(qok.numel())
            quad_ok += int(qok.sum().item())
            specialist_total += int(specialist_row_ok.numel())
            specialist_ok += int(specialist_row_ok.sum().item())
            conditional_all_ok += int(all_edge_ok.sum().item())
            conditional_same_ok += int(same_edge_ok.sum().item())
            margins.extend(float(x) for x in margin.cpu().tolist())
            specialist_probs.extend(float(x) for x in specialist.cpu().tolist())
            target_conditional_probs.extend(
                float(x) for x in target_conditional.cpu().tolist()
            )

            for local in range(batch_quads):
                first = int(global_indices[local, 0].item())
                relation = str(payload["relations"][first])
                family_quads[relation].append(int(qok[local].item()))
                family_rows[relation].extend(int(x) for x in ok4[local].cpu().tolist())
                family_specialist[relation].extend(
                    int(x) for x in spec4[local].cpu().tolist()
                )
                family_margins[relation].extend(
                    float(x) for x in margin4[local].cpu().tolist()
                )

    family_row = {
        key: sum(v) / len(v) for key, v in sorted(family_rows.items())
    }
    family_quad = {
        key: sum(v) / len(v) for key, v in sorted(family_quads.items())
    }
    family_spec = {
        key: sum(v) / len(v) for key, v in sorted(family_specialist.items())
    }
    family_margin = {
        key: sum(v) / len(v) for key, v in sorted(family_margins.items())
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
        "specialist_top1_accuracy": specialist_ok / max(specialist_total, 1),
        "family_specialist_top1_accuracy": family_spec,
        "family_min_specialist_top1_accuracy": min(family_spec.values()),
        "mean_specialist_probability": (
            sum(specialist_probs) / max(len(specialist_probs), 1)
        ),
        "conditional_edge_all_four_top1_accuracy": (
            conditional_all_ok / max(rows, 1)
        ),
        "conditional_edge_same_relation_top1_accuracy": (
            conditional_same_ok / max(rows, 1)
        ),
        "mean_target_conditional_edge_probability": (
            sum(target_conditional_probs)
            / max(len(target_conditional_probs), 1)
        ),
    }


def evaluate_ordinary(
    graph: DualViewLateInteractionEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: DualViewRowDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mass_by_family: dict[str, list[float]] = defaultdict(list)
    top1_by_family: dict[str, list[int]] = defaultdict(list)
    noops: list[float] = []
    noop_ok = total = 0
    graph.eval()
    adapter.eval()
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
            noop = out["noop_route_probability"]
            noop_top1 = noop.gt(0.5)
            noop_ok += int(noop_top1.sum().item())
            total += int(noop_top1.numel())
            noops.extend(float(x) for x in noop.cpu().tolist())
            for local, index in enumerate(global_index.tolist()):
                family = str(payload["families"][int(index)])
                mass_by_family[family].append(float(mass[local].item()))
                top1_by_family[family].append(int(top1[local].item()))
    mass = {
        key: sum(v) / len(v) for key, v in sorted(mass_by_family.items())
    }
    top1 = {
        key: sum(v) / len(v) for key, v in sorted(top1_by_family.items())
    }
    metrics = {
        "family_macro_target_support_mass": sum(mass.values()) / len(mass),
        "family_min_target_support_mass": min(mass.values()),
        "family_target_support_mass": mass,
        "family_macro_top1_support_accuracy": sum(top1.values()) / len(top1),
        "family_min_top1_support_accuracy": min(top1.values()),
        "family_top1_support_accuracy": top1,
    }
    activation = {
        "noop_top1_accuracy": noop_ok / max(total, 1),
        "mean_noop_probability": sum(noops) / max(len(noops), 1),
    }
    return metrics, activation


def evaluate_endpoint(
    graph: DualViewLateInteractionEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: DualViewPairDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pair_total = pair_ok_total = row_total = row_ok_total = 0
    margins: list[float] = []
    family_ok: dict[str, list[int]] = defaultdict(list)
    noop_ok = noop_total = 0
    noops: list[float] = []
    graph.eval()
    adapter.eval()
    with torch.inference_mode():
        for batch in loader:
            a_global = batch["a_global_index"]
            outputs: dict[str, dict[str, torch.Tensor]] = {}
            rows: dict[str, dict[str, torch.Tensor]] = {}
            for prefix in ("a", "b"):
                row = {
                    key[2:]: value
                    for key, value in batch.items()
                    if key.startswith(prefix + "_") and key != prefix + "_global_index"
                }
                row = to_device(row, device)
                rows[prefix] = row
                outputs[prefix] = candidate_forward(graph, adapter, row)
            a, b = rows["a"], rows["b"]
            ao, bo = outputs["a"], outputs["b"]
            at = a["target_distribution"].argmax(dim=-1)
            bt = b["target_distribution"].argmax(dim=-1)
            aok = ao["field_weights"].argmax(dim=-1).eq(at)
            bok = bo["field_weights"].argmax(dim=-1).eq(bt)
            pok = aok & bok
            am = ml.target_margin(
                ao["field_weights"], a["target_distribution"], a["valid_mask"]
            )
            bm = ml.target_margin(
                bo["field_weights"], b["target_distribution"], b["valid_mask"]
            )
            pm = torch.minimum(am, bm)
            pair_total += int(pok.numel())
            pair_ok_total += int(pok.sum().item())
            row_total += int(aok.numel() + bok.numel())
            row_ok_total += int(aok.sum().item() + bok.sum().item())
            margins.extend(float(x) for x in pm.cpu().tolist())
            for out in (ao, bo):
                noop = out["noop_route_probability"]
                noop_top1 = noop.gt(0.5)
                noop_ok += int(noop_top1.sum().item())
                noop_total += int(noop_top1.numel())
                noops.extend(float(x) for x in noop.cpu().tolist())
            for local, index in enumerate(a_global.tolist()):
                family = str(payload["families"][int(index)])
                family_ok[family].append(int(pok[local].item()))
    fam = {
        key: sum(v) / len(v) for key, v in sorted(family_ok.items())
    }
    metrics = {
        "pairs": pair_total,
        "pair_accuracy": pair_ok_total / max(pair_total, 1),
        "family_pair_accuracy": fam,
        "family_min_pair_accuracy": min(fam.values()),
        "row_accuracy": row_ok_total / max(row_total, 1),
        "mean_graph_target_margin": sum(margins) / max(len(margins), 1),
    }
    activation = {
        "noop_top1_accuracy": noop_ok / max(noop_total, 1),
        "mean_noop_probability": sum(noops) / max(len(noops), 1),
    }
    return metrics, activation


def readiness(
    current: dict[str, Any],
    baseline: dict[str, Any],
    ordinary_noop: dict[str, float],
    endpoint_noop: dict[str, float],
    initial_ordinary_noop: dict[str, float],
    initial_endpoint_noop: dict[str, float],
) -> tuple[bool, dict[str, bool]]:
    no_family_regression = all(
        float(current["family_row_accuracy"][relation]) + 1e-12
        >= float(baseline["family_row_accuracy"][relation])
        for relation in baseline["family_row_accuracy"]
    )
    detail = {
        "conditional_all_edge_binding_still_perfect": (
            float(current["conditional_edge_all_four_top1_accuracy"]) == 1.0
        ),
        "conditional_same_relation_binding_still_perfect": (
            float(current["conditional_edge_same_relation_top1_accuracy"]) == 1.0
        ),
        "no_relation_family_row_accuracy_regression": no_family_regression,
        "overall_row_accuracy_strictly_improves": (
            float(current["row_accuracy"])
            > float(baseline["row_accuracy"]) + 1e-12
        ),
        "overall_quad_accuracy_strictly_improves": (
            float(current["quad_accuracy"])
            > float(baseline["quad_accuracy"]) + 1e-12
        ),
        "worst_family_quad_accuracy_not_worse": (
            float(current["family_min_quad_accuracy"]) + 1e-12
            >= float(baseline["family_min_quad_accuracy"])
        ),
        "mean_target_margin_strictly_improves": (
            float(current["mean_target_margin"])
            > float(baseline["mean_target_margin"]) + 1e-12
        ),
        "causal_specialist_top1_is_perfect": (
            float(current["specialist_top1_accuracy"]) == 1.0
        ),
        "causal_family_min_specialist_top1_is_perfect": (
            float(current["family_min_specialist_top1_accuracy"]) == 1.0
        ),
        "ordinary_noop_top1_is_perfect": (
            float(ordinary_noop["noop_top1_accuracy"]) == 1.0
        ),
        "endpoint_noop_top1_is_perfect": (
            float(endpoint_noop["noop_top1_accuracy"]) == 1.0
        ),
        "causal_specialist_probability_strictly_improves": (
            float(current["mean_specialist_probability"])
            > float(baseline["mean_specialist_probability"]) + 1e-12
        ),
        "ordinary_noop_probability_strictly_improves": (
            float(ordinary_noop["mean_noop_probability"])
            > float(initial_ordinary_noop["mean_noop_probability"]) + 1e-12
        ),
        "endpoint_noop_probability_strictly_improves": (
            float(endpoint_noop["mean_noop_probability"])
            > float(initial_endpoint_noop["mean_noop_probability"]) + 1e-12
        ),
    }
    return all(detail.values()), detail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--qualification-receipt", required=True)
    p.add_argument("--causal-cache", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--curriculum-manifest", required=True)
    p.add_argument("--preservation-anchors", required=True)
    p.add_argument("--calibration-receipt", required=True)
    p.add_argument("--ordinary-source-curriculum", required=True)
    p.add_argument("--endpoint-source-curriculum", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--structured-config", required=True)
    p.add_argument("--structured-checkpoint", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--ordinary-replay-cache", required=True)
    p.add_argument("--endpoint-replay-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=24)
    p.add_argument("--quad-batch-size", type=int, default=4)
    p.add_argument("--preservation-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=12)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--margin-weight", type=float, default=0.25)
    p.add_argument("--activation-weight", type=float, default=1.0)
    p.add_argument("--preservation-weight", type=float, default=1.0)
    p.add_argument("--noop-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("dual-view specialist training requires one CUDA device")
    device = torch.device("cuda")
    old.seed_all(args.seed)

    root = Path(args.repo_root).resolve()
    qual_path = Path(args.qualification_receipt).resolve()
    causal_cache_path = Path(args.causal_cache).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    anchors_path = Path(args.preservation_anchors).resolve()
    cal_path = Path(args.calibration_receipt).resolve()
    ordinary_source_path = Path(args.ordinary_source_curriculum).resolve()
    endpoint_source_path = Path(args.endpoint_source_curriculum).resolve()
    semantic_config = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    structured_config = Path(args.structured_config).resolve()
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    map_path = Path(args.layer_map).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    ordinary_replay_path = Path(args.ordinary_replay_cache).resolve()
    endpoint_replay_path = Path(args.endpoint_replay_cache).resolve()
    output = Path(args.output_dir).resolve()

    for path in (
        qual_path, causal_cache_path, curriculum_path, manifest_path,
        anchors_path, cal_path, ordinary_source_path, endpoint_source_path,
        semantic_config, semantic_checkpoint / "alice_n0_v02.safetensors",
        tokenizer_dir / "tokenizer.json", structured_config,
        structured_checkpoint / "structured_state.safetensors",
        map_path, adapter_path, graph_path,
        ordinary_replay_path, endpoint_replay_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing dual-view training input: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite dual-view training output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    exact = {
        qual_path: EXPECTED_QUAL_SHA256,
        causal_cache_path: EXPECTED_CAUSAL_CACHE_SHA256,
        curriculum_path: EXPECTED_CURRICULUM_SHA256,
        manifest_path: EXPECTED_MANIFEST_SHA256,
        anchors_path: EXPECTED_ANCHORS_SHA256,
        cal_path: EXPECTED_CALIBRATION_SHA256,
        adapter_path: EXPECTED_PARENT_ADAPTER_SHA256,
        graph_path: EXPECTED_PARENT_GRAPH_SHA256,
    }
    for path, digest in exact.items():
        if old.sha(path) != digest:
            raise SystemExit(f"dual-view training lineage drift: {path.name}")

    qual = json.loads(qual_path.read_text(encoding="utf-8"))
    if qual.get("schema") != QUAL_SCHEMA:
        raise SystemExit("dual-view qualification schema drift")
    if qual.get("status") != "PASS_DUAL_VIEW_LATE_INTERACTION_NO_GRADIENT_RUNTIME_CONTRACT":
        raise SystemExit("dual-view qualification did not pass")
    for key in (
        "exact_parent_field_weights",
        "exact_parent_pooled_state",
        "parent_parameters_exactly_unchanged",
    ):
        if qual.get(key) is not True:
            raise SystemExit(f"qualification invariant drift: {key}")
    if float(qual["dev_conditional_edge_all_four_top1_accuracy"]) != 1.0:
        raise SystemExit("qualified edge identity is not perfect")
    if float(qual["dev_conditional_edge_same_relation_top1_accuracy"]) != 1.0:
        raise SystemExit("qualified same-relation edge identity is not perfect")
    if qual.get("training_authorized") is not False:
        raise SystemExit("qualification receipt self-authorized training")

    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    if anchors.get("schema") != ANCHOR_SCHEMA:
        raise SystemExit("preservation anchor schema drift")
    if anchors["gradient_eligibility"]["dev_rows_may_be_used_for_gradient"] is not False:
        raise SystemExit("preservation DEV leakage")
    if anchors["gradient_eligibility"]["heldout_or_test_rows_may_be_used_for_gradient"] is not False:
        raise SystemExit("preservation heldout leakage")

    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal.get("status") != "PASS_PARENT_ONLY_PRESERVATION_CALIBRATION":
        raise SystemExit("preservation calibration drift")
    policy = cal["policy"]

    causal_rows_all = old.read_jsonl(curriculum_path)
    causal_rows = [
        row for row in causal_rows_all if str(row["split"]) in {"train", "dev"}
    ]
    causal = torch.load(causal_cache_path, map_location="cpu")
    if causal["ids"] != [str(row["id"]) for row in causal_rows]:
        raise SystemExit("causal cache/curriculum order drift")

    ordinary_payload = torch.load(ordinary_replay_path, map_location="cpu")
    endpoint_payload = torch.load(endpoint_replay_path, map_location="cpu")
    ordinary_rows = old.read_jsonl(ordinary_source_path)
    endpoint_rows = old.read_jsonl(endpoint_source_path)

    ordinary_train_indices = old.indices_from_ids(
        ordinary_payload,
        [str(x) for x in anchors["ordinary"]["train_anchor_ids"]],
        "ordinary",
    )
    endpoint_train_indices = old.indices_from_ids(
        endpoint_payload,
        [str(x) for x in anchors["endpoint"]["train_anchor_ids"]],
        "endpoint",
    )
    ordinary_dev_indices = [
        i for i, split in enumerate(ordinary_payload["splits"])
        if str(split) == "dev"
    ]
    endpoint_dev_pairs = rr.pair_indices(endpoint_payload, "dev")
    endpoint_dev_indices = sorted({i for pair in endpoint_dev_pairs for i in pair})

    semantic_cfg = load_n0_config(semantic_config)
    semantic = AliceN0V02Model(semantic_cfg)
    semantic_state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"),
        device="cpu",
    )
    missing, unexpected = semantic.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}"
        )
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    causal_indices = list(range(len(causal_rows)))
    causal_query_content = build_query_content_lookup(
        causal,
        causal_rows,
        causal_indices,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    causal_fields = build_field_lookup(
        causal,
        causal_rows,
        causal_indices,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )

    ordinary_needed = sorted(set(ordinary_train_indices + ordinary_dev_indices))
    endpoint_needed = sorted(set(endpoint_train_indices + endpoint_dev_indices))
    ordinary_hidden = ml.attach_query_hidden_states(
        ordinary_payload,
        ordinary_rows,
        indices=ordinary_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    endpoint_hidden = ml.attach_query_hidden_states(
        endpoint_payload,
        endpoint_rows,
        indices=endpoint_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    ordinary_query_content = build_query_content_lookup(
        ordinary_payload,
        ordinary_rows,
        ordinary_needed,
        tokenizer=tokenizer,
        max_length=args.max_length,
        attention_lookup=ordinary_hidden,
    )
    endpoint_query_content = build_query_content_lookup(
        endpoint_payload,
        endpoint_rows,
        endpoint_needed,
        tokenizer=tokenizer,
        max_length=args.max_length,
        attention_lookup=endpoint_hidden,
    )
    ordinary_fields = build_field_lookup(
        ordinary_payload,
        ordinary_rows,
        ordinary_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    endpoint_fields = build_field_lookup(
        endpoint_payload,
        endpoint_rows,
        endpoint_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    torch.save(
        {
            "causal_field_tokens": causal_fields,
            "ordinary_field_tokens": ordinary_fields,
            "endpoint_field_tokens": endpoint_fields,
        },
        output / "dual_view_field_token_cache.pt",
    )

    del semantic, semantic_state
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

    bridge = DualViewLateInteractionBridge.from_layer_map_file(
        layer_map_path=map_path,
        semantic_size=rr.expanded_graph_config().semantic_size,
        graph_size=rr.expanded_graph_config().graph_size,
        num_relation_types=rr.expanded_graph_config().num_relation_types,
        num_hidden_states=EXPECTED_HIDDEN_STATES,
        interface_size=rr.expanded_graph_config().graph_size,
        dropout=0.0,
        binding_logit_scale_init=10.0,
    )
    graph = DualViewLateInteractionEvidenceGraphEncoder(
        config=rr.expanded_graph_config(),
        bridge=bridge,
    ).to(device)
    missing, unexpected = graph.load_state_dict(parent_state, strict=False)
    if any(
        not name.startswith("dual_view_query_edge_bridge.")
        for name in missing
    ) or unexpected:
        raise SystemExit(
            f"dual-view parent load drift missing={list(missing)} unexpected={list(unexpected)}"
        )
    graph.freeze_parent_for_bridge_training()
    graph.dual_view_query_edge_bridge.binding_logit_scale.requires_grad = False
    trainable = [
        p for p in graph.parameters() if p.requires_grad
    ]
    trainable_names = [
        name for name, p in graph.named_parameters() if p.requires_grad
    ]
    if not trainable or any(
        not name.startswith("dual_view_query_edge_bridge.")
        for name in trainable_names
    ):
        raise SystemExit("trainable scope escaped dual-view bridge")
    if "dual_view_query_edge_bridge.binding_logit_scale" in trainable_names:
        raise SystemExit("binding scale is not frozen")
    parent_hash_before = old.tensor_state_sha256(
        graph,
        exclude_prefixes=("dual_view_query_edge_bridge.",),
    )

    train_quads = DualViewQuadDataset(
        causal,
        "train",
        query_content_lookup=causal_query_content,
        field_lookup=causal_fields,
    )
    dev_quads = DualViewQuadDataset(
        causal,
        "dev",
        query_content_lookup=causal_query_content,
        field_lookup=causal_fields,
    )
    ordinary_train = DualViewRowDataset(
        ordinary_payload,
        ordinary_train_indices,
        hidden_lookup=ordinary_hidden,
        query_content_lookup=ordinary_query_content,
        field_lookup=ordinary_fields,
    )
    endpoint_train = DualViewRowDataset(
        endpoint_payload,
        endpoint_train_indices,
        hidden_lookup=endpoint_hidden,
        query_content_lookup=endpoint_query_content,
        field_lookup=endpoint_fields,
    )
    ordinary_dev = DualViewRowDataset(
        ordinary_payload,
        ordinary_dev_indices,
        hidden_lookup=ordinary_hidden,
        query_content_lookup=ordinary_query_content,
        field_lookup=ordinary_fields,
    )
    endpoint_dev_rows = DualViewRowDataset(
        endpoint_payload,
        endpoint_dev_indices,
        hidden_lookup=endpoint_hidden,
        query_content_lookup=endpoint_query_content,
        field_lookup=endpoint_fields,
    )
    endpoint_dev = DualViewPairDataset(endpoint_dev_rows, endpoint_dev_pairs)

    baseline_dev = evaluate_causal(
        graph, adapter, dev_quads, causal, device, args.eval_batch_size
    )
    initial_ordinary, initial_ordinary_noop = evaluate_ordinary(
        graph, adapter, ordinary_dev, ordinary_payload, device, args.eval_batch_size
    )
    initial_endpoint, initial_endpoint_noop = evaluate_endpoint(
        graph, adapter, endpoint_dev, endpoint_payload, device, args.eval_batch_size
    )
    initial_preserved, initial_checks = ml.preservation_pass(
        initial_ordinary, initial_endpoint, policy
    )
    if not initial_preserved:
        raise SystemExit("zero-init dual-view candidate failed preservation")
    if baseline_dev["conditional_edge_all_four_top1_accuracy"] != 1.0:
        raise SystemExit("baseline dual-view edge identity drift")
    if baseline_dev["conditional_edge_same_relation_top1_accuracy"] != 1.0:
        raise SystemExit("baseline same-relation edge identity drift")
    if abs(float(baseline_dev["mean_specialist_probability"]) - 0.5) > 1e-7:
        raise SystemExit("specialist activation initialization drift")

    print("baseline_dual_view_dev=" + json.dumps(baseline_dev, sort_keys=True), flush=True)
    print(
        "initial_preservation=" + json.dumps(
            {
                "ordinary": initial_ordinary,
                "endpoint": initial_endpoint,
                "ordinary_noop": initial_ordinary_noop,
                "endpoint_noop": initial_endpoint_noop,
            },
            sort_keys=True,
        ),
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
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    causal_loader = DataLoader(
        train_quads,
        batch_size=args.quad_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    ordinary_loader = DataLoader(
        ordinary_train,
        batch_size=args.preservation_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=0,
    )
    endpoint_loader = DataLoader(
        endpoint_train,
        batch_size=args.preservation_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
        num_workers=0,
    )
    causal_it = iter(causal_loader)
    ordinary_it = iter(ordinary_loader)
    endpoint_it = iter(endpoint_loader)
    receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, args.max_steps + 1):
        causal_batch, causal_it = old.next_batch(causal_it, causal_loader)
        ordinary_batch, ordinary_it = old.next_batch(ordinary_it, ordinary_loader)
        endpoint_batch, endpoint_it = old.next_batch(endpoint_it, endpoint_loader)

        graph.train()
        adapter.eval()
        optimizer.zero_grad(set_to_none=True)

        causal_flat = to_device(flatten_quad_batch(causal_batch), device)
        out = candidate_forward(graph, adapter, causal_flat)
        selection_ce = ml.dist_ce(
            out["field_weights"], causal_flat["target_distribution"]
        )
        margins = ml.target_margin(
            out["field_weights"],
            causal_flat["target_distribution"],
            causal_flat["valid_mask"],
        )
        margin_loss = F.relu(args.margin - margins).mean()
        activation_bce = specialist_bce(out, 1.0)
        causal_total = (
            selection_ce
            + args.margin_weight * margin_loss
            + args.activation_weight * activation_bce
        )
        causal_total.backward()

        preservation_log: list[dict[str, float]] = []
        for preservation_batch in (ordinary_batch, endpoint_batch):
            preserve = to_device(preservation_batch, device)
            with torch.inference_mode():
                teacher = ml.parent_forward(baseline_graph, adapter, preserve)
            student = candidate_forward(graph, adapter, preserve)
            kl = old.distill_kl(student["field_weights"], teacher["field_weights"])
            noop_bce = specialist_bce(student, 0.0)
            weighted = (
                0.5 * args.preservation_weight * kl
                + 0.5 * args.noop_weight * noop_bce
            )
            weighted.backward()
            preservation_log.append(
                {
                    "parent_kl": float(kl.detach().cpu()),
                    "noop_activation_bce": float(noop_bce.detach().cpu()),
                }
            )

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "selection_ce": float(selection_ce.detach().cpu()),
                        "margin_loss": float(margin_loss.detach().cpu()),
                        "causal_specialist_bce": float(activation_bce.detach().cpu()),
                        "ordinary_parent_kl": preservation_log[0]["parent_kl"],
                        "ordinary_noop_bce": preservation_log[0]["noop_activation_bce"],
                        "endpoint_parent_kl": preservation_log[1]["parent_kl"],
                        "endpoint_noop_bce": preservation_log[1]["noop_activation_bce"],
                        "binding_scale": float(
                            graph.dual_view_query_edge_bridge.binding_logit_scale
                            .exp().detach().cpu()
                        ),
                        "lr": scheduler.get_last_lr()[0],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if step % args.save_every == 0 or step == args.max_steps:
            dev = evaluate_causal(
                graph, adapter, dev_quads, causal, device, args.eval_batch_size
            )
            ordinary, ordinary_noop = evaluate_ordinary(
                graph, adapter, ordinary_dev, ordinary_payload,
                device, args.eval_batch_size
            )
            endpoint, endpoint_noop = evaluate_endpoint(
                graph, adapter, endpoint_dev, endpoint_payload,
                device, args.eval_batch_size
            )
            preserved, preservation_checks = ml.preservation_pass(
                ordinary, endpoint, policy
            )
            causal_ok, causal_checks = readiness(
                dev,
                baseline_dev,
                ordinary_noop,
                endpoint_noop,
                initial_ordinary_noop,
                initial_endpoint_noop,
            )

            cp = output / f"step-{step:08d}"
            cp.mkdir(parents=True, exist_ok=True)
            candidate_path = cp / "dual_view_query_edge_bridge.safetensors"
            bridge_state = {
                name: value.detach().cpu().contiguous()
                for name, value in graph.state_dict().items()
                if name.startswith("dual_view_query_edge_bridge.")
            }
            save_file(bridge_state, str(candidate_path))
            binding_scale_now = float(
                graph.dual_view_query_edge_bridge.binding_logit_scale
                .detach().cpu().item()
            )
            receipt = {
                "schema": CHECKPOINT_SCHEMA,
                "status": "TRAINED_NOT_RATIFIED",
                "step": step,
                "git_revision": old.git_revision(root),
                "candidate_sha256": old.sha(candidate_path),
                "qualification_receipt_sha256": old.sha(qual_path),
                "causal_cache_sha256": old.sha(causal_cache_path),
                "curriculum_sha256": old.sha(curriculum_path),
                "curriculum_manifest_sha256": old.sha(manifest_path),
                "preservation_anchors_sha256": old.sha(anchors_path),
                "calibration_receipt_sha256": old.sha(cal_path),
                "parent_graph_sha256": old.sha(graph_path),
                "parent_adapter_sha256": old.sha(adapter_path),
                "trainable_scope": "dual_view_bridge_except_binding_scale",
                "trainable_parameters": sum(p.numel() for p in trainable),
                "binding_logit_scale_frozen": True,
                "binding_logit_scale_raw_value": binding_scale_now,
                "causal_dev_metrics": dev,
                "causal_dev_checks": causal_checks,
                "causal_dev_ready": causal_ok,
                "ordinary_preservation_metrics": ordinary,
                "endpoint_preservation_metrics": endpoint,
                "ordinary_noop_metrics": ordinary_noop,
                "endpoint_noop_metrics": endpoint_noop,
                "preservation_checks": preservation_checks,
                "preservation_pass": preserved,
                "eligible_for_heldout_decision": preserved and causal_ok,
                "parent_parameters_exactly_unchanged": (
                    old.tensor_state_sha256(
                        graph,
                        exclude_prefixes=("dual_view_query_edge_bridge.",),
                    )
                    == parent_hash_before
                ),
                "causal_test_split_evaluated": False,
                "frozen_challenge_evaluated": False,
                "preservation_dev_rows_used_for_gradient": False,
                "scale_authorized": False,
                "private_identity_gradient": False,
                "production_promotion_authorized": False,
                "n0_complete": False,
            }
            if receipt["parent_parameters_exactly_unchanged"] is not True:
                raise SystemExit("frozen parent changed during dual-view training")
            (cp / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipts[cp.name] = receipt
            print(
                "dual_view_checkpoint=" + json.dumps(receipt, sort_keys=True),
                flush=True,
            )

    eligible = {
        key: value for key, value in receipts.items()
        if value["eligible_for_heldout_decision"]
    }

    def score(receipt: dict[str, Any]) -> tuple[float, ...]:
        dev = receipt["causal_dev_metrics"]
        return (
            float(dev["family_min_quad_accuracy"]),
            float(dev["quad_accuracy"]),
            float(dev["row_accuracy"]),
            float(dev["mean_target_margin"]),
            float(dev["mean_specialist_probability"]),
            -int(receipt["step"]),
        )

    selected = max(eligible, key=lambda key: score(eligible[key])) if eligible else None
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_DUAL_VIEW_SPECIALIST_DEV_AND_PRESERVATION_READY_FOR_SEPARATE_HELDOUT_DECISION"
            if selected is not None
            else
            "FAIL_DUAL_VIEW_SPECIALIST_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
        ),
        "git_revision": old.git_revision(root),
        "qualification_receipt_sha256": old.sha(qual_path),
        "causal_cache_sha256": old.sha(causal_cache_path),
        "curriculum_sha256": old.sha(curriculum_path),
        "curriculum_manifest_sha256": old.sha(manifest_path),
        "preservation_anchors_sha256": old.sha(anchors_path),
        "calibration_receipt_sha256": old.sha(cal_path),
        "parent_graph_sha256": old.sha(graph_path),
        "parent_adapter_sha256": old.sha(adapter_path),
        "field_token_cache_sha256": old.sha(output / "dual_view_field_token_cache.pt"),
        "baseline_dual_view_dev": baseline_dev,
        "initial_parent_preservation": {
            "ordinary": initial_ordinary,
            "endpoint": initial_endpoint,
            "ordinary_noop": initial_ordinary_noop,
            "endpoint_noop": initial_endpoint_noop,
            "checks": initial_checks,
        },
        "checkpoints": receipts,
        "eligible_checkpoint_keys": sorted(eligible),
        "selected_checkpoint": selected,
        "selected_candidate_sha256": (
            eligible[selected]["candidate_sha256"] if selected else None
        ),
        "binding_logit_scale_frozen": True,
        "edge_identity_training_loss": False,
        "training_objective": {
            "causal_field_selection_ce": True,
            "causal_margin_weight": args.margin_weight,
            "causal_specialist_activation_bce_weight": args.activation_weight,
            "parent_distillation_weight": args.preservation_weight,
            "preservation_noop_activation_bce_weight": args.noop_weight,
            "conditional_edge_binding_loss": False,
            "proxy_routing_loss": False,
            "load_balancing_loss": False,
            "gradient_surgery": False,
        },
        "causal_test_split_evaluated": False,
        "causal_test_split_opened_after_training": False,
        "frozen_challenge_evaluated": False,
        "parent_graph_parameters_exactly_unchanged": (
            old.tensor_state_sha256(
                graph,
                exclude_prefixes=("dual_view_query_edge_bridge.",),
            )
            == parent_hash_before
        ),
        "automatic_rerun_or_hotfix_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "scale_authorized": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "make_one_explicit_heldout_opening_decision_without_changing_candidate"
            if selected is not None
            else
            "stop_and_localize_failure_between_specialist_activation_and_directional_residual_reasoning"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
