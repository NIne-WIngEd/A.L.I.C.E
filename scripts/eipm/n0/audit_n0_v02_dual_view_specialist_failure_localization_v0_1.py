#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_query_edge_dual_view import (
    DualViewLateInteractionEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.query_edge_dual_view_late_interaction import (
    DualViewLateInteractionBridge,
)
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_relation_conditioned_multilayer_interface_v0_1 as ml
import train_n0_v02_query_edge_cross_attention_bridge_v0_1 as old
import train_n0_v02_dual_view_specialist_v0_1 as tr


SCHEMA = "alice.eipm.n0.dual-view-specialist-failure-localization.v0.1"

EXPECTED_RESULT_SHA256 = (
    "1a38c37a41de68bea0bb9bc897b86d62cbc457b8b93189c2d16731110c1694ec"
)
EXPECTED_QUAL_SHA256 = (
    "714f06d4939abad2b816a7e3ffb5c5fa10112129ef0b520fd838f2d182286278"
)
EXPECTED_CAUSAL_CACHE_SHA256 = (
    "5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
)
EXPECTED_CURRICULUM_SHA256 = (
    "c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
)
EXPECTED_FIELD_TOKEN_CACHE_SHA256 = (
    "8bdf72561bafd2bbb88b56e9202570401f76cd7a0bcf48aa0abd5d8e2400ad5c"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
)
EXPECTED_LAYER_MAP_SHA256 = (
    "ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
)

EXPECTED_CHECKPOINTS = {
    40: "baae065f9869af316ac6ba82695b6076cedd223a319f36fd15cd24337e993f06",
    80: "18eb6b4026472f64d12d87c2b048e1551dfe488a6e6e3ff07785c9ec4997b07f",
    120: "5296ee7ffc1b8a5d85d70f8e14a643db088eef18a14f6b9267786baba0300664",
    160: "f7d7ff95dd59758b326b3f4b70ae85e2328fb00f18a8175f02a145a4e1d9fdf0",
    200: "bbe0e38955f5887acf24649007b8bf755aa9cdda8c95ead7afaf6e1d5bbebdd7",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    if values.numel() == 0:
        raise ValueError("empty probability vector")
    q = torch.tensor([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    out = torch.quantile(values, q)
    return {
        "min": float(out[0]),
        "p10": float(out[1]),
        "p25": float(out[2]),
        "median": float(out[3]),
        "p75": float(out[4]),
        "p90": float(out[5]),
        "max": float(out[6]),
        "mean": float(values.mean()),
    }


def pairwise_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    p = pos.float().flatten()[:, None]
    n = neg.float().flatten()[None, :]
    greater = (p > n).float()
    equal = (p == n).float()
    return float((greater + 0.5 * equal).mean())


@torch.inference_mode()
def collect_specialist_probabilities(
    graph: DualViewLateInteractionEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values: list[torch.Tensor] = []
    graph.eval()
    adapter.eval()
    for batch in loader:
        batch = tr.to_device(batch, device)
        out = tr.candidate_forward(graph, adapter, batch)
        values.append(out["specialist_probability"].detach().float().cpu())
    return torch.cat(values, dim=0)


@torch.inference_mode()
def counterfactual_components(
    graph: DualViewLateInteractionEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapted = ml.adapter_forward(adapter, batch)
    field_states = adapted["field_states"]
    valid_mask = batch["valid_mask"]
    edge_index = batch["edge_index"]
    edge_type_ids = batch["edge_type_ids"]
    edge_confidence = batch["edge_confidence"]
    edge_valid_mask = batch["edge_valid_mask"]
    query_semantic = batch["query_semantic"]

    batch_size, fields = valid_mask.shape
    x = graph.input_norm(graph.input_projection(field_states))
    for layer in graph.layers:
        x, _layer_norm = layer(
            x,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
        )

    query = graph.pool_query.unsqueeze(0).expand(batch_size, -1)
    query = query + graph.query_projection(query_semantic)
    query = F.normalize(query, dim=-1)

    parent_relation_bias = graph._query_conditioned_relation_bias(
        x=x,
        query=query,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_confidence=edge_confidence,
        edge_valid_mask=edge_valid_mask,
        valid_mask=valid_mask,
    )

    hidden = batch["query_hidden_states"].float()
    hidden_tuple = tuple(hidden[:, layer] for layer in range(hidden.size(1)))
    bridge = graph.dual_view_query_edge_bridge(
        hidden_states=hidden_tuple,
        attention_mask=batch["query_attention_mask"],
        query_content_mask=batch["query_content_mask"],
        field_token_states=batch["field_token_states"].float(),
        field_content_mask=batch["field_content_mask"],
        graph_states=x,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_valid_mask=edge_valid_mask,
        valid_mask=valid_mask,
    )

    base_scores = (
        torch.einsum("bfd,bd->bf", x, query)
        / math.sqrt(graph.config.graph_size)
    )
    base_scores = base_scores + parent_relation_bias
    if graph.config.base_weight_scale > 0.0:
        prior = adapted["field_weights"].to(x.dtype).clamp_min(1e-6).log()
        base_scores = base_scores + graph.config.base_weight_scale * prior

    return {
        "x": x,
        "base_scores": base_scores,
        "bridge": bridge,
        "valid_mask": valid_mask,
        "edge_index": edge_index,
        "edge_confidence": edge_confidence,
    }


def hard_top1_distribution(
    conditional: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    pred = conditional.argmax(dim=-1)
    hard = torch.zeros_like(conditional)
    hard.scatter_(1, pred[:, None], 1.0)
    any_active = active.any(dim=-1)
    return hard * any_active[:, None].to(hard.dtype)


def weights_from_mode(
    components: dict[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    bridge = components["bridge"]
    conditional = bridge["conditional_edge_probability"]
    specialist = bridge["specialist_probability"]
    active = bridge["route_active_mask"]
    hard = hard_top1_distribution(conditional, active)

    if mode == "actual_activation_soft_binding":
        route = specialist[:, None] * conditional
    elif mode == "forced_specialist_soft_binding":
        route = conditional
    elif mode == "actual_activation_hard_top1_binding":
        route = specialist[:, None] * hard
    elif mode == "forced_specialist_hard_top1_binding":
        route = hard
    else:
        raise ValueError(f"unknown counterfactual mode: {mode}")

    edge_index = components["edge_index"]
    confidence = components["edge_confidence"].squeeze(-1)
    valid = components["valid_mask"]
    batch, fields = valid.shape
    source = edge_index[..., 0].clamp(0, fields - 1)
    target = edge_index[..., 1].clamp(0, fields - 1)
    active_f = active.to(route.dtype)
    confidence = confidence.to(route.dtype).clamp(0.0, 1.0)

    source_delta = (
        route * bridge["source_residual_proposal"] * confidence * active_f
    )
    target_delta = (
        route * bridge["target_residual_proposal"] * confidence * active_f
    )

    flat = torch.zeros(
        batch * fields,
        device=source_delta.device,
        dtype=source_delta.dtype,
    )
    offsets = (
        torch.arange(batch, device=source_delta.device) * fields
    ).unsqueeze(1)
    flat.index_add_(
        0,
        (source + offsets).reshape(-1),
        source_delta.reshape(-1),
    )
    flat.index_add_(
        0,
        (target + offsets).reshape(-1),
        target_delta.reshape(-1),
    )
    bridge_bias = flat.reshape(batch, fields)

    scores = components["base_scores"] + bridge_bias
    scores = scores.masked_fill(
        ~valid,
        torch.finfo(scores.dtype).min,
    )
    return torch.softmax(scores, dim=-1)


def evaluate_counterfactual_modes(
    graph: DualViewLateInteractionEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: tr.DualViewQuadDataset,
    payload: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    modes = (
        "actual_activation_soft_binding",
        "forced_specialist_soft_binding",
        "actual_activation_hard_top1_binding",
        "forced_specialist_hard_top1_binding",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    accum: dict[str, dict[str, Any]] = {
        mode: {
            "rows": 0,
            "row_ok": 0,
            "quads": 0,
            "quad_ok": 0,
            "margins": [],
            "pair_endpoint_ok": 0,
            "family_rows": defaultdict(list),
            "family_quads": defaultdict(list),
            "family_pair_endpoint": defaultdict(list),
        }
        for mode in modes
    }
    proposal_total = proposal_ok = 0
    proposal_margins: list[float] = []
    proposal_family: dict[str, list[int]] = defaultdict(list)
    reconstructed_actual_max_delta = 0.0
    hard_binding_top1_ok = 0
    hard_binding_rows = 0

    graph.eval()
    adapter.eval()
    for batch in loader:
        global_indices = batch["global_indices"]
        flat = tr.to_device(tr.flatten_quad_batch(batch), device)
        components = counterfactual_components(graph, adapter, flat)
        bridge = components["bridge"]

        runtime = tr.candidate_forward(graph, adapter, flat)
        reconstructed = weights_from_mode(
            components,
            "actual_activation_soft_binding",
        )
        reconstructed_actual_max_delta = max(
            reconstructed_actual_max_delta,
            float((runtime["field_weights"] - reconstructed).abs().max()),
        )

        relevant = flat["relevant_edge_index"].long()
        conditional = bridge["conditional_edge_probability"]
        hard_binding_top1_ok += int(
            conditional.argmax(dim=-1).eq(relevant).sum().item()
        )
        hard_binding_rows += int(relevant.numel())

        row_idx = torch.arange(relevant.size(0), device=device)
        source_field = flat["edge_index"][row_idx, relevant, 0]
        target_field = flat["edge_index"][row_idx, relevant, 1]
        desired_field = flat["target_distribution"].argmax(dim=-1)

        source_prop = bridge["source_residual_proposal"][row_idx, relevant]
        target_prop = bridge["target_residual_proposal"][row_idx, relevant]
        desired_is_source = desired_field.eq(source_field)
        desired_is_target = desired_field.eq(target_field)
        if not torch.all(desired_is_source | desired_is_target):
            raise SystemExit("target field escaped relevant edge")

        proposal_margin = torch.where(
            desired_is_source,
            source_prop - target_prop,
            target_prop - source_prop,
        )
        proposal_correct = proposal_margin.gt(0.0)
        proposal_total += int(proposal_correct.numel())
        proposal_ok += int(proposal_correct.sum().item())
        proposal_margins.extend(
            float(x) for x in proposal_margin.detach().cpu().tolist()
        )

        batch_quads = global_indices.size(0)
        for local in range(batch_quads):
            first = int(global_indices[local, 0].item())
            relation = str(payload["relations"][first])
            start = 4 * local
            proposal_family[relation].extend(
                int(x)
                for x in proposal_correct[
                    start : start + 4
                ].detach().cpu().tolist()
            )

        for mode in modes:
            weights = weights_from_mode(components, mode)
            target = desired_field
            pred = weights.argmax(dim=-1)
            ok = pred.eq(target)
            margin = ml.target_margin(
                weights,
                flat["target_distribution"],
                flat["valid_mask"],
            )
            other_endpoint = torch.where(
                desired_is_source,
                target_field,
                source_field,
            )
            endpoint_ok = (
                weights[row_idx, target] > weights[row_idx, other_endpoint]
            )

            ok4 = ok.reshape(batch_quads, 4)
            ep4 = endpoint_ok.reshape(batch_quads, 4)
            margin4 = margin.reshape(batch_quads, 4)
            qok = ok4.all(dim=1)

            a = accum[mode]
            a["rows"] += int(ok.numel())
            a["row_ok"] += int(ok.sum().item())
            a["quads"] += int(qok.numel())
            a["quad_ok"] += int(qok.sum().item())
            a["pair_endpoint_ok"] += int(endpoint_ok.sum().item())
            a["margins"].extend(float(x) for x in margin.detach().cpu().tolist())

            for local in range(batch_quads):
                first = int(global_indices[local, 0].item())
                relation = str(payload["relations"][first])
                a["family_quads"][relation].append(int(qok[local].item()))
                a["family_rows"][relation].extend(
                    int(x) for x in ok4[local].detach().cpu().tolist()
                )
                a["family_pair_endpoint"][relation].extend(
                    int(x) for x in ep4[local].detach().cpu().tolist()
                )

    mode_metrics: dict[str, Any] = {}
    for mode, a in accum.items():
        family_rows = {
            key: sum(v) / len(v)
            for key, v in sorted(a["family_rows"].items())
        }
        family_quads = {
            key: sum(v) / len(v)
            for key, v in sorted(a["family_quads"].items())
        }
        family_endpoint = {
            key: sum(v) / len(v)
            for key, v in sorted(a["family_pair_endpoint"].items())
        }
        mode_metrics[mode] = {
            "row_accuracy": a["row_ok"] / max(a["rows"], 1),
            "quad_accuracy": a["quad_ok"] / max(a["quads"], 1),
            "mean_target_margin": (
                sum(a["margins"]) / max(len(a["margins"]), 1)
            ),
            "within_relevant_pair_endpoint_accuracy": (
                a["pair_endpoint_ok"] / max(a["rows"], 1)
            ),
            "family_row_accuracy": family_rows,
            "family_quad_accuracy": family_quads,
            "family_within_pair_endpoint_accuracy": family_endpoint,
        }

    proposal_family_accuracy = {
        key: sum(v) / len(v)
        for key, v in sorted(proposal_family.items())
    }
    return {
        "modes": mode_metrics,
        "reconstructed_actual_field_weight_max_delta": (
            reconstructed_actual_max_delta
        ),
        "conditional_edge_top1_accuracy": (
            hard_binding_top1_ok / max(hard_binding_rows, 1)
        ),
        "relevant_edge_residual_proposal_role_accuracy": (
            proposal_ok / max(proposal_total, 1)
        ),
        "relevant_edge_residual_proposal_mean_role_margin": (
            sum(proposal_margins) / max(len(proposal_margins), 1)
        ),
        "family_residual_proposal_role_accuracy": proposal_family_accuracy,
    }


def load_candidate(
    *,
    parent_state: dict[str, torch.Tensor],
    checkpoint_path: Path,
    layer_map: Path,
    device: torch.device,
) -> DualViewLateInteractionEvidenceGraphEncoder:
    bridge = DualViewLateInteractionBridge.from_layer_map_file(
        layer_map_path=layer_map,
        semantic_size=rr.expanded_graph_config().semantic_size,
        graph_size=rr.expanded_graph_config().graph_size,
        num_relation_types=rr.expanded_graph_config().num_relation_types,
        num_hidden_states=17,
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
            f"parent/candidate load drift missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )

    saved = load_file(str(checkpoint_path), device="cpu")
    prefix = "dual_view_query_edge_bridge."
    bridge_state = {
        name[len(prefix) :]: tensor
        for name, tensor in saved.items()
        if name.startswith(prefix)
    }
    if len(bridge_state) != len(saved):
        raise SystemExit("candidate checkpoint contains non-bridge parameters")
    graph.dual_view_query_edge_bridge.load_state_dict(
        bridge_state,
        strict=True,
    )
    for parameter in graph.parameters():
        parameter.requires_grad = False
    graph.eval()
    return graph


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--training-result", required=True)
    p.add_argument("--training-root", required=True)
    p.add_argument("--qualification-receipt", required=True)
    p.add_argument("--causal-cache", required=True)
    p.add_argument("--field-token-cache", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--ordinary-source-curriculum", required=True)
    p.add_argument("--endpoint-source-curriculum", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--ordinary-replay-cache", required=True)
    p.add_argument("--endpoint-replay-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=24)
    p.add_argument("--eval-batch-size", type=int, default=24)
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    result_path = Path(args.training_result).resolve()
    training_root = Path(args.training_root).resolve()
    qual_path = Path(args.qualification_receipt).resolve()
    causal_cache_path = Path(args.causal_cache).resolve()
    field_cache_path = Path(args.field_token_cache).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    ordinary_source_path = Path(args.ordinary_source_curriculum).resolve()
    endpoint_source_path = Path(args.endpoint_source_curriculum).resolve()
    semantic_config = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    layer_map = Path(args.layer_map).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    ordinary_replay_path = Path(args.ordinary_replay_cache).resolve()
    endpoint_replay_path = Path(args.endpoint_replay_cache).resolve()
    output = Path(args.output).resolve()

    required = (
        result_path,
        qual_path,
        causal_cache_path,
        field_cache_path,
        curriculum_path,
        ordinary_source_path,
        endpoint_source_path,
        semantic_config,
        semantic_checkpoint / "alice_n0_v02.safetensors",
        tokenizer_dir / "tokenizer.json",
        layer_map,
        adapter_path,
        graph_path,
        ordinary_replay_path,
        endpoint_replay_path,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"missing localization input: {path}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite localization audit: {output}")

    exact = {
        result_path: EXPECTED_RESULT_SHA256,
        qual_path: EXPECTED_QUAL_SHA256,
        causal_cache_path: EXPECTED_CAUSAL_CACHE_SHA256,
        field_cache_path: EXPECTED_FIELD_TOKEN_CACHE_SHA256,
        curriculum_path: EXPECTED_CURRICULUM_SHA256,
        layer_map: EXPECTED_LAYER_MAP_SHA256,
        graph_path: EXPECTED_PARENT_GRAPH_SHA256,
        adapter_path: EXPECTED_PARENT_ADAPTER_SHA256,
    }
    for path, digest in exact.items():
        if sha(path) != digest:
            raise SystemExit(f"localization lineage drift: {path.name}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != (
        "FAIL_DUAL_VIEW_SPECIALIST_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
    ):
        raise SystemExit("source experiment is not the expected failed run")
    if result.get("selected_checkpoint") is not None:
        raise SystemExit("failed experiment unexpectedly selected a checkpoint")
    if result.get("eligible_checkpoint_keys") != []:
        raise SystemExit("failed experiment unexpectedly has eligible checkpoint")
    if result.get("causal_test_split_evaluated") is not False:
        raise SystemExit("TEST contamination in source experiment")
    if result.get("frozen_challenge_evaluated") is not False:
        raise SystemExit("challenge contamination in source experiment")
    if result.get("binding_logit_scale_frozen") is not True:
        raise SystemExit("binding scale was not frozen in source experiment")
    if result.get("edge_identity_training_loss") is not False:
        raise SystemExit("edge identity was trained in source experiment")

    checkpoint_paths: dict[int, Path] = {}
    for step, digest in EXPECTED_CHECKPOINTS.items():
        cp = training_root / f"step-{step:08d}" / "dual_view_query_edge_bridge.safetensors"
        if not cp.is_file():
            raise SystemExit(f"missing trained checkpoint: {cp}")
        if sha(cp) != digest:
            raise SystemExit(f"checkpoint hash drift at step {step}")
        checkpoint_paths[step] = cp

    device = torch.device("cpu")
    causal_rows_all = old.read_jsonl(curriculum_path)
    causal_rows = [
        row for row in causal_rows_all
        if str(row["split"]) in {"train", "dev"}
    ]
    causal = torch.load(causal_cache_path, map_location="cpu")
    if causal["ids"] != [str(row["id"]) for row in causal_rows]:
        raise SystemExit("causal cache/curriculum drift")

    tokenizer = load_tokenizer(tokenizer_dir)
    causal_dev_indices = [
        i for i, split in enumerate(causal["splits"])
        if str(split) == "dev"
    ]
    causal_query_content = tr.build_query_content_lookup(
        causal,
        causal_rows,
        causal_dev_indices,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    field_cache = torch.load(field_cache_path, map_location="cpu")
    causal_fields = field_cache["causal_field_tokens"]
    ordinary_fields = field_cache["ordinary_field_tokens"]
    endpoint_fields = field_cache["endpoint_field_tokens"]
    for index in causal_dev_indices:
        if index not in causal_fields:
            raise SystemExit(f"missing causal field tokens: {index}")

    ordinary_payload = torch.load(ordinary_replay_path, map_location="cpu")
    endpoint_payload = torch.load(endpoint_replay_path, map_location="cpu")
    ordinary_rows = old.read_jsonl(ordinary_source_path)
    endpoint_rows = old.read_jsonl(endpoint_source_path)

    ordinary_dev_indices = [
        i for i, split in enumerate(ordinary_payload["splits"])
        if str(split) == "dev"
    ]
    endpoint_dev_pairs = rr.pair_indices(endpoint_payload, "dev")
    endpoint_dev_indices = sorted({
        i for pair in endpoint_dev_pairs for i in pair
    })

    semantic_cfg = load_n0_config(semantic_config)
    semantic = AliceN0V02Model(semantic_cfg)
    semantic_state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"),
        device="cpu",
    )
    missing, unexpected = semantic.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint drift missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.eval()

    ordinary_hidden = ml.attach_query_hidden_states(
        ordinary_payload,
        ordinary_rows,
        indices=ordinary_dev_indices,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    endpoint_hidden = ml.attach_query_hidden_states(
        endpoint_payload,
        endpoint_rows,
        indices=endpoint_dev_indices,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    ordinary_query_content = tr.build_query_content_lookup(
        ordinary_payload,
        ordinary_rows,
        ordinary_dev_indices,
        tokenizer=tokenizer,
        max_length=args.max_length,
        attention_lookup=ordinary_hidden,
    )
    endpoint_query_content = tr.build_query_content_lookup(
        endpoint_payload,
        endpoint_rows,
        endpoint_dev_indices,
        tokenizer=tokenizer,
        max_length=args.max_length,
        attention_lookup=endpoint_hidden,
    )
    del semantic, semantic_state

    causal_dev = tr.DualViewQuadDataset(
        causal,
        "dev",
        query_content_lookup=causal_query_content,
        field_lookup=causal_fields,
    )
    ordinary_dev = tr.DualViewRowDataset(
        ordinary_payload,
        ordinary_dev_indices,
        hidden_lookup=ordinary_hidden,
        query_content_lookup=ordinary_query_content,
        field_lookup=ordinary_fields,
    )
    endpoint_dev_rows = tr.DualViewRowDataset(
        endpoint_payload,
        endpoint_dev_indices,
        hidden_lookup=endpoint_hidden,
        query_content_lookup=endpoint_query_content,
        field_lookup=endpoint_fields,
    )

    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()
    parent_state = load_file(str(graph_path), device="cpu")

    checkpoint_results: dict[str, Any] = {}
    for step in sorted(EXPECTED_CHECKPOINTS):
        graph = load_candidate(
            parent_state=parent_state,
            checkpoint_path=checkpoint_paths[step],
            layer_map=layer_map,
            device=device,
        )
        counterfactual = evaluate_counterfactual_modes(
            graph,
            adapter,
            causal_dev,
            causal,
            device=device,
            batch_size=args.eval_batch_size,
        )

        causal_probs: list[torch.Tensor] = []
        causal_loader = DataLoader(
            causal_dev,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.inference_mode():
            for batch in causal_loader:
                flat = tr.to_device(tr.flatten_quad_batch(batch), device)
                out = tr.candidate_forward(graph, adapter, flat)
                causal_probs.append(
                    out["specialist_probability"].detach().float().cpu()
                )
        causal_prob = torch.cat(causal_probs)
        ordinary_prob = collect_specialist_probabilities(
            graph,
            adapter,
            ordinary_dev,
            device=device,
            batch_size=args.eval_batch_size,
        )
        endpoint_prob = collect_specialist_probabilities(
            graph,
            adapter,
            endpoint_dev_rows,
            device=device,
            batch_size=args.eval_batch_size,
        )
        preservation_prob = torch.cat([ordinary_prob, endpoint_prob], dim=0)
        gate = {
            "causal": quantiles(causal_prob),
            "ordinary_preservation": quantiles(ordinary_prob),
            "endpoint_preservation": quantiles(endpoint_prob),
            "combined_preservation": quantiles(preservation_prob),
            "causal_vs_preservation_auc": pairwise_auc(
                causal_prob,
                preservation_prob,
            ),
            "strict_rank_separation": bool(
                causal_prob.min() > preservation_prob.max()
            ),
            "rank_separation_gap": float(
                causal_prob.min() - preservation_prob.max()
            ),
            "causal_top1_at_0_5": float(
                causal_prob.gt(0.5).float().mean()
            ),
            "preservation_noop_top1_at_0_5": float(
                preservation_prob.lt(0.5).float().mean()
            ),
        }

        actual = counterfactual["modes"][
            "actual_activation_soft_binding"
        ]
        forced_soft = counterfactual["modes"][
            "forced_specialist_soft_binding"
        ]
        actual_hard = counterfactual["modes"][
            "actual_activation_hard_top1_binding"
        ]
        forced_hard = counterfactual["modes"][
            "forced_specialist_hard_top1_binding"
        ]
        deltas = {
            "forced_activation_soft_binding_row_gain": (
                forced_soft["row_accuracy"] - actual["row_accuracy"]
            ),
            "hard_top1_actual_activation_row_gain": (
                actual_hard["row_accuracy"] - actual["row_accuracy"]
            ),
            "hard_top1_forced_activation_row_gain_vs_soft_forced": (
                forced_hard["row_accuracy"] - forced_soft["row_accuracy"]
            ),
            "fully_forced_hard_top1_row_gain_vs_actual": (
                forced_hard["row_accuracy"] - actual["row_accuracy"]
            ),
            "fully_forced_hard_top1_quad_gain_vs_actual": (
                forced_hard["quad_accuracy"] - actual["quad_accuracy"]
            ),
            "fully_forced_hard_top1_margin_gain_vs_actual": (
                forced_hard["mean_target_margin"]
                - actual["mean_target_margin"]
            ),
        }
        checkpoint_results[str(step)] = {
            "candidate_sha256": EXPECTED_CHECKPOINTS[step],
            "gate_separation": gate,
            "counterfactual": counterfactual,
            "diagnostic_deltas": deltas,
        }

    best_gate_auc_step = max(
        checkpoint_results,
        key=lambda k: checkpoint_results[k][
            "gate_separation"
        ]["causal_vs_preservation_auc"],
    )
    best_forced_hard_row_step = max(
        checkpoint_results,
        key=lambda k: checkpoint_results[k][
            "counterfactual"
        ]["modes"]["forced_specialist_hard_top1_binding"][
            "row_accuracy"
        ],
    )
    best_proposal_role_step = max(
        checkpoint_results,
        key=lambda k: checkpoint_results[k][
            "counterfactual"
        ]["relevant_edge_residual_proposal_role_accuracy"],
    )

    audit = {
        "schema": SCHEMA,
        "status": "PASS_DUAL_VIEW_SPECIALIST_FAILURE_LOCALIZATION_NO_GRADIENT",
        "source_failed_result_sha256": sha(result_path),
        "source_failed_status": result["status"],
        "qualification_receipt_sha256": sha(qual_path),
        "causal_cache_sha256": sha(causal_cache_path),
        "field_token_cache_sha256": sha(field_cache_path),
        "curriculum_sha256": sha(curriculum_path),
        "parent_graph_sha256": sha(graph_path),
        "parent_adapter_sha256": sha(adapter_path),
        "layer_map_sha256": sha(layer_map),
        "checkpoint_results": checkpoint_results,
        "summary": {
            "best_gate_auc_step": int(best_gate_auc_step),
            "best_gate_auc": checkpoint_results[best_gate_auc_step][
                "gate_separation"
            ]["causal_vs_preservation_auc"],
            "best_gate_strict_rank_separation": checkpoint_results[
                best_gate_auc_step
            ]["gate_separation"]["strict_rank_separation"],
            "best_forced_hard_row_step": int(best_forced_hard_row_step),
            "best_forced_hard_row_accuracy": checkpoint_results[
                best_forced_hard_row_step
            ]["counterfactual"]["modes"][
                "forced_specialist_hard_top1_binding"
            ]["row_accuracy"],
            "best_forced_hard_quad_accuracy": checkpoint_results[
                best_forced_hard_row_step
            ]["counterfactual"]["modes"][
                "forced_specialist_hard_top1_binding"
            ]["quad_accuracy"],
            "best_proposal_role_step": int(best_proposal_role_step),
            "best_relevant_edge_residual_proposal_role_accuracy": (
                checkpoint_results[best_proposal_role_step][
                    "counterfactual"
                ]["relevant_edge_residual_proposal_role_accuracy"]
            ),
        },
        "interpretation_contract": {
            "activation_failure_evidence": (
                "causal and preservation specialist scores overlap in rank "
                "or forcing specialist activation materially repairs DEV"
            ),
            "soft_edge_mixture_failure_evidence": (
                "hard predicted top1 materially improves over soft conditional "
                "edge execution while edge top1 remains 1.0"
            ),
            "directional_residual_failure_evidence": (
                "forced specialist plus hard predicted top1 remains weak and/or "
                "relevant-edge residual proposal role accuracy is weak"
            ),
            "multiple_failures_may_coexist": True,
            "no_threshold_tuning_authorized": True,
            "no_binding_scale_tuning_authorized": True,
        },
        "optimizer_created": False,
        "gradient_performed": False,
        "gpu_required": False,
        "training_authorized": False,
        "causal_test_split_evaluated": False,
        "frozen_challenge_evaluated": False,
        "heldout_opening_authorized": False,
        "scale_authorized": False,
        "semantic_retraining_authorized": False,
        "graph_parent_retraining_authorized": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "interpret activation-vs-mixture-vs-directional-residual localization "
            "before any architecture change or retraining"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
