#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

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
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_relation_conditioned_multilayer_interface_v0_1 as ml
import train_n0_v02_query_edge_cross_attention_bridge_v0_1 as qe


SCHEMA = "alice.eipm.n0.dual-view-late-interaction-qualification.v0.1"

EXPECTED_AUDIT_SHA256 = (
    "6d166af91240cb67cae270c95bb963cafca0597d65da77de3b0ada53bb33b721"
)
EXPECTED_CACHE_SHA256 = (
    "5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
)
EXPECTED_CURRICULUM_SHA256 = (
    "c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
)
EXPECTED_LAYER_MAP_SHA256 = (
    "ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
)
EXPECTED_HIDDEN_STATES = 17
SPECIAL_IDS = {0, 1, 2, 3, 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def content_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for token_id in SPECIAL_IDS:
        mask &= input_ids.ne(token_id)
    return mask


@torch.inference_mode()
def encode_unique_final_field_tokens(
    *,
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    unique = list(dict.fromkeys(texts))
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    model.eval().to(device)
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
        cmask = content_mask(ids.cpu(), attention.cpu())
        for pos, text in enumerate(chunk):
            out[text] = (final[pos], cmask[pos])
    return out


def query_content_masks(
    *,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    indices: list[int],
    payload: dict[str, Any],
    max_length: int,
) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for index in indices:
        encoded = tokenizer(
            str(rows[index]["query_text"]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention = encoded["attention_mask"][0].bool()
        cached = payload["query_attention_mask"][index].bool()
        if not torch.equal(attention, cached):
            raise SystemExit(
                f"query tokenization/cache drift: {rows[index]['id']}"
            )
        out[index] = content_mask(
            encoded["input_ids"][0],
            encoded["attention_mask"][0],
        )
    return out


def batch_from_indices(
    *,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    query_masks: dict[int, torch.Tensor],
    field_token_lookup: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    tensor_keys = (
        "valid_mask",
        "edge_index",
        "edge_type_ids",
        "edge_confidence",
        "edge_valid_mask",
        "query_semantic",
        "query_hidden_states",
        "query_attention_mask",
        "parent_field_states",
        "parent_field_weights",
    )
    batch: dict[str, Any] = {
        key: payload[key][indices].to(device)
        for key in tensor_keys
    }
    batch["query_content_mask"] = torch.stack(
        [query_masks[index] for index in indices],
        dim=0,
    ).to(device)

    field_states: list[torch.Tensor] = []
    field_masks: list[torch.Tensor] = []
    for index in indices:
        row_states: list[torch.Tensor] = []
        row_masks: list[torch.Tensor] = []
        for field in rows[index]["fields"]:
            state, mask = field_token_lookup[str(field["text"])]
            row_states.append(state)
            row_masks.append(mask)
        field_states.append(torch.stack(row_states, dim=0))
        field_masks.append(torch.stack(row_masks, dim=0))
    batch["field_token_states"] = torch.stack(
        field_states,
        dim=0,
    ).to(device)
    batch["field_content_mask"] = torch.stack(
        field_masks,
        dim=0,
    ).to(device)
    return batch


def parent_forward(
    graph: DualEndpointEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapted = ml.adapter_forward(adapter, batch)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--audit-receipt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=12)
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    audit_path = Path(args.audit_receipt).resolve()
    cache_path = Path(args.cache).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    map_path = Path(args.layer_map).resolve()
    semantic_config = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    output = Path(args.output_dir).resolve()

    for path in (
        audit_path,
        cache_path,
        curriculum_path,
        map_path,
        semantic_config,
        semantic_checkpoint / "alice_n0_v02.safetensors",
        tokenizer_dir / "tokenizer.json",
        adapter_path,
        graph_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing dual-view qualification input: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"refusing to overwrite dual-view qualification output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    expected = {
        audit_path: EXPECTED_AUDIT_SHA256,
        cache_path: EXPECTED_CACHE_SHA256,
        curriculum_path: EXPECTED_CURRICULUM_SHA256,
        map_path: EXPECTED_LAYER_MAP_SHA256,
        adapter_path: EXPECTED_PARENT_ADAPTER_SHA256,
        graph_path: EXPECTED_PARENT_GRAPH_SHA256,
    }
    for path, digest in expected.items():
        observed = sha256(path)
        if observed != digest:
            raise SystemExit(
                f"dual-view qualification lineage drift: {path.name} "
                f"{observed} != {digest}"
            )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_BINDING_IDENTIFIABILITY_AUDIT_NO_GRADIENT":
        raise SystemExit("binding identifiability audit did not pass")
    if audit.get("localization") != (
        "FIELD_POOLING_BOTTLENECK_SUPPORTED_TOKEN_LATE_INTERACTION_RECOVERS_BINDING"
    ):
        raise SystemExit("binding localization drift")
    comparisons = audit["diagnostic_comparisons"]
    if float(comparisons["dev_best_token_all_edge_accuracy"]) != 1.0:
        raise SystemExit("token-level all-edge identifiability drift")
    if float(comparisons["dev_best_token_same_relation_accuracy"]) != 1.0:
        raise SystemExit("token-level same-relation identifiability drift")
    final_layer = audit["dev_token_level_late_interaction"]["layers"]["16"]
    if float(final_layer["all_four_edge_top1_accuracy"]) != 1.0:
        raise SystemExit("final semantic token layer lost edge identifiability")
    if float(final_layer["same_relation_three_edge_top1_accuracy"]) != 1.0:
        raise SystemExit(
            "final semantic token layer lost same-relation edge identifiability"
        )
    if audit.get("setwise_training_authorized") is not False:
        raise SystemExit("source audit unexpectedly authorized setwise training")

    rows_all = read_jsonl(curriculum_path)
    rows = [row for row in rows_all if str(row["split"]) in {"train", "dev"}]
    payload = torch.load(cache_path, map_location="cpu")
    if len(rows) != len(payload["ids"]):
        raise SystemExit("cache/curriculum row-count drift")
    for i, row in enumerate(rows):
        if str(row["id"]) != str(payload["ids"][i]):
            raise SystemExit(f"cache/curriculum order drift at {i}")

    dev_indices = [
        i for i, split in enumerate(payload["splits"])
        if str(split) == "dev"
    ]
    if len(dev_indices) != 144:
        raise SystemExit(f"DEV row-count drift: {len(dev_indices)}")

    device = torch.device("cpu")
    tokenizer = load_tokenizer(tokenizer_dir)

    semantic_cfg = load_n0_config(semantic_config)
    semantic = AliceN0V02Model(semantic_cfg)
    semantic_state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"),
        device="cpu",
    )
    missing, unexpected = semantic.load_state_dict(
        semantic_state,
        strict=False,
    )
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint mismatch "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.eval()

    dev_field_texts = [
        str(field["text"])
        for index in dev_indices
        for field in rows[index]["fields"]
    ]
    field_token_lookup = encode_unique_final_field_tokens(
        model=semantic,
        tokenizer=tokenizer,
        texts=dev_field_texts,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        device=device,
    )
    query_masks = query_content_masks(
        tokenizer=tokenizer,
        rows=rows,
        indices=dev_indices,
        payload=payload,
        max_length=args.max_length,
    )
    del semantic, semantic_state

    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(
        load_file(str(adapter_path), device="cpu"),
        strict=True,
    )
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()

    parent_state = load_file(str(graph_path), device="cpu")
    baseline = DualEndpointEvidenceGraphEncoder(
        rr.expanded_graph_config()
    ).to(device)
    baseline.load_state_dict(parent_state, strict=True)
    for parameter in baseline.parameters():
        parameter.requires_grad = False
    baseline.eval()

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
    candidate = DualViewLateInteractionEvidenceGraphEncoder(
        config=rr.expanded_graph_config(),
        bridge=bridge,
    ).to(device)
    missing, unexpected = candidate.load_state_dict(
        parent_state,
        strict=False,
    )
    if any(
        not name.startswith("dual_view_query_edge_bridge.")
        for name in missing
    ) or unexpected:
        raise SystemExit(
            f"dual-view parent load drift "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )

    future_trainable = candidate.freeze_parent_for_bridge_training()
    parent_hash_before = tensor_state_sha256(
        candidate,
        exclude_prefixes=("dual_view_query_edge_bridge.",),
    )

    all_correct = same_correct = rows_seen = 0
    role_correct: dict[str, list[int]] = defaultdict(list)
    relation_correct: dict[str, list[int]] = defaultdict(list)
    route_sum_max_error = 0.0
    specialist_init_max_delta = 0.0
    exact_parent_weights = True
    exact_parent_pooled = True
    saved: dict[tuple[str, str, str], torch.Tensor] = {}
    permutation_checked = False
    permutation_max_delta = 0.0

    for start in range(0, len(dev_indices), args.eval_batch_size):
        indices = dev_indices[start : start + args.eval_batch_size]
        batch = batch_from_indices(
            payload=payload,
            rows=rows,
            indices=indices,
            query_masks=query_masks,
            field_token_lookup=field_token_lookup,
            device=device,
        )
        with torch.inference_mode():
            parent = parent_forward(baseline, adapter, batch)
            out = candidate_forward(candidate, adapter, batch)

        exact_parent_weights = exact_parent_weights and torch.equal(
            parent["field_weights"],
            out["field_weights"],
        )
        exact_parent_pooled = exact_parent_pooled and torch.equal(
            parent["pooled_state"],
            out["pooled_state"],
        )

        conditional = out["conditional_edge_probability"].cpu()
        route = out["route_probabilities_with_noop"].cpu()
        specialist = out["specialist_probability"].cpu()
        route_sum_max_error = max(
            route_sum_max_error,
            float((route.sum(dim=-1) - 1.0).abs().max().item()),
        )
        specialist_init_max_delta = max(
            specialist_init_max_delta,
            float((specialist - 0.5).abs().max().item()),
        )

        for pos, index in enumerate(indices):
            relevant = int(payload["relevant_edge_index"][index])
            all_pred = int(conditional[pos].argmax().item())
            same_pred = int(conditional[pos, :3].argmax().item())
            ok_all = int(all_pred == relevant)
            ok_same = int(same_pred == relevant)
            all_correct += ok_all
            same_correct += ok_same
            rows_seen += 1
            role = str(payload["query_roles"][index])
            relation = str(payload["relations"][index])
            role_correct[role].append(ok_all)
            relation_correct[relation].append(ok_all)

            key = (
                str(payload["quad_ids"][index]),
                str(payload["query_roles"][index]),
                str(payload["edge_direction_variants"][index]),
            )
            saved[key] = out["edge_binding_scores"][pos].detach().cpu()

        if not permutation_checked:
            permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
            inverse = torch.argsort(permutation)
            permuted = dict(batch)
            for key in (
                "edge_index",
                "edge_type_ids",
                "edge_confidence",
                "edge_valid_mask",
            ):
                permuted[key] = batch[key][:, permutation]
            with torch.inference_mode():
                permuted_out = candidate_forward(
                    candidate,
                    adapter,
                    permuted,
                )
            original_prob = out["conditional_edge_probability"]
            recovered = permuted_out[
                "conditional_edge_probability"
            ][:, inverse]
            permutation_max_delta = float(
                (original_prob - recovered).abs().max().item()
            )
            permutation_checked = True

    direction_max_delta = 0.0
    direction_pair_keys = {
        (
            str(payload["quad_ids"][index]),
            str(payload["query_roles"][index]),
        )
        for index in dev_indices
    }
    if len(direction_pair_keys) != 72:
        raise SystemExit(
            "direction-invariance unique pair coverage drift: "
            f"{len(direction_pair_keys)}"
        )

    quad_role_pairs = 0
    for qid, role in sorted(direction_pair_keys):
        a = saved.get((qid, role, "A"))
        b = saved.get((qid, role, "B"))
        if a is None or b is None:
            raise SystemExit(
                "direction-invariance missing A/B member for "
                f"{qid}:{role}"
            )
        direction_max_delta = max(
            direction_max_delta,
            float((a - b).abs().max().item()),
        )
        quad_role_pairs += 1
    if quad_role_pairs != 72:
        raise SystemExit(
            f"direction-invariance pair coverage drift: {quad_role_pairs}"
        )

    parent_hash_after = tensor_state_sha256(
        candidate,
        exclude_prefixes=("dual_view_query_edge_bridge.",),
    )
    parent_unchanged = parent_hash_before == parent_hash_after

    all_accuracy = all_correct / rows_seen
    same_accuracy = same_correct / rows_seen
    role_accuracy = {
        key: sum(values) / len(values)
        for key, values in sorted(role_correct.items())
    }
    relation_accuracy = {
        key: sum(values) / len(values)
        for key, values in sorted(relation_correct.items())
    }

    pass_gate = (
        all_accuracy == 1.0
        and same_accuracy == 1.0
        and all(value == 1.0 for value in role_accuracy.values())
        and all(value == 1.0 for value in relation_accuracy.values())
        and direction_max_delta <= 1e-6
        and permutation_max_delta <= 1e-6
        and route_sum_max_error <= 1e-6
        and specialist_init_max_delta <= 1e-7
        and exact_parent_weights
        and exact_parent_pooled
        and parent_unchanged
    )

    result = {
        "schema": SCHEMA,
        "status": (
            "PASS_DUAL_VIEW_LATE_INTERACTION_NO_GRADIENT_RUNTIME_CONTRACT"
            if pass_gate
            else "FAIL_DUAL_VIEW_LATE_INTERACTION_RUNTIME_CONTRACT"
        ),
        "git_revision": git_revision(root),
        "source_audit_sha256": sha256(audit_path),
        "source_cache_sha256": sha256(cache_path),
        "curriculum_sha256": sha256(curriculum_path),
        "layer_map_sha256": sha256(map_path),
        "parent_graph_sha256": sha256(graph_path),
        "parent_adapter_sha256": sha256(adapter_path),
        "source_audit_localization": audit["localization"],
        "final_token_layer_source_audit_all_edge_accuracy": float(
            final_layer["all_four_edge_top1_accuracy"]
        ),
        "final_token_layer_source_audit_same_relation_accuracy": float(
            final_layer["same_relation_three_edge_top1_accuracy"]
        ),
        "dev_rows": rows_seen,
        "dev_conditional_edge_all_four_top1_accuracy": all_accuracy,
        "dev_conditional_edge_same_relation_top1_accuracy": same_accuracy,
        "dev_query_role_all_edge_accuracy": role_accuracy,
        "dev_relation_family_all_edge_accuracy": relation_accuracy,
        "edge_binding_direction_reversal_max_delta": direction_max_delta,
        "edge_permutation_equivariance_max_delta": permutation_max_delta,
        "route_probability_sum_max_error": route_sum_max_error,
        "specialist_probability_initialization_max_delta_from_half": (
            specialist_init_max_delta
        ),
        "exact_parent_field_weights": exact_parent_weights,
        "exact_parent_pooled_state": exact_parent_pooled,
        "parent_parameters_exactly_unchanged": parent_unchanged,
        "future_trainable_scope": "dual_view_query_edge_bridge_only",
        "future_trainable_parameter_count": sum(
            p.numel() for p in future_trainable
        ),
        "binding_architecture": {
            "query_binding_layer": "final_semantic_token_layer",
            "hardcoded_dev_best_layer": False,
            "source_audit_dev_best_layer": int(
                audit["dev_token_level_late_interaction"][
                    "best_all_edge_layer"
                ]
            ),
            "field_binding_bypasses_structured_state": True,
            "edge_binding_score": (
                "mean_of_source_and_target_query_to_field_maxsim"
            ),
            "edge_binding_order_trainable": False,
            "edge_binding_scale_trainable": True,
            "source_target_pair_binding_symmetric": True,
            "parent_specialist_decision_separate_from_edge_identity": True,
            "setwise_edge_router_transformer": False,
            "flat_parent_plus_edge_router": False,
            "relation_direction_reasoning_remains_multilayer": True,
            "residual_runtime_control": (
                "specialist_probability_times_conditional_edge_probability"
            ),
        },
        "optimizer_created": False,
        "gradient_performed": False,
        "gpu_required": False,
        "training_authorized": False,
        "gpu_training_authorized": False,
        "causal_test_split_evaluated": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_evaluated": False,
        "frozen_challenge_rerun_authorized": False,
        "scale_authorized": False,
        "semantic_retraining_authorized": False,
        "graph_parent_retraining_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "review dual-view qualification and decide separately whether "
            "one bounded residual-plus-specialist-activation experiment is justified"
            if pass_gate
            else "stop_and_reopen_dual_view_runtime_contract"
        ),
    }

    result_path = output / "runtime_contract_qualification.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
