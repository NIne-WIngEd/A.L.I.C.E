#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from alice_personality.n0.evidence_graph import EvidenceRelationType
from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_graph_query_edge_bridge import (
    QueryEdgeBridgeEvidenceGraphEncoder,
)
from alice_personality.n0.query_edge_cross_attention_bridge import (
    QueryEdgeCrossAttentionBridge,
)

import train_n0_v02_relation_repair as rr


SCHEMA = "alice.eipm.n0.query-edge-bridge-runtime-qualification.v0.1"
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_FAILED_RESULT_SHA256 = (
    "6f2f3fe6ddab764d947a7c63fe94ef24d5cd77e3cc68821c8678fdb8f2ea36bd"
)
EXPECTED_HIDDEN_STATES = 17


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def exact_parent_state_unchanged(
    baseline: DualEndpointEvidenceGraphEncoder,
    candidate: QueryEdgeBridgeEvidenceGraphEncoder,
) -> bool:
    base = baseline.state_dict()
    current = candidate.state_dict()
    for name, tensor in base.items():
        other = current.get(name)
        if other is None or not torch.equal(tensor.cpu(), other.cpu()):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--failed-result", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--curriculum-manifest", required=True)
    p.add_argument("--preservation-anchors", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    layer_map = Path(args.layer_map).resolve()
    parent_graph_path = Path(args.parent_graph).resolve()
    failed_result_path = Path(args.failed_result).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    curriculum_manifest_path = Path(args.curriculum_manifest).resolve()
    preservation_anchors_path = Path(args.preservation_anchors).resolve()
    output = Path(args.output).resolve()

    for path in (
        layer_map,
        parent_graph_path,
        failed_result_path,
        curriculum_path,
        curriculum_manifest_path,
        preservation_anchors_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing query-edge qualification input: {path}")
    if output.exists():
        raise SystemExit("refusing to overwrite query-edge qualification receipt")
    output.parent.mkdir(parents=True, exist_ok=True)

    if sha256(parent_graph_path) != EXPECTED_PARENT_GRAPH_SHA256:
        raise SystemExit("selected parent graph hash drift")
    if sha256(failed_result_path) != EXPECTED_FAILED_RESULT_SHA256:
        raise SystemExit("failed experiment result hash drift")

    failed = json.loads(failed_result_path.read_text(encoding="utf-8"))
    if failed.get("status") != (
        "FAIL_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
    ):
        raise SystemExit("failed experiment status drift")
    if failed.get("automatic_rerun_or_hotfix_authorized") is not False:
        raise SystemExit("failed experiment unexpectedly authorizes rerun")
    if failed.get("selected_checkpoint") is not None:
        raise SystemExit("failed experiment unexpectedly selected a checkpoint")

    manifest = json.loads(curriculum_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-query-edge-binding-curriculum.v0.1":
        raise SystemExit("query-edge curriculum schema drift")
    if manifest.get("compiled_sha256") != sha256(curriculum_path):
        raise SystemExit("query-edge curriculum hash drift")
    if manifest.get("query_to_specific_edge_binding_required") is not True:
        raise SystemExit("query-edge binding requirement missing")
    if manifest.get("same_relation_edges_per_row") != 3:
        raise SystemExit("same-relation distractor coverage drift")
    if manifest.get("gradient_authorized") is not False:
        raise SystemExit("gradient leaked into query-edge curriculum")

    anchors = json.loads(preservation_anchors_path.read_text(encoding="utf-8"))
    if anchors.get("schema") != (
        "alice.eipm.n0.query-edge-preservation-train-anchors.v0.1"
    ):
        raise SystemExit("preservation anchor schema drift")
    if anchors.get("gradient_authorized") is not False:
        raise SystemExit("preservation anchors prematurely authorize gradient")
    if anchors["gradient_eligibility"]["dev_rows_may_be_used_for_gradient"] is not False:
        raise SystemExit("preservation dev leakage")

    torch.manual_seed(20260919)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    config = rr.expanded_graph_config()
    parent_state = load_file(str(parent_graph_path), device="cpu")

    baseline = DualEndpointEvidenceGraphEncoder(config).to(device)
    baseline.load_state_dict(parent_state, strict=True)
    baseline.eval()
    for parameter in baseline.parameters():
        parameter.requires_grad = False

    bridge = QueryEdgeCrossAttentionBridge.from_layer_map_file(
        layer_map_path=layer_map,
        semantic_size=config.semantic_size,
        graph_size=config.graph_size,
        num_relation_types=config.num_relation_types,
        num_hidden_states=EXPECTED_HIDDEN_STATES,
        interface_size=config.graph_size,
        dropout=0.0,
    )
    candidate = QueryEdgeBridgeEvidenceGraphEncoder(
        config=config,
        bridge=bridge,
    ).to(device)
    missing, unexpected = candidate.load_state_dict(parent_state, strict=False)
    allowed_missing = [
        name for name in missing if name.startswith("query_edge_bridge.")
    ]
    if len(allowed_missing) != len(missing) or unexpected:
        raise SystemExit(
            f"parent checkpoint load drift missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )

    future_trainable = candidate.freeze_parent_for_bridge_training()
    trainable_names = [
        name for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("query_edge_bridge.") for name in trainable_names
    ):
        raise SystemExit("future trainable scope escapes query-edge bridge")
    future_trainable_parameters = sum(p.numel() for p in future_trainable)

    # Qualification itself is inference-only even though the future trainable
    # scope is inspected above.
    for parameter in candidate.parameters():
        parameter.requires_grad = False
    candidate.eval()

    batch = 2
    fields = 8
    tokens = 12
    edges = 5
    field_states = torch.randn(batch, fields, config.semantic_size, device=device)
    valid_mask = torch.ones(batch, fields, dtype=torch.bool, device=device)
    query_semantic = torch.randn(batch, config.semantic_size, device=device)
    base_field_weights = torch.softmax(
        torch.randn(batch, fields, device=device), dim=-1
    )
    edge_index = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5], [6, 7], [0, 0]],
            [[1, 0], [3, 2], [5, 4], [7, 6], [0, 0]],
        ],
        dtype=torch.long,
        device=device,
    )
    causes = int(EvidenceRelationType.CAUSES)
    conflict = int(EvidenceRelationType.CONFLICTS_WITH)
    pad = int(EvidenceRelationType.PAD)
    edge_type_ids = torch.tensor(
        [
            [causes, causes, causes, conflict, pad],
            [causes, causes, causes, conflict, pad],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_confidence = torch.ones(batch, edges, 1, device=device)
    edge_valid_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, False]],
        dtype=torch.bool,
        device=device,
    )
    attention_mask = torch.ones(batch, tokens, dtype=torch.long, device=device)
    hidden_states = tuple(
        torch.randn(batch, tokens, config.semantic_size, device=device)
        for _ in range(EXPECTED_HIDDEN_STATES)
    )

    with torch.inference_mode():
        parent_out = baseline(
            field_states=field_states,
            valid_mask=valid_mask,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            query_semantic=query_semantic,
            base_field_weights=base_field_weights,
        )
        candidate_out = candidate(
            field_states=field_states,
            valid_mask=valid_mask,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_confidence=edge_confidence,
            edge_valid_mask=edge_valid_mask,
            query_semantic=query_semantic,
            base_field_weights=base_field_weights,
            query_hidden_states=hidden_states,
            query_attention_mask=attention_mask,
        )

    if not torch.equal(parent_out["field_weights"], candidate_out["field_weights"]):
        delta = float(
            (parent_out["field_weights"] - candidate_out["field_weights"])
            .abs().max().item()
        )
        raise SystemExit(f"zero-gated bridge is not exact parent: weights delta={delta}")
    if not torch.equal(parent_out["pooled_state"], candidate_out["pooled_state"]):
        delta = float(
            (parent_out["pooled_state"] - candidate_out["pooled_state"])
            .abs().max().item()
        )
        raise SystemExit(f"zero-gated bridge is not exact parent: pooled delta={delta}")

    gate = candidate_out["edge_specialist_gate"]
    source_residual = candidate_out["source_residual"]
    target_residual = candidate_out["target_residual"]
    if not torch.equal(gate, torch.zeros_like(gate)):
        raise SystemExit("zero-init specialist gate contract failed")
    if not torch.equal(source_residual, torch.zeros_like(source_residual)):
        raise SystemExit("zero-init source residual contract failed")
    if not torch.equal(target_residual, torch.zeros_like(target_residual)):
        raise SystemExit("zero-init target residual contract failed")

    # Same-relation edges with different endpoint content must produce distinct
    # edge-conditioned queries before the zero gate.
    binding = candidate_out["edge_binding_query"]
    edge01_delta = float((binding[:, 0] - binding[:, 1]).abs().max().item())
    edge12_delta = float((binding[:, 1] - binding[:, 2]).abs().max().item())
    if edge01_delta <= 1e-8 or edge12_delta <= 1e-8:
        raise SystemExit("same-relation edges collapsed to one global query state")

    token_attention = candidate_out["relation_token_attention"]
    attention_edge_delta = float(
        (token_attention[:, 0] - token_attention[:, 1]).abs().max().item()
    )
    if attention_edge_delta <= 1e-10:
        raise SystemExit("edge content failed to condition query-token attention")

    has_interface = candidate_out["relation_has_interface"]
    if bool(has_interface[:, 3].any().item()):
        raise SystemExit("CONFLICTS_WITH received directional specialist")
    if bool(has_interface[:, 4].any().item()):
        raise SystemExit("PAD received specialist")

    report = candidate.parameter_report()
    required_report = {
        "edge_specific_query_binding": True,
        "independent_source_target_residuals": True,
        "forced_antisymmetry": False,
        "generic_endpoint_classifier": False,
        "raw_mean_pool_router": False,
        "hardcoded_single_layer": False,
        "parent_graph_can_remain_frozen": True,
        "hard_parameter_ceiling": None,
    }
    for key, value in required_report.items():
        if report.get(key) != value:
            raise SystemExit(f"architecture report drift: {key}")

    if not exact_parent_state_unchanged(baseline, candidate):
        raise SystemExit("parent tensors changed during no-gradient qualification")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_QUERY_EDGE_BRIDGE_NO_GRADIENT_RUNTIME_CONTRACT",
        "git_revision": git_revision(root),
        "failed_experiment_result_sha256": sha256(failed_result_path),
        "parent_graph_sha256": sha256(parent_graph_path),
        "layer_map_sha256": sha256(layer_map),
        "curriculum_sha256": sha256(curriculum_path),
        "curriculum_manifest_sha256": sha256(curriculum_manifest_path),
        "preservation_anchors_sha256": sha256(preservation_anchors_path),
        "future_trainable_parameter_count": future_trainable_parameters,
        "future_trainable_scope": "query_edge_bridge_only",
        "parent_parameters_exactly_unchanged": True,
        "exact_parent_field_weights": True,
        "exact_parent_pooled_state": True,
        "zero_initialized_specialist_gate": True,
        "same_relation_edges_produce_distinct_edge_queries": True,
        "edge_content_conditions_query_token_attention": True,
        "same_relation_edge_query_max_delta_01": edge01_delta,
        "same_relation_edge_query_max_delta_12": edge12_delta,
        "same_relation_token_attention_max_delta_01": attention_edge_delta,
        "conflicts_with_directional_specialist": False,
        "pad_specialist": False,
        "forced_antisymmetry": False,
        "optimizer_created": False,
        "gradient_performed": False,
        "gpu_required": False,
        "training_authorized": False,
        "gpu_training_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "scale_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "review qualified edge-specific architecture, multi-edge causal "
            "curriculum, and train-only preservation anchors before one explicit "
            "training-objective decision"
        ),
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
