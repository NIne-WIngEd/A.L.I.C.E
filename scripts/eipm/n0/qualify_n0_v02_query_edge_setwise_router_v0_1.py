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
from alice_personality.n0.evidence_graph_query_edge_setwise_router import (
    SetwiseQueryEdgeEvidenceGraphEncoder,
)
from alice_personality.n0.query_edge_setwise_router_bridge import (
    SetwiseQueryEdgeBridge,
)

import train_n0_v02_relation_repair as rr


SCHEMA = "alice.eipm.n0.query-edge-setwise-router-qualification.v0.1"
EXPECTED_FAILED_RESULT_SHA256 = (
    "08bef659779c2e6fcfb0d60ef209d9f6c7f03b0198bcf370e097e278f5436328"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
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


def parent_state_exact(
    parent: DualEndpointEvidenceGraphEncoder,
    candidate: SetwiseQueryEdgeEvidenceGraphEncoder,
) -> bool:
    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    for name, tensor in parent_state.items():
        other = candidate_state.get(name)
        if other is None or not torch.equal(tensor.cpu(), other.cpu()):
            return False
    return True


def make_models(
    *,
    parent_state: dict[str, torch.Tensor],
    map_path: Path,
    device: torch.device,
) -> tuple[DualEndpointEvidenceGraphEncoder, SetwiseQueryEdgeEvidenceGraphEncoder]:
    config = rr.expanded_graph_config()

    parent = DualEndpointEvidenceGraphEncoder(config).to(device)
    parent.load_state_dict(parent_state, strict=True)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad = False

    bridge = SetwiseQueryEdgeBridge.from_layer_map_file(
        layer_map_path=map_path,
        semantic_size=config.semantic_size,
        graph_size=config.graph_size,
        num_relation_types=config.num_relation_types,
        num_hidden_states=EXPECTED_HIDDEN_STATES,
        interface_size=config.graph_size,
        dropout=0.0,
        route_context_layers=2,
        route_context_heads=8,
        route_context_ffn_multiplier=4,
    )
    candidate = SetwiseQueryEdgeEvidenceGraphEncoder(
        config=config,
        bridge=bridge,
    ).to(device)
    missing, unexpected = candidate.load_state_dict(parent_state, strict=False)
    if any(
        not name.startswith("setwise_query_edge_bridge.")
        for name in missing
    ) or unexpected:
        raise SystemExit(
            f"parent load drift missing={list(missing)} unexpected={list(unexpected)}"
        )
    candidate.eval()
    return parent, candidate


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
    map_path = Path(args.layer_map).resolve()
    graph_path = Path(args.parent_graph).resolve()
    failed_path = Path(args.failed_result).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    anchors_path = Path(args.preservation_anchors).resolve()
    output = Path(args.output).resolve()

    for path in (
        map_path,
        graph_path,
        failed_path,
        curriculum_path,
        manifest_path,
        anchors_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing setwise-router qualification input: {path}")
    if output.exists():
        raise SystemExit("refusing to overwrite setwise-router qualification")
    output.parent.mkdir(parents=True, exist_ok=True)

    expected_hashes = {
        failed_path: EXPECTED_FAILED_RESULT_SHA256,
        graph_path: EXPECTED_PARENT_GRAPH_SHA256,
        curriculum_path: EXPECTED_CURRICULUM_SHA256,
        manifest_path: EXPECTED_MANIFEST_SHA256,
        anchors_path: EXPECTED_ANCHORS_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"qualification lineage drift: {path.name} "
                f"expected={expected} actual={actual}"
            )

    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    if failed.get("status") != (
        "FAIL_COMPETITIVE_EDGE_ROUTER_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
    ):
        raise SystemExit("source failure status drift")
    if failed.get("selected_checkpoint") is not None:
        raise SystemExit("source failure unexpectedly selected candidate")
    if failed.get("eligible_checkpoint_keys") not in ([], None):
        raise SystemExit("source failure unexpectedly has eligible checkpoints")
    if failed.get("automatic_rerun_or_hotfix_authorized") is not False:
        raise SystemExit("source failure unexpectedly authorized rerun")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("query_to_specific_edge_binding_required") is not True:
        raise SystemExit("query-edge binding curriculum drift")
    if manifest.get("test_split_opening_authorized") is not False:
        raise SystemExit("query-edge test unexpectedly opened")

    torch.manual_seed(20260920)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    parent_state = load_file(str(graph_path), device="cpu")
    parent, candidate = make_models(
        parent_state=parent_state,
        map_path=map_path,
        device=device,
    )

    future_trainable = candidate.freeze_parent_for_bridge_training()
    future_trainable_names = [
        name
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    ]
    if not future_trainable_names or any(
        not name.startswith("setwise_query_edge_bridge.")
        for name in future_trainable_names
    ):
        raise SystemExit("future trainable scope escapes setwise bridge")
    future_trainable_count = sum(p.numel() for p in future_trainable)

    for parameter in candidate.parameters():
        parameter.requires_grad = False

    config = rr.expanded_graph_config()
    batch = 2
    fields = 8
    tokens = 12
    edges = 6

    field_states = torch.randn(batch, fields, config.semantic_size)
    valid_mask = torch.ones(batch, fields, dtype=torch.bool)
    query_semantic = torch.randn(batch, config.semantic_size)
    base_field_weights = torch.softmax(torch.randn(batch, fields), dim=-1)
    edge_index = torch.tensor(
        [
            [[0,1],[2,3],[4,5],[6,7],[0,2],[0,0]],
            [[1,0],[3,2],[5,4],[7,6],[2,0],[0,0]],
        ],
        dtype=torch.long,
    )
    causes = int(EvidenceRelationType.CAUSES)
    supports = int(EvidenceRelationType.SUPPORTS)
    conflict = int(EvidenceRelationType.CONFLICTS_WITH)
    pad = int(EvidenceRelationType.PAD)
    edge_type_ids = torch.tensor(
        [
            [causes,causes,causes,supports,conflict,pad],
            [causes,causes,causes,supports,conflict,pad],
        ],
        dtype=torch.long,
    )
    edge_confidence = torch.ones(batch, edges, 1)
    edge_valid_mask = torch.tensor(
        [
            [True,True,True,True,True,False],
            [True,True,True,True,True,False],
        ],
        dtype=torch.bool,
    )
    attention_mask = torch.ones(batch, tokens, dtype=torch.long)
    hidden_states = tuple(
        torch.randn(batch, tokens, config.semantic_size)
        for _ in range(EXPECTED_HIDDEN_STATES)
    )

    with torch.inference_mode():
        parent_out = parent(
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
        raise SystemExit("zero-residual setwise bridge lost exact parent weights")
    if not torch.equal(parent_out["pooled_state"], candidate_out["pooled_state"]):
        raise SystemExit("zero-residual setwise bridge lost exact parent pooled state")
    if not parent_state_exact(parent, candidate):
        raise SystemExit("parent tensors changed during qualification")

    for key in ("source_residual_proposal", "target_residual_proposal"):
        value = candidate_out[key]
        if not torch.equal(value, torch.zeros_like(value)):
            raise SystemExit(f"{key} is not zero initialized")
    for key in ("source_residual", "target_residual"):
        value = candidate_out[key]
        if not torch.equal(value, torch.zeros_like(value)):
            raise SystemExit(f"{key} is not zero at initialization")

    route_probs = candidate_out["route_probabilities_with_noop"]
    if route_probs.shape != (batch, edges + 1):
        raise SystemExit("setwise route shape drift")
    if not torch.allclose(
        route_probs.sum(dim=-1),
        torch.ones(batch),
        atol=1e-7,
        rtol=0.0,
    ):
        raise SystemExit("setwise route probabilities do not sum to one")
    if not bool((candidate_out["noop_route_probability"] > 0).all().item()):
        raise SystemExit("parent/no-op route missing probability mass")

    edge_probs = candidate_out["edge_route_probability"]
    if not bool((edge_probs[:, 4].abs() <= 1e-12).all().item()):
        raise SystemExit("conflict edge received setwise route probability")
    if not bool((edge_probs[:, 5].abs() <= 1e-12).all().item()):
        raise SystemExit("PAD edge received setwise route probability")

    binding = candidate_out["edge_binding_query"]
    binding_delta = float((binding[:,0] - binding[:,1]).abs().max().item())
    if binding_delta <= 1e-8:
        raise SystemExit("same-relation edge binding collapsed")
    attention = candidate_out["relation_token_attention"]
    attention_delta = float((attention[:,0] - attention[:,1]).abs().max().item())
    if attention_delta <= 1e-10:
        raise SystemExit("edge content no longer changes token attention")

    # Prove edge-order permutation equivariance on the bridge itself.  There are
    # no positional edge embeddings, so reordering active edges must only
    # reorder the corresponding route probabilities.
    bridge = candidate.setwise_query_edge_bridge
    bridge.eval()
    graph_states = torch.randn(batch, fields, config.graph_size)
    with torch.inference_mode():
        base_bridge = bridge(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            graph_states=graph_states,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
        )
        perm = torch.tensor([2,0,1,3,4,5], dtype=torch.long)
        perm_bridge = bridge(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            graph_states=graph_states,
            edge_index=edge_index[:, perm],
            edge_type_ids=edge_type_ids[:, perm],
            edge_valid_mask=edge_valid_mask[:, perm],
            valid_mask=valid_mask,
        )

    noop_equivariant = torch.allclose(
        perm_bridge["route_probabilities_with_noop"][:, :1],
        base_bridge["route_probabilities_with_noop"][:, :1],
        atol=2e-6,
        rtol=0.0,
    )
    edges_equivariant = torch.allclose(
        perm_bridge["route_probabilities_with_noop"][:, 1:],
        base_bridge["route_probabilities_with_noop"][:, 1:][:, perm],
        atol=2e-6,
        rtol=0.0,
    )
    if not noop_equivariant or not edges_equivariant:
        raise SystemExit("edge-set route permutation equivariance failed")

    # Prove the new control plane is actually setwise.  Perturb an endpoint used
    # only by edge 0 and require the route logit for disjoint edge 1 to change.
    perturbed_states = graph_states.clone()
    perturbed_states[:, 0] = perturbed_states[:, 0] + 4.0
    with torch.inference_mode():
        perturbed_bridge = bridge(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            graph_states=perturbed_states,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_valid_mask=edge_valid_mask,
            valid_mask=valid_mask,
        )
    nonlocal_delta = float(
        (
            perturbed_bridge["edge_route_logits"][:, 1]
            - base_bridge["edge_route_logits"][:, 1]
        ).abs().max().item()
    )
    if nonlocal_delta <= 1e-9:
        raise SystemExit(
            "setwise router did not propagate competitor context across disjoint edges"
        )

    # Non-persisted route-control probe.  Every edge gets a constant non-zero
    # proposal so exact route probabilities must equal exact runtime multipliers.
    _, probe = make_models(
        parent_state=parent_state,
        map_path=map_path,
        device=device,
    )
    with torch.no_grad():
        probe.setwise_query_edge_bridge.source_residual_read.bias.fill_(1.0)
        probe.setwise_query_edge_bridge.target_residual_read.bias.fill_(-1.0)
    probe.eval()
    with torch.inference_mode():
        probe_out = probe(
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
    if not torch.allclose(
        probe_out["source_residual"],
        probe_out["edge_route_probability"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise SystemExit("route probability is not actual source contribution control")
    if not torch.allclose(
        probe_out["target_residual"],
        -probe_out["edge_route_probability"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise SystemExit("route probability is not actual target contribution control")

    report = candidate.parameter_report()
    required = {
        "setwise_contextual_routing": True,
        "joint_competitor_context": True,
        "permutation_equivariant_edge_set": True,
        "edge_order_positional_embedding": False,
        "explicit_parent_noop_route": True,
        "supervised_route_equals_runtime_contribution_route": True,
        "independent_source_target_residuals": True,
        "independent_per_edge_route_scoring": False,
        "independent_per_edge_tanh_gate": False,
        "proxy_dot_product_router": False,
        "parent_graph_can_remain_frozen": True,
        "hard_parameter_ceiling": None,
    }
    for key, value in required.items():
        if report.get(key) != value:
            raise SystemExit(f"setwise router report drift: {key}")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_SETWISE_QUERY_EDGE_ROUTER_NO_GRADIENT_RUNTIME_CONTRACT",
        "git_revision": git_revision(root),
        "source_failed_result_sha256": sha256(failed_path),
        "parent_graph_sha256": sha256(graph_path),
        "curriculum_sha256": sha256(curriculum_path),
        "curriculum_manifest_sha256": sha256(manifest_path),
        "preservation_anchors_sha256": sha256(anchors_path),
        "layer_map_sha256": sha256(map_path),
        "future_trainable_scope": "setwise_query_edge_bridge_only",
        "future_trainable_parameter_count": future_trainable_count,
        "parent_parameters_exactly_unchanged": True,
        "exact_parent_field_weights": True,
        "exact_parent_pooled_state": True,
        "zero_initialized_source_residual_readout": True,
        "zero_initialized_target_residual_readout": True,
        "route_probabilities_sum_to_one": True,
        "explicit_parent_noop_route": True,
        "all_active_directed_edges_jointly_contextualized": True,
        "edge_set_permutation_equivariance": True,
        "nonlocal_competitor_context": True,
        "nonlocal_competitor_route_logit_delta": nonlocal_delta,
        "conflict_directional_route_probability": False,
        "pad_route_probability": False,
        "same_relation_edges_produce_distinct_edge_queries": True,
        "edge_content_conditions_query_token_attention": True,
        "same_relation_edge_query_max_delta_01": binding_delta,
        "same_relation_token_attention_max_delta_01": attention_delta,
        "route_probability_is_actual_source_contribution_control": True,
        "route_probability_is_actual_target_contribution_control": True,
        "independent_per_edge_route_scoring": False,
        "proxy_dot_product_router": False,
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
            "review setwise route-control qualification and decide separately "
            "whether one bounded training objective is justified"
        ),
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
