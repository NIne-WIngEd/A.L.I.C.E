#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from alice_personality.n0.evidence_graph import EvidenceRelationType
from alice_personality.n0.relation_conditioned_multilayer_query import (
    RELATION_ID_TO_NAME,
    RelationConditionedMultiLayerQueryInterface,
)

SCHEMA = "alice.eipm.n0.relation-conditioned-multilayer-interface-qualification.v0.1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def tensor_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--layer-map", required=True)
    parser.add_argument("--expected-layer-map-sha256", required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(575804)

    root = Path(args.repo_root).resolve()
    layer_map_path = Path(args.layer_map).resolve()
    output = Path(args.output).resolve()

    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")
    if not layer_map_path.is_file():
        raise SystemExit(f"missing layer map: {layer_map_path}")

    observed_map_sha = sha256(layer_map_path)
    if observed_map_sha != args.expected_layer_map_sha256:
        raise SystemExit(
            "compiled layer-map hash drift: "
            f"{observed_map_sha} != {args.expected_layer_map_sha256}"
        )

    payload = json.loads(layer_map_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "alice.eipm.n0.relation-conditioned-layer-map.v0.1":
        raise SystemExit("compiled layer-map schema drift")
    if payload.get("status") != "COMPILED_FROM_FROZEN_LAYERWISE_AUDIT":
        raise SystemExit("compiled layer-map status drift")
    if payload.get("source_audit_sha256") != args.expected_source_audit_sha256:
        raise SystemExit("575804 source-audit binding drift")
    if payload.get("gradient_performed") is not False:
        raise SystemExit("layer map claims gradient activity")
    if payload.get("model_parameters_mutated") is not False:
        raise SystemExit("layer map claims parameter mutation")
    if payload.get("training_authorized") is not False:
        raise SystemExit("layer map prematurely authorizes training")
    if payload.get("scale_authorized") is not False:
        raise SystemExit("layer map prematurely authorizes scaling")
    if payload.get("n0_complete") is not False:
        raise SystemExit("layer map prematurely completes N0")
    if payload.get("hard_parameter_ceiling") is not None:
        raise SystemExit("hard parameter ceiling reintroduced")

    relation_map = payload.get("relation_map")
    if not isinstance(relation_map, dict):
        raise SystemExit("compiled layer map has no relation_map")

    directed_names = set(RELATION_ID_TO_NAME.values())
    if set(relation_map) != directed_names:
        raise SystemExit(
            "directed relation coverage drift: "
            f"expected={sorted(directed_names)} observed={sorted(relation_map)}"
        )

    candidate_sets: dict[str, tuple[int, ...]] = {}
    for name in sorted(directed_names):
        candidates = tuple(
            sorted({int(x) for x in relation_map[name].get("candidate_layers", [])})
        )
        if not candidates:
            raise SystemExit(f"directed relation {name} has no candidate layers")
        candidate_sets[name] = candidates

    candidate_union = sorted(
        {layer for values in candidate_sets.values() for layer in values}
    )
    if len(candidate_union) < 2:
        raise SystemExit("pathological universal single-layer map")
    if len(set(candidate_sets.values())) < 2:
        raise SystemExit("relation-conditioned map collapsed to one universal candidate set")

    max_layer = max(candidate_union)
    num_hidden_states = max_layer + 1
    semantic_size = 16
    graph_size = 12
    num_relation_types = int(EvidenceRelationType.TEMPORAL_SUCCESSOR) + 1

    interface = RelationConditionedMultiLayerQueryInterface.from_layer_map_file(
        layer_map_path=layer_map_path,
        semantic_size=semantic_size,
        graph_size=graph_size,
        num_relation_types=num_relation_types,
        num_hidden_states=num_hidden_states,
        interface_size=12,
        dropout=0.0,
        audit_prior_scale=2.0,
    )
    interface.eval()

    report = interface.parameter_report()
    require(report["single_universal_layer"] is False, "universal-layer drift")
    require(
        report["candidate_layer_mask_is_permanent_architecture_limit"] is False,
        "candidate mask became a permanent capability ceiling",
    )
    require(report["hard_parameter_ceiling"] is None, "hard parameter ceiling drift")
    require(report["parent_semantic_backbone_mutated"] is False, "parent mutation drift")

    before_sha = tensor_state_sha256(interface)
    before_grads = [parameter.grad for parameter in interface.parameters()]
    require(all(grad is None for grad in before_grads), "unexpected pre-existing gradients")

    batch = 2
    tokens = 7
    hidden_states: list[torch.Tensor] = []
    base = torch.linspace(
        -1.0,
        1.0,
        steps=batch * tokens * semantic_size,
        dtype=torch.float32,
    ).reshape(batch, tokens, semantic_size)
    for layer in range(num_hidden_states):
        hidden_states.append(base + float(layer) / 17.0)

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )

    relation_ids = [
        int(EvidenceRelationType.PAD),
        int(EvidenceRelationType.SUPPORTS),
        int(EvidenceRelationType.CORRECTS),
        int(EvidenceRelationType.SUPERSEDES),
        int(EvidenceRelationType.CONFLICTS_WITH),
        int(EvidenceRelationType.DERIVED_FROM),
        int(EvidenceRelationType.CAUSES),
        int(EvidenceRelationType.TEMPORAL_SUCCESSOR),
    ]
    edge_type_ids = torch.tensor([relation_ids, relation_ids], dtype=torch.long)
    edge_valid_mask = torch.ones_like(edge_type_ids, dtype=torch.bool)

    with torch.inference_mode():
        result = interface(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            edge_type_ids=edge_type_ids,
            edge_valid_mask=edge_valid_mask,
        )

    edge_state = result["edge_relation_query_state"]
    layer_weights = result["relation_layer_weights"]
    token_attention = result["relation_token_attention"]
    has_interface = result["relation_has_interface"]
    observed_candidate_mask = result["relation_layer_candidate_mask"]

    require(
        tuple(edge_state.shape) == (batch, len(relation_ids), graph_size),
        "edge semantic-state shape mismatch",
    )
    require(
        tuple(layer_weights.shape)
        == (batch, len(relation_ids), num_hidden_states),
        "layer-weight shape mismatch",
    )
    require(
        tuple(token_attention.shape)
        == (batch, len(relation_ids), num_hidden_states, tokens),
        "token-attention shape mismatch",
    )
    require(torch.isfinite(edge_state).all().item(), "non-finite edge states")
    require(torch.isfinite(layer_weights).all().item(), "non-finite layer weights")
    require(torch.isfinite(token_attention).all().item(), "non-finite token attention")

    directed_ids = set(RELATION_ID_TO_NAME)
    per_relation: dict[str, Any] = {}
    for edge_index, relation_id in enumerate(relation_ids):
        expected_has_interface = relation_id in directed_ids
        observed_has = bool(has_interface[:, edge_index].all().item())
        if expected_has_interface:
            require(observed_has, f"directed relation id {relation_id} lost interface")
            sums = layer_weights[:, edge_index].sum(dim=-1)
            require(
                torch.allclose(sums, torch.ones_like(sums), atol=1e-6),
                f"directed relation id {relation_id} layer weights do not sum to one",
            )
            mask = observed_candidate_mask[:, edge_index]
            non_candidate_weight = layer_weights[:, edge_index].masked_select(~mask)
            if non_candidate_weight.numel():
                require(
                    float(non_candidate_weight.abs().max()) <= 1e-7,
                    f"directed relation id {relation_id} leaked weight outside candidates",
                )
            relation_name = RELATION_ID_TO_NAME[relation_id]
            expected_layers = list(candidate_sets[relation_name])
            actual_layers = torch.nonzero(
                interface.relation_layer_candidate_mask[relation_id],
                as_tuple=False,
            ).flatten().tolist()
            require(
                actual_layers == expected_layers,
                f"{relation_name} candidate-mask mismatch",
            )
            per_relation[relation_name] = {
                "relation_id": relation_id,
                "candidate_layers": expected_layers,
                "layer_weight_sums": [float(x) for x in sums.tolist()],
                "has_interface": True,
            }
        else:
            require(
                not bool(has_interface[:, edge_index].any().item()),
                f"unsupported/symmetric relation id {relation_id} did not fail closed",
            )
            require(
                float(layer_weights[:, edge_index].abs().max()) <= 1e-7,
                f"unsupported/symmetric relation id {relation_id} has layer weight",
            )
            require(
                float(edge_state[:, edge_index].abs().max()) <= 1e-7,
                f"unsupported/symmetric relation id {relation_id} has semantic state",
            )

    for batch_index in range(batch):
        masked = ~attention_mask[batch_index]
        if masked.any():
            masked_attention = token_attention[
                batch_index, :, :, masked
            ]
            require(
                float(masked_attention.abs().max()) <= 1e-7,
                "masked query tokens received attention mass",
            )

    after_sha = tensor_state_sha256(interface)
    after_grads = [parameter.grad for parameter in interface.parameters()]
    require(before_sha == after_sha, "interface parameters mutated during qualification")
    require(all(grad is None for grad in after_grads), "qualification created gradients")

    receipt = {
        "schema": SCHEMA,
        "status": "PASS_EXACT_MAP_NO_GRADIENT_RUNTIME_CONTRACT",
        "git_revision": git_revision(root),
        "layer_map_sha256": observed_map_sha,
        "source_audit_sha256": payload["source_audit_sha256"],
        "source_audit_job": 575804,
        "candidate_layer_bank": candidate_union,
        "candidate_layer_sets": {
            key: list(value) for key, value in sorted(candidate_sets.items())
        },
        "relation_conditioned_candidate_sets_distinct": True,
        "universal_single_layer_rejected": True,
        "symmetric_conflicts_with_fails_closed": True,
        "pad_fails_closed": True,
        "directed_relation_count": len(directed_ids),
        "parameter_report": report,
        "parameter_state_sha256_before": before_sha,
        "parameter_state_sha256_after": after_sha,
        "parameter_state_exactly_unchanged": before_sha == after_sha,
        "gradient_performed": False,
        "optimizer_created": False,
        "parameter_gradients_created": False,
        "model_parameters_mutated": False,
        "gpu_required": False,
        "heldout_rows_used": False,
        "frozen_challenge_rows_used": False,
        "private_identity_data_used": False,
        "runtime_shapes": {
            "edge_relation_query_state": list(edge_state.shape),
            "relation_layer_weights": list(layer_weights.shape),
            "relation_token_attention": list(token_attention.shape),
        },
        "all_outputs_finite": True,
        "per_relation": per_relation,
        "training_authorized": False,
        "scale_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "n0_complete": False,
        "next_action": (
            "define_one_fresh_causal_interface_curriculum_and_one_preservation_contract_"
            "before_any_training_decision"
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== RELATION-CONDITIONED MULTILAYER INTERFACE QUALIFICATION =====")
    print(f"status={receipt['status']}")
    print(f"git_revision={receipt['git_revision']}")
    print(f"layer_map_sha256={receipt['layer_map_sha256']}")
    print(f"source_audit_sha256={receipt['source_audit_sha256']}")
    print(f"candidate_layer_bank={receipt['candidate_layer_bank']}")
    print(
        "parameter_state_exactly_unchanged="
        f"{receipt['parameter_state_exactly_unchanged']}"
    )
    print("gradient_performed=false")
    print("optimizer_created=false")
    print("gpu_required=false")
    print("training_authorized=false")
    print("scale_authorized=false")
    print("heldout_opening_authorized=false")
    print("frozen_challenge_rerun_authorized=false")
    print("n0_complete=false")
    print(f"next_action={receipt['next_action']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
