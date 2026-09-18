#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "alice.eipm.n0.relation-conditioned-layer-map.v0.1"
EXPECTED_AUDIT_SCHEMA = "alice.eipm.n0.query-semantics-layerwise-audit.v0.1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_candidates(
    ranked_layers: list[dict[str, Any]],
    *,
    max_candidates: int,
    accuracy_quantum: float,
) -> list[int]:
    if not ranked_layers:
        return []
    best = float(ranked_layers[0]["token_accuracy"])
    near = [
        item
        for item in ranked_layers
        if float(item["token_accuracy"]) >= best - accuracy_quantum - 1e-12
    ]
    exact_best = [
        int(item["layer_index"])
        for item in near
        if abs(float(item["token_accuracy"]) - best) <= 1e-12
    ]
    selected: list[int] = []
    for layer in exact_best:
        if layer not in selected:
            selected.append(layer)
        if len(selected) >= max_candidates:
            return sorted(selected)

    # Add near-best layers while preserving depth diversity rather than simply
    # taking adjacent layers from one plateau.
    remaining = [
        item for item in near if int(item["layer_index"]) not in selected
    ]
    if remaining:
        remaining = sorted(
            remaining,
            key=lambda item: (
                -float(item["token_accuracy"]),
                -float(item["token_pair_distance"]),
                int(item["layer_index"]),
            ),
        )
        while remaining and len(selected) < max_candidates:
            if not selected:
                chosen = remaining.pop(0)
            else:
                chosen = max(
                    remaining,
                    key=lambda item: (
                        min(
                            abs(int(item["layer_index"]) - existing)
                            for existing in selected
                        ),
                        float(item["token_accuracy"]),
                        float(item["token_pair_distance"]),
                        -int(item["layer_index"]),
                    ),
                )
                remaining.remove(chosen)
            selected.append(int(chosen["layer_index"]))
    return sorted(selected)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-candidates-per-relation", type=int, default=4)
    p.add_argument(
        "--accuracy-quantum",
        type=float,
        default=0.125,
        help="One observation step for the 8-example per-relation audit.",
    )
    args = p.parse_args()

    audit_path = Path(args.audit).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA:
        raise SystemExit("layerwise audit schema drift")
    if audit.get("gradient_performed") is not False:
        raise SystemExit("layerwise audit gradient contract drift")
    if audit.get("model_parameters_mutated") is not False:
        raise SystemExit("layerwise audit model-mutation contract drift")

    layers = audit.get("layer_results", {})
    if not layers:
        raise SystemExit("layerwise audit contains no layer results")

    relations: set[str] = set()
    for payload in layers.values():
        relations.update(payload.get("per_relation", {}).keys())
    if not relations:
        raise SystemExit("layerwise audit contains no relation metrics")

    relation_map: dict[str, Any] = {}
    union_candidates: set[int] = set()

    for relation in sorted(relations):
        ranked: list[dict[str, Any]] = []
        for key, payload in layers.items():
            per_relation = payload.get("per_relation", {}).get(relation)
            if per_relation is None:
                continue
            ranked.append(
                {
                    "layer_index": int(payload.get("layer_index", int(key))),
                    "token_accuracy": float(
                        per_relation["token_late_interaction_role_accuracy"]
                    ),
                    "pool_accuracy": float(
                        per_relation["mean_pool_role_accuracy"]
                    ),
                    "token_pair_distance": float(
                        payload["token_mean_opposite_role_pair_distance"]
                    ),
                    "is_embedding_output": bool(
                        payload.get("is_embedding_output", False)
                    ),
                    "is_final_layer": bool(payload.get("is_final_layer", False)),
                }
            )

        ranked.sort(
            key=lambda item: (
                -float(item["token_accuracy"]),
                -float(item["token_pair_distance"]),
                int(item["layer_index"]),
            )
        )
        best_accuracy = float(ranked[0]["token_accuracy"])
        exact_best = sorted(
            int(item["layer_index"])
            for item in ranked
            if abs(float(item["token_accuracy"]) - best_accuracy) <= 1e-12
        )
        near_best = sorted(
            int(item["layer_index"])
            for item in ranked
            if float(item["token_accuracy"])
            >= best_accuracy - args.accuracy_quantum - 1e-12
        )
        candidates = compact_candidates(
            ranked,
            max_candidates=args.max_candidates_per_relation,
            accuracy_quantum=args.accuracy_quantum,
        )
        union_candidates.update(candidates)

        relation_map[relation] = {
            "best_token_role_accuracy": best_accuracy,
            "exact_best_layers": exact_best,
            "near_best_layers_within_one_accuracy_quantum": near_best,
            "candidate_layers": candidates,
            "ranked_layers": ranked,
            "candidate_selection_is_permanent_architecture_limit": False,
        }

    payload = {
        "schema": SCHEMA,
        "status": "COMPILED_FROM_FROZEN_LAYERWISE_AUDIT",
        "source_audit_sha256": sha(audit_path),
        "source_audit_schema": EXPECTED_AUDIT_SCHEMA,
        "accuracy_quantum": args.accuracy_quantum,
        "max_candidates_per_relation_operating_budget":
            args.max_candidates_per_relation,
        "max_candidates_is_permanent_architecture_limit": False,
        "relation_map": relation_map,
        "global_candidate_layer_bank": sorted(union_candidates),
        "global_candidate_layer_bank_is_permanent_architecture_limit": False,
        "design_contract": {
            "no_single_universal_layer": True,
            "relation_conditioned_layer_selection": True,
            "audit_scores_are_priors_not_hard_routing_truth": True,
            "future_interface_may_mix_multiple_candidate_layers": True,
            "future_training_must_not_treat_map_as_historical_identity_authority": True,
        },
        "gradient_performed": False,
        "model_parameters_mutated": False,
        "training_authorized": False,
        "scale_authorized": False,
        "n0_complete": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== RELATION-CONDITIONED LAYER MAP =====")
    for relation, info in relation_map.items():
        print(
            f"{relation}: best={info['best_token_role_accuracy']} "
            f"exact={info['exact_best_layers']} "
            f"candidates={info['candidate_layers']}"
        )
    print(f"global_candidate_layer_bank={payload['global_candidate_layer_bank']}")
    print(f"output={output}")
    print("training_authorized=false")
    print("scale_authorized=false")
    print("n0_complete=false")


if __name__ == "__main__":
    main()
