#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
    AdaptiveMultiViewLatentPoolV02,
)
from alice_personality.n0.qsre_n0_bridge import QSREN0EvidenceBridge
from alice_personality.n0.qsre_production_core import QSREProductionOperatorState
from alice_personality.n0.qsre_production_binder_v2 import (
    QSREProductionBinderV2,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import (
    config_from_plan,
    downstream_success,
    focus_metrics,
    load_plan,
    operator_metrics,
    paired_view_consistency,
    summarize_downstream,
    support_metrics,
)
from train_n0_v02_adaptive_multi_view_latent_pool_full_scale import (
    LatentPoolDataset,
    latent_forward,
    load_ratified_fusion,
    precompute_fusion_cache,
    to_device,
)
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_2_full_scale import (
    corrected_evaluate,
)
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale import (
    load_config as load_latent_config,
)
from train_n0_v02_cross_context_fusion_full_scale import build_parent_cache
from eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1 import (
    load_public_parents,
)
from train_n0_v02_qsre_production_p3_v3 import (
    bind,
    execute,
    infer_operator,
    load_parents,
    load_selected,
)


EXPECTED_LATENT_SHA256 = "503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4"
EXPECTED_FUSION_SHA256 = "4d51494beb788f74ddc03590da05ea00f0e36574294649c2cd5d438f9472e577"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def concat_operator_states(values: list[QSREProductionOperatorState]) -> QSREProductionOperatorState:
    return QSREProductionOperatorState(
        **{
            name: torch.cat(
                [getattr(value, name).detach().cpu() for value in values],
                dim=0,
            )
            for name in QSREProductionOperatorState.__dataclass_fields__
        }
    )


def qsre_final_metrics(
    *,
    split: dict[str, Any],
    schema,
    schema_encoder,
    executor,
    operator_model,
    binder,
    config,
    max_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    encoded = schema_encoder(schema)
    per_view_ops: list[list[QSREProductionOperatorState]] = [[], []]
    per_view_support: list[list[torch.Tensor]] = [[], []]
    per_view_compat: list[list[torch.Tensor]] = [[], []]
    per_view_focus: list[list[torch.Tensor]] = [[], []]
    per_view_probability: list[list[torch.Tensor]] = [[], []]
    per_view_summary: list[list[torch.Tensor]] = [[], []]

    for start in range(0, len(split["ids"]), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(split["ids"])))
        for view in (0, 1):
            op = infer_operator(
                split=split,
                indices=idx,
                view=view,
                schema=schema,
                encoded=encoded,
                operator=operator_model,
                max_steps=max_steps,
                device=device,
            )
            bound = bind(
                split=split,
                indices=idx,
                view=view,
                schema=schema,
                encoded=encoded,
                binder=binder,
                operator_state=op,
                device=device,
            )
            out = execute(
                split=split,
                indices=idx,
                encoded=encoded,
                executor=executor,
                operator_state=op,
                support=bound["edge_support_weight"],
                focus=bound["focus_field_weight"],
                device=device,
            )
            per_view_ops[view].append(op)
            per_view_support[view].append(bound["edge_support_weight"].detach().cpu())
            per_view_compat[view].append(bound["type_compatible"].detach().cpu())
            per_view_focus[view].append(bound["focus_field_weight"].detach().cpu())
            per_view_probability[view].append(out["relational_probability"].detach().cpu())
            per_view_summary[view].append(out["relational_summary"].detach().cpu())

    ops = [concat_operator_states(per_view_ops[0]), concat_operator_states(per_view_ops[1])]
    supports = [torch.cat(per_view_support[0]), torch.cat(per_view_support[1])]
    compatible = [torch.cat(per_view_compat[0]), torch.cat(per_view_compat[1])]
    focuses = [torch.cat(per_view_focus[0]), torch.cat(per_view_focus[1])]
    probability = [torch.cat(per_view_probability[0]), torch.cat(per_view_probability[1])]
    summaries = [torch.cat(per_view_summary[0]), torch.cat(per_view_summary[1])]
    indices = torch.arange(len(split["ids"]))

    view_results = []
    success_tensors = []
    for view in (0, 1):
        om = operator_metrics(operator=ops[view], split=split, indices=indices)
        om_public = {
            key: value
            for key, value in om.items()
            if not key.endswith("_tensor")
        }
        sm = support_metrics(
            predicted=supports[view],
            oracle=split["oracle_edge_support"].float(),
            type_compatible=compatible[view],
        )
        fm = focus_metrics(
            predicted=focuses[view],
            oracle=split["focus_field_weight"].float(),
            traversal_target=split["traversal_target"].long(),
        )
        dm = summarize_downstream(
            probability=probability[view],
            target=split["target_distribution"].float(),
            control_target=split["control_target"].long(),
            families=split["families"],
            open_schema=split["open_schema"].bool(),
            causal_groups=split["causal_groups"],
        )
        success = dm.pop("row_success_tensor")
        success_tensors.append(success)
        nonrel = split["control_target"].ne(1)
        false_assertion = (
            probability[view][nonrel].sum(dim=-1).gt(0.01).float().mean().item()
            if bool(nonrel.any())
            else 0.0
        )
        dm["nonrelational_false_assertion_rate"] = float(false_assertion)
        final_only = split["final_only_relation"].bool()
        dm["final_only_relation_success"] = (
            float(success[final_only].float().mean().item())
            if bool(final_only.any())
            else 1.0
        )
        view_results.append(
            {"operator": om_public, "support": sm, "focus": fm, "downstream": dm}
        )

    family_keys = (
        set(view_results[0]["downstream"]["family_success"])
        | set(view_results[1]["downstream"]["family_success"])
    )
    family_worst = {
        family: min(
            float(view_results[0]["downstream"]["family_success"].get(family, 1.0)),
            float(view_results[1]["downstream"]["family_success"].get(family, 1.0)),
        )
        for family in sorted(family_keys)
    }

    aggregate = {
        "downstream_row_success_accuracy": min(
            float(v["downstream"]["row_success_accuracy"]) for v in view_results
        ),
        "family_min_success": min(family_worst.values()) if family_worst else 1.0,
        "family_success": family_worst,
        "final_only_relation_success": min(
            float(v["downstream"]["final_only_relation_success"]) for v in view_results
        ),
        "open_schema_success": min(
            float(v["downstream"]["open_schema_success"]) for v in view_results
        ),
        "relation_sequence_exact_accuracy": min(
            float(v["operator"]["relation_sequence_exact_accuracy"]) for v in view_results
        ),
        "direction_accuracy_relational": min(
            float(v["operator"]["direction_accuracy_relational"]) for v in view_results
        ),
        "role_accuracy_relational": min(
            float(v["operator"]["role_accuracy_relational"]) for v in view_results
        ),
        "traversal_accuracy_relational": min(
            float(v["operator"]["traversal_accuracy_relational"]) for v in view_results
        ),
        "modifier_exact_accuracy_relational": min(
            float(v["operator"]["modifier_exact_accuracy_relational"]) for v in view_results
        ),
        "support_edge_f1": min(
            float(v["support"]["edge_f1"]) for v in view_results
        ),
        "path_focus_top1_accuracy": min(
            float(v["focus"]["path_focus_top1_accuracy"]) for v in view_results
        ),
        "unknown_fail_closed_accuracy": min(
            float(v["operator"]["unknown_termination_accuracy"]) for v in view_results
        ),
        "paraphrase_pair_consistency": paired_view_consistency(
            first=ops[0], second=ops[1]
        ),
        "nonrelational_false_assertion_rate": max(
            float(v["downstream"]["nonrelational_false_assertion_rate"])
            for v in view_results
        ),
        "outside_support_invariance_max_delta": max(
            float(v["downstream"]["outside_support_invariance_max_delta"])
            for v in view_results
        ),
    }
    runtime = {
        "probability": torch.stack(probability, dim=1),
        "relational_summary": torch.stack(summaries, dim=1),
        "support": torch.stack(supports, dim=1),
        "focus": torch.stack(focuses, dim=1),
    }
    return {"aggregate": aggregate, "view_results": view_results}, runtime


def latent_row_scores(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    dataset: LatentPoolDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    best_parts = []
    pooled_parts = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = latent_forward(model, batch)
            target = F.normalize(batch["semantic_target"].float(), dim=-1).unsqueeze(1)
            best = (
                F.normalize(output["latent_slots"].float(), dim=-1) * target
            ).sum(dim=-1).max(dim=-1).values
            pooled = F.cosine_similarity(
                output["pooled_state"].float(),
                batch["semantic_target"].float(),
                dim=-1,
            )
            best_parts.append(best.cpu())
            pooled_parts.append(pooled.cpu())
    return torch.cat(best_parts), torch.cat(pooled_parts)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def duplicate_relational_parent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        for view in (0, 1):
            item = copy.deepcopy(row)
            item["id"] = f"{row['id']}:view{view}"
            item["query"] = str(row["query_views"][view])
            item["query_text"] = str(row["query_views"][view])
            item["split"] = "final_self_validation"
            out.append(item)
    return out


def gate_general(metrics: dict[str, Any], threshold: dict[str, Any]) -> dict[str, bool]:
    return {
        "best_slot_semantic": float(metrics["best_slot_semantic_cosine"])
        >= float(threshold["best_slot_semantic_cosine_min"]),
        "family_min_best_slot": float(metrics["family_min_best_slot_semantic_cosine"])
        >= float(threshold["family_min_best_slot_semantic_cosine_min"]),
        "pooled_semantic": float(metrics["pooled_semantic_cosine"])
        >= float(threshold["pooled_semantic_cosine_min"]),
        "family_min_pooled": float(metrics["family_min_pooled_semantic_cosine"])
        >= float(threshold["family_min_pooled_semantic_cosine_min"]),
        "slot_diversity": float(metrics["mean_pairwise_offdiag_slot_cosine"])
        <= float(threshold["mean_pairwise_offdiag_slot_cosine_max"]),
        "effective_rank": float(metrics["mean_centered_slot_effective_rank"])
        >= float(threshold["mean_centered_slot_effective_rank_min"]),
        "source_view_recoverability": float(
            metrics["mean_min_available_view_best_slot_semantic_cosine"]
        )
        >= float(threshold["mean_min_available_view_best_slot_semantic_cosine_min"]),
        "view_specialization": float(
            metrics["mean_disagreement_weighted_view_specialization"]
        )
        >= float(threshold["mean_disagreement_weighted_view_specialization_min"]),
        "view_attention": float(metrics["mean_min_available_view_best_slot_attention"])
        >= float(threshold["mean_min_available_view_best_slot_attention_min"]),
        "channel_attention": float(metrics["mean_min_channel_best_slot_attention"])
        >= float(threshold["mean_min_channel_best_slot_attention_min"]),
        "missing_view_safety": float(metrics["missing_view_attention_max"])
        <= float(threshold["missing_view_attention_max"]),
    }


def gate_qsre(metrics: dict[str, Any], threshold: dict[str, Any]) -> dict[str, bool]:
    return {
        "downstream": metrics["downstream_row_success_accuracy"]
        >= threshold["downstream_row_success_accuracy_min"],
        "family_floor": metrics["family_min_success"]
        >= threshold["family_min_success_min"],
        "final_only": metrics["final_only_relation_success"]
        >= threshold["final_only_relation_success_min"],
        "open_schema": metrics["open_schema_success"]
        >= threshold["open_schema_success_min"],
        "relation_sequence": metrics["relation_sequence_exact_accuracy"]
        >= threshold["relation_sequence_exact_accuracy_min"],
        "direction": metrics["direction_accuracy_relational"]
        >= threshold["direction_accuracy_relational_min"],
        "role": metrics["role_accuracy_relational"]
        >= threshold["role_accuracy_relational_min"],
        "traversal": metrics["traversal_accuracy_relational"]
        >= threshold["traversal_accuracy_relational_min"],
        "modifiers": metrics["modifier_exact_accuracy_relational"]
        >= threshold["modifier_exact_accuracy_relational_min"],
        "support": metrics["support_edge_f1"]
        >= threshold["support_edge_f1_min"],
        "focus": metrics["path_focus_top1_accuracy"]
        >= threshold["path_focus_top1_accuracy_min"],
        "unknown": metrics["unknown_fail_closed_accuracy"]
        >= threshold["unknown_fail_closed_accuracy_min"],
        "paraphrase": metrics["paraphrase_pair_consistency"]
        >= threshold["paraphrase_pair_consistency_min"],
        "nonrelational": metrics["nonrelational_false_assertion_rate"]
        <= threshold["nonrelational_false_assertion_rate_max"],
        "outside_support": metrics["outside_support_invariance_max_delta"]
        <= threshold["outside_support_invariance_max_delta"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--freeze-receipt", required=True)
    p.add_argument("--general-corpus", required=True)
    p.add_argument("--relational-corpus", required=True)
    p.add_argument("--relational-cache", required=True)
    p.add_argument("--final-schema-cache", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--p1-result", required=True)
    p.add_argument("--p1-root", required=True)
    p.add_argument("--p2-result", required=True)
    p.add_argument("--p2-root", required=True)
    p.add_argument("--factor-schema-cache", required=True)
    p.add_argument("--p3-result", required=True)
    p.add_argument("--p3-root", required=True)
    p.add_argument("--p4-result", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--structured-config", required=True)
    p.add_argument("--structured-checkpoint", required=True)
    p.add_argument("--evidence-adapter", required=True)
    p.add_argument("--evidence-graph", required=True)
    p.add_argument("--fusion-checkpoint", required=True)
    p.add_argument("--fusion-config", required=True)
    p.add_argument("--fusion-ratification", required=True)
    p.add_argument("--latent-candidate", required=True)
    p.add_argument("--latent-config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("final native N0 self-validation requires CUDA")
    device = torch.device("cuda")

    contract_path = Path(args.contract)
    manifest_path = Path(args.manifest)
    freeze_path = Path(args.freeze_receipt)
    general_path = Path(args.general_corpus)
    relational_path = Path(args.relational_corpus)
    relational_cache_path = Path(args.relational_cache)
    final_schema_cache_path = Path(args.final_schema_cache)
    plan_path = Path(args.plan)
    p4_path = Path(args.p4_result)

    contract = json.loads(contract_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if contract.get("status") != "FROZEN_NATIVE_N0_OBJECTIVE_GATE_BEFORE_PRODUCTION_RESULTS":
        raise SystemExit("final N0 objective contract drift")
    if manifest.get("status") != "FROZEN_BY_BUILD_SOURCE_BEFORE_PRODUCTION_MODEL_RESULTS":
        raise SystemExit("final N0 validation manifest is not frozen")
    if freeze.get("status") != "FROZEN_N0_FINAL_SELF_VALIDATION_BEFORE_P1_RESULTS":
        raise SystemExit("final N0 validation freeze receipt missing")
    for path, key in (
        (contract_path, "contract_sha256"),
        (manifest_path, "manifest_sha256"),
        (general_path, "general_sha256"),
        (relational_path, "relational_sha256"),
        (final_schema_cache_path, "final_schema_cache_sha256"),
        (relational_cache_path, "relational_cache_sha256"),
    ):
        if freeze.get(key) != sha256(path):
            raise SystemExit(f"final N0 frozen artifact drift: {key}")
    if freeze.get("training_authorized") is not False:
        raise SystemExit("final validation leaked into training authority")

    general_rows = read_jsonl(general_path)
    relational_rows = read_jsonl(relational_path)
    if len(general_rows) != int(contract["corpus"]["expected_general_rows"]):
        raise SystemExit("final general row count drift")
    if len(relational_rows) != int(contract["corpus"]["expected_relational_rows"]):
        raise SystemExit("final relational row count drift")

    plan = load_plan(plan_path)
    config = config_from_plan(plan)
    p1_path, _ = load_selected(
        Path(args.p1_result), Path(args.p1_root),
        "qsre_production_p1.pt", "PASS_QSRE_PRODUCTION_P1_EXECUTOR",
    )
    p2_path, _ = load_selected(
        Path(args.p2_result), Path(args.p2_root),
        "qsre_production_p2.pt", "PASS_QSRE_PRODUCTION_P2_OPERATOR",
    )
    p3_path, _ = load_selected(
        Path(args.p3_result), Path(args.p3_root),
        "qsre_production_p3.pt", "PASS_QSRE_PRODUCTION_P3_BINDER",
    )
    p4 = json.loads(p4_path.read_text())
    if p4.get("status") != "PASS_QSRE_PRODUCTION_P4_END_TO_END":
        raise SystemExit("P4 did not authorize final N0 self-validation")
    if (
        p4.get("p1_checkpoint_sha256") != sha256(p1_path)
        or p4.get("p2_checkpoint_sha256") != sha256(p2_path)
        or p4.get("p3_checkpoint_sha256") != sha256(p3_path)
    ):
        raise SystemExit("P4 selected lineage drift")

    schema_encoder, executor, operator_model = load_parents(
        p1_path=p1_path,
        p2_path=p2_path,
        factor_schema_cache_path=Path(args.factor_schema_cache),
        config=config,
        device=device,
    )
    binder = QSREProductionBinderV2(config)
    binder.load_state_dict(torch.load(p3_path, map_location="cpu")["binder"], strict=True)
    for parameter in binder.parameters():
        parameter.requires_grad = False
    binder = binder.to(device).eval()

    final_schema, schema_payload = load_dynamic_schema_cache(
        final_schema_cache_path, device=device
    )
    final_only_keys = set(contract["corpus"]["final_only_relation_keys"])
    if not final_only_keys.issubset(set(schema_payload["relation_keys"])):
        raise SystemExit("final-only dynamic relation schema missing")
    relational_cache = torch.load(relational_cache_path, map_location="cpu")
    if relational_cache.get("schema") != "alice.eipm.n0.qsre-final-self-validation-cache.v1":
        raise SystemExit("final relational cache schema drift")
    split = relational_cache["rows"]

    qsre_result, qsre_runtime = qsre_final_metrics(
        split=split,
        schema=final_schema,
        schema_encoder=schema_encoder,
        executor=executor,
        operator_model=operator_model,
        binder=binder,
        config=config,
        max_steps=int(plan["stages"]["P4"]["operator_runtime_max_steps"]),
        batch_size=8,
        device=device,
    )
    qsre_checks = gate_qsre(
        qsre_result["aggregate"], contract["qsre_gate"]
    )

    del binder, operator_model, executor, schema_encoder, final_schema
    torch.cuda.empty_cache()

    semantic_model, tokenizer, structured, adapter, graph = load_public_parents(
        semantic_config_path=Path(args.semantic_config),
        semantic_checkpoint=Path(args.semantic_checkpoint).parent,
        tokenizer_dir=Path(args.tokenizer_dir),
        structured_config_path=Path(args.structured_config),
        structured_checkpoint=Path(args.structured_checkpoint),
        evidence_adapter_path=Path(args.evidence_adapter),
        evidence_graph_path=Path(args.evidence_graph),
        device=device,
    )
    general_parent = build_parent_cache(
        general_rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured=structured,
        evidence_adapter=adapter,
        evidence_graph=graph,
        device=device,
        encode_batch_size=16,
        raw_max_length=96,
        field_max_length=96,
    )
    relational_parent_rows = duplicate_relational_parent_rows(relational_rows)
    relational_parent = build_parent_cache(
        relational_parent_rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured=structured,
        evidence_adapter=adapter,
        evidence_graph=graph,
        device=device,
        encode_batch_size=16,
        raw_max_length=96,
        field_max_length=96,
    )
    del semantic_model, structured, adapter, graph
    torch.cuda.empty_cache()

    fusion = load_ratified_fusion(
        checkpoint_dir=Path(args.fusion_checkpoint),
        fusion_config_path=Path(args.fusion_config),
        ratification_path=Path(args.fusion_ratification),
        device=device,
    )
    fusion_model_path = Path(args.fusion_checkpoint) / "cross_context_fusion.safetensors"
    if sha256(fusion_model_path) != EXPECTED_FUSION_SHA256:
        raise SystemExit("final validation fusion candidate drift")

    latent_path = Path(args.latent_candidate)
    if sha256(latent_path) != EXPECTED_LATENT_SHA256:
        raise SystemExit("final validation latent candidate drift")
    latent = AdaptiveMultiViewLatentPoolV02(
        load_latent_config(Path(args.latent_config))
    )
    latent.load_state_dict(load_file(str(latent_path), device="cpu"), strict=True)
    for parameter in latent.parameters():
        parameter.requires_grad = False
    latent = latent.to(device).eval()

    general_fusion = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=general_parent,
        device=device,
        batch_size=16,
    )
    general_dataset = LatentPoolDataset(
        general_parent, general_fusion, list(range(len(general_rows)))
    )
    general_metrics = corrected_evaluate(
        model=latent,
        dataset=general_dataset,
        parent_cache=general_parent,
        device=device,
        batch_size=16,
    )
    general_checks = gate_general(
        general_metrics, contract["general_fabric_gate"]
    )

    # Flatten QSRE view order to the duplicated parent order:
    # row0/view0,row0/view1,row1/view0,row1/view1,...
    probability = qsre_runtime["probability"].reshape(
        -1, qsre_runtime["probability"].size(-1)
    )
    relational_summary = qsre_runtime["relational_summary"].reshape(
        -1, qsre_runtime["relational_summary"].size(-1)
    )
    field_state = split["field_state"].repeat_interleave(2, dim=0).float()
    field_valid = split["field_valid_mask"].repeat_interleave(2, dim=0).bool()

    if relational_summary.size(-1) != int(plan["model"]["semantic_dim"]):
        raise SystemExit(
            "Production QSRE relational summary width does not match N0 semantic fabric"
        )
    bridge = QSREN0EvidenceBridge(semantic_dim=int(plan["model"]["semantic_dim"]))
    bridged = bridge(
        base_evidence_tokens=relational_parent["evidence_tokens"].float(),
        base_evidence_mask=relational_parent["evidence_valid_mask"].bool(),
        field_semantic_state=field_state,
        field_valid_mask=field_valid,
        relational_probability=probability.float(),
        relational_summary=relational_summary.float(),
    )
    augmented_parent = copy.deepcopy(relational_parent)
    augmented_parent["evidence_tokens"] = bridged.evidence_tokens
    augmented_parent["evidence_valid_mask"] = bridged.evidence_mask
    augmented_parent["source_view_summaries"] = (
        augmented_parent["source_view_summaries"].clone()
    )
    augmented_parent["source_view_summaries"][:, 2] = masked_mean(
        bridged.evidence_tokens, bridged.evidence_mask
    )

    # Keep the ablation tensor geometry identical to the integrated path.
    # Otherwise merely adding two masked token positions can change fp16
    # attention kernels and contaminate the causal QSRE delta with a sequence-
    # shape effect. The ablation removes QSRE content and validity, not tensor
    # shape, routing topology, batch order, or the ratified parent evidence.
    ablated_parent = copy.deepcopy(relational_parent)
    zero_extra = torch.zeros(
        relational_parent["evidence_tokens"].size(0),
        2,
        relational_parent["evidence_tokens"].size(-1),
        dtype=relational_parent["evidence_tokens"].dtype,
    )
    false_extra = torch.zeros(
        relational_parent["evidence_valid_mask"].size(0),
        2,
        dtype=torch.bool,
    )
    ablated_parent["evidence_tokens"] = torch.cat(
        [relational_parent["evidence_tokens"], zero_extra],
        dim=1,
    )
    ablated_parent["evidence_valid_mask"] = torch.cat(
        [relational_parent["evidence_valid_mask"], false_extra],
        dim=1,
    )
    ablated_parent["source_view_summaries"] = (
        relational_parent["source_view_summaries"].clone()
    )
    ablated_parent["source_view_summaries"][:, 2] = masked_mean(
        ablated_parent["evidence_tokens"],
        ablated_parent["evidence_valid_mask"],
    )

    if ablated_parent["evidence_tokens"].shape != augmented_parent["evidence_tokens"].shape:
        raise RuntimeError("QSRE ablation evidence-token geometry drift")
    if ablated_parent["evidence_valid_mask"].shape != augmented_parent["evidence_valid_mask"].shape:
        raise RuntimeError("QSRE ablation evidence-mask geometry drift")

    augmented_fusion = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=augmented_parent,
        device=device,
        batch_size=16,
    )
    ablated_fusion = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=ablated_parent,
        device=device,
        batch_size=16,
    )
    augmented_dataset = LatentPoolDataset(
        augmented_parent,
        augmented_fusion,
        list(range(len(relational_parent_rows))),
    )
    ablated_dataset = LatentPoolDataset(
        ablated_parent,
        ablated_fusion,
        list(range(len(relational_parent_rows))),
    )
    integrated_metrics = corrected_evaluate(
        model=latent,
        dataset=augmented_dataset,
        parent_cache=augmented_parent,
        device=device,
        batch_size=16,
    )
    normal_best, normal_pooled = latent_row_scores(
        model=latent,
        dataset=augmented_dataset,
        device=device,
        batch_size=16,
    )
    ablated_best, ablated_pooled = latent_row_scores(
        model=latent,
        dataset=ablated_dataset,
        device=device,
        batch_size=16,
    )
    delta = normal_best - ablated_best

    required_row = split[
        "relational_required_for_integrated_answer"
    ].repeat_interleave(2)
    final_only_row = split["final_only_relation"].repeat_interleave(2)
    nonrel_row = split["control_target"].ne(1).repeat_interleave(2)
    families = [
        family
        for family in split["families"]
        for _ in (0, 1)
    ]

    family_drop: dict[str, list[float]] = defaultdict(list)
    for index, family in enumerate(families):
        if bool(required_row[index]):
            family_drop[str(family)].append(float(delta[index]))
    family_drop_mean = {
        family: sum(values) / len(values)
        for family, values in sorted(family_drop.items())
    }
    required_mean_drop = (
        float(delta[required_row].mean().item())
        if bool(required_row.any())
        else 0.0
    )
    required_family_min_drop = (
        min(family_drop_mean.values()) if family_drop_mean else 0.0
    )
    final_only_mean_drop = (
        float(delta[final_only_row].mean().item())
        if bool(final_only_row.any())
        else 0.0
    )
    final_only_best = (
        float(normal_best[final_only_row].mean().item())
        if bool(final_only_row.any())
        else 1.0
    )
    nonrel_delta = (
        float((normal_best[nonrel_row] - ablated_best[nonrel_row]).abs().max().item())
        if bool(nonrel_row.any())
        else 0.0
    )

    integrated_summary = {
        **integrated_metrics,
        "final_only_best_slot_semantic_cosine": final_only_best,
        "relational_required_mean_qsre_ablation_drop": required_mean_drop,
        "relational_required_family_mean_qsre_ablation_drop": family_drop_mean,
        "relational_required_family_min_qsre_ablation_drop": required_family_min_drop,
        "final_only_mean_qsre_ablation_drop": final_only_mean_drop,
        "nonrelational_base_preservation_max_abs_delta": nonrel_delta,
        "relational_required_rows": int(required_row.sum().item()),
        "final_only_rows": int(final_only_row.sum().item()),
    }
    ig = contract["integrated_relational_fabric_gate"]
    integrated_checks = {
        "best_slot": integrated_metrics["best_slot_semantic_cosine"]
        >= ig["best_slot_semantic_cosine_min"],
        "family_min_best": integrated_metrics["family_min_best_slot_semantic_cosine"]
        >= ig["family_min_best_slot_semantic_cosine_min"],
        "pooled": integrated_metrics["pooled_semantic_cosine"]
        >= ig["pooled_semantic_cosine_min"],
        "family_min_pooled": integrated_metrics["family_min_pooled_semantic_cosine"]
        >= ig["family_min_pooled_semantic_cosine_min"],
        "final_only_best": final_only_best
        >= ig["final_only_best_slot_semantic_cosine_min"],
        "required_ablation_mean": required_mean_drop
        >= ig["relational_required_mean_qsre_ablation_drop_min"],
        "required_ablation_family_floor": required_family_min_drop
        >= ig["relational_required_family_min_qsre_ablation_drop_min"],
        "final_only_ablation": final_only_mean_drop
        >= ig["final_only_mean_qsre_ablation_drop_min"],
        "nonrelational_preservation": nonrel_delta
        <= ig["nonrelational_base_preservation_max_abs_delta"],
    }

    all_checks = {
        "qsre": qsre_checks,
        "general_fabric": general_checks,
        "integrated_relational_fabric": integrated_checks,
    }
    passed = all(
        all(block.values()) for block in all_checks.values()
    )

    result = {
        "schema": "alice.eipm.n0.final-self-validation-result.v3",
        "qsre_architecture": "operator-v3-plus-binder-v2",
        "status": (
            "PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE"
            if passed
            else "FAIL_N0_FINAL_SELF_VALIDATION_STOP_AND_LOCALIZE"
        ),
        "n0_objective_reached": passed,
        "n0_complete": passed,
        "n1_authorized": passed,
        "validation_owner": "alice_native_self_validation_harness",
        "external_validator": False,
        "single_frozen_validation": True,
        "checkpoint_selection_performed": False,
        "threshold_changed_after_results": False,
        "gradient_performed": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "qsre": qsre_result,
        "general_fabric": general_metrics,
        "integrated_relational_fabric": integrated_summary,
        "gate_checks": all_checks,
        "lineage": {
            "contract_sha256": sha256(contract_path),
            "manifest_sha256": sha256(manifest_path),
            "freeze_receipt_sha256": sha256(freeze_path),
            "general_corpus_sha256": sha256(general_path),
            "relational_corpus_sha256": sha256(relational_path),
            "relational_cache_sha256": sha256(relational_cache_path),
            "final_schema_cache_sha256": sha256(final_schema_cache_path),
            "p1_checkpoint_sha256": sha256(p1_path),
            "p2_checkpoint_sha256": sha256(p2_path),
            "p3_checkpoint_sha256": sha256(p3_path),
            "p4_result_sha256": sha256(p4_path),
            "fusion_sha256": EXPECTED_FUSION_SHA256,
            "latent_pool_sha256": EXPECTED_LATENT_SHA256,
        },
        "objective_interpretation": (
            "N0 supplies the identity-neutral semantic, structured, evidence, "
            "dynamic relational, uncertainty, sparse relevance, heterogeneous "
            "fusion and multi-view latent substrate required for N1 identity learning."
            if passed
            else "At least one precommitted N0 objective block failed; do not start N1."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(24)


if __name__ == "__main__":
    main()
