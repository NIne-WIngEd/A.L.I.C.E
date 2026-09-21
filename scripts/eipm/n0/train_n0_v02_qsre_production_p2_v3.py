from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from alice_personality.n0.qsre_production_core import (
    CONTROL_RELATIONAL,
    QSREDynamicRelationSchema,
    QSREProductionExecutor,
    QSREProductionOperatorState,
    QSREProductionSchemaEncoder,
)
from alice_personality.n0.qsre_production_operator_v3 import (
    EVENT_CONTINUE,
    EVENT_STOP,
    EVENT_UNKNOWN,
    QSREProductionOperatorInducerV3,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import (
    batch_indices,
    config_from_plan,
    load_plan,
    operator_metrics,
    oracle_operator_from_targets,
    paired_view_consistency,
    summarize_downstream,
    target_distribution_loss,
)


def load_p1(
    *,
    p1_result_path: Path,
    p1_root: Path,
    config,
    device: torch.device,
) -> tuple[QSREProductionSchemaEncoder, QSREProductionExecutor, dict]:
    result = json.loads(p1_result_path.read_text())
    if result.get("status") != "PASS_QSRE_PRODUCTION_P1_EXECUTOR":
        raise SystemExit("P1 did not pass")
    selected = result.get("selected")
    if not selected:
        raise SystemExit("P1 selected checkpoint missing")
    step = int(selected["step"])
    path = p1_root / f"step-{step:08d}" / "qsre_production_p1.pt"
    if sha256(path) != selected["checkpoint_sha256"]:
        raise SystemExit("P1 selected checkpoint hash drift")
    payload = torch.load(path, map_location="cpu")
    schema_encoder = QSREProductionSchemaEncoder(config)
    executor = QSREProductionExecutor(config)
    schema_encoder.load_state_dict(payload["schema_encoder"], strict=True)
    executor.load_state_dict(payload["executor"], strict=True)
    for module in (schema_encoder, executor):
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.to(device).eval()
    return schema_encoder, executor, {
        "path": path,
        "sha256": sha256(path),
        "step": step,
    }


def load_closure_matcher(
    *,
    result_path: Path,
    root: Path,
    factor_cache_path: Path,
) -> tuple[dict, dict, dict]:
    result = json.loads(result_path.read_text())
    if result.get("schema") != "alice.eipm.n0.qsre-closure-matcher-result.v2":
        raise SystemExit("closure schema matcher result version drift")
    if result.get("status") != "PASS_QSRE_CLOSURE_SCHEMA_MATCHER":
        raise SystemExit("closure schema matcher did not pass")
    selected = result.get("selected")
    if not selected:
        raise SystemExit("closure schema matcher selected checkpoint missing")
    step = int(selected["step"])
    path = root / f"step-{step:08d}" / "qsre_closure_matcher.pt"
    if sha256(path) != selected["checkpoint_sha256"]:
        raise SystemExit("closure schema matcher checkpoint hash drift")
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "alice.eipm.n0.qsre-closure-matcher-checkpoint.v2":
        raise SystemExit("closure schema matcher checkpoint schema drift")
    factor_cache = torch.load(factor_cache_path, map_location="cpu")
    if factor_cache.get("schema") != "alice.eipm.n0.qsre-closure-factor-schema-cache.v1":
        raise SystemExit("closure factor schema cache drift")
    if sha256(factor_cache_path) != result["factor_schema_cache_sha256"]:
        raise SystemExit("closure factor schema cache hash drift")
    return payload, factor_cache, {
        "path": path,
        "sha256": sha256(path),
        "factor_schema_cache_sha256": sha256(factor_cache_path),
    }


def subset_schema(
    schema: QSREDynamicRelationSchema,
    count: int,
) -> QSREDynamicRelationSchema:
    if count <= 0 or count > schema.token_states.size(0):
        raise ValueError("invalid schema subset size")
    return QSREDynamicRelationSchema(
        token_states=schema.token_states[:count],
        token_mask=schema.token_mask[:count],
        domain_type_mask=schema.domain_type_mask[:count],
        range_type_mask=schema.range_type_mask[:count],
        symmetric=schema.symmetric[:count],
    )


def macro_class_nll(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if probability.ndim < 2:
        raise ValueError("probability requires class axis")
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    mask = mask.bool()
    if not bool(mask.any()):
        return probability.sum() * 0.0
    p = probability[mask].clamp_min(1.0e-8)
    y = target[mask].long()
    losses = []
    for cls in torch.unique(y).tolist():
        cls_mask = y.eq(int(cls))
        losses.append(-p[cls_mask, int(cls)].log().mean())
    return torch.stack(losses).mean()


def macro_class_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits require class axis")
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    mask = mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    selected_logits = logits[mask]
    selected_target = target[mask].long()
    losses = []
    for cls in torch.unique(selected_target).tolist():
        cls_mask = selected_target.eq(int(cls))
        losses.append(
            F.cross_entropy(
                selected_logits[cls_mask],
                selected_target[cls_mask],
            )
        )
    return torch.stack(losses).mean()


def macro_binary_factor_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("binary factor logits/target shape drift")
    per_factor = []
    for factor in range(logits.size(-1)):
        x = logits[:, factor]
        y = target[:, factor]
        class_losses = []
        for value in (0.0, 1.0):
            mask = y.eq(value)
            if bool(mask.any()):
                class_losses.append(
                    F.binary_cross_entropy_with_logits(
                        x[mask],
                        y[mask],
                    )
                )
        if class_losses:
            per_factor.append(torch.stack(class_losses).mean())
    if not per_factor:
        return logits.sum() * 0.0
    return torch.stack(per_factor).mean()


def macro_binary_factor_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if probability.shape != target.shape:
        raise ValueError("binary factor probability/target shape drift")
    per_factor = []
    for factor in range(probability.size(-1)):
        p = probability[:, factor].clamp(1.0e-6, 1.0 - 1.0e-6)
        y = target[:, factor]
        class_losses = []
        for value in (0.0, 1.0):
            mask = y.eq(value)
            if bool(mask.any()):
                class_losses.append(
                    F.binary_cross_entropy(p[mask], y[mask])
                )
        if class_losses:
            per_factor.append(torch.stack(class_losses).mean())
    if not per_factor:
        return probability.sum() * 0.0
    return torch.stack(per_factor).mean()


def build_event_targets(
    *,
    relation_mask: torch.Tensor,
    termination_target: torch.Tensor,
    pred_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = relation_mask.size(0)
    event_target = torch.full(
        (batch, pred_steps),
        EVENT_STOP,
        dtype=torch.long,
        device=relation_mask.device,
    )
    supervised = torch.zeros(
        batch,
        pred_steps,
        dtype=torch.bool,
        device=relation_mask.device,
    )
    lengths = relation_mask.long().sum(dim=-1)
    for row, length in enumerate(lengths.tolist()):
        active = min(int(length), pred_steps)
        if active:
            event_target[row, :active] = EVENT_CONTINUE
            supervised[row, :active] = True
        if length < pred_steps:
            event_target[row, length] = (
                EVENT_UNKNOWN
                if int(termination_target[row].item()) == 1
                else EVENT_STOP
            )
            supervised[row, length] = True
    return event_target, supervised


def operator_supervised_loss_v3(
    *,
    model_output: dict,
    split: dict,
    indices: torch.Tensor,
    oracle_continuous: torch.Tensor,
    weights: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    operator: QSREProductionOperatorState = model_output["operator"]
    relation_target = split["relation_target"][indices].to(device).long()
    relation_mask = split["relation_target_mask"][indices].to(device).bool()
    role_target = split["role_target"][indices].to(device).long()
    traversal_target = split["traversal_target"][indices].to(device).long()
    direction_target = split["direction_target"][indices].to(device).long()
    modifier_target = split["modifier_target"][indices].to(device).float()
    applicability_target = split["applicability_target"][indices].to(device).float()
    control_target = split["control_target"][indices].to(device).long()
    termination_target = split["termination_target"][indices].to(device).long()

    _, pred_steps, _ = operator.relation_distribution.shape
    target_steps = relation_target.size(1)
    if pred_steps < target_steps:
        raise RuntimeError("operator runtime budget shorter than supervision")
    if pred_steps > target_steps:
        pad = pred_steps - target_steps
        relation_target = F.pad(relation_target, (0, pad), value=0)
        relation_mask = F.pad(relation_mask, (0, pad), value=False)

    relation_logits = model_output["relation_logits"]
    relation_loss = macro_class_cross_entropy(
        relation_logits,
        relation_target,
        mask=relation_mask,
    )

    event_target, event_supervised = build_event_targets(
        relation_mask=relation_mask,
        termination_target=termination_target,
        pred_steps=pred_steps,
    )
    event_loss = macro_class_cross_entropy(
        model_output["event_logits"],
        event_target,
        mask=event_supervised,
    )

    factor_logits = model_output["factor_logits"]
    relational = control_target.eq(CONTROL_RELATIONAL)
    if bool(relational.any()):
        role_loss = macro_class_cross_entropy(
            factor_logits["role"][relational],
            role_target[relational],
        )
        traversal_loss = macro_class_cross_entropy(
            factor_logits["traversal"][relational],
            traversal_target[relational],
        )
        direction_loss = macro_class_cross_entropy(
            factor_logits["direction"][relational],
            direction_target[relational],
        )
        modifier_loss = macro_binary_factor_loss_with_logits(
            factor_logits["modifier"][relational],
            modifier_target[relational],
        )
    else:
        zero = operator.continuous_state.sum() * 0.0
        role_loss = zero
        traversal_loss = zero
        direction_loss = zero
        modifier_loss = zero

    applicability_loss = F.binary_cross_entropy_with_logits(
        factor_logits["applicability"],
        applicability_target,
    )
    control_loss = macro_class_cross_entropy(
        factor_logits["control"],
        control_target,
    )

    relational_cont = relational & relation_mask.any(dim=-1)
    if bool(relational_cont.any()):
        continuous_loss = (
            1.0
            - F.cosine_similarity(
                operator.continuous_state[relational_cont],
                oracle_continuous[relational_cont].detach(),
                dim=-1,
            )
        ).mean()
    else:
        continuous_loss = operator.continuous_state.sum() * 0.0

    total = (
        float(weights["relation"]) * relation_loss
        + float(weights["event"]) * event_loss
        + float(weights["role"]) * role_loss
        + float(weights["traversal"]) * traversal_loss
        + float(weights["direction"]) * direction_loss
        + float(weights["modifiers"]) * modifier_loss
        + float(weights["applicability"]) * applicability_loss
        + float(weights["control"]) * control_loss
        + float(weights["continuous_alignment"]) * continuous_loss
    )
    parts = {
        "relation": float(relation_loss.detach().item()),
        "event": float(event_loss.detach().item()),
        "role": float(role_loss.detach().item()),
        "traversal": float(traversal_loss.detach().item()),
        "direction": float(direction_loss.detach().item()),
        "modifiers": float(modifier_loss.detach().item()),
        "applicability": float(applicability_loss.detach().item()),
        "control": float(control_loss.detach().item()),
        "continuous_alignment": float(continuous_loss.detach().item()),
    }
    return total, parts


def pair_loss_v3(a: dict, b: dict) -> torch.Tensor:
    op_a: QSREProductionOperatorState = a["operator"]
    op_b: QSREProductionOperatorState = b["operator"]
    factor_a = a["factor_logits"]
    factor_b = b["factor_logits"]

    relation_consistency = F.mse_loss(
        torch.softmax(a["relation_logits"], dim=-1),
        torch.softmax(b["relation_logits"], dim=-1),
    )
    event_consistency = F.mse_loss(
        torch.softmax(a["event_logits"], dim=-1),
        torch.softmax(b["event_logits"], dim=-1),
    )
    factor_consistency = (
        F.mse_loss(
            torch.softmax(factor_a["role"], dim=-1),
            torch.softmax(factor_b["role"], dim=-1),
        )
        + F.mse_loss(
            torch.softmax(factor_a["traversal"], dim=-1),
            torch.softmax(factor_b["traversal"], dim=-1),
        )
        + F.mse_loss(
            torch.softmax(factor_a["direction"], dim=-1),
            torch.softmax(factor_b["direction"], dim=-1),
        )
        + F.mse_loss(
            torch.sigmoid(factor_a["modifier"]),
            torch.sigmoid(factor_b["modifier"]),
        )
        + F.mse_loss(
            torch.softmax(factor_a["control"], dim=-1),
            torch.softmax(factor_b["control"], dim=-1),
        )
        + F.mse_loss(
            torch.sigmoid(factor_a["applicability"]),
            torch.sigmoid(factor_b["applicability"]),
        )
    )
    continuous_consistency = (
        1.0
        - F.cosine_similarity(
            op_a.continuous_state,
            op_b.continuous_state,
            dim=-1,
        )
    ).mean()
    return (
        relation_consistency
        + event_consistency
        + factor_consistency
        + 0.25 * continuous_consistency
    )


def execute_with_oracle_support(
    *,
    split: dict,
    indices: torch.Tensor,
    operator: QSREProductionOperatorState,
    schema_relation_state: torch.Tensor,
    executor: QSREProductionExecutor,
    device: torch.device,
):
    return executor(
        field_state=split["field_state"][indices].to(device).float(),
        field_metadata=split["field_metadata"][indices].to(device).float(),
        field_valid_mask=split["field_valid_mask"][indices].to(device).bool(),
        edge_index=split["edge_index"][indices].to(device).long(),
        edge_relation_index=split["edge_relation_index"][indices].to(device).long(),
        edge_valid_mask=split["edge_valid_mask"][indices].to(device).bool(),
        edge_support_weight=split["oracle_edge_support"][indices].to(device).float(),
        edge_reliability=split["edge_reliability"][indices].to(device).float(),
        edge_recency=split["edge_recency"][indices].to(device).float(),
        edge_temporal_match=split["edge_temporal_match"][indices].to(device).float(),
        edge_provenance_match=split["edge_provenance_match"][indices].to(device).float(),
        schema_relation_state=schema_relation_state,
        operator=operator,
        focus_field_weight=split["focus_field_weight"][indices].to(device).float(),
    )


def infer_view_output(
    *,
    split: dict,
    indices: torch.Tensor,
    view: int,
    schema: QSREDynamicRelationSchema,
    encoded: dict[str, torch.Tensor],
    operator_model: QSREProductionOperatorInducerV3,
    max_steps: int,
    device: torch.device,
):
    return operator_model(
        query_hidden_states=split["query_hidden_states"][indices, view]
        .to(device)
        .float(),
        query_token_mask=split["query_token_mask"][indices, view]
        .to(device)
        .bool(),
        schema=schema,
        schema_token_state=encoded["schema_token_state"],
        schema_relation_state=encoded["schema_relation_state"],
        max_steps=max_steps,
    )


@torch.no_grad()
def evaluate(
    *,
    split: dict,
    schema: QSREDynamicRelationSchema,
    schema_encoder: QSREProductionSchemaEncoder,
    executor: QSREProductionExecutor,
    operator_model: QSREProductionOperatorInducerV3,
    config,
    max_steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    schema_encoder.eval()
    executor.eval()
    operator_model.eval()
    encoded = schema_encoder(schema)
    per_view_operator = [[], []]
    per_view_probability = [[], []]

    for start in range(0, len(split["ids"]), batch_size):
        idx = torch.arange(
            start,
            min(start + batch_size, len(split["ids"])),
        )
        for view in (0, 1):
            output = infer_view_output(
                split=split,
                indices=idx,
                view=view,
                schema=schema,
                encoded=encoded,
                operator_model=operator_model,
                max_steps=max_steps,
                device=device,
            )
            op = output["operator"]
            per_view_operator[view].append(op)
            downstream = execute_with_oracle_support(
                split=split,
                indices=idx,
                operator=op,
                schema_relation_state=encoded["schema_relation_state"],
                executor=executor,
                device=device,
            )
            per_view_probability[view].append(
                downstream["relational_probability"].cpu()
            )

    def concat_ops(values):
        fields = QSREProductionOperatorState.__dataclass_fields__
        return QSREProductionOperatorState(
            **{
                name: torch.cat(
                    [getattr(value, name).cpu() for value in values],
                    dim=0,
                )
                for name in fields
            }
        )

    ops = [
        concat_ops(per_view_operator[0]),
        concat_ops(per_view_operator[1]),
    ]
    probs = [
        torch.cat(per_view_probability[0], dim=0),
        torch.cat(per_view_probability[1], dim=0),
    ]
    all_idx = torch.arange(len(split["ids"]))

    op_metrics = []
    downstream = []
    for view in (0, 1):
        metric = operator_metrics(
            operator=ops[view],
            split=split,
            indices=all_idx,
        )
        op_metrics.append(metric)
        summary = summarize_downstream(
            probability=probs[view],
            target=split["target_distribution"].float(),
            control_target=split["control_target"].long(),
            families=split["families"],
            open_schema=split["open_schema"].bool(),
            causal_groups=split["causal_groups"],
        )
        summary.pop("row_success_tensor", None)
        downstream.append(summary)

    def min_metric(key):
        return min(
            float(op_metrics[0][key]),
            float(op_metrics[1][key]),
        )

    operator_summary = {
        "relation_sequence_exact_accuracy": min_metric(
            "relation_sequence_exact_accuracy"
        ),
        "open_schema_relation_exact_accuracy": min_metric(
            "open_schema_relation_exact_accuracy"
        ),
        "role_accuracy_relational": min_metric("role_accuracy_relational"),
        "traversal_accuracy_relational": min_metric(
            "traversal_accuracy_relational"
        ),
        "direction_accuracy_relational": min_metric(
            "direction_accuracy_relational"
        ),
        "modifier_exact_accuracy_relational": min_metric(
            "modifier_exact_accuracy_relational"
        ),
        "control_accuracy": min_metric("control_accuracy"),
        "termination_accuracy": min_metric("termination_accuracy"),
        "unknown_termination_accuracy": min_metric(
            "unknown_termination_accuracy"
        ),
        "pair_consistency": paired_view_consistency(
            first=ops[0],
            second=ops[1],
        ),
    }

    family_keys = (
        set(downstream[0]["family_success"])
        | set(downstream[1]["family_success"])
    )
    family_worst = {
        family: min(
            float(
                downstream[0]["family_success"].get(family, 1.0)
            ),
            float(
                downstream[1]["family_success"].get(family, 1.0)
            ),
        )
        for family in sorted(family_keys)
    }
    downstream_summary = {
        "row_success_accuracy": min(
            float(downstream[0]["row_success_accuracy"]),
            float(downstream[1]["row_success_accuracy"]),
        ),
        "family_min_success": min(family_worst.values())
        if family_worst
        else 1.0,
        "family_success": family_worst,
        "open_schema_success": min(
            float(downstream[0]["open_schema_success"]),
            float(downstream[1]["open_schema_success"]),
        ),
        "single_target_top1_accuracy": min(
            float(downstream[0]["single_target_top1_accuracy"]),
            float(downstream[1]["single_target_top1_accuracy"]),
        ),
        "plural_l1": max(
            float(downstream[0]["plural_l1"]),
            float(downstream[1]["plural_l1"]),
        ),
        "outside_support_invariance_max_delta": max(
            float(
                downstream[0]["outside_support_invariance_max_delta"]
            ),
            float(
                downstream[1]["outside_support_invariance_max_delta"]
            ),
        ),
    }
    return {
        "operator": operator_summary,
        "downstream": downstream_summary,
    }


def eligible(metrics: dict, thresholds: dict) -> bool:
    op = metrics["operator"]
    down = metrics["downstream"]
    return (
        op["relation_sequence_exact_accuracy"]
        >= thresholds["relation_sequence_exact_accuracy"]
        and op["open_schema_relation_exact_accuracy"]
        >= thresholds["open_schema_relation_exact_accuracy"]
        and op["role_accuracy_relational"]
        >= thresholds["role_accuracy_relational"]
        and op["traversal_accuracy_relational"]
        >= thresholds["traversal_accuracy_relational"]
        and op["direction_accuracy_relational"]
        >= thresholds["direction_accuracy_relational"]
        and op["modifier_exact_accuracy_relational"]
        >= thresholds["modifier_exact_accuracy_relational"]
        and op["control_accuracy"] >= thresholds["control_accuracy"]
        and op["termination_accuracy"]
        >= thresholds["termination_accuracy"]
        and op["unknown_termination_accuracy"]
        >= thresholds["unknown_termination_accuracy"]
        and op["pair_consistency"] >= thresholds["pair_consistency"]
        and down["row_success_accuracy"]
        >= thresholds["downstream_row_success_accuracy"]
        and down["family_min_success"]
        >= thresholds["downstream_family_min_success"]
    )


def score_tuple(metrics: dict) -> tuple[float, ...]:
    op = metrics["operator"]
    down = metrics["downstream"]
    return (
        op["relation_sequence_exact_accuracy"],
        op["open_schema_relation_exact_accuracy"],
        op["unknown_termination_accuracy"],
        op["traversal_accuracy_relational"],
        op["direction_accuracy_relational"],
        op["pair_consistency"],
        down["row_success_accuracy"],
        down["family_min_success"],
    )


def save_checkpoint(
    path: Path,
    operator_model,
    config,
    step: int,
    lineage: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "alice.eipm.n0.qsre-production-p2-checkpoint.v3",
            "architecture": "closure-schema-matcher-plus-ordered-continuous-operator",
            "stage": "P2",
            "step": step,
            "config": config.__dict__,
            "operator": operator_model.state_dict(),
            **lineage,
        },
        path,
    )
    return sha256(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--prepared-cache", required=True)
    p.add_argument("--schema-cache", required=True)
    p.add_argument("--p1-result", required=True)
    p.add_argument("--p1-root", required=True)
    p.add_argument("--failed-p2-result", required=True)
    p.add_argument("--closure-matcher-result", required=True)
    p.add_argument("--closure-matcher-root", required=True)
    p.add_argument("--factor-schema-cache", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("P2 v3 requires CUDA")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("refusing to overwrite P2 v3 evidence")

    failed_p2_path = Path(args.failed_p2_result)
    failed_p2 = json.loads(failed_p2_path.read_text())
    if failed_p2.get("status") != "FAIL_QSRE_PRODUCTION_P2_OPERATOR_V3":
        raise SystemExit(
            "closure P2 requires preserved Magnolia 575958 P2-v3 failure evidence"
        )
    if failed_p2.get("selected") is not None:
        raise SystemExit("failed P2 unexpectedly selected a checkpoint")
    if failed_p2.get("p3_authorized") is not False:
        raise SystemExit("failed P2 unexpectedly authorized P3")

    plan_path = Path(args.plan)
    prepared_path = Path(args.prepared_cache)
    schema_cache_path = Path(args.schema_cache)
    plan = load_plan(plan_path)
    stage = plan["stages"]["P2"]
    config = config_from_plan(plan)
    prepared = torch.load(prepared_path, map_location="cpu")
    if (
        prepared.get("private_identity_data") is not False
        or prepared.get("test_present") is not False
    ):
        raise SystemExit("P2 v3 boundary violation")

    device = torch.device("cuda")
    full_schema, schema_payload = load_dynamic_schema_cache(
        schema_cache_path,
        device=device,
    )
    core_keys = list(schema_payload["core_train_relation_keys"])
    relation_keys = list(schema_payload["relation_keys"])
    core_count = len(core_keys)
    if relation_keys[:core_count] != core_keys:
        raise SystemExit(
            "core relation schema must occupy the stable training prefix"
        )
    core_schema = subset_schema(full_schema, core_count)

    schema_encoder, executor, p1 = load_p1(
        p1_result_path=Path(args.p1_result),
        p1_root=Path(args.p1_root),
        config=config,
        device=device,
    )
    matcher_payload, factor_cache, closure_matcher = load_closure_matcher(
        result_path=Path(args.closure_matcher_result),
        root=Path(args.closure_matcher_root),
        factor_cache_path=Path(args.factor_schema_cache),
    )
    operator_model = QSREProductionOperatorInducerV3(config).to(device)
    operator_model.load_pretrained_schema_matcher(
        matcher_payload["matcher"],
        freeze=True,
    )
    operator_model.configure_factor_schema_cache(factor_cache)
    if any(
        parameter.requires_grad
        for parameter in operator_model.schema_matcher.parameters()
    ):
        raise SystemExit("closure schema matcher must remain frozen in P2")
    if any(
        hasattr(operator_model, name)
        for name in (
            "role_head",
            "traversal_head",
            "direction_head",
            "modifier_head",
            "control_head",
        )
    ):
        raise SystemExit("fixed factor class head reintroduced in closure P2")

    trainable_parameters = [
        parameter
        for parameter in operator_model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(stage["optimizer"]["learning_rate"]),
        weight_decay=float(stage["optimizer"]["weight_decay"]),
    )

    seed = int(stage["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_size = int(stage["training"]["batch_size"])
    max_train_steps = int(stage["training"]["max_steps"])
    eval_every = int(stage["training"]["eval_every"])
    runtime_steps = int(stage["training"]["operator_runtime_max_steps"])
    weights = stage["loss_weights"]

    output_dir.mkdir(parents=True)
    train = prepared["train"]
    dev = prepared["dev"]

    # Query-supervised training is strictly core-schema only. DEV/open
    # relation descriptions never enter a gradient-bearing query path and are
    # not repeatedly taught as negatives. Dynamic candidate expansion is an
    # architectural requirement, tested only during DEV/final runtime.
    with torch.no_grad():
        encoded_core = schema_encoder(core_schema)

    train_relation_mask = train["relation_target_mask"].bool()
    if bool(train_relation_mask.any()):
        train_relation_ids = train["relation_target"][train_relation_mask]
        if int(train_relation_ids.max().item()) >= core_count:
            raise SystemExit(
                "open-schema relation label leaked into P2 v3 training"
            )

    batches = []
    epoch = 0
    step = 0
    history = []
    selected = None
    best = None
    best_score = None

    while step < max_train_steps:
        if not batches:
            batches = batch_indices(
                len(train["ids"]),
                batch_size=batch_size,
                seed=seed + epoch,
            )
            epoch += 1
        idx = batches.pop(0)
        operator_model.train()
        operator_model.schema_matcher.eval()
        optimizer.zero_grad(set_to_none=True)

        oracle = oracle_operator_from_targets(
            train,
            idx,
            schema_relation_state=encoded_core["schema_relation_state"],
            config=config,
            device=device,
        )
        view_ops = []
        view_outputs = []
        supervised = []
        downstream_losses = []
        supervised_parts = []

        for view in (0, 1):
            model_output = infer_view_output(
                split=train,
                indices=idx,
                view=view,
                schema=core_schema,
                encoded=encoded_core,
                operator_model=operator_model,
                max_steps=runtime_steps,
                device=device,
            )
            op = model_output["operator"]
            view_ops.append(op)
            view_outputs.append(model_output)
            sup, parts = operator_supervised_loss_v3(
                model_output=model_output,
                split=train,
                indices=idx,
                oracle_continuous=oracle.continuous_state,
                weights=weights,
                device=device,
            )
            supervised.append(sup)
            supervised_parts.append(parts)
            down = execute_with_oracle_support(
                split=train,
                indices=idx,
                operator=op,
                schema_relation_state=encoded_core[
                    "schema_relation_state"
                ],
                executor=executor,
                device=device,
            )
            target = train["target_distribution"][idx].to(device).float()
            downstream_losses.append(
                target_distribution_loss(
                    down["relational_probability"],
                    target,
                )
            )

        consistency = pair_loss_v3(view_outputs[0], view_outputs[1])
        # The semantic matcher already passed an unseen-schema stage and is
        # frozen here. Production P2 may learn ordering/event/applicability and
        # continuous execution state, but it cannot rotate the semantic metric
        # around the six core relations or destroy runtime zero-shot matching.
        matcher_anchor = sum(
            parameter.sum() * 0.0
            for parameter in operator_model.schema_matcher.parameters()
        )
        loss = (
            0.5 * (supervised[0] + supervised[1])
            + float(weights["downstream"])
            * 0.5
            * (downstream_losses[0] + downstream_losses[1])
            + float(weights["pair_consistency"]) * consistency
            + matcher_anchor
        )
        if not torch.isfinite(loss):
            raise RuntimeError("P2 v3 nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            operator_model.parameters(),
            1.0,
        )
        optimizer.step()
        step += 1

        if step % eval_every == 0 or step == max_train_steps:
            metrics = evaluate(
                split=dev,
                schema=full_schema,
                schema_encoder=schema_encoder,
                executor=executor,
                operator_model=operator_model,
                config=config,
                max_steps=runtime_steps,
                batch_size=batch_size,
                device=device,
            )
            is_eligible = eligible(metrics, stage["eligibility"])
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_pair_consistency_loss": float(
                    consistency.detach().item()
                ),
                "closure_schema_matcher_frozen": True,
                "fixed_factor_class_heads": False,
                "train_supervised_parts_view0": supervised_parts[0],
                "train_supervised_parts_view1": supervised_parts[1],
                "eligible": is_eligible,
                "dev": metrics,
            }
            history.append(record)
            print(
                "P2_V3_EVAL=" + json.dumps(record, sort_keys=True),
                flush=True,
            )

            path = (
                output_dir
                / f"step-{step:08d}"
                / "qsre_production_p2.pt"
            )
            ckpt_sha = save_checkpoint(
                path,
                operator_model,
                config,
                step,
                {
                    "plan_sha256": sha256(plan_path),
                    "prepared_cache_sha256": sha256(prepared_path),
                    "schema_cache_sha256": sha256(schema_cache_path),
                    "p1_checkpoint_sha256": p1["sha256"],
                    "source_failed_p2_result_sha256": sha256(
                        failed_p2_path
                    ),
                    "closure_matcher_checkpoint_sha256": closure_matcher["sha256"],
                    "factor_schema_cache_sha256": closure_matcher[
                        "factor_schema_cache_sha256"
                    ],
                },
            )
            score = score_tuple(metrics)
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "step": step,
                    "checkpoint_sha256": ckpt_sha,
                    "metrics": metrics,
                }
            if is_eligible:
                selected = {
                    "step": step,
                    "checkpoint_sha256": ckpt_sha,
                    "metrics": metrics,
                }
                break

    status = (
        "PASS_QSRE_PRODUCTION_P2_OPERATOR"
        if selected
        else "FAIL_QSRE_PRODUCTION_P2_OPERATOR_V3"
    )
    result = {
        "schema": "alice.eipm.n0.qsre-production-p2-result.v3",
        "architecture": "continuous-relation-open-schema-operator-v3",
        "status": status,
        "selected": selected,
        "best_observed": best,
        "history": history,
        "plan_sha256": sha256(plan_path),
        "prepared_cache_sha256": sha256(prepared_path),
        "schema_cache_sha256": sha256(schema_cache_path),
        "p1_checkpoint_sha256": p1["sha256"],
        "source_failed_p2_result_sha256": sha256(failed_p2_path),
        "train_positive_relation_count": core_count,
        "runtime_relation_count": int(full_schema.token_states.size(0)),
        "open_schema_training_query_labels_used": False,
        "open_schema_relation_descriptions_used_in_gradient": False,
        "open_schema_runtime_candidates_used_during_training": False,
        "core_schema_only_query_training": True,
        "schema_only_self_calibration_used": False,
        "closure_schema_matcher_frozen": True,
        "closure_matcher_checkpoint_sha256": closure_matcher["sha256"],
        "factor_schema_cache_sha256": closure_matcher[
            "factor_schema_cache_sha256"
        ],
        "fixed_factor_class_heads": False,
        "semantic_factor_schemas": True,
        "ordered_query_evidence_coverage": True,
        "continuous_relation_hypotheses": True,
        "exact_relation_sparsity_before_binder": False,
        "runtime_max_steps": runtime_steps,
        "schema_encoder_gradient": False,
        "executor_gradient": False,
        "operator_gradient": True,
        "binder_gradient": False,
        "private_identity_gradient": False,
        "automatic_rerun": False,
        "automatic_hotfix": False,
        "p3_authorized": bool(selected),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        "P2_V3_RESULT=" + json.dumps(result, sort_keys=True),
        flush=True,
    )
    if not selected:
        raise SystemExit(31)


if __name__ == "__main__":
    main()
