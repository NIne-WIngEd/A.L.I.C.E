from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from alice_personality.n0.qsre_production_core import (
    QSREProductionOperatorInducer,
    QSREProductionSchemaEncoder,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import config_from_plan, load_plan


EXPECTED_FAILED_STATUS = "FAIL_QSRE_PRODUCTION_P2_OPERATOR"
EXPECTED_P1_STATUS = "PASS_QSRE_PRODUCTION_P1_EXECUTOR"
EXPECTED_FAILED_SOURCE = "dba3d4b100f081b0c09fe63ae002205244af5b44"
EXPECTED_BEST_STEP = 1600
EXPECTED_BEST_CHECKPOINT_SHA = (
    "16dd2681a18e9b5e371232c9206ca4a3b4d0ace8e5ec47e7ee492e9ab25364a0"
)


def _mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().item())


def _cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        a.reshape(a.size(0), -1).float(),
        b.reshape(b.size(0), -1).float(),
        dim=-1,
        eps=1.0e-8,
    )


def _pad_targets(
    target: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.size(1) > steps:
        raise RuntimeError("runtime step budget shorter than target program")
    if target.size(1) < steps:
        pad = steps - target.size(1)
        target = F.pad(target, (0, pad), value=0)
        mask = F.pad(mask, (0, pad), value=False)
    return target, mask


def _group_boolean(
    labels: list[str],
    value: torch.Tensor,
) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for label, ok in zip(labels, value.tolist()):
        buckets[str(label)].append(bool(ok))
    return {
        key: sum(items) / len(items)
        for key, items in sorted(buckets.items())
    }


def _length_breakdown(
    *,
    lengths: torch.Tensor,
    identity: torch.Tensor,
    active_exact: torch.Tensor,
    full_exact: torch.Tensor,
    extra_tail: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for length in sorted(set(int(x) for x in lengths.tolist())):
        select = lengths.eq(length)
        out[str(length)] = {
            "rows": int(select.sum().item()),
            "identity_exact": _mean(identity[select]),
            "active_mask_exact": _mean(active_exact[select]),
            "full_program_exact": _mean(full_exact[select]),
            "extra_active_tail_rate": _mean(extra_tail[select]),
        }
    return out


def _terminal_summary(
    *,
    unknown_probability: torch.Tensor,
    stop_probability: torch.Tensor,
    relation_step_mass: torch.Tensor,
    lengths: torch.Tensor,
    termination_target: torch.Tensor,
) -> dict[str, float | int]:
    unknown_rows = termination_target.eq(1)
    if not bool(unknown_rows.any()):
        return {"rows": 0}
    row = torch.arange(lengths.size(0))
    terminal = lengths.clamp(max=unknown_probability.size(1) - 1)
    u = unknown_probability[row, terminal][unknown_rows]
    s = stop_probability[row, terminal][unknown_rows]
    m = relation_step_mass[row, terminal][unknown_rows]
    return {
        "rows": int(unknown_rows.sum().item()),
        "unknown_win_rate": _mean(u.gt(s)),
        "mean_unknown_probability": _mean(u),
        "mean_stop_probability": _mean(s),
        "mean_effective_relation_mass_at_terminal": _mean(m),
        "relation_mass_ge_0_5_rate_at_terminal": _mean(m.ge(0.5)),
    }


def _alignment_report(
    *,
    schema_encoder: QSREProductionSchemaEncoder,
    operator: QSREProductionOperatorInducer,
) -> dict[str, float]:
    sw = schema_encoder.schema_norm.weight.detach().cpu().float()
    sb = schema_encoder.schema_norm.bias.detach().cpu().float()
    qw = operator.query_norm.weight.detach().cpu().float()
    qb = operator.query_norm.bias.detach().cpu().float()
    sp = schema_encoder.schema_projection.weight.detach().cpu().float()
    qp = operator.query_projection.weight.detach().cpu().float()
    sl = schema_encoder.layer_embedding.weight.detach().cpu().float()
    ql = operator.layer_embedding.weight.detach().cpu().float()

    def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            ((a - b).norm() / b.norm().clamp_min(1.0e-12)).item()
        )

    return {
        "schema_norm_weight_vs_identity_l2": float(
            (sw - torch.ones_like(sw)).norm().item()
        ),
        "schema_norm_bias_vs_zero_l2": float(sb.norm().item()),
        "query_vs_schema_norm_weight_relative_l2": rel_l2(qw, sw),
        "query_vs_schema_norm_bias_l2": float((qb - sb).norm().item()),
        "query_vs_schema_projection_relative_l2": rel_l2(qp, sp),
        "query_vs_schema_layer_embedding_relative_l2": rel_l2(ql, sl),
        "query_schema_projection_cosine": float(
            F.cosine_similarity(
                qp.reshape(1, -1),
                sp.reshape(1, -1),
                dim=-1,
            ).item()
        ),
    }


@torch.inference_mode()
def infer_view(
    *,
    split: dict,
    view: int,
    schema,
    encoded: dict[str, torch.Tensor],
    operator: QSREProductionOperatorInducer,
    max_steps: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, len(split["ids"]), batch_size):
        stop = min(start + batch_size, len(split["ids"]))
        idx = torch.arange(start, stop)
        output = operator(
            query_hidden_states=split["query_hidden_states"][idx, view].float(),
            query_token_mask=split["query_token_mask"][idx, view].bool(),
            schema=schema,
            schema_token_state=encoded["schema_token_state"],
            schema_relation_state=encoded["schema_relation_state"],
            max_steps=max_steps,
        )
        op = output["operator"]
        chunks["relation_distribution"].append(
            op.relation_distribution.detach().cpu()
        )
        chunks["relation_step_mass"].append(
            op.relation_step_mass.detach().cpu()
        )
        chunks["stop_probability"].append(
            op.stop_probability.detach().cpu()
        )
        chunks["unknown_probability"].append(
            op.unknown_probability.detach().cpu()
        )
        chunks["step_query_attention"].append(
            output["step_query_attention"].detach().cpu()
        )
    return {
        key: torch.cat(values, dim=0)
        for key, values in chunks.items()
    }


def diagnose_view(
    *,
    split: dict,
    inferred: dict[str, torch.Tensor],
) -> dict:
    relation = inferred["relation_distribution"]
    mass = inferred["relation_step_mass"]
    stop_probability = inferred["stop_probability"]
    unknown_probability = inferred["unknown_probability"]
    attention = inferred["step_query_attention"]

    pred = relation.argmax(dim=-1)
    active = mass.ge(0.5)
    target = split["relation_target"].cpu().long()
    target_mask = split["relation_target_mask"].cpu().bool()
    target, target_mask = _pad_targets(target, target_mask, pred.size(1))
    lengths = target_mask.long().sum(dim=-1)

    identity_position = torch.where(
        target_mask,
        pred.eq(target),
        torch.ones_like(target_mask),
    )
    identity_exact = identity_position.all(dim=-1)
    active_exact = active.eq(target_mask).all(dim=-1)
    full_exact = identity_exact & active_exact
    extra_tail = (active & ~target_mask).any(dim=-1)

    step_accuracy: dict[str, dict[str, float | int]] = {}
    step_mass: dict[str, dict[str, float | int]] = {}
    for step in range(pred.size(1)):
        select = target_mask[:, step]
        step_accuracy[str(step)] = {
            "rows": int(select.sum().item()),
            "relation_top1": (
                _mean(pred[select, step].eq(target[select, step]))
                if bool(select.any())
                else None
            ),
        }
        step_mass[str(step)] = {
            "rows": int(select.sum().item()),
            "mean_effective_relation_mass": (
                _mean(mass[select, step])
                if bool(select.any())
                else float("nan")
            ),
            "active_ge_0_5_rate": (
                _mean(mass[select, step].ge(0.5))
                if bool(select.any())
                else float("nan")
            ),
        }

    multi = lengths.ge(2)
    reversed_match = torch.zeros_like(multi)
    repeated_collapse = torch.zeros_like(multi)
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        if length < 2:
            continue
        predicted = pred[row, :length]
        gold = target[row, :length]
        reversed_match[row] = bool(predicted.eq(gold.flip(0)).all())
        repeated_collapse[row] = bool(predicted.eq(predicted[0]).all())

    attention_similarity: dict[str, dict[str, float | int]] = {}
    relation_similarity: dict[str, dict[str, float | int]] = {}
    for left in range(pred.size(1) - 1):
        right = left + 1
        select = lengths.gt(right)
        key = f"{left}->{right}"
        if bool(select.any()):
            attention_similarity[key] = {
                "rows": int(select.sum().item()),
                "mean_cosine": _mean(
                    _cosine_rows(
                        attention[select, left],
                        attention[select, right],
                    )
                ),
            }
            relation_similarity[key] = {
                "rows": int(select.sum().item()),
                "mean_cosine": _mean(
                    _cosine_rows(
                        relation[select, left],
                        relation[select, right],
                    )
                ),
            }

    families = [str(x) for x in split["families"]]
    open_schema = split["open_schema"].cpu().bool()
    core = ~open_schema
    termination_target = split["termination_target"].cpu().long()

    return {
        "rows": len(families),
        "relation_identity_exact_ignoring_activity": _mean(identity_exact),
        "active_mask_exact": _mean(active_exact),
        "full_program_exact": _mean(full_exact),
        "identity_correct_but_activity_wrong_rate": _mean(
            identity_exact & ~active_exact
        ),
        "activity_correct_but_identity_wrong_rate": _mean(
            active_exact & ~identity_exact
        ),
        "extra_active_tail_rate": _mean(extra_tail),
        "core_relation_identity_exact": _mean(identity_exact[core]),
        "open_schema_relation_identity_exact": _mean(
            identity_exact[open_schema]
        ),
        "by_target_length": _length_breakdown(
            lengths=lengths,
            identity=identity_exact,
            active_exact=active_exact,
            full_exact=full_exact,
            extra_tail=extra_tail,
        ),
        "relation_top1_by_step": step_accuracy,
        "effective_relation_mass_by_step": step_mass,
        "multi_step_reversed_target_match_rate": (
            _mean(reversed_match[multi])
            if bool(multi.any())
            else float("nan")
        ),
        "multi_step_repeated_relation_collapse_rate": (
            _mean(repeated_collapse[multi])
            if bool(multi.any())
            else float("nan")
        ),
        "step_attention_cosine": attention_similarity,
        "step_relation_distribution_cosine": relation_similarity,
        "family_relation_identity_exact": _group_boolean(
            families,
            identity_exact,
        ),
        "family_active_mask_exact": _group_boolean(
            families,
            active_exact,
        ),
        "family_full_program_exact": _group_boolean(
            families,
            full_exact,
        ),
        "unknown_terminal": _terminal_summary(
            unknown_probability=unknown_probability,
            stop_probability=stop_probability,
            relation_step_mass=mass,
            lengths=lengths,
            termination_target=termination_target,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--prepared-cache", required=True)
    parser.add_argument("--schema-cache", required=True)
    parser.add_argument("--p1-result", required=True)
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--p2-result", required=True)
    parser.add_argument("--p2-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cpu")
    plan_path = Path(args.plan).resolve()
    prepared_path = Path(args.prepared_cache).resolve()
    schema_cache_path = Path(args.schema_cache).resolve()
    p1_result_path = Path(args.p1_result).resolve()
    p1_root = Path(args.p1_root).resolve()
    p2_result_path = Path(args.p2_result).resolve()
    p2_root = Path(args.p2_root).resolve()
    output_path = Path(args.output).resolve()

    if output_path.exists():
        raise SystemExit("refusing to overwrite P2 failure-localization evidence")

    plan = load_plan(plan_path)
    if plan.get("schema") != "alice.eipm.n0.qsre-production-training-plan.v1":
        raise SystemExit("diagnostic must use failed P2 v1 training plan")
    config = config_from_plan(plan)

    p2_result = json.loads(p2_result_path.read_text())
    if p2_result.get("status") != EXPECTED_FAILED_STATUS:
        raise SystemExit("source P2 result is not the governed failed result")
    best = p2_result.get("best_observed") or {}
    if int(best.get("step", -1)) != EXPECTED_BEST_STEP:
        raise SystemExit("failed P2 best-step drift")
    if best.get("checkpoint_sha256") != EXPECTED_BEST_CHECKPOINT_SHA:
        raise SystemExit("failed P2 best-checkpoint hash drift")

    checkpoint = (
        p2_root
        / f"step-{EXPECTED_BEST_STEP:08d}"
        / "qsre_production_p2.pt"
    )
    if sha256(checkpoint) != EXPECTED_BEST_CHECKPOINT_SHA:
        raise SystemExit("failed P2 checkpoint file hash drift")

    prepared = torch.load(prepared_path, map_location="cpu")
    if prepared.get("private_identity_data") is not False:
        raise SystemExit("private identity data entered diagnostic")

    schema, _ = load_dynamic_schema_cache(
        schema_cache_path,
        device=device,
    )

    p1_result = json.loads(p1_result_path.read_text())
    if p1_result.get("status") != EXPECTED_P1_STATUS:
        raise SystemExit("P1 did not pass")
    p1_selected = p1_result.get("selected") or {}
    p1_step = int(p1_selected.get("step", -1))
    p1_checkpoint = (
        p1_root
        / f"step-{p1_step:08d}"
        / "qsre_production_p1.pt"
    )
    if sha256(p1_checkpoint) != p1_selected.get("checkpoint_sha256"):
        raise SystemExit("P1 checkpoint hash drift")
    p1_payload = torch.load(p1_checkpoint, map_location="cpu")

    schema_encoder = QSREProductionSchemaEncoder(config)
    schema_encoder.load_state_dict(
        p1_payload["schema_encoder"],
        strict=True,
    )
    schema_encoder.eval()
    for parameter in schema_encoder.parameters():
        parameter.requires_grad = False

    p2_payload = torch.load(checkpoint, map_location="cpu")
    if p2_payload.get("schema") != "alice.eipm.n0.qsre-production-p2-checkpoint.v1":
        raise SystemExit("P2 checkpoint schema drift")
    operator = QSREProductionOperatorInducer(config)
    operator.load_state_dict(p2_payload["operator"], strict=True)
    operator.eval()
    for parameter in operator.parameters():
        parameter.requires_grad = False

    if any(parameter.requires_grad for parameter in operator.parameters()):
        raise SystemExit("operator gradient unexpectedly enabled")
    if any(parameter.requires_grad for parameter in schema_encoder.parameters()):
        raise SystemExit("schema-encoder gradient unexpectedly enabled")

    encoded = schema_encoder(schema)
    split = prepared["dev"]
    runtime_steps = int(
        plan["stages"]["P2"]["training"]["operator_runtime_max_steps"]
    )

    view_results = {}
    for view in (0, 1):
        inferred = infer_view(
            split=split,
            view=view,
            schema=schema,
            encoded=encoded,
            operator=operator,
            max_steps=runtime_steps,
            batch_size=int(args.batch_size),
        )
        view_results[str(view)] = diagnose_view(
            split=split,
            inferred=inferred,
        )

    report = {
        "schema": "alice.eipm.n0.qsre-production-p2-failure-localization.v1",
        "status": "PASS_ZERO_GRADIENT_P2_FAILURE_LOCALIZATION",
        "failed_source_revision": EXPECTED_FAILED_SOURCE,
        "source_p2_status": p2_result["status"],
        "source_p2_result_sha256": sha256(p2_result_path),
        "source_p2_best_step": EXPECTED_BEST_STEP,
        "source_p2_best_checkpoint_sha256": EXPECTED_BEST_CHECKPOINT_SHA,
        "p1_checkpoint_sha256": sha256(p1_checkpoint),
        "prepared_cache_sha256": sha256(prepared_path),
        "schema_cache_sha256": sha256(schema_cache_path),
        "plan_sha256": sha256(plan_path),
        "runtime_steps": runtime_steps,
        "views": view_results,
        "query_schema_alignment": _alignment_report(
            schema_encoder=schema_encoder,
            operator=operator,
        ),
        "optimizer_created": False,
        "gradient_performed": False,
        "parameters_mutated": False,
        "test_split_opened": False,
        "frozen_final_opened": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "gpu_required": False,
        "automatic_rerun_authorized": False,
        "automatic_hotfix_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
