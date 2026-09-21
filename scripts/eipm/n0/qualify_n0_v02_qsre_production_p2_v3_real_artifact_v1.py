from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alice_personality.n0.qsre_production_operator_v3 import (
    QSREProductionOperatorInducerV3,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import config_from_plan, load_plan
from train_n0_v02_qsre_production_p2_v3 import load_p1


EXPECTED_LOCALIZATION_SHA = (
    "da87d9436a7d4a928e45ea65582ab432c9d43254e2d1cbbaa8235ad28627e55b"
)
EXPECTED_FAILED_P2_SHA = (
    "3bcd557c38b27b6f7768901015aeee1ece8f7ccb8a2de0ae0e4c495ef79c4a6b"
)


def initialize_from_p1(operator, schema_encoder) -> None:
    with torch.no_grad():
        operator.query_norm.weight.copy_(schema_encoder.schema_norm.weight)
        operator.query_norm.bias.copy_(schema_encoder.schema_norm.bias)
        operator.query_projection.weight.copy_(
            schema_encoder.schema_projection.weight
        )
        operator.layer_embedding.weight.copy_(
            schema_encoder.layer_embedding.weight
        )


def assert_output(
    *,
    output: dict,
    batch: int,
    relations: int,
    steps: int,
    tokens: int,
) -> dict[str, float]:
    operator = output["operator"]
    relation = operator.relation_distribution
    event = output["event_distribution"]
    coverage = output["step_query_coverage"]
    remaining = output["step_query_remaining"]

    assert relation.shape == (batch, steps, relations)
    assert event.shape == (batch, steps, 3)
    assert coverage.shape == (batch, steps, tokens)
    assert remaining.shape == (batch, steps, tokens)

    for tensor in (
        relation,
        event,
        coverage,
        remaining,
        output["relation_logits"],
        output["event_logits"],
        output["step_query_attention"],
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("nonfinite ordered-evidence runtime output")

    if not torch.allclose(
        relation.sum(dim=-1),
        torch.ones(batch, steps),
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise RuntimeError("relation distribution normalization drift")
    if not torch.allclose(
        event.sum(dim=-1),
        torch.ones(batch, steps),
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise RuntimeError("event distribution normalization drift")

    if bool((coverage < -1.0e-7).any()) or bool((coverage > 1.0 + 1.0e-7).any()):
        raise RuntimeError("coverage outside [0,1]")
    if bool((remaining < 0.05 - 1.0e-6).any()) or bool((remaining > 1.0 + 1.0e-7).any()):
        raise RuntimeError("remaining evidence outside governed bounds")
    if steps > 1 and not bool(
        (coverage[:, 1:] + 1.0e-7 >= coverage[:, :-1]).all()
    ):
        raise RuntimeError("query coverage is not monotone")

    return {
        "coverage_max": float(coverage.max().item()),
        "coverage_mean_final": float(coverage[:, -1].mean().item()),
        "remaining_min": float(remaining.min().item()),
        "relation_sum_max_error": float(
            (relation.sum(dim=-1) - 1.0).abs().max().item()
        ),
        "event_sum_max_error": float(
            (event.sum(dim=-1) - 1.0).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--prepared-cache", required=True)
    parser.add_argument("--schema-cache", required=True)
    parser.add_argument("--p1-result", required=True)
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--failed-p2-result", required=True)
    parser.add_argument("--localization-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    plan_path = Path(args.plan).resolve()
    prepared_path = Path(args.prepared_cache).resolve()
    schema_cache_path = Path(args.schema_cache).resolve()
    p1_result_path = Path(args.p1_result).resolve()
    p1_root = Path(args.p1_root).resolve()
    failed_p2_path = Path(args.failed_p2_result).resolve()
    localization_path = Path(args.localization_result).resolve()
    output_path = Path(args.output).resolve()

    if output_path.exists():
        raise SystemExit("refusing to overwrite v3 real-artifact qualification")

    if sha256(localization_path) != EXPECTED_LOCALIZATION_SHA:
        raise SystemExit("P2 failure-localization hash drift")
    if sha256(failed_p2_path) != EXPECTED_FAILED_P2_SHA:
        raise SystemExit("failed P2 result hash drift")

    localization = json.loads(localization_path.read_text())
    if localization.get("status") != "PASS_ZERO_GRADIENT_P2_FAILURE_LOCALIZATION":
        raise SystemExit("P2 localization did not pass")
    if localization.get("gradient_performed") is not False:
        raise SystemExit("localization unexpectedly performed gradient")

    plan = load_plan(plan_path)
    if plan.get("schema") != "alice.eipm.n0.qsre-production-training-plan.v3":
        raise SystemExit("ordered-evidence v3 plan drift")
    config = config_from_plan(plan)

    prepared = torch.load(prepared_path, map_location="cpu")
    if prepared.get("private_identity_data") is not False:
        raise SystemExit("private identity data entered v3 qualification")
    if prepared.get("test_present") is not False:
        raise SystemExit("TEST entered v3 qualification")

    schema, schema_payload = load_dynamic_schema_cache(
        schema_cache_path,
        device=device,
    )
    schema_info = schema.validate(
        num_hidden_states=config.num_hidden_states,
        semantic_dim=config.semantic_dim,
    )
    relations = int(schema_info["relations"])

    schema_encoder, executor, p1 = load_p1(
        p1_result_path=p1_result_path,
        p1_root=p1_root,
        config=config,
        device=device,
    )
    del executor

    model = QSREProductionOperatorInducerV3(config).to(device)
    initialize_from_p1(model, schema_encoder)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    schema_encoder.eval()

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit("operator gradient unexpectedly enabled")

    encoded = schema_encoder(schema)
    dev = prepared["dev"]
    rows = min(8, len(dev["ids"]))
    indices = torch.arange(rows)
    steps = int(plan["stages"]["P2"]["training"]["operator_runtime_max_steps"])

    summaries = {}
    with torch.inference_mode():
        for view in (0, 1):
            hidden = dev["query_hidden_states"][indices, view].float()
            mask = dev["query_token_mask"][indices, view].bool()
            output = model(
                query_hidden_states=hidden,
                query_token_mask=mask,
                schema=schema,
                schema_token_state=encoded["schema_token_state"],
                schema_relation_state=encoded["schema_relation_state"],
                max_steps=steps,
            )
            summaries[f"view_{view}"] = assert_output(
                output=output,
                batch=rows,
                relations=relations,
                steps=steps,
                tokens=hidden.size(2),
            )

    report = model.parameter_report()
    required_true = (
        "runtime_dynamic_relation_schema",
        "shared_query_schema_metric",
        "relation_selection_decoupled_from_stop_unknown",
        "dense_relation_logits_exposed_for_trainability",
        "dense_event_and_factor_logits_exposed_for_trainability",
        "ordered_query_evidence_consumption",
        "coverage_is_token_position_not_hop_parameter",
        "coverage_preserves_nonzero_evidence_floor",
    )
    for key in required_true:
        if report.get(key) is not True:
            raise SystemExit(f"v3 parameter-report contract failed: {key}")
    if report.get("relation_count_dependent_parameters") != 0:
        raise SystemExit("relation-count parameter axis reintroduced")
    if report.get("hop_count_dependent_parameters") != 0:
        raise SystemExit("hop-count parameter axis reintroduced")

    result = {
        "schema": "alice.eipm.n0.qsre-production-p2-v3-real-artifact-qualification.v1",
        "status": "PASS_QSRE_PRODUCTION_P2_V3_ZERO_GRADIENT_REAL_ARTIFACT_RUNTIME",
        "plan_sha256": sha256(plan_path),
        "prepared_cache_sha256": sha256(prepared_path),
        "schema_cache_sha256": sha256(schema_cache_path),
        "failed_p2_result_sha256": sha256(failed_p2_path),
        "localization_result_sha256": sha256(localization_path),
        "p1_checkpoint_sha256": p1["sha256"],
        "runtime_relation_count": relations,
        "runtime_steps": steps,
        "rows_per_view": rows,
        "views": summaries,
        "parameter_report": report,
        "schema_core_train_relation_count": len(
            schema_payload["core_train_relation_keys"]
        ),
        "optimizer_created": False,
        "gradient_performed": False,
        "parameters_mutated": False,
        "gpu_required": False,
        "test_split_opened": False,
        "frozen_final_opened": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "p2_v3_gpu_training_authorized_by_this_receipt": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
