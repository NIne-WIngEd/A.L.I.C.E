from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from alice_personality.n0.qsre_production_core import (
    QSREDynamicRelationSchema,
    QSREProductionExecutor,
    QSREProductionSchemaEncoder,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import (
    batch_indices,
    config_from_plan,
    family_metrics,
    load_plan,
    oracle_operator_from_targets,
    summarize_downstream,
    target_distribution_loss,
)


def subset_schema(schema: QSREDynamicRelationSchema, count: int) -> QSREDynamicRelationSchema:
    return QSREDynamicRelationSchema(
        token_states=schema.token_states[:count],
        token_mask=schema.token_mask[:count],
        domain_type_mask=schema.domain_type_mask[:count],
        range_type_mask=schema.range_type_mask[:count],
        symmetric=schema.symmetric[:count],
    )


def move(tensor: torch.Tensor, indices: torch.Tensor, device: torch.device, *, dtype=None):
    value=tensor[indices].to(device)
    return value.to(dtype=dtype) if dtype is not None else value


def schema_geometry_loss(
    *,
    schema: QSREDynamicRelationSchema,
    encoded_relation_state: torch.Tensor,
) -> torch.Tensor:
    token = schema.token_states.float()
    mask = schema.token_mask[:, None, :, None].to(token.dtype)
    per_layer = (token * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
    raw_summary = per_layer.mean(dim=1)
    raw_norm = torch.nn.functional.normalize(raw_summary, dim=-1)
    encoded_norm = torch.nn.functional.normalize(encoded_relation_state, dim=-1)
    raw_geometry = raw_norm @ raw_norm.transpose(0, 1)
    encoded_geometry = encoded_norm @ encoded_norm.transpose(0, 1)
    return torch.nn.functional.mse_loss(encoded_geometry, raw_geometry)


def execute_batch(
    *,
    split: dict,
    indices: torch.Tensor,
    schema: QSREDynamicRelationSchema,
    schema_encoder: QSREProductionSchemaEncoder,
    executor: QSREProductionExecutor,
    config,
    device: torch.device,
):
    encoded=schema_encoder(schema)
    operator=oracle_operator_from_targets(
        split,
        indices,
        schema_relation_state=encoded["schema_relation_state"],
        config=config,
        device=device,
    )
    output=executor(
        field_state=move(split["field_state"],indices,device,dtype=torch.float32),
        field_metadata=move(split["field_metadata"],indices,device,dtype=torch.float32),
        field_valid_mask=move(split["field_valid_mask"],indices,device).bool(),
        edge_index=move(split["edge_index"],indices,device).long(),
        edge_relation_index=move(split["edge_relation_index"],indices,device).long(),
        edge_valid_mask=move(split["edge_valid_mask"],indices,device).bool(),
        edge_support_weight=move(split["oracle_edge_support"],indices,device,dtype=torch.float32),
        edge_reliability=move(split["edge_reliability"],indices,device,dtype=torch.float32),
        edge_recency=move(split["edge_recency"],indices,device,dtype=torch.float32),
        edge_temporal_match=move(split["edge_temporal_match"],indices,device,dtype=torch.float32),
        edge_provenance_match=move(split["edge_provenance_match"],indices,device,dtype=torch.float32),
        schema_relation_state=encoded["schema_relation_state"],
        operator=operator,
        focus_field_weight=move(split["focus_field_weight"],indices,device,dtype=torch.float32),
    )
    target=move(split["target_distribution"],indices,device,dtype=torch.float32)
    return output,target


@torch.no_grad()
def evaluate(
    *,
    split: dict,
    schema: QSREDynamicRelationSchema,
    schema_encoder: QSREProductionSchemaEncoder,
    executor: QSREProductionExecutor,
    config,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    schema_encoder.eval()
    executor.eval()
    probabilities=[]
    for start in range(0,len(split["ids"]),batch_size):
        idx=torch.arange(start,min(start+batch_size,len(split["ids"])))
        output,_=execute_batch(
            split=split,
            indices=idx,
            schema=schema,
            schema_encoder=schema_encoder,
            executor=executor,
            config=config,
            device=device,
        )
        probabilities.append(output["relational_probability"].detach().cpu())
    probability=torch.cat(probabilities,dim=0)
    metrics=summarize_downstream(
        probability=probability,
        target=split["target_distribution"].float(),
        control_target=split["control_target"].long(),
        families=split["families"],
        open_schema=split["open_schema"].bool(),
        causal_groups=split["causal_groups"],
    )
    metrics.pop("row_success_tensor",None)
    return metrics


def eligible(metrics: dict, thresholds: dict) -> bool:
    return (
        metrics["row_success_accuracy"] >= thresholds["row_success_accuracy"]
        and metrics["single_target_top1_accuracy"] >= thresholds["single_target_top1_accuracy"]
        and metrics["plural_l1"] <= thresholds["plural_l1_max"]
        and metrics["family_min_success"] >= thresholds["family_min_success"]
        and metrics["open_schema_success"] >= thresholds["open_schema_success"]
        and metrics["path_family_min_success"] >= thresholds["path_family_min_success"]
        and metrics["outside_support_invariance_max_delta"]
        <= thresholds["outside_support_invariance_max_delta"]
    )


def score_tuple(metrics: dict) -> tuple[float,...]:
    return (
        float(metrics["row_success_accuracy"]),
        float(metrics["family_min_success"]),
        float(metrics["open_schema_success"]),
        float(metrics["path_family_min_success"]),
        -float(metrics["plural_l1"]),
    )


def save_checkpoint(
    *,
    path: Path,
    schema_encoder,
    executor,
    config,
    step: int,
    plan_sha: str,
    prepared_sha: str,
    schema_cache_sha: str,
) -> str:
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "schema":"alice.eipm.n0.qsre-production-p1-checkpoint.v1",
        "stage":"P1",
        "step":int(step),
        "config":config.__dict__,
        "schema_encoder":schema_encoder.state_dict(),
        "executor":executor.state_dict(),
        "plan_sha256":plan_sha,
        "prepared_cache_sha256":prepared_sha,
        "schema_cache_sha256":schema_cache_sha,
    }
    torch.save(payload,path)
    return sha256(path)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--plan",required=True)
    p.add_argument("--p0-receipt",required=True)
    p.add_argument("--prepared-cache",required=True)
    p.add_argument("--schema-cache",required=True)
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()

    plan_path=Path(args.plan)
    p0_path=Path(args.p0_receipt)
    prepared_path=Path(args.prepared_cache)
    schema_cache_path=Path(args.schema_cache)
    output_dir=Path(args.output_dir)

    if output_dir.exists():
        raise SystemExit("refusing to overwrite P1 evidence")
    if not torch.cuda.is_available():
        raise SystemExit("P1 requires CUDA")

    plan=load_plan(plan_path)
    stage=plan["stages"]["P1"]
    if int(stage["max_gpu_runs"])!=1:
        raise SystemExit("P1 one-shot run count drift")
    p0=json.loads(p0_path.read_text())
    if p0.get("status")!="PASS_QSRE_PRODUCTION_P0_REAL_ARTIFACT_RUNTIME":
        raise SystemExit("P0 did not authorize P1")
    if p0.get("p1_authorized_by_this_receipt") is not True:
        raise SystemExit("P0 P1 authorization missing")
    if p0.get("prepared_cache_sha256")!=sha256(prepared_path):
        raise SystemExit("P0/prepared cache hash drift")
    if p0.get("schema_cache_sha256")!=sha256(schema_cache_path):
        raise SystemExit("P0/schema cache hash drift")

    prepared=torch.load(prepared_path,map_location="cpu")
    if prepared.get("private_identity_data") is not False or prepared.get("test_present") is not False:
        raise SystemExit("P1 boundary violation")

    device=torch.device("cuda")
    full_schema,schema_payload=load_dynamic_schema_cache(schema_cache_path,device=device)
    core_count=len(schema_payload["core_train_relation_keys"])
    train_schema=subset_schema(full_schema,core_count)
    config=config_from_plan(plan)

    schema_encoder=QSREProductionSchemaEncoder(config).to(device)
    executor=QSREProductionExecutor(config).to(device)

    for name,module in (("schema_encoder",schema_encoder),("executor",executor)):
        if name not in stage["trainable"]:
            raise SystemExit(f"P1 trainable-scope drift: {name}")
    parameters=list(schema_encoder.parameters())+list(executor.parameters())
    loss_weights=stage["loss_weights"]
    optimizer=torch.optim.AdamW(
        parameters,
        lr=float(stage["optimizer"]["learning_rate"]),
        weight_decay=float(stage["optimizer"]["weight_decay"]),
    )

    training=stage["training"]
    seed=int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_size=int(training["batch_size"])
    max_steps=int(training["max_steps"])
    eval_every=int(training["eval_every"])

    output_dir.mkdir(parents=True)
    plan_sha=sha256(plan_path)
    prepared_sha=sha256(prepared_path)
    schema_cache_sha=sha256(schema_cache_path)
    train=prepared["train"]
    dev=prepared["dev"]

    history=[]
    selected=None
    best=None
    best_score=None
    train_batches=[]
    epoch=0
    step=0

    while step < max_steps:
        if not train_batches:
            train_batches=batch_indices(
                len(train["ids"]),
                batch_size=batch_size,
                seed=seed+epoch,
            )
            epoch+=1
        idx=train_batches.pop(0)
        schema_encoder.train()
        executor.train()
        optimizer.zero_grad(set_to_none=True)
        output,target=execute_batch(
            split=train,
            indices=idx,
            schema=train_schema,
            schema_encoder=schema_encoder,
            executor=executor,
            config=config,
            device=device,
        )
        downstream_loss=target_distribution_loss(
            output["relational_probability"],
            target,
        )
        encoded_now=schema_encoder(train_schema)
        geometry_loss=schema_geometry_loss(
            schema=train_schema,
            encoded_relation_state=encoded_now["schema_relation_state"],
        )
        loss=(
            float(loss_weights["downstream"]) * downstream_loss
            + float(loss_weights["schema_geometry"]) * geometry_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("P1 nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.0)
        optimizer.step()
        step+=1

        if step % eval_every==0 or step==max_steps:
            metrics=evaluate(
                split=dev,
                schema=full_schema,
                schema_encoder=schema_encoder,
                executor=executor,
                config=config,
                device=device,
                batch_size=batch_size,
            )
            is_eligible=eligible(metrics,stage["eligibility"])
            record={
                "step":step,
                "train_loss":float(loss.item()),
                "train_downstream_loss":float(downstream_loss.detach().item()),
                "train_schema_geometry_loss":float(geometry_loss.detach().item()),
                "eligible":is_eligible,
                "dev":metrics,
            }
            history.append(record)
            print("P1_EVAL="+json.dumps(record,sort_keys=True),flush=True)

            ckpt_path=output_dir/f"step-{step:08d}"/"qsre_production_p1.pt"
            ckpt_sha=save_checkpoint(
                path=ckpt_path,
                schema_encoder=schema_encoder,
                executor=executor,
                config=config,
                step=step,
                plan_sha=plan_sha,
                prepared_sha=prepared_sha,
                schema_cache_sha=schema_cache_sha,
            )
            score=score_tuple(metrics)
            if best_score is None or score>best_score:
                best_score=score
                best={"step":step,"checkpoint_sha256":ckpt_sha,"metrics":metrics}
            if is_eligible:
                selected={"step":step,"checkpoint_sha256":ckpt_sha,"metrics":metrics}
                break

    status="PASS_QSRE_PRODUCTION_P1_EXECUTOR" if selected else "FAIL_QSRE_PRODUCTION_P1_EXECUTOR"
    result={
        "schema":"alice.eipm.n0.qsre-production-p1-result.v1",
        "status":status,
        "selected":selected,
        "best_observed":best,
        "history":history,
        "plan_sha256":plan_sha,
        "prepared_cache_sha256":prepared_sha,
        "schema_cache_sha256":schema_cache_sha,
        "train_relation_count":core_count,
        "dev_relation_count":int(full_schema.token_states.size(0)),
        "semantic_backbone_gradient":False,
        "operator_gradient":False,
        "binder_gradient":False,
        "schema_encoder_gradient":True,
        "executor_gradient":True,
        "private_identity_gradient":False,
        "automatic_rerun":False,
        "automatic_hotfix":False,
        "p2_authorized":bool(selected),
    }
    result_path=output_dir/"result.json"
    result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("P1_RESULT="+json.dumps(result,sort_keys=True),flush=True)
    if not selected:
        raise SystemExit(20)


if __name__=="__main__":
    main()
