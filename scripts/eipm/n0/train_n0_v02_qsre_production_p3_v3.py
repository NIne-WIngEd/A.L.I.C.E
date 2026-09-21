from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from alice_personality.n0.qsre_production_core import (
    QSREDynamicRelationSchema,
    QSREProductionExecutor,
    QSREProductionSchemaEncoder,
)
from alice_personality.n0.qsre_production_binder_v2 import (
    QSREProductionBinderV2,
)
from alice_personality.n0.qsre_production_operator_v3 import (
    QSREProductionOperatorInducerV3,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import (
    batch_indices,
    config_from_plan,
    load_plan,
    oracle_operator_from_targets,
    focus_metrics,
    summarize_downstream,
    support_metrics,
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


def balanced_binary_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    selected_logits = logits[mask]
    selected_target = target[mask]
    losses = []
    for value in (0.0, 1.0):
        cls = selected_target.eq(value)
        if bool(cls.any()):
            losses.append(
                F.binary_cross_entropy_with_logits(
                    selected_logits[cls],
                    selected_target[cls],
                )
            )
    return torch.stack(losses).mean()


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


def load_selected(result_path:Path,root:Path,filename:str,expected_status:str)->tuple[Path,dict]:
    result=json.loads(result_path.read_text())
    if result.get("status")!=expected_status or not result.get("selected"):
        raise SystemExit(f"required prior stage did not pass: {expected_status}")
    step=int(result["selected"]["step"])
    path=root/f"step-{step:08d}"/filename
    if sha256(path)!=result["selected"]["checkpoint_sha256"]:
        raise SystemExit("selected checkpoint hash drift")
    return path,result


def load_parents(*,p1_path:Path,p2_path:Path,factor_schema_cache_path:Path,config,device):
    p1=torch.load(p1_path,map_location="cpu")
    p2=torch.load(p2_path,map_location="cpu")
    factor_cache=torch.load(factor_schema_cache_path,map_location="cpu")
    if p2.get("factor_schema_cache_sha256") != sha256(factor_schema_cache_path):
        raise SystemExit("P2 factor-schema cache lineage drift")
    schema_encoder=QSREProductionSchemaEncoder(config)
    executor=QSREProductionExecutor(config)
    operator=QSREProductionOperatorInducerV3(config)
    operator.configure_factor_schema_cache(factor_cache)
    schema_encoder.load_state_dict(p1["schema_encoder"],strict=True)
    executor.load_state_dict(p1["executor"],strict=True)
    operator.load_state_dict(p2["operator"],strict=True)
    for module in (schema_encoder,executor,operator):
        for parameter in module.parameters():
            parameter.requires_grad=False
        module.to(device).eval()
    return schema_encoder,executor,operator


def infer_operator(*,split,indices,view,schema,encoded,operator,max_steps,device):
    return operator(
        query_hidden_states=split["query_hidden_states"][indices,view].to(device).float(),
        query_token_mask=split["query_token_mask"][indices,view].to(device).bool(),
        schema=schema,
        schema_token_state=encoded["schema_token_state"],
        schema_relation_state=encoded["schema_relation_state"],
        max_steps=max_steps,
    )["operator"]


def bind(
    *,
    split,indices,view,schema,encoded,binder,operator_state,device,
):
    return binder(
        query_hidden_states=split["query_hidden_states"][indices,view].to(device).float(),
        query_token_mask=split["query_token_mask"][indices,view].to(device).bool(),
        field_token_states=split["field_token_states"][indices].to(device).float(),
        field_token_mask=split["field_token_mask"][indices].to(device).bool(),
        field_type_id=split["field_type_id"][indices].to(device).long(),
        edge_index=split["edge_index"][indices].to(device).long(),
        edge_relation_index=split["edge_relation_index"][indices].to(device).long(),
        edge_valid_mask=split["edge_valid_mask"][indices].to(device).bool(),
        edge_reliability=split["edge_reliability"][indices].to(device).float(),
        edge_recency=split["edge_recency"][indices].to(device).float(),
        schema=schema,
        schema_relation_state=encoded["schema_relation_state"],
        operator=operator_state,
    )


def execute(*,split,indices,encoded,executor,operator_state,support,focus,device):
    return executor(
        field_state=split["field_state"][indices].to(device).float(),
        field_metadata=split["field_metadata"][indices].to(device).float(),
        field_valid_mask=split["field_valid_mask"][indices].to(device).bool(),
        edge_index=split["edge_index"][indices].to(device).long(),
        edge_relation_index=split["edge_relation_index"][indices].to(device).long(),
        edge_valid_mask=split["edge_valid_mask"][indices].to(device).bool(),
        edge_support_weight=support.float(),
        edge_reliability=split["edge_reliability"][indices].to(device).float(),
        edge_recency=split["edge_recency"][indices].to(device).float(),
        edge_temporal_match=split["edge_temporal_match"][indices].to(device).float(),
        edge_provenance_match=split["edge_provenance_match"][indices].to(device).float(),
        schema_relation_state=encoded["schema_relation_state"],
        operator=operator_state,
        focus_field_weight=focus.float(),
    )


@torch.no_grad()
def evaluate_mode(
    *,
    split,schema,encoded,schema_encoder,executor,operator_model,binder,
    config,max_steps,batch_size,device,predicted_operator:bool,
):
    supports=[]; oracle_supports=[]; compatible=[]; probabilities=[]; focuses=[]; oracle_focuses=[]
    for start in range(0,len(split["ids"]),batch_size):
        idx=torch.arange(start,min(start+batch_size,len(split["ids"])))
        # Strict across both paraphrase views.
        view_support=[]; view_compat=[]; view_prob=[]
        for view in (0,1):
            if predicted_operator:
                op=infer_operator(
                    split=split,indices=idx,view=view,schema=schema,encoded=encoded,
                    operator=operator_model,max_steps=max_steps,device=device,
                )
            else:
                op=oracle_operator_from_targets(
                    split,idx,
                    schema_relation_state=encoded["schema_relation_state"],
                    config=config,device=device,
                )
            bound=bind(
                split=split,indices=idx,view=view,schema=schema,encoded=encoded,
                binder=binder,operator_state=op,device=device,
            )
            out=execute(
                split=split,indices=idx,encoded=encoded,executor=executor,
                operator_state=op,support=bound["edge_support_weight"],
                focus=bound["focus_field_weight"],device=device,
            )
            focuses.append(bound["focus_field_weight"].cpu())
            oracle_focuses.append(split["focus_field_weight"][idx].float())
            view_support.append(bound["edge_support_weight"].cpu())
            view_compat.append(bound["type_compatible"].cpu())
            view_prob.append(out["relational_probability"].cpu())
        # Concatenate views so any weak paraphrase lowers aggregate metrics.
        supports.extend(view_support)
        compatible.extend(view_compat)
        probabilities.extend(view_prob)
        oracle=split["oracle_edge_support"][idx].float()
        oracle_supports.extend([oracle,oracle])

    predicted=torch.cat(supports,dim=0)
    oracle=torch.cat(oracle_supports,dim=0)
    type_ok=torch.cat(compatible,dim=0)
    probability=torch.cat(probabilities,dim=0)

    support=support_metrics(predicted=predicted,oracle=oracle,type_compatible=type_ok)
    focus_predicted=torch.cat(focuses,dim=0)
    focus_oracle=torch.cat(oracle_focuses,dim=0)
    traversal=torch.cat(
        [split["traversal_target"].long(),split["traversal_target"].long()],
        dim=0,
    )
    focus=focus_metrics(
        predicted=focus_predicted,
        oracle=focus_oracle,
        traversal_target=traversal,
    )
    target=torch.cat([split["target_distribution"].float(),split["target_distribution"].float()],dim=0)
    control=torch.cat([split["control_target"].long(),split["control_target"].long()],dim=0)
    families=list(split["families"])+list(split["families"])
    groups=list(split["causal_groups"])+[
        str(group)+":view1" for group in split["causal_groups"]
    ]
    open_schema=torch.cat([split["open_schema"].bool(),split["open_schema"].bool()],dim=0)
    downstream=summarize_downstream(
        probability=probability,target=target,control_target=control,
        families=families,open_schema=open_schema,causal_groups=groups,
    )
    downstream.pop("row_success_tensor",None)
    return {"support":support,"focus":focus,"downstream":downstream}


@torch.no_grad()
def evaluate(**kwargs):
    oracle=evaluate_mode(predicted_operator=False,**kwargs)
    predicted=evaluate_mode(predicted_operator=True,**kwargs)
    return {"oracle_operator":oracle,"predicted_operator":predicted}


def eligible(metrics:dict,thresholds:dict)->bool:
    o=metrics["oracle_operator"]; p=metrics["predicted_operator"]
    return (
        o["support"]["edge_f1"]>=thresholds["oracle_support_edge_f1"]
        and o["support"]["exact_set_accuracy"]>=thresholds["oracle_support_exact_set_accuracy"]
        and o["support"]["type_violation_rate"]<=thresholds["type_violation_rate_max"]
        and o["focus"]["path_focus_top1_accuracy"]>=thresholds["oracle_path_focus_top1_accuracy"]
        and o["focus"]["path_focus_exact_set_accuracy"]>=thresholds["oracle_path_focus_exact_set_accuracy"]
        and o["downstream"]["row_success_accuracy"]>=thresholds["oracle_downstream_row_success_accuracy"]
        and p["support"]["edge_f1"]>=thresholds["predicted_operator_support_edge_f1"]
        and p["focus"]["path_focus_top1_accuracy"]>=thresholds["predicted_operator_path_focus_top1_accuracy"]
        and p["downstream"]["row_success_accuracy"]>=thresholds["predicted_operator_downstream_row_success_accuracy"]
    )


def score_tuple(metrics:dict)->tuple[float,...]:
    o=metrics["oracle_operator"]; p=metrics["predicted_operator"]
    return (
        o["support"]["edge_f1"],
        o["support"]["exact_set_accuracy"],
        o["focus"]["path_focus_top1_accuracy"],
        p["support"]["edge_f1"],
        p["focus"]["path_focus_top1_accuracy"],
        p["downstream"]["row_success_accuracy"],
        o["downstream"]["row_success_accuracy"],
    )


def save_checkpoint(path:Path,binder,config,step,lineage)->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"alice.eipm.n0.qsre-production-p3-checkpoint.v3",
        "architecture":"operator-v3-plus-open-schema-binder-v2",
        "stage":"P3","step":step,"config":config.__dict__,
        "binder":binder.state_dict(),**lineage,
    },path)
    return sha256(path)


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--plan",required=True)
    p.add_argument("--prepared-cache",required=True)
    p.add_argument("--schema-cache",required=True)
    p.add_argument("--p1-result",required=True)
    p.add_argument("--p1-root",required=True)
    p.add_argument("--p2-result",required=True)
    p.add_argument("--p2-root",required=True)
    p.add_argument("--factor-schema-cache",required=True)
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("P3 requires CUDA")
    output_dir=Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("refusing to overwrite P3 evidence")

    plan_path=Path(args.plan); prepared_path=Path(args.prepared_cache); schema_cache_path=Path(args.schema_cache)
    plan=load_plan(plan_path); stage=plan["stages"]["P3"]; config=config_from_plan(plan)
    prepared=torch.load(prepared_path,map_location="cpu")
    if prepared.get("private_identity_data") is not False or prepared.get("test_present") is not False:
        raise SystemExit("P3 boundary violation")

    p1_path,_=load_selected(
        Path(args.p1_result),Path(args.p1_root),
        "qsre_production_p1.pt","PASS_QSRE_PRODUCTION_P1_EXECUTOR",
    )
    p2_path,_=load_selected(
        Path(args.p2_result),Path(args.p2_root),
        "qsre_production_p2.pt","PASS_QSRE_PRODUCTION_P2_OPERATOR",
    )
    device=torch.device("cuda")
    schema_encoder,executor,operator_model=load_parents(
        p1_path=p1_path,p2_path=p2_path,
        factor_schema_cache_path=Path(args.factor_schema_cache),
        config=config,device=device,
    )
    full_schema,schema_payload=load_dynamic_schema_cache(schema_cache_path,device=device)
    core_keys=list(schema_payload["core_train_relation_keys"])
    relation_keys=list(schema_payload["relation_keys"])
    core_count=len(core_keys)
    if relation_keys[:core_count] != core_keys:
        raise SystemExit("core relation schema must occupy the stable training prefix")
    core_schema=subset_schema(full_schema,core_count)
    with torch.no_grad():
        encoded_train=schema_encoder(core_schema)
        encoded_dev=schema_encoder(full_schema)

    binder=QSREProductionBinderV2(config).to(device)
    # Query and field tokens share one trainable metric initialized in the
    # proven P1 semantic coordinate system. Relation semantics remain owned by
    # the operator rather than being relearned as binder-specific relation IDs.
    with torch.no_grad():
        binder.semantic_projection.weight.copy_(
            schema_encoder.schema_projection.weight
        )
        binder.semantic_norm.weight.copy_(schema_encoder.schema_norm.weight)
        binder.semantic_norm.bias.copy_(schema_encoder.schema_norm.bias)
    optimizer=torch.optim.AdamW(
        binder.parameters(),
        lr=float(stage["optimizer"]["learning_rate"]),
        weight_decay=float(stage["optimizer"]["weight_decay"]),
    )
    seed=int(stage["training"]["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    batch_size=int(stage["training"]["batch_size"]); max_train_steps=int(stage["training"]["max_steps"])
    eval_every=int(stage["training"]["eval_every"]); runtime_steps=int(stage["training"]["operator_runtime_max_steps"])
    weights=stage["loss_weights"]
    train=prepared["train"]; dev=prepared["dev"]

    output_dir.mkdir(parents=True)
    batches=[]; epoch=0; step=0; history=[]; selected=None; best=None; best_score=None
    while step<max_train_steps:
        if not batches:
            batches=batch_indices(len(train["ids"]),batch_size=batch_size,seed=seed+epoch)
            epoch+=1
        idx=batches.pop(0)
        oracle_op=oracle_operator_from_targets(
            train,idx,schema_relation_state=encoded_train["schema_relation_state"],
            config=config,device=device,
        )
        binder.train(); optimizer.zero_grad(set_to_none=True)
        view_losses=[]
        for view in (0,1):
            bound=bind(
                split=train,indices=idx,view=view,schema=core_schema,
                encoded=encoded_train,binder=binder,operator_state=oracle_op,device=device,
            )
            valid=train["edge_valid_mask"][idx].to(device).bool()
            target=train["oracle_edge_support"][idx].to(device).float()
            support_loss=balanced_binary_bce_with_logits(
                bound["support_logits"],
                target,
                valid,
            )
            path_rows=train["traversal_target"][idx].to(device).eq(1)
            field_valid=train["field_valid_mask"][idx].to(device).bool()
            focus_target=train["focus_field_weight"][idx].to(device).float()
            focus_supervised=path_rows[:,None] & field_valid
            focus_loss=balanced_binary_bce_with_logits(
                bound["focus_logits"],
                focus_target,
                focus_supervised,
            )
            impossible=valid & ~bound["type_compatible"]
            type_penalty=(
                torch.sigmoid(bound["support_logits"][impossible]).mean()
                if bool(impossible.any())
                else bound["support_logits"].sum()*0.0
            )
            downstream=execute(
                split=train,indices=idx,encoded=encoded_train,executor=executor,
                operator_state=oracle_op,support=bound["edge_support_weight"],
                focus=bound["focus_field_weight"],device=device,
            )
            down_loss=target_distribution_loss(
                downstream["relational_probability"],
                train["target_distribution"][idx].to(device).float(),
            )
            view_losses.append(
                float(weights["support_bce"])*support_loss
                +float(weights["focus_bce"])*focus_loss
                +float(weights["downstream"])*down_loss
                +float(weights["type_violation"])*type_penalty
            )
        loss=0.5*(view_losses[0]+view_losses[1])
        if not torch.isfinite(loss):
            raise RuntimeError("P3 nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(binder.parameters(),1.0)
        optimizer.step(); step+=1

        if step%eval_every==0 or step==max_train_steps:
            binder.eval()
            metrics=evaluate(
                split=dev,schema=full_schema,encoded=encoded_dev,
                schema_encoder=schema_encoder,executor=executor,operator_model=operator_model,
                binder=binder,config=config,max_steps=runtime_steps,
                batch_size=batch_size,device=device,
            )
            is_eligible=eligible(metrics,stage["eligibility"])
            record={"step":step,"train_loss":float(loss.item()),"eligible":is_eligible,"dev":metrics}
            history.append(record)
            print("P3_EVAL="+json.dumps(record,sort_keys=True),flush=True)
            path=output_dir/f"step-{step:08d}"/"qsre_production_p3.pt"
            ckpt_sha=save_checkpoint(
                path,binder,config,step,
                {
                    "plan_sha256":sha256(plan_path),
                    "prepared_cache_sha256":sha256(prepared_path),
                    "schema_cache_sha256":sha256(schema_cache_path),
                    "p1_checkpoint_sha256":sha256(p1_path),
                    "p2_checkpoint_sha256":sha256(p2_path),
                },
            )
            score=score_tuple(metrics)
            if best_score is None or score>best_score:
                best_score=score; best={"step":step,"checkpoint_sha256":ckpt_sha,"metrics":metrics}
            if is_eligible:
                selected={"step":step,"checkpoint_sha256":ckpt_sha,"metrics":metrics}
                break

    status="PASS_QSRE_PRODUCTION_P3_BINDER" if selected else "FAIL_QSRE_PRODUCTION_P3_BINDER"
    result={
        "schema":"alice.eipm.n0.qsre-production-p3-result.v3",
        "architecture":"operator-v3-plus-open-schema-binder-v2",
        "status":status,"selected":selected,"best_observed":best,"history":history,
        "plan_sha256":sha256(plan_path),
        "prepared_cache_sha256":sha256(prepared_path),
        "schema_cache_sha256":sha256(schema_cache_path),
        "p1_checkpoint_sha256":sha256(p1_path),
        "p2_checkpoint_sha256":sha256(p2_path),
        "schema_encoder_gradient":False,"executor_gradient":False,
        "operator_gradient":False,"binder_gradient":True,
        "private_identity_gradient":False,
        "automatic_rerun":False,"automatic_hotfix":False,
        "p4_authorized":bool(selected),
        "train_positive_relation_count":core_count,
        "runtime_relation_count":int(full_schema.token_states.size(0)),
        "open_schema_training_query_labels_used":False,
        "open_schema_relation_descriptions_used_in_gradient":False,
        "core_schema_only_binder_training":True,
        "full_runtime_schema_candidates_used_during_training":False,
        "balanced_support_and_focus_supervision":True,
        "continuous_relation_hypotheses_from_operator":True,
        "exact_sparsity_owned_by_binder":True,
    }
    (output_dir/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("P3_RESULT="+json.dumps(result,sort_keys=True),flush=True)
    if not selected:
        raise SystemExit(22)


if __name__=="__main__":
    main()
