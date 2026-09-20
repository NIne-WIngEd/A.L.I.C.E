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
    QSREProductionOperatorInducer,
    QSREProductionOperatorState,
    QSREProductionSchemaEncoder,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import (
    batch_indices,
    config_from_plan,
    load_plan,
    modifier_exact,
    operator_metrics,
    oracle_operator_from_targets,
    paired_view_consistency,
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


def load_p1(
    *,
    p1_result_path: Path,
    p1_root: Path,
    config,
    device: torch.device,
) -> tuple[QSREProductionSchemaEncoder,QSREProductionExecutor,dict]:
    result=json.loads(p1_result_path.read_text())
    if result.get("status")!="PASS_QSRE_PRODUCTION_P1_EXECUTOR":
        raise SystemExit("P1 did not pass")
    selected=result.get("selected")
    if not selected:
        raise SystemExit("P1 selected checkpoint missing")
    step=int(selected["step"])
    path=p1_root/f"step-{step:08d}"/"qsre_production_p1.pt"
    if sha256(path)!=selected["checkpoint_sha256"]:
        raise SystemExit("P1 selected checkpoint hash drift")
    payload=torch.load(path,map_location="cpu")
    schema_encoder=QSREProductionSchemaEncoder(config)
    executor=QSREProductionExecutor(config)
    schema_encoder.load_state_dict(payload["schema_encoder"],strict=True)
    executor.load_state_dict(payload["executor"],strict=True)
    for module in (schema_encoder,executor):
        for parameter in module.parameters():
            parameter.requires_grad=False
        module.to(device).eval()
    return schema_encoder,executor,{"path":path,"sha256":sha256(path),"step":step}


def operator_supervised_loss(
    *,
    operator: QSREProductionOperatorState,
    split: dict,
    indices: torch.Tensor,
    oracle_continuous: torch.Tensor,
    weights: dict,
    device: torch.device,
) -> tuple[torch.Tensor,dict[str,float]]:
    relation_target=split["relation_target"][indices].to(device).long()
    relation_mask=split["relation_target_mask"][indices].to(device).bool()
    role_target=split["role_target"][indices].to(device).long()
    traversal_target=split["traversal_target"][indices].to(device).long()
    modifier_target=split["modifier_target"][indices].to(device).float()
    applicability_target=split["applicability_target"][indices].to(device).float()
    control_target=split["control_target"][indices].to(device).long()
    termination_target=split["termination_target"][indices].to(device).long()

    batch,pred_steps,_=operator.relation_distribution.shape
    target_steps=relation_target.size(1)
    if pred_steps<target_steps:
        raise RuntimeError("operator runtime budget shorter than supervision")
    if pred_steps>target_steps:
        pad=pred_steps-target_steps
        relation_target=F.pad(relation_target,(0,pad),value=0)
        relation_mask=F.pad(relation_mask,(0,pad),value=False)

    relation_prob=operator.relation_distribution.clamp_min(1.0e-8)
    gather=relation_prob.gather(-1,relation_target.unsqueeze(-1)).squeeze(-1)
    if bool(relation_mask.any()):
        relation_nll=-(gather[relation_mask].log()).mean()
    else:
        relation_nll=relation_prob.sum()*0.0

    active_target=relation_mask.float()
    step_mass_loss=F.binary_cross_entropy(
        operator.relation_step_mass.clamp(1e-6,1-1e-6),
        active_target,
    )

    lengths=relation_mask.long().sum(dim=-1)
    termination_supervised=relation_mask.clone()
    stop_target=torch.zeros_like(operator.stop_probability)
    unknown_target=torch.zeros_like(operator.unknown_probability)
    for row,length in enumerate(lengths.tolist()):
        if length<pred_steps:
            termination_supervised[row,length]=True
            if int(termination_target[row].item()) == 1:
                unknown_target[row,length]=1.0
            else:
                stop_target[row,length]=1.0
    stop_loss=F.binary_cross_entropy(
        operator.stop_probability[termination_supervised].clamp(1e-6,1-1e-6),
        stop_target[termination_supervised],
    )
    unknown_loss=F.binary_cross_entropy(
        operator.unknown_probability[termination_supervised].clamp(1e-6,1-1e-6),
        unknown_target[termination_supervised],
    )

    relational=control_target.eq(CONTROL_RELATIONAL)
    if bool(relational.any()):
        role_loss=F.nll_loss(
            operator.role_distribution[relational].clamp_min(1e-8).log(),
            role_target[relational],
        )
        traversal_loss=F.nll_loss(
            operator.traversal_distribution[relational].clamp_min(1e-8).log(),
            traversal_target[relational],
        )
        modifier_loss=F.binary_cross_entropy(
            operator.modifier_weight[relational],
            modifier_target[relational],
        )
    else:
        zero=operator.continuous_state.sum()*0.0
        role_loss=zero
        traversal_loss=zero
        modifier_loss=zero

    applicability_loss=F.mse_loss(operator.applicability,applicability_target)
    control_loss=F.nll_loss(
        operator.control_distribution.clamp_min(1e-8).log(),
        control_target,
    )

    relational_cont=relational & relation_mask.any(dim=-1)
    if bool(relational_cont.any()):
        continuous_loss=(
            1.0-F.cosine_similarity(
                operator.continuous_state[relational_cont],
                oracle_continuous[relational_cont].detach(),
                dim=-1,
            )
        ).mean()
    else:
        continuous_loss=operator.continuous_state.sum()*0.0

    relation_total=relation_nll+0.5*step_mass_loss+0.25*stop_loss+0.10*unknown_loss
    total=(
        float(weights["relation"])*relation_total
        +float(weights["role"])*role_loss
        +float(weights["traversal"])*traversal_loss
        +float(weights["modifiers"])*modifier_loss
        +float(weights["applicability"])*applicability_loss
        +float(weights["control"])*control_loss
        +float(weights["continuous_alignment"])*continuous_loss
    )
    parts={
        "relation":float(relation_total.detach().item()),
        "role":float(role_loss.detach().item()),
        "traversal":float(traversal_loss.detach().item()),
        "modifiers":float(modifier_loss.detach().item()),
        "applicability":float(applicability_loss.detach().item()),
        "control":float(control_loss.detach().item()),
        "continuous_alignment":float(continuous_loss.detach().item()),
    }
    return total,parts


def pair_loss(a: QSREProductionOperatorState,b: QSREProductionOperatorState) -> torch.Tensor:
    return (
        F.mse_loss(a.relation_distribution,b.relation_distribution)
        +F.mse_loss(a.relation_step_mass,b.relation_step_mass)
        +F.mse_loss(a.stop_probability,b.stop_probability)
        +F.mse_loss(a.unknown_probability,b.unknown_probability)
        +F.mse_loss(a.role_distribution,b.role_distribution)
        +F.mse_loss(a.traversal_distribution,b.traversal_distribution)
        +F.mse_loss(a.modifier_weight,b.modifier_weight)
        +F.mse_loss(a.applicability,b.applicability)
        +F.mse_loss(a.control_distribution,b.control_distribution)
        +0.25*(
            1.0-F.cosine_similarity(a.continuous_state,b.continuous_state,dim=-1)
        ).mean()
    )


def execute_with_oracle_support(
    *,
    split:dict,
    indices:torch.Tensor,
    operator:QSREProductionOperatorState,
    schema_relation_state:torch.Tensor,
    executor:QSREProductionExecutor,
    device:torch.device,
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


def infer_view(
    *,
    split:dict,
    indices:torch.Tensor,
    view:int,
    schema:QSREDynamicRelationSchema,
    encoded:dict[str,torch.Tensor],
    operator_model:QSREProductionOperatorInducer,
    max_steps:int,
    device:torch.device,
):
    return operator_model(
        query_hidden_states=split["query_hidden_states"][indices,view].to(device).float(),
        query_token_mask=split["query_token_mask"][indices,view].to(device).bool(),
        schema=schema,
        schema_token_state=encoded["schema_token_state"],
        schema_relation_state=encoded["schema_relation_state"],
        max_steps=max_steps,
    )["operator"]


@torch.no_grad()
def evaluate(
    *,
    split:dict,
    schema:QSREDynamicRelationSchema,
    schema_encoder:QSREProductionSchemaEncoder,
    executor:QSREProductionExecutor,
    operator_model:QSREProductionOperatorInducer,
    config,
    max_steps:int,
    batch_size:int,
    device:torch.device,
) -> dict[str,object]:
    schema_encoder.eval(); executor.eval(); operator_model.eval()
    encoded=schema_encoder(schema)
    per_view_operator=[[],[]]
    per_view_probability=[[],[]]

    # Store operator tensors in simple lists, then concatenate into dataclasses.
    for start in range(0,len(split["ids"]),batch_size):
        idx=torch.arange(start,min(start+batch_size,len(split["ids"])))
        for view in (0,1):
            op=infer_view(
                split=split,indices=idx,view=view,schema=schema,encoded=encoded,
                operator_model=operator_model,max_steps=max_steps,device=device,
            )
            per_view_operator[view].append(op)
            downstream=execute_with_oracle_support(
                split=split,indices=idx,operator=op,
                schema_relation_state=encoded["schema_relation_state"],
                executor=executor,device=device,
            )
            per_view_probability[view].append(downstream["relational_probability"].cpu())

    def concat_ops(values):
        fields=QSREProductionOperatorState.__dataclass_fields__
        return QSREProductionOperatorState(**{
            name:torch.cat([getattr(value,name).cpu() for value in values],dim=0)
            for name in fields
        })

    ops=[concat_ops(per_view_operator[0]),concat_ops(per_view_operator[1])]
    probs=[torch.cat(per_view_probability[0],dim=0),torch.cat(per_view_probability[1],dim=0)]
    all_idx=torch.arange(len(split["ids"]))

    op_metrics=[]
    downstream=[]
    for view in (0,1):
        metric=operator_metrics(operator=ops[view],split=split,indices=all_idx)
        op_metrics.append(metric)
        summary=summarize_downstream(
            probability=probs[view],
            target=split["target_distribution"].float(),
            control_target=split["control_target"].long(),
            families=split["families"],
            open_schema=split["open_schema"].bool(),
            causal_groups=split["causal_groups"],
        )
        summary.pop("row_success_tensor",None)
        downstream.append(summary)

    def min_metric(key):
        return min(float(op_metrics[0][key]),float(op_metrics[1][key]))
    operator_summary={
        "relation_sequence_exact_accuracy":min_metric("relation_sequence_exact_accuracy"),
        "open_schema_relation_exact_accuracy":min_metric("open_schema_relation_exact_accuracy"),
        "role_accuracy_relational":min_metric("role_accuracy_relational"),
        "traversal_accuracy_relational":min_metric("traversal_accuracy_relational"),
        "modifier_exact_accuracy_relational":min_metric("modifier_exact_accuracy_relational"),
        "control_accuracy":min_metric("control_accuracy"),
        "termination_accuracy":min_metric("termination_accuracy"),
        "unknown_termination_accuracy":min_metric("unknown_termination_accuracy"),
        "pair_consistency":paired_view_consistency(first=ops[0],second=ops[1]),
    }

    family_keys=set(downstream[0]["family_success"])|set(downstream[1]["family_success"])
    family_worst={
        family:min(
            float(downstream[0]["family_success"].get(family,1.0)),
            float(downstream[1]["family_success"].get(family,1.0)),
        )
        for family in sorted(family_keys)
    }
    downstream_summary={
        "row_success_accuracy":min(float(downstream[0]["row_success_accuracy"]),float(downstream[1]["row_success_accuracy"])),
        "family_min_success":min(family_worst.values()) if family_worst else 1.0,
        "family_success":family_worst,
        "open_schema_success":min(float(downstream[0]["open_schema_success"]),float(downstream[1]["open_schema_success"])),
        "single_target_top1_accuracy":min(float(downstream[0]["single_target_top1_accuracy"]),float(downstream[1]["single_target_top1_accuracy"])),
        "plural_l1":max(float(downstream[0]["plural_l1"]),float(downstream[1]["plural_l1"])),
        "outside_support_invariance_max_delta":max(float(downstream[0]["outside_support_invariance_max_delta"]),float(downstream[1]["outside_support_invariance_max_delta"])),
    }
    return {"operator":operator_summary,"downstream":downstream_summary}


def eligible(metrics:dict,thresholds:dict)->bool:
    op=metrics["operator"]; down=metrics["downstream"]
    return (
        op["relation_sequence_exact_accuracy"]>=thresholds["relation_sequence_exact_accuracy"]
        and op["open_schema_relation_exact_accuracy"]>=thresholds["open_schema_relation_exact_accuracy"]
        and op["role_accuracy_relational"]>=thresholds["role_accuracy_relational"]
        and op["traversal_accuracy_relational"]>=thresholds["traversal_accuracy_relational"]
        and op["modifier_exact_accuracy_relational"]>=thresholds["modifier_exact_accuracy_relational"]
        and op["control_accuracy"]>=thresholds["control_accuracy"]
        and op["termination_accuracy"]>=thresholds["termination_accuracy"]
        and op["unknown_termination_accuracy"]>=thresholds["unknown_termination_accuracy"]
        and op["pair_consistency"]>=thresholds["pair_consistency"]
        and down["row_success_accuracy"]>=thresholds["downstream_row_success_accuracy"]
        and down["family_min_success"]>=thresholds["downstream_family_min_success"]
    )


def score_tuple(metrics:dict)->tuple[float,...]:
    op=metrics["operator"]; down=metrics["downstream"]
    return (
        op["relation_sequence_exact_accuracy"],
        op["open_schema_relation_exact_accuracy"],
        op["pair_consistency"],
        down["row_success_accuracy"],
        down["family_min_success"],
    )


def save_checkpoint(path:Path,operator_model,config,step:int,lineage:dict)->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"alice.eipm.n0.qsre-production-p2-checkpoint.v1",
        "stage":"P2",
        "step":step,
        "config":config.__dict__,
        "operator":operator_model.state_dict(),
        **lineage,
    },path)
    return sha256(path)


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--plan",required=True)
    p.add_argument("--prepared-cache",required=True)
    p.add_argument("--schema-cache",required=True)
    p.add_argument("--p1-result",required=True)
    p.add_argument("--p1-root",required=True)
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("P2 requires CUDA")
    output_dir=Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("refusing to overwrite P2 evidence")

    plan_path=Path(args.plan); prepared_path=Path(args.prepared_cache); schema_cache_path=Path(args.schema_cache)
    plan=load_plan(plan_path); stage=plan["stages"]["P2"]; config=config_from_plan(plan)
    prepared=torch.load(prepared_path,map_location="cpu")
    if prepared.get("private_identity_data") is not False or prepared.get("test_present") is not False:
        raise SystemExit("P2 boundary violation")
    device=torch.device("cuda")
    full_schema,schema_payload=load_dynamic_schema_cache(schema_cache_path,device=device)
    core_count=len(schema_payload["core_train_relation_keys"])
    train_schema=subset_schema(full_schema,core_count)

    schema_encoder,executor,p1=load_p1(
        p1_result_path=Path(args.p1_result),
        p1_root=Path(args.p1_root),
        config=config,
        device=device,
    )
    operator_model=QSREProductionOperatorInducer(config).to(device)
    optimizer=torch.optim.AdamW(
        operator_model.parameters(),
        lr=float(stage["optimizer"]["learning_rate"]),
        weight_decay=float(stage["optimizer"]["weight_decay"]),
    )

    seed=int(stage["training"]["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    batch_size=int(stage["training"]["batch_size"])
    max_train_steps=int(stage["training"]["max_steps"])
    eval_every=int(stage["training"]["eval_every"])
    runtime_steps=int(stage["training"]["operator_runtime_max_steps"])
    weights=stage["loss_weights"]

    output_dir.mkdir(parents=True)
    train=prepared["train"]; dev=prepared["dev"]
    with torch.no_grad():
        encoded_train=schema_encoder(train_schema)

    batches=[]; epoch=0; step=0; history=[]; selected=None; best=None; best_score=None
    while step<max_train_steps:
        if not batches:
            batches=batch_indices(len(train["ids"]),batch_size=batch_size,seed=seed+epoch)
            epoch+=1
        idx=batches.pop(0)
        operator_model.train()
        optimizer.zero_grad(set_to_none=True)

        oracle=oracle_operator_from_targets(
            train,idx,
            schema_relation_state=encoded_train["schema_relation_state"],
            config=config,device=device,
        )
        view_ops=[]; supervised=[]; downstream_losses=[]
        for view in (0,1):
            op=infer_view(
                split=train,indices=idx,view=view,schema=train_schema,encoded=encoded_train,
                operator_model=operator_model,max_steps=runtime_steps,device=device,
            )
            view_ops.append(op)
            sup,parts=operator_supervised_loss(
                operator=op,split=train,indices=idx,
                oracle_continuous=oracle.continuous_state,
                weights=weights,device=device,
            )
            supervised.append(sup)
            down=execute_with_oracle_support(
                split=train,indices=idx,operator=op,
                schema_relation_state=encoded_train["schema_relation_state"],
                executor=executor,device=device,
            )
            target=train["target_distribution"][idx].to(device).float()
            downstream_losses.append(target_distribution_loss(down["relational_probability"],target))

        consistency=pair_loss(view_ops[0],view_ops[1])
        loss=(
            0.5*(supervised[0]+supervised[1])
            +float(weights["downstream"])*0.5*(downstream_losses[0]+downstream_losses[1])
            +float(weights["pair_consistency"])*consistency
        )
        if not torch.isfinite(loss):
            raise RuntimeError("P2 nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(operator_model.parameters(),1.0)
        optimizer.step()
        step+=1

        if step%eval_every==0 or step==max_train_steps:
            metrics=evaluate(
                split=dev,schema=full_schema,schema_encoder=schema_encoder,
                executor=executor,operator_model=operator_model,config=config,
                max_steps=runtime_steps,batch_size=batch_size,device=device,
            )
            is_eligible=eligible(metrics,stage["eligibility"])
            record={"step":step,"train_loss":float(loss.item()),"eligible":is_eligible,"dev":metrics}
            history.append(record)
            print("P2_EVAL="+json.dumps(record,sort_keys=True),flush=True)
            path=output_dir/f"step-{step:08d}"/"qsre_production_p2.pt"
            ckpt_sha=save_checkpoint(
                path,operator_model,config,step,
                {
                    "plan_sha256":sha256(plan_path),
                    "prepared_cache_sha256":sha256(prepared_path),
                    "schema_cache_sha256":sha256(schema_cache_path),
                    "p1_checkpoint_sha256":p1["sha256"],
                },
            )
            score=score_tuple(metrics)
            if best_score is None or score>best_score:
                best_score=score; best={"step":step,"checkpoint_sha256":ckpt_sha,"metrics":metrics}
            if is_eligible:
                selected={"step":step,"checkpoint_sha256":ckpt_sha,"metrics":metrics}
                break

    status="PASS_QSRE_PRODUCTION_P2_OPERATOR" if selected else "FAIL_QSRE_PRODUCTION_P2_OPERATOR"
    result={
        "schema":"alice.eipm.n0.qsre-production-p2-result.v1",
        "status":status,
        "selected":selected,
        "best_observed":best,
        "history":history,
        "plan_sha256":sha256(plan_path),
        "prepared_cache_sha256":sha256(prepared_path),
        "schema_cache_sha256":sha256(schema_cache_path),
        "p1_checkpoint_sha256":p1["sha256"],
        "train_relation_count":core_count,
        "dev_relation_count":int(full_schema.token_states.size(0)),
        "runtime_max_steps":runtime_steps,
        "schema_encoder_gradient":False,
        "executor_gradient":False,
        "operator_gradient":True,
        "binder_gradient":False,
        "private_identity_gradient":False,
        "automatic_rerun":False,
        "automatic_hotfix":False,
        "p3_authorized":bool(selected),
    }
    (output_dir/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("P2_RESULT="+json.dumps(result,sort_keys=True),flush=True)
    if not selected:
        raise SystemExit(21)


if __name__=="__main__":
    main()
