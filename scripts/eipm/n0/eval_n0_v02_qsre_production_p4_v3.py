from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alice_personality.n0.qsre_production_binder_v2 import (
    QSREProductionBinderV2,
)
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import (
    config_from_plan,
    focus_metrics,
    load_plan,
    operator_metrics,
    paired_view_consistency,
    summarize_downstream,
    support_metrics,
)
from train_n0_v02_qsre_production_p3_v3 import (
    bind,
    execute,
    infer_operator,
    load_parents,
    load_selected,
)


@torch.no_grad()
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
    p.add_argument("--p3-result",required=True)
    p.add_argument("--p3-root",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("P4 governed full-stack evaluation requires CUDA")

    plan_path=Path(args.plan); prepared_path=Path(args.prepared_cache); schema_cache_path=Path(args.schema_cache)
    plan=load_plan(plan_path); stage=plan["stages"]["P4"]; config=config_from_plan(plan)
    prepared=torch.load(prepared_path,map_location="cpu")
    if prepared.get("private_identity_data") is not False or prepared.get("test_present") is not False:
        raise SystemExit("P4 boundary violation")

    p1_path,_=load_selected(Path(args.p1_result),Path(args.p1_root),"qsre_production_p1.pt","PASS_QSRE_PRODUCTION_P1_EXECUTOR")
    p2_path,_=load_selected(Path(args.p2_result),Path(args.p2_root),"qsre_production_p2.pt","PASS_QSRE_PRODUCTION_P2_OPERATOR")
    p3_path,_=load_selected(Path(args.p3_result),Path(args.p3_root),"qsre_production_p3.pt","PASS_QSRE_PRODUCTION_P3_BINDER")

    device=torch.device("cuda")
    schema_encoder,executor,operator_model=load_parents(
        p1_path=p1_path,p2_path=p2_path,
        factor_schema_cache_path=Path(args.factor_schema_cache),
        config=config,device=device,
    )
    binder=QSREProductionBinderV2(config)
    binder.load_state_dict(torch.load(p3_path,map_location="cpu")["binder"],strict=True)
    for parameter in binder.parameters():
        parameter.requires_grad=False
    binder.to(device).eval()

    schema,_=load_dynamic_schema_cache(schema_cache_path,device=device)
    encoded=schema_encoder(schema)
    split=prepared["dev"]
    max_steps=int(stage["operator_runtime_max_steps"])
    batch_size=8

    all_ops=[[],[]]; all_support=[[],[]]; all_focus=[[],[]]; all_compat=[[],[]]; all_probability=[[],[]]
    for start in range(0,len(split["ids"]),batch_size):
        idx=torch.arange(start,min(start+batch_size,len(split["ids"])))
        for view in (0,1):
            op=infer_operator(
                split=split,indices=idx,view=view,schema=schema,encoded=encoded,
                operator=operator_model,max_steps=max_steps,device=device,
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
            all_ops[view].append(op)
            all_support[view].append(bound["edge_support_weight"].cpu())
            all_focus[view].append(bound["focus_field_weight"].cpu())
            all_compat[view].append(bound["type_compatible"].cpu())
            all_probability[view].append(out["relational_probability"].cpu())

    from alice_personality.n0.qsre_production_core import QSREProductionOperatorState
    def concat_ops(values):
        return QSREProductionOperatorState(**{
            name:torch.cat([getattr(value,name).cpu() for value in values],dim=0)
            for name in QSREProductionOperatorState.__dataclass_fields__
        })
    ops=[concat_ops(all_ops[0]),concat_ops(all_ops[1])]
    probabilities=[torch.cat(all_probability[0]),torch.cat(all_probability[1])]
    supports=[torch.cat(all_support[0]),torch.cat(all_support[1])]
    focuses=[torch.cat(all_focus[0]),torch.cat(all_focus[1])]
    compatible=[torch.cat(all_compat[0]),torch.cat(all_compat[1])]
    indices=torch.arange(len(split["ids"]))

    view_results=[]
    for view in (0,1):
        om=operator_metrics(operator=ops[view],split=split,indices=indices)
        om={k:v for k,v in om.items() if not k.endswith("_tensor")}
        sm=support_metrics(
            predicted=supports[view],
            oracle=split["oracle_edge_support"].float(),
            type_compatible=compatible[view],
        )
        fm=focus_metrics(
            predicted=focuses[view],
            oracle=split["focus_field_weight"].float(),
            traversal_target=split["traversal_target"].long(),
        )
        dm=summarize_downstream(
            probability=probabilities[view],
            target=split["target_distribution"].float(),
            control_target=split["control_target"].long(),
            families=split["families"],
            open_schema=split["open_schema"].bool(),
            causal_groups=split["causal_groups"],
        )
        dm.pop("row_success_tensor",None)
        nonrel=split["control_target"].ne(1)
        false_assertion=(
            probabilities[view][nonrel].sum(dim=-1).gt(0.01).float().mean().item()
            if bool(nonrel.any()) else 0.0
        )
        dm["nonrelational_false_assertion_rate"]=float(false_assertion)
        view_results.append({"operator":om,"support":sm,"focus":fm,"downstream":dm})

    family_keys=set(view_results[0]["downstream"]["family_success"])|set(view_results[1]["downstream"]["family_success"])
    family_worst={
        family:min(
            float(view_results[0]["downstream"]["family_success"].get(family,1.0)),
            float(view_results[1]["downstream"]["family_success"].get(family,1.0)),
        )
        for family in sorted(family_keys)
    }
    aggregate={
        "downstream_row_success_accuracy":min(v["downstream"]["row_success_accuracy"] for v in view_results),
        "family_min_success":min(family_worst.values()) if family_worst else 1.0,
        "family_success":family_worst,
        "open_schema_row_success_accuracy":min(v["downstream"]["open_schema_success"] for v in view_results),
        "outside_support_invariance_max_delta":max(v["downstream"]["outside_support_invariance_max_delta"] for v in view_results),
        "nonrelational_false_assertion_rate":max(v["downstream"]["nonrelational_false_assertion_rate"] for v in view_results),
        "operator_pair_consistency":paired_view_consistency(first=ops[0],second=ops[1]),
        "support_edge_f1":min(v["support"]["edge_f1"] for v in view_results),
        "path_focus_top1_accuracy":min(v["focus"]["path_focus_top1_accuracy"] for v in view_results),
        "path_focus_exact_set_accuracy":min(v["focus"]["path_focus_exact_set_accuracy"] for v in view_results),
        "direction_accuracy_relational":min(v["operator"]["direction_accuracy_relational"] for v in view_results),
        "relation_sequence_exact_accuracy":min(v["operator"]["relation_sequence_exact_accuracy"] for v in view_results),
        "termination_accuracy":min(v["operator"]["termination_accuracy"] for v in view_results),
        "unknown_fail_closed_accuracy":min(v["operator"]["unknown_termination_accuracy"] for v in view_results),
    }
    threshold=stage["eligibility"]
    passed=(
        aggregate["downstream_row_success_accuracy"]>=threshold["downstream_row_success_accuracy"]
        and aggregate["family_min_success"]>=threshold["family_min_success"]
        and aggregate["open_schema_row_success_accuracy"]>=threshold["open_schema_row_success_accuracy"]
        and aggregate["outside_support_invariance_max_delta"]<=threshold["outside_support_invariance_max_delta"]
        and aggregate["nonrelational_false_assertion_rate"]<=threshold["nonrelational_false_assertion_rate_max"]
        and aggregate["unknown_fail_closed_accuracy"]>=threshold["unknown_fail_closed_accuracy"]
        and aggregate["path_focus_top1_accuracy"]>=threshold["path_focus_top1_accuracy"]
        and aggregate["direction_accuracy_relational"]>=threshold["direction_accuracy_relational"]
    )
    result={
        "schema":"alice.eipm.n0.qsre-production-p4-result.v3",
        "architecture":"operator-v3-plus-binder-v2",
        "status":"PASS_QSRE_PRODUCTION_P4_END_TO_END" if passed else "FAIL_QSRE_PRODUCTION_P4_END_TO_END",
        "aggregate":aggregate,
        "view_results":view_results,
        "plan_sha256":sha256(plan_path),
        "prepared_cache_sha256":sha256(prepared_path),
        "schema_cache_sha256":sha256(schema_cache_path),
        "p1_checkpoint_sha256":sha256(p1_path),
        "p2_checkpoint_sha256":sha256(p2_path),
        "p3_checkpoint_sha256":sha256(p3_path),
        "factor_schema_cache_sha256":sha256(Path(args.factor_schema_cache)),
        "gradient":False,
        "test_open":False,
        "private_identity_data":False,
        "automatic_joint_finetune":False,
        "p5_authorized":passed,
    }
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    if not passed:
        raise SystemExit(23)


if __name__=="__main__":
    main()
