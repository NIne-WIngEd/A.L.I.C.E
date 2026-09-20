from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from alice_personality.n0.qsre_production_core import QSREProductionCore
from qsre_production_runtime import load_dynamic_schema_cache, sha256
from qsre_production_training_utils import config_from_plan, load_plan


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--plan",required=True)
    p.add_argument("--curriculum",required=True)
    p.add_argument("--schema-cache",required=True)
    p.add_argument("--prepared-cache",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    plan_path=Path(args.plan)
    curriculum_path=Path(args.curriculum)
    schema_cache_path=Path(args.schema_cache)
    prepared_path=Path(args.prepared_cache)
    output_path=Path(args.output)

    plan=load_plan(plan_path)
    if sha256(curriculum_path)!=plan["curriculum_sha256"]:
        raise SystemExit("P0 curriculum hash drift")

    prepared=torch.load(prepared_path,map_location="cpu")
    if prepared.get("schema")!="alice.eipm.n0.qsre-production-public-cache.v1":
        raise SystemExit("P0 prepared-cache schema drift")
    if prepared.get("curriculum_sha256")!=plan["curriculum_sha256"]:
        raise SystemExit("P0 prepared curriculum lineage drift")
    if prepared.get("private_identity_data") is not False:
        raise SystemExit("private identity data entered P0")
    if prepared.get("test_present") is not False:
        raise SystemExit("TEST entered P0")

    device=torch.device("cpu")
    schema,schema_payload=load_dynamic_schema_cache(schema_cache_path,device=device)
    config=config_from_plan(plan)
    core=QSREProductionCore(config).eval()

    split=prepared["dev"]
    n=min(4,len(split["ids"]))
    idx=torch.arange(n)
    encoded=core.encode_schema(schema)

    # Use first paraphrase view for the real-artifact full-path smoke.
    query_hidden=split["query_hidden_states"][idx,0].float()
    query_mask=split["query_token_mask"][idx,0].bool()
    operator_output=core.operator(
        query_hidden_states=query_hidden,
        query_token_mask=query_mask,
        schema=schema,
        schema_token_state=encoded["schema_token_state"],
        schema_relation_state=encoded["schema_relation_state"],
        max_steps=int(split["relation_target"].size(1)),
    )
    operator=operator_output["operator"]

    binder=core.binder(
        query_hidden_states=query_hidden,
        query_token_mask=query_mask,
        field_token_states=split["field_token_states"][idx].float(),
        field_token_mask=split["field_token_mask"][idx].bool(),
        field_type_id=split["field_type_id"][idx].long(),
        edge_index=split["edge_index"][idx].long(),
        edge_relation_index=split["edge_relation_index"][idx].long(),
        edge_valid_mask=split["edge_valid_mask"][idx].bool(),
        edge_reliability=split["edge_reliability"][idx].float(),
        edge_recency=split["edge_recency"][idx].float(),
        schema=schema,
        schema_relation_state=encoded["schema_relation_state"],
        operator=operator,
    )

    execution=core.executor(
        field_state=split["field_state"][idx].float(),
        field_metadata=split["field_metadata"][idx].float(),
        field_valid_mask=split["field_valid_mask"][idx].bool(),
        edge_index=split["edge_index"][idx].long(),
        edge_relation_index=split["edge_relation_index"][idx].long(),
        edge_valid_mask=split["edge_valid_mask"][idx].bool(),
        edge_support_weight=binder["edge_support_weight"].float(),
        edge_reliability=split["edge_reliability"][idx].float(),
        edge_recency=split["edge_recency"][idx].float(),
        edge_temporal_match=split["edge_temporal_match"][idx].float(),
        edge_provenance_match=split["edge_provenance_match"][idx].float(),
        schema_relation_state=encoded["schema_relation_state"],
        operator=operator,
        focus_field_weight=split["focus_field_weight"][idx].float(),
    )

    for name,tensor in (
        ("relation_distribution",operator.relation_distribution),
        ("continuous_state",operator.continuous_state),
        ("edge_support_weight",binder["edge_support_weight"]),
        ("relational_probability",execution["relational_probability"]),
        ("relational_summary",execution["relational_summary"]),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"P0 nonfinite tensor: {name}")

    # Dynamic-cardinality and dynamic-step real-tensor checks use the exact
    # same checkpoint/source module. Core train relations are first six.
    core_count=len(schema_payload["core_train_relation_keys"])
    schema6=type(schema)(
        token_states=schema.token_states[:core_count],
        token_mask=schema.token_mask[:core_count],
        domain_type_mask=schema.domain_type_mask[:core_count],
        range_type_mask=schema.range_type_mask[:core_count],
        symmetric=schema.symmetric[:core_count],
    )
    encoded6=core.encode_schema(schema6)
    dynamic_shapes={}
    for steps in (1,2,3,5):
        out=core.operator(
            query_hidden_states=query_hidden[:1],
            query_token_mask=query_mask[:1],
            schema=schema6,
            schema_token_state=encoded6["schema_token_state"],
            schema_relation_state=encoded6["schema_relation_state"],
            max_steps=steps,
        )
        dynamic_shapes[str(steps)]=list(out["operator"].relation_distribution.shape)

    report=core.parameter_report()
    for section in ("schema_encoder","operator","binder","executor"):
        if report[section].get("relation_count_dependent_parameters") not in (0,None):
            raise RuntimeError(f"P0 relation-count parameter axis reintroduced in {section}")
    if report["operator"]["hop_count_dependent_parameters"]!=0:
        raise RuntimeError("P0 hop-count parameter axis reintroduced")

    receipt={
        "schema":"alice.eipm.n0.qsre-production-p0-runtime-result.v1",
        "status":"PASS_QSRE_PRODUCTION_P0_REAL_ARTIFACT_RUNTIME",
        "plan_sha256":sha256(plan_path),
        "curriculum_sha256":sha256(curriculum_path),
        "schema_cache_sha256":sha256(schema_cache_path),
        "prepared_cache_sha256":sha256(prepared_path),
        "rows_smoked":n,
        "full_relation_count":int(schema.token_states.size(0)),
        "core_train_relation_count":core_count,
        "dynamic_step_shapes":dynamic_shapes,
        "query_view_count":int(split["query_hidden_states"].size(1)),
        "query_hidden_depth":int(split["query_hidden_states"].size(2)),
        "parameter_report":report,
        "optimizer":False,
        "gradient":False,
        "gpu":False,
        "test_open":False,
        "private_identity_data":False,
        "p1_authorized_by_this_receipt":True,
        "automatic_rerun":False,
        "automatic_hotfix":False,
    }
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
