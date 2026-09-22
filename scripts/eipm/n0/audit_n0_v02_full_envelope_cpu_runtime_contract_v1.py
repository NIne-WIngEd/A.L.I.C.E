#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PASS = "PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_CONTRACT_AUDIT_V1"
FAIL = "FAIL_N0_FULL_ENVELOPE_CPU_RUNTIME_CONTRACT_AUDIT_V1"


def shape2(rows, outer, inner, name):
    if len(rows) != outer or any(len(row) != inner for row in rows):
        raise ValueError(f"{name} must be [{outer},{inner}]")


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    config=Path(args.config)
    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite runtime-contract audit")
    o=json.loads(config.read_text())
    errors=[]
    try:
        if o.get("schema")!="alice.eipm.n0.full-envelope-cpu-runtime-qualification.v1":
            raise ValueError("schema drift")
        q=o["qualification"]
        for key in ("optimizer","gradient","model_training","final_validation_opening","private_identity_data"):
            if q.get(key) is not False:
                raise ValueError(f"{key} must remain false")
        if q.get("require_cpu_only") is not True or q.get("require_inference_mode") is not True:
            raise ValueError("CPU/inference-mode requirement missing")

        long_cfg=o.get("long_context_bridge_runtime")
        if not isinstance(long_cfg,dict):
            raise ValueError("long_context_bridge_runtime missing")
        if int(long_cfg.get("native_window_tokens",0)) <= 0:
            raise ValueError("long-context native window must be positive")
        if not 0 <= int(long_cfg.get("overlap_tokens",-1)) < int(long_cfg["native_window_tokens"]):
            raise ValueError("long-context overlap geometry invalid")
        if int(long_cfg.get("bridge_heads",0)) <= 0 or int(long_cfg.get("bridge_layers",0)) <= 0:
            raise ValueError("long-context bridge depth/head geometry invalid")
        if int(long_cfg.get("minimum_segments",0)) < 2:
            raise ValueError("long-context runtime must require multiple native windows")
        if int(long_cfg.get("max_reasoning_steps",0)) <= 0:
            raise ValueError("long-context operator steps must be positive")
        if not isinstance(long_cfg.get("text"),list) or len(long_cfg["text"]) < 3:
            raise ValueError("long-context fixture must contain multiple separated passages")
        if o["qualification"].get("cross_window_bridge_required") is not True:
            raise ValueError("cross-window bridge must be required")
        if o["qualification"].get("standalone_window_stitching_insufficient") is not True:
            raise ValueError("standalone window stitching must not count as semantic completion")

        case=o["runtime_case"]
        b=int(case["batch_size"])
        if len(case["queries"])!=b:
            raise ValueError("query batch drift")
        relations=case["relations"]
        r=len(relations)
        relation_keys=[str(x["key"]) for x in relations]
        if len(set(relation_keys))!=r:
            raise ValueError("duplicate runtime relation key")
        types=[str(x) for x in case["type_descriptions"]]
        type_set=set(types)
        for row in relations:
            if not row["domain"] or not row["range"]:
                raise ValueError(f"empty relation type schema: {row['key']}")
            if not set(map(str,row["domain"])) <= type_set:
                raise ValueError(f"unknown domain type: {row['key']}")
            if not set(map(str,row["range"])) <= type_set:
                raise ValueError(f"unknown range type: {row['key']}")
        shape2(case["relation_candidate_masks"],b,r,"relation_candidate_masks")
        if any(not any(row) for row in case["relation_candidate_masks"]):
            raise ValueError("empty per-example relation candidate set")

        banks=case["factor_banks"]
        factor_masks=case["factor_candidate_masks"]
        if set(factor_masks)!=set(banks):
            raise ValueError("factor mask/bank name drift")
        for name,texts in banks.items():
            shape2(factor_masks[name],b,len(texts),f"factor mask {name}")
            if any(not any(row) for row in factor_masks[name]):
                raise ValueError(f"empty per-example factor candidate set: {name}")
        structural=case["structural_factor_opcodes"]
        if not set(structural) <= set(banks):
            raise ValueError("structural opcode mapping refers to missing factor bank")
        for name,opcodes in structural.items():
            if len(opcodes)!=len(banks[name]):
                raise ValueError(f"structural opcode cardinality drift: {name}")
        if "open_semantic_factor" in structural:
            raise ValueError("open semantic factor must not be forced into fixed executor opcode vocabulary")

        field_texts=case["field_texts"]
        if len(field_texts)!=b:
            raise ValueError("field batch drift")
        fields=len(field_texts[0])
        if fields<1 or any(len(row)!=fields for row in field_texts):
            raise ValueError("field cardinality drift")
        for key in ("field_valid_mask","field_type_index","field_confidence","field_missing","field_reliability","field_metadata"):
            shape2(case[key],b,fields,key)
        for bi in range(b):
            if not any(case["field_valid_mask"][bi]):
                raise ValueError("every runtime example needs one valid field")
            for fi,valid in enumerate(case["field_valid_mask"][bi]):
                t=int(case["field_type_index"][bi][fi])
                if valid and not 0 <= t < len(types):
                    raise ValueError("valid field has invalid runtime type")
                if not valid and t != -1:
                    raise ValueError("padded field type must be -1")
                for key in ("field_confidence","field_missing","field_reliability"):
                    value=float(case[key][bi][fi])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"{key} outside [0,1]")

        descriptor_banks=case["descriptor_banks"]
        if set(case["descriptor_indices"])!=set(descriptor_banks):
            raise ValueError("descriptor bank/index name drift")
        for name,rows in case["descriptor_indices"].items():
            shape2(rows,b,fields,f"descriptor_indices {name}")
            count=len(descriptor_banks[name])
            for bi,row in enumerate(rows):
                for fi,index in enumerate(row):
                    index=int(index)
                    valid=bool(case["field_valid_mask"][bi][fi])
                    if valid and not 0 <= index < count:
                        raise ValueError(f"valid field descriptor index invalid: {name}")
                    if not valid and index != -1:
                        raise ValueError(f"padded descriptor index must be -1: {name}")

        edges=case["edges"]
        if len(edges)!=b:
            raise ValueError("edge batch drift")
        edge_count=len(edges[0])
        if any(len(row)!=edge_count for row in edges):
            raise ValueError("edge cardinality drift")
        relation_set=set(relation_keys)
        for bi,rows in enumerate(edges):
            for edge in rows:
                if str(edge["relation"]) not in relation_set:
                    raise ValueError("edge relation absent from runtime schema")
                if bool(edge.get("valid",True)):
                    s=int(edge["source"]); t=int(edge["target"])
                    if not (0<=s<fields and 0<=t<fields):
                        raise ValueError("valid edge endpoint outside runtime field set")
                    if not case["field_valid_mask"][bi][s] or not case["field_valid_mask"][bi][t]:
                        raise ValueError("valid edge touches padded field")

        candidate_rows=case["candidate_texts"]
        if len(candidate_rows)!=b:
            raise ValueError("candidate batch drift")
        candidates=len(candidate_rows[0])
        if any(len(row)!=candidates for row in candidate_rows):
            raise ValueError("candidate cardinality drift")
        shape2(case["candidate_valid_mask"],b,candidates,"candidate_valid_mask")
        if any(not any(row) for row in case["candidate_valid_mask"]):
            raise ValueError("every example needs one valid public judgment candidate")

        if len(case["internal_view_descriptions"])!=6:
            raise ValueError("internal view descriptor count must remain six for v1")
        if len(case["additional_views"])<1:
            raise ValueError("runtime qualification must exercise dynamic additional views")
        if int(case["latent_slot_count"])<2:
            raise ValueError("runtime qualification must exercise multi-slot latent fabric")
        if int(case["max_reasoning_steps"])<2:
            raise ValueError("runtime qualification must exercise multi-step operator")
    except Exception as exc:
        errors.append(str(exc))

    result={
        "schema":"alice.eipm.n0.full-envelope-cpu-runtime-contract-audit.v1",
        "status":PASS if not errors else FAIL,
        "errors":errors,
        "optimizer":False,
        "gradient":False,
        "gpu_training":False,
        "n0_complete":False
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
