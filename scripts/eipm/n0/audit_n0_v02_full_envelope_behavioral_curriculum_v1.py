#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PASS="PASS_N0_FULL_ENVELOPE_BEHAVIORAL_CURRICULUM_AUDIT_V1"
FAIL="FAIL_N0_FULL_ENVELOPE_BEHAVIORAL_CURRICULUM_AUDIT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str,Any]]:
    rows=[]
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            row=json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL line {line_no}: {exc}") from exc
        if not isinstance(row,dict):
            raise ValueError(f"row {line_no} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("behavioral curriculum is empty")
    return rows


def normalized_surface(text: str, entities: list[str]) -> str:
    value=str(text).lower()
    for entity in sorted((str(x) for x in entities),key=len,reverse=True):
        value=value.replace(entity.lower(),"<entity>")
    return " ".join(value.split())


def field_surface_signature(row: dict[str,Any]) -> str:
    entities=list(row.get("entities") or [])
    normalized=" || ".join(
        normalized_surface(str(item.get("text","")),entities)
        for item in list(row.get("fields") or [])
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def candidate_surface_signature(row: dict[str,Any]) -> str:
    entities=list(row.get("entities") or [])
    normalized=sorted(
        normalized_surface(str(text),entities)
        for text in list(row.get("candidate_answers") or [])
    )
    return hashlib.sha256(" || ".join(normalized).encode("utf-8")).hexdigest()


def semantic_text(row: dict[str,Any]) -> str:
    pieces=[str(row["query"])]
    pieces.extend(str(x["text"]) for x in row["type_schema"])
    pieces.extend(str(x["text"]) for x in row["relation_candidates"])
    for bank in row["factor_schemas"].values():
        pieces.extend(str(x["text"]) for x in bank)
    pieces.extend(str(x["text"]) for x in row["fields"])
    pieces.extend(str(x) for x in row["candidate_answers"])
    for field in row["fields"]:
        pieces.extend(str(v) for v in field["descriptors"].values())
    return " ".join(pieces).lower()


def audit_row(row: dict[str,Any], contract: dict[str,Any]) -> list[str]:
    errors=[]
    rid=str(row.get("id",""))
    prefix=f"{rid}: " if rid else "<missing-id>: "
    if row.get("schema") != contract["row_schema"]:
        errors.append(prefix+"row schema drift")
    if row.get("lane") != "full_envelope_behavioral_fabric":
        errors.append(prefix+"lane drift")
    if row.get("private_identity_data") is not False:
        errors.append(prefix+"private identity data forbidden")
    split=str(row.get("split",""))
    if split not in {"train","dev"}:
        errors.append(prefix+"split must be train/dev")
    if row.get("training_authorized") is not (split=="train"):
        errors.append(prefix+"training authorization drift")
    if row.get("model_selection_authorized") is not (split=="dev"):
        errors.append(prefix+"model-selection authorization drift")
    if row.get("final_validation_only") is not False:
        errors.append(prefix+"behavioral supplement may not be FINAL")
    if not row.get("relation_keys_are_metadata_only",False):
        errors.append(prefix+"relation keys must be metadata only")
    if not row.get("type_keys_are_metadata_only",False):
        errors.append(prefix+"type keys must be metadata only")

    type_schema=list(row.get("type_schema") or [])
    type_keys=[str(x.get("key","")) for x in type_schema]
    if not type_keys or len(type_keys)!=len(set(type_keys)):
        errors.append(prefix+"runtime type keys missing or duplicated")
    relation_candidates=list(row.get("relation_candidates") or [])
    relation_keys=[str(x.get("key","")) for x in relation_candidates]
    if not relation_keys or len(relation_keys)!=len(set(relation_keys)):
        errors.append(prefix+"runtime relation keys missing or duplicated")
    relation_index={key:i for i,key in enumerate(relation_keys)}

    text=semantic_text(row)
    for key in type_keys + relation_keys:
        if key and key.lower() in text:
            errors.append(prefix+f"opaque metadata key leaked into semantic text: {key}")

    factor_schemas=dict(row.get("factor_schemas") or {})
    factor_targets=dict(row.get("factor_target_keys") or {})
    factor_counter=dict(row.get("counterfactual_factor_keys") or {})
    step_factor_targets=list(row.get("step_factor_target_keys") or [])
    if set(factor_targets)!=set(factor_schemas):
        errors.append(prefix+"factor target/schema bank mismatch")
    if set(factor_counter)!=set(factor_schemas):
        errors.append(prefix+"factor counterfactual/schema bank mismatch")
    factor_key_sets={}
    for name,bank_raw in factor_schemas.items():
        bank=list(bank_raw)
        keys=[str(x.get("key","")) for x in bank]
        if not keys or len(keys)!=len(set(keys)):
            errors.append(prefix+f"factor keys missing/duplicated in {name}")
            continue
        factor_key_sets[str(name)]=set(keys)
        target=str(factor_targets.get(name,""))
        if target not in factor_key_sets[str(name)]:
            errors.append(prefix+f"factor target absent from bank {name}: {target}")
        counter=factor_counter.get(name)
        if counter is not None:
            counter=str(counter)
            if counter not in factor_key_sets[str(name)]:
                errors.append(prefix+f"factor counterfactual absent from bank {name}")
            if counter==target:
                errors.append(prefix+f"factor counterfactual equals target in {name}")

    relation_target=[int(x) for x in row.get("relation_sequence_target") or []]
    relation_counter=[int(x) for x in row.get("counterfactual_relation_sequence_target") or []]
    if len(relation_target)!=len(relation_counter):
        errors.append(prefix+"relation target/counterfactual length drift")
    for i,target in enumerate(relation_target):
        if not 0 <= target < len(relation_keys):
            errors.append(prefix+f"relation target outside candidate bank at step {i}")
        counter=relation_counter[i] if i < len(relation_counter) else -1
        if counter != -1 and not 0 <= counter < len(relation_keys):
            errors.append(prefix+f"relation counterfactual outside bank at step {i}")
        if counter == target:
            errors.append(prefix+f"relation counterfactual equals target at step {i}")

    event=list(row.get("event_sequence_target") or [])
    if relation_target:
        if len(event)!=len(relation_target)+1:
            errors.append(prefix+"relational event sequence must end after relation steps")
        if event and event[:-1] != ["CONTINUE"]*len(relation_target):
            errors.append(prefix+"relation steps must align to CONTINUE events")
        if event and event[-1]!="STOP":
            errors.append(prefix+"relational program must end in STOP")
    else:
        if event != ["UNKNOWN"]:
            errors.append(prefix+"zero-step row must explicitly UNKNOWN")
        if float(row.get("applicability_target",-1)) != 0.0:
            errors.append(prefix+"UNKNOWN row applicability must be zero")

    if len(step_factor_targets)!=len(relation_target):
        errors.append(prefix+"step-factor/program length drift")
    for step,targets in enumerate(step_factor_targets):
        if set(targets)!=set(factor_schemas):
            errors.append(prefix+f"step factor bank mismatch at {step}")
            continue
        for name,target in targets.items():
            if str(target) not in factor_key_sets.get(str(name),set()):
                errors.append(prefix+f"step factor target absent from bank {name} at {step}")

    fields=list(row.get("fields") or [])
    if not fields:
        errors.append(prefix+"runtime field set empty")
    for i,item in enumerate(fields):
        if str(item.get("type_key","")) not in set(type_keys):
            errors.append(prefix+f"field {i} type absent from runtime type schema")
        for scalar in ("confidence","missing","reliability"):
            value=float(item.get(scalar,-1))
            if not 0.0 <= value <= 1.0:
                errors.append(prefix+f"field {i} {scalar} outside [0,1]")
        metadata=item.get("metadata")
        if not isinstance(metadata,list) or len(metadata)!=3:
            errors.append(prefix+f"field {i} metadata must have width 3")
        descriptors=item.get("descriptors")
        if not isinstance(descriptors,dict) or not {"provenance","temporal_scope"} <= set(descriptors):
            errors.append(prefix+f"field {i} descriptor coverage drift")

    edges=list(row.get("edges") or [])
    relation_key_set=set(relation_keys)
    for i,item in enumerate(edges):
        s=int(item.get("source_field",-1)); t=int(item.get("target_field",-1))
        if not (0<=s<len(fields) and 0<=t<len(fields)):
            errors.append(prefix+f"edge {i} endpoint outside field set")
        if str(item.get("relation_key","")) not in relation_key_set:
            errors.append(prefix+f"edge {i} relation absent from runtime relation candidates")
        for scalar in ("reliability","recency","temporal_match","provenance_match"):
            value=float(item.get(scalar,-1))
            if not 0.0 <= value <= 1.0:
                errors.append(prefix+f"edge {i} {scalar} outside [0,1]")
        metadata=item.get("metadata")
        if not isinstance(metadata,list) or len(metadata)!=4:
            errors.append(prefix+f"edge {i} metadata must have width 4")

    support=[int(x) for x in row.get("support_edge_indices") or []]
    if len(support)!=len(set(support)):
        errors.append(prefix+"duplicate support edge target")
    for i in support:
        if not 0<=i<len(edges):
            errors.append(prefix+"support edge target outside edge bank")
        elif not bool(edges[i].get("support_target")):
            errors.append(prefix+f"support target flag missing on edge {i}")
    flagged={i for i,e in enumerate(edges) if bool(e.get("support_target"))}
    if set(support)!=flagged:
        errors.append(prefix+"support edge list/flags disagree")

    endpoint=dict(row.get("endpoint_target") or {})
    endpoint_active=bool(endpoint.get("active",False))
    source=int(endpoint.get("source_field",-1)); target=int(endpoint.get("target_field",-1))
    if endpoint_active:
        if not (0<=source<len(fields) and 0<=target<len(fields)):
            errors.append(prefix+"active endpoint target outside field set")
    elif source!=-1 or target!=-1:
        errors.append(prefix+"inactive endpoint target must use -1 sentinels")

    scenario=str(row.get("scenario_family",""))
    if scenario=="unknown_defer" and support:
        errors.append(prefix+"unknown/defer row must have zero positive support edges")

    decisive=set(int(x) for x in row.get("decisive_field_indices") or [])
    irrelevant=set(int(x) for x in row.get("irrelevant_field_indices") or [])
    if decisive & irrelevant:
        errors.append(prefix+"field cannot be both decisive and irrelevant")
    if any(not 0<=i<len(fields) for i in decisive|irrelevant):
        errors.append(prefix+"decisive/irrelevant field outside field set")
    if bool(row.get("decisive_view_active")) != bool(decisive):
        errors.append(prefix+"decisive intervention activity flag drift")
    if bool(row.get("irrelevant_view_active")) != bool(irrelevant):
        errors.append(prefix+"irrelevant intervention activity flag drift")
    if decisive:
        if not any(
            i in support and (
                int(edges[i]["source_field"]) in decisive
                or int(edges[i]["target_field"]) in decisive
            )
            for i in support
        ):
            errors.append(prefix+"decisive removal does not touch a governed positive support edge")
    for i in support:
        if int(edges[i]["source_field"]) in irrelevant or int(edges[i]["target_field"]) in irrelevant:
            errors.append(prefix+"irrelevant removal would delete positive target support")

    answers=list(row.get("candidate_answers") or [])
    public_target=int(row.get("public_target_index",-1))
    if len(answers)<2 or not 0<=public_target<len(answers):
        errors.append(prefix+"public judgment target geometry drift")
    if len(set(map(str,answers)))!=len(answers):
        errors.append(prefix+"duplicate public answer candidate")

    recoverable=list(row.get("recoverable_view_names") or [])
    valid_views={
        "raw_semantic","structured","evidence_specialist",
        "graph_source","graph_target","relational_executor"
    }
    if not recoverable or not set(recoverable)<=valid_views:
        errors.append(prefix+"recoverable view labels invalid")

    perm=[int(x) for x in row.get("field_permutation") or []]
    if sorted(perm)!=list(range(len(fields))):
        errors.append(prefix+"field permutation is not a bijection")
    if len(fields)>1 and perm==list(range(len(fields))):
        errors.append(prefix+"field permutation variant is identity")

    if int(row.get("runtime_relation_count",-1)) != len(relation_candidates):
        errors.append(prefix+"runtime relation count drift")
    if int(row.get("runtime_field_count",-1)) != len(fields):
        errors.append(prefix+"runtime field count drift")
    if int(row.get("runtime_edge_count",-1)) != len(edges):
        errors.append(prefix+"runtime edge count drift")
    if int(row.get("runtime_reasoning_steps",-1)) != len(relation_target):
        errors.append(prefix+"runtime reasoning-step count drift")
    if int(row.get("runtime_candidate_answer_count",-1)) != len(answers):
        errors.append(prefix+"runtime answer count drift")

    return errors


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--rows",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--contract",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    rows_path=Path(args.rows)
    manifest_path=Path(args.manifest)
    contract_path=Path(args.contract)
    output_path=Path(args.output)
    if output_path.exists():
        raise SystemExit("refusing to overwrite behavioral audit")
    rows=load_jsonl(rows_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    contract=json.loads(contract_path.read_text(encoding="utf-8"))
    errors=[]

    if contract.get("schema")!="alice.eipm.n0.full-envelope-behavioral-curriculum-contract.v1":
        errors.append("contract schema drift")
    if manifest.get("schema")!="alice.eipm.n0.full-envelope-behavioral-manifest.v1":
        errors.append("manifest schema drift")
    if manifest.get("sha256")!=sha256(rows_path):
        errors.append("row SHA mismatch")
    if int(manifest.get("rows",-1))!=len(rows):
        errors.append("manifest row count drift")

    ids=[str(x.get("id","")) for x in rows]
    if not all(ids) or len(ids)!=len(set(ids)):
        errors.append("row IDs missing or duplicated")

    for row in rows:
        errors.extend(audit_row(row,contract))

    train=[x for x in rows if x.get("split")=="train"]
    dev=[x for x in rows if x.get("split")=="dev"]
    train_entities={str(e) for x in train for e in x.get("entities",[])}
    dev_entities={str(e) for x in dev for e in x.get("entities",[])}
    entity_overlap=sorted(train_entities & dev_entities)
    if entity_overlap:
        errors.append("TRAIN/DEV entity overlap: "+repr(entity_overlap))

    train_templates={str(x.get("template_signature_sha256","")) for x in train}
    dev_templates={str(x.get("template_signature_sha256","")) for x in dev}
    template_overlap=sorted(train_templates & dev_templates)
    if "" in train_templates or "" in dev_templates:
        errors.append("template signature missing")
    if template_overlap:
        errors.append("TRAIN/DEV normalized query-template overlap")

    train_field_surfaces={field_surface_signature(x) for x in train}
    dev_field_surfaces={field_surface_signature(x) for x in dev}
    field_surface_overlap=sorted(train_field_surfaces & dev_field_surfaces)
    if field_surface_overlap:
        errors.append(
            "TRAIN/DEV normalized field-surface overlap; DEV may reuse TRAIN "
            "scenario wording outside the query"
        )

    train_candidate_surfaces={candidate_surface_signature(x) for x in train}
    dev_candidate_surfaces={candidate_surface_signature(x) for x in dev}
    candidate_surface_overlap=sorted(
        train_candidate_surfaces & dev_candidate_surfaces
    )
    if candidate_surface_overlap:
        errors.append(
            "TRAIN/DEV normalized candidate-surface overlap; DEV may reuse "
            "TRAIN answer templates"
        )

    train_causal={str(x.get("causal_group","")) for x in train}
    dev_causal={str(x.get("causal_group","")) for x in dev}
    if train_causal & dev_causal:
        errors.append("TRAIN/DEV causal-group overlap")

    required=set(contract["scenario_families"])
    train_scenarios={str(x.get("scenario_family","")) for x in train}
    dev_scenarios={str(x.get("scenario_family","")) for x in dev}
    missing_train=sorted(required-train_scenarios)
    missing_dev=sorted(required-dev_scenarios)
    if missing_train:
        errors.append("TRAIN missing scenario families: "+repr(missing_train))
    if missing_dev:
        errors.append("DEV missing scenario families: "+repr(missing_dev))

    relation_points=sorted({int(x["runtime_relation_count"]) for x in rows})
    field_points=sorted({int(x["runtime_field_count"]) for x in rows})
    edge_points=sorted({int(x["runtime_edge_count"]) for x in rows})
    step_points=sorted({int(x["runtime_reasoning_steps"]) for x in rows})
    answer_points=sorted({int(x["runtime_candidate_answer_count"]) for x in rows})
    if len(relation_points)<4:
        errors.append("relation-cardinality coverage too narrow")
    if len(field_points)<4:
        errors.append("field-cardinality coverage too narrow")
    if len(edge_points)<4:
        errors.append("edge-cardinality coverage too narrow")
    if not {0,1,2} <= set(step_points):
        errors.append("reasoning-step coverage must include 0,1,2")
    if len(answer_points)<3:
        errors.append("candidate-answer cardinality coverage too narrow")

    answer_hist=Counter(int(x["public_target_index"]) for x in rows)
    max_answer_share=max(answer_hist.values())/len(rows)
    if max_answer_share>0.45:
        errors.append(
            f"public answer position shortcut too strong: max share={max_answer_share:.4f}"
        )

    relation_target_positions=Counter()
    for row in rows:
        for target in row["relation_sequence_target"]:
            relation_target_positions[int(target)]+=1
    if relation_target_positions:
        max_relation_share=max(relation_target_positions.values())/sum(relation_target_positions.values())
        if max_relation_share>0.55:
            errors.append(
                f"relation target position shortcut too strong: max share={max_relation_share:.4f}"
            )

    result={
        "schema":"alice.eipm.n0.full-envelope-behavioral-curriculum-audit.v1",
        "status":PASS if not errors else FAIL,
        "errors":errors,
        "rows":len(rows),
        "train_rows":len(train),
        "dev_rows":len(dev),
        "scenario_histogram":dict(sorted(Counter(str(x["scenario_family"]) for x in rows).items())),
        "relation_count_points":relation_points,
        "field_count_points":field_points,
        "edge_count_points":edge_points,
        "reasoning_step_points":step_points,
        "answer_count_points":answer_points,
        "public_target_position_histogram":dict(sorted(answer_hist.items())),
        "relation_target_position_histogram":dict(sorted(relation_target_positions.items())),
        "train_dev_entity_overlap":entity_overlap,
        "train_dev_template_overlap":template_overlap,
        "train_dev_field_surface_overlap":field_surface_overlap,
        "train_dev_candidate_surface_overlap":candidate_surface_overlap,
        "private_identity_data":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "n0_complete":False,
    }
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
