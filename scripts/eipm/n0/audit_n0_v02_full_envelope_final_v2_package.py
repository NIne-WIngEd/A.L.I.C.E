#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PASS="PASS_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_AUDIT_V1"
FAIL="FAIL_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_AUDIT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str, entities: list[str]) -> str:
    value=str(text).lower()
    for entity in sorted((str(x) for x in entities),key=len,reverse=True):
        value=value.replace(entity.lower(),"<entity>")
    return " ".join(value.split())


def field_signature(row: dict[str,Any]) -> str:
    entities=list(row.get("entities") or [])
    payload=" || ".join(
        normalized(str(item.get("text","")),entities)
        for item in row.get("fields") or []
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_signature(row: dict[str,Any]) -> str:
    entities=list(row.get("entities") or [])
    payload=" || ".join(sorted(
        normalized(str(x),entities)
        for x in row.get("candidate_answers") or []
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def type_schema_signature(row: dict[str,Any]) -> str:
    payload=" || ".join(sorted(str(x.get("text","")).lower() for x in row.get("type_schema") or []))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def full_text_characters(row: dict[str,Any]) -> int:
    pieces=[str(row.get("query",""))]
    pieces.extend(str(x.get("text","")) for x in row.get("fields") or [])
    pieces.extend(str(x) for x in row.get("candidate_answers") or [])
    pieces.extend(str(x.get("text","")) for x in row.get("relation_candidates") or [])
    for bank in (row.get("factor_schemas") or {}).values():
        pieces.extend(str(x.get("text","")) for x in bank)
    pieces.extend(str(x.get("text","")) for x in row.get("type_schema") or [])
    return sum(len(x) for x in pieces)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--package-config",required=True)
    p.add_argument("--evaluator-contract",required=True)
    p.add_argument("--final-contract",required=True)
    p.add_argument("--behavioral-train-dev-rows",required=True)
    p.add_argument("--semantic-train-dev-rows",required=True)
    p.add_argument("--synthetic-final-rows",required=True)
    p.add_argument("--semantic-final-rows",required=True)
    p.add_argument("--package-manifest",required=True)
    p.add_argument("--fewrel-final-rows",required=True)
    p.add_argument("--fewrel-final-bank",required=True)
    p.add_argument("--fewrel-manifest",required=True)
    p.add_argument("--fewrel-audit",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    paths={name:Path(getattr(args,name)) for name in (
        "package_config","evaluator_contract","final_contract","behavioral_train_dev_rows",
        "semantic_train_dev_rows","synthetic_final_rows","semantic_final_rows","package_manifest",
        "fewrel_final_rows","fewrel_final_bank","fewrel_manifest","fewrel_audit","output"
    )}
    if paths["output"].exists():
        raise SystemExit("refusing to overwrite final-v2 audit")

    package=json.loads(paths["package_config"].read_text(encoding="utf-8"))
    evaluator=json.loads(paths["evaluator_contract"].read_text(encoding="utf-8"))
    final_contract=json.loads(paths["final_contract"].read_text(encoding="utf-8"))
    manifest=json.loads(paths["package_manifest"].read_text(encoding="utf-8"))
    fewrel_manifest=json.loads(paths["fewrel_manifest"].read_text(encoding="utf-8"))
    fewrel_audit=json.loads(paths["fewrel_audit"].read_text(encoding="utf-8"))
    behavioral=read_jsonl(paths["behavioral_train_dev_rows"])
    semantic=read_jsonl(paths["semantic_train_dev_rows"])
    final_rows=read_jsonl(paths["synthetic_final_rows"])
    semantic_final=read_jsonl(paths["semantic_final_rows"])
    natural_final=read_jsonl(paths["fewrel_final_rows"])
    errors=[]

    if package.get("schema")!="alice.eipm.n0.full-envelope-final-package.v1":
        errors.append("package config schema drift")
    if evaluator.get("registered_system")!="N0FullEnvelopeTrainableSystemV1":
        errors.append("final evaluator not bound to registered full-envelope trainable system")
    if evaluator.get("legacy_final_self_validation_v1_forbidden") is not True:
        errors.append("legacy final-v1 evaluator not explicitly forbidden")
    if evaluator.get("legacy_final_self_validation_v3_forbidden") is not True:
        errors.append("legacy final-v3 evaluator not explicitly forbidden")
    if final_contract.get("schema")!="alice.eipm.n0.full-envelope-final-validation-contract.v2":
        errors.append("final contract schema drift")
    if manifest.get("status")!="MATERIALIZED_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_NO_RESULTS":
        errors.append("package manifest status drift")
    if fewrel_audit.get("status")!="PASS_FEWREL_NATURAL_RELATION_AUDIT_V3":
        errors.append("FewRel natural-final audit did not pass")
    if any([
        package.get("training_authorized") is not False,
        package.get("model_selection_authorized") is not False,
        package.get("final_opening_authorized") is not False,
        package.get("results_observed") is not False,
        evaluator.get("results_observed") is not False,
        evaluator.get("final_opening_authorized") is not False,
        manifest.get("results_observed") is not False,
    ]):
        errors.append("final-v2 package leaked training/model-selection/opening/result authority")

    expected_hashes={
        "synthetic_final_rows_sha256":paths["synthetic_final_rows"],
        "semantic_final_rows_sha256":paths["semantic_final_rows"],
        "fewrel_final_rows_sha256":paths["fewrel_final_rows"],
        "fewrel_final_bank_sha256":paths["fewrel_final_bank"],
        "fewrel_manifest_sha256":paths["fewrel_manifest"],
        "final_contract_sha256":paths["final_contract"],
        "behavioral_train_dev_reference_sha256":paths["behavioral_train_dev_rows"],
    }
    for key,path in expected_hashes.items():
        if manifest.get(key)!=sha256(path):
            errors.append(f"package manifest hash drift: {key}")
    if fewrel_manifest.get("final_rows_sha256")!=sha256(paths["fewrel_final_rows"]):
        errors.append("FewRel final rows hash drift")
    if fewrel_manifest.get("final_bank_sha256")!=sha256(paths["fewrel_final_bank"]):
        errors.append("FewRel final bank hash drift")

    if not final_rows:
        errors.append("synthetic FINAL component empty")
    if not natural_final:
        errors.append("natural FINAL component empty")
    for row in final_rows:
        rid=str(row.get("id","<missing>"))
        if row.get("split")!="final":
            errors.append(f"{rid}: synthetic row not FINAL")
        if row.get("training_authorized") is not False:
            errors.append(f"{rid}: synthetic FINAL training-authorized")
        if row.get("model_selection_authorized") is not False:
            errors.append(f"{rid}: synthetic FINAL model-selection-authorized")
        if row.get("final_validation_only") is not True:
            errors.append(f"{rid}: synthetic FINAL lacks final-only marker")
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data in public N0 final")
    for row in natural_final:
        rid=str(row.get("id","<missing>"))
        if row.get("split")!="final" or row.get("final_validation_only") is not True:
            errors.append(f"{rid}: natural row not sealed FINAL")
        if row.get("training_authorized") is not False or row.get("model_selection_authorized") is not False:
            errors.append(f"{rid}: natural FINAL leaked training/model-selection authority")

    train_dev_entities={str(e) for row in behavioral for e in row.get("entities") or []}
    final_entities={str(e) for row in final_rows for e in row.get("entities") or []}
    entity_overlap=sorted(train_dev_entities & final_entities)
    if entity_overlap:
        errors.append(f"FINAL entity overlap with TRAIN/DEV: {entity_overlap}")

    train_templates={str(row.get("template_signature_sha256")) for row in behavioral}
    final_templates={str(row.get("template_signature_sha256")) for row in final_rows}
    template_overlap=sorted(train_templates & final_templates)
    if template_overlap:
        errors.append("FINAL query-template overlap with TRAIN/DEV")

    field_overlap=sorted({field_signature(x) for x in behavioral} & {field_signature(x) for x in final_rows})
    candidate_overlap=sorted({candidate_signature(x) for x in behavioral} & {candidate_signature(x) for x in final_rows})
    if field_overlap:
        errors.append("FINAL normalized field-surface overlap with TRAIN/DEV")
    if candidate_overlap:
        errors.append("FINAL normalized candidate-surface overlap with TRAIN/DEV")

    train_domains={str(x.get("domain_family","")) for x in behavioral}
    final_domains={str(x.get("domain_family","")) for x in final_rows}
    domain_overlap=sorted((train_domains & final_domains)-{""})
    if domain_overlap:
        errors.append(f"FINAL domain-family overlap with TRAIN/DEV: {domain_overlap}")

    type_overlap=sorted({type_schema_signature(x) for x in behavioral} & {type_schema_signature(x) for x in final_rows})
    if type_overlap:
        errors.append("FINAL type-schema surface overlaps TRAIN/DEV")

    def max_axis(rows: list[dict[str,Any]], key: str) -> int:
        return max((int(x.get(key,0)) for x in rows),default=0)
    axis_checks={
        "relation":max_axis(final_rows,"runtime_relation_count") > max_axis(behavioral,"runtime_relation_count"),
        "field":max_axis(final_rows,"runtime_field_count") > max_axis(behavioral,"runtime_field_count"),
        "candidate":max_axis(final_rows,"runtime_candidate_answer_count") > max_axis(behavioral,"runtime_candidate_answer_count"),
        "reasoning_steps":max_axis(final_rows,"runtime_reasoning_steps") > max_axis(behavioral,"runtime_reasoning_steps"),
        "context_characters":max((full_text_characters(x) for x in final_rows),default=0) > max((full_text_characters(x) for x in behavioral),default=0),
    }
    train_factor_max=max((max(len(bank) for bank in x.get("factor_schemas",{}).values()) for x in behavioral),default=0)
    final_factor_max=max((max(len(bank) for bank in x.get("factor_schemas",{}).values()) for x in final_rows),default=0)
    axis_checks["factor"]=final_factor_max > train_factor_max
    for name,passed in axis_checks.items():
        if not passed:
            errors.append(f"FINAL lacks {name} extrapolation beyond TRAIN/DEV operating points")

    families={str(x.get("scenario_family")) for x in final_rows}
    for required in ("long_causal_chain","heldout_recency_provenance_combo"):
        if required not in families:
            errors.append(f"missing required final-only scenario: {required}")

    behavioral_combo=any(
        (x.get("factor_target_keys") or {}).get("recency")=="MOD_RECENCY_ON"
        and (x.get("factor_target_keys") or {}).get("provenance")=="MOD_PROVENANCE_ON"
        for x in behavioral
    )
    semantic_combo=any(
        int((x.get("factor_targets") or {}).get("recency_modifier",0))==1
        and int((x.get("factor_targets") or {}).get("provenance_constraint_modifier",0))==1
        for x in semantic
    )
    final_combo=any(
        (x.get("factor_target_keys") or {}).get("recency")=="MOD_RECENCY_ON"
        and (x.get("factor_target_keys") or {}).get("provenance")=="MOD_PROVENANCE_ON"
        for x in final_rows
    )
    if behavioral_combo or semantic_combo:
        errors.append("heldout FINAL recency+provenance factor combination appears in TRAIN/DEV")
    if not final_combo:
        errors.append("heldout FINAL recency+provenance factor combination absent")


    if not semantic_final:
        errors.append("synthetic semantic-operator FINAL component empty")
    for row in semantic_final:
        rid=str(row.get("id","<missing>"))
        if row.get("split")!="final":
            errors.append(f"{rid}: semantic FINAL row not final")
        if row.get("training_authorized") is not False:
            errors.append(f"{rid}: semantic FINAL training-authorized")
        if row.get("model_selection_authorized") is not False:
            errors.append(f"{rid}: semantic FINAL model-selection-authorized")
        if row.get("final_validation_only") is not True:
            errors.append(f"{rid}: semantic FINAL lacks final-only marker")
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data in semantic FINAL")
        if row.get("relation_partition")!="sealed_final_relation_family":
            errors.append(f"{rid}: semantic FINAL relation partition drift")
        if row.get("template_partition")!="sealed_final_templates":
            errors.append(f"{rid}: semantic FINAL template partition drift")

    semantic_train_dev_families={
        str(x.get("relation_family","")) for x in semantic
    }
    semantic_final_families={
        str(x.get("relation_family","")) for x in semantic_final
    }
    semantic_relation_family_overlap=sorted(
        (semantic_train_dev_families & semantic_final_families)-{""}
    )
    if semantic_relation_family_overlap:
        errors.append(
            "semantic FINAL relation-family overlap with TRAIN/DEV: "
            +repr(semantic_relation_family_overlap)
        )

    semantic_train_dev_entities={
        str(e) for x in semantic for e in x.get("entities") or []
    }
    semantic_final_entities={
        str(e) for x in semantic_final for e in x.get("entities") or []
    }
    semantic_entity_overlap=sorted(
        semantic_train_dev_entities & semantic_final_entities
    )
    if semantic_entity_overlap:
        errors.append(
            "semantic FINAL entity overlap with TRAIN/DEV: "
            +repr(semantic_entity_overlap)
        )

    train_dev_relation_text={
        str(item.get("text","")).strip().lower()
        for row in semantic
        for item in row.get("relation_candidates") or []
    }
    final_relation_text={
        str(item.get("text","")).strip().lower()
        for row in semantic_final
        for item in row.get("relation_candidates") or []
    }
    semantic_relation_surface_overlap=sorted(
        train_dev_relation_text & final_relation_text
    )
    if semantic_relation_surface_overlap:
        errors.append("semantic FINAL relation descriptions overlap TRAIN/DEV")

    semantic_interventions={
        str(x.get("intervention","")) for x in semantic_final
    }
    required_semantic_interventions={
        "source_target_role","ordered_composition",
        "reverse_ordered_composition","reliability_modifier",
        "recency_modifier","temporal_constraint","provenance_constraint",
        "unknown_defer","plurality","mixed_direction_composition",
        "mixed_step_modifier_composition",
    }
    missing_semantic_interventions=sorted(
        required_semantic_interventions-semantic_interventions
    )
    if missing_semantic_interventions:
        errors.append(
            "semantic FINAL missing interventions: "
            +repr(missing_semantic_interventions)
        )
    plurality_rows=[
        row for row in semantic_final
        if bool(row.get("plurality_supervision_required"))
    ]
    if not plurality_rows or any(
        len(row.get("relation_plurality_target_indices") or [])<2
        for row in plurality_rows
    ):
        errors.append("semantic FINAL lacks true multi-hypothesis plurality")
    pair_groups={}
    for row in semantic_final:
        pair=row.get("counterfactual_pair_id")
        if pair:
            pair_groups.setdefault(str(pair),set()).add(
                str(row.get("counterfactual_variant"))
            )
    if not pair_groups or any(
        variants!={"source","target"} for variants in pair_groups.values()
    ):
        errors.append("semantic FINAL source/target causal pairs incomplete")

    natural_relation_overlap={
        "train_final":fewrel_audit.get("relation_overlap",{}).get("train_final"),
        "dev_final":fewrel_audit.get("relation_overlap",{}).get("dev_final"),
    }
    if natural_relation_overlap["train_final"] or natural_relation_overlap["dev_final"]:
        errors.append("natural FINAL relation-family overlap detected")
    if fewrel_audit.get("final_relation_descriptions_absent_from_train_dev_bank") is not True:
        errors.append("natural FINAL relation descriptions leaked into TRAIN/DEV bank")

    result={
        "schema":"alice.eipm.n0.full-envelope-final-v2-package-audit.v1",
        "status":PASS if not errors else FAIL,
        "errors":errors,
        "synthetic_rows":len(final_rows),
        "semantic_final_rows":len(semantic_final),
        "semantic_final_relation_family_overlap":semantic_relation_family_overlap,
        "semantic_final_entity_overlap":semantic_entity_overlap,
        "semantic_final_relation_surface_overlap":semantic_relation_surface_overlap,
        "semantic_final_interventions":sorted(semantic_interventions),
        "semantic_final_plurality_rows":len(plurality_rows),
        "semantic_final_source_target_pair_count":len(pair_groups),
        "natural_rows":len(natural_final),
        "entity_overlap":entity_overlap,
        "template_overlap":template_overlap,
        "field_surface_overlap":field_overlap,
        "candidate_surface_overlap":candidate_overlap,
        "domain_overlap":domain_overlap,
        "type_schema_overlap":type_overlap,
        "axis_extrapolation":axis_checks,
        "train_factor_cardinality_max":train_factor_max,
        "final_factor_cardinality_max":final_factor_max,
        "heldout_recency_provenance_combo_present":final_combo,
        "heldout_recency_provenance_combo_seen_in_behavioral_train_dev":behavioral_combo,
        "heldout_recency_provenance_combo_seen_in_semantic_train_dev":semantic_combo,
        "natural_relation_overlap":natural_relation_overlap,
        "results_observed":False,
        "training_authorized":False,
        "model_selection_authorized":False,
        "final_opening_authorized":False,
        "private_identity_data":False,
        "n0_complete":False,
    }
    paths["output"].parent.mkdir(parents=True,exist_ok=True)
    paths["output"].write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
