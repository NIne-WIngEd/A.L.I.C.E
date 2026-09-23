#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROW_SCHEMA = "alice.eipm.n0.semantic-operator-intervention-row.v1"
MANIFEST_SCHEMA = "alice.eipm.n0.semantic-operator-intervention-manifest.v1"
AUDIT_SCHEMA = "alice.eipm.n0.semantic-operator-curriculum-audit.v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows_path = Path(args.rows)
    manifest_path = Path(args.manifest)
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(rows_path)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite curriculum audit")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("intervention manifest schema drift")
    if manifest.get("sha256") != sha256(rows_path):
        raise SystemExit("intervention row hash drift")
    if contract.get("schema") != "alice.eipm.n0.semantic-operator-curriculum-contract.v1":
        raise SystemExit("curriculum contract schema drift")

    errors: list[str] = []
    ids: set[str] = set()
    train_families: set[str] = set()
    dev_families: set[str] = set()
    train_templates: set[str] = set()
    dev_templates: set[str] = set()
    train_entities: set[str] = set()
    dev_entities: set[str] = set()
    counts = Counter()
    interventions = Counter()

    for row in rows:
        if row.get("schema") != ROW_SCHEMA:
            errors.append(f"{row.get('id')}: row schema drift")
            continue
        rid = str(row["id"])
        if rid in ids:
            errors.append(f"{rid}: duplicate id")
        ids.add(rid)
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data")
        split = str(row.get("split"))
        if split not in {"train", "dev"}:
            errors.append(f"{rid}: invalid split {split}")
        if split == "dev" and row.get("training_authorized") is not False:
            errors.append(f"{rid}: DEV row authorized for training")
        candidates = list(row.get("relation_candidates") or [])
        if len(candidates) < 1:
            errors.append(f"{rid}: empty relation candidate bank")
        keys = [str(x["key"]) for x in candidates]
        if len(keys) != len(set(keys)):
            errors.append(f"{rid}: duplicate relation candidate key")
        semantic_text = " ".join(
            [str(row.get("query",""))]
            + [str(x.get("text","")) for x in candidates]
            + [
                str(item.get("text",""))
                for bank in (row.get("factor_schemas") or {}).values()
                for item in bank
            ]
        ).lower()
        for key in keys:
            if key.lower() in semantic_text:
                errors.append(f"{rid}: opaque relation key leaked into semantic text")
        relation_targets = list(row.get("relation_sequence_target") or [])
        for target in relation_targets:
            if int(target) < 0 or int(target) >= len(candidates):
                errors.append(f"{rid}: relation target outside runtime candidate bank")

        query_evidence_spans = list(
            row.get("query_relation_evidence_char_spans") or []
        )
        schema_evidence_spans = list(
            row.get("relation_schema_evidence_char_spans") or []
        )
        if len(query_evidence_spans) != len(relation_targets):
            errors.append(
                f"{rid}: query evidence span count does not match relation program"
            )
        if len(schema_evidence_spans) != len(relation_targets):
            errors.append(
                f"{rid}: schema evidence span count does not match relation program"
            )
        query_text = str(row.get("query",""))
        for step, span in enumerate(query_evidence_spans):
            start = int(span.get("start",-1))
            end = int(span.get("end",-1))
            evidence_text = str(span.get("text",""))
            if int(span.get("step",-1)) != step:
                errors.append(f"{rid}: query evidence step index drift")
            if not (0 <= start < end <= len(query_text)):
                errors.append(f"{rid}: query evidence span outside query")
            elif query_text[start:end] != evidence_text or not evidence_text:
                errors.append(f"{rid}: query evidence substring mismatch")
        for step, span in enumerate(schema_evidence_spans):
            start = int(span.get("start",-1))
            end = int(span.get("end",-1))
            evidence_text = str(span.get("text",""))
            candidate_index = int(span.get("candidate_index",-1))
            if int(span.get("step",-1)) != step:
                errors.append(f"{rid}: schema evidence step index drift")
            if step < len(relation_targets) and candidate_index != int(
                relation_targets[step]
            ):
                errors.append(f"{rid}: schema evidence candidate target drift")
            if not (0 <= candidate_index < len(candidates)):
                errors.append(f"{rid}: schema evidence candidate outside bank")
            else:
                candidate_text = str(candidates[candidate_index].get("text",""))
                if not (0 <= start < end <= len(candidate_text)):
                    errors.append(f"{rid}: schema evidence span outside candidate text")
                elif candidate_text[start:end] != evidence_text or not evidence_text:
                    errors.append(f"{rid}: schema evidence substring mismatch")
        events = list(row.get("event_sequence_target") or [])
        expected_slots = int(row.get("runtime_operator_slots", -1))
        if len(events) != expected_slots:
            errors.append(
                f"{rid}: event sequence length {len(events)} != runtime_operator_slots {expected_slots}"
            )
        allowed_events = {"CONTINUE", "STOP", "UNKNOWN"}
        if any(str(event) not in allowed_events for event in events):
            errors.append(f"{rid}: invalid event target")
        if row.get("intervention") == "unknown_defer":
            if events != ["UNKNOWN"]:
                errors.append(f"{rid}: unknown/defer must terminate with UNKNOWN")
            if relation_targets:
                errors.append(f"{rid}: unknown/defer cannot execute a relation")
            if int(row.get("applicability_target", -1)) != 0:
                errors.append(f"{rid}: unknown/defer applicability target must be 0")
        else:
            expected_events = ["CONTINUE"] * len(relation_targets) + ["STOP"]
            if events != expected_events:
                errors.append(
                    f"{rid}: relation program must execute each relation with CONTINUE then STOP"
                )
            if int(row.get("applicability_target", -1)) != 1:
                errors.append(f"{rid}: relational applicability target must be 1")
        uncertainty_target = row.get("uncertainty_target")
        if not isinstance(uncertainty_target,(int,float)) or not (
            0.0 <= float(uncertainty_target) <= 1.0
        ):
            errors.append(f"{rid}: explicit uncertainty_target must be inside [0,1]")
        elif row.get("intervention")=="unknown_defer":
            if float(uncertainty_target) != 1.0:
                errors.append(f"{rid}: unknown/defer uncertainty target must be 1")
        elif float(uncertainty_target) != 0.0:
            errors.append(
                f"{rid}: deterministic intervention uncertainty target must be 0"
            )

        factor_schemas = row.get("factor_schemas") or {}
        factor_targets = row.get("factor_targets") or {}
        if set(factor_schemas) != set(factor_targets):
            errors.append(f"{rid}: factor schema/target bank mismatch")
        for name, bank in factor_schemas.items():
            target = int(factor_targets[name])
            if target < 0 or target >= len(bank):
                errors.append(f"{rid}: factor target outside {name} bank")

        factor_counter = row.get("counterfactual_factor_targets") or {}
        if set(factor_counter) != set(factor_schemas):
            errors.append(f"{rid}: counterfactual factor bank mismatch")
        else:
            for name,value in factor_counter.items():
                if value is None:
                    continue
                index=int(value)
                if not 0 <= index < len(factor_schemas[name]):
                    errors.append(
                        f"{rid}: counterfactual factor target outside {name} bank"
                    )
                elif index == int(factor_targets[name]):
                    errors.append(
                        f"{rid}: counterfactual factor target equals correct target for {name}"
                    )

        factor_evidence = row.get("factor_schema_evidence_char_spans") or {}
        if set(factor_evidence) != set(factor_schemas):
            errors.append(f"{rid}: factor schema evidence bank mismatch")
        for name, bank in factor_schemas.items():
            span = factor_evidence.get(name) or {}
            candidate_index = int(span.get("candidate_index",-1))
            target = int(factor_targets[name])
            if candidate_index != target:
                errors.append(f"{rid}: factor schema evidence target drift for {name}")
                continue
            if not (0 <= candidate_index < len(bank)):
                continue
            factor_text = str(bank[candidate_index].get("text",""))
            start = int(span.get("start",-1))
            end = int(span.get("end",-1))
            evidence_text = str(span.get("text",""))
            if not (0 <= start < end <= len(factor_text)):
                errors.append(f"{rid}: factor evidence span outside {name} text")
            elif factor_text[start:end] != evidence_text or not evidence_text:
                errors.append(f"{rid}: factor evidence substring mismatch for {name}")

        step_targets = row.get("step_factor_targets") or {}
        required_step_factor_names = {
            "direction",
            "reliability_modifier",
            "recency_modifier",
            "temporal_constraint_modifier",
            "provenance_constraint_modifier",
        }
        if set(step_targets) != required_step_factor_names:
            errors.append(f"{rid}: step factor target names drift")
        step_count = int(row.get("runtime_reasoning_steps", -1))
        if step_count < 0:
            errors.append(f"{rid}: invalid runtime_reasoning_steps")
            step_count = 0
        step_factor_evidence = (
            row.get("step_factor_schema_evidence_char_spans") or {}
        )
        if set(step_factor_evidence) != required_step_factor_names:
            errors.append(f"{rid}: step factor schema evidence bank mismatch")
        for name in required_step_factor_names:
            values = list(step_targets.get(name) or [])
            evidence_spans = list(step_factor_evidence.get(name) or [])
            if len(values) != step_count:
                errors.append(
                    f"{rid}: step factor {name} length {len(values)} != reasoning steps {step_count}"
                )
                continue
            if len(evidence_spans) != step_count:
                errors.append(
                    f"{rid}: step factor evidence {name} length {len(evidence_spans)} != reasoning steps {step_count}"
                )
            bank = factor_schemas.get(name)
            if bank is None:
                errors.append(f"{rid}: step factor {name} has no semantic bank")
                continue
            for step, target in enumerate(values):
                target = int(target)
                if target < 0 or target >= len(bank):
                    errors.append(
                        f"{rid}: step factor target outside {name} bank"
                    )
                    continue
                if step >= len(evidence_spans):
                    continue
                span = evidence_spans[step]
                candidate_index = int(span.get("candidate_index",-1))
                if int(span.get("step",-1)) != step:
                    errors.append(
                        f"{rid}: step factor evidence step index drift for {name}"
                    )
                if candidate_index != target:
                    errors.append(
                        f"{rid}: step factor evidence target drift for {name}"
                    )
                    continue
                factor_text = str(bank[candidate_index].get("text",""))
                start = int(span.get("start",-1))
                end = int(span.get("end",-1))
                evidence_text = str(span.get("text",""))
                if not (0 <= start < end <= len(factor_text)):
                    errors.append(
                        f"{rid}: step factor evidence span outside {name} text"
                    )
                elif factor_text[start:end] != evidence_text or not evidence_text:
                    errors.append(
                        f"{rid}: step factor evidence substring mismatch for {name}"
                    )
        counts[int(row.get("runtime_relation_count", -1))] += 1
        interventions[str(row.get("intervention"))] += 1
        family = str(row.get("relation_family"))
        template = str(row.get("template_id"))
        entities = {str(x) for x in row.get("entities") or []}
        if split == "train":
            train_families.add(family); train_templates.add(template); train_entities |= entities
        else:
            dev_families.add(family); dev_templates.add(template); dev_entities |= entities

    family_overlap = sorted(train_families & dev_families)
    template_overlap = sorted(train_templates & dev_templates)
    entity_overlap = sorted(train_entities & dev_entities)
    if family_overlap:
        errors.append("train/dev relation-family leakage: " + repr(family_overlap))
    if template_overlap:
        errors.append("train/dev template leakage: " + repr(template_overlap))
    if entity_overlap:
        errors.append("train/dev entity leakage: " + repr(entity_overlap))

    required_interventions = {
        "source_target_role",
        "ordered_composition",
        "reverse_ordered_composition",
        "reliability_modifier",
        "recency_modifier",
        "temporal_constraint",
        "provenance_constraint",
        "unknown_defer",
        "plurality",
        "mixed_direction_composition",
        "mixed_step_modifier_composition",
    }
    missing_interventions = sorted(required_interventions - set(interventions))
    if missing_interventions:
        errors.append("missing intervention families: " + repr(missing_interventions))

    expected_points = set(
        int(x)
        for x in contract["runtime_axis_coverage"]["relation_candidate_counts"]["training_coverage_points"]
    )
    observed_train_points = {
        int(row["runtime_relation_count"])
        for row in rows
        if row["split"] == "train"
    }
    missing_points = sorted(expected_points - observed_train_points)
    if missing_points:
        errors.append("missing candidate-count operating points: " + repr(missing_points))

    result = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS_SEMANTIC_OPERATOR_CURRICULUM_AUDIT" if not errors else "FAIL_SEMANTIC_OPERATOR_CURRICULUM_AUDIT",
        "rows_sha256": sha256(rows_path),
        "manifest_sha256": sha256(manifest_path),
        "rows": len(rows),
        "errors": errors,
        "train_relation_families": sorted(train_families),
        "dev_relation_families": sorted(dev_families),
        "relation_family_overlap": family_overlap,
        "template_overlap": template_overlap,
        "entity_overlap": entity_overlap,
        "candidate_count_histogram": {str(k): v for k, v in sorted(counts.items())},
        "terminal_event_contract_explicit": True,
        "intervention_histogram": dict(sorted(interventions.items())),
        "step_conditioned_direction_present": interventions["mixed_direction_composition"] > 0,
        "step_conditioned_modifier_present": interventions["mixed_step_modifier_composition"] > 0,
        "query_relation_evidence_spans_required": True,
        "relation_schema_evidence_spans_required": True,
        "factor_schema_evidence_spans_required": True,
        "step_factor_schema_evidence_spans_required": True,
        "private_identity_data": False,
        "gradient": False,
        "optimizer": False,
        "training_authorized_by_audit": False,
        "complete_training_corpus": False,
        "natural_public_replay_still_required": True,
        "governed_teacher_replay_still_required": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
