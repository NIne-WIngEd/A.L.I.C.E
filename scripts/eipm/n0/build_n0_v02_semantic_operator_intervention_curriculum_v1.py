#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


ROW_SCHEMA = "alice.eipm.n0.semantic-operator-intervention-row.v1"
MANIFEST_SCHEMA = "alice.eipm.n0.semantic-operator-intervention-manifest.v1"


def rel(
    key: str,
    family: str,
    description: str,
    source: str,
    target: str,
    phrases: list[str],
    *,
    symmetric: bool = False,
) -> dict[str, Any]:
    return {
        "key": key,
        "family": family,
        "description": description,
        "source_argument": source,
        "target_argument": target,
        "phrases": phrases,
        "symmetric": symmetric,
    }


TRAIN_RELATIONS = [
    rel("r001","evidential_support","The source provides evidence or independent backing for the target.","evidence or supporting record","claim or conclusion being supported",["provides evidence for","gives independent backing to","supports the conclusion in"]),
    rel("r002","correction","The source corrects an error or false statement in the target.","correcting record","record containing the corrected error",["corrects an error in","rectifies a mistake in","fixes the false statement in"]),
    rel("r003","supersession","The source is the controlling replacement for the target.","newer controlling version","older replaced version",["supersedes","replaces as the controlling version","makes obsolete"]),
    rel("r004","derivation","The source was derived or constructed from the target.","derived artifact","source material used for derivation",["was derived from","was synthesized from","was constructed using"]),
    rel("r005","causation","The source event causes or produces the target event.","causal event","effect event",["causes","produces","leads directly to"]),
    rel("r006","temporal_successor","The source is the immediately later state after the target.","later state","earlier state",["comes immediately after","is the next state after","succeeds temporally"]),
    rel("r007","quotation","The source reproduces exact wording from the target.","quoting artifact","source whose wording is reproduced",["quotes","reproduces words from","uses the exact wording of"]),
    rel("r008","summary","The source gives a condensed account of the target.","summary or synopsis","material summarized",["summarizes","gives a condensed account of","provides a synopsis of"]),
    rel("r009","publication","The source work was issued by the target organization.","published work","publishing organization",["was published by","was issued by","came out through"]),
    rel("r010","testing","The source was evaluated using the target method or apparatus.","tested component","test method or apparatus",["was tested with","was evaluated using","underwent testing with"]),
    rel("r011","storage","The source data or artifact is retained in the target repository.","stored item","repository or storage medium",["is stored in","is retained in","resides in"]),
    rel("r012","display","The source content is visually presented on the target interface.","displayed content","display or interface",["is displayed on","is shown on","appears on"]),
    rel("r013","prediction","The source model or statement forecasts the target outcome.","predictor","predicted outcome",["predicts","forecasts","anticipates"]),
    rel("r014","observation_time","The source event was observed during the target interval or activity.","observed event","observation interval or activity",["was observed during","was detected during","appeared during"]),
    rel("r015","labeling","The source item is assigned the target label or category.","labeled item","assigned label or category",["is labeled as","is tagged as","is classified under"]),
    rel("r016","mention","The source text mentions the target without a stronger asserted relation.","mentioning text","mentioned entity or item",["mentions","names","refers to"]),
    rel("r017","translation","The source expression is translated into the target language or representation.","source expression","target language or representation",["is translated to","is rendered into","has a translation into"]),
    rel("r018","communication","The source participant exchanges messages or signals with the target participant.","participant","other participant",["communicates with","exchanges messages with","maintains communication with"],symmetric=True),
    rel("r019","containment","The source item belongs as a constituent inside the target.","component","containing whole",["is part of","belongs inside","is a component of"]),
    rel("r020","prerequisite","The source condition must hold before the target action or state.","required condition","dependent action or state",["is a prerequisite for","must hold before","is required before"]),
    rel("r021","exemplification","The source is a concrete example illustrating the target principle.","example","principle or category illustrated",["exemplifies","is a concrete case of","illustrates"]),
    rel("r022","conflict","The source and target cannot both hold as stated.","one conflicting claim","other conflicting claim",["conflicts with","cannot be simultaneously true with","is inconsistent with"],symmetric=True),
    rel("r023","ownership","The source item belongs to or is owned by the target.","owned item","owner",["belongs to","is owned by","is the property of"]),
    rel("r024","measurement","The source measurement was produced by the target device or observer.","measurement","measuring device or observer",["was measured by","was recorded by","was captured by"]),
]

DEV_RELATIONS = [
    rel("h001","verification","The source claim or artifact was independently verified by the target.","verified claim or artifact","reviewer or verification procedure",["was verified by","was independently checked by","received verification from"]),
    rel("h002","routing","The source flow passes through the target intermediary.","traffic or flow","intermediary",["is routed through","passes through","travels by way of"]),
    rel("h003","purchase","The source item was purchased or acquired from the target seller.","purchased item","seller",["was purchased from","was bought from","was acquired from"]),
    rel("h004","schedule_order","The source activity is scheduled after the target activity.","later scheduled activity","earlier activity",["is scheduled after","is planned after","will occur later than"]),
    rel("h005","filtering","The source stream is filtered using the target mechanism.","stream or set being filtered","filter or rule",["is filtered by","is screened using","passes through the filter"]),
    rel("h006","naming","The source entity received its name in reference to the target.","named entity","name source or honoree",["is named after","takes its name from","was named in reference to"]),
    rel("h007","enablement","The source condition makes the target action possible.","enabling condition","enabled action or state",["enables","makes possible","allows"]),
    rel("h008","prevention","The source condition blocks the target event.","preventing condition","prevented event",["prevents","blocks","stops the occurrence of"]),
]


FACTOR_OPCODES = {
    "role": ["ROLE_SOURCE", "ROLE_TARGET", "ROLE_SYMMETRIC", "ROLE_NONE"],
    "traversal": ["TRAVERSAL_LOCAL", "TRAVERSAL_PATH", "TRAVERSAL_AGGREGATE"],
    "direction": ["DIRECTION_FORWARD", "DIRECTION_REVERSE", "DIRECTION_BIDIRECTIONAL"],
    "control": ["CONTROL_FALLBACK", "CONTROL_RELATIONAL", "CONTROL_DEFER"],
    "reliability_modifier": ["MOD_RELIABILITY_OFF", "MOD_RELIABILITY_ON"],
    "recency_modifier": ["MOD_RECENCY_OFF", "MOD_RECENCY_ON"],
    "temporal_constraint_modifier": ["MOD_TEMPORAL_OFF", "MOD_TEMPORAL_ON"],
    "provenance_constraint_modifier": ["MOD_PROVENANCE_OFF", "MOD_PROVENANCE_ON"],
}


FACTOR_BANKS = {
    "role": [
        "Select the endpoint that originates, provides, or performs the requested relation.",
        "Select the receiving endpoint reached or acted on by the requested relation.",
        "Preserve both endpoints equally because neither is the requested directional winner.",
        "Select no endpoint role because relational execution is not grounded.",
    ],
    "traversal": [
        "Resolve one immediate local relation step.",
        "Follow an ordered sequence of relation steps.",
        "Consider multiple matching links or endpoints together.",
    ],
    "direction": [
        "Traverse from relation source toward target.",
        "Traverse backward from relation target toward source.",
        "Preserve both relation directions without one directed winner.",
    ],
    "control": [
        "Use the ordinary non-relational path instead of relation execution.",
        "Execute relational reasoning using the supplied runtime schema.",
        "Preserve uncertainty and defer because the supplied schema is insufficient.",
    ],
    "reliability_modifier": [
        "Do not use reliability as an arbitration criterion.",
        "Prefer stronger reliability, verification, or provenance when alternatives remain.",
    ],
    "recency_modifier": [
        "Do not use recency as an arbitration criterion.",
        "Prefer the latest applicable state or relation when alternatives remain.",
    ],
    "temporal_constraint_modifier": [
        "Do not filter support by an explicit time window.",
        "Require support to satisfy the stated temporal scope.",
    ],
    "provenance_constraint_modifier": [
        "Do not filter support by provenance class.",
        "Require support to come from the authorized provenance class.",
    ],
}


TRAIN_ENTITIES = [
    "Aster","Beryl","Cinder","Dorian","Elio","Fenn","Galen","Hera","Ivo","Juno",
    "Kora","Lyra","Miro","Nola","Orin","Pia","Quill","Rhea","Soren","Tala",
]
DEV_ENTITIES = [
    "Umbra","Vega","Willow","Xanthe","Yarrow","Zephyr","Arden","Brio","Cyra","Delta",
    "Ember","Fjord","Glint","Haven","Indigo","Juniper","Kestrel","Lumen","Mica","Nova",
]


def schema_text(row: dict[str, Any]) -> str:
    symmetry = "The relation is symmetric." if row["symmetric"] else "The relation is directional."
    return (
        f"Relation meaning: {row['description']} "
        f"Source argument: {row['source_argument']}. "
        f"Target argument: {row['target_argument']}. {symmetry}"
    )


def exact_char_span(text: str, needle: str) -> dict[str, Any]:
    start = text.find(needle)
    if start < 0:
        raise ValueError(f"evidence substring not found: {needle!r}")
    return {
        "start": start,
        "end": start + len(needle),
        "text": needle,
    }


def factor_schema() -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            {
                "key": f"{name}:{i}",
                "opcode": FACTOR_OPCODES[name][i],
                "text": text,
            }
            for i, text in enumerate(values)
        ]
        for name, values in FACTOR_BANKS.items()
    }


def factor_targets(
    *,
    role: int,
    traversal: int,
    direction: int,
    control: int,
    reliability: int = 0,
    recency: int = 0,
    temporal: int = 0,
    provenance: int = 0,
) -> dict[str, int]:
    return {
        "role": role,
        "traversal": traversal,
        "direction": direction,
        "control": control,
        "reliability_modifier": reliability,
        "recency_modifier": recency,
        "temporal_constraint_modifier": temporal,
        "provenance_constraint_modifier": provenance,
    }


def candidate_bank(
    *,
    correct: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[int]]:
    if count < len(correct):
        raise ValueError("candidate count smaller than required correct relations")
    chosen = list(correct)
    available = [x for x in pool if x["key"] not in {r["key"] for r in correct}]
    rng.shuffle(available)
    chosen.extend(available[: max(0, count - len(chosen))])
    if len(chosen) < count:
        raise ValueError("relation pool too small for requested candidate cardinality")
    rng.shuffle(chosen)
    index = {row["key"]: i for i, row in enumerate(chosen)}
    return [
        {
            "key": row["key"],
            "family": row["family"],
            "text": schema_text(row),
            "symmetric": bool(row["symmetric"]),
        }
        for row in chosen
    ], [index[row["key"]] for row in correct]


def make_row(
    *,
    split: str,
    relation: dict[str, Any],
    second: dict[str, Any],
    example: int,
    candidates: int,
    rng: random.Random,
) -> dict[str, Any]:
    entities = TRAIN_ENTITIES if split == "train" else DEV_ENTITIES
    mode = example % 12
    # Modes 0/1 are an explicit source-vs-target role counterfactual pair.
    # Hold entities, relation schema bank/order and candidate cardinality fixed
    # so the requested endpoint role is the only governed semantic change.
    pair_anchor = example - 1 if mode == 1 else example
    entity_example = pair_anchor if mode in {0, 1} else example
    left = entities[(entity_example * 3) % len(entities)]
    middle = entities[(entity_example * 5 + 1) % len(entities)]
    right = entities[(entity_example * 7 + 2) % len(entities)]
    phrase_example = pair_anchor if mode in {0, 1} else example
    phrase = relation["phrases"][phrase_example % len(relation["phrases"])]
    phrase2 = second["phrases"][(example + 1) % len(second["phrases"])]

    role = 1
    traversal = 0
    direction = 0
    control = 1
    reliability = recency = temporal = provenance = 0
    sequence = [relation]
    step_relation_phrases = [phrase]
    step_direction = [0]
    step_reliability = [0]
    step_recency = [0]
    step_temporal = [0]
    step_provenance = [0]
    target_entities = [right]
    intervention = "single_target"
    plurality_alias: dict[str, Any] | None = None

    if mode == 0:
        query = f"{left} {phrase} {right}. Which participant is doing the relational work?"
        role = 0
        target_entities = [left]
        intervention = "source_target_role"
    elif mode == 1:
        query = f"{left} {phrase} {right}. Which participant is on the receiving side?"
        intervention = "source_target_role"
    elif mode == 2:
        sequence = [relation, second]
        step_relation_phrases = [phrase, phrase2]
        step_direction = [0, 0]
        step_reliability = [0, 0]
        step_recency = [0, 0]
        step_temporal = [0, 0]
        step_provenance = [0, 0]
        query = (
            f"Begin with {left}. First use the relationship where it {phrase} {middle}; "
            f"then use the relationship where {middle} {phrase2} {right}. Which endpoint is reached last?"
        )
        traversal = 1
        target_entities = [right]
        intervention = "ordered_composition"
    elif mode == 3:
        sequence = [second, relation]
        step_relation_phrases = [phrase2, phrase]
        step_direction = [1, 1]
        step_reliability = [0, 0]
        step_recency = [0, 0]
        step_temporal = [0, 0]
        step_provenance = [0, 0]
        query = (
            f"Starting at {right}, trace backward first through the relation described by '{phrase2}' "
            f"and then through the relation described by '{phrase}'. Which original source is reached?"
        )
        role = 0
        traversal = 1
        direction = 1
        target_entities = [left]
        intervention = "reverse_ordered_composition"
    elif mode == 4:
        query = (
            f"Two candidate links express the same requested meaning as '{phrase}'. "
            "Prefer the receiving endpoint backed by the stronger verification record."
        )
        traversal = 2
        reliability = 1
        intervention = "reliability_modifier"
    elif mode == 5:
        query = (
            f"Two candidate links express the same requested meaning as '{phrase}'. "
            "Prefer the receiving endpoint belonging to the latest applicable state."
        )
        traversal = 2
        recency = 1
        intervention = "recency_modifier"
    elif mode == 6:
        query = (
            f"Use the relationship expressed by '{phrase}', but only when it is valid inside the requested time window."
        )
        temporal = 1
        intervention = "temporal_constraint"
    elif mode == 7:
        query = (
            f"Use the relationship expressed by '{phrase}', but only from the authorized provenance class."
        )
        provenance = 1
        intervention = "provenance_constraint"
    elif mode == 8:
        query = (
            "The requested relation is not represented by any supplied schema description. "
            "Do not force a nearby relation; preserve uncertainty."
        )
        sequence = []
        step_relation_phrases = []
        step_direction = []
        step_reliability = []
        step_recency = []
        step_temporal = []
        step_provenance = []
        role = 3
        control = 2
        target_entities = []
        intervention = "unknown_defer"
    elif mode == 9:
        # The semantic operator never receives downstream field IDs, so
        # plurality must be expressed on an interface it actually sees.
        # Supply two independently worded runtime relation descriptions that
        # are both correct for the same requested relation.
        plurality_alias=rel(
            f"co_valid_{relation['key']}",
            relation["family"],
            (
                "An independently worded schema expresses the same relation: "
                + str(relation["description"]).rstrip(".").lower()
                + "."
            ),
            relation["source_argument"],
            relation["target_argument"],
            list(relation["phrases"]),
            symmetric=bool(relation["symmetric"]),
        )
        query = (
            f"{left} {phrase} {right}. Two supplied runtime relation descriptions "
            "both correctly express this same relationship. Preserve probability "
            "mass on both co-valid schema hypotheses instead of forcing one label."
        )
        traversal = 2
        target_entities = [right]
        intervention = "plurality"
    elif mode == 10:
        sequence = [relation, second]
        step_relation_phrases = [phrase, phrase2]
        step_direction = [0, 1]
        step_reliability = [0, 0]
        step_recency = [0, 0]
        step_temporal = [0, 0]
        step_provenance = [0, 0]
        query = (
            f"{left} {phrase} {middle}. Separately, {right} {phrase2} {middle}. "
            f"Start from {left}; traverse the first relation forward to {middle}, "
            f"then traverse the second relation backward from {middle} to {right}. "
            "Which endpoint is reached last?"
        )
        role = 0
        traversal = 1
        direction = 2
        target_entities = [right]
        intervention = "mixed_direction_composition"
    else:
        sequence = [relation, second]
        step_relation_phrases = [phrase, phrase2]
        step_direction = [0, 0]
        step_reliability = [1, 0]
        step_recency = [0, 0]
        step_temporal = [0, 1]
        step_provenance = [0, 0]
        query = (
            f"First use the relationship where {left} {phrase} {middle}, "
            "preferring the stronger verified support for that first step. "
            f"Then use the relationship where {middle} {phrase2} {right}, "
            "but require the second step to satisfy the requested time window. "
            "Which endpoint is reached last?"
        )
        role = 1
        traversal = 1
        reliability = 1
        temporal = 1
        target_entities = [right]
        intervention = "mixed_step_modifier_composition"

    pool = TRAIN_RELATIONS if split == "train" else TRAIN_RELATIONS + DEV_RELATIONS
    if plurality_alias is not None:
        required=[relation,plurality_alias]
        effective_count=max(int(candidates),len(required))
        bank,plurality_indices=candidate_bank(
            correct=required,
            pool=pool,
            count=effective_count,
            rng=rng,
        )
        # Keep one representative index only for execution/evidence geometry.
        # The optimizer-facing relation loss replaces hard CE with the explicit
        # multi-positive target on this plurality step.
        target_sequence=[int(plurality_indices[0])]
    else:
        required=sequence if sequence else [relation]
        effective_count=max(int(candidates),len(required))
        bank,indices=candidate_bank(
            correct=required,
            pool=pool,
            count=effective_count,
            rng=rng,
        )
        target_sequence=indices if sequence else []
        plurality_indices=[]

    if len(step_relation_phrases) != len(target_sequence):
        raise RuntimeError("relation evidence phrase/program length drift")
    query_relation_evidence_char_spans = [
        {
            "step": step,
            **exact_char_span(query, evidence_phrase),
        }
        for step, evidence_phrase in enumerate(step_relation_phrases)
    ]
    relation_by_key = {
        str(item["key"]): item
        for item in (TRAIN_RELATIONS + DEV_RELATIONS)
    }
    relation_schema_evidence_char_spans = []
    for step, candidate_index in enumerate(target_sequence):
        candidate = bank[candidate_index]
        source_relation = relation_by_key[str(candidate["key"])]
        span = exact_char_span(
            str(candidate["text"]),
            str(source_relation["description"]),
        )
        relation_schema_evidence_char_spans.append(
            {
                "step": step,
                "candidate_index": int(candidate_index),
                **span,
            }
        )

    factor_banks = factor_schema()
    global_factor_targets = factor_targets(
        role=role,
        traversal=traversal,
        direction=direction,
        control=control,
        reliability=reliability,
        recency=recency,
        temporal=temporal,
        provenance=provenance,
    )
    factor_schema_evidence_char_spans = {}
    for name, target_index in global_factor_targets.items():
        factor_text = str(factor_banks[name][int(target_index)]["text"])
        factor_schema_evidence_char_spans[name] = {
            "candidate_index": int(target_index),
            "start": 0,
            "end": len(factor_text),
            "text": factor_text,
        }

    step_factor_target_map = {
        "direction": step_direction,
        "reliability_modifier": step_reliability,
        "recency_modifier": step_recency,
        "temporal_constraint_modifier": step_temporal,
        "provenance_constraint_modifier": step_provenance,
    }
    step_factor_schema_evidence_char_spans = {}
    for name, targets in step_factor_target_map.items():
        bank_values = factor_banks[name]
        spans = []
        for step, target_index in enumerate(targets):
            factor_text = str(bank_values[int(target_index)]["text"])
            spans.append(
                {
                    "step": step,
                    "candidate_index": int(target_index),
                    "start": 0,
                    "end": len(factor_text),
                    "text": factor_text,
                }
            )
        step_factor_schema_evidence_char_spans[name] = spans

    if intervention == "unknown_defer":
        event_sequence_target = ["UNKNOWN"]
        applicability_target = 0
        uncertainty_target = 1.0
    else:
        event_sequence_target = ["CONTINUE"] * len(target_sequence) + ["STOP"]
        applicability_target = 1
        if intervention=="plurality":
            # SchemaConditionedSemanticOperator normalizes relation entropy by
            # log(active candidate count). Match that contract exactly for a
            # uniform target over the precommitted co-valid hypotheses.
            uncertainty_target=(
                math.log(float(len(plurality_indices)))
                / math.log(float(len(bank)))
            )
        else:
            uncertainty_target = 0.0

    counterfactual_factor_targets = {
        name: None for name in factor_banks
    }
    if intervention == "source_target_role":
        current_role = int(global_factor_targets["role"])
        if current_role in {0,1}:
            counterfactual_factor_targets["role"] = 1-current_role

    semantic_text = " ".join([query] + [x["text"] for x in bank])
    for item in bank:
        if item["key"].lower() in semantic_text.lower():
            raise RuntimeError("opaque relation key leaked into semantic text")

    template_partition = "train_templates" if split == "train" else "dev_templates"
    relation_partition = "seen_relation_family" if split == "train" else "heldout_relation_family"
    row_id = f"so_{split}_{relation['key']}_{example:04d}_{intervention}"
    counterfactual_pair_id = (
        f"cf_role_{split}_{relation['key']}_{pair_anchor:04d}"
        if mode in {0, 1}
        else None
    )
    counterfactual_variant = (
        "source" if mode == 0 else "target" if mode == 1 else None
    )
    return {
        "schema": ROW_SCHEMA,
        "id": row_id,
        "split": split,
        "lane": "schema_operator_intervention",
        "relation_partition": relation_partition,
        "relation_family": relation["family"],
        "template_partition": template_partition,
        "template_id": f"{template_partition}:{intervention}",
        "domain": "public_synthetic_relational_reasoning",
        "entities": [left, middle, right],
        "query": query,
        "relation_candidates": bank,
        "relation_sequence_target": target_sequence,
        "relation_plurality_target_indices":[
            int(x) for x in plurality_indices
        ],
        "plurality_supervision_required":bool(plurality_indices),
        "event_sequence_target": event_sequence_target,
        "applicability_target": applicability_target,
        "uncertainty_target": uncertainty_target,
        "factor_schemas": factor_banks,
        "factor_targets": global_factor_targets,
        "counterfactual_factor_targets": counterfactual_factor_targets,
        "query_relation_evidence_char_spans": query_relation_evidence_char_spans,
        "relation_schema_evidence_char_spans": relation_schema_evidence_char_spans,
        "factor_schema_evidence_char_spans": factor_schema_evidence_char_spans,
        "step_factor_targets": step_factor_target_map,
        "step_factor_schema_evidence_char_spans": (
            step_factor_schema_evidence_char_spans
        ),
        "target_entities": target_entities,
        "intervention": intervention,
        "counterfactual_pair_id": counterfactual_pair_id,
        "counterfactual_dimension": (
            "requested_endpoint_role" if counterfactual_pair_id else None
        ),
        "counterfactual_variant": counterfactual_variant,
        "runtime_relation_count": len(bank),
        "runtime_reasoning_steps": len(target_sequence),
        "runtime_operator_slots": len(event_sequence_target),
        "private_identity_data": False,
        "training_authorized": split == "train",
        "generated_text": True,
        "data_origin": "deterministic_public_semantic_operator_intervention_v1",
        "relation_keys_are_metadata_only": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--examples-per-relation", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument(
        "--candidate-counts",
        default="2,4,8,16",
        help="comma-separated operating coverage points; not a capability ceiling",
    )
    args = p.parse_args()
    if args.examples_per_relation <= 0:
        raise SystemExit("examples-per-relation must be positive")
    candidate_counts = sorted({int(x) for x in args.candidate_counts.split(",") if x.strip()})
    if not candidate_counts or min(candidate_counts) < 1:
        raise SystemExit("candidate-counts must contain integers >=1")
    if max(candidate_counts) > len(TRAIN_RELATIONS):
        raise SystemExit("candidate-count operating point exceeds available training relation definitions")

    rows: list[dict[str, Any]] = []
    for split, relations in (("train", TRAIN_RELATIONS), ("dev", DEV_RELATIONS)):
        for ri, relation in enumerate(relations):
            second_pool = TRAIN_RELATIONS if split == "train" else DEV_RELATIONS
            second = second_pool[(ri + 1) % len(second_pool)]
            for example in range(args.examples_per_relation):
                mode = example % 12
                pair_anchor = example - 1 if mode == 1 else example
                randomization_example = pair_anchor if mode in {0, 1} else example
                rng = random.Random(
                    args.seed
                    + ri * 10007
                    + randomization_example * 97
                    + (0 if split == "train" else 1_000_000)
                )
                count = candidate_counts[
                    randomization_example % len(candidate_counts)
                ]
                rows.append(
                    make_row(
                        split=split,
                        relation=relation,
                        second=second,
                        example=example,
                        candidates=count,
                        rng=rng,
                    )
                )

    output = Path(args.output)
    manifest_path = Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite curriculum artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    train = [x for x in rows if x["split"] == "train"]
    dev = [x for x in rows if x["split"] == "dev"]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "MATERIALIZED_PUBLIC_INTERVENTION_SUPPLEMENT",
        "sha256": digest,
        "rows": len(rows),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_relation_families": sorted({x["relation_family"] for x in train}),
        "dev_relation_families": sorted({x["relation_family"] for x in dev}),
        "relation_family_overlap": sorted(
            {x["relation_family"] for x in train}
            & {x["relation_family"] for x in dev}
        ),
        "candidate_count_points": sorted({x["runtime_relation_count"] for x in rows}),
        "reasoning_step_points": sorted({x["runtime_reasoning_steps"] for x in rows}),
        "operator_slot_points": sorted({x["runtime_operator_slots"] for x in rows}),
        "mixed_direction_rows": sum(
            1 for x in rows if x["intervention"] == "mixed_direction_composition"
        ),
        "mixed_step_modifier_rows": sum(
            1 for x in rows if x["intervention"] == "mixed_step_modifier_composition"
        ),
        "counterfactual_pair_count": len(
            {
                str(x["counterfactual_pair_id"])
                for x in rows
                if x.get("counterfactual_pair_id")
            }
        ),
        "counterfactual_pair_dimension": "requested_endpoint_role",
        "template_partition_overlap": sorted(
            {x["template_id"] for x in train}
            & {x["template_id"] for x in dev}
        ),
        "entity_overlap": sorted(
            {e for x in train for e in x["entities"]}
            & {e for x in dev for e in x["entities"]}
        ),
        "private_identity_data": False,
        "is_complete_training_corpus": False,
        "role": "schema/operator intervention supplement mixed with broad public semantic and governed teacher replay under the full curriculum contract",
        "row_count_is_capability_ceiling": False,
        "candidate_count_is_capability_ceiling": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
