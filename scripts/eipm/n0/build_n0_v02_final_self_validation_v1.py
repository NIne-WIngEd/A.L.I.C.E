#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1_1 as fabric
from alice_personality.n0.evidence_graph import relation_type_id


SCHEMA = "alice.eipm.n0.final-self-validation-row.v1"
ROWS_PER_FAMILY = 8

GENERAL_SUBJECTS = [
    "Vanguard Acorn", "Vanguard Brook", "Vanguard Cobalt", "Vanguard Dune",
    "Vanguard Estuary", "Vanguard Fern", "Vanguard Granite", "Vanguard Hollow",
    "Vanguard Ivory", "Vanguard Jet", "Vanguard Kelp", "Vanguard Linen",
    "Vanguard Meadow", "Vanguard Nickel", "Vanguard Orchid", "Vanguard Prairie",
]
GENERAL_ATTRIBUTES = [
    ("handover setting", "queued", "direct"),
    ("inspection epoch", "23", "57"),
    ("service posture", "conservative", "adaptive"),
    ("refresh window", "14 seconds", "43 seconds"),
    ("replica layout", "single", "mirrored"),
    ("telemetry mode", "compact", "extended"),
    ("routing track", "reserve", "primary"),
    ("archive interval", "16 minutes", "39 minutes"),
    ("worker allotment", "5", "11"),
    ("sampling policy", "coarse", "fine"),
]

ENTITIES = [
    "Aquila", "Banyan", "Cirrus", "Dahlia", "Equinox", "Flint", "Gannet", "Halcyon",
    "Isobar", "Jade", "Kepler", "Lotus", "Mistral", "Nacre", "Osprey", "Palisade",
    "Quasar", "Rook", "Sable", "Topaz", "Umber", "Vireo", "Warden", "Xylem",
]

CORE_RELATIONS = (
    "SUPPORTS",
    "CORRECTS",
    "SUPERSEDES",
    "DERIVED_FROM",
    "CAUSES",
    "TEMPORAL_SUCCESSOR",
)
OPEN_RELATIONS = (
    "CONFLICTS_WITH",
    "EXEMPLIFIES",
    "PREREQUISITE_FOR",
    "PART_OF",
)
FINAL_ONLY_RELATIONS = ("ENABLES", "PREVENTS")

PHRASES = {
    "SUPPORTS": "gives independent backing to",
    "CORRECTS": "rectifies a mistake recorded in",
    "SUPERSEDES": "is the controlling replacement for",
    "DERIVED_FROM": "was synthesized using the source material in",
    "CAUSES": "is the initiating event that produces",
    "TEMPORAL_SUCCESSOR": "occupies the immediately later state after",
    "CONFLICTS_WITH": "cannot be simultaneously true with",
    "EXEMPLIFIES": "is a concrete case illustrating",
    "PREREQUISITE_FOR": "must hold before",
    "PART_OF": "belongs as a constituent inside",
    "ENABLES": "makes it possible to carry out",
    "PREVENTS": "blocks the occurrence of",
}

TYPE_PAIR = {
    "SUPPORTS": ("EVIDENCE", "CLAIM"),
    "CORRECTS": ("CORRECTION", "RECORD"),
    "SUPERSEDES": ("VERSION", "VERSION"),
    "DERIVED_FROM": ("ARTIFACT", "SOURCE"),
    "CAUSES": ("EVENT", "EVENT"),
    "TEMPORAL_SUCCESSOR": ("VERSION", "VERSION"),
    "CONFLICTS_WITH": ("CLAIM", "CLAIM"),
    "EXEMPLIFIES": ("EXAMPLE", "PRINCIPLE"),
    "PREREQUISITE_FOR": ("CONDITION", "ACTION"),
    "PART_OF": ("COMPONENT", "WHOLE"),
    "ENABLES": ("CONDITION", "ACTION"),
    "PREVENTS": ("CONDITION", "EVENT"),
}

ROLE = {"SOURCE": 0, "TARGET": 1, "SYMMETRIC": 2, "NONE": 3}
TRAVERSAL = {"LOCAL_SELECT": 0, "PATH": 1, "AGGREGATE": 2}
DIRECTION = {"FORWARD": 0, "REVERSE": 1, "BIDIRECTIONAL": 2}
CONTROL = {"FALLBACK": 0, "RELATIONAL": 1, "DEFER": 2}

RELATIONAL_FAMILIES = (
    "single_source",
    "single_target",
    "path_source",
    "path_target",
    "path_order",
    "three_hop",
    "plural_targets",
    "reliability_arbitration",
    "recency_arbitration",
    "temporal_constraint",
    "provenance_constraint",
    "combined_constraints",
    "unknown_fail_closed",
    "existing_open_schema",
    "final_unseen_enables",
    "final_unseen_prevents",
    "symmetric_conflict",
    "outside_support",
    "nonrel_fallback",
    "ambiguous_plurality",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alternate_query(text: str) -> str:
    replacements = [
        ("Which record", "Identify the record"),
        ("Return", "Keep"),
        ("Starting from", "Beginning with"),
        ("Start at", "Begin with"),
        ("follow", "trace"),
        ("Follow", "Trace"),
        ("Use", "Apply"),
        ("use", "apply"),
        ("current", "currently applicable"),
    ]
    out = text
    for old, new in replacements:
        if old in out:
            out = out.replace(old, new, 1)
    if out == text:
        out = "Using the same evidence, " + text[0].lower() + text[1:]
    return out


def field(
    *,
    row_id: str,
    index: int,
    entity: str,
    type_name: str,
    detail: str,
    reliability: float = 0.8,
    recency: float = 0.5,
) -> dict[str, Any]:
    text = f"{entity} {type_name.lower()} record. {detail}"
    return {
        "id": f"{row_id}_f{index}",
        "entity": entity,
        "type": type_name,
        "text": text,
        "metadata": {
            "reliability": float(reliability),
            "normalized_time": float(recency),
            "temporal_match": 1.0,
            "provenance_match": 1.0,
        },
        # Existing public N0 parent-stack fields. These are generic public
        # structural ids; QSRE carries the runtime-open semantic type schema.
        "field_type_id": 1,
        "provenance_id": 1,
        "relation_role_id": 1,
        "temporal_scope_id": 1,
        "confidence": float(reliability),
        "missing": False,
    }


def edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    *,
    reliability: float = 0.8,
    recency: float = 0.5,
    temporal_match: float = 1.0,
    provenance_match: float = 1.0,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "metadata": {
            "reliability": float(reliability),
            "normalized_time": float(recency),
            "temporal_match": float(temporal_match),
            "provenance_match": float(provenance_match),
        },
    }


def parent_relations(edges: list[dict[str, Any]], positions: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for item in edges:
        try:
            type_id = relation_type_id(str(item["relation"]).lower())
        except Exception:
            # A final-only open-schema relation is deliberately unavailable to
            # the historical parent graph. QSRE must add that capability.
            continue
        out.append(
            {
                "source": positions[item["source"]],
                "target": positions[item["target"]],
                "relation": str(item["relation"]).lower(),
                "relation_type_id": int(type_id),
                "confidence": float(item["metadata"]["reliability"]),
            }
        )
    return out


def make_row(
    *,
    family: str,
    index: int,
    query: str,
    fields: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    relation_sequence: list[str],
    role: str,
    traversal: str,
    modifier_target: list[float],
    direction: str = "FORWARD",
    support_edge_ids: list[str],
    target_distribution: dict[str, float],
    focus_field_id: str | None,
    applicability: float = 0.95,
    control: str = "RELATIONAL",
    termination: str = "STOP",
    relational_required_for_integrated_answer: bool = True,
) -> dict[str, Any]:
    row_id = f"N0FINAL-{family.upper()}-{index + 1:02d}"
    positions = {value["id"]: pos for pos, value in enumerate(fields)}
    if target_distribution:
        target_names = [
            fields[positions[field_id]]["entity"]
            for field_id, weight in target_distribution.items()
            if weight > 0
        ]
        target_summary = (
            "The requested relational answer is "
            + " and ".join(target_names)
            + "."
        )
    elif termination == "UNKNOWN":
        target_summary = "The requested relation is outside the active schema, so the relation must remain unresolved."
    else:
        target_summary = "No relational execution is required for this request."

    return {
        "schema": SCHEMA,
        "id": row_id,
        "family": family,
        "split": "final_self_validation",
        "query": query,
        "query_views": [query, alternate_query(query)],
        "fields": fields,
        "edges": edges,
        "operator_target": {
            "relation_sequence": relation_sequence,
            "role_id": ROLE[role],
            "traversal_id": TRAVERSAL[traversal],
            "direction_id": DIRECTION[direction],
            "modifier_target": {
                "RELIABILITY": modifier_target[0],
                "RECENCY": modifier_target[1],
                "TEMPORAL_CONSTRAINT": modifier_target[2],
                "PROVENANCE_CONSTRAINT": modifier_target[3],
            },
            "focus_field_id": focus_field_id,
            "applicability": float(applicability),
            "control_id": CONTROL[control],
            "termination": termination,
        },
        "support_edge_ids": support_edge_ids,
        "target_distribution": target_distribution,
        "open_schema_relation": any(
            relation in OPEN_RELATIONS + FINAL_ONLY_RELATIONS
            for relation in relation_sequence
        ),
        "final_only_relation": any(
            relation in FINAL_ONLY_RELATIONS
            for relation in relation_sequence
        ),
        "relational_required_for_integrated_answer": relational_required_for_integrated_answer,
        "private_identity_data": False,
        # Full public N0 parent/fusion/latent inputs.
        "raw_text": (
            "Several public records are available. Their relationship, order, "
            "authority, and requested argument role determine the answer."
        ),
        "query_text": query,
        "relations": parent_relations(edges, positions),
        "target_summary_text": target_summary,
        "target_view_distribution": [0.10, 0.25, 0.65],
        "target_view_distribution_role": "compatibility_feature_not_validation_ground_truth",
        "view_reliability": [0.35, 0.80, 0.98],
        "view_available": [True, True, True],
        "cross_view_conflict": len(edges) > 1,
        "counterfactual_required": relational_required_for_integrated_answer,
        "counterfactual_view": 2 if relational_required_for_integrated_answer else None,
        "private_identity_content": False,
        "generated_text": True,
        "data_origin": "deterministic_public_n0_final_self_validation",
        "identity_authority": False,
        "training_authorized": False,
    }


def fields_for(index: int, family: str, types: list[str]) -> list[dict[str, Any]]:
    offset = RELATIONAL_FAMILIES.index(family) * 3 + index
    names = [ENTITIES[(offset + step * 5) % len(ENTITIES)] for step in range(6)]
    details = [
        "The record concerns a verified deployment transition.",
        "The record concerns an earlier operating state.",
        "The record concerns a signed evidence update.",
        "The record concerns a later service event.",
        "The record concerns a provenance-limited observation.",
        "The record concerns an unrelated distractor.",
    ]
    return [
        field(
            row_id=f"N0FINAL-{family.upper()}-{index + 1:02d}",
            index=i,
            entity=names[i],
            type_name=types[i] if i < len(types) else "GENERIC",
            detail=details[i],
        )
        for i in range(6)
    ]


def build_relational_row(family: str, index: int) -> dict[str, Any]:
    r1 = CORE_RELATIONS[(index + RELATIONAL_FAMILIES.index(family)) % len(CORE_RELATIONS)]
    r2 = CORE_RELATIONS[(CORE_RELATIONS.index(r1) + 2) % len(CORE_RELATIONS)]
    p1 = PHRASES[r1]
    p2 = PHRASES[r2]
    t1 = TYPE_PAIR[r1]
    t2 = TYPE_PAIR[r2]

    if family in {"single_source", "single_target"}:
        fields = fields_for(index, family, [t1[0], t1[1], "GENERIC", "GENERIC", "GENERIC", "GENERIC"])
        ids = [x["id"] for x in fields]
        edges = [edge("e0", ids[0], ids[1], r1)]
        role = "SOURCE" if family == "single_source" else "TARGET"
        target = ids[0] if role == "SOURCE" else ids[1]
        if role == "SOURCE":
            query = (
                f"The relation whose meaning is '{p1}' ends at {fields[1]['entity']}. "
                "Which record is the source endpoint?"
            )
            focus = ids[1]
        else:
            query = (
                f"Starting from {fields[0]['entity']}, use the relation whose meaning is '{p1}'. "
                "Which record is the receiving endpoint?"
            )
            focus = ids[0]
        return make_row(
            family=family, index=index, fields=fields, edges=edges,
            query=query,
            relation_sequence=[r1], role=role, traversal="LOCAL_SELECT",
            modifier_target=[0,0,0,0], support_edge_ids=["e0"],
            target_distribution={target:1.0}, focus_field_id=focus,
        )

    if family in {"path_source", "path_target", "path_order"}:
        fields = fields_for(index, family, ["GENERIC"] * 6)
        ids = [x["id"] for x in fields]
        edges = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[1], ids[2], r2),
            edge("e2", ids[0], ids[3], r2),
            edge("e3", ids[3], ids[4], r1),
        ]
        if family == "path_source":
            role, seq, target = "SOURCE", [r1,r2], ids[0]
        elif family == "path_target":
            role, seq, target = "TARGET", [r1,r2], ids[2]
        else:
            role, seq, target = "TARGET", [r2,r1], ids[4]
        query = (
            f"Start at {fields[0]['entity']}. Follow '{p1}' then '{p2}'."
            if seq == [r1,r2]
            else f"Start at {fields[0]['entity']}. Follow '{p2}' first and '{p1}' second."
        )
        query += (
            " Return the origin of that ordered route."
            if role == "SOURCE"
            else " Return the final endpoint of that ordered route."
        )
        return make_row(
            family=family,index=index,fields=fields,edges=edges,query=query,
            relation_sequence=seq,role=role,traversal="PATH",
            modifier_target=[0,0,0,0],
            support_edge_ids=["e0","e1"] if seq==[r1,r2] else ["e2","e3"],
            target_distribution={target:1.0},focus_field_id=ids[0],
        )

    if family == "three_hop":
        r3=CORE_RELATIONS[(CORE_RELATIONS.index(r2)+2)%len(CORE_RELATIONS)]
        p3=PHRASES[r3]
        fields=fields_for(index,family,["GENERIC"]*6); ids=[x["id"] for x in fields]
        edges=[edge("e0",ids[0],ids[1],r1),edge("e1",ids[1],ids[2],r2),edge("e2",ids[2],ids[3],r3)]
        return make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=f"Starting from {fields[0]['entity']}, trace '{p1}', then '{p2}', then '{p3}'. Which record is the endpoint?",
            relation_sequence=[r1,r2,r3],role="TARGET",traversal="PATH",
            modifier_target=[0,0,0,0],support_edge_ids=["e0","e1","e2"],
            target_distribution={ids[3]:1.0},focus_field_id=ids[0],
        )

    if family in {"plural_targets","ambiguous_plurality"}:
        fields=fields_for(index,family,[t1[0],t1[1],t1[0],t1[1],"GENERIC","GENERIC"]); ids=[x["id"] for x in fields]
        edges=[edge("e0",ids[0],ids[1],r1),edge("e1",ids[2],ids[3],r1)]
        return make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=f"Two equally valid links use the relation where a source {p1} a target. Preserve both receiving records.",
            relation_sequence=[r1],role="TARGET",traversal="AGGREGATE",
            modifier_target=[0,0,0,0],support_edge_ids=["e0","e1"],
            target_distribution={ids[1]:0.5,ids[3]:0.5},focus_field_id=None,
        )

    if family in {"reliability_arbitration","recency_arbitration","temporal_constraint","provenance_constraint","combined_constraints"}:
        fields=fields_for(index,family,[t1[0],t1[1],t1[0],t1[1],t1[0],t1[1]]); ids=[x["id"] for x in fields]
        if family=="reliability_arbitration":
            edges=[edge("e0",ids[0],ids[1],r1,reliability=0.97),edge("e1",ids[2],ids[3],r1,reliability=0.12)]
            mods=[1,0,0,0]; target=ids[1]; extra="Use the better verified relation."
        elif family=="recency_arbitration":
            edges=[edge("e0",ids[0],ids[1],r1,recency=0.15),edge("e1",ids[2],ids[3],r1,recency=0.96)]
            mods=[0,1,0,0]; target=ids[3]; extra="Use the latest applicable relation."
        elif family=="temporal_constraint":
            edges=[edge("e0",ids[0],ids[1],r1,temporal_match=0.0),edge("e1",ids[2],ids[3],r1,temporal_match=1.0)]
            mods=[0,0,1,0]; target=ids[3]; extra="Only the record valid for the requested time window may answer."
        elif family=="provenance_constraint":
            edges=[edge("e0",ids[0],ids[1],r1,provenance_match=1.0),edge("e1",ids[2],ids[3],r1,provenance_match=0.0)]
            mods=[0,0,0,1]; target=ids[1]; extra="Only the requested provenance class may answer."
        else:
            edges=[
                edge("e0",ids[0],ids[1],r1,temporal_match=1.0,provenance_match=0.0),
                edge("e1",ids[2],ids[3],r1,temporal_match=0.0,provenance_match=1.0),
                edge("e2",ids[4],ids[5],r1,temporal_match=1.0,provenance_match=1.0),
            ]
            mods=[0,0,1,1]; target=ids[5]; extra="The answer must satisfy both the time window and provenance requirement."
        return make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=f"Several links match the relation where a source {p1} a target. {extra} Which record is the receiving endpoint?",
            relation_sequence=[r1],role="TARGET",traversal="AGGREGATE",
            modifier_target=mods,support_edge_ids=[x["id"] for x in edges],
            target_distribution={target:1.0},focus_field_id=None,
        )

    if family == "unknown_fail_closed":
        fields=fields_for(index,family,["GENERIC"]*6); ids=[x["id"] for x in fields]
        edges=[edge("e0",ids[0],ids[1],r1),edge("e1",ids[2],ids[3],r2)]
        return make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=f"Starting from {fields[0]['entity']}, follow the relation 'is certified by the same external legal trustee'. Which record is reached?",
            relation_sequence=[],role="NONE",traversal="LOCAL_SELECT",
            direction="BIDIRECTIONAL", modifier_target=[0,0,0,0],support_edge_ids=[],target_distribution={},
            focus_field_id=ids[0],applicability=0.6,control="DEFER",termination="UNKNOWN",
            relational_required_for_integrated_answer=False,
        )

    if family == "existing_open_schema":
        relation=OPEN_RELATIONS[index%len(OPEN_RELATIONS)]
        types=TYPE_PAIR[relation]
        fields=fields_for(index,family,[types[0],types[1],"GENERIC","GENERIC","GENERIC","GENERIC"]); ids=[x["id"] for x in fields]
        edges=[edge("e0",ids[0],ids[1],relation)]
        symmetric=relation=="CONFLICTS_WITH"
        return make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=(
                f"Starting from {fields[0]['entity']}, use the supplied schema relation meaning "
                f"'{PHRASES[relation]}'. Return the requested endpoint semantics."
            ),
            relation_sequence=[relation],role="SYMMETRIC" if symmetric else "TARGET",
            traversal="LOCAL_SELECT",modifier_target=[0,0,0,0],support_edge_ids=["e0"],
            target_distribution={ids[0]:0.5,ids[1]:0.5} if symmetric else {ids[1]:1.0},
            focus_field_id=ids[0],
        )

    if family in {"final_unseen_enables","final_unseen_prevents"}:
        relation="ENABLES" if family.endswith("enables") else "PREVENTS"
        types=TYPE_PAIR[relation]
        fields=fields_for(index,family,[types[0],types[1],"GENERIC","GENERIC","GENERIC","GENERIC"]); ids=[x["id"] for x in fields]
        edges=[edge("e0",ids[0],ids[1],relation)]
        return make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=(
                f"Starting from {fields[0]['entity']}, use the new schema relation meaning "
                f"'{PHRASES[relation]}'. Which record is the receiving endpoint?"
            ),
            relation_sequence=[relation],role="TARGET",traversal="LOCAL_SELECT",
            modifier_target=[0,0,0,0],support_edge_ids=["e0"],
            target_distribution={ids[1]:1.0},focus_field_id=ids[0],
        )

    if family=="symmetric_conflict":
        relation="CONFLICTS_WITH"; types=TYPE_PAIR[relation]
        fields=fields_for(index,family,[types[0],types[1],"GENERIC","GENERIC","GENERIC","GENERIC"]); ids=[x["id"] for x in fields]
        edges=[edge("e0",ids[0],ids[1],relation)]
        return make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=(
                f"{fields[0]['entity']} participates in a relation whose supplied meaning is "
                f"'{PHRASES[relation]}'. Preserve both conflicting participants."
            ),
            relation_sequence=[relation],role="SYMMETRIC",traversal="AGGREGATE",
            direction="BIDIRECTIONAL",
            modifier_target=[0,0,0,0],support_edge_ids=["e0"],
            target_distribution={ids[0]:0.5,ids[1]:0.5},focus_field_id=None,
        )

    if family=="outside_support":
        pair=index//2
        fields=fields_for(pair,family,[t1[0],t1[1],"GENERIC","GENERIC","GENERIC","GENERIC"]); ids=[x["id"] for x in fields]
        value=0.05 if index%2==0 else 0.99
        fields[5]["metadata"]["reliability"]=value
        fields[5]["metadata"]["normalized_time"]=value
        edges=[edge("e0",ids[0],ids[1],r1)]
        row=make_row(
            family=family,index=index,fields=fields,edges=edges,
            query=f"{fields[0]['entity']} {p1} {fields[1]['entity']}. Which record is the receiving endpoint?",
            relation_sequence=[r1],role="TARGET",traversal="LOCAL_SELECT",
            modifier_target=[0,0,0,0],support_edge_ids=["e0"],
            target_distribution={ids[1]:1.0},focus_field_id=ids[0],
        )
        row["causal_group"]=f"final:outside_support:{pair:02d}"
        return row

    if family=="nonrel_fallback":
        fields=fields_for(index,family,["GENERIC"]*6)
        return make_row(
            family=family,index=index,fields=fields,edges=[],
            query="Interpret the ordinary phrase 'sometime after lunch' without performing graph relation execution.",
            relation_sequence=[],role="NONE",traversal="LOCAL_SELECT",
            direction="BIDIRECTIONAL", modifier_target=[0,0,0,0],support_edge_ids=[],target_distribution={},
            focus_field_id=None,applicability=0.05,control="FALLBACK",termination="STOP",
            relational_required_for_integrated_answer=False,
        )

    raise KeyError(family)


def build_general_rows() -> list[dict[str, Any]]:
    fabric.base.SUBJECTS = GENERAL_SUBJECTS
    fabric.base.ATTRIBUTES = GENERAL_ATTRIBUTES
    rows = []
    for family in fabric.base.FAMILIES:
        for index in range(ROWS_PER_FAMILY):
            item = fabric.build_row(family, index)
            item["id"] = item["id"].replace("N0V02-LATCHAL1-", "N0FINAL-FABRIC-")
            item["split"] = "final_self_validation"
            item["challenge_version"] = "n0_final_self_validation_v1"
            item["training_authorized"] = False
            item["private_identity_content"] = False
            rows.append(item)
    return rows


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--general-output",required=True)
    p.add_argument("--relational-output",required=True)
    p.add_argument("--manifest",required=True)
    args=p.parse_args()

    general=build_general_rows()
    relational=[
        build_relational_row(family,index)
        for family in RELATIONAL_FAMILIES
        for index in range(ROWS_PER_FAMILY)
    ]
    if len(general)!=160 or len(relational)!=160:
        raise SystemExit("final self-validation row count drift")
    if len({r["id"] for r in general+relational})!=320:
        raise SystemExit("final self-validation id collision")
    if any(r.get("training_authorized") is not False for r in general+relational):
        raise SystemExit("final validation rows unexpectedly authorize training")
    if any(r.get("private_identity_content",r.get("private_identity_data",False)) for r in general+relational):
        raise SystemExit("private identity data entered final N0 validation")

    general_path=Path(args.general_output)
    relational_path=Path(args.relational_output)
    manifest_path=Path(args.manifest)
    general_path.parent.mkdir(parents=True,exist_ok=True)
    relational_path.parent.mkdir(parents=True,exist_ok=True)
    general_path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in general),encoding="utf-8")
    relational_path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in relational),encoding="utf-8")

    manifest={
        "schema":"alice.eipm.n0.final-self-validation-manifest.v1",
        "status":"FROZEN_BY_BUILD_SOURCE_BEFORE_PRODUCTION_MODEL_RESULTS",
        "general_rows":len(general),
        "relational_rows":len(relational),
        "total_rows":len(general)+len(relational),
        "general_families":dict(sorted(Counter(r["family"] for r in general).items())),
        "relational_families":dict(sorted(Counter(r["family"] for r in relational).items())),
        "general_sha256":sha(general_path),
        "relational_sha256":sha(relational_path),
        "final_only_relation_keys":list(FINAL_ONLY_RELATIONS),
        "final_only_relation_rows":sum(r.get("final_only_relation",False) for r in relational),
        "unknown_rows":sum(r["operator_target"]["termination"]=="UNKNOWN" for r in relational),
        "counterfactual_required_relational_rows":sum(r["relational_required_for_integrated_answer"] for r in relational),
        "training_authorized":False,
        "results_observed_at_build_time":False,
        "identity_authority":False,
        "private_identity_content":False,
        "external_validator":False,
        "validation_owner":"alice_native_self_validation_harness",
    }
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
