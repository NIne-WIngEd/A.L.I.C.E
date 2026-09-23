#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROW_SCHEMA = "alice.eipm.n0.full-envelope-behavioral-row.v1"
MANIFEST_SCHEMA = "alice.eipm.n0.full-envelope-behavioral-manifest.v1"

TYPE_SCHEMA = [
    {"key":"t_actor","text":"a person, organization, or social actor"},
    {"key":"t_record","text":"a report, note, message, document, or recorded source"},
    {"key":"t_claim","text":"a claim, proposition, or conclusion that can be supported or challenged"},
    {"key":"t_event","text":"an event, occurrence, or state transition"},
    {"key":"t_context","text":"a contextual item that may be relevant without itself deciding the conclusion"},
]

RELATIONS = [
    {
        "key":"r_support",
        "family":"evidential_support",
        "text":"the source provides evidence in favor of the target claim",
        "domain":["t_record","t_claim"],
        "range":["t_claim"],
        "symmetric":False,
    },
    {
        "key":"r_contradict",
        "family":"conflict",
        "text":"the two endpoints express claims that conflict and cannot both hold as stated",
        "domain":["t_claim"],
        "range":["t_claim"],
        "symmetric":True,
    },
    {
        "key":"r_supersede",
        "family":"supersession",
        "text":"the source is a newer controlling replacement that supersedes the target record",
        "domain":["t_record"],
        "range":["t_record"],
        "symmetric":False,
    },
    {
        "key":"r_cause",
        "family":"causation",
        "text":"the source event causally contributes to producing the target event or state",
        "domain":["t_event"],
        "range":["t_event"],
        "symmetric":False,
    },
    {
        "key":"r_derived",
        "family":"derivation",
        "text":"the source record was derived or constructed from the target record",
        "domain":["t_record"],
        "range":["t_record"],
        "symmetric":False,
    },
    {
        "key":"r_authored",
        "family":"authorship",
        "text":"the source record was authored or issued by the target actor",
        "domain":["t_record"],
        "range":["t_actor"],
        "symmetric":False,
    },
    {
        "key":"r_refers",
        "family":"reference",
        "text":"the source record or claim refers to and is about the target context item",
        "domain":["t_record","t_claim"],
        "range":["t_context","t_actor","t_event"],
        "symmetric":False,
    },
    {
        "key":"r_corroborate",
        "family":"corroboration",
        "text":"the two endpoints independently reinforce the same underlying information",
        "domain":["t_record","t_claim"],
        "range":["t_record","t_claim"],
        "symmetric":True,
    },
    {
        "key":"r_follow",
        "family":"temporal_order",
        "text":"the source event occurs immediately after the target event in the relevant sequence",
        "domain":["t_event"],
        "range":["t_event"],
        "symmetric":False,
    },
    {
        "key":"r_context",
        "family":"contextual_relation",
        "text":"the two endpoints are contextually related without asserting a causal direction",
        "domain":["t_actor","t_record","t_claim","t_event","t_context"],
        "range":["t_actor","t_record","t_claim","t_event","t_context"],
        "symmetric":True,
    },
]

FACTOR_BANKS = {
    "role":[
        {"opcode":"ROLE_SOURCE","text":"read the semantic source endpoint that originates or performs the requested relation"},
        {"opcode":"ROLE_TARGET","text":"read the semantic target endpoint reached or acted on by the requested relation"},
        {"opcode":"ROLE_SYMMETRIC","text":"preserve both endpoints symmetrically because neither endpoint is the directional winner"},
        {"opcode":"ROLE_NONE","text":"do not select an endpoint role because relational execution is not grounded"},
    ],
    "traversal":[
        {"opcode":"TRAVERSAL_LOCAL","text":"use one immediate local relation step"},
        {"opcode":"TRAVERSAL_PATH","text":"follow an ordered multi-step relation path"},
        {"opcode":"TRAVERSAL_AGGREGATE","text":"aggregate multiple matching support links without forcing one path winner"},
    ],
    "direction":[
        {"opcode":"DIRECTION_FORWARD","text":"follow the stored semantic relation from source toward target"},
        {"opcode":"DIRECTION_REVERSE","text":"follow the semantic relation backward from target toward source"},
        {"opcode":"DIRECTION_BIDIRECTIONAL","text":"treat traversal as bidirectional or preserve both directions"},
    ],
    "control":[
        {"opcode":"CONTROL_FALLBACK","text":"use ordinary semantic judgment because relational execution is not applicable"},
        {"opcode":"CONTROL_RELATIONAL","text":"execute relational reasoning over the supplied runtime schema and evidence"},
        {"opcode":"CONTROL_DEFER","text":"defer because the supplied relation schema or evidence is insufficient"},
    ],
    "reliability":[
        {"opcode":"MOD_RELIABILITY_OFF","text":"do not use reliability as an arbitration criterion"},
        {"opcode":"MOD_RELIABILITY_ON","text":"prefer evidence with stronger verified reliability when alternatives conflict"},
    ],
    "recency":[
        {"opcode":"MOD_RECENCY_OFF","text":"do not use recency as an arbitration criterion"},
        {"opcode":"MOD_RECENCY_ON","text":"prefer the latest applicable controlling evidence when alternatives conflict"},
    ],
    "temporal":[
        {"opcode":"MOD_TEMPORAL_OFF","text":"do not require evidence to satisfy an explicit time-window constraint"},
        {"opcode":"MOD_TEMPORAL_ON","text":"require evidence to satisfy the active temporal scope"},
    ],
    "provenance":[
        {"opcode":"MOD_PROVENANCE_OFF","text":"do not require a particular provenance class"},
        {"opcode":"MOD_PROVENANCE_ON","text":"require evidence to come from the authorized provenance class"},
    ],
    "open_semantic_factor":[
        {"opcode":None,"text":"an explicit correction is present and should remain semantically available"},
        {"opcode":None,"text":"multiple co-valid interpretations should remain represented"},
        {"opcode":None,"text":"a conflict is unresolved and should preserve uncertainty"},
        {"opcode":None,"text":"no additional open semantic qualifier is active"},
    ],
}

for _bank_name, _items in FACTOR_BANKS.items():
    for _index, _item in enumerate(_items):
        _item["key"] = (
            str(_item["opcode"])
            if _item["opcode"] is not None
            else f"OPEN_{_index}"
        )


TRAIN_ENTITIES = [
    "Aster","Beryl","Cinder","Dorian","Elio","Fenn","Galen","Hera",
    "Ivo","Juno","Kora","Lyra","Miro","Nola","Orin","Pia",
]
DEV_ENTITIES = [
    "Quartz","Riven","Solace","Tern","Umber","Vega","Wren","Xanthe",
    "Yarrow","Zephyr","Arden","Brio","Cyra","Delta","Ember","Fjord",
]

FINAL_ENTITIES = [
    "Grove","Harbor","Ion","Juniper","Kestrel","Lattice","Meridian","Nimbus",
    "Onyx","Pioneer","Quill","Radian","Summit","Trellis","Vale","Willow",
]

DEV_TYPE_SCHEMA = [
    {"key":"t_actor","text":"a responsible person, team, or public organization participating in a tracked process"},
    {"key":"t_record","text":"an auditable docket, report, notice, or recorded public source"},
    {"key":"t_claim","text":"a finding, proposition, or conclusion whose support must be established from evidence"},
    {"key":"t_event","text":"a process occurrence, handoff, or state change that can participate in ordered reasoning"},
    {"key":"t_context","text":"background material that may inform context without controlling the requested conclusion"},
]

FINAL_TYPE_SCHEMA = [
    {"key":"t_actor","text":"an accountable human or institutional participant in a public process"},
    {"key":"t_record","text":"a traceable public memorandum, instrument record, notice, or source document"},
    {"key":"t_claim","text":"a proposition or operational conclusion whose support can change with evidence"},
    {"key":"t_event","text":"an observed occurrence, transition, or process state in a public system"},
    {"key":"t_context","text":"background material that can supply context without deciding the requested conclusion"},
]

INTERNAL_VIEWS = [
    "raw_semantic",
    "structured",
    "evidence_specialist",
    "graph_source",
    "graph_target",
    "relational_executor",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(
    text: str,
    type_key: str,
    *,
    confidence: float = 0.9,
    reliability: float = 0.9,
    recency: float = 0.5,
    temporal: str = "time compatible",
    provenance: str = "ordinary public record",
) -> dict[str, Any]:
    return {
        "text":text,
        "type_key":type_key,
        "confidence":confidence,
        "missing":0.0,
        "reliability":reliability,
        "metadata":[recency,1.0,1.0],
        "descriptors":{
            "provenance":provenance,
            "temporal_scope":temporal,
        },
    }


def edge(
    source: int,
    target: int,
    relation_key: str,
    *,
    reliability: float = 0.9,
    recency: float = 0.5,
    temporal_match: float = 1.0,
    provenance_match: float = 1.0,
    support: bool = False,
    decisive: bool = False,
    irrelevant: bool = False,
) -> dict[str, Any]:
    return {
        "source_field":source,
        "target_field":target,
        "relation_key":relation_key,
        "valid":True,
        "reliability":reliability,
        "recency":recency,
        "temporal_match":temporal_match,
        "provenance_match":provenance_match,
        "metadata":[reliability,recency,temporal_match,provenance_match],
        "support_target":support,
        "decisive":decisive,
        "irrelevant":irrelevant,
    }


def default_factor_targets() -> dict[str,str]:
    return {
        "role":"ROLE_TARGET",
        "traversal":"TRAVERSAL_LOCAL",
        "direction":"DIRECTION_FORWARD",
        "control":"CONTROL_RELATIONAL",
        "reliability":"MOD_RELIABILITY_OFF",
        "recency":"MOD_RECENCY_OFF",
        "temporal":"MOD_TEMPORAL_OFF",
        "provenance":"MOD_PROVENANCE_OFF",
        "open_semantic_factor":"OPEN_3",
    }


def factor_counterfactuals(targets: dict[str,str]) -> dict[str,str | None]:
    result: dict[str,str | None] = {}
    for name, target in targets.items():
        options = [str(x["key"]) for x in FACTOR_BANKS[name]]
        alternatives = [x for x in options if x != target]
        result[name] = alternatives[0] if alternatives else None
    return result


def unique_entity_sequence(
    entities: list[str],
    *,
    example: int,
    count: int,
    stride: int,
) -> list[str]:
    if count <= 0:
        raise ValueError("unique entity sequence count must be positive")
    unique=list(dict.fromkeys(str(x) for x in entities))
    if len(unique) < count:
        raise ValueError(
            f"entity pool has {len(unique)} unique values but {count} are required"
        )
    # Pick a deterministic coprime-ish walk, then fall back to the remaining
    # pool if a stride aliases modulo the current entity count. The result is
    # row-local and contains no learned identity or cardinality parameter.
    start=(int(example)*7 + count*3) % len(unique)
    selected=[]
    seen=set()
    for offset in range(len(unique)*2):
        value=unique[(start + offset*int(stride)) % len(unique)]
        if value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected)==count:
            return selected
    for value in unique:
        if value not in seen:
            selected.append(value)
            seen.add(value)
            if len(selected)==count:
                return selected
    raise RuntimeError("unable to construct deterministic unique entity sequence")


def scenario(mode: int, entities: list[str], example: int, *, split: str) -> dict[str,Any]:
    pair_anchor=example-1 if mode==16 else example
    entity_example=pair_anchor if mode in {15,16} else example
    if mode in {15,16}:
        base=entity_example % len(entities)
        a=entities[base]
        b=entities[(base+1) % len(entities)]
        c=entities[(base+2) % len(entities)]
        d=entities[(base+3) % len(entities)]
    else:
        a=entities[(entity_example*3) % len(entities)]
        b=entities[(entity_example*5+1) % len(entities)]
        c=entities[(entity_example*7+2) % len(entities)]
        d=entities[(entity_example*11+3) % len(entities)]
    entities_used=[a,b,c,d]
    targets=default_factor_targets()
    step_targets: list[dict[str,str]] = []
    relation_sequence: list[str] = []
    relation_counterfactuals: list[str | None] = []
    event_sequence: list[str] = []
    endpoint={"active":True,"source_field":0,"target_field":1}
    uncertainty=0.05
    decisive_fields: list[int] = []
    irrelevant_fields: list[int] = []
    support_edges: list[int] = []
    recoverable=list(INTERNAL_VIEWS)
    family=""
    candidate_context_swap_pair_anchor: int | None = None
    candidate_context_swap_variant: str | None = None

    if mode == 0:
        family="source_target_role"
        fields=[
            field(f"Verified report from {a} states that the tested claim is supported by direct measurements.","t_record",reliability=0.96,provenance="verified primary public record"),
            field(f"Claim {b}: the measured condition occurred as reported.","t_claim"),
            field(f"Claim {c}: a different condition occurred instead.","t_claim",confidence=0.6),
            field(f"Context note {d} discusses packaging and does not bear on either claim.","t_context",reliability=0.4),
        ]
        edges=[
            edge(0,1,"r_support",reliability=0.96,support=True,decisive=True),
            edge(0,2,"r_contradict",reliability=0.45),
            edge(3,0,"r_context",reliability=0.3,irrelevant=True),
        ]
        query="Which claim is supported by the verified measurement report?"
        answers=[f"The supported conclusion is Claim {b}.",f"The supported conclusion is Claim {c}.","The packaging note should decide the conclusion.","Defer even though the verified report directly supports one claim."]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_contradict"]
        event_sequence=["CONTINUE","STOP"]
        decisive_fields=[0]
        irrelevant_fields=[3]
        support_edges=[0]
    elif mode == 1:
        family="ordered_composition"
        fields=[
            field(f"Event {a}: an initiating state transition occurred.","t_event"),
            field(f"Event {b}: the intermediate state followed from the initiating event.","t_event"),
            field(f"Event {c}: the final state followed from the intermediate event.","t_event"),
            field(f"Context item {d} is unrelated to the causal sequence.","t_context",reliability=0.3),
        ]
        edges=[
            edge(0,1,"r_cause",support=True,decisive=True),
            edge(1,2,"r_cause",support=True,decisive=True),
            edge(3,0,"r_context",irrelevant=True,reliability=0.2),
        ]
        query=f"Starting from Event {a}, follow the causal relation twice in order. Which event is reached last?"
        answers=[f"Event {c} is reached last.",f"Event {b} is reached last.",f"Context item {d} is reached last.","The path should be treated as unordered."]
        correct=0
        relation_sequence=["r_cause","r_cause"]
        relation_counterfactuals=["r_context","r_context"]
        event_sequence=["CONTINUE","CONTINUE","STOP"]
        targets["traversal"]="TRAVERSAL_PATH"
        step_targets=[
            {**targets,"direction":"DIRECTION_FORWARD"},
            {**targets,"direction":"DIRECTION_FORWARD"},
        ]
        endpoint={"active":True,"source_field":0,"target_field":2}
        decisive_fields=[1]
        irrelevant_fields=[3]
        support_edges=[0,1]
    elif mode == 2:
        family="reverse_traversal"
        fields=[
            field(f"Record {a} is a public technical report.","t_record"),
            field(f"Actor {b} authored Record {a}.","t_actor"),
            field(f"Actor {c} did not author the report.","t_actor",confidence=0.7),
            field(f"Context note {d} is unrelated to authorship.","t_context",reliability=0.3),
        ]
        edges=[
            edge(0,1,"r_authored",support=True,decisive=True),
            edge(0,2,"r_refers",reliability=0.35),
            edge(3,0,"r_context",irrelevant=True,reliability=0.2),
        ]
        query=f"Start at Actor {b} and traverse the authored-by relation in reverse. Which record is reached?"
        answers=[f"Record {a} is reached.",f"Actor {c} is reached.",f"Context note {d} is reached.","No record can be reached by reversing the relation."]
        correct=0
        relation_sequence=["r_authored"]
        relation_counterfactuals=["r_refers"]
        event_sequence=["CONTINUE","STOP"]
        targets["role"]="ROLE_SOURCE"
        targets["direction"]="DIRECTION_REVERSE"
        endpoint={"active":True,"source_field":0,"target_field":1}
        decisive_fields=[0]
        irrelevant_fields=[3]
        support_edges=[0]
    elif mode == 3:
        family="symmetric_relation"
        fields=[
            field(f"Audit record {a} independently reports the same observation.","t_record"),
            field(f"Audit record {b} independently reports the same observation.","t_record"),
            field(f"Record {c} reports an unrelated observation.","t_record",reliability=0.5),
            field(f"Context item {d} is unrelated.","t_context",reliability=0.3),
        ]
        edges=[
            edge(1,0,"r_corroborate",support=True,decisive=True),
            edge(2,0,"r_context",reliability=0.3),
            edge(3,2,"r_context",irrelevant=True,reliability=0.2),
        ]
        query=f"Which record mutually corroborates Audit record {a}?"
        answers=[f"Audit record {b} corroborates it.",f"Record {c} corroborates it.",f"Context item {d} corroborates it.","The storage orientation makes corroboration impossible."]
        correct=0
        relation_sequence=["r_corroborate"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["role"]="ROLE_SYMMETRIC"
        targets["direction"]="DIRECTION_BIDIRECTIONAL"
        endpoint={"active":True,"source_field":0,"target_field":1}
        decisive_fields=[1]
        irrelevant_fields=[3]
        support_edges=[0]
    elif mode == 4:
        family="reliability_arbitration"
        fields=[
            field(f"Verified source {a} reports strong direct evidence for Claim {b}.","t_record",reliability=0.98,provenance="verified primary public record"),
            field(f"Claim {b}: the measured result is accepted.","t_claim"),
            field(f"Unverified source {c} weakly supports a competing claim.","t_record",reliability=0.30,provenance="unverified secondary public record"),
            field(f"Claim {d}: the competing result is accepted.","t_claim",confidence=0.55),
            field("A scheduling notice is newer but unrelated to the evidence question.","t_context",reliability=0.4,recency=0.95),
        ]
        edges=[
            edge(0,1,"r_support",reliability=0.98,support=True,decisive=True),
            edge(2,3,"r_support",reliability=0.30),
            edge(4,0,"r_context",reliability=0.2,recency=0.95,irrelevant=True),
        ]
        query="Two sources support competing claims. Which claim should be selected when verified reliability is the required arbitration criterion?"
        answers=[f"Select Claim {b} using the verified high-reliability source.",f"Select Claim {d} using the weak unverified source.","Select the unrelated scheduling notice.","Ignore reliability and choose arbitrarily."]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["traversal"]="TRAVERSAL_AGGREGATE"
        targets["reliability"]="MOD_RELIABILITY_ON"
        decisive_fields=[0]
        irrelevant_fields=[4]
        support_edges=[0,1]
    elif mode == 5:
        family="recency_supersession"
        fields=[
            field(f"Record {a} is the older version of the report.","t_record",recency=0.15),
            field(f"Record {b} is a signed later correction replacing Record {a}.","t_record",reliability=0.97,recency=0.95,provenance="verified correction record"),
            field(f"Claim {c} is supported by the corrected record.","t_claim"),
            field(f"Context note {d} is newer than both records but unrelated.","t_context",recency=1.0,reliability=0.3),
        ]
        edges=[
            edge(1,0,"r_supersede",recency=0.95,support=True,decisive=True),
            edge(1,2,"r_support",recency=0.95,support=True),
            edge(3,1,"r_context",recency=1.0,irrelevant=True,reliability=0.2),
        ]
        query=f"Which record controls when the signed later correction explicitly replaces the older report?"
        answers=[f"Record {b} controls because it supersedes the older record.",f"Record {a} controls because it existed first.",f"Context note {d} controls because it is newest.","Treat the two report versions as equally controlling."]
        correct=0
        relation_sequence=["r_supersede"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["role"]="ROLE_SOURCE"
        targets["recency"]="MOD_RECENCY_ON"
        endpoint={"active":True,"source_field":1,"target_field":0}
        decisive_fields=[1]
        irrelevant_fields=[3]
        support_edges=[0]
    elif mode == 6:
        family="temporal_constraint"
        fields=[
            field(f"Event {a} occurred inside the requested observation window.","t_event",temporal="inside requested time window"),
            field(f"Event {b} followed from Event {a} inside that same window.","t_event",temporal="inside requested time window"),
            field(f"Event {c} occurred outside the requested window.","t_event",temporal="outside requested time window"),
            field(f"Event {d} followed from Event {c} outside the requested window.","t_event",temporal="outside requested time window"),
        ]
        edges=[
            edge(0,1,"r_cause",temporal_match=1.0,support=True,decisive=True),
            edge(2,3,"r_cause",temporal_match=0.0),
        ]
        query="Which effect is supported when causal evidence must satisfy the requested time window?"
        answers=[f"Event {b} is the supported in-window effect.",f"Event {d} is the supported effect even though it is out of window.","Both effects must be accepted regardless of time.","Defer despite one exact in-window causal link."]
        correct=0
        relation_sequence=["r_cause"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["temporal"]="MOD_TEMPORAL_ON"
        decisive_fields=[0]
        support_edges=[0,1]
    elif mode == 7:
        family="provenance_constraint"
        fields=[
            field(f"Authorized record {a} provides evidence for Claim {b}.","t_record",reliability=0.95,provenance="authorized verified provenance class"),
            field(f"Claim {b}: the authorized record's conclusion.","t_claim"),
            field(f"Unverified record {c} supports a competing Claim {d}.","t_record",reliability=0.8,provenance="outside authorized provenance class"),
            field(f"Claim {d}: the competing unverified conclusion.","t_claim"),
        ]
        edges=[
            edge(0,1,"r_support",provenance_match=1.0,support=True,decisive=True),
            edge(2,3,"r_support",provenance_match=0.0),
        ]
        query="Which claim is supported when evidence must come from the authorized provenance class?"
        answers=[f"Claim {b} is supported by the authorized record.",f"Claim {d} is supported even though its provenance is outside the allowed class.","Both claims are equally authorized.","Ignore provenance and choose the longer record."]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["provenance"]="MOD_PROVENANCE_ON"
        decisive_fields=[0]
        support_edges=[0,1]
    elif mode == 8:
        family="conflict_plurality"
        fields=[
            field(f"Claim {a}: the component failed before the test ended.","t_claim",confidence=0.9),
            field(f"Claim {b}: the component remained operational through the test.","t_claim",confidence=0.9),
            field(f"Record {c} reports both claims from independent sources without resolving them.","t_record",reliability=0.85),
            field(f"Context item {d} does not resolve the conflict.","t_context",reliability=0.4),
        ]
        edges=[
            edge(0,1,"r_contradict",support=True,decisive=True),
            edge(2,0,"r_refers",reliability=0.8),
            edge(2,1,"r_refers",reliability=0.8),
            edge(3,2,"r_context",irrelevant=True,reliability=0.2),
        ]
        query="Two well-supported claims directly conflict and the available record does not resolve them. What should the system preserve?"
        answers=[f"Preserve the unresolved conflict between Claim {a} and Claim {b}.",f"Force Claim {a} as true without additional evidence.",f"Force Claim {b} as true without additional evidence.","Use the unrelated context item as the tiebreaker."]
        correct=0
        relation_sequence=["r_contradict"]
        relation_counterfactuals=["r_support"]
        event_sequence=["CONTINUE","STOP"]
        targets["role"]="ROLE_SYMMETRIC"
        targets["traversal"]="TRAVERSAL_AGGREGATE"
        targets["direction"]="DIRECTION_BIDIRECTIONAL"
        targets["open_semantic_factor"]="OPEN_2"
        uncertainty=0.85
        endpoint={"active":True,"source_field":0,"target_field":1}
        decisive_fields=[0,1]
        irrelevant_fields=[3]
        support_edges=[0]
    elif mode == 9:
        family="unknown_defer"
        fields=[
            field(f"Record {a} supports an ordinary Claim {b}.","t_record"),
            field(f"Claim {b}: an ordinary supported conclusion.","t_claim"),
            field(f"Context item {c} mentions Actor {d} without asserting the requested relationship.","t_context",reliability=0.5),
            field(f"Actor {d} appears only in contextual material.","t_actor",reliability=0.5),
        ]
        edges=[
            edge(0,1,"r_support",reliability=0.9),
            edge(2,3,"r_context",irrelevant=True,reliability=0.3),
        ]
        query="The requested relation is a licensing relationship, but no supplied runtime relation description expresses licensing. What should the system do?"
        answers=["Defer because the requested relation is absent from the supplied runtime schema.","Force the evidential-support relation as a substitute.","Use contextual relatedness as if it meant licensing.","Choose a relation from candidate position alone."]
        correct=0
        relation_sequence=[]
        relation_counterfactuals=[]
        event_sequence=["UNKNOWN"]
        targets["role"]="ROLE_NONE"
        targets["control"]="CONTROL_DEFER"
        targets["open_semantic_factor"]="OPEN_3"
        endpoint={"active":False,"source_field":-1,"target_field":-1}
        uncertainty=1.0
        irrelevant_fields=[2,3]
        recoverable=["raw_semantic","structured","evidence_specialist"]
    elif mode == 10:
        family="decisive_source"
        fields=[
            field(f"Primary record {a} contains the only verified measurement establishing Claim {b}.","t_record",reliability=0.99,provenance="verified decisive primary record"),
            field(f"Claim {b}: the verified measurement establishes the requested conclusion.","t_claim"),
            field(f"Background record {c} gives general context but no measurement of the claim.","t_record",reliability=0.7),
            field(f"Context item {d} is unrelated to the requested conclusion.","t_context",reliability=0.3),
        ]
        edges=[
            edge(0,1,"r_support",reliability=0.99,support=True,decisive=True),
            edge(2,3,"r_context",reliability=0.6),
            edge(3,0,"r_context",irrelevant=True,reliability=0.2),
        ]
        query="Which conclusion is justified by the decisive verified measurement?"
        answers=[f"Claim {b} is justified by the decisive verified measurement.","The general background record decides the claim.","The unrelated context item decides the claim.","There is no difference between decisive and merely available evidence."]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        decisive_fields=[0]
        irrelevant_fields=[3]
        support_edges=[0]
    elif mode == 11:
        family="irrelevant_distractor"
        fields=[
            field(f"Verified record {a} supports Claim {b}.","t_record",reliability=0.96),
            field(f"Claim {b}: the supported technical conclusion.","t_claim"),
            field(f"Unrelated record {c} supports an unrelated cafeteria scheduling claim.","t_record",reliability=0.95,recency=1.0),
            field(f"Unrelated Claim {d}: the cafeteria schedule changed.","t_claim",reliability=0.8,recency=1.0),
        ]
        edges=[
            edge(0,1,"r_support",reliability=0.96,support=True,decisive=True),
            # Same runtime relation, but query-irrelevant. Binder supervision
            # must not collapse query relevance into relation-key equality.
            edge(2,3,"r_support",reliability=0.9,recency=1.0,irrelevant=True),
        ]
        query="Which conclusion is supported by the relevant evidence, ignoring newer but unrelated context?"
        answers=[f"Claim {b} remains the supported conclusion.",f"Notice {c} should override the technical evidence because it is newer.",f"Context item {d} should decide the technical question.","All available text should influence the answer equally."]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        decisive_fields=[0]
        irrelevant_fields=[2,3]
        support_edges=[0]
    elif mode == 12:
        family="multiple_co_valid_support"
        fields=[
            field(f"Independent record {a} supports Claim {c}.","t_record",reliability=0.93),
            field(f"Independent record {b} separately supports Claim {c}.","t_record",reliability=0.92),
            field(f"Claim {c}: both independent records support the same conclusion.","t_claim"),
            field(f"Context item {d} is unrelated.","t_context",reliability=0.3),
        ]
        edges=[
            edge(0,2,"r_support",reliability=0.93,support=True,decisive=True),
            edge(1,2,"r_support",reliability=0.92,support=True,decisive=True),
            edge(0,1,"r_corroborate",reliability=0.9,support=True),
            edge(3,2,"r_context",irrelevant=True,reliability=0.2),
        ]
        query="How should two independent reliable records supporting the same claim be treated?"
        answers=[f"Use both independent records as co-valid support for Claim {c}.",f"Discard record {a} only because record {b} also exists.",f"Discard record {b} only because record {a} also exists.","Use the unrelated context item instead."]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["traversal"]="TRAVERSAL_AGGREGATE"
        targets["open_semantic_factor"]="OPEN_1"
        decisive_fields=[0,1]
        irrelevant_fields=[3]
        support_edges=[0,1]
        endpoint={"active":True,"source_field":0,"target_field":2}
    elif mode == 13:
        family="mixed_direction_composition"
        fields=[
            field(f"Event {a} causes Event {b}.","t_event"),
            field(f"Event {b} is the shared intermediate event.","t_event"),
            field(f"Event {c} also causes Event {b} from the opposite side of the requested path.","t_event"),
            field(f"Context item {d} is unrelated to the path.","t_context",reliability=0.3),
        ]
        edges=[
            edge(0,1,"r_cause",support=True,decisive=True),
            edge(2,1,"r_cause",support=True,decisive=True),
            edge(3,1,"r_context",irrelevant=True,reliability=0.2),
        ]
        query=f"Start at Event {a}, traverse causation forward to Event {b}, then traverse causation backward to the other causal source. Which event is reached?"
        answers=[f"Event {c} is reached.",f"Event {b} is the final source.",f"Context item {d} is reached.","The two traversal directions commute and cannot change the result."]
        correct=0
        relation_sequence=["r_cause","r_cause"]
        relation_counterfactuals=["r_context","r_context"]
        event_sequence=["CONTINUE","CONTINUE","STOP"]
        targets["traversal"]="TRAVERSAL_PATH"
        targets["direction"]="DIRECTION_BIDIRECTIONAL"
        step_targets=[
            {**targets,"direction":"DIRECTION_FORWARD"},
            {**targets,"direction":"DIRECTION_REVERSE"},
        ]
        endpoint={"active":False,"source_field":-1,"target_field":-1}
        decisive_fields=[1]
        irrelevant_fields=[3]
        support_edges=[0,1]
    elif mode == 14:
        family="causal_chain"
        # TRAIN covers ordinary 3/4-step composition. DEV deliberately
        # adds one strictly deeper public operating point so checkpoint
        # selection has real reasoning-depth extrapolation evidence. This is
        # not a product ceiling; longer programs remain runtime-shaped.
        chain_steps=(
            5
            if split=="dev"
            else (3 if example % 2 == 0 else 4)
        )
        names=unique_entity_sequence(
            entities,
            example=example,
            count=chain_steps+1,
            stride=5,
        )
        entities_used=list(names)
        fields=[
            field(
                f"Event {names[i]} is causal stage {i+1} in a verified multi-stage process.",
                "t_event",
                reliability=0.93 - 0.02*i,
            )
            for i in range(chain_steps + 1)
        ]
        fields.append(
            field(
                "A distant background context item is unrelated to the causal chain.",
                "t_context",
                reliability=0.25,
            )
        )
        edges=[]
        support_edges=[]
        for i in range(chain_steps):
            support_edges.append(len(edges))
            edges.append(
                edge(
                    i,
                    i+1,
                    "r_cause",
                    reliability=0.92 - 0.02*i,
                    support=True,
                    decisive=True,
                )
            )
        irrelevant_index=len(fields)-1
        edges.append(
            edge(
                irrelevant_index,
                0,
                "r_context",
                reliability=0.2,
                irrelevant=True,
            )
        )
        query=(
            f"Starting from Event {names[0]}, follow the causal relation exactly "
            f"{chain_steps} times in order. Which event is the terminal result?"
        )
        answers=[
            f"Event {names[chain_steps]} is the terminal result.",
            f"Event {names[1]} is the terminal result.",
            "The unrelated background context is the terminal result.",
            "The chain order may be ignored.",
        ]
        correct=0
        relation_sequence=["r_cause"] * chain_steps
        relation_counterfactuals=["r_context"] * chain_steps
        event_sequence=["CONTINUE"] * chain_steps + ["STOP"]
        targets["traversal"]="TRAVERSAL_PATH"
        step_targets=[
            {**targets,"direction":"DIRECTION_FORWARD"}
            for _ in range(chain_steps)
        ]
        endpoint={
            "active":True,
            "source_field":0,
            "target_field":chain_steps,
        }
        decisive_fields=list(range(1,chain_steps))
        irrelevant_fields=[irrelevant_index]
    elif mode in {15,16}:
        family="candidate_context_swap"
        first_is_stronger=mode==15
        first_reliability=0.97 if first_is_stronger else 0.31
        second_reliability=0.31 if first_is_stronger else 0.97
        fields=[
            field(
                f"Source {a} provides measurement evidence for Proposition {b}.",
                "t_record",
                reliability=first_reliability,
            ),
            field(
                f"Proposition {b}: the first candidate conclusion.",
                "t_claim",
            ),
            field(
                f"Source {c} provides measurement evidence for Proposition {d}.",
                "t_record",
                reliability=second_reliability,
            ),
            field(
                f"Proposition {d}: the second candidate conclusion.",
                "t_claim",
            ),
        ]
        edges=[
            edge(
                0,1,"r_support",
                reliability=first_reliability,
                support=True,
                decisive=first_is_stronger,
            ),
            edge(
                2,3,"r_support",
                reliability=second_reliability,
                support=True,
                decisive=not first_is_stronger,
            ),
        ]
        query=(
            "Two sources support different propositions. Which proposition is "
            "better supported when source reliability is the active arbitration criterion?"
        )
        answers=[
            f"Select Proposition {b} as the better-supported conclusion.",
            f"Select Proposition {d} as the better-supported conclusion.",
            "Treat both propositions as equally supported.",
            "Defer even though the evidence clearly distinguishes their support.",
        ]
        correct=0 if first_is_stronger else 1
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["traversal"]="TRAVERSAL_AGGREGATE"
        targets["reliability"]="MOD_RELIABILITY_ON"
        endpoint={"active":False,"source_field":-1,"target_field":-1}
        decisive_fields=[0] if first_is_stronger else [2]
        support_edges=[0,1]
        candidate_context_swap_pair_anchor=pair_anchor
        candidate_context_swap_variant=(
            "first_source_stronger"
            if first_is_stronger
            else "second_source_stronger"
        )
    elif mode == 17 and split=="dev":
        family="heldout_reliability_temporal_combo"
        fields=[
            field(
                f"Inspection record {a} has the highest source reliability but falls outside the required review window.",
                "t_record",reliability=0.99,temporal="outside required review window",
            ),
            field(f"Finding {b}: conclusion supported only by the out-of-window record.","t_claim"),
            field(
                f"Inspection record {c} is highly reliable and falls inside the required review window.",
                "t_record",reliability=0.93,temporal="inside required review window",
            ),
            field(f"Finding {d}: conclusion supported by the strongest in-window record.","t_claim"),
            field(
                f"A second in-window inspection note has weaker reliability and supports Finding {b}.",
                "t_record",reliability=0.55,temporal="inside required review window",
            ),
        ]
        edges=[
            edge(0,1,"r_support",reliability=0.99,temporal_match=0.0,support=True),
            edge(2,3,"r_support",reliability=0.93,temporal_match=1.0,support=True,decisive=True),
            edge(4,1,"r_support",reliability=0.55,temporal_match=1.0,support=True),
        ]
        query=(
            "Apply the required review-window constraint first. Among evidence "
            "that remains temporally admissible, use verified source reliability "
            "to choose the controlling supported finding."
        )
        answers=[
            f"Select Finding {d}; it has the strongest reliable support inside the required window.",
            f"Select Finding {b}; the highest-reliability source wins even though it is outside the window.",
            f"Select Finding {b}; any in-window source is sufficient regardless of reliability.",
            "Ignore both the time-window and reliability criteria.",
        ]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["traversal"]="TRAVERSAL_AGGREGATE"
        targets["reliability"]="MOD_RELIABILITY_ON"
        targets["temporal"]="MOD_TEMPORAL_ON"
        decisive_fields=[2]
        support_edges=[0,1,2]
        endpoint={"active":False,"source_field":-1,"target_field":-1}
    elif mode == 17:
        family="long_causal_chain"
        # FINAL stays beyond TRAIN/DEV geometry after DEV gains a deeper
        # five-step operating point. Seven is a sealed evaluation operating
        # point only, never a model or serving ceiling.
        chain_steps=7
        names=unique_entity_sequence(
            entities,
            example=example,
            count=chain_steps+1,
            stride=3,
        )
        entities_used=list(names)
        fields=[
            field(
                f"Process observation {names[i]} records verified stage {i+1} of a five-link public causal sequence.",
                "t_event",
                reliability=0.95-0.015*i,
            )
            for i in range(chain_steps+1)
        ]
        fields.append(
            field(
                "A separate public maintenance bulletin concerns an unrelated subsystem and is not part of the causal sequence.",
                "t_context",
                reliability=0.25,
            )
        )
        edges=[]
        support_edges=[]
        for i in range(chain_steps):
            support_edges.append(len(edges))
            edges.append(
                edge(
                    i,i+1,"r_cause",
                    reliability=0.94-0.015*i,
                    support=True,
                    decisive=True,
                )
            )
        irrelevant_index=len(fields)-1
        edges.append(
            edge(
                irrelevant_index,0,"r_context",
                reliability=0.2,
                irrelevant=True,
            )
        )
        query=(
            f"Beginning with process observation {names[0]}, execute all five "
            f"cause-and-effect links in their stated order. Which observation "
            "is reached only after completing the entire five-step program?"
        )
        answers=[
            f"Process observation {names[-1]} is the endpoint after all five causal steps.",
            f"Process observation {names[1]} is the final endpoint.",
            "The unrelated maintenance bulletin is the terminal causal result.",
            "The ordered five-step program can be shortened without changing the endpoint.",
        ]
        correct=0
        relation_sequence=["r_cause"]*chain_steps
        relation_counterfactuals=["r_context"]*chain_steps
        event_sequence=["CONTINUE"]*chain_steps+["STOP"]
        targets["traversal"]="TRAVERSAL_PATH"
        step_targets=[
            {**targets,"direction":"DIRECTION_FORWARD"}
            for _ in range(chain_steps)
        ]
        endpoint={"active":True,"source_field":0,"target_field":chain_steps}
        decisive_fields=list(range(1,chain_steps))
        irrelevant_fields=[irrelevant_index]
    elif mode == 18:
        family="heldout_recency_provenance_combo"
        fields=[
            field(
                f"Authorized memorandum {a} supports Proposition {b}, but it is the older authorized source.",
                "t_record",
                reliability=0.94,
                recency=0.35,
                provenance="authorized audited public source",
            ),
            field(f"Proposition {b}: conclusion from the older authorized source.","t_claim"),
            field(
                f"External bulletin {c} supports a competing conclusion and is newest overall, but its provenance is outside the authorized class.",
                "t_record",
                reliability=0.96,
                recency=0.99,
                provenance="outside authorized public provenance class",
            ),
            field(f"Proposition {c}: conclusion from the unauthorized newest source.","t_claim"),
            field(
                f"Authorized memorandum {d} supports the controlling conclusion and is newer than the other authorized memorandum.",
                "t_record",
                reliability=0.95,
                recency=0.82,
                provenance="authorized audited public source",
            ),
            field(f"Proposition {d}: conclusion from the newest authorized source.","t_claim"),
        ]
        edges=[
            edge(0,1,"r_support",recency=0.35,provenance_match=1.0,support=True),
            edge(2,3,"r_support",recency=0.99,provenance_match=0.0,support=True),
            edge(4,5,"r_support",recency=0.82,provenance_match=1.0,support=True,decisive=True),
        ]
        query=(
            "Several sources support competing propositions. Restrict evidence "
            "to the authorized provenance class and, within that class, use "
            "recency to identify the controlling supported conclusion."
        )
        answers=[
            f"Select Proposition {d}; it is supported by the newest authorized source.",
            f"Select Proposition {c}; it is newest overall even though its provenance is unauthorized.",
            f"Select Proposition {b}; it is authorized but older than the other authorized evidence.",
            "Ignore both provenance and recency and choose an arbitrary candidate.",
        ]
        correct=0
        relation_sequence=["r_support"]
        relation_counterfactuals=["r_context"]
        event_sequence=["CONTINUE","STOP"]
        targets["traversal"]="TRAVERSAL_AGGREGATE"
        targets["recency"]="MOD_RECENCY_ON"
        targets["provenance"]="MOD_PROVENANCE_ON"
        decisive_fields=[4]
        support_edges=[0,1,2]
        endpoint={"active":False,"source_field":-1,"target_field":-1}
    else:
        raise ValueError(f"unsupported behavioral scenario mode: {mode}")

    if not step_targets and relation_sequence:
        step_targets=[dict(targets) for _ in relation_sequence]
    return {
        "scenario_family":family,
        "entities_used":entities_used,
        "query":query,
        "fields":fields,
        "edges":edges,
        "answers":answers,
        "correct_answer_index":correct,
        "relation_sequence_keys":relation_sequence,
        "counterfactual_relation_sequence_keys":relation_counterfactuals,
        "event_sequence_target":event_sequence,
        "factor_target_opcodes":targets,
        "counterfactual_factor_opcodes":factor_counterfactuals(targets),
        "step_factor_target_opcodes":step_targets,
        "applicability_target":0.0 if family=="unknown_defer" else 1.0,
        "uncertainty_target":uncertainty,
        "endpoint_target":endpoint,
        "decisive_field_indices":decisive_fields,
        "irrelevant_field_indices":irrelevant_fields,
        "support_edge_indices":support_edges,
        "recoverable_view_names":recoverable,
        "candidate_context_swap_pair_anchor":candidate_context_swap_pair_anchor,
        "candidate_context_swap_variant":candidate_context_swap_variant,
    }


DEV_QUERY_PARAPHRASES = {
    "source_target_role":"Using the verified evidence relation in the records, identify the conclusion that receives direct support.",
    "ordered_composition":"Trace the two causal links from the stated starting event in sequence and report the terminal event rather than the intermediate one.",
    "reverse_traversal":"Begin with the named author, invert the authorship relation, and identify the document at the opposite endpoint.",
    "symmetric_relation":"Identify the other record joined by mutual corroboration; stored endpoint order must not decide the answer.",
    "reliability_arbitration":"Competing conclusions have supporting sources of different verification quality. Select the conclusion backed by the stronger reliable evidence.",
    "recency_supersession":"A later signed correction replaces an earlier version. Identify the record that has controlling status after that replacement.",
    "temporal_constraint":"Only causal evidence inside the required temporal scope is admissible. Identify the effect supported by an in-scope link.",
    "provenance_constraint":"Only evidence from the authorized provenance class is admissible. Identify the conclusion supported under that restriction.",
    "conflict_plurality":"Two independently supported propositions remain mutually inconsistent and no resolving evidence is present. State the appropriate unresolved interpretation.",
    "unknown_defer":"The requested relationship has no matching description in the runtime schema. Choose the response that avoids substituting a merely similar relation.",
    "decisive_source":"One verified measurement is the sole evidence that establishes the requested conclusion. Identify the conclusion that depends on that decisive source.",
    "irrelevant_distractor":"Newer contextual material is unrelated to the technical question. Identify the conclusion that remains supported when irrelevant context is ignored.",
    "multiple_co_valid_support":"Two independent reliable records converge on one conclusion. Choose the response that preserves both as co-valid supporting evidence.",
    "mixed_direction_composition":"Follow the first causal edge in its stored direction, then invert the second causal edge at the shared intermediate event. Identify the endpoint reached after both operations.",
    "causal_chain":"Trace every causal stage in the stated order from the initial event to the terminal effect; intermediate stages and unrelated context are not the requested endpoint.",
    "candidate_context_swap":"Two evidence sources back different propositions. Select the proposition whose support is stronger under the active reliability criterion.",
    "heldout_reliability_temporal_combo":"Apply the review-window requirement, then compare reliability among the remaining admissible evidence and select the controlling finding.",
}

FINAL_QUERY_PARAPHRASES = {
    "source_target_role":"Within this held-out public case, determine which proposition receives direct evidential backing from the validated measurement source.",
    "ordered_composition":"Execute the two cause-and-effect transitions in sequence and return only the endpoint reached after the second transition.",
    "reverse_traversal":"Starting from the identified author, traverse the authorship relation against its stored direction and name the document reached.",
    "symmetric_relation":"Find the peer source connected by mutual corroboration while treating stored endpoint orientation as irrelevant.",
    "reliability_arbitration":"When competing propositions have differently verified evidence, choose the proposition supported by the more dependable source.",
    "recency_supersession":"A signed revision replaces an earlier public record. Determine which version governs after the replacement.",
    "temporal_constraint":"Apply the stated time-scope restriction to the causal evidence and identify the admissible effect.",
    "provenance_constraint":"Apply the authorized-source restriction and identify the proposition that remains supported.",
    "conflict_plurality":"The evidence sustains incompatible propositions without resolving them. Preserve the unresolved interpretation rather than forcing one side.",
    "unknown_defer":"No runtime relation description represents the requested relationship. Select the response that preserves uncertainty instead of inventing a mapping.",
    "decisive_source":"Only one validated measurement establishes the requested conclusion. Identify the conclusion whose justification depends on that source.",
    "irrelevant_distractor":"Separate evidence bearing on the technical question from newer background material and identify the conclusion that remains justified.",
    "multiple_co_valid_support":"Independent sources converge on one proposition. Preserve both sources as valid support rather than discarding one without cause.",
    "mixed_direction_composition":"Traverse the first causal link forward and the second causal link backward from the shared intermediate state, then report the resulting endpoint.",
    "causal_chain":"Follow the entire ordered causal program from its initial observation to the terminal state and ignore unrelated background material.",
    "candidate_context_swap":"Use the active reliability criterion to compare the two evidence-backed propositions and select the one with stronger support.",
    "heldout_reliability_temporal_combo":"Enforce the stated review window, then compare the reliability of the remaining admissible evidence and choose the supported finding that controls.",
    "long_causal_chain":"Execute the complete five-step causal program and identify the endpoint that appears only after every transition has been applied.",
    "heldout_recency_provenance_combo":"First enforce the authorized-provenance requirement, then use recency among the remaining eligible sources to select the controlling proposition.",
}

FINAL_QUERY_SECONDARY_REWRITES = (
    ("determine", "identify"),
    ("execute", "carry out"),
    ("find", "locate"),
    ("choose", "select"),
    ("apply", "enforce"),
    ("preserve", "retain"),
    ("separate", "distinguish"),
    ("follow", "trace"),
    ("starting from", "beginning at"),
    ("starting at", "beginning at"),
    ("within this held-out public case", "for this independent public evaluation"),
    ("when competing propositions", "when the candidate propositions conflict"),
    ("first enforce", "begin by enforcing"),
    ("then use", "afterward use"),
    ("identify", "name"),
    ("which", "what"),
)


def final_query_secondary_paraphrase(text: str) -> str:
    """Second frozen FINAL-only query wording for paired paraphrase evaluation."""
    value=str(text)
    lower=value.lower()
    changed=False
    for source,replacement in FINAL_QUERY_SECONDARY_REWRITES:
        source_lower=source.lower()
        index=lower.find(source_lower)
        if index < 0:
            continue
        value=value[:index]+replacement+value[index+len(source):]
        lower=value.lower()
        changed=True
    if not changed or value==text:
        value="Answer the same held-out semantic task in equivalent wording: "+str(text)
    return value


FINAL_SURFACE_REWRITES = (
    ("provides evidence for", "supplies documented support for"),
    ("supports", "backs with evidence"),
    ("supported", "justified by the evidence"),
    ("verified", "independently validated"),
    ("record", "memorandum"),
    ("claim", "proposition"),
    ("context item", "background material"),
    ("context note", "background memorandum"),
    ("unrelated", "not germane"),
    ("causal relation", "cause-and-effect transition"),
    ("causal sequence", "ordered process sequence"),
    ("causal stage", "process transition"),
    ("causal chain", "multi-stage process"),
    ("event", "process observation"),
    ("actor", "participant"),
    ("is reached last", "is obtained after the complete sequence"),
    ("is reached", "is obtained"),
    ("select proposition", "choose proposition"),
    ("use unrelated option", "choose a non-germane alternative"),
    ("candidate position", "response position"),
)

def final_surface_paraphrase(text: str) -> str:
    """Deterministic FINAL-only lexical variant with no split marker in semantics."""
    value=str(text)
    lower=value.lower()
    for source,replacement in FINAL_SURFACE_REWRITES:
        start=0
        source_lower=source.lower()
        while True:
            index=lower.find(source_lower,start)
            if index < 0:
                break
            value=value[:index]+replacement+value[index+len(source):]
            lower=value.lower()
            start=index+len(replacement)
    return value


DEV_SURFACE_REWRITES = (
    ("provides measurement evidence for", "contains measured support for"),
    ("the first candidate conclusion", "the first proposed conclusion"),
    ("the second candidate conclusion", "the second proposed conclusion"),
    ("select proposition", "choose proposition"),
    ("as the better-supported conclusion.", "as the conclusion with stronger evidence."),
    ("treat both propositions as equally supported.", "regard both propositions as equally backed."),
    ("defer even though the evidence clearly distinguishes their support.", "abstain despite evidence that clearly separates their support."),
    ("defer because the requested relation is absent from the supplied runtime schema.", "Abstain because no runtime relation description represents the requested relationship."),
    ("force the evidential-support relation as a substitute.", "Do not substitute evidence-support semantics for a relationship absent from the schema."),
    ("use contextual relatedness as if it meant licensing.", "Do not reinterpret generic contextual association as the missing licensing relationship."),
    ("the supported conclusion is", "the evidence-backed conclusion is"),
    ("the terminal result", "the endpoint after the full chain"),
    ("is reached last", "is the final endpoint"),
    ("is reached", "is the endpoint obtained"),
    ("mutually corroborates", "independently confirms"),
    ("corroborates it", "confirms the same observation"),
    ("verified high-reliability source", "validated source with stronger reliability"),
    ("weak unverified source", "lower-confidence unverified source"),
    ("signed later correction", "later signed correction"),
    ("supersedes the older record", "replaces the earlier document as controlling evidence"),
    ("older version", "earlier version"),
    ("newer than", "more recent than"),
    ("verified measurement", "validated measurement"),
    ("verified report", "validated report"),
    ("verified source", "validated source"),
    ("verified record", "validated record"),
    ("verified decisive primary record", "validated decisive primary document"),
    ("authorized record", "record from the allowed provenance class"),
    ("unverified record", "record outside the verified provenance class"),
    ("independent record", "separate record"),
    ("independent reliable records", "separate reliable records"),
    ("supports claim", "provides evidence for proposition"),
    ("supports a competing claim", "provides evidence for a competing proposition"),
    ("supports an ordinary claim", "provides evidence for an ordinary proposition"),
    ("supports the same conclusion", "provides separate evidence for the same conclusion"),
    ("supports", "provides evidence for"),
    ("supported", "backed by the evidence"),
    ("claim", "proposition"),
    ("record", "document"),
    ("context item", "background item"),
    ("context note", "background note"),
    ("unrelated", "not pertinent"),
    ("causal relation", "cause-and-effect link"),
    ("causal sequence", "ordered cause-and-effect sequence"),
    ("causal stage", "cause-and-effect stage"),
    ("causal chain", "ordered cause-and-effect chain"),
    ("authored-by relation", "authorship link"),
    ("actor", "person"),
    ("event", "process event"),
    ("use unrelated option", "select irrelevant alternative"),
    ("candidate position", "answer position"),
)


def dev_surface_paraphrase(text: str) -> str:
    """Deterministic DEV-only lexical variant for fields and answer candidates."""
    value=str(text)
    lower=value.lower()
    # Rewrite longest semantic phrases first while preserving entity tokens and
    # punctuation. This is deliberately lexical only: graph/schema semantics
    # remain identical so DEV isolates transfer across surface wording.
    for source,replacement in DEV_SURFACE_REWRITES:
        start=0
        source_lower=source.lower()
        while True:
            index=lower.find(source_lower,start)
            if index < 0:
                break
            value=(
                value[:index]
                + replacement
                + value[index+len(source):]
            )
            lower=value.lower()
            start=index+len(replacement)
    return value


def template_signature(query: str, entities: list[str]) -> str:
    normalized=query.lower()
    for entity in sorted(entities,key=len,reverse=True):
        normalized=normalized.replace(entity.lower(),"<entity>")
    normalized=" ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def add_distractors(row: dict[str,Any], desired_fields: int, entities: list[str]) -> None:
    while len(row["fields"]) < desired_fields:
        index=len(row["fields"])
        name=entities[(index*7+3) % len(entities)]
        row["fields"].append(
            field(
                f"Distractor context {name} contains public background information unrelated to the requested reasoning chain.",
                "t_context",
                confidence=0.5,
                reliability=0.35,
                recency=0.8,
                provenance="ordinary unrelated public context",
            )
        )
        if len(row["fields"]) > 1:
            row["edges"].append(
                edge(
                    index,
                    max(0,index-1),
                    "r_context",
                    reliability=0.2,
                    recency=0.8,
                    irrelevant=True,
                )
            )
        row["irrelevant_field_indices"].append(index)


def materialize_row(
    *,
    split: str,
    example: int,
    seed: int,
    relation_count: int,
    field_count: int,
    answer_count: int,
) -> dict[str,Any]:
    if split not in {"train","dev","final"}:
        raise ValueError("behavioral split must be train/dev/final")
    entities={
        "train":TRAIN_ENTITIES,
        "dev":DEV_ENTITIES,
        "final":FINAL_ENTITIES,
    }[split]
    mode=example % (19 if split=="final" else (18 if split=="dev" else 17))
    base=scenario(mode,entities,example,split=split)
    if split=="dev":
        base["query"]=DEV_QUERY_PARAPHRASES[base["scenario_family"]]
    elif split=="final":
        base["query"]=FINAL_QUERY_PARAPHRASES[base["scenario_family"]]
        base["final_paraphrase_query"]=final_query_secondary_paraphrase(base["query"])
    add_distractors(base,max(field_count,len(base["fields"])),entities)
    if split=="dev":
        for item in base["fields"]:
            item["text"]=dev_surface_paraphrase(item["text"])
        base["answers"]=[
            dev_surface_paraphrase(text)
            for text in base["answers"]
        ]
    elif split=="final":
        for item in base["fields"]:
            item["text"]=final_surface_paraphrase(item["text"])
        base["answers"]=[
            final_surface_paraphrase(text)
            for text in base["answers"]
        ]
        if example % 4 == 0:
            long_context=(
                " Additional public context records calibration history, procedural "
                "background, noncontrolling observations, and provenance notes that "
                "must remain available without displacing the explicitly relevant evidence."
            )
            base["query"] += long_context * 10
            if base["fields"]:
                base["fields"][-1]["text"] += long_context * 8
        if example % 5 == 0:
            base["answers"]=[
                text + (
                    " The comparison must still be made from the governed evidence "
                    "state rather than from response length or wording."
                ) * 6
                for text in base["answers"]
            ]
    randomization_example=example-1 if mode==16 else example
    split_offset={"train":0,"dev":10_000_000,"final":20_000_000}[split]
    rng=random.Random(
        seed
        + randomization_example*1009
        + split_offset
    )

    required_relations=list(dict.fromkeys(
        base["relation_sequence_keys"]
        + [e["relation_key"] for e in base["edges"]]
    ))
    relation_by_key={r["key"]:r for r in RELATIONS}
    if any(key not in relation_by_key for key in required_relations):
        raise RuntimeError("scenario relation missing from runtime pool")
    desired=max(relation_count,len(required_relations))
    pool=[r["key"] for r in RELATIONS if r["key"] not in required_relations]
    rng.shuffle(pool)
    relation_keys=required_relations + pool[:max(0,desired-len(required_relations))]
    rng.shuffle(relation_keys)
    # Runtime relation candidates belong to the row. Do not return aliases
    # into the module-level public relation bank: downstream curriculum
    # interventions may safely rewrite semantic surface text and must never
    # contaminate later TRAIN/DEV/FINAL materialization.
    relation_candidates=[
        copy.deepcopy(relation_by_key[key]) for key in relation_keys
    ]

    answers=list(base["answers"])
    while len(answers)<answer_count:
        answers.append(
            f"Use unrelated option {len(answers)+1} because it happens to occupy a candidate position."
        )
    answers=answers[:max(answer_count,2)]
    correct_text=base["answers"][base["correct_answer_index"]]
    rng.shuffle(answers)
    public_target=answers.index(correct_text)

    field_perm=list(range(len(base["fields"])))
    rng.shuffle(field_perm)
    if field_perm == list(range(len(field_perm))) and len(field_perm)>1:
        field_perm=field_perm[1:]+field_perm[:1]

    template_id=f"{split}:behavioral-template:{mode}"
    causal_group=f"{split}:causal:{mode}:{example:05d}"
    type_schema=(
        [dict(x) for x in FINAL_TYPE_SCHEMA]
        if split=="final"
        else (
            [dict(x) for x in DEV_TYPE_SCHEMA]
            if split=="dev"
            else [dict(x) for x in TYPE_SCHEMA]
        )
    )
    factor_schemas={
        name:[dict(item) for item in bank]
        for name,bank in FACTOR_BANKS.items()
    }
    if split=="dev":
        factor_schemas["open_semantic_factor"].append(
            {
                "key":"OPEN_DEV_4",
                "opcode":None,
                "text":"a public semantic qualifier marks the conclusion as contingent on an unresolved procedural dependency",
            }
        )
    elif split=="final":
        factor_schemas["open_semantic_factor"].extend(
            [
                {
                    "key":"OPEN_FINAL_4",
                    "opcode":None,
                    "text":"a held-out public semantic qualifier indicates that evidence remains conditional on an unresolved operational dependency",
                },
                {
                    "key":"OPEN_FINAL_5",
                    "opcode":None,
                    "text":"a second held-out public qualifier marks an independent unresolved condition without changing the executable relation program",
                },
            ]
        )
    semantic_text=" ".join(
        [base["query"]]
        + [x["text"] for x in relation_candidates]
        + [x["text"] for bank in factor_schemas.values() for x in bank]
        + [x["text"] for x in base["fields"]]
        + answers
        + [x["text"] for x in type_schema]
    ).lower()
    opaque_keys=[x["key"] for x in type_schema] + [x["key"] for x in RELATIONS]
    if any(key.lower() in semantic_text for key in opaque_keys):
        raise RuntimeError("opaque metadata key leaked into semantic text")

    # Binder supervision represents query-relevant structural/semantic
    # support before step-local evidence-quality arbitration. Relation-key
    # membership is necessary but not sufficient: an unrelated edge can carry
    # the same runtime relation and must remain a hard negative. Preserve the
    # scenario's explicit support annotations rather than reconstructing them
    # from relation identity.
    structural_relation_keys=set(base["relation_sequence_keys"])
    support_set={int(i) for i in base["support_edge_indices"]}
    if any(i < 0 or i >= len(base["edges"]) for i in support_set):
        raise RuntimeError("scenario support edge outside runtime edge bank")
    if any(
        base["edges"][i]["relation_key"] not in structural_relation_keys
        for i in support_set
    ):
        raise RuntimeError(
            "scenario support edge relation absent from selected relation program"
        )
    base["support_edge_indices"]=sorted(support_set)
    decisive_set=set(base["decisive_field_indices"])
    irrelevant_set=set(base["irrelevant_field_indices"])
    for i,e in enumerate(base["edges"]):
        e["support_target"]=i in support_set
        e["decisive"]=e["decisive"] or (
            e["source_field"] in decisive_set or e["target_field"] in decisive_set
        )
        e["irrelevant"]=e["irrelevant"] or (
            e["source_field"] in irrelevant_set and e["target_field"] in irrelevant_set
        )

    relation_index={key:i for i,key in enumerate(relation_keys)}
    relation_targets=[
        relation_index[key] for key in base["relation_sequence_keys"]
    ]
    relation_counter_targets=[
        relation_index[key]
        if key is not None and key in relation_index and key != base["relation_sequence_keys"][i]
        else -1
        for i,key in enumerate(base["counterfactual_relation_sequence_keys"])
    ]

    return {
        "schema":ROW_SCHEMA,
        "id":f"feb_{split}_{example:06d}_{base['scenario_family']}",
        "split":split,
        "lane":"full_envelope_behavioral_fabric",
        "scenario_family":base["scenario_family"],
        "candidate_context_swap_pair_id":(
            f"{split}:candidate-context-swap:{int(base['candidate_context_swap_pair_anchor']):06d}"
            if base.get("candidate_context_swap_pair_anchor") is not None
            else None
        ),
        "candidate_context_swap_variant":base.get(
            "candidate_context_swap_variant"
        ),
        "entities":base["entities_used"],
        "template_id":template_id,
        "template_signature_sha256":template_signature(
            base["query"],
            base["entities_used"],
        ),
        "causal_group":causal_group,
        "query":base["query"],
        "final_paraphrase_query":base.get("final_paraphrase_query"),
        "type_schema":type_schema,
        "relation_candidates":relation_candidates,
        "factor_schemas":factor_schemas,
        "fields":base["fields"],
        "edges":base["edges"],
        "candidate_answers":answers,
        "public_target_index":public_target,
        "relation_sequence_target":relation_targets,
        "counterfactual_relation_sequence_target":relation_counter_targets,
        "event_sequence_target":base["event_sequence_target"],
        "factor_target_keys":base["factor_target_opcodes"],
        "counterfactual_factor_keys":base["counterfactual_factor_opcodes"],
        "step_factor_target_keys":base["step_factor_target_opcodes"],
        "applicability_target":base["applicability_target"],
        "uncertainty_target":base["uncertainty_target"],
        "support_edge_indices":[
            i for i,e in enumerate(base["edges"]) if e["support_target"]
        ],
        "endpoint_target":base["endpoint_target"],
        "decisive_field_indices":sorted(set(base["decisive_field_indices"])),
        "irrelevant_field_indices":sorted(set(base["irrelevant_field_indices"])),
        "decisive_view_active":bool(base["decisive_field_indices"]),
        "irrelevant_view_active":bool(base["irrelevant_field_indices"]),
        "recoverable_view_names":base["recoverable_view_names"],
        "field_permutation":field_perm,
        "runtime_relation_count":len(relation_candidates),
        "runtime_field_count":len(base["fields"]),
        "runtime_edge_count":len(base["edges"]),
        "runtime_reasoning_steps":len(relation_targets),
        "runtime_candidate_answer_count":len(answers),
        "candidate_cardinality_extrapolation":(
            split=="dev" and len(answers)>6
        ),
        "composition_extrapolation":(
            split=="dev" and len(relation_targets)>4
        ),
        "factor_cardinality_extrapolation":(
            split=="dev"
            and len(factor_schemas["open_semantic_factor"])>len(FACTOR_BANKS["open_semantic_factor"])
        ),
        "factor_combination_transfer":(
            split=="dev" and base["scenario_family"]=="heldout_reliability_temporal_combo"
        ),
        "type_schema_transfer":split=="dev",
        "domain_transfer":split=="dev",
        "dev_extrapolation_operating_point_is_capability_ceiling":False,
        "generated_text":True,
        "data_origin":("deterministic_public_full_envelope_final_v2" if split=="final" else "deterministic_public_full_envelope_behavioral_fabric_v1"),
        "private_identity_data":False,
        "relation_keys_are_metadata_only":True,
        "type_keys_are_metadata_only":True,
        "training_authorized":split=="train",
        "model_selection_authorized":split=="dev",
        "final_validation_only":split=="final",
        "domain_family":(
            "heldout_public_operations_and_instrumentation"
            if split=="final"
            else (
                "heldout_public_quality_and_logistics"
                if split=="dev"
                else "public_synthetic_relational_reasoning"
            )
        ),
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--train-rows",type=int,default=280)
    p.add_argument("--dev-rows",type=int,default=112)
    p.add_argument("--seed",type=int,default=20260922)
    args=p.parse_args()
    if args.train_rows<=0 or args.dev_rows<=0:
        raise SystemExit("train/dev row counts must be positive")

    relation_points=[1,2,4,6,8]
    field_points=[4,6,8,12,16]
    train_answer_points=[2,3,4,6]
    # DEV must test unseen candidate geometry rather than only repeat TRAIN.
    # Seven is a public evaluation operating point, not a candidate ceiling.
    dev_answer_points=[3,4,5,7]
    rows=[]
    for split,count in (("train",args.train_rows),("dev",args.dev_rows)):
        answer_points=(
            train_answer_points if split=="train" else dev_answer_points
        )
        for i in range(count):
            # Candidate-context-swap pairs live at modes 15/16 inside the
            # split's actual scenario cycle. DEV has one extra transfer-only
            # mode, so using the historical 17-mode cycle here silently gave
            # later DEV pair members different relation/field/answer geometry.
            # Anchor operating-point selection to the same split-aware cycle
            # used by materialize_row so each causal pair differs only in the
            # intended governing evidence/reliability state.
            cycle=18 if split=="dev" else 17
            mode=i % cycle
            axis_i=i-1 if mode==16 else i
            rows.append(
                materialize_row(
                    split=split,
                    example=i,
                    seed=args.seed,
                    relation_count=relation_points[axis_i % len(relation_points)],
                    field_count=field_points[
                        (axis_i//len(relation_points)) % len(field_points)
                    ],
                    answer_count=answer_points[
                        (axis_i//3) % len(answer_points)
                    ],
                )
            )

    output=Path(args.output)
    manifest_path=Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite behavioral curriculum artifact")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        "".join(json.dumps(row,sort_keys=True)+"\n" for row in rows),
        encoding="utf-8",
    )
    train=[x for x in rows if x["split"]=="train"]
    dev=[x for x in rows if x["split"]=="dev"]
    manifest={
        "schema":MANIFEST_SCHEMA,
        "status":"MATERIALIZED_PUBLIC_FULL_ENVELOPE_BEHAVIORAL_FABRIC",
        "sha256":sha256(output),
        "rows":len(rows),
        "train_rows":len(train),
        "dev_rows":len(dev),
        "scenario_histogram":dict(sorted(Counter(x["scenario_family"] for x in rows).items())),
        "train_scenario_histogram":dict(sorted(Counter(x["scenario_family"] for x in train).items())),
        "dev_scenario_histogram":dict(sorted(Counter(x["scenario_family"] for x in dev).items())),
        "relation_count_points":sorted({x["runtime_relation_count"] for x in rows}),
        "field_count_points":sorted({x["runtime_field_count"] for x in rows}),
        "edge_count_points":sorted({x["runtime_edge_count"] for x in rows}),
        "reasoning_step_points":sorted({x["runtime_reasoning_steps"] for x in rows}),
        "answer_count_points":sorted({x["runtime_candidate_answer_count"] for x in rows}),
        "train_templates":sorted({x["template_id"] for x in train}),
        "dev_templates":sorted({x["template_id"] for x in dev}),
        "train_causal_groups":len({x["causal_group"] for x in train}),
        "dev_causal_groups":len({x["causal_group"] for x in dev}),
        "private_identity_data":False,
        "is_complete_training_corpus":False,
        "row_count_is_capability_ceiling":False,
        "runtime_axis_points_are_capability_ceilings":False,
    }
    manifest_path.write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
