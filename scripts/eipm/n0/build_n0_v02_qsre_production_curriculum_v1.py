from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SCHEMA = "alice.eipm.n0.qsre-production-public-row.v1"
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

ROLE = {"SOURCE": 0, "TARGET": 1, "SYMMETRIC": 2, "NONE": 3}
TRAVERSAL = {"LOCAL_SELECT": 0, "PATH": 1, "AGGREGATE": 2}
CONTROL = {"FALLBACK": 0, "RELATIONAL": 1, "DEFER": 2}
MODIFIERS = (
    "RELIABILITY",
    "RECENCY",
    "TEMPORAL_CONSTRAINT",
    "PROVENANCE_CONSTRAINT",
)

TYPE_TEXT = {
    "GENERIC": "general record",
    "EVIDENCE": "observation log",
    "CLAIM": "analysis note",
    "CORRECTION": "signed amendment",
    "POLICY": "policy document",
    "ARTIFACT": "processed artifact",
    "SOURCE": "source archive",
    "EVENT": "event report",
    "VERSION": "release record",
    "PRINCIPLE": "principle note",
    "EXAMPLE": "case example",
    "CONDITION": "condition record",
    "ACTION": "action record",
    "COMPONENT": "component record",
    "WHOLE": "assembly record",
    "RECORD": "official record",
}

RELATION_TYPES = {
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
}

PHRASES = {
    "SUPPORTS": {
        "train": [
            "backs the conclusion",
            "provides evidence for the finding",
            "makes the claim more credible",
            "corroborates the interpretation",
        ],
        "dev": [
            "gives independent reason to trust the conclusion",
            "strengthens the case for the finding",
            "serves as evidence behind the claim",
        ],
    },
    "CORRECTS": {
        "train": [
            "repairs the error in",
            "fixes the inaccurate detail in",
            "amends the mistaken statement in",
            "supplies the correction to",
        ],
        "dev": [
            "sets right the wrong detail in",
            "revises an error contained in",
            "provides the erratum for",
        ],
    },
    "SUPERSEDES": {
        "train": [
            "takes precedence over",
            "replaces as the governing version",
            "becomes authoritative instead of",
            "displaces the older version",
        ],
        "dev": [
            "is now controlling in place of",
            "is the successor authority to",
            "renders the older version obsolete",
        ],
    },
    "DERIVED_FROM": {
        "train": [
            "was produced using",
            "was computed from",
            "came from the source material in",
            "uses as its provenance base",
        ],
        "dev": [
            "was generated out of",
            "traces its source material to",
            "was built from the contents of",
        ],
    },
    "CAUSES": {
        "train": [
            "brings about",
            "leads to",
            "produces the resulting event",
            "is the initiating condition for",
        ],
        "dev": [
            "sets off",
            "is responsible for the occurrence of",
            "results in",
        ],
    },
    "TEMPORAL_SUCCESSOR": {
        "train": [
            "comes immediately after",
            "follows in the chronology",
            "is later than",
            "is the next state after",
        ],
        "dev": [
            "appears subsequently to",
            "occupies the later position after",
            "is the chronological successor of",
        ],
    },
    "CONFLICTS_WITH": {
        "dev": [
            "cannot be reconciled with",
            "makes an incompatible claim with",
            "contradicts the account in",
        ],
    },
    "EXEMPLIFIES": {
        "dev": [
            "is a concrete instance of",
            "illustrates the broader principle in",
            "serves as an example of",
        ],
    },
    "PREREQUISITE_FOR": {
        "dev": [
            "must be satisfied before",
            "is required in order for",
            "has to hold before proceeding with",
        ],
    },
    "PART_OF": {
        "dev": [
            "is a component of",
            "belongs inside the larger whole",
            "is one constituent of",
        ],
    },
}

TRAIN_NAMES = [
    "Orion", "Cedar", "Lumen", "Harbor", "Nadir", "Mosaic", "Kestrel",
    "Quartz", "Atlas", "Juniper", "Nimbus", "Solace", "Tundra", "Vela",
    "Copper", "Maple", "Falcon", "Iris", "Delta", "Aster",
]
DEV_NAMES = [
    "Meridian", "Willow", "Pioneer", "Helix", "Saffron", "Beacon", "Lyric",
    "Summit", "Cobalt", "Ember", "Ridge", "Nova", "Mariner", "Opal",
    "Fjord", "Tempo", "Sterling", "Echo", "Boreal", "Canyon",
]


def stable_rng(*parts: object) -> random.Random:
    raw = "|".join(str(value) for value in parts)
    seed = int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def field_text(entity: str, type_name: str, detail: str) -> str:
    return (
        f"{entity} {TYPE_TEXT[type_name]}. "
        f"The record concerns {detail}. "
        f"Identifier {hashlib.sha256((entity + detail).encode()).hexdigest()[:8]}."
    )


def make_fields(split: str, family: str, group: int, types: list[str]) -> list[dict]:
    names = TRAIN_NAMES if split == "train" else DEV_NAMES
    rng = stable_rng(split, family, group, "fields")
    chosen = rng.sample(names, k=6)
    details = [
        "pump temperature during the morning inspection",
        "a signed release decision for the current procedure",
        "the source readings used in a later analysis",
        "an incident timeline around the service interruption",
        "a calibration issue reported by the audit team",
        "a component change recorded during assembly",
    ]
    rng.shuffle(details)
    fields = []
    for index in range(6):
        type_name = types[index] if index < len(types) else "GENERIC"
        fields.append(
            {
                "id": f"f_{split}_{family}_{group:04d}_{index}",
                "entity": chosen[index],
                "type": type_name,
                "text": field_text(chosen[index], type_name, details[index]),
                "metadata": {
                    "reliability": 0.5,
                    "normalized_time": 0.5,
                    "temporal_match": 1.0,
                    "provenance_match": 1.0,
                },
            }
        )
    return fields


def edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    *,
    reliability: float = 0.5,
    recency: float = 0.5,
    temporal_match: float = 1.0,
    provenance_match: float = 1.0,
) -> dict:
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


def relation_phrase(relation: str, split: str, key: str) -> str:
    bank = PHRASES[relation][split]
    rng = stable_rng(split, relation, key)
    return bank[rng.randrange(len(bank))]


def modifiers(**kwargs: bool) -> dict[str, float]:
    out = {name: 0.0 for name in MODIFIERS}
    for key, value in kwargs.items():
        name = key.upper()
        if name not in out:
            raise KeyError(name)
        out[name] = 1.0 if value else 0.0
    return out



def alternate_query(text: str) -> str:
    replacements = [
        ("Which record", "Identify the record"),
        ("Which endpoint", "Identify the endpoint"),
        ("Which receiving record", "Identify the receiving record"),
        ("Which record or records", "Identify the record or records"),
        ("Which records", "Identify the records"),
        ("Where does", "At which record does"),
        ("Start at", "Begin with"),
        ("Starting at", "Beginning with"),
        ("Beginning with", "Starting from"),
        ("follow", "trace"),
        ("Follow", "Trace"),
        ("Return", "Give"),
        ("Preserve", "Keep"),
        ("use the relationship", "apply the relationship"),
        ("Use the relationship", "Apply the relationship"),
        ("Prefer", "Choose according to"),
        ("latest applicable state", "most recent applicable state"),
        ("stronger provenance and verification record", "better verified and better sourced record"),
    ]
    out = text
    for old, new in replacements:
        if old in out:
            out = out.replace(old, new, 1)
    if out == text:
        out = "Using the same evidence and requested semantics, " + text[0].lower() + text[1:]
    return out

def make_row(
    *,
    split: str,
    family: str,
    group: int,
    variant: str,
    fields: list[dict],
    edges: list[dict],
    query: str,
    relation_sequence: list[str],
    role: str,
    traversal: str,
    modifier_target: dict[str, float],
    focus_field_id: str | None,
    support_edge_ids: list[str],
    target_distribution: dict[str, float],
    applicability: float = 0.9,
    control: str = "RELATIONAL",
    open_schema: bool = False,
) -> dict:
    return {
        "schema": SCHEMA,
        "id": f"prod_{split}_{family}_{group:04d}_{variant}",
        "split": split,
        "family": family,
        "causal_group": f"{split}:{family}:{group:04d}",
        "variant": variant,
        "query": query,
        "query_views": [query, alternate_query(query)],
        "fields": fields,
        "edges": edges,
        "operator_target": {
            "relation_sequence": relation_sequence,
            "role_id": ROLE[role],
            "traversal_id": TRAVERSAL[traversal],
            "modifier_target": modifier_target,
            "focus_field_id": focus_field_id,
            "applicability": float(applicability),
            "control_id": CONTROL[control],
        },
        "support_edge_ids": support_edge_ids,
        "target_distribution": target_distribution,
        "open_schema_relation": bool(open_schema),
        "private_identity_data": False,
    }


def pair_rows(
    *,
    split: str,
    family: str,
    group: int,
    relation_pool: tuple[str, ...],
) -> list[dict]:
    rng = stable_rng(split, family, group)
    r1 = relation_pool[rng.randrange(len(relation_pool))]
    r2 = relation_pool[(relation_pool.index(r1) + 1 + rng.randrange(len(relation_pool) - 1)) % len(relation_pool)]
    src_type, tgt_type = RELATION_TYPES[r1]
    fields = make_fields(
        split,
        family,
        group,
        [src_type, tgt_type, RELATION_TYPES[r2][0], RELATION_TYPES[r2][1], "GENERIC", "GENERIC"],
    )
    ids = [row["id"] for row in fields]
    names = [row["entity"] for row in fields]
    p1 = relation_phrase(r1, split, f"{family}:{group}:r1")
    p2 = relation_phrase(r2, split, f"{family}:{group}:r2")

    if family == "endpoint_role":
        edges = [edge("e0", ids[0], ids[1], r1)]
        return [
            make_row(
                split=split, family=family, group=group, variant="SOURCE",
                fields=fields, edges=edges,
                query=f"{names[0]} {p1} {names[1]}. Which record is doing that relational work?",
                relation_sequence=[r1], role="SOURCE", traversal="LOCAL_SELECT",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e0"], target_distribution={ids[0]: 1.0},
            ),
            make_row(
                split=split, family=family, group=group, variant="TARGET",
                fields=fields, edges=edges,
                query=f"{names[0]} {p1} {names[1]}. Which record is on the receiving side of that relationship?",
                relation_sequence=[r1], role="TARGET", traversal="LOCAL_SELECT",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e0"], target_distribution={ids[1]: 1.0},
            ),
        ]

    if family == "relation_filter":
        edges = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[0], ids[3], r2),
        ]
        return [
            make_row(
                split=split, family=family, group=group, variant="R1",
                fields=fields, edges=edges,
                query=f"From {names[0]}, follow the relationship where it {p1} {names[1]}. Which record is reached?",
                relation_sequence=[r1], role="TARGET", traversal="LOCAL_SELECT",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e0"], target_distribution={ids[1]: 1.0},
            ),
            make_row(
                split=split, family=family, group=group, variant="R2",
                fields=fields, edges=edges,
                query=f"From {names[0]}, use the different relationship where it {p2} {names[3]}. Which record is reached?",
                relation_sequence=[r2], role="TARGET", traversal="LOCAL_SELECT",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e1"], target_distribution={ids[3]: 1.0},
            ),
        ]

    if family == "ordered_path":
        # Ensure both orderings exist as different structural paths.
        fields = make_fields(split, family, group, ["GENERIC"] * 6)
        ids = [row["id"] for row in fields]
        names = [row["entity"] for row in fields]
        edges = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[1], ids[2], r2),
            edge("e2", ids[0], ids[3], r2),
            edge("e3", ids[3], ids[4], r1),
        ]
        return [
            make_row(
                split=split, family=family, group=group, variant="R1_R2",
                fields=fields, edges=edges,
                query=(
                    f"Start at {names[0]}. First use the link described as '{p1}', "
                    f"then use the one described as '{p2}'. Which record is the endpoint?"
                ),
                relation_sequence=[r1, r2], role="TARGET", traversal="PATH",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e0", "e1"], target_distribution={ids[2]: 1.0},
            ),
            make_row(
                split=split, family=family, group=group, variant="R2_R1",
                fields=fields, edges=edges,
                query=(
                    f"Start at {names[0]}. Take the '{p2}' relationship first and "
                    f"the '{p1}' relationship second. Where does that ordered route end?"
                ),
                relation_sequence=[r2, r1], role="TARGET", traversal="PATH",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e2", "e3"], target_distribution={ids[4]: 1.0},
            ),
        ]

    if family == "three_hop":
        r3 = relation_pool[(relation_pool.index(r2) + 1) % len(relation_pool)]
        fields = make_fields(split, family, group, ["GENERIC"] * 6)
        ids = [row["id"] for row in fields]
        names = [row["entity"] for row in fields]
        p3 = relation_phrase(r3, split, f"{family}:{group}:r3")
        edges = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[1], ids[2], r2),
            edge("e2", ids[2], ids[3], r3),
            edge("e3", ids[0], ids[4], r3),
        ]
        return [
            make_row(
                split=split, family=family, group=group, variant="CHAIN",
                fields=fields, edges=edges,
                query=(
                    f"Beginning with {names[0]}, follow three links in this order: "
                    f"'{p1}', then '{p2}', then '{p3}'. Which record is reached last?"
                ),
                relation_sequence=[r1, r2, r3], role="TARGET", traversal="PATH",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e0", "e1", "e2"], target_distribution={ids[3]: 1.0},
            ),
            make_row(
                split=split, family=family, group=group, variant="WRONG_FIRST_DISTRACTOR",
                fields=fields, edges=edges,
                query=(
                    f"From {names[0]}, the route must begin with '{p1}' before '{p2}' "
                    f"and '{p3}'. Ignore a tempting link that starts with the last relation. "
                    f"Which endpoint satisfies the full sequence?"
                ),
                relation_sequence=[r1, r2, r3], role="TARGET", traversal="PATH",
                modifier_target=modifiers(), focus_field_id=ids[0],
                support_edge_ids=["e0", "e1", "e2"], target_distribution={ids[3]: 1.0},
            ),
        ]

    if family == "aggregate":
        fields = make_fields(split, family, group, [src_type, tgt_type, src_type, tgt_type, "GENERIC", "GENERIC"])
        ids = [row["id"] for row in fields]
        names = [row["entity"] for row in fields]
        edges = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[2], ids[3], r1),
        ]
        return [
            make_row(
                split=split, family=family, group=group, variant="SOURCES",
                fields=fields, edges=edges,
                query=f"Both {names[0]} and {names[2]} {p1} separate records. Return the two records doing the relational work.",
                relation_sequence=[r1], role="SOURCE", traversal="AGGREGATE",
                modifier_target=modifiers(), focus_field_id=None,
                support_edge_ids=["e0", "e1"],
                target_distribution={ids[0]: 0.5, ids[2]: 0.5},
            ),
            make_row(
                split=split, family=family, group=group, variant="TARGETS",
                fields=fields, edges=edges,
                query=f"Two separate links use the same kind of relationship: each source {p1} another record. Return both receiving records.",
                relation_sequence=[r1], role="TARGET", traversal="AGGREGATE",
                modifier_target=modifiers(), focus_field_id=None,
                support_edge_ids=["e0", "e1"],
                target_distribution={ids[1]: 0.5, ids[3]: 0.5},
            ),
        ]

    if family == "reliability":
        fields = make_fields(split, family, group, [src_type, tgt_type, src_type, tgt_type, "GENERIC", "GENERIC"])
        ids = [row["id"] for row in fields]
        names = [row["entity"] for row in fields]
        left = [
            edge("e0", ids[0], ids[1], r1, reliability=0.95),
            edge("e1", ids[2], ids[3], r1, reliability=0.15),
        ]
        right = [
            edge("e0", ids[0], ids[1], r1, reliability=0.15),
            edge("e1", ids[2], ids[3], r1, reliability=0.95),
        ]
        base_query = (
            f"Two {r1.lower().replace('_',' ')}-type links are relevant. "
            "Prefer the one backed by the stronger provenance and verification record."
        )
        return [
            make_row(
                split=split, family=family, group=group, variant="LEFT_HIGH",
                fields=fields, edges=left, query=base_query,
                relation_sequence=[r1], role="TARGET", traversal="AGGREGATE",
                modifier_target=modifiers(reliability=True), focus_field_id=None,
                support_edge_ids=["e0", "e1"], target_distribution={ids[1]: 1.0},
            ),
            make_row(
                split=split, family=family, group=group, variant="RIGHT_HIGH",
                fields=fields, edges=right, query=base_query,
                relation_sequence=[r1], role="TARGET", traversal="AGGREGATE",
                modifier_target=modifiers(reliability=True), focus_field_id=None,
                support_edge_ids=["e0", "e1"], target_distribution={ids[3]: 1.0},
            ),
        ]

    if family == "temporal":
        fields = make_fields(split, family, group, [src_type, tgt_type, src_type, tgt_type, "GENERIC", "GENERIC"])
        ids = [row["id"] for row in fields]
        left = [
            edge("e0", ids[0], ids[1], r1, recency=0.95),
            edge("e1", ids[2], ids[3], r1, recency=0.10),
        ]
        right = [
            edge("e0", ids[0], ids[1], r1, recency=0.10),
            edge("e1", ids[2], ids[3], r1, recency=0.95),
        ]
        q = "Two structurally valid records apply at different times. Use the relationship that belongs to the latest applicable state."
        return [
            make_row(
                split=split, family=family, group=group, variant="LEFT_LATEST",
                fields=fields, edges=left, query=q,
                relation_sequence=[r1], role="TARGET", traversal="AGGREGATE",
                modifier_target=modifiers(recency=True), focus_field_id=None,
                support_edge_ids=["e0", "e1"], target_distribution={ids[1]: 1.0},
            ),
            make_row(
                split=split, family=family, group=group, variant="RIGHT_LATEST",
                fields=fields, edges=right, query=q,
                relation_sequence=[r1], role="TARGET", traversal="AGGREGATE",
                modifier_target=modifiers(recency=True), focus_field_id=None,
                support_edge_ids=["e0", "e1"], target_distribution={ids[3]: 1.0},
            ),
        ]

    if family == "path_latest":
        fields = make_fields(split, family, group, ["GENERIC"] * 6)
        ids = [row["id"] for row in fields]
        edges_a = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[1], ids[2], r2, recency=0.90),
            edge("e2", ids[1], ids[3], r2, recency=0.20),
        ]
        edges_b = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[1], ids[2], r2, recency=0.20),
            edge("e2", ids[1], ids[3], r2, recency=0.90),
        ]
        q = (
            f"Start with {fields[0]['entity']}. Follow '{p1}' and then '{p2}', "
            "but when the second step has more than one valid continuation use the latest one."
        )
        return [
            make_row(
                split=split, family=family, group=group, variant="LEFT_LATEST",
                fields=fields, edges=edges_a, query=q,
                relation_sequence=[r1, r2], role="TARGET", traversal="PATH",
                modifier_target=modifiers(recency=True), focus_field_id=ids[0],
                support_edge_ids=["e0", "e1", "e2"], target_distribution={ids[2]: 1.0},
            ),
            make_row(
                split=split, family=family, group=group, variant="RIGHT_LATEST",
                fields=fields, edges=edges_b, query=q,
                relation_sequence=[r1, r2], role="TARGET", traversal="PATH",
                modifier_target=modifiers(recency=True), focus_field_id=ids[0],
                support_edge_ids=["e0", "e1", "e2"], target_distribution={ids[3]: 1.0},
            ),
        ]

    if family == "ambiguity":
        fields = make_fields(split, family, group, [src_type, tgt_type, src_type, tgt_type, "GENERIC", "GENERIC"])
        ids = [row["id"] for row in fields]
        edges = [
            edge("e0", ids[0], ids[1], r1),
            edge("e1", ids[2], ids[3], r1),
        ]
        return [
            make_row(
                split=split, family=family, group=group, variant="PLURAL_TARGET",
                fields=fields, edges=edges,
                query=f"Two equally valid links both match the requested relationship. Preserve both receiving records rather than inventing a winner.",
                relation_sequence=[r1], role="TARGET", traversal="AGGREGATE",
                modifier_target=modifiers(), focus_field_id=None,
                support_edge_ids=["e0", "e1"], target_distribution={ids[1]: 0.5, ids[3]: 0.5},
            ),
            make_row(
                split=split, family=family, group=group, variant="PLURAL_SOURCE",
                fields=fields, edges=edges,
                query=f"Two equally valid links both match. Preserve both source-side records rather than collapsing the answer to one.",
                relation_sequence=[r1], role="SOURCE", traversal="AGGREGATE",
                modifier_target=modifiers(), focus_field_id=None,
                support_edge_ids=["e0", "e1"], target_distribution={ids[0]: 0.5, ids[2]: 0.5},
            ),
        ]

    if family == "outside_support":
        fields = make_fields(split, family, group, [src_type, tgt_type, "GENERIC", "GENERIC", "GENERIC", "GENERIC"])
        ids = [row["id"] for row in fields]
        edges = [edge("e0", ids[0], ids[1], r1)]
        q = f"Which receiving record is connected to {fields[0]['entity']} by the relationship where it {p1} the other record?"
        rows = []
        for variant, value in (("DISTRACTOR_LOW", 0.1), ("DISTRACTOR_HIGH", 1.0)):
            variant_fields = json.loads(json.dumps(fields))
            variant_fields[5]["metadata"]["reliability"] = value
            variant_fields[5]["metadata"]["normalized_time"] = value
            rows.append(
                make_row(
                    split=split, family=family, group=group, variant=variant,
                    fields=variant_fields, edges=edges, query=q,
                    relation_sequence=[r1], role="TARGET", traversal="LOCAL_SELECT",
                    modifier_target=modifiers(), focus_field_id=ids[0],
                    support_edge_ids=["e0"], target_distribution={ids[1]: 1.0},
                )
            )
        return rows

    if family == "fallback_defer":
        fields = make_fields(split, family, group, ["GENERIC"] * 6)
        return [
            make_row(
                split=split, family=family, group=group, variant="FALLBACK",
                fields=fields, edges=[],
                query="What does the phrase 'only after lunch' imply about the timing of the request?",
                relation_sequence=[], role="NONE", traversal="LOCAL_SELECT",
                modifier_target=modifiers(), focus_field_id=None,
                support_edge_ids=[], target_distribution={}, applicability=0.1,
                control="FALLBACK",
            ),
            make_row(
                split=split, family=family, group=group, variant="DEFER",
                fields=fields, edges=[
                    edge("e0", fields[0]["id"], fields[1]["id"], r1),
                    edge("e1", fields[2]["id"], fields[3]["id"], r2),
                ],
                query=(
                    "Several records are connected, but the request never specifies "
                    "which relationship or direction should govern. Preserve that uncertainty."
                ),
                relation_sequence=[], role="NONE", traversal="LOCAL_SELECT",
                modifier_target=modifiers(), focus_field_id=None,
                support_edge_ids=[], target_distribution={}, applicability=0.5,
                control="DEFER",
            ),
        ]

    raise ValueError(family)


FAMILIES = (
    "endpoint_role",
    "relation_filter",
    "ordered_path",
    "three_hop",
    "aggregate",
    "reliability",
    "temporal",
    "path_latest",
    "ambiguity",
    "outside_support",
    "fallback_defer",
)


def open_schema_rows(group: int) -> list[dict]:
    split = "dev"
    relation = OPEN_RELATIONS[group % len(OPEN_RELATIONS)]
    src_type, tgt_type = RELATION_TYPES[relation]
    fields = make_fields(split, "open_schema", group, [src_type, tgt_type, src_type, tgt_type, "GENERIC", "GENERIC"])
    ids = [row["id"] for row in fields]
    names = [row["entity"] for row in fields]
    phrase = relation_phrase(relation, "dev", f"open:{group}")
    edges = [
        edge("e0", ids[0], ids[1], relation),
        edge("e1", ids[2], ids[3], relation),
    ]
    symmetric = relation == "CONFLICTS_WITH"
    role = "SYMMETRIC" if symmetric else "TARGET"
    target = (
        {ids[0]: 0.5, ids[1]: 0.5}
        if symmetric
        else {ids[1]: 1.0}
    )
    query = (
        f"Without relying on a memorized relation label, use the supplied schema meaning: "
        f"{names[0]} {phrase} {names[1]}. Which record or records satisfy the requested endpoint semantics?"
    )
    row = make_row(
        split=split, family="open_schema", group=group, variant="ZERO_SHOT_SCHEMA",
        fields=fields, edges=edges, query=query,
        relation_sequence=[relation], role=role, traversal="LOCAL_SELECT",
        modifier_target=modifiers(), focus_field_id=ids[0],
        support_edge_ids=["e0"], target_distribution=target,
        open_schema=True,
    )
    # Cue-collision sibling: same unseen relation is present twice, entity names select support.
    sibling = make_row(
        split=split, family="open_schema", group=group, variant="CUE_COLLISION",
        fields=fields, edges=edges,
        query=(
            f"Use the same supplied schema relation, but answer for the pair involving "
            f"{names[2]} and {names[3]} rather than the first pair. The relationship {phrase}."
        ),
        relation_sequence=[relation], role=role, traversal="LOCAL_SELECT",
        modifier_target=modifiers(), focus_field_id=ids[2],
        support_edge_ids=["e1"],
        target_distribution=(
            {ids[2]: 0.5, ids[3]: 0.5}
            if symmetric
            else {ids[3]: 1.0}
        ),
        open_schema=True,
    )
    return [row, sibling]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--train-groups-per-family", type=int, default=20)
    p.add_argument("--dev-groups-per-family", type=int, default=8)
    p.add_argument("--open-schema-groups", type=int, default=32)
    args = p.parse_args()

    rows: list[dict] = []
    for family in FAMILIES:
        for group in range(args.train_groups_per_family):
            rows.extend(
                pair_rows(
                    split="train",
                    family=family,
                    group=group,
                    relation_pool=CORE_RELATIONS,
                )
            )
        for group in range(args.dev_groups_per_family):
            rows.extend(
                pair_rows(
                    split="dev",
                    family=family,
                    group=group,
                    relation_pool=CORE_RELATIONS,
                )
            )
    for group in range(args.open_schema_groups):
        rows.extend(open_schema_rows(group))

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate production curriculum row id")
    if any(row["private_identity_data"] is not False for row in rows):
        raise SystemExit("private identity data entered production curriculum")
    if any(
        row["open_schema_relation"] and row["split"] != "dev"
        for row in rows
    ):
        raise SystemExit("open-schema relation supervision leaked into TRAIN")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "dev")
    }
    open_count = sum(bool(row["open_schema_relation"]) for row in rows)
    max_steps = max(len(row["operator_target"]["relation_sequence"]) for row in rows)
    print("status=PASS_QSRE_PRODUCTION_PUBLIC_CURRICULUM")
    print(f"rows={len(rows)} train={counts['train']} dev={counts['dev']}")
    print(f"open_schema_dev_rows={open_count}")
    print(f"max_relation_steps={max_steps}")
    print(
        "sha256="
        + hashlib.sha256(output.read_bytes()).hexdigest()
    )


if __name__ == "__main__":
    main()
