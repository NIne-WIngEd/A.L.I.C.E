from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RELATION_NAMES = {
    0: "SUPPORTS",
    1: "CORRECTS",
    2: "SUPERSEDES",
    3: "DERIVED_FROM",
    4: "CAUSES",
    5: "TEMPORAL_SUCCESSOR",
}

CONTROL = {"FALLBACK": 0, "RELATIONAL": 1, "DEFER": 2}

TRAIN_NAMES = [
    ("Mara", "Noah"),
    ("Priya", "Evan"),
    ("Lena", "Omar"),
    ("Sofia", "Caleb"),
    ("Nadia", "Theo"),
    ("Ari", "Jonah"),
]
DEV_NAMES = [
    ("Mina", "Elliot"),
    ("Tara", "Isaac"),
    ("Leila", "Marcus"),
    ("Rina", "Adrian"),
    ("Zara", "Miles"),
    ("Nora", "Felix"),
]

TRAIN_OBJECTS = [
    ("calibration log", "drift claim"),
    ("erratum", "lab note"),
    ("June procedure", "March procedure"),
    ("weekly summary", "raw export"),
    ("seal leak report", "pressure-test failure"),
    ("revision C", "revision B"),
]
DEV_OBJECTS = [
    ("diagnostic trace", "fault hypothesis"),
    ("correction memo", "incident note"),
    ("release policy", "draft policy"),
    ("cleaned dataset", "instrument dump"),
    ("scheduler outage", "missed backup"),
    ("migration update", "initial report"),
]

RELATION_CLAUSES = {
    0: {
        "train": [
            "{a} gives concrete evidence for {b}.",
            "{a} makes the conclusion in {b} more credible.",
            "The evidence in {a} strengthens the claim recorded in {b}.",
            "{b} is better supported because of {a}.",
        ],
        "dev": [
            "{a} materially backs the interpretation written in {b}.",
            "The observation in {a} increases confidence in {b}.",
            "{b} gains evidential support from {a}.",
            "The case stated in {b} is strengthened by {a}.",
        ],
    },
    1: {
        "train": [
            "{a} fixes an error that appears in {b}.",
            "{a} repairs the mistaken value recorded in {b}.",
            "The mistake in {b} is amended by {a}.",
            "{a} supplies the correction for {b}.",
        ],
        "dev": [
            "{a} revises the inaccurate detail in {b}.",
            "{b} contains an error that {a} repairs.",
            "The later note {a} fixes what was wrong in {b}.",
            "{a} carries the amendment needed by {b}.",
        ],
    },
    2: {
        "train": [
            "{a} takes over from {b} as the version people should use.",
            "{b} loses precedence when {a} becomes authoritative.",
            "{a} replaces {b} without erasing the older history.",
            "The governing version changes from {b} to {a}.",
        ],
        "dev": [
            "{a} displaces {b} as the current authority.",
            "{b} is no longer current after {a} takes effect.",
            "{a} becomes the active version in place of {b}.",
            "Authority moves from {b} to {a}.",
        ],
    },
    3: {
        "train": [
            "{a} was produced from the information in {b}.",
            "{b} is the underlying material used to create {a}.",
            "{a} comes from {b} as its source record.",
            "The contents of {b} are the basis for {a}.",
        ],
        "dev": [
            "{a} was assembled using {b} as its source.",
            "{b} is the provenance base from which {a} was obtained.",
            "{a} originates from the material recorded in {b}.",
            "The record {a} was generated on the basis of {b}.",
        ],
    },
    4: {
        "train": [
            "{a} led to the event recorded in {b}.",
            "The condition in {a} brought about {b}.",
            "{b} happened because of what is recorded in {a}.",
            "{a} produced the resulting event in {b}.",
        ],
        "dev": [
            "{a} is the antecedent that resulted in {b}.",
            "The event in {b} followed as a consequence of {a}.",
            "{a} triggered the outcome recorded in {b}.",
            "What happened in {a} is responsible for {b}.",
        ],
    },
    5: {
        "train": [
            "{a} comes after {b} in the documented sequence.",
            "{b} is earlier, and {a} is the record that follows it.",
            "The timeline places {a} after {b}.",
            "{a} is the later member of the pair containing {b}.",
        ],
        "dev": [
            "{a} succeeds {b} chronologically.",
            "{b} occurs first and {a} follows in time.",
            "The ordering of the records puts {a} after {b}.",
            "{a} is later than the predecessor record {b}.",
        ],
    },
}

ROLE_QUESTIONS = {
    0: {
        "train": [
            "Which record is on the first side of that relation?",
            "Which record is doing the evidential or relational work described here?",
            "Which record occupies the originating side of this link?",
            "Which record should be selected as the source-side answer?",
        ],
        "dev": [
            "Which record is the relation's originating endpoint?",
            "Which record is playing the active side of that connection?",
            "Which record belongs on the source side of the relation?",
            "Which record supplies the relation described in the scenario?",
        ],
    },
    1: {
        "train": [
            "Which record is on the receiving side of that relation?",
            "Which record is the endpoint affected by the connection described here?",
            "Which record occupies the destination side of this link?",
            "Which record should be selected as the target-side answer?",
        ],
        "dev": [
            "Which record is the relation's receiving endpoint?",
            "Which record is playing the affected side of that connection?",
            "Which record belongs on the target side of the relation?",
            "Which record receives the relation described in the scenario?",
        ],
    },
}

TRAIN_FALLBACK = [
    "A passenger asks, 'Could you stop near the library?' What is the likely speech act?",
    "The sentence says, 'Again, the printer jammed.' What does the word 'again' normally presuppose?",
    "A structured status says delivered while the raw note says not shipped. What inconsistency should be reported?",
    "The witness remembers either 8KJ or BKJ from a plate seen at night. How should uncertainty be expressed?",
    "In this paragraph, what does the word 'bank' mean given that the discussion is about a river?",
    "A manager says 'That was quick' after a two-hour delay. What pragmatic reading is plausible?",
]

DEV_FALLBACK = [
    "A user asks, 'Can you open the window?' What action is most likely being requested?",
    "The sentence says, 'Even Mira passed.' What does 'even' contribute to the meaning?",
    "A table says approved while the signed note says rejected. What conflict should be surfaced?",
    "A blurry label could read 1O7 or 107. How should the answer communicate confidence?",
    "In a biology passage, what sense of 'cell' is intended?",
    "Someone says 'Perfect timing' immediately after a late arrival. What implied attitude is likely?",
]

TRAIN_DEFER = [
    "Two records are clearly connected, but the request never establishes whether the link is evidential, causal, corrective, temporal, provenance-based, or replacement. Which operator is justified?",
    "One note refers to another, but the text does not establish whether it backs it, fixes it, replaces it, comes from it, causes it, or follows it in time. What relation can be chosen safely?",
    "Several link types are present and the request only asks for 'the related record' without identifying which connection matters. Which relational interpretation is warranted?",
    "The documents concern the same event, but no direction or relation type is stated. What relational action should be taken?",
]

DEV_DEFER = [
    "The query asks for 'the connected record' even though evidence, chronology, and provenance links all exist. Which one is actually licensed by the wording?",
    "Two notes mention the same incident, but nothing says whether one corrects, supports, replaces, derives from, causes, or follows the other. What operator is supported?",
    "A relation may matter here, but the request does not identify the relation family or its direction. Which relational interpretation can be selected?",
    "The records are associated, yet the wording leaves both the type and direction of the connection unresolved. What should the operator extractor commit to?",
]

TRAIN_COLLISIONS = [
    "The later audited note is available, but the question is simply which record supplies evidence for the claim.",
    "Several records exist, but only one is the record that fixes the mistaken value.",
    "The most recent file is not the point of the question; identify the record that the new policy replaced.",
    "The trusted record is mentioned only as context. Which record was produced from the raw source?",
]
DEV_COLLISIONS = [
    "A newer verified note is present, but the requested endpoint is the record providing evidence.",
    "Many records are listed, yet the question asks only for the one that repairs the error.",
    "The post-audit file is a distractor. Which record lost authority when the replacement took effect?",
    "A high-confidence source is mentioned, but the requested record is the item created from that source.",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pick(values, key: str):
    index = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return values[index % len(values)]


def entities(split: str, key: str) -> tuple[str, str, str, str]:
    names = TRAIN_NAMES if split == "train" else DEV_NAMES
    objects = TRAIN_OBJECTS if split == "train" else DEV_OBJECTS
    n1, n2 = pick(names, key + "|names")
    o1, o2 = pick(objects, key + "|objects")
    # Use both named actors and concrete records to make the query look like
    # ordinary language rather than a bare relation instruction.
    a = f"{n1}'s {o1}"
    b = f"{n2}'s {o2}"
    return a, b, n1, n2


def relation_clause(relation_id: int, split: str, key: str) -> str:
    a, b, _, _ = entities(split, key)
    template = pick(RELATION_CLAUSES[relation_id][split], key + "|clause")
    return template.format(a=a, b=b)


def role_question(role_id: int, split: str, key: str) -> str:
    return pick(ROLE_QUESTIONS[role_id][split], key + "|roleq")


def role_select_query(row: dict, *, query_view: int) -> str:
    split = str(row["split"])
    op = row["operator"]
    relation_id = int(op["relation_sequence_id"][0])
    role_id = int(op["role_id"])
    key = f"{split}|{row['family']}|{row['causal_group']}|{row['variant']}|{query_view}"
    clause = relation_clause(relation_id, split, key)
    question = role_question(role_id, split, key)

    collision_bank = TRAIN_COLLISIONS if split == "train" else DEV_COLLISIONS
    collision = pick(collision_bank, key + "|collision")
    if query_view == 0:
        return f"{clause} {question}"
    return f"{collision} In this case, {clause.lower()} {question}"


def aggregate_query(row: dict, *, query_view: int) -> str:
    split = str(row["split"])
    op = row["operator"]
    relation_id = int(op["relation_sequence_id"][0])
    role_id = int(op["role_id"])
    key = f"{split}|{row['family']}|{row['causal_group']}|{row['variant']}|{query_view}"
    clause = relation_clause(relation_id, split, key)
    side = (
        "the records on the originating side of those links"
        if role_id == 0
        else "the records on the receiving side of those links"
    )
    if query_view == 0:
        return (
            f"Across several independently valid records, the same kind of connection occurs: "
            f"{clause} Which set contains {side}? Keep every co-valid answer."
        )
    return (
        f"More than one link is genuinely relevant here. {clause} "
        f"Do not force the result to one record; identify {side}."
    )


def reliability_query(row: dict, *, query_view: int) -> str:
    split = str(row["split"])
    op = row["operator"]
    relation_id = int(op["relation_sequence_id"][0])
    role_id = int(op["role_id"])
    key = f"{split}|{row['family']}|{row['causal_group']}|{row['variant']}|{query_view}"
    clause = relation_clause(relation_id, split, key)
    question = role_question(role_id, split, key)
    if query_view == 0:
        return (
            f"Two candidate links fit the relation. One is backed by an audited record with intact provenance; "
            f"the other comes from an unverified note already flagged as uncertain. {clause} "
            f"Use the better-supported link when answering. {question}"
        )
    return (
        f"The alternatives disagree. One link is reproduced in complete logs while the other depends on a "
        f"single unsupported recollection. {clause} Which candidate should govern the answer? {question}"
    )


def temporal_query(row: dict, *, query_view: int) -> str:
    split = str(row["split"])
    op = row["operator"]
    relation_id = int(op["relation_sequence_id"][0])
    role_id = int(op["role_id"])
    key = f"{split}|{row['family']}|{row['causal_group']}|{row['variant']}|{query_view}"
    clause = relation_clause(relation_id, split, key)
    question = role_question(role_id, split, key)
    if query_view == 0:
        return (
            f"The same type of link existed before a system change and again after that change completed. "
            f"For the state after the change, {clause.lower()} {question}"
        )
    return (
        f"An earlier relation instance belongs to the pre-review state; another instance belongs to the "
        f"post-review state. Resolve the question for the post-review state. {clause} {question}"
    )


def path_query(row: dict, *, query_view: int) -> str:
    split = str(row["split"])
    op = row["operator"]
    seq = [int(x) for x in op["relation_sequence_id"]]
    key = f"{split}|{row['family']}|{row['causal_group']}|{row['variant']}|{query_view}"
    a, b, n1, n2 = entities(split, key + "|p1")
    c, d, n3, n4 = entities(split, key + "|p2")
    first = pick(RELATION_CLAUSES[seq[0]][split], key + "|first").format(a=a, b=b)
    second = pick(RELATION_CLAUSES[seq[1]][split], key + "|second").format(a=c, b=d)
    if query_view == 0:
        return (
            f"Start from the focused record. The first connection is described by: {first} "
            f"After reaching that endpoint, the next connection is described by: {second} "
            f"Which record is reached after carrying out both steps in that order?"
        )
    return (
        f"A two-link chain must be composed without swapping the links. First use the connection expressed by "
        f"'{first}' Then use the connection expressed by '{second}' What is the terminal endpoint?"
    )


def fallback_or_defer_query(row: dict, *, query_view: int) -> str:
    split = str(row["split"])
    variant = str(row["variant"])
    key = f"{split}|{row['causal_group']}|{variant}|{query_view}"
    if variant == "FALLBACK":
        bank = TRAIN_FALLBACK if split == "train" else DEV_FALLBACK
    else:
        bank = TRAIN_DEFER if split == "train" else DEV_DEFER
    return pick(bank, key)


def query_text(row: dict, *, query_view: int) -> tuple[str, bool]:
    family = str(row["family"])
    if family == "fallback_defer":
        return fallback_or_defer_query(row, query_view=query_view), False
    if family == "ordered_path":
        return path_query(row, query_view=query_view), False
    if family in ("multi_support_aggregate", "ambiguity_plurality"):
        return aggregate_query(row, query_view=query_view), False
    if family == "reliability_arbitration":
        return reliability_query(row, query_view=query_view), False
    if family == "temporal_arbitration":
        return temporal_query(row, query_view=query_view), False

    text = role_select_query(row, query_view=query_view)
    cue_collision = query_view == 1
    return text, cue_collision


def validate(rows: list[dict], contract: dict) -> None:
    expected = contract["expected_rows"]
    counts = {
        "train": sum(r["split"] == "train" for r in rows),
        "dev": sum(r["split"] == "dev" for r in rows),
    }
    counts["total"] = len(rows)
    if counts != expected:
        raise SystemExit(f"row-count drift: {counts} != {expected}")

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate T2 v0.2 ids")
    if any(r.get("private_identity_data") is not False for r in rows):
        raise SystemExit("private identity data present")
    if any(r.get("training_authorized") is not False for r in rows):
        raise SystemExit("curriculum rows must not self-authorize training")
    if any(r["split"] not in ("train", "dev") for r in rows):
        raise SystemExit("unexpected split")

    train_text = " ".join(r["query_text"].lower() for r in rows if r["split"] == "train")
    dev_text = " ".join(r["query_text"].lower() for r in rows if r["split"] == "dev")
    forbidden = (" non-relational ", " defer ", " fallback ", " path_follow ", " role_select ")
    padded_train = " " + train_text + " "
    padded_dev = " " + dev_text + " "
    if any(term in padded_train or term in padded_dev for term in forbidden):
        raise SystemExit("control/operation label word leaked into curriculum")

    collision_count = sum(bool(r.get("cue_collision")) for r in rows)
    if collision_count < 100:
        raise SystemExit("insufficient cue-collision rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--t1-curriculum", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    t1_path = Path(args.t1_curriculum)
    output_path = Path(args.output)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "alice.eipm.n0.qsre-t2-curriculum-contract.v0.2":
        raise SystemExit("T2 v0.2 curriculum contract drift")
    if sha256(t1_path) != contract["source_t1_curriculum_sha256"]:
        raise SystemExit("T1 curriculum hash drift")

    source_rows = read_jsonl(t1_path)
    out: list[dict] = []
    for row in source_rows:
        split = str(row["split"])
        if split not in ("train", "dev"):
            raise SystemExit("T2 v0.2 remains TRAIN/DEV only")
        views = int(
            contract[
                "query_views_per_train_row"
                if split == "train"
                else "query_views_per_dev_row"
            ]
        )
        expected_control = str(row["expected_control"])
        relational = expected_control == "RELATIONAL"
        relation_sequence = (
            [int(x) for x in row["operator"]["relation_sequence_id"]]
            if relational
            else []
        )
        role_id = int(row["operator"]["role_id"]) if relational else 3
        operation_id = int(row["operator"]["operation_id"]) if relational else 0

        for query_view in range(views):
            text, cue_collision = query_text(row, query_view=query_view)
            payload = {
                "schema": "alice.eipm.n0.qsre-t2-operator-row.v0.2",
                "id": f"{row['id']}:r{query_view}",
                "split": split,
                "family": row["family"],
                "variant": row["variant"],
                "source_t1_row_id": row["id"],
                "source_t1_causal_group": row["causal_group"],
                "query_view": query_view,
                "query_text": text,
                "cue_collision": bool(cue_collision),
                "operator_target": {
                    "relation_sequence_id": relation_sequence,
                    "role_id": role_id,
                    "operation_id": operation_id,
                    "control_id": CONTROL[expected_control],
                },
                "private_identity_data": False,
                "training_authorized": False,
            }
            out.append(payload)

    validate(out, contract)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in out),
        encoding="utf-8",
    )
    print(f"status=PASS_QSRE_T2_REALISTIC_CURRICULUM_BUILD")
    print(f"rows={len(out)}")
    print(f"cue_collision_rows={sum(int(r['cue_collision']) for r in out)}")
    print(f"sha256={sha256(output_path)}")


if __name__ == "__main__":
    main()
