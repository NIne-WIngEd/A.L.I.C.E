from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RELATIONS = {
    0: {
        "name": "SUPPORTS",
        "train_source": [
            "provides evidence in favor of another record",
            "backs another record",
            "supports another record",
        ],
        "train_target": [
            "is backed by another record",
            "receives supporting evidence from another record",
            "is the record being supported",
        ],
        "dev_source": [
            "strengthens the case for another record",
            "serves as evidence for another record",
        ],
        "dev_target": [
            "has its case strengthened by another record",
            "is supported by evidence from another record",
        ],
        "train_edge": ["supports", "backs", "provides evidence for"],
        "dev_edge": ["strengthens the case for", "serves as evidence for"],
    },
    1: {
        "name": "CORRECTS",
        "train_source": [
            "repairs an error in another record",
            "corrects another record",
            "amends an inaccurate record",
        ],
        "train_target": [
            "is the record being corrected",
            "has an error repaired by another record",
            "is amended by another record",
        ],
        "dev_source": [
            "fixes a mistake in another record",
            "revises an inaccurate earlier record",
        ],
        "dev_target": [
            "is revised because it contains a mistake",
            "receives a correction from another record",
        ],
        "train_edge": ["corrects", "repairs an error in", "amends"],
        "dev_edge": ["fixes a mistake in", "revises an inaccurate"],
    },
    2: {
        "name": "SUPERSEDES",
        "train_source": [
            "replaces an older record as the current authority",
            "supersedes another record",
            "takes precedence over an older record",
        ],
        "train_target": [
            "is the older record being replaced",
            "is superseded by a newer authority",
            "loses precedence to a newer record",
        ],
        "dev_source": [
            "displaces an older record as current",
            "renders an older record obsolete",
        ],
        "dev_target": [
            "is displaced by the newer current record",
            "becomes obsolete when another record takes over",
        ],
        "train_edge": ["supersedes", "replaces", "takes precedence over"],
        "dev_edge": ["displaces", "renders obsolete"],
    },
    3: {
        "name": "DERIVED_FROM",
        "train_source": [
            "was derived from another record",
            "is based on another record",
            "comes from another record as its source",
        ],
        "train_target": [
            "is the basis from which another record was derived",
            "is the source record another record is based on",
            "provides the basis for a derived record",
        ],
        "dev_source": [
            "was obtained from another record",
            "has another record as its basis",
        ],
        "dev_target": [
            "is the underlying basis for another record",
            "is the origin from which another record was obtained",
        ],
        "train_edge": ["is derived from", "is based on", "comes from"],
        "dev_edge": ["was obtained from", "has as its basis"],
    },
    4: {
        "name": "CAUSES",
        "train_source": [
            "brings about another event or record",
            "causes another record",
            "produces the resulting record",
        ],
        "train_target": [
            "is the result caused by another record",
            "is brought about by another record",
            "is the consequence produced by another record",
        ],
        "dev_source": [
            "leads to another event or record",
            "acts as the causal antecedent of another record",
        ],
        "dev_target": [
            "is the consequence of another record",
            "results from another record",
        ],
        "train_edge": ["causes", "brings about", "produces"],
        "dev_edge": ["leads to", "is the causal antecedent of"],
    },
    5: {
        "name": "TEMPORAL_SUCCESSOR",
        "train_source": [
            "is the later record that follows another in time",
            "is the temporal successor of another record",
            "comes after another record",
        ],
        "train_target": [
            "is the earlier record followed by a later one",
            "is the temporal predecessor receiving a successor",
            "comes before the later successor record",
        ],
        "dev_source": [
            "succeeds another record chronologically",
            "is later than the record it follows",
        ],
        "dev_target": [
            "is earlier than its chronological successor",
            "is the record that a later one follows",
        ],
        "train_edge": ["comes after", "is the temporal successor of", "follows in time"],
        "dev_edge": ["succeeds chronologically", "is later than"],
    },
}

CONTROL = {"FALLBACK": 0, "RELATIONAL": 1, "DEFER": 2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def choose(values: list[str], key: str) -> str:
    index = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return values[index % len(values)]


def relation_role_phrase(
    relation_id: int,
    role_id: int,
    split: str,
    key: str,
) -> str:
    prefix = "train" if split == "train" else "dev"
    role = "source" if role_id == 0 else "target"
    return choose(RELATIONS[relation_id][f"{prefix}_{role}"], key)


def edge_phrase(relation_id: int, split: str, key: str) -> str:
    prefix = "train" if split == "train" else "dev"
    return choose(RELATIONS[relation_id][f"{prefix}_edge"], key)


def query_text(row: dict, *, query_view: int) -> str:
    split = row["split"]
    family = row["family"]
    variant = row["variant"]
    op = row["operator"]
    seq = [int(x) for x in op["relation_sequence_id"]]
    role = int(op["role_id"])
    key = f"{split}|{family}|{row['causal_group']}|{variant}|{query_view}"

    if family == "fallback_defer":
        if variant == "FALLBACK":
            bank = (
                [
                    "This request does not call for a relational operation. Keep the ordinary evidence path in control.",
                    "No relation needs to be executed for this request. Use general evidence handling instead.",
                ]
                if split == "train"
                else [
                    "Relational execution is not applicable here. Leave this to the normal evidence path.",
                    "The request is non-relational. Do not manufacture a relation; fall back to ordinary handling.",
                ]
            )
        else:
            bank = (
                [
                    "The request may involve a relation, but the intended relational interpretation is underspecified. Preserve the uncertainty and defer.",
                    "There is not enough relational intent to choose an operator safely. Do not guess; defer the relational decision.",
                ]
                if split == "train"
                else [
                    "Relational intent is unresolved. Keep the ambiguity instead of selecting an operation.",
                    "A relation might matter, but the query does not determine which one. Defer rather than inventing a choice.",
                ]
            )
        return choose(bank, key)

    if family == "ordered_path":
        first = edge_phrase(seq[0], split, key + "|0")
        second = edge_phrase(seq[1], split, key + "|1")
        bank = (
            [
                f"Starting from the focused record, follow a directed link where the current record {first} the next record, then a link where that record {second} the next record. Return the final reached record.",
                f"Use the focused record as the start. Traverse first the relation in which it {first} the next record and then the relation in which that record {second} the next one. Select the final endpoint.",
            ]
            if split == "train"
            else [
                f"Begin at the focused record. Take the {first} relation first and the {second} relation second, preserving that order, and return the last endpoint.",
                f"From the focus, compose two directed steps: first a link that {first} the next record, then one that {second} the following record. The answer is the terminal record.",
            ]
        )
        return choose(bank, key)

    phrase = relation_role_phrase(seq[0], role, split, key)

    if family in ("multi_support_aggregate", "ambiguity_plurality"):
        bank = (
            [
                f"Considering every valid supported link, return all records that {phrase}. Preserve multiple co-valid answers.",
                f"Aggregate across the supported relation instances and keep every record that {phrase}; do not collapse them to one.",
            ]
            if split == "train"
            else [
                f"Keep the full plural answer across supported links: every record that {phrase} should remain active.",
                f"Several relation instances may be simultaneously valid. Return all records that {phrase} rather than forcing a single winner.",
            ]
        )
        return choose(bank, key)

    if family == "reliability_arbitration":
        bank = (
            [
                f"Among the supported relation instances, choose the record that {phrase} on the most reliable link.",
                f"Use reliability to arbitrate the supported links and select the record that {phrase} for the strongest-reliability relation.",
            ]
            if split == "train"
            else [
                f"Prefer the highest-confidence supported link, then return the record that {phrase}.",
                f"Resolve the alternatives by evidence reliability. The answer is the record that {phrase} on the most trustworthy link.",
            ]
        )
        return choose(bank, key)

    if family == "temporal_arbitration":
        bank = (
            [
                f"Among the supported relation instances, prefer the latest one in time and return the record that {phrase}.",
                f"Use temporal recency to arbitrate the supported links. Select the record that {phrase} on the newest relation instance.",
            ]
            if split == "train"
            else [
                f"Choose the most recent supported relation and return the record that {phrase}.",
                f"Resolve the alternatives by which relation is latest in time, then select the record that {phrase}.",
            ]
        )
        return choose(bank, key)

    if family == "outside_support_distractor":
        bank = (
            [
                f"Ignore information outside the claimed support. Within the supported relation, return the record that {phrase}.",
                f"Only the supported link defines the relational domain. Select the record that {phrase}.",
            ]
            if split == "train"
            else [
                f"Unrelated evidence must not change the operator. On the supported link, identify the record that {phrase}.",
                f"Stay inside the claimed support and return the record that {phrase}; outside distractors are irrelevant.",
            ]
        )
        # Intentionally omit row variant so the counterfactual distractor pair
        # receives exactly the same query.
        return choose(bank, f"{split}|{family}|{row['causal_group']}|{query_view}")

    bank = (
        [
            f"Within the supported relation, identify the record that {phrase}.",
            f"Use the active relation and return the record that {phrase}.",
        ]
        if split == "train"
        else [
            f"For the relevant supported link, select the record that {phrase}.",
            f"Resolve the active relation by choosing the record that {phrase}.",
        ]
    )
    return choose(bank, key)


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
    if contract.get("schema") != "alice.eipm.n0.qsre-t2-curriculum-contract.v0.1":
        raise SystemExit("T2 curriculum contract drift")
    if sha256(t1_path) != contract["t1_curriculum_sha256"]:
        raise SystemExit("T1 curriculum hash drift")

    rows = read_jsonl(t1_path)
    out: list[dict] = []
    for row in rows:
        split = str(row["split"])
        if split not in ("train", "dev"):
            raise SystemExit("T2 initial curriculum must remain TRAIN/DEV only")
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
            payload = dict(row)
            payload["schema"] = "alice.eipm.n0.qsre-t2-operator-row.v0.1"
            payload["id"] = f"{row['id']}:q{query_view}"
            payload["source_t1_row_id"] = row["id"]
            payload["source_t1_causal_group"] = row["causal_group"]
            payload["query_view"] = query_view
            payload["query_template_bank"] = split
            payload["query_text"] = query_text(row, query_view=query_view)
            payload["operator_target"] = {
                "relation_sequence_id": relation_sequence,
                "role_id": role_id,
                "operation_id": operation_id,
                "control_id": CONTROL[expected_control],
            }
            payload["private_identity_data"] = False
            out.append(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in out),
        encoding="utf-8",
    )
    print(f"rows={len(out)}")
    print(f"sha256={sha256(output_path)}")


if __name__ == "__main__":
    main()
