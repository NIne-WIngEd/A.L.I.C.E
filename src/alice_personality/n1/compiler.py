from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ACTIVE_FILES = {
    "E0": "source_authority/canonical_v2_e0_units.jsonl",
    "EINF": "curated/curated_einf_v2.jsonl",
    "ASYN_DIRECT": "curated/curated_asyn_direct_v2.jsonl",
    "ASYN_BASE": "curated/curated_asyn_base_policies_v2.jsonl",
    "ASYN_TARGETED": "curated/curated_asyn_targeted_v2.jsonl",
    "ASYN_CONTEXT": "curated/curated_asyn_context_variants_v2.jsonl",
}
UNKNOWN_FILE = "curated/historical_unknown_bank_v2.jsonl"
ALTERNATIVE_FILE = "blueprints/alternative_policy_competitors_v1.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def bool_authority(row: dict[str, Any]) -> bool:
    if "training_authority" in row:
        return bool(row["training_authority"])
    if "model_training_authority" in row:
        return bool(row["model_training_authority"])
    return False


def record_id(kind: str, row: dict[str, Any]) -> str:
    candidates = {
        "E0": ["unit_id", "substrate_id"],
        "EINF": ["curated_id", "source_proposal_id"],
        "ASYN_DIRECT": ["curated_id", "source_proposal_id"],
        "ASYN_BASE": ["curated_base_id"],
        "ASYN_TARGETED": ["curated_id", "proposal_id"],
        "ASYN_CONTEXT": ["curated_id", "proposal_id"],
    }[kind]
    for key in candidates:
        if row.get(key):
            return str(row[key])
    raise ValueError(f"missing stable id for {kind}")


def normalize_active_record(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    rid = record_id(kind, row)
    if kind == "E0":
        text = str(row["text"])
        if text_sha256(text).lower() != str(row.get("text_sha256", "")).lower():
            raise ValueError(f"E0 text hash mismatch for {rid}")
        return {
            "record_id": rid,
            "source_kind": kind,
            "provenance_class": row.get("provenance_class", "E0"),
            "source_text": text,
            "scenario": None,
            "behavioral_policy": None,
            "personality_dimensions": list(row.get("semantic_labels", [])),
            "relationship_context": list(row.get("relationship_contexts", [])),
            "emotional_context": None,
            "social_context": None,
            "boundary_conditions": [],
            "disconfirmation_conditions": [],
            "identity_supporting_e0_ids": [],
            "context_only_e0_ids": [],
            "excluded_context_e0_ids": [],
            "supporting_einf_ids": [],
            "uncertainty_reason": None,
            "confidence_or_support_strength": None,
            "historical_truth_allowed": bool(row.get("historical_truth_allowed", True)),
            "runtime_behavioral_prior_allowed": False,
            "alice_lived_memory": bool(row.get("alice_lived_memory", False)),
            "autobiographical_recall_allowed": bool(row.get("historical_truth_allowed", True)),
            "supervision_lanes": list(row.get("use_lanes", [])),
            "loss_mask": dict(row.get("loss_mask", {})),
            "training_authority": bool_authority(row),
            "owner_final_review_required": True,
            "coverage_cell_id": None,
            "related_active_ids": [],
            "source_ref": row.get("record_ref"),
        }

    policy = row.get("behavioral_proposal")
    if policy is None:
        policy = row.get("base_behavioral_policy")
    scenario = row.get("scenario")
    if scenario is None and row.get("scenario_family") is not None:
        scenario = row.get("scenario_family")

    return {
        "record_id": rid,
        "source_kind": kind,
        "provenance_class": row.get("provenance_class"),
        "source_text": None,
        "scenario": scenario,
        "behavioral_policy": policy,
        "personality_dimensions": list(row.get("personality_dimensions", row.get("raw_personality_dimensions", []))),
        "relationship_context": row.get("relationship_context", row.get("relationship_contexts_seen")),
        "emotional_context": row.get("emotional_context", row.get("emotional_contexts_seen")),
        "social_context": row.get("social_context"),
        "boundary_conditions": as_list(row.get("boundary_conditions")),
        "disconfirmation_conditions": as_list(row.get("disconfirmation_conditions")),
        "identity_supporting_e0_ids": list(row.get("identity_supporting_E0_unit_ids", [])),
        "context_only_e0_ids": list(row.get("context_only_E0_unit_ids", [])),
        "excluded_context_e0_ids": list(row.get("excluded_context_E0_unit_ids", [])),
        "supporting_einf_ids": list(row.get("supporting_curated_EINF_ids", row.get("supporting_raw_EINF_ids", []))),
        "uncertainty_reason": row.get("uncertainty_reason"),
        "confidence_or_support_strength": row.get("confidence_or_support_strength"),
        "historical_truth_allowed": bool(row.get("historical_Elaina_truth", False)),
        "runtime_behavioral_prior_allowed": bool(row.get("runtime_behavioral_prior_allowed", False)),
        "alice_lived_memory": bool(row.get("Alice_lived_memory", False)),
        "autobiographical_recall_allowed": bool(row.get("autobiographical_recall_allowed", False)),
        "supervision_lanes": as_list(row.get("recommended_supervision_lane")),
        "loss_mask": None,
        "training_authority": bool_authority(row),
        "owner_final_review_required": bool(row.get("owner_final_review_required", True)),
        "coverage_cell_id": row.get("coverage_cell_id"),
        "related_active_ids": list(row.get("related_curated_ASYN_ids", [])),
        "source_ref": row.get("source_proposal_id", row.get("proposal_id", row.get("source_domain"))),
    }


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def family_splits(records: list[dict[str, Any]]) -> dict[str, str]:
    ids = {record["record_id"] for record in records}
    uf = UnionFind(ids)
    for record in records:
        if record["source_kind"].startswith("ASYN"):
            for related in record.get("related_active_ids", []):
                if related in ids:
                    uf.union(record["record_id"], related)

    split_map: dict[str, str] = {}
    for record in records:
        rid = record["record_id"]
        if record["source_kind"].startswith("ASYN"):
            family = f"ASYN:{uf.find(rid)}"
        elif record["source_kind"] == "EINF":
            family = f"EINF:{record.get('source_ref') or rid}"
        else:
            family = f"E0:{rid}"
        bucket = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:8], 16) % 1000
        split_map[rid] = "test" if bucket < 50 else "dev" if bucket < 100 else "train"
    return split_map


def validate_supports(records: list[dict[str, Any]], e0_ids: set[str], einf_ids: set[str]) -> None:
    for record in records:
        rid = record["record_id"]
        identity = set(record.get("identity_supporting_e0_ids", []))
        context = set(record.get("context_only_e0_ids", []))
        excluded = set(record.get("excluded_context_e0_ids", []))
        missing = (identity | context | excluded) - e0_ids
        if missing:
            raise ValueError(f"{rid}: missing E0 support ids: {sorted(missing)[:5]}")
        if identity & excluded:
            raise ValueError(f"{rid}: identity support overlaps excluded context")
        if identity & context:
            raise ValueError(f"{rid}: identity support overlaps context-only support")
        for support in record.get("supporting_einf_ids", []):
            if support not in einf_ids and support:
                # Raw historical E-INF IDs may be preserved as lineage references.
                continue


def compile_identity_substrate(package_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(package_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((root / "curation_manifest.json").read_text(encoding="utf-8"))
    flags = manifest.get("authority_flags", {})
    if flags.get("E0_mutated"):
        raise ValueError("curation manifest reports E0 mutation")
    if flags.get("candidate_promotion_to_E0_performed"):
        raise ValueError("curation manifest reports candidate promotion to E0")
    if flags.get("model_training_performed") or flags.get("weights_created"):
        raise ValueError("source frontier is not a pre-weight package")

    source_rows = {kind: read_jsonl(root / rel) for kind, rel in ACTIVE_FILES.items()}
    records = [normalize_active_record(kind, row) for kind, rows in source_rows.items() for row in rows]
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("duplicate active record ids")

    e0_ids = {record_id("E0", row) for row in source_rows["E0"]}
    einf_ids = {record_id("EINF", row) for row in source_rows["EINF"]}
    validate_supports(records, e0_ids, einf_ids)

    split_map = family_splits(records)
    for record in records:
        record["split"] = split_map[record["record_id"]]

    unknown_rows = read_jsonl(root / UNKNOWN_FILE)
    alternative_rows = read_jsonl(root / ALTERNATIVE_FILE)

    expected = manifest.get("counts", {})
    observed_manifest_counts = {
        "curated_EINF": len(source_rows["EINF"]),
        "curated_direct_ASYN_carried": len(source_rows["ASYN_DIRECT"]),
        "factorized_base_policies": len(source_rows["ASYN_BASE"]),
        "targeted_ASYN_added": len(source_rows["ASYN_TARGETED"]),
        "context_variants_added": len(source_rows["ASYN_CONTEXT"]),
        "historical_UNKNOWN_bank": len(unknown_rows),
    }
    for key, observed in observed_manifest_counts.items():
        if key in expected and int(expected[key]) != observed:
            raise ValueError(f"curation manifest count mismatch for {key}: expected {expected[key]}, got {observed}")
    if int(expected.get("targeted_gap_queue_remaining", 0)) != 0:
        raise ValueError("targeted gap queue is not closed")

    if any(bool_authority(row) for row in unknown_rows):
        raise ValueError("historical UNKNOWN bank unexpectedly carries training authority")
    if any(bool_authority(row) or bool(row.get("hard_negative_authorized", False)) for row in alternative_rows):
        raise ValueError("alternative competitors unexpectedly authorized as training negatives")

    concepts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for value in record.get("personality_dimensions", []):
            concepts["personality_or_semantic_dimension"][str(value)] += 1
        for value in as_list(record.get("relationship_context")):
            concepts["relationship_context"][str(value)] += 1
        for value in as_list(record.get("emotional_context")):
            concepts["emotional_context"][str(value)] += 1

    support_edges: list[dict[str, Any]] = []
    for record in records:
        rid = record["record_id"]
        for target in record.get("identity_supporting_e0_ids", []):
            support_edges.append({"from_record_id": rid, "to_record_id": target, "edge_type": "identity_support"})
        for target in record.get("context_only_e0_ids", []):
            support_edges.append({"from_record_id": rid, "to_record_id": target, "edge_type": "context_only"})
        for target in record.get("excluded_context_e0_ids", []):
            support_edges.append({"from_record_id": rid, "to_record_id": target, "edge_type": "excluded_context"})
        for target in record.get("supporting_einf_ids", []):
            support_edges.append({"from_record_id": rid, "to_record_id": target, "edge_type": "inference_support"})

    def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    write_jsonl(output / "active_identity_records.jsonl", records)
    write_jsonl(output / "support_edges.jsonl", support_edges)
    write_jsonl(output / "historical_unknown_bank.jsonl", unknown_rows)
    write_jsonl(output / "alternative_competitors_unordered.jsonl", alternative_rows)

    concept_payload = {
        category: [{"value": value, "count": count} for value, count in counter.most_common()]
        for category, counter in sorted(concepts.items())
    }
    (output / "observed_concept_inventory.json").write_text(
        json.dumps(concept_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema": "alice.eipm.n1.private-substrate-compile.v0.1",
        "active_record_count": len(records),
        "kind_counts": dict(sorted(Counter(record["source_kind"] for record in records).items())),
        "split_counts": dict(sorted(Counter(record["split"] for record in records).items())),
        "support_edge_count": len(support_edges),
        "historical_unknown_count": len(unknown_rows),
        "alternative_competitor_count": len(alternative_rows),
        "training_authority_counts": {str(k).lower(): v for k, v in Counter(record["training_authority"] for record in records).items()},
        "owner_final_review_required_counts": {str(k).lower(): v for k, v in Counter(record["owner_final_review_required"] for record in records).items()},
        "private_gradient_authorized": False,
        "source_training_authority_granted": bool(flags.get("training_authority_granted", False)),
        "source_owner_final_preweight_review_required": bool(flags.get("owner_final_preweight_review_required", True)),
        "alternatives_are_unordered_not_negatives": True,
        "source_package_root": str(root),
    }
    (output / "compile_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
