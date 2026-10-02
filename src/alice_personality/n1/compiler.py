"""Compile pinned private identity sources without approving data or gradients.

The original EIPM compiler supplied the field mapping. This successor binds
every archive member, validates source namespaces and actual Boolean flags,
and assigns connected evidence/derivative families to a single split. ZIP
members are streamed, never extracted. Importing the module reads no source.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping
import zipfile


SCHEMA = "alice.eipm.n1.private-substrate-compile.v1"
PIN_SCHEMA = "alice.eipm.identity-package-pin.v1"
STATE = "COMPILED_UNQUALIFIED"
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
_FLAGS = {"E0_mutated", "candidate_promotion_to_E0_performed", "model_training_performed",
          "weights_created", "training_authority_granted", "owner_final_preweight_review_required"}
_BOOL_FIELDS = {
    "training_authority", "model_training_authority", "historical_truth_allowed",
    "historical_Elaina_truth", "runtime_behavioral_prior_allowed", "Alice_lived_memory",
    "alice_lived_memory", "autobiographical_recall_allowed", "owner_final_review_required",
    "hard_negative_authorized", "acceptance_authority", "private_gradient_authorized",
    "is_training_negative", "is_negative",
}
_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")
_ID = re.compile(r"[^\s\x00-\x1f\x7f]{1,512}\Z")
_ARTIFACTS = {"active_identity_records.jsonl", "support_edges.jsonl",
              "historical_unknown_bank.jsonl", "alternative_competitors_unordered.jsonl",
              "observed_concept_inventory.json", "split_families.jsonl"}


class IdentitySubstrateError(ValueError):
    """Source custody, authority, lineage or compiled artifacts are invalid."""


@dataclass(frozen=True)
class PackagePin:
    """A caller-reviewed exact archive and membership pin, not data approval."""

    archive_sha256: str
    package_root: str
    members_sha256: Mapping[str, str]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise IdentitySubstrateError(f"{label} requires a SHA256 digest")
    return value.lower()


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise IdentitySubstrateError(f"{label} requires a nonempty stable identifier")
    return value


def _member(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value \
            or value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/")) \
            or any(ord(char) < 32 for char in value):
        raise IdentitySubstrateError("archive member path is unsafe or ambiguous")
    return value


def _pin(pin: PackagePin) -> dict:
    if not isinstance(pin, PackagePin) or not isinstance(pin.members_sha256, Mapping):
        raise IdentitySubstrateError("an explicit reviewed PackagePin is required")
    root = _member(pin.package_root)
    members = {_member(name): _digest(digest, "member pin")
               for name, digest in pin.members_sha256.items()}
    if not members or any(not name.startswith(root + "/") for name in members):
        raise IdentitySubstrateError("pin members must belong to its explicit package root")
    required = {root + "/" + path for path in [*ACTIVE_FILES.values(), UNKNOWN_FILE,
                ALTERNATIVE_FILE, "curation_manifest.json", "SHA256SUMS.txt"]}
    if not required <= set(members):
        raise IdentitySubstrateError("pin omits required curated-frontier members")
    return {"schema": PIN_SCHEMA, "archive_sha256": _digest(pin.archive_sha256, "archive pin"),
            "package_root": root, "members_sha256": dict(sorted(members.items()))}


def _pairs(pairs: list[tuple]) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise IdentitySubstrateError("JSON contains duplicate field names")
        output[key] = value
    return output


def _json(data: bytes, label: str) -> object:
    try:
        return json.loads(data.decode("utf-8-sig"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise IdentitySubstrateError(f"invalid JSON in {label}") from exc


def load_package_pin(path: str | Path) -> PackagePin:
    value = _json(_file(path).read_bytes(), "package pin")
    if not isinstance(value, dict) or set(value) != {"schema", "archive_sha256",
                                                       "package_root", "members_sha256"} \
            or value["schema"] != PIN_SCHEMA:
        raise IdentitySubstrateError("invalid package pin schema")
    pin = PackagePin(value["archive_sha256"], value["package_root"], value["members_sha256"])
    _pin(pin)
    return pin


def curated_frontier_v2_pin() -> PackagePin:
    """Return public receipt/checksum metadata; never open the private package."""
    return load_package_pin(Path(__file__).with_name("frontier_v2_pin.json"))


def _no_links(path: Path) -> None:
    if any(part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction())
           for part in (path, *path.parents)):
        raise IdentitySubstrateError("source and output paths must not traverse filesystem links")


def _file(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise IdentitySubstrateError("source file requires a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise IdentitySubstrateError("source file path must be absolute")
    _no_links(path)
    if not path.is_file():
        raise IdentitySubstrateError("source file must be a regular file")
    return path.resolve(strict=True)


def _hash_stream(stream) -> str:
    digest = sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return _hash_stream(stream)


def _implementation() -> list[dict]:
    root = Path(__file__).resolve().parent
    names = ("__init__.py", "compiler.py", "frontier_v2_pin.json")
    return [{"path": name, "sha256": _file_hash(_file(root / name))} for name in names]


def _bool(row: dict, field: str, *, default: bool = False) -> bool:
    value = row.get(field, default)
    if type(value) is not bool:
        raise IdentitySubstrateError(f"{field} must be a Boolean")
    return value


def _flags(row: dict) -> None:
    for field in _BOOL_FIELDS & row.keys():
        _bool(row, field)
    if "training_authority" in row and "model_training_authority" in row \
            and row["training_authority"] != row["model_training_authority"]:
        raise IdentitySubstrateError("row training authority flags conflict")


def _authority(row: dict) -> bool:
    _flags(row)
    return _bool(row, "training_authority", default=_bool(row, "model_training_authority"))


def _strings(value: object, label: str, *, scalar: bool = False) -> list[str]:
    if value is None:
        return []
    if scalar and isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise IdentitySubstrateError(f"{label} must be an array of nonempty strings")
    if len(set(value)) != len(value):
        raise IdentitySubstrateError(f"{label} contains duplicate identifiers or labels")
    return list(value)


def _refs(row: dict, field: str) -> list[str]:
    return [_id(value, field) for value in _strings(row.get(field), field)]


def _record_id(kind: str, row: dict) -> str:
    fields = {"E0": ("unit_id", "substrate_id"), "EINF": ("curated_id", "source_proposal_id"),
              "ASYN_DIRECT": ("curated_id", "source_proposal_id"), "ASYN_BASE": ("curated_base_id",),
              "ASYN_TARGETED": ("curated_id", "proposal_id"), "ASYN_CONTEXT": ("curated_id", "proposal_id")}
    for field in fields[kind]:
        if field in row:
            return _id(row[field], "active record ID")
    raise IdentitySubstrateError("active record lacks a stable ID")


def _loss_mask(row: dict, lanes: list[str]) -> dict:
    mask = row.get("loss_mask", {})
    if not isinstance(mask, dict) or any(not isinstance(key, str) or not key or type(value) is not bool
                                       for key, value in mask.items()):
        raise IdentitySubstrateError("loss_mask requires named Boolean values")
    context_only = {"context_only_conditioning", "exclude_from_identity_loss"} & set(lanes)
    if context_only and any(value and "identity" in key for key, value in mask.items()):
        raise IdentitySubstrateError("context-only or excluded evidence cannot enable identity loss")
    return dict(sorted(mask.items()))


def _normalize(kind: str, row: dict) -> dict:
    _flags(row)
    authority = _authority(row)
    # A row may declare source eligibility while global package acceptance is
    # pending. Preserve that declaration; it is not a compiler permission to
    # train, and does not change the independent global receipt flags.
    provenance = row.get("provenance_class")
    expected = {"E0": {"E0"}, "EINF": {"E-INF", "EINF"}}.get(kind, {"A-SYN", "ASYN"})
    if provenance not in expected:
        raise IdentitySubstrateError("active source kind and provenance class disagree")
    is_e0 = kind == "E0"
    if is_e0:
        text = row.get("text")
        if not isinstance(text, str) or sha256(text.encode("utf-8")).hexdigest() != \
                _digest(row.get("text_sha256"), "E0 text"):
            raise IdentitySubstrateError("canonical E0 text hash is invalid")
        if "historical_truth_allowed" not in row:
            raise IdentitySubstrateError("canonical E0 requires an explicit historical truth flag")
        historical = _bool(row, "historical_truth_allowed")
        lived = _bool(row, "alice_lived_memory")
        lanes = _strings(row.get("use_lanes"), "use_lanes")
        policy, scenario = None, None
    else:
        text = None
        historical = _bool(row, "historical_Elaina_truth")
        lived = _bool(row, "Alice_lived_memory")
        if historical or lived or _bool(row, "autobiographical_recall_allowed"):
            raise IdentitySubstrateError("inferred or synthetic policy cannot become history or Alice memory")
        lanes = _strings(row.get("recommended_supervision_lane"), "recommended_supervision_lane", scalar=True)
        policy = row.get("behavioral_proposal", row.get("base_behavioral_policy"))
        scenario = row.get("scenario", row.get("scenario_family"))
    if lived or _bool(row, "private_gradient_authorized") or _bool(row, "acceptance_authority"):
        raise IdentitySubstrateError("source compilation cannot accept Alice memory, acceptance or gradient authority")
    identity = _refs(row, "identity_supporting_E0_unit_ids")
    context = _refs(row, "context_only_E0_unit_ids")
    excluded = _refs(row, "excluded_context_E0_unit_ids")
    if (set(identity) & set(context)) or (set(identity) & set(excluded)) or (set(context) & set(excluded)):
        raise IdentitySubstrateError("identity, context-only and excluded support classes overlap")
    source_ref = row.get("record_ref") if is_e0 else row.get("source_proposal_id", row.get("proposal_id"))
    if source_ref is not None:
        source_ref = _id(source_ref, "source lineage reference")
    return {
        "record_id": _record_id(kind, row), "source_kind": kind,
        "provenance_class": "E0" if is_e0 else "E-INF" if kind == "EINF" else "A-SYN",
        "source_text": text, "scenario": scenario, "behavioral_policy": policy,
        "personality_dimensions": _strings(row.get("semantic_labels") if is_e0 else
            row.get("personality_dimensions", row.get("raw_personality_dimensions")), "personality dimensions"),
        "relationship_context": row.get("relationship_context", row.get("relationship_contexts",
                                                              row.get("relationship_contexts_seen"))),
        "emotional_context": row.get("emotional_context", row.get("emotional_contexts_seen")),
        "social_context": row.get("social_context"),
        "boundary_conditions": row.get("boundary_conditions", []),
        "disconfirmation_conditions": row.get("disconfirmation_conditions", []),
        "identity_supporting_e0_ids": identity, "context_only_e0_ids": context,
        "excluded_context_e0_ids": excluded,
        "supporting_curated_einf_ids": _refs(row, "supporting_curated_EINF_ids"),
        "supporting_raw_einf_ids": _refs(row, "supporting_raw_EINF_ids"),
        "supporting_einf_ids": _refs(row, "supporting_curated_EINF_ids") + _refs(row, "supporting_raw_EINF_ids"),
        "uncertainty_reason": row.get("uncertainty_reason"),
        "confidence_or_support_strength": row.get("confidence_or_support_strength"),
        "historical_truth_allowed": historical,
        "runtime_behavioral_prior_allowed": _bool(row, "runtime_behavioral_prior_allowed"),
        "alice_lived_memory": lived,
        "autobiographical_recall_allowed": _bool(row, "autobiographical_recall_allowed"),
        "supervision_lanes": lanes, "loss_mask": _loss_mask(row, lanes),
        "training_authority": authority,
        "declared_authority_flags": {field: row[field] for field in sorted(_BOOL_FIELDS & row.keys())},
        "owner_final_review_required": _bool(row, "owner_final_review_required", default=True),
        "coverage_cell_id": row.get("coverage_cell_id"),
        "related_active_ids": _refs(row, "related_curated_ASYN_ids"), "source_ref": source_ref,
        "declared_family_refs": {name: _id(row[name], name) for name in
            ("source_family_id", "generator_family_id", "template_family_id") if name in row},
    }


def _registry(value: Mapping | None) -> dict:
    value = {} if value is None else value
    if not isinstance(value, Mapping) or set(value) - {"EINF"}:
        raise IdentitySubstrateError("raw lineage registry supports the explicit EINF namespace")
    refs = _strings(value.get("EINF", []), "raw EINF registry")
    return {"EINF": sorted(_id(item, "raw EINF registry ID") for item in refs)}


def _families(records: list[dict], registry: dict, seed: str) -> tuple[list[dict], list[dict]]:
    by_id = {record["record_id"]: record for record in records}
    if len(by_id) != len(records):
        raise IdentitySubstrateError("duplicate active record IDs")
    parent = {"active:" + rid: "active:" + rid for rid in by_id}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    edges = []
    for record in records:
        origin = "active:" + record["record_id"]
        for field, edge_type in (("identity_supporting_e0_ids", "identity_support"),
                                 ("context_only_e0_ids", "context_only"),
                                 ("excluded_context_e0_ids", "excluded_context"),
                                 ("supporting_curated_einf_ids", "curated_inference_support"),
                                 ("related_active_ids", "derived_variant")):
            required_kind = "EINF" if field == "supporting_curated_einf_ids" else "ASYN" \
                if field == "related_active_ids" else "E0"
            for target in record[field]:
                if target not in by_id or not by_id[target]["source_kind"].startswith(required_kind):
                    raise IdentitySubstrateError(f"unresolved or wrong-namespace {required_kind} support")
                union(origin, "active:" + target)
                edges.append({"from_record_id": record["record_id"], "to_record_id": target,
                              "target_namespace": "active", "edge_type": edge_type})
        for target in record["supporting_raw_einf_ids"]:
            if target not in registry["EINF"]:
                raise IdentitySubstrateError("raw EINF support is absent from the explicit lineage registry")
            union(origin, "raw-EINF:" + target)
            edges.append({"from_record_id": record["record_id"], "to_record_id": target,
                          "target_namespace": "raw-EINF", "edge_type": "raw_inference_support"})
        if record["source_ref"]:
            namespace = "source-E0:" if record["source_kind"] == "E0" else \
                        "raw-EINF:" if record["source_kind"] == "EINF" else "source-ASYN:"
            union(origin, namespace + record["source_ref"])
        for name, value in record["declared_family_refs"].items():
            union(origin, "declared-" + name + ":" + value)
    groups = defaultdict(list)
    for rid in by_id:
        groups[find("active:" + rid)].append(rid)
    families = []
    for values in sorted(groups.values(), key=lambda group: sorted(group)):
        values = sorted(values)
        family_id = sha256(_canonical(values)).hexdigest()
        bucket = int(sha256((seed + ":" + family_id).encode()).hexdigest()[:16], 16) % 1000
        split = "test" if bucket < 50 else "dev" if bucket < 100 else "train"
        for rid in values:
            by_id[rid]["split"], by_id[rid]["source_family_id"] = split, family_id
        families.append({"family_id": family_id, "split": split, "record_ids": values})
    return edges, families


def _jsonl(archive: zipfile.ZipFile, name: str) -> list[dict]:
    rows = []
    with archive.open(name) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = _json(line, f"declared JSONL member line {line_number}")
            if not isinstance(row, dict):
                raise IdentitySubstrateError("source JSONL rows must be objects")
            rows.append(row)
    return rows


def _verify_members(archive: zipfile.ZipFile, pin: dict) -> None:
    entries = {}
    for entry in archive.infolist():
        name = _member(entry.filename.rstrip("/") if entry.is_dir() else entry.filename)
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or mode == stat.S_IFLNK:
            raise IdentitySubstrateError("archive links and special files are forbidden")
        if entry.flag_bits & 1:
            raise IdentitySubstrateError("encrypted archive members are not admissible")
        if entry.is_dir():
            if not any(member.startswith(name + "/") for member in pin["members_sha256"]):
                raise IdentitySubstrateError("unexpected archive directory")
            continue
        if name in entries:
            raise IdentitySubstrateError("duplicate archive member names")
        entries[name] = entry
    if set(entries) != set(pin["members_sha256"]):
        raise IdentitySubstrateError("archive membership differs from the exact package pin")
    for name, expected in pin["members_sha256"].items():
        with archive.open(entries[name]) as stream:
            if _hash_stream(stream) != expected:
                raise IdentitySubstrateError("archive member hash differs from the exact package pin")
    covered = set()
    checksum_files = {name for name in entries if PurePosixPath(name).name == "SHA256SUMS.txt"}
    for sums in checksum_files:
        seen = set()
        for line in archive.read(sums).decode("utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise IdentitySubstrateError("invalid internal checksum line")
            expected, relative = parts
            target = str(PurePosixPath(sums).parent / _member(relative.lstrip("*")))
            if target in seen or target not in entries or _digest(expected, "internal checksum") \
                    != pin["members_sha256"][target]:
                raise IdentitySubstrateError("internal checksum membership or digest conflicts with package pin")
            seen.add(target)
            covered.add(target)
    if covered | checksum_files != set(entries):
        raise IdentitySubstrateError("internal checksums do not cover the complete archive membership")


def _compile(archive: zipfile.ZipFile, pin: dict, registry: dict, seed: str) -> tuple[dict, dict]:
    root = pin["package_root"] + "/"
    manifest = _json(archive.read(root + "curation_manifest.json"), "curation manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("authority_flags"), dict):
        raise IdentitySubstrateError("curation manifest requires authority flags")
    flags = manifest["authority_flags"]
    if not _FLAGS <= flags.keys() or any(type(value) is not bool for value in flags.values()):
        raise IdentitySubstrateError("manifest authority flags must be explicit Booleans")
    if any(flags[name] for name in ("E0_mutated", "candidate_promotion_to_E0_performed",
                                     "model_training_performed", "weights_created")):
        raise IdentitySubstrateError("source is not an unchanged pre-weight candidate frontier")
    rows = {kind: _jsonl(archive, root + path) for kind, path in ACTIVE_FILES.items()}
    records = [_normalize(kind, row) for kind, source in rows.items() for row in source]
    if not records:
        raise IdentitySubstrateError("active identity source is empty")
    unknown = _jsonl(archive, root + UNKNOWN_FILE)
    alternatives = _jsonl(archive, root + ALTERNATIVE_FILE)
    for row in unknown:
        if _authority(row) or any(_bool(row, flag) for flag in
            ("private_gradient_authorized", "acceptance_authority", "historical_truth_allowed",
             "historical_Elaina_truth", "Alice_lived_memory", "alice_lived_memory",
             "autobiographical_recall_allowed", "hard_negative_authorized", "is_training_negative", "is_negative")):
            raise IdentitySubstrateError("historical UNKNOWN cannot carry historical, memory or learning authority")
    for row in alternatives:
        if _authority(row) or _bool(row, "hard_negative_authorized") \
                or any(_bool(row, flag) for flag in ("private_gradient_authorized", "acceptance_authority",
                                                     "is_training_negative", "is_negative")):
            raise IdentitySubstrateError("unordered alternatives cannot be authorized training negatives")
    observed = {"curated_EINF": len(rows["EINF"]), "curated_direct_ASYN_carried": len(rows["ASYN_DIRECT"]),
                "factorized_base_policies": len(rows["ASYN_BASE"]), "targeted_ASYN_added": len(rows["ASYN_TARGETED"]),
                "context_variants_added": len(rows["ASYN_CONTEXT"]), "historical_UNKNOWN_bank": len(unknown),
                "targeted_gap_queue_remaining": 0}
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or not observed.keys() <= counts.keys():
        raise IdentitySubstrateError("curation manifest omits required source counts or gap status")
    if any(type(counts[key]) is not int or counts[key] != count for key, count in observed.items()):
        raise IdentitySubstrateError("curation manifest count or closed-gap status mismatch")
    edges, families = _families(records, registry, seed)
    concepts = defaultdict(Counter)
    for record in records:
        for value in record["personality_dimensions"]:
            concepts["personality_or_semantic_dimension"][value] += 1
    inventory = {category: [{"value": value, "count": count} for value, count in sorted(counter.items())]
                 for category, counter in sorted(concepts.items())}
    records.sort(key=lambda row: row["record_id"])
    edges.sort(key=lambda row: _canonical(row))
    output = {"active_identity_records.jsonl": records, "support_edges.jsonl": edges,
              "historical_unknown_bank.jsonl": unknown,
              "alternative_competitors_unordered.jsonl": alternatives,
              "observed_concept_inventory.json": inventory, "split_families.jsonl": families}
    split_counts = dict(sorted(Counter(row["split"] for row in records).items()))
    info = {"active_record_count": len(records),
            "kind_counts": dict(sorted(Counter(row["source_kind"] for row in records).items())),
            "split_counts": split_counts, "source_family_count": len(families),
            "family_split_counts": dict(sorted(Counter(row["split"] for row in families).items())),
            "largest_family_record_count": max(len(row["record_ids"]) for row in families),
            "heldout_family_coverage_complete": all(split_counts.get(name, 0) > 0 for name in ("train", "dev", "test")),
            "support_edge_count": len(edges), "historical_unknown_count": len(unknown),
            "alternative_competitor_count": len(alternatives),
            "training_authority_counts": dict(sorted(Counter(str(row["training_authority"]).lower()
                                                               for row in records).items())),
            "source_authority_flags": flags,
            "source_training_authority_granted": flags["training_authority_granted"],
            "source_owner_final_preweight_review_required": flags["owner_final_preweight_review_required"]}
    return output, info


def audit_loss_role_structure(package_path: str | Path, *, pin: PackagePin) -> dict:
    """Return only fixed aggregate schema diagnostics after exact source custody.

    No source text, IDs, unrecognized keys or values leave this audit. The
    aggregate predicates diagnose compiler assumptions, never approve a lane.
    The caller must establish private execution isolation before invoking it.
    """
    package, binding = _file(package_path), _pin(pin)
    if _file_hash(package) != binding["archive_sha256"]:
        raise IdentitySubstrateError("archive SHA256 differs from the explicit package pin")
    keys = ("direct_identity", "identity_core", "identity_loss", "exclude_from_identity_loss")
    counts = {name: 0 for name in ("e0_rows", "legacy_conflicting_rows",
        "context_only_rows", "exclusion_lane_rows", "context_and_direct_supervision_rows",
        "conflicting_positive_direct_identity_rows", "conflicting_positive_identity_core_rows",
        "conflicting_identity_exclusion_name_rows", "conflicting_other_identity_name_rows")}
    counts["enabled_known_mask_fields"] = {key: 0 for key in keys}
    with zipfile.ZipFile(package) as archive:
        _verify_members(archive, binding)
        rows = _jsonl(archive, binding["package_root"] + "/" + ACTIVE_FILES["E0"])
        for row in rows:
            lanes, mask = row.get("use_lanes", []), row.get("loss_mask", {})
            if not isinstance(lanes, list) or any(not isinstance(value, str) for value in lanes) \
                    or not isinstance(mask, dict) or any(not isinstance(key, str) or type(value) is not bool
                                                       for key, value in mask.items()):
                raise IdentitySubstrateError("loss role structural audit requires explicit schema types")
            counts["e0_rows"] += 1
            context = "context_only_conditioning" in lanes
            excluded = "exclude_from_identity_loss" in lanes
            conflict_keys = [key for key, value in mask.items() if value and "identity" in key]
            counts["context_only_rows"] += int(context)
            counts["exclusion_lane_rows"] += int(excluded)
            counts["context_and_direct_supervision_rows"] += int(context and "direct_identity_supervision" in lanes)
            for key in keys:
                counts["enabled_known_mask_fields"][key] += int(mask.get(key) is True)
            if (context or excluded) and conflict_keys:
                counts["legacy_conflicting_rows"] += 1
                counts["conflicting_positive_direct_identity_rows"] += int(mask.get("direct_identity") is True)
                counts["conflicting_positive_identity_core_rows"] += int(mask.get("identity_core") is True)
                counts["conflicting_identity_exclusion_name_rows"] += int(any("exclu" in key.lower() for key in conflict_keys))
                counts["conflicting_other_identity_name_rows"] += int(any(key not in keys and "exclu" not in key.lower()
                                                                         for key in conflict_keys))
    if _file_hash(package) != binding["archive_sha256"]:
        raise IdentitySubstrateError("source archive changed during compilation")
    return {"schema": "alice-personality-source-loss-structure-diagnostic-v1",
            "acceptance_authority": False, "training_authorized": False, "counts": counts}


def compile_package(package_path: str | Path, output_dir: str | Path, *, pin: PackagePin,
                    split_seed: str = "alice-eipm-evidence-families-v1",
                    raw_lineage_registry: Mapping | None = None) -> dict:
    """Compile to a new immutable artifact directory; no acceptance or gradient.

    Exact hash/membership pins are mandatory. Raw inference references require
    an explicit namespace registry, whose content hash is bound in the receipt.
    Failed publication may leave a partial directory without a valid receipt;
    retry must use a new directory. Source payloads never appear in exceptions.
    """
    package = _file(package_path)
    binding = _pin(pin)
    registry = _registry(raw_lineage_registry)
    seed = _id(split_seed, "split seed")
    output = Path(output_dir).expanduser()
    if not output.is_absolute():
        raise IdentitySubstrateError("compiled output directory must be absolute")
    _no_links(output)
    if output.exists() or not output.parent.is_dir() or output == package:
        raise IdentitySubstrateError("compiled output requires a new directory under an existing parent")
    implementation = _implementation()
    initial = package.stat()
    with package.open("rb") as stream:
        if _hash_stream(stream) != binding["archive_sha256"]:
            raise IdentitySubstrateError("archive SHA256 differs from the explicit package pin")
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream) as archive:
                _verify_members(archive, binding)
                artifacts, info = _compile(archive, binding, registry, seed)
        except (zipfile.BadZipFile, UnicodeError, KeyError, RuntimeError, OSError) as exc:
            raise IdentitySubstrateError("source archive cannot satisfy governed compilation") from exc
        stream.seek(0)
        if _hash_stream(stream) != binding["archive_sha256"]:
            raise IdentitySubstrateError("source archive changed during compilation")
    final = package.stat()
    if (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns) != \
            (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) or implementation != _implementation():
        raise IdentitySubstrateError("source file or compiler implementation changed during compilation")
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise IdentitySubstrateError("compiled output already exists; use a new artifact path") from exc
    files = []
    for name, value in sorted(artifacts.items()):
        data = _canonical(value) + b"\n" if name.endswith(".json") else \
            b"".join(_canonical(row) + b"\n" for row in value)
        with (output / name).open("xb") as stream:
            stream.write(data)
        files.append({"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()})
    receipt = {"schema": SCHEMA, "state": STATE, "source_package_path": str(package),
               "source_package_sha256": binding["archive_sha256"], "package_pin": binding,
               "package_pin_sha256": sha256(_canonical(binding)).hexdigest(),
               "compiler_implementation": implementation, "split_seed": seed,
               "raw_lineage_registry": registry,
               "raw_lineage_registry_sha256": sha256(_canonical(registry)).hexdigest(),
               "private_gradient_authorized": False, "acceptance_authority": False,
               "row_authority_semantics": "declared_source_eligibility_not_package_acceptance",
               "behavior_qualification": None, "alternatives_are_unordered_not_negatives": True,
               "unknown_is_behavior_void": False, "voice_overlay_ingested": False,
               "files": files, **info}
    receipt["receipt_sha256"] = sha256(_canonical(receipt)).hexdigest()
    with (output / "compile_receipt.json").open("xb") as stream:
        stream.write(_canonical(receipt) + b"\n")
    return receipt


def verify_compiled(output_dir: str | Path, *, expected_source_archive_sha256: str | None = None,
                    expected_receipt_sha256: str | None = None) -> dict:
    """Rehash compiled/source/code bindings without approving personality data."""
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise IdentitySubstrateError("compiled directory must be absolute")
    _no_links(root)
    receipt = _json(_file(root / "compile_receipt.json").read_bytes(), "compile receipt")
    if not isinstance(receipt, dict):
        raise IdentitySubstrateError("compile receipt must be an object")
    supplied = _digest(receipt.pop("receipt_sha256", None), "compile receipt")
    if sha256(_canonical(receipt)).hexdigest() != supplied:
        raise IdentitySubstrateError("compile receipt digest mismatch")
    if expected_receipt_sha256 is not None and supplied != _digest(expected_receipt_sha256, "expected receipt"):
        raise IdentitySubstrateError("compiled receipt differs from the caller-bound artifact")
    receipt["receipt_sha256"] = supplied
    if receipt.get("schema") != SCHEMA or receipt.get("state") != STATE \
            or receipt.get("behavior_qualification") is not None \
            or receipt.get("private_gradient_authorized") is not False \
            or receipt.get("acceptance_authority") is not False \
            or receipt.get("row_authority_semantics") != "declared_source_eligibility_not_package_acceptance" \
            or receipt.get("voice_overlay_ingested") is not False \
            or receipt.get("unknown_is_behavior_void") is not False \
            or receipt.get("alternatives_are_unordered_not_negatives") is not True:
        raise IdentitySubstrateError("compilation cannot claim data acceptance, personality qualification or gradient")
    pin = receipt.get("package_pin")
    if not isinstance(pin, dict) or set(pin) != {"schema", "package_root", "archive_sha256", "members_sha256"} \
            or pin["schema"] != PIN_SCHEMA:
        raise IdentitySubstrateError("invalid compiled package pin")
    if pin != _pin(PackagePin(pin["archive_sha256"], pin["package_root"], pin["members_sha256"])) \
            or receipt.get("package_pin_sha256") != sha256(_canonical(pin)).hexdigest():
        raise IdentitySubstrateError("compiled package pin binding changed")
    expected = pin["archive_sha256"]
    if receipt.get("source_package_sha256") != expected or (expected_source_archive_sha256 is not None
            and _digest(expected_source_archive_sha256, "expected source archive") != expected):
        raise IdentitySubstrateError("compiled source archive identity differs")
    source = _file(receipt.get("source_package_path"))
    if _file_hash(source) != expected:
        raise IdentitySubstrateError("compiled source archive changed")
    if receipt.get("compiler_implementation") != _implementation():
        raise IdentitySubstrateError("identity compiler implementation changed")
    registry = _registry(receipt.get("raw_lineage_registry"))
    if receipt.get("raw_lineage_registry_sha256") != sha256(_canonical(registry)).hexdigest():
        raise IdentitySubstrateError("compiled raw lineage binding changed")
    files = receipt.get("files")
    if not isinstance(files, list) or len(files) != len(_ARTIFACTS):
        raise IdentitySubstrateError("compiled artifact inventory is incomplete")
    if {row.get("path") for row in files if isinstance(row, dict)} != _ARTIFACTS \
            or {path.name for path in root.iterdir()} != _ARTIFACTS | {"compile_receipt.json"}:
        raise IdentitySubstrateError("compiled artifact membership changed")
    for row in files:
        if set(row) != {"path", "bytes", "sha256"} or type(row["bytes"]) is not int:
            raise IdentitySubstrateError("invalid compiled artifact metadata")
        path = _file(root / row["path"])
        if path.stat().st_size != row["bytes"] or _file_hash(path) != _digest(row["sha256"], "compiled artifact"):
            raise IdentitySubstrateError("compiled artifact content changed")
    flags = receipt.get("source_authority_flags")
    if not isinstance(flags, dict) or not _FLAGS <= flags.keys() or any(type(value) is not bool for value in flags.values()):
        raise IdentitySubstrateError("compiled source authority flags must be Booleans")
    with zipfile.ZipFile(source) as archive:
        source_manifest = _json(archive.read(pin["package_root"] + "/curation_manifest.json"), "source manifest")
    if flags != source_manifest.get("authority_flags") \
            or receipt.get("source_training_authority_granted") is not flags["training_authority_granted"] \
            or receipt.get("source_owner_final_preweight_review_required") is not flags["owner_final_preweight_review_required"]:
        raise IdentitySubstrateError("compiled source authority claims differ from the pinned source manifest")
    return receipt
