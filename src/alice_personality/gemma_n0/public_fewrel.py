"""Offline admission of one pinned, public FewRel TRAIN/DEV artifact pair.

Only four explicitly named files are read. Publisher data and historical FINAL
artifacts are never opened. Admission checks provenance, split integrity and
bytes; it does not authorize training, establish heldout competence or approve
Gemma/personality behavior. Opaque relation keys and entity IDs stay metadata.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import unicodedata


SCHEMA = "alice-personality-gemma-n0-public-fewrel-admission-v1"
STATE = "ELIGIBLE_PUBLIC_SOURCE_UNQUALIFIED"
SOURCE_ID = "thunlp_fewrel_1_0"
SOURCE_REVISION = "278a2315d2138810a379cd8d5718914dc56e2582"
SPLIT_RULE = "alice-n0-fewrel-train-dev-family-split-v1"
SOURCE_BLOBS = {
    "LICENSE": "6fec9b6238e7b4f2037aefcf4110821a2fd16c70",
    "data/pid2name.json": "00b632be58e00b5337c054ac8a46bda1f4c27b0e",
    "data/train_wiki.json": "4fee283724b455bab1fcbf76bfdef02283bb44f5",
    "data/val_wiki.json": "fe103ba5f52de1f0c070d66c7a8ba9de20c9f96c",
}
# External, reviewed 78-file V2 inventory pins. These are this artifact's
# identity, never a row/cardinality ceiling for future personality capability.
ARTIFACT_PINS = {
    "rows": {"name": "train_dev_rows.jsonl", "size": 67814760,
             "sha256": "0841234b78b8c7dfc3a4cc7b0ca56c251ad6077053ed61e4e6a7eea2a7b62ec5"},
    "bank": {"name": "train_dev_bank.json", "size": 22340,
             "sha256": "1646ccc6e98aca107b377657ffa918c10a7b7f7ea63a395dd4ca4ca3c7dcab10"},
    "manifest": {"name": "manifest.json", "size": 2134,
                 "sha256": "4364e585f53a1712f15ea4fc81285acdbf6548dcaab7999fca49f67b41ad9ba8"},
    "audit": {"name": "audit.json", "size": 1560,
              "sha256": "7f5eb9fbabaaef40b020a328c17ba2ce12dd8a2a7c42b5d15eeaf36dbb0d4dca"},
}
EXPECTED_COUNTS = {"train_rows": 39200, "dev_rows": 5600,
                   "train_relation_count": 56, "dev_relation_count": 8,
                   "official_training_relation_count": 64}
PROVENANCE = {
    "repository": "thunlp/FewRel", "revision": SOURCE_REVISION,
    "license": "MIT", "attribution": "Copyright (c) 2018 THUNLP",
    "source_git_blob_sha1": SOURCE_BLOBS,
    "materializer_source_revision": "5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2",
    "transfer_manifest_sha256": "2d83e34f58918ad36d31142ba0e6f77447f2d647c6edc786f06678502f010b7d",
    "full_mixture_manifest_sha256": "c5f18c6f7ebf8c1a198cd420740966b5e9c7ffe9143b1125c06edfb94de51b97",
    "label_authority": "historically receipted natural human relation annotation",
    "license_bytes_opened_by_this_admission": False,
}
IMPLEMENTATION_PATHS = {
    "public_fewrel.py": Path(__file__),
    "admit_public_fewrel.py": Path(__file__).resolve().parents[3]
        / "scripts/eipm/gemma_n0/admit_public_fewrel.py",
}
ROW_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-row.v1"
BANK_SCHEMA = "alice.eipm.n0.fewrel-runtime-relation-bank.v1"
MANIFEST_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-manifest.v3"
AUDIT_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-audit.v3"
INSTRUCTION = ("Given the natural sentence and marked head/tail entities, "
               "select the supplied relation description that best expresses "
               "how the head entity relates to the tail entity.")
_PROPERTY = re.compile(r"\bP\d+\b")
_KEY = re.compile(r"P[1-9]\d*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ROW_KEYS = {"schema", "id", "split", "source_id", "source_revision",
             "natural_source_text", "generated_instruction_only", "generated_relation_label",
             "sentence", "tokens", "head", "tail", "instruction", "candidate_relation_keys",
             "target_relation_key", "target_candidate_index", "runtime_relation_count",
             "training_authorized", "model_selection_authorized", "final_validation_only",
             "relation_keys_are_metadata_only", "private_identity_data"}


class PublicSourceError(ValueError):
    """A public source violated custody, provenance or TRAIN/DEV isolation."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicSourceError(message)


def _file(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    _require(path.is_absolute() and not path.is_symlink() and path.is_file(),
             "inputs must be absolute regular nonsymlink files")
    resolved = path.resolve(strict=True)
    _require(path == resolved, "input ancestors must not redirect through symlinks")
    return resolved


def _identity(path: Path) -> tuple:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _digest(path: Path) -> tuple[int, str]:
    before = _identity(path)
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    _require(before == _identity(path), "source changed while hashing")
    return before[2], digest.hexdigest()


def _json(payload: str | bytes) -> object:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            _require(key not in out, "duplicate JSON field")
            out[key] = value
        return out
    try:
        return json.loads(payload, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              PublicSourceError("nonfinite JSON value")))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PublicSourceError("invalid source JSON") from exc


def _bound_files(paths: dict) -> tuple[dict, list[dict]]:
    _require(set(paths) == set(ARTIFACT_PINS), "admission requires exactly four public files")
    files, records = {}, []
    for kind in ("rows", "bank", "manifest", "audit"):
        path = _file(paths[kind])
        pin = ARTIFACT_PINS[kind]
        _require(path.name == pin["name"], "input is not an allowlisted TRAIN/DEV artifact")
        _require(path.stat().st_size == pin["size"], f"{kind} size differs from external pin")
        size, digest = _digest(path)
        _require(digest == pin["sha256"], f"{kind} SHA256 differs from external pin")
        files[kind] = path
        records.append({"kind": kind, "path": str(path), "size": size, "sha256": digest})
    _require(len({path.parent for path in files.values()}) == 1,
             "the four receipted files must share one isolated source directory")
    return files, records


def _metadata(files: dict) -> tuple[dict, dict, dict]:
    bank, manifest, audit = (_json(files[key].read_bytes())
                             for key in ("bank", "manifest", "audit"))
    _require(all(isinstance(value, dict) for value in (bank, manifest, audit)),
             "public bank/receipts must be JSON objects")
    _require(manifest.get("schema") == MANIFEST_SCHEMA and audit.get("schema") == AUDIT_SCHEMA,
             "historical receipt schema differs")
    _require(manifest.get("status") ==
             "MATERIALIZED_NATURAL_RELATION_CURRICULUM_WITH_OFFICIAL_VALIDATION_FINAL_FAMILIES"
             and audit.get("status") == "PASS_FEWREL_NATURAL_RELATION_AUDIT_V3"
             and audit.get("errors") == [], "historical materialization/audit did not pass")
    _require(manifest.get("source_id") == SOURCE_ID
             and manifest.get("source_revision") == SOURCE_REVISION
             and manifest.get("source_git_blob_sha1") == SOURCE_BLOBS
             and manifest.get("license") == "MIT"
             and manifest.get("dev_family_split_rule") == SPLIT_RULE,
             "public provenance/license/family split differs")
    for receipt in (manifest, audit):
        for key, expected in EXPECTED_COUNTS.items():
            if key == "official_training_relation_count" and receipt is audit:
                continue
            _require(type(receipt.get(key)) is int and receipt[key] == expected,
                     f"historical {key} differs")
        for key in ("natural_source_text", "human_relation_labels",
                    "final_relation_descriptions_absent_from_train_dev_bank"):
            _require(receipt.get(key) is True, f"historical {key} must be true")
        for key in ("private_identity_data", "final_training_authorized",
                    "final_model_selection_authorized"):
            _require(receipt.get(key) is False, f"historical {key} must be false")
        for kind in ("rows", "bank"):
            _require(receipt.get(f"train_dev_{kind}_sha256") == ARTIFACT_PINS[kind]["sha256"],
                     "historical TRAIN/DEV digest differs from external pin")
    _require(manifest.get("relation_ids_omitted_from_manifest") is True
             and manifest.get("final_rows_separate_artifact") is True
             and manifest.get("all_official_validation_relation_families_are_final") is True
             and manifest.get("final_family_subset_selected_after_observation") is False
             and manifest.get("operating_cap_is_capability_ceiling") is False
             and type(manifest.get("max_per_relation_operating_cap")) is int
             and manifest["max_per_relation_operating_cap"] == 0,
             "historical public/final/capability boundary differs")
    _require(audit.get("training_authorized_by_audit") is False
             and isinstance(audit.get("relation_overlap"), dict)
             and audit["relation_overlap"].get("train_dev") == []
             and isinstance(audit.get("source_overlap"), dict)
             and type(audit["source_overlap"].get("train_dev")) is int
             and audit["source_overlap"]["train_dev"] == 0,
             "historical audit does not preserve TRAIN/DEV isolation")
    expected_bank_keys = {"schema", "source_id", "source_revision", "role", "relations",
                          "relation_keys_are_metadata_only", "semantic_text_field",
                          "private_identity_data"}
    _require(set(bank) == expected_bank_keys and bank["schema"] == BANK_SCHEMA
             and bank["source_id"] == SOURCE_ID and bank["source_revision"] == SOURCE_REVISION
             and bank["role"] == "TRAIN_AND_MODEL_SELECTION_ONLY_NO_FINAL_RELATIONS"
             and bank["relation_keys_are_metadata_only"] is True
             and bank["private_identity_data"] is False and bank["semantic_text_field"] == "semantic_text"
             and isinstance(bank["relations"], dict), "bank is not the public TRAIN/DEV semantic bank")
    relations = bank["relations"]
    _require(len(relations) == EXPECTED_COUNTS["official_training_relation_count"],
             "bank family count differs")
    for key, relation in relations.items():
        _require(isinstance(key, str) and bool(_KEY.fullmatch(key))
                 and isinstance(relation, dict) and set(relation) == {"name", "description", "semantic_text"}
                 and all(isinstance(value, str) for value in relation.values()),
                 "invalid semantic bank relation")
        name = _PROPERTY.sub("referenced property", relation["name"]).strip()
        description = _PROPERTY.sub("referenced property", relation["description"]).strip()
        expected = (f"Relation meaning: {name}." if not description
                    or description.lower() == "no description defined" else
                    f"Relation name: {name}. Relation meaning: {description}")
        _require(relation["semantic_text"] == expected and bool(expected.strip())
                 and not _PROPERTY.search(expected), "opaque relation ID leaked into semantic text")
    return bank, manifest, audit


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _entity(value: object, token_count: int) -> dict:
    _require(isinstance(value, dict) and set(value) == {"text", "type", "token_indices"},
             "row entity schema differs")
    _require(isinstance(value["text"], str) and bool(value["text"].strip())
             and isinstance(value["type"], str), "invalid entity metadata")
    indices = value["token_indices"]
    _require(isinstance(indices, list) and bool(indices)
             and all(type(index) is int and 0 <= index < token_count for index in indices)
             and indices == sorted(set(indices)), "entity token positions are not source-aligned")
    return value


def _rows(path: Path, bank: dict, manifest: dict, audit: dict) -> dict:
    relations = set(bank["relations"])
    ordered = sorted(relations, key=lambda key: sha256(f"{SPLIT_RULE}:{key}".encode()).hexdigest())
    dev_families = set(ordered[:EXPECTED_COUNTS["dev_relation_count"]])
    train_families = relations - dev_families
    counters = {"train": Counter(), "dev": Counter()}
    cardinalities = {"train": set(), "dev": set()}
    signatures = {"train": set(), "dev": set()}
    normalized = {"train": set(), "dev": set()}
    sentences = {"train": set(), "dev": set()}
    ids = set()
    before = _identity(path)
    streamed_digest = sha256()
    with path.open("rb") as stream:
        for line_number, payload in enumerate(stream, 1):
            streamed_digest.update(payload)
            _require(bool(payload.strip()), f"blank row at line {line_number}")
            row = _json(payload)
            _require(isinstance(row, dict) and set(row) == _ROW_KEYS, "row schema fields differ")
            split = row["split"]
            _require(isinstance(split, str) and split in {"train", "dev"},
                     "public rows include a non-TRAIN/DEV split")
            _require(row["schema"] == ROW_SCHEMA and row["source_id"] == SOURCE_ID
                     and row["source_revision"] == SOURCE_REVISION and row["instruction"] == INSTRUCTION,
                     "row provenance/schema/instruction differs")
            for key in ("natural_source_text", "generated_instruction_only", "relation_keys_are_metadata_only"):
                _require(row[key] is True, f"row {key} must be true")
            for key in ("generated_relation_label", "private_identity_data", "final_validation_only"):
                _require(row[key] is False, f"row {key} must be false")
            _require(row["training_authorized"] is (split == "train")
                     and row["model_selection_authorized"] is (split == "dev"),
                     "historical row split authority differs")
            rid = row["id"]
            _require(isinstance(rid, str) and re.fullmatch(f"fewrel:{split}:[0-9a-f]{{20}}", rid)
                     and rid not in ids, "row identifier is invalid or duplicate")
            ids.add(rid)
            tokens = row["tokens"]
            _require(isinstance(tokens, list) and bool(tokens)
                     and all(isinstance(token, str) for token in tokens)
                     and isinstance(row["sentence"], str) and bool(row["sentence"].strip())
                     and row["sentence"] == " ".join(tokens), "sentence/token reconstruction differs")
            head, tail = (_entity(row[key], len(tokens)) for key in ("head", "tail"))
            target, candidates, index = row["target_relation_key"], row["candidate_relation_keys"], row["target_candidate_index"]
            expected_families = train_families if split == "train" else dev_families
            allowed_candidates = train_families if split == "train" else relations
            _require(isinstance(target, str) and target in expected_families,
                     "target family violates the deterministic TRAIN/DEV split")
            _require(isinstance(candidates, list) and bool(candidates)
                     and all(isinstance(key, str) and key in allowed_candidates for key in candidates)
                     and len(candidates) == len(set(candidates)) and type(index) is int
                     and 0 <= index < len(candidates) and candidates[index] == target,
                     "candidate pool/target alignment differs")
            _require(type(row["runtime_relation_count"]) is int
                     and row["runtime_relation_count"] == len(candidates), "candidate cardinality differs")
            counters[split][target] += 1
            cardinalities[split].add(len(candidates))
            signatures[split].add(sha256(_canonical({"sentence": row["sentence"],
                                                     "head": head, "tail": tail})).hexdigest())
            normalized_source = {"sentence": _normalized(row["sentence"])}
            for key, entity in (("head", head), ("tail", tail)):
                normalized_source[key] = {"text": _normalized(entity["text"]),
                                          "type": _normalized(entity["type"]),
                                          "token_indices": entity["token_indices"]}
            normalized[split].add(sha256(_canonical(normalized_source)).hexdigest())
            sentences[split].add(sha256(_normalized(row["sentence"]).encode()).hexdigest())
    _require(before == _identity(path) and streamed_digest.hexdigest() == ARTIFACT_PINS["rows"]["sha256"],
             "row file changed between custody verification and complete parsing")
    for split, families in (("train", train_families), ("dev", dev_families)):
        _require(set(counters[split]) == families, "observed family coverage differs")
        _require(sum(counters[split].values()) == EXPECTED_COUNTS[f"{split}_rows"],
                 "complete row count differs from original artifact")
        expected_per_family, remainder = divmod(EXPECTED_COUNTS[f"{split}_rows"], len(families))
        _require(remainder == 0 and set(counters[split].values()) == {expected_per_family},
                 "source family completeness differs")
        for receipt in (manifest, audit):
            _require(_canonical(receipt.get(f"{split}_candidate_count_points")) ==
                     _canonical(sorted(cardinalities[split])),
                     "observed cardinality points differ from historical metadata")
    _require(not signatures["train"] & signatures["dev"], "exact TRAIN/DEV source overlap")
    _require(not normalized["train"] & normalized["dev"], "normalized TRAIN/DEV source overlap")
    _require(_canonical(audit.get("dev_unseen_candidate_count_points")) ==
             _canonical(sorted(cardinalities["dev"] - cardinalities["train"])),
             "unseen DEV cardinality metadata differs")
    return {"rows": {split: sum(counters[split].values()) for split in counters},
            "relation_families": {"train": len(train_families), "dev": len(dev_families)},
            "candidate_count_points": {split: sorted(points) for split, points in cardinalities.items()},
            "exact_train_dev_source_overlap": 0, "normalized_train_dev_source_overlap": 0,
            "normalized_sentence_only_overlap": len(sentences["train"] & sentences["dev"]),
            "normalization": "NFKC_casefold_whitespace_sentence_and_ordered_entities_with_token_positions",
            "all_train_dev_records_checked": True, "final_payload_opened": False}


def _implementation() -> list[dict]:
    _require(set(IMPLEMENTATION_PATHS) == {"public_fewrel.py", "admit_public_fewrel.py"},
             "both admission implementations must be bound")
    return [{"name": name, "sha256": _digest(_file(path))[1]}
            for name, path in sorted(IMPLEMENTATION_PATHS.items())]


def _admit(paths: dict, *, declared_public: bool) -> dict:
    _require(declared_public is True, "explicit public TRAIN/DEV declaration is required")
    implementation = _implementation()
    files, records = _bound_files(paths)
    before = {kind: _identity(path) for kind, path in files.items()}
    bank, manifest, audit = _metadata(files)
    statistics = _rows(files["rows"], bank, manifest, audit)
    _require(all(before[kind] == _identity(path) for kind, path in files.items()),
             "public artifact changed during admission")
    # The row stream checked its digest as it parsed. Rehash small metadata
    # too, including a change between initial custody and the stat snapshot.
    for kind in ("bank", "manifest", "audit"):
        _require(_digest(files[kind]) == (ARTIFACT_PINS[kind]["size"], ARTIFACT_PINS[kind]["sha256"]),
                 "public metadata differs after complete validation")
    _require(implementation == _implementation(), "admission implementation changed during validation")
    return {"schema": SCHEMA, "state": STATE, "source_id": SOURCE_ID,
            "source_revision": SOURCE_REVISION, "provenance": _json(_canonical(PROVENANCE)),
            "files": records, "implementation": implementation, "statistics": statistics,
            "public_train_dev_declared": True, "split_rule": SPLIT_RULE,
            "historical_audit_status": audit["status"],
            "relation_keys_and_entity_ids_are_metadata_only": True,
            "permitted_future_semantic_inputs": ["sentence", "head_tail_positions_and_text",
                                                 "candidate_relation_semantic_text"],
            "target_keys_indices_and_ids_are_model_input": False,
            "train_split_use": "eligible_for_separately_precommitted_public_readout_protocol",
            "dev_split_use": "selection_or_evaluation_only_never_gradient",
            "historical_pass_authorizes_new_training": False,
            "source_counts_define_capability_ceiling": False,
            "training_authorized": False, "gradient_performed": False,
            "model_forward_performed": False, "private_identity_data": False,
            "n0_approved": False, "behavior_qualification": None}


def admit_fewrel_train_dev(rows_path: str | Path, bank_path: str | Path,
                          manifest_path: str | Path, audit_path: str | Path,
                          output_receipt_path: str | Path, *, declared_public: bool) -> dict:
    """Verify exact public artifacts and write fresh source-only eligibility.

    No payloads are returned and no directory is enumerated. Frozen pins name
    this historical corpus, not an upper bound for future experiments.
    """
    paths = {"rows": rows_path, "bank": bank_path, "manifest": manifest_path, "audit": audit_path}
    output = Path(output_receipt_path).expanduser()
    _require(output.is_absolute() and not output.is_symlink() and not output.exists(),
             "source receipt must use a fresh absolute nonsymlink path")
    _require(output.parent.is_dir() and output.parent == output.parent.resolve(strict=True),
             "source receipt requires an existing nonsymlink parent")
    output = output.resolve(strict=False)
    for raw in paths.values():
        source_dir = _file(raw).parent
        _require(source_dir != output and source_dir not in output.parents,
                 "source eligibility receipt must remain outside source artifacts")
    receipt = _admit(paths, declared_public=declared_public)
    receipt["receipt_sha256"] = sha256(_canonical(receipt)).hexdigest()
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PublicSourceError("source receipt already exists") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(receipt) + b"\n")
    return receipt


def verify_fewrel_admission(receipt_path: str | Path) -> dict:
    """Recheck a source receipt and every public record before later reuse."""
    receipt_file = _file(receipt_path)
    record = _json(receipt_file.read_bytes())
    _require(isinstance(record, dict) and isinstance(record.get("receipt_sha256"), str)
             and bool(_DIGEST.fullmatch(record["receipt_sha256"])), "invalid source admission receipt")
    original = dict(record)
    digest = record.pop("receipt_sha256")
    _require(sha256(_canonical(record)).hexdigest() == digest, "source receipt digest differs")
    rows = record.get("files")
    _require(isinstance(rows, list) and len(rows) == 4
             and all(isinstance(row, dict) and set(row) == {"kind", "path", "size", "sha256"}
                     for row in rows), "invalid source receipt file bindings")
    paths = {row["kind"]: row["path"] for row in rows}
    _require(len(paths) == 4, "duplicate source receipt file binding")
    for raw in paths.values():
        _require(_file(raw).parent not in receipt_file.parents,
                 "source admission receipt must remain outside source artifacts")
    current = _admit(paths, declared_public=record.get("public_train_dev_declared"))
    _require(_canonical(current) == _canonical(record), "source admission no longer matches current artifacts/code")
    return original
