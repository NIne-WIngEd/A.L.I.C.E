"""Verify an exact public-data pack before native MFM foundation training.

The candidate inventory is a menu, never an admission. This verifier checks
local bytes and recorded item-level reviews; it cannot authenticate a grant,
the identity of a reviewer, or whether extracted bytes truly came from an
upstream archive. Those facts need a separate human/source audit. No downloads,
media decoding, training, or model qualification occur here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path


CANDIDATE_SCHEMA = "mfm-native-foundation-source-candidates-v1"
ADMISSION_SCHEMA = "mfm-native-foundation-admission-v1"
RIGHTS_SCHEMA = "mfm-native-public-item-rights-v1"
INPUTS_SCHEMA = "mfm-native-foundation-inputs-v1"
SPLITS = frozenset({"train", "development", "final"})
RESERVED_EVALUATIONS = frozenset({"LongMemEval-V2", "LoCoMo", "CareCall"})


def _fail(message: str) -> None:
    raise ValueError(message)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail(f"{name} must be a nonblank exact string")
    return value


def _digest(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _obj(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list:
    if not isinstance(value, list):
        _fail(f"{name} must be an array")
    return value


def _date(value: object, name: str) -> str:
    value = _text(value, name)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        _fail(f"{name} must be a canonical ISO date")
    return value


def _path(root: Path, value: object, name: str) -> tuple[str, Path]:
    value = _text(value, name)
    if ("\\" in value or value.startswith("/") or
            any(part in ("", ".", "..") for part in value.split("/"))):
        _fail(f"{name} escapes admission root")
    absolute = (root / value).resolve()
    if not absolute.is_relative_to(root):
        _fail(f"{name} escapes admission root")
    return value, absolute


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checked_file(root: Path, value: object, name: str,
                  cache: dict[str, str], *, open_bytes: bool = True) -> "FileRef":
    row = _obj(value, name)
    path, absolute = _path(root, row.get("path"), f"{name}.path")
    digest = _digest(row.get("sha256"), f"{name}.sha256")
    if open_bytes:
        try:
            if path not in cache:
                cache[path] = _sha256_file(absolute)
            actual = cache[path]
        except OSError as exc:
            raise ValueError(f"{name} cannot be read") from exc
        if actual != digest:
            _fail(f"{name} bytes differ from admission manifest")
    return FileRef(path, digest)


def _frozen_json(path: Path, expected_sha256: str, name: str) -> tuple[dict, str]:
    expected = _digest(expected_sha256, f"{name} expected_sha256")
    try:
        raw = path.read_bytes()
        content = _obj(json.loads(raw), name)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} cannot be read as JSON") from exc
    actual = sha256(raw).hexdigest()
    if actual != expected:
        _fail(f"{name} differs from frozen digest")
    return content, actual


@dataclass(frozen=True)
class FileRef:
    path: str
    sha256: str


@dataclass(frozen=True)
class AdmittedPublicItem:
    item_id: str
    source_id: str
    candidate_id: str
    modality: str
    split: str
    payload: FileRef
    upstream_file: FileRef
    rights: FileRef
    terms: FileRef | None
    review_evidence: FileRef | None
    lineage_group: str
    duplicate_group: str
    parent_item_ids: tuple[str, ...]
    root: Path

    @property
    def payload_path(self) -> Path:
        return self.root / self.payload.path

    @property
    def payload_sha256(self) -> str:
        return self.payload.sha256


@dataclass(frozen=True)
class SourceAdmission:
    pack_id: str
    candidate_inventory_sha256: str
    manifest_sha256: str
    candidate_path: Path
    manifest_path: Path
    root: Path
    train_items: tuple[AdmittedPublicItem, ...]
    development_items: tuple[AdmittedPublicItem, ...]
    final_metadata: tuple[AdmittedPublicItem, ...]
    exclusion_source_ids: frozenset[str]
    exclusion_digests: frozenset[str]
    exclusion_lineage_groups: frozenset[str]

    @property
    def items(self) -> tuple[AdmittedPublicItem, ...]:
        return (*self.train_items, *self.development_items, *self.final_metadata)

    def audit_handoff(self, *, gradient_paths: tuple[str, ...],
                      development_paths: tuple[str, ...]) -> dict[str, object]:
        """Rehash all admitted inputs and receipts at the optimizer boundary."""
        for name, file, digest in (("candidate inventory", self.candidate_path,
                                   self.candidate_inventory_sha256),
                                  ("admission manifest", self.manifest_path,
                                   self.manifest_sha256)):
            try:
                actual = _sha256_file(file)
            except OSError as exc:
                raise ValueError(f"{name} cannot be read at handoff") from exc
            if actual != digest:
                _fail(f"{name} changed before optimizer handoff")
        cache: dict[str, str] = {}
        for item in (*self.train_items, *self.development_items):
            for name, ref in (("payload", item.payload), ("upstream file", item.upstream_file),
                              ("rights", item.rights), ("source terms", item.terms),
                              ("review evidence", item.review_evidence)):
                if ref is not None:
                    _checked_file(self.root, vars(ref), name, cache)
        final_paths = {item.payload.path for item in self.final_metadata}
        final_digests = {item.payload.sha256 for item in self.final_metadata}
        def check(paths: tuple[str, ...], items: tuple[AdmittedPublicItem, ...],
                  label: str) -> list[dict[str, str]]:
            if len(paths) != len(set(paths)):
                _fail(f"duplicate {label} path")
            permitted = {item.payload.path: item for item in items}
            output = []
            for given in paths:
                path, _ = _path(self.root, given, label)
                item = permitted.get(path)
                if (item is None or path in final_paths or
                        item.payload.sha256 in final_digests or
                        item.payload.sha256 in self.exclusion_digests or
                        item.source_id in self.exclusion_source_ids or
                        item.lineage_group in self.exclusion_lineage_groups):
                    _fail(f"{label} contains unadmitted, excluded or FINAL payload")
                _checked_file(self.root, vars(item.payload), label, cache)
                output.append({"item_id": item.item_id, "path": path,
                               "sha256": item.payload.sha256})
            return output
        return {"schema": INPUTS_SCHEMA,
                "status": "validated_recorded_assertions_not_rights_authentication",
                "foundation_trained": False,
                "full_capability_qualified": False,
                "pack_id": self.pack_id,
                "candidate_inventory_sha256": self.candidate_inventory_sha256,
                "admission_manifest_sha256": self.manifest_sha256,
                "gradient_inputs": check(gradient_paths, self.train_items, "gradient"),
                "development_inputs": check(development_paths, self.development_items,
                                            "development"),
                "final_payloads_excluded": True,
                "final_exclusion_paths": sorted(final_paths),
                "excluded_sha256s": sorted(final_digests | self.exclusion_digests),
                "reserved_evaluations": sorted(RESERVED_EVALUATIONS)}


def admit_native_sources(candidate_path: str | Path, admission_path: str | Path, *,
                         expected_candidate_sha256: str,
                         expected_manifest_sha256: str) -> SourceAdmission:
    """Validate exact local Stage A items; FINAL is metadata only."""
    candidates, candidate_digest = _frozen_json(Path(candidate_path),
                                                 expected_candidate_sha256,
                                                 "candidate inventory")
    if (candidates.get("schema") != CANDIDATE_SCHEMA or
            candidates.get("status") != "candidate_only_no_training_admission" or
            candidates.get("model_lineage", {}).get("third_party_pretrained_weights_allowed") is not False):
        _fail("candidate inventory has unsupported status or weight lineage")
    candidate_rows = _array(candidates.get("candidates"), "candidates")
    candidate_by_id: dict[str, dict] = {}
    for value in candidate_rows:
        row = _obj(value, "candidate")
        candidate_id = _text(row.get("candidate_id"), "candidate_id")
        if candidate_id in candidate_by_id or row.get("status") != "candidate_not_admitted":
            _fail("candidate inventory must contain unique, unadmitted candidates")
        candidate_by_id[candidate_id] = row
    file = Path(admission_path)
    manifest, manifest_digest = _frozen_json(file, expected_manifest_sha256,
                                             "admission manifest")
    if manifest.get("schema") != ADMISSION_SCHEMA:
        _fail("expected separate native foundation admission manifest")
    if manifest.get("candidate_inventory_sha256") != candidate_digest:
        _fail("admission manifest does not bind frozen candidate inventory")
    pack_id = _text(manifest.get("pack_id"), "pack_id")
    exclusions = _obj(manifest.get("exclusions"), "exclusions")
    def frozen_set(field: str, validate) -> frozenset[str]:
        values = _array(exclusions.get(field), f"exclusions.{field}")
        normalized = [validate(value, field) for value in values]
        if len(set(normalized)) != len(normalized):
            _fail(f"duplicate exclusion {field}")
        return frozenset(normalized)
    excluded_ids = frozen_set("source_ids", _text)
    excluded_hashes = frozen_set("sha256s", _digest)
    excluded_groups = frozen_set("lineage_groups", _text)
    reserved = frozen_set("reserved_evaluations", _text)
    if not RESERVED_EVALUATIONS.issubset(reserved):
        _fail("reserved benchmark exclusion declarations are missing")
    root = file.parent.resolve()
    cache: dict[str, str] = {}
    seen_ids: set[str] = set()
    item_splits: dict[str, str] = {}
    paths: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    groups: dict[tuple[str, str], str] = {}
    parent_graph: dict[str, tuple[str, ...]] = {}
    optimizer_paths: set[str] = set()
    review_paths: set[str] = set()
    grouped: dict[str, list[AdmittedPublicItem]] = {name: [] for name in SPLITS}
    rows = _array(manifest.get("items"), "items")
    if not rows or not any(_obj(row, "item").get("split") == "train" for row in rows):
        _fail("admission manifest has no training item")
    for value in rows:
        row = _obj(value, "item")
        item_id = _text(row.get("item_id"), "item_id")
        source_id = _text(row.get("source_id"), "source_id")
        if item_id in seen_ids:
            _fail("duplicate item ID")
        seen_ids.add(item_id)
        split = row.get("split")
        if split not in SPLITS:
            _fail("unsupported public source split")
        item_splits[item_id] = split
        candidate_id = _text(row.get("candidate_id"), "candidate_id")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            _fail("item source is absent from candidate inventory")
        if row.get("candidate_source_url") != candidate.get("source_url"):
            _fail("item candidate URL differs from frozen inventory")
        modality = _text(row.get("modality"), "modality")
        if modality not in candidate.get("modalities", ()):
            _fail("modality is absent from candidate inventory")
        original_url = _text(row.get("source_url"), "source_url")
        revision = _text(row.get("upstream_revision"), "upstream_revision")
        upstream_item_ref = _text(row.get("upstream_item_ref"), "upstream_item_ref")
        _digest(row.get("extraction_process_sha256"), "extraction_process_sha256")
        lineage = _text(row.get("lineage_group"), "lineage_group")
        duplicate = _text(row.get("duplicate_group"), "duplicate_group")
        parents = tuple(_text(parent, "parent_item_id") for parent in
                        _array(row.get("parent_item_ids"), "parent_item_ids"))
        if item_id in parents or len(parents) != len(set(parents)):
            _fail("invalid parent item references")
        parent_graph[item_id] = parents
        final = split == "final"
        upstream = _checked_file(root, row.get("upstream_file"), "upstream file", cache,
                                 open_bytes=not final)
        payload = _checked_file(root, row.get("payload"), "payload", cache,
                                open_bytes=not final)
        rights = _checked_file(root, row.get("rights"), "rights", cache,
                               open_bytes=not final)
        if not final:
            try:
                rights_data = _obj(json.loads((root / rights.path).read_bytes()), "rights")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("rights receipt cannot be parsed") from exc
            if rights_data.get("schema") != RIGHTS_SCHEMA:
                _fail("unsupported item rights schema")
            for field, expected in (("item_id", item_id), ("source_id", source_id),
                                    ("candidate_id", candidate_id),
                                    ("payload_sha256", payload.sha256),
                                    ("upstream_file_sha256", upstream.sha256),
                                    ("source_url", original_url),
                                    ("upstream_revision", revision),
                                    ("upstream_item_ref", upstream_item_ref)):
                if rights_data.get(field) != expected:
                    _fail(f"item rights receipt has wrong {field} binding")
            for field in ("license_id", "author", "attribution", "reviewer_id",
                          "authority_ref"):
                value = _text(rights_data.get(field), f"rights.{field}")
                if value.lower() in {"unknown", "pending", "unverified", "none", "n/a"}:
                    _fail(f"rights.{field} needs specific reviewed evidence")
            if rights_data.get("data_class") != "public_source":
                _fail("private owner data is not eligible for shared foundation")
            _date(rights_data.get("reviewed_at"), "rights.reviewed_at")
            permissions = _obj(rights_data.get("permissions"), "permissions")
            if (permissions.get("commercial_training") is not True or
                    permissions.get("model_distribution") is not True):
                _fail("per-item rights do not permit commercial training and model distribution")
            privacy = _obj(rights_data.get("privacy"), "privacy")
            if (privacy.get("decision") != "approved" or
                    privacy.get("private_owner_data_present") is not False):
                _fail("privacy review is not approved")
            _text(privacy.get("reviewer_id"), "privacy.reviewer_id")
            _date(privacy.get("reviewed_at"), "privacy.reviewed_at")
            removal = _obj(rights_data.get("removal"), "removal")
            if removal.get("state") != "active":
                _fail("item is removed or removal state is unknown")
            _date(removal.get("checked_at"), "removal.checked_at")
            terms = _checked_file(root, rights_data.get("source_terms"), "source terms", cache)
            evidence = _checked_file(root, rights_data.get("review_evidence"),
                                     "review evidence", cache)
            review_paths.update((rights.path, terms.path, evidence.path))
            optimizer_paths.add(payload.path)
        else:
            terms = None
            evidence = None
        if split != "final" and (source_id in excluded_ids or
                                 payload.sha256 in excluded_hashes or
                                 lineage in excluded_groups):
            _fail("item is in frozen evaluation or removal exclusions")
        for key in (("source_id", source_id), ("lineage_group", lineage),
                    ("duplicate_group", duplicate), ("payload_sha256", payload.sha256),
                    ("upstream_source", f"{original_url}#{revision}:{upstream_item_ref}")):
            previous = groups.setdefault(key, split)
            if previous != split:
                _fail("related source lineage leaks across splits")
        old_split = source_splits.setdefault(source_id, split)
        if old_split != split:
            _fail("source ID leaks across splits")
        for ref in (payload, upstream, rights):
            prior = paths.setdefault(ref.path, ref.sha256)
            if prior != ref.sha256:
                _fail("same path has conflicting source bytes")
        grouped[split].append(AdmittedPublicItem(item_id, source_id, candidate_id,
                                                   modality, split, payload, upstream,
                                                   rights, terms, evidence, lineage, duplicate,
                                                   parents, root))
    for child, parents in parent_graph.items():
        for parent in parents:
            if parent not in parent_graph:
                _fail("unregistered parent item")
            if item_splits[child] != item_splits[parent]:
                _fail("related source lineage leaks across splits")
    if optimizer_paths & review_paths:
        _fail("review or rights material cannot be an optimizer payload")
    active: set[str] = set()
    complete: set[str] = set()
    for node in parent_graph:
        if node in complete:
            continue
        stack = [(node, False)]
        while stack:
            current, closing = stack.pop()
            if closing:
                active.remove(current)
                complete.add(current)
            elif current not in complete:
                if current in active:
                    _fail("source lineage cycle")
                active.add(current)
                stack.append((current, True))
                stack.extend((parent, False) for parent in reversed(parent_graph[current]))
    return SourceAdmission(pack_id, candidate_digest, manifest_digest,
                           Path(candidate_path).resolve(), file.resolve(), root,
                           tuple(grouped["train"]), tuple(grouped["development"]),
                           tuple(grouped["final"]), excluded_ids, excluded_hashes,
                           excluded_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--admission-manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    admission = admit_native_sources(
        args.candidate_inventory, args.admission_manifest,
        expected_candidate_sha256=args.candidate_sha256,
        expected_manifest_sha256=args.manifest_sha256)
    receipt = admission.audit_handoff(
        gradient_paths=tuple(item.payload.path for item in admission.train_items),
        development_paths=tuple(item.payload.path for item in admission.development_items))
    output = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(json.dumps({"receipt_sha256": sha256(output).hexdigest(),
                      "gradient_items": len(admission.train_items),
                      "development_items": len(admission.development_items),
                      "status": receipt["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
