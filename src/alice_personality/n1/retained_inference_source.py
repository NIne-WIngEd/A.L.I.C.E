"""Inspect retained candidate proposals inside the caller's private boundary.

No private I/O occurs on import. The caller must enter original-owner USER+NET
and P2 isolation BEFORE invoking this API, including before resolving its path.
Returned field names and IDs are PROTECTED. Only verify_result's fixed summary
may leave that boundary. This establishes retained candidate existence, never
original delivery, historical validity, acceptance, or gradient permission.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import unicodedata
import zipfile

from . import compiler as c
from . import raw_inference_lineage as raw
from .manifest_structure_audit import _PinnedFile
from . import manifest_structure_audit as structural


MEMBER = "reserve/legacy_einf_720_raw_reserve.jsonl"
SCHEMA = "alice-personality-retained-inference-inspection-v1"
ENVELOPE_SCHEMA = "alice-personality-retained-inference-inspection-v2"
STATE = "RETAINED_SOURCE_INSPECTED_UNQUALIFIED"
FLAGS = ("acceptance_authority", "private_gradient_authorized", "historical_authority_granted",
         "training_authorized", "source_admission", "original_source_independently_verified")
CONTRACTS = frozenset({"exact_proposal_contract", "invalid_proposal_namespace",
                      "invalid_original_proposal_id", "duplicate_original_proposal_id"})
TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
CODE_NAMES = {"retained.py", "raw.py", "compiler.py", "descriptor.py", "__init__.py"}
MAX_PROFILE_BYTES = 4 * 1024 * 1024
ENVELOPE_ROLES = {"declared_source_id": "string", "pool_id": "string", "provenance": "string",
                  "historical_authority": "boolean", "training_authority": "boolean", "original_record": "object"}


class RetainedSourceError(ValueError):
    """Fixed refusal; no arbitrary source text, IDs, keys, paths or errors."""


def _fail():
    raise RetainedSourceError("retained_source_refused")


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    paths = {"retained.py": Path(__file__), "raw.py": root / "raw_inference_lineage.py",
             "compiler.py": root / "compiler.py", "descriptor.py": root / "manifest_structure_audit.py",
             "__init__.py": root / "__init__.py"}
    return {name: sha256(path.read_bytes()).hexdigest() for name, path in sorted(paths.items())}


def _profile(payload: bytes) -> tuple[int, dict]:
    lines = payload.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines or len(lines) > raw.MAX_PROPOSAL_ROWS:
        _fail()
    counts = {}
    for line in lines:
        if not line or len(line) > raw.MAX_JSONL_LINE_BYTES:
            _fail()
        row = raw._json(line)
        if type(row) is not dict:
            _fail()
        for name, value in row.items():
            kind = {dict: "object", list: "array", str: "string", int: "integer", float: "number",
                    bool: "boolean", type(None): "null"}.get(type(value))
            if kind is None:
                _fail()
            table = counts.setdefault(name, {})
            table[kind] = table.get(kind, 0) + 1
        if len(counts) > 50_000:
            _fail()
    if len(c._canonical(counts)) > MAX_PROFILE_BYTES:
        _fail()
    return len(lines), counts


def _identifier(value: object) -> bool:
    return type(value) is str and raw._ID.fullmatch(value) is not None and \
        all(unicodedata.category(char) not in ("Cc", "Cs") for char in value)


def _envelope_layout(layout: dict, previous: dict, binding: dict, rows: int, profile: dict) -> str:
    keys = set(FLAGS) | {"schema", "state", "source_archive_sha256", "retained_member_sha256",
                         "profile_file_sha256", "profile_canonical_sha256", "pointers", "review_id", "selection_basis"}
    if type(layout) is not dict or set(layout) != keys or \
            layout["schema"] != "alice-personality-retained-envelope-layout-review-v1" or layout["state"] != "PROPOSED_UNQUALIFIED" or \
            any(layout[key] is not False for key in FLAGS) or \
            layout["source_archive_sha256"] != binding["archive_sha256"] or \
            layout["retained_member_sha256"] != binding["members_sha256"][binding["package_root"] + "/" + MEMBER] or \
            any(type(layout[key]) is not str or not 0 < len(layout[key]) <= 1024 for key in ("review_id", "selection_basis")) or \
            type(previous) is not dict or previous.get("schema") != SCHEMA or previous.get("state") != STATE or \
            previous.get("source_archive_sha256") != layout["source_archive_sha256"] or \
            previous.get("retained_member_sha256") != layout["retained_member_sha256"] or \
            previous.get("retained_member_path") != binding["package_root"] + "/" + MEMBER or \
            previous.get("row_count") != rows or previous.get("top_level_field_type_counts") != profile or \
            layout["profile_canonical_sha256"] != sha256(c._canonical(previous)).hexdigest() or \
            layout["profile_file_sha256"] != sha256(c._canonical(previous) + b"\n").hexdigest() or \
            type(layout["pointers"]) is not dict or set(layout["pointers"]) != set(ENVELOPE_ROLES):
        _fail()
    pointers = layout["pointers"]
    for role, expected_type in ENVELOPE_ROLES.items():
        path = pointers[role]
        if type(path) is not list or len(path) != 1 or type(path[0]) is not str or \
                profile.get(path[0]) != {expected_type: rows}:
            _fail()
    if len({tuple(path) for path in pointers.values()}) != len(ENVELOPE_ROLES):
        _fail()
    return sha256(c._canonical(layout)).hexdigest()


def _envelope_ids_and_structure(payload: bytes, layout: dict) -> tuple[list[str], list[dict]]:
    pointers = {role: path[0] for role, path in layout["pointers"].items()}
    identifiers, pool_ids, variants = set(), set(), {}
    lines = payload.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    retained_bytes = 2  # JSON list brackets; enforce before keeping each new tree.
    for line in lines:
        row = raw._json(line)
        if row[pointers["provenance"]] != "E-INF" or row[pointers["historical_authority"]] is not False or \
                row[pointers["training_authority"]] is not False or type(row[pointers["original_record"]]) is not dict:
            _fail()
        identifier, pool_id = row[pointers["declared_source_id"]], row[pointers["pool_id"]]
        if not _identifier(identifier) or not _identifier(pool_id) or identifier in identifiers or pool_id in pool_ids:
            _fail()
        identifiers.add(identifier)
        pool_ids.add(pool_id)
        tree, _ = structural.manifest_structure(row)
        key = sha256(c._canonical(tree)).hexdigest()
        if key not in variants:
            variant = {"count": 1, "structure": tree}
            additional = len(c._canonical(variant)) + (1 if variants else 0)
            if retained_bytes + additional > MAX_PROFILE_BYTES:
                _fail()
            variants[key] = variant
            retained_bytes += additional
        else:
            variant = variants[key]
            count = variant["count"] + 1
            additional = len(str(count)) - len(str(variant["count"]))
            if retained_bytes + additional > MAX_PROFILE_BYTES:
                _fail()
            variant["count"] = count
            retained_bytes += additional
    result = [variants[key] for key in sorted(variants)]
    if len(c._canonical(result)) > MAX_PROFILE_BYTES:
        _fail()
    return sorted(identifiers), result


def _verify_structure(tree: dict, depth: int = 0) -> int:
    if type(tree) is not dict or depth > structural.MAX_STRUCTURE_DEPTH or tree.get("type") not in TYPES:
        _fail()
    kind = tree["type"]
    if kind == "object":
        if set(tree) != {"type", "fields"} or type(tree["fields"]) is not dict or \
                any(type(key) is not str for key in tree["fields"]):
            _fail()
        nodes = 1 + sum(_verify_structure(child, depth + 1) for child in tree["fields"].values())
    elif kind == "array":
        if set(tree) != {"type", "length", "items"} or type(tree["items"]) is not list or \
                type(tree["length"]) is not int or tree["length"] != len(tree["items"]):
            _fail()
        nodes = 1 + sum(_verify_structure(child, depth + 1) for child in tree["items"])
    else:
        if set(tree) != {"type"}:
            _fail()
        nodes = 1
    if nodes > structural.MAX_STRUCTURE_NODES:
        _fail()
    return nodes


def _verify_coverage(value: dict, registry: dict, archive_sha: str) -> None:
    digest_fields = {"source_archive_sha256", "package_pin_sha256", "compiler_implementation_sha256",
                     "raw_lineage_registry_sha256", "receipt_sha256"}
    flags = set(FLAGS[:4])
    keys = digest_fields | flags | {"schema", "state", "registry_count", "source_kind_counts",
                                   "global_counts", "einf_source_origin_counts", "behavior_qualification"}
    metrics = {"total_rows", "reference_occurrences", "unique_references", "resolved_reference_occurrences",
               "unresolved_reference_occurrences", "resolved_unique_references", "unresolved_unique_references",
               "rows_with_missing"}
    globals_ = metrics | {"reference_union_count", "reference_registry_intersection_count",
                          "reference_registry_difference_count", "registry_reference_difference_count"}
    if type(value) is not dict or set(value) != keys or \
            value["schema"] != "alice-personality-raw-lineage-coverage-diagnostic-v1" or \
            value["state"] != "UNQUALIFIED" or value["behavior_qualification"] is not None or \
            any(value[key] is not False for key in flags) or \
            any(type(value[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None for key in digest_fields) or \
            value["source_archive_sha256"] != archive_sha or \
            value["raw_lineage_registry_sha256"] != sha256(c._canonical(registry)).hexdigest() or \
            type(value["registry_count"]) is not int or value["registry_count"] != len(registry["EINF"]) or \
            type(value["source_kind_counts"]) is not dict or set(value["source_kind_counts"]) != set(c.ACTIVE_FILES):
        _fail()
    for table, expected in [(t, metrics) for t in value["source_kind_counts"].values()] + [
            (value["global_counts"], globals_), (value["einf_source_origin_counts"], metrics)]:
        if type(table) is not dict or set(table) != expected or \
                any(type(count) is not int or count < 0 for count in table.values()):
            _fail()
        if table["reference_occurrences"] != table["resolved_reference_occurrences"] + table["unresolved_reference_occurrences"] or \
                table["unique_references"] != table["resolved_unique_references"] + table["unresolved_unique_references"] or \
                table["unique_references"] > table["reference_occurrences"] or \
                table["rows_with_missing"] > min(table["total_rows"], table["unresolved_reference_occurrences"]) or \
                table["resolved_unique_references"] > min(table["resolved_reference_occurrences"], value["registry_count"]) or \
                table["unresolved_unique_references"] > table["unresolved_reference_occurrences"]:
            _fail()
    global_counts = value["global_counts"]
    if any(global_counts[key] != sum(table[key] for table in value["source_kind_counts"].values())
            for key in ("total_rows", "reference_occurrences", "resolved_reference_occurrences",
                        "unresolved_reference_occurrences", "rows_with_missing")) or \
            global_counts["reference_union_count"] != global_counts["unique_references"] or \
            global_counts["reference_registry_intersection_count"] != global_counts["resolved_unique_references"] or \
            global_counts["reference_registry_difference_count"] != global_counts["unresolved_unique_references"] or \
            global_counts["registry_reference_difference_count"] + global_counts["resolved_unique_references"] != value["registry_count"] or \
            value["einf_source_origin_counts"]["total_rows"] != value["source_kind_counts"]["EINF"]["total_rows"]:
        _fail()
    if value["receipt_sha256"] != sha256(c._canonical(
            {key: item for key, item in value.items() if key != "receipt_sha256"})).hexdigest():
        _fail()


def verify_result(protected: dict, summary: dict) -> None:
    """Allowlist every public field, including nested coverage; bind protected data."""
    try:
        envelope = type(summary) is dict and summary.get("schema") == ENVELOPE_SCHEMA
        keys = set(FLAGS) | {"schema", "state", "source_archive_sha256", "package_pin_sha256",
            "retained_member_sha256", "namespace_provenance", "row_count", "field_count",
            "namespace_contract", "observed_proposal_count", "candidate_registry_sha256",
            "retained_candidate_coverage", "implementation_sha256", "protected_sha256", "receipt_sha256"}
        pkeys = {"schema", "state", "source_archive_sha256", "retained_member_path", "retained_member_sha256",
                 "row_count", "top_level_field_type_counts", "namespace_contract", "candidate_registry"}
        if envelope:
            keys |= {"reviewed_layout_sha256", "previous_profile_file_sha256", "observed_envelope_identifier_count",
                     "nested_schema_variant_count"}
            pkeys |= {"reviewed_layout_sha256", "nested_schema_variants"}
        expected_schema = ENVELOPE_SCHEMA if envelope else SCHEMA
        expected_provenance = "member_pinned_retained_envelope_declaration" if envelope else "member_pinned_retained_candidate"
        contracts = {"reviewed_retained_envelope_contract"} if envelope else CONTRACTS
        if type(summary) is not dict or set(summary) != keys or type(protected) is not dict or set(protected) != pkeys or \
                summary["schema"] != expected_schema or protected["schema"] != expected_schema or \
                summary["state"] != STATE or protected["state"] != STATE or \
                summary["namespace_provenance"] != expected_provenance or \
                summary["namespace_contract"] not in contracts or \
                any(summary[key] is not False for key in FLAGS) or \
                any(type(summary[key]) is not int or summary[key] < 0 for key in
                    ("row_count", "field_count", "observed_proposal_count")) or \
                not 0 < summary["row_count"] <= raw.MAX_PROPOSAL_ROWS or \
                type(summary["implementation_sha256"]) is not dict or set(summary["implementation_sha256"]) != CODE_NAMES:
            _fail()
        digests = [summary[key] for key in ("source_archive_sha256", "package_pin_sha256", "retained_member_sha256",
                   "protected_sha256", "receipt_sha256")] + list(summary["implementation_sha256"].values())
        if envelope:
            digests += [summary["reviewed_layout_sha256"], summary["previous_profile_file_sha256"]]
        if any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests) or \
                any(protected[key] != summary[key] for key in
                    ("state", "source_archive_sha256", "retained_member_sha256", "row_count", "namespace_contract")) or \
                type(protected["retained_member_path"]) is not str or \
                not protected["retained_member_path"].endswith("/" + MEMBER):
            _fail()
        profile = protected["top_level_field_type_counts"]
        if type(profile) is not dict or len(profile) != summary["field_count"]:
            _fail()
        for name, table in profile.items():
            if type(name) is not str or type(table) is not dict or not table or not set(table) <= TYPES or \
                    any(type(count) is not int or count <= 0 for count in table.values()) or \
                    sum(table.values()) > summary["row_count"]:
                _fail()
        registry = protected["candidate_registry"]
        if envelope:
            variants = protected["nested_schema_variants"]
            if type(variants) is not list or type(summary["nested_schema_variant_count"]) is not int or \
                    len(variants) != summary["nested_schema_variant_count"] or not variants or \
                    protected["reviewed_layout_sha256"] != summary["reviewed_layout_sha256"] or \
                    type(summary["observed_envelope_identifier_count"]) is not int or \
                    summary["observed_envelope_identifier_count"] != summary["row_count"] or summary["observed_proposal_count"] != 0:
                _fail()
            for variant in variants:
                if type(variant) is not dict or set(variant) != {"count", "structure"} or \
                        type(variant["count"]) is not int or variant["count"] <= 0:
                    _fail()
                _verify_structure(variant["structure"])
            if sum(variant["count"] for variant in variants) != summary["row_count"] or len(c._canonical(variants)) > MAX_PROFILE_BYTES:
                _fail()
        if envelope or summary["namespace_contract"] == "exact_proposal_contract":
            if type(registry) is not dict or set(registry) != {"EINF"} or type(registry["EINF"]) is not list or \
                    any(not _identifier(value) for value in registry["EINF"]) or \
                    registry["EINF"] != sorted(set(registry["EINF"])) or \
                    len(registry["EINF"]) != summary["row_count"] or \
                    (not envelope and summary["observed_proposal_count"] != summary["row_count"]) or \
                    summary["candidate_registry_sha256"] != sha256(c._canonical(registry)).hexdigest():
                _fail()
            _verify_coverage(summary["retained_candidate_coverage"], registry, summary["source_archive_sha256"])
            if summary["retained_candidate_coverage"]["package_pin_sha256"] != summary["package_pin_sha256"]:
                _fail()
        elif registry is not None or summary["candidate_registry_sha256"] is not None or \
                summary["retained_candidate_coverage"] is not None or summary["observed_proposal_count"] != 0:
            _fail()
        if len(c._canonical(protected)) > raw.MAX_PROPOSAL_BYTES or \
                summary["protected_sha256"] != sha256(c._canonical(protected)).hexdigest() or \
                summary["receipt_sha256"] != sha256(c._canonical(
                    {key: value for key, value in summary.items() if key != "receipt_sha256"})).hexdigest():
            _fail()
    except Exception:
        raise RetainedSourceError("retained_source_refused") from None


def inspect_retained_source(package_path: str | Path, *, pin: c.PackagePin,
                            reviewed_layout: dict | None = None, reviewed_profile: dict | None = None) -> tuple[dict, dict]:
    """Private construction API; no files written and no registry installed."""
    try:
        binding = c._pin(pin)
        code = implementation_hashes()
        member = binding["package_root"] + "/" + MEMBER
        ledger_name = binding["package_root"] + "/SHA256SUMS.txt"
        with _PinnedFile(package_path, binding["archive_sha256"], raw.MAX_ARCHIVE_BYTES) as source:
            source.stream.seek(0)
            with zipfile.ZipFile(source.stream) as archive:
                files = raw._inventory(archive)
                if set(files) != set(binding["members_sha256"]) or member not in files:
                    _fail()
                ledger_bytes = raw._read_member(archive, files[ledger_name], raw.MAX_METADATA_BYTES)
                ledger = raw._ledger(ledger_bytes, files, binding["package_root"], ledger_name)
                if sha256(ledger_bytes).hexdigest() != binding["members_sha256"][ledger_name] or \
                        ledger.get(member) != binding["members_sha256"][member] or \
                        any(digest != binding["members_sha256"][name] for name, digest in ledger.items()):
                    _fail()
                payload = raw._read_member(archive, files[member], raw.MAX_PROPOSAL_BYTES)
                if sha256(payload).hexdigest() != binding["members_sha256"][member]:
                    _fail()
                rows, profile = _profile(payload)
                envelope = reviewed_layout is not None or reviewed_profile is not None
                layout_sha, variants = None, None
                if envelope:
                    layout_sha = _envelope_layout(reviewed_layout, reviewed_profile, binding, rows, profile)
                    observed, variants = _envelope_ids_and_structure(payload, reviewed_layout)
                    contract = "reviewed_retained_envelope_contract"
                else:
                    contract = "exact_proposal_contract"
                    try:
                        observed = raw._proposal_ids(payload)
                    except raw.RawInferenceLineageError as exc:
                        if len(exc.args) != 1 or type(exc.args[0]) is not str or exc.args[0] not in CONTRACTS - {contract}:
                            _fail()
                        contract, observed = exc.args[0], None
            registry = {"EINF": observed} if observed is not None else None
            # This is diagnostic coverage against retained candidate existence.
            # It never installs the registry or calls the compiler/trainer.
            coverage = None if registry is None else c.audit_raw_lineage_coverage(
                source.path, pin=pin, raw_lineage_registry=registry)
            source.verify()
            if c._pin(pin) != binding or implementation_hashes() != code:
                _fail()
            selected_schema = ENVELOPE_SCHEMA if envelope else SCHEMA
            protected = {"schema": selected_schema, "state": STATE, "source_archive_sha256": binding["archive_sha256"],
                "retained_member_path": member, "retained_member_sha256": binding["members_sha256"][member],
                "row_count": rows, "top_level_field_type_counts": profile, "namespace_contract": contract,
                "candidate_registry": registry}
            if envelope:
                protected.update(reviewed_layout_sha256=layout_sha, nested_schema_variants=variants)
            summary = {"schema": selected_schema, "state": STATE, "source_archive_sha256": binding["archive_sha256"],
                "package_pin_sha256": sha256(c._canonical(binding)).hexdigest(),
                "retained_member_sha256": binding["members_sha256"][member],
                "namespace_provenance": "member_pinned_retained_envelope_declaration" if envelope else "member_pinned_retained_candidate",
                "row_count": rows, "field_count": len(profile),
                "namespace_contract": contract, "observed_proposal_count": len(observed) if observed is not None and not envelope else 0,
                "candidate_registry_sha256": sha256(c._canonical(registry)).hexdigest() if registry is not None else None,
                "retained_candidate_coverage": coverage, "implementation_sha256": code,
                "protected_sha256": sha256(c._canonical(protected)).hexdigest(), **{key: False for key in FLAGS}}
            if envelope:
                if sha256(c._canonical(reviewed_layout)).hexdigest() != layout_sha or \
                        sha256(c._canonical(reviewed_profile) + b"\n").hexdigest() != reviewed_layout["profile_file_sha256"]:
                    _fail()
                summary.update(reviewed_layout_sha256=layout_sha,
                    previous_profile_file_sha256=reviewed_layout["profile_file_sha256"],
                    observed_envelope_identifier_count=len(observed), nested_schema_variant_count=len(variants))
            summary["receipt_sha256"] = sha256(c._canonical(summary)).hexdigest()
            verify_result(protected, summary)
        return protected, summary
    except Exception:
        raise RetainedSourceError("retained_source_refused") from None
