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


MEMBER = "reserve/legacy_einf_720_raw_reserve.jsonl"
SCHEMA = "alice-personality-retained-inference-inspection-v1"
STATE = "RETAINED_SOURCE_INSPECTED_UNQUALIFIED"
FLAGS = ("acceptance_authority", "private_gradient_authorized", "historical_authority_granted",
         "training_authorized", "source_admission", "original_source_independently_verified")
CONTRACTS = frozenset({"exact_proposal_contract", "invalid_proposal_namespace",
                      "invalid_original_proposal_id", "duplicate_original_proposal_id"})
TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
CODE_NAMES = {"retained.py", "raw.py", "compiler.py", "descriptor.py", "__init__.py"}
MAX_PROFILE_BYTES = 4 * 1024 * 1024


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
        keys = set(FLAGS) | {"schema", "state", "source_archive_sha256", "package_pin_sha256",
            "retained_member_sha256", "namespace_provenance", "row_count", "field_count",
            "namespace_contract", "observed_proposal_count", "candidate_registry_sha256",
            "retained_candidate_coverage", "implementation_sha256", "protected_sha256", "receipt_sha256"}
        pkeys = {"schema", "state", "source_archive_sha256", "retained_member_path", "retained_member_sha256",
                 "row_count", "top_level_field_type_counts", "namespace_contract", "candidate_registry"}
        if type(summary) is not dict or set(summary) != keys or type(protected) is not dict or set(protected) != pkeys or \
                summary["schema"] != SCHEMA or protected["schema"] != SCHEMA or \
                summary["state"] != STATE or protected["state"] != STATE or \
                summary["namespace_provenance"] != "member_pinned_retained_candidate" or \
                summary["namespace_contract"] not in CONTRACTS or \
                any(summary[key] is not False for key in FLAGS) or \
                any(type(summary[key]) is not int or summary[key] < 0 for key in
                    ("row_count", "field_count", "observed_proposal_count")) or \
                not 0 < summary["row_count"] <= raw.MAX_PROPOSAL_ROWS or \
                type(summary["implementation_sha256"]) is not dict or set(summary["implementation_sha256"]) != CODE_NAMES:
            _fail()
        digests = [summary[key] for key in ("source_archive_sha256", "package_pin_sha256", "retained_member_sha256",
                   "protected_sha256", "receipt_sha256")] + list(summary["implementation_sha256"].values())
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
        if summary["namespace_contract"] == "exact_proposal_contract":
            if type(registry) is not dict or set(registry) != {"EINF"} or type(registry["EINF"]) is not list or \
                    any(type(value) is not str or not raw._ID.fullmatch(value) or
                        any(unicodedata.category(char) in ("Cc", "Cs") for char in value)
                        for value in registry["EINF"]) or \
                    registry["EINF"] != sorted(set(registry["EINF"])) or \
                    len(registry["EINF"]) != summary["row_count"] or summary["observed_proposal_count"] != summary["row_count"] or \
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


def inspect_retained_source(package_path: str | Path, *, pin: c.PackagePin) -> tuple[dict, dict]:
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
            protected = {"schema": SCHEMA, "state": STATE, "source_archive_sha256": binding["archive_sha256"],
                "retained_member_path": member, "retained_member_sha256": binding["members_sha256"][member],
                "row_count": rows, "top_level_field_type_counts": profile, "namespace_contract": contract,
                "candidate_registry": registry}
            summary = {"schema": SCHEMA, "state": STATE, "source_archive_sha256": binding["archive_sha256"],
                "package_pin_sha256": sha256(c._canonical(binding)).hexdigest(),
                "retained_member_sha256": binding["members_sha256"][member],
                "namespace_provenance": "member_pinned_retained_candidate", "row_count": rows, "field_count": len(profile),
                "namespace_contract": contract, "observed_proposal_count": len(observed) if observed is not None else 0,
                "candidate_registry_sha256": sha256(c._canonical(registry)).hexdigest() if registry is not None else None,
                "retained_candidate_coverage": coverage, "implementation_sha256": code,
                "protected_sha256": sha256(c._canonical(protected)).hexdigest(), **{key: False for key in FLAGS}}
            summary["receipt_sha256"] = sha256(c._canonical(summary)).hexdigest()
            verify_result(protected, summary)
        return protected, summary
    except Exception:
        raise RetainedSourceError("retained_source_refused") from None
