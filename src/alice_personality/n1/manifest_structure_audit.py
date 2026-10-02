"""Protected manifest structure discovery, never a provenance comparison.

The caller MUST establish its reviewed private-source isolation boundary before
calling this module, including before any path resolution/stat/hash. Importing
the module opens no private source. Only two selected JSON manifest streams and
their checksum ledgers are opened; whole-archive hashing is opaque byte custody.
No reserve/proposal/identity row stream, extraction, reference registry, model or
gradient is used. Arbitrary JSON keys can themselves be IDs or filenames: the
returned structure is PROTECTED metadata, never a safe public schema dump.

The original discovery never interprets values. A separate targeted comparison
uses only the exact source_hashes path established by the pinned phase-one
review; it does not scan strings, attribute labels or prove original input
authenticity. Bounds are parser/resource limits.
"""
from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile
import zlib
import lzma

from . import compiler
from . import raw_inference_lineage as raw


STRUCTURE_SCHEMA = "alice-personality-provenance-manifest-structure-v1"
SUMMARY_SCHEMA = "alice-personality-provenance-manifest-structure-summary-v1"
STATE = "STRUCTURE_DISCOVERED_UNQUALIFIED"
MAX_STRUCTURE_DEPTH = 32
MAX_STRUCTURE_NODES = 50_000
MAX_STRUCTURE_KEY_BYTES = 1024 * 1024
MAX_STRUCTURE_BYTES = 4 * 1024 * 1024
COMPARISON_SCHEMA = "alice-personality-declared-source-hash-comparison-v1"
COMPARISON_STATE = "DECLARED_SOURCE_HASHES_COMPARED_UNQUALIFIED"
LEGACY_RESERVE_SHA256 = "83901fc75ff25c1dda6188ad3b1fab65faf8f833df74b8c5c9e85fc6ea755327"
LEGACY_RESERVE_PUBLIC_MEMBER = "reserve/legacy_einf_720_raw_reserve.jsonl"
REVIEWED_STRUCTURE_FILE_SHA256 = "4e55e0c8ca124565d45cdf7f05b5ae5ad01754c7505e5ae84e9aed32f07e1788"
REVIEWED_CURATED_MANIFEST_SHA256 = "f5fc12c09d3304b6327570bda43cced0919bf1f67a16c0725e0ffafd75adc22f"
REVIEWED_RAW_MANIFEST_SHA256 = "a720bdc756ae0c798461c0733ad9d3093a65a8bd6c6f265491e2667dfd56b810"
DECLARED_SOURCE_HASH_COUNT = 30
_FLAGS = ("acceptance_authority", "private_gradient_authorized",
          "historical_authority_granted", "training_authorized")
_FAILURES = frozenset({"audit_refused", "audit_input_invalid", "audit_custody_refused", "audit_inventory_refused",
                      "audit_metadata_refused", "audit_structure_bound", "audit_code_changed"})


class ManifestStructureAuditError(ValueError):
    """Fixed sanitized failure, with no source values or arbitrary key paths."""


def _fail(reason: str) -> None:
    raise ManifestStructureAuditError(reason)


def safe_reason(exc: BaseException) -> str:
    # Even a forged exception carrying a private value cannot echo it publicly.
    if isinstance(exc, ManifestStructureAuditError) and str(exc) in _FAILURES:
        return str(exc)
    return "audit_refused"


class _PinnedFile:
    """An open regular descriptor, exact byte pin and stable path rechecks."""

    def __init__(self, path: str | Path, expected: str, max_bytes: int):
        self.value, self.expected, self.max_bytes = path, raw._digest(expected), max_bytes
        self.stream = None

    def __enter__(self):
        try:
            self.path = raw._archive_path(self.value)
            self.before = self.path.stat()
            if self.before.st_size > self.max_bytes:
                _fail("audit_custody_refused")
            descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                                 | getattr(os, "O_NOFOLLOW", 0))
            self.stream = os.fdopen(descriptor, "rb")
            self.descriptor_before = os.fstat(self.stream.fileno())
            if not stat.S_ISREG(self.descriptor_before.st_mode) or \
                    raw._file_identity(self.descriptor_before) != raw._file_identity(self.before):
                _fail("audit_custody_refused")
            self.verify()
            return self
        except BaseException:
            if self.stream is not None:
                self.stream.close()
            raise

    def verify(self) -> None:
        self.stream.seek(0)
        digest, count = sha256(), 0
        for block in iter(lambda: self.stream.read(1024 * 1024), b""):
            count += len(block)
            if count > self.max_bytes:
                _fail("audit_custody_refused")
            digest.update(block)
        if digest.hexdigest() != self.expected or count != self.before.st_size \
                or raw._file_state(os.fstat(self.stream.fileno())) != raw._file_state(self.descriptor_before) \
                or raw._file_state(raw._archive_path(self.path).stat()) != raw._file_state(self.before):
            _fail("audit_custody_refused")

    def bounded_bytes(self) -> bytes:
        self.stream.seek(0)
        payload = self.stream.read(self.max_bytes + 1)
        if len(payload) > self.max_bytes or sha256(payload).hexdigest() != self.expected:
            _fail("audit_custody_refused")
        return payload

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.verify()
        finally:
            self.stream.close()


def implementation_hashes() -> dict[str, str]:
    """Public implementation dependencies; reading these grants no authority."""
    paths = {"manifest_structure_audit.py": Path(__file__),
             "raw_inference_lineage.py": Path(raw.__file__),
             "compiler.py": Path(compiler.__file__),
             "__init__.py": Path(__file__).with_name("__init__.py")}
    return {name: sha256(raw._archive_path(path).read_bytes()).hexdigest()
            for name, path in sorted(paths.items())}


def manifest_structure(value: object) -> tuple[dict, int]:
    """Return a bounded key/type-only tree. ALL of this result stays protected.

    Scalars retain only their JSON type. Arrays retain per-item structures and
    length, never string/numeric/Boolean values. Dict keys remain exact because
    the next review must establish a legitimate field path, not guess aliases.
    No key is interpreted as a provenance role or digest-bearing field.
    """
    nodes = key_bytes = 0

    def visit(item: object, depth: int) -> dict:
        nonlocal nodes, key_bytes
        nodes += 1
        if nodes > MAX_STRUCTURE_NODES or depth > MAX_STRUCTURE_DEPTH:
            _fail("audit_structure_bound")
        if type(item) is dict:
            fields = {}
            for key in sorted(item):
                if type(key) is not str:
                    _fail("audit_metadata_refused")
                key_bytes += len(key.encode("utf-8"))
                if key_bytes > MAX_STRUCTURE_KEY_BYTES:
                    _fail("audit_structure_bound")
                fields[key] = visit(item[key], depth + 1)
            return {"type": "object", "fields": fields}
        if type(item) is list:
            return {"type": "array", "length": len(item),
                    "items": [visit(child, depth + 1) for child in item]}
        primitive = {str: "string", int: "integer", float: "number", bool: "boolean",
                     type(None): "null"}.get(type(item))
        if primitive is None:
            _fail("audit_metadata_refused")
        return {"type": primitive}

    try:
        if type(value) is not dict:
            _fail("audit_metadata_refused")
        structure = visit(value, 0)
        if len(raw._canonical(structure)) > MAX_STRUCTURE_BYTES:
            _fail("audit_structure_bound")
        return structure, nodes
    except ManifestStructureAuditError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise ManifestStructureAuditError("audit_metadata_refused") from None


def _selection(value: object, expected_archive: str) -> dict:
    paths = {"source_member_path", "generation_manifest_path", "checksum_ledger_path"}
    digests = {"source_archive_sha256", "source_layout_receipt_file_sha256", "source_layout_receipt_sha256"}
    keys = paths | digests | set(_FLAGS) | {"schema", "state", "selection_basis"}
    if type(value) is not dict or set(value) != keys or \
            value["schema"] != "alice-personality-raw-inference-source-layout-review-v1" or \
            value["state"] != "PROPOSED_UNQUALIFIED" or \
            value["source_archive_sha256"] != expected_archive or \
            any(value[key] is not False for key in _FLAGS) or \
            type(value["selection_basis"]) is not str or not 0 < len(value["selection_basis"]) <= 512:
        _fail("audit_input_invalid")
    for key in digests:
        if raw._digest(value[key]) != value[key]:
            _fail("audit_input_invalid")
    selected = [raw._member_path(value[key]) for key in sorted(paths)]
    if any(any(char in name for char in "*?[]") for name in selected) \
            or len(set(selected)) != 3 or len({str(PurePosixPath(name).parent) for name in selected}) != 1 \
            or PurePosixPath(value["generation_manifest_path"]).name != "generation_manifest.json" \
            or PurePosixPath(value["checksum_ledger_path"]).name != "SHA256SUMS.txt":
        _fail("audit_input_invalid")
    return dict(value)


def _read_structure(archive, files, name, expected_digest) -> tuple[dict, int, str, dict]:
    payload = raw._read_member(archive, files[name], raw.MAX_METADATA_BYTES)
    digest = sha256(payload).hexdigest()
    if digest != expected_digest:
        _fail("audit_metadata_refused")
    manifest = raw._json(payload)
    structure, count = manifest_structure(manifest)
    return structure, count, digest, manifest


def verify_discovery_result(protected: dict, summary: dict) -> None:
    """Check complete fixed public schema and protected canonical binding.

    A scalar value field is forbidden in the protected tree too. The only
    source-defined strings permitted there are exact dictionary keys/member
    paths, and they never appear in the public summary or error text.
    """
    digest_keys = {"curated_archive_sha256", "raw_archive_sha256", "curated_manifest_sha256",
                   "raw_generation_manifest_sha256", "curated_checksum_ledger_sha256",
                   "raw_checksum_ledger_sha256", "protected_structure_sha256"}
    count_keys = {"curated_structure_node_count", "raw_structure_node_count",
                  "opened_json_manifest_count", "opened_checksum_ledger_count",
                  "opened_identity_payload_member_count"}
    false_keys = set(_FLAGS) | {"link_compared", "digest_search_performed", "source_admission_performed",
                               "reference_registry_written", "model_training_performed"}
    summary_keys = digest_keys | count_keys | false_keys | {"schema", "state", "implementation_sha256"}
    if type(summary) is not dict or set(summary) != summary_keys or \
            summary["schema"] != SUMMARY_SCHEMA or summary["state"] != STATE or \
            any(summary[key] is not False for key in false_keys) or \
            any(type(summary[key]) is not int or summary[key] < 0 for key in count_keys) or \
            summary["opened_json_manifest_count"] != 2 or summary["opened_checksum_ledger_count"] != 2 or \
            summary["opened_identity_payload_member_count"] != 0:
        _fail("audit_metadata_refused")
    for key in digest_keys:
        if raw._digest(summary[key]) != summary[key]:
            _fail("audit_metadata_refused")
    code_keys = {"manifest_structure_audit.py", "raw_inference_lineage.py", "compiler.py", "__init__.py"}
    if type(summary["implementation_sha256"]) is not dict or set(summary["implementation_sha256"]) != code_keys:
        _fail("audit_metadata_refused")
    for digest in summary["implementation_sha256"].values():
        if raw._digest(digest) != digest:
            _fail("audit_metadata_refused")
    if type(protected) is not dict or set(protected) != set(_FLAGS) | {
            "schema", "state", "key_names_are_protected_metadata", "curated", "raw", "link_compared"} or \
            protected["schema"] != STRUCTURE_SCHEMA or protected["state"] != STATE or \
            protected["key_names_are_protected_metadata"] is not True or protected["link_compared"] is not False or \
            any(protected[key] is not False for key in _FLAGS):
        _fail("audit_metadata_refused")
    for label, digest_key, count_key, basename in (
            ("curated", "curated_manifest_sha256", "curated_structure_node_count", "curation_manifest.json"),
            ("raw", "raw_generation_manifest_sha256", "raw_structure_node_count", "generation_manifest.json")):
        item = protected[label]
        if type(item) is not dict or set(item) != {"manifest_member_path", "manifest_sha256", "structure"} or \
                PurePosixPath(raw._member_path(item["manifest_member_path"])).name != basename or \
                item["manifest_sha256"] != summary[digest_key]:
            _fail("audit_metadata_refused")
        nodes = key_bytes = 0

        def check(tree, depth):
            nonlocal nodes, key_bytes
            nodes += 1
            if nodes > MAX_STRUCTURE_NODES or depth > MAX_STRUCTURE_DEPTH:
                _fail("audit_structure_bound")
            if type(tree) is not dict or type(tree.get("type")) is not str:
                _fail("audit_metadata_refused")
            if tree["type"] == "object":
                if set(tree) != {"type", "fields"} or type(tree["fields"]) is not dict:
                    _fail("audit_metadata_refused")
                for key, child in tree["fields"].items():
                    if type(key) is not str:
                        _fail("audit_metadata_refused")
                    key_bytes += len(key.encode("utf-8"))
                    if key_bytes > MAX_STRUCTURE_KEY_BYTES:
                        _fail("audit_structure_bound")
                    check(child, depth + 1)
            elif tree["type"] == "array":
                if set(tree) != {"type", "length", "items"} or type(tree["items"]) is not list or \
                        type(tree["length"]) is not int or tree["length"] != len(tree["items"]):
                    _fail("audit_metadata_refused")
                for child in tree["items"]:
                    check(child, depth + 1)
            elif tree["type"] not in {"string", "integer", "number", "boolean", "null"} or set(tree) != {"type"}:
                _fail("audit_metadata_refused")

        check(item["structure"], 0)
        if item["structure"]["type"] != "object" or nodes != summary[count_key]:
            _fail("audit_metadata_refused")
    payload = raw._canonical(protected)
    if len(payload) > MAX_STRUCTURE_BYTES or sha256(payload).hexdigest() != summary["protected_structure_sha256"]:
        _fail("audit_metadata_refused")


def _inspect_manifests(curated_archive_path: str | Path, *,
                                curated_pin: compiler.PackagePin,
                                raw_archive_path: str | Path,
                                expected_raw_archive_sha256: str,
                                raw_member_map: dict, observer=None) -> tuple[dict, dict]:
    """Private shared custody seam. Observer runs before both postchecks."""
    try:
        pin = compiler._pin(curated_pin)
        expected_raw = raw._digest(expected_raw_archive_sha256)
        selection = _selection(raw_member_map, expected_raw)
        code = implementation_hashes()
        with ExitStack() as stack:
            curated_file = stack.enter_context(_PinnedFile(curated_archive_path, pin["archive_sha256"], raw.MAX_ARCHIVE_BYTES))
            raw_file = stack.enter_context(_PinnedFile(raw_archive_path, expected_raw, raw.MAX_ARCHIVE_BYTES))
            if curated_file.path == raw_file.path:
                _fail("audit_input_invalid")
            with zipfile.ZipFile(curated_file.stream) as archive:
                files = raw._inventory(archive)
                if set(files) != set(pin["members_sha256"]):
                    _fail("audit_inventory_refused")
                root = pin["package_root"]
                curation_name, ledger_name = root + "/curation_manifest.json", root + "/SHA256SUMS.txt"
                ledger_bytes = raw._read_member(archive, files[ledger_name], raw.MAX_METADATA_BYTES)
                if sha256(ledger_bytes).hexdigest() != pin["members_sha256"][ledger_name]:
                    _fail("audit_metadata_refused")
                ledger = raw._ledger(ledger_bytes, files, root, ledger_name)
                # A pinned package may contain nested checksum ledgers. The root
                # ledger need not flatten their inventories: do not open them or
                # assume undocumented complete root-ledger coverage. Whole ZIP
                # custody + exact inventory pins omitted siblings independently.
                if curation_name not in ledger or any(
                        digest != pin["members_sha256"][name] for name, digest in ledger.items()):
                    _fail("audit_inventory_refused")
                curated_structure, curated_nodes, curated_digest, _ = _read_structure(
                    archive, files, curation_name, pin["members_sha256"][curation_name])
                curated_ledger_digest = sha256(ledger_bytes).hexdigest()
            with zipfile.ZipFile(raw_file.stream) as archive:
                files = raw._inventory(archive)
                root, source_name, manifest_name, ledger_name = raw._selected(files, selection["source_member_path"])
                if manifest_name != selection["generation_manifest_path"] or ledger_name != selection["checksum_ledger_path"]:
                    _fail("audit_input_invalid")
                ledger_bytes = raw._read_member(archive, files[ledger_name], raw.MAX_METADATA_BYTES)
                ledger = raw._ledger(ledger_bytes, files, root, ledger_name)
                if manifest_name not in ledger or source_name not in ledger:
                    _fail("audit_metadata_refused")
                raw_structure, raw_nodes, raw_digest, raw_manifest = _read_structure(archive, files, manifest_name, ledger[manifest_name])
                raw_ledger_digest = sha256(ledger_bytes).hexdigest()
            protected = {"schema": STRUCTURE_SCHEMA, "state": STATE,
                         "key_names_are_protected_metadata": True,
                         "curated": {"manifest_member_path": curation_name,
                                     "manifest_sha256": curated_digest, "structure": curated_structure},
                         "raw": {"manifest_member_path": manifest_name,
                                 "manifest_sha256": raw_digest, "structure": raw_structure},
                         "link_compared": False, **{key: False for key in _FLAGS}}
            if observer is not None:
                observer(protected, raw_manifest)
            # Both original descriptors remain open through the complete audit.
            curated_file.verify()
            raw_file.verify()
            if implementation_hashes() != code:
                _fail("audit_code_changed")
        protected_bytes = raw._canonical(protected)
        if len(protected_bytes) > MAX_STRUCTURE_BYTES:
            _fail("audit_structure_bound")
        summary = {"schema": SUMMARY_SCHEMA, "state": STATE,
                   "curated_archive_sha256": pin["archive_sha256"], "raw_archive_sha256": expected_raw,
                   "curated_manifest_sha256": curated_digest, "raw_generation_manifest_sha256": raw_digest,
                   "curated_checksum_ledger_sha256": curated_ledger_digest, "raw_checksum_ledger_sha256": raw_ledger_digest,
                   "protected_structure_sha256": sha256(protected_bytes).hexdigest(),
                   "curated_structure_node_count": curated_nodes, "raw_structure_node_count": raw_nodes,
                   "opened_json_manifest_count": 2, "opened_checksum_ledger_count": 2,
                   "opened_identity_payload_member_count": 0, "link_compared": False,
                   "digest_search_performed": False, "source_admission_performed": False,
                   "reference_registry_written": False, "model_training_performed": False,
                   "implementation_sha256": code, **{key: False for key in _FLAGS}}
        verify_discovery_result(protected, summary)
        return protected, summary
    except ManifestStructureAuditError:
        raise
    except (raw.RawInferenceLineageError, compiler.IdentitySubstrateError, OSError, ValueError,
            UnicodeError, RuntimeError, KeyError, EOFError, NotImplementedError, RecursionError,
            zlib.error, lzma.LZMAError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise ManifestStructureAuditError("audit_refused") from None


def discover_manifest_structure(curated_archive_path: str | Path, *,
                                curated_pin: compiler.PackagePin,
                                raw_archive_path: str | Path,
                                expected_raw_archive_sha256: str,
                                raw_member_map: dict) -> tuple[dict, dict]:
    """Return (PROTECTED structure, fixed public summary), both unqualified.

    Caller-established isolation is mandatory, without a bypass parameter. A
    tiny explicit fixture pin can test custody mechanics but never accepts data.
    Public structural node counts do not count proposals or establish lineage.
    This original discovery API/schema always leaves link_compared=False. The
    map's layout receipt hashes are declared metadata, not receipt verification.
    """
    return _inspect_manifests(curated_archive_path, curated_pin=curated_pin,
        raw_archive_path=raw_archive_path, expected_raw_archive_sha256=expected_raw_archive_sha256,
        raw_member_map=raw_member_map)


def _declared_matches(manifest: dict) -> int:
    # Exactly one reviewed path, with no recursive scan, labels, key aliases,
    # filename/prefix normalization or source_package value interpretation.
    values = manifest.get("source_hashes")
    if type(values) is not dict or len(values) != DECLARED_SOURCE_HASH_COUNT or \
            type(manifest.get("source_package")) is not str:
        _fail("audit_metadata_refused")
    if any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in values.values()):
        _fail("audit_metadata_refused")
    # Validate EVERY declaration before any equality comparison.
    return sum(value == LEGACY_RESERVE_SHA256 for value in values.values())


def verify_comparison_result(value: dict) -> None:
    digest_keys = {"curated_archive_sha256", "raw_archive_sha256", "curated_manifest_sha256",
        "raw_generation_manifest_sha256", "curated_checksum_ledger_sha256", "raw_checksum_ledger_sha256",
        "reviewed_structure_file_sha256", "protected_structure_sha256", "legacy_reserve_member_sha256"}
    count_keys = {"declared_source_hash_count", "reserve_digest_match_count", "opened_json_manifest_count",
                  "opened_checksum_ledger_count", "opened_identity_payload_member_count"}
    false_keys = set(_FLAGS) | {"original_source_independently_verified", "legacy_namespace_proven",
        "source_admission", "digest_search_performed", "reference_registry_written", "model_training_performed"}
    true_keys = {"link_compared", "reviewed_structure_matched", "reviewed_manifest_pins_matched"}
    keys = digest_keys | count_keys | false_keys | true_keys | {
        "schema", "state", "comparison_basis", "generator_declares_matching_digest", "implementation_sha256"}
    if type(value) is not dict or set(value) != keys or value["schema"] != COMPARISON_SCHEMA or \
            value["state"] != COMPARISON_STATE or value["comparison_basis"] != "reviewed_generation_source_hash_map" or \
            any(value[key] is not False for key in false_keys) or any(value[key] is not True for key in true_keys) or \
            any(type(value[key]) is not int for key in count_keys) or \
            value["declared_source_hash_count"] != DECLARED_SOURCE_HASH_COUNT or \
            not 0 <= value["reserve_digest_match_count"] <= DECLARED_SOURCE_HASH_COUNT or \
            type(value["generator_declares_matching_digest"]) is not bool or \
            value["generator_declares_matching_digest"] != (value["reserve_digest_match_count"] > 0) or \
            value["opened_json_manifest_count"] != 2 or value["opened_checksum_ledger_count"] != 2 or \
            value["opened_identity_payload_member_count"] != 0 or value["legacy_reserve_member_sha256"] != LEGACY_RESERVE_SHA256:
        _fail("audit_metadata_refused")
    for key in digest_keys:
        if raw._digest(value[key]) != value[key]:
            _fail("audit_metadata_refused")
    code = value["implementation_sha256"]
    if type(code) is not dict or set(code) != {"manifest_structure_audit.py", "raw_inference_lineage.py", "compiler.py", "__init__.py"}:
        _fail("audit_metadata_refused")
    if any(raw._digest(digest) != digest for digest in code.values()):
        _fail("audit_metadata_refused")


def compare_declared_source_hashes(curated_archive_path: str | Path, *,
        curated_pin: compiler.PackagePin, raw_archive_path: str | Path,
        expected_raw_archive_sha256: str, raw_member_map: dict,
        reviewed_structure_path: str | Path,
        expected_reviewed_structure_file_sha256: str,
        expected_curated_manifest_sha256: str,
        expected_raw_generation_manifest_sha256: str) -> dict:
    """Compare one literal reviewed declaration map, never original authenticity.

    The caller MUST first establish isolation. Both opened manifest byte hashes
    and the entire scalar-free phase-one structure (including exact private key
    labels) must match independently supplied review pins. Matching labels and
    source_package contents are never returned, even as protected new evidence.
    A match/no-match is scoped only to this pinned generator-declared source map;
    it cannot prove an original source, legacy ID namespace, registry or approval.
    legacy_reserve_member_sha256 is the externally reviewed PUBLIC member pin,
    not a freshly read reserve-member hash. The production wrapper additionally
    checks this exact named public pin entry; tiny fixtures test mechanics only.
    """
    try:
        curated_digest = raw._digest(expected_curated_manifest_sha256)
        raw_digest = raw._digest(expected_raw_generation_manifest_sha256)
        with _PinnedFile(reviewed_structure_path, expected_reviewed_structure_file_sha256,
                         MAX_STRUCTURE_BYTES + 1) as reviewed_file:
            reviewed_bytes = reviewed_file.bounded_bytes()
            reviewed = raw._json(reviewed_bytes)
            matches = None

            def compare(protected, manifest):
                nonlocal matches
                if reviewed_bytes != raw._canonical(reviewed) + b"\n" or raw._canonical(protected) != raw._canonical(reviewed) or \
                        protected["curated"]["manifest_sha256"] != curated_digest or \
                        protected["raw"]["manifest_sha256"] != raw_digest:
                    _fail("audit_metadata_refused")
                matches = _declared_matches(manifest)

            _, discovered = _inspect_manifests(curated_archive_path, curated_pin=curated_pin,
                raw_archive_path=raw_archive_path, expected_raw_archive_sha256=expected_raw_archive_sha256,
                raw_member_map=raw_member_map, observer=compare)
            reviewed_file.verify()
            if implementation_hashes() != discovered["implementation_sha256"]:
                _fail("audit_code_changed")
        digest_keys = ("curated_archive_sha256", "raw_archive_sha256", "curated_manifest_sha256",
                      "raw_generation_manifest_sha256", "curated_checksum_ledger_sha256", "raw_checksum_ledger_sha256",
                      "protected_structure_sha256")
        result = {"schema": COMPARISON_SCHEMA, "state": COMPARISON_STATE,
            **{key: discovered[key] for key in digest_keys},
            "reviewed_structure_file_sha256": raw._digest(expected_reviewed_structure_file_sha256),
            "legacy_reserve_member_sha256": LEGACY_RESERVE_SHA256,
            "comparison_basis": "reviewed_generation_source_hash_map", "declared_source_hash_count": DECLARED_SOURCE_HASH_COUNT,
            "reserve_digest_match_count": matches, "generator_declares_matching_digest": matches > 0,
            "link_compared": True, "reviewed_structure_matched": True, "reviewed_manifest_pins_matched": True,
            "original_source_independently_verified": False, "legacy_namespace_proven": False, "source_admission": False,
            "digest_search_performed": False, "reference_registry_written": False, "model_training_performed": False,
            "opened_json_manifest_count": 2, "opened_checksum_ledger_count": 2,
            "opened_identity_payload_member_count": 0, "implementation_sha256": discovered["implementation_sha256"],
            **{key: False for key in _FLAGS}}
        verify_comparison_result(result)
        return result
    except ManifestStructureAuditError:
        raise
    except (raw.RawInferenceLineageError, OSError, ValueError, UnicodeError, TypeError, RuntimeError, KeyError,
            EOFError, NotImplementedError, RecursionError, zlib.error, lzma.LZMAError,
            zipfile.BadZipFile, zipfile.LargeZipFile):
        raise ManifestStructureAuditError("audit_refused") from None
