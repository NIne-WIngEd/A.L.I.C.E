"""Derive original E-INF existence lineage without accepting source proposals.

The caller MUST establish its private-source isolation boundary before calling:
even resolving the archive path or hashing it is a private operation. This
stdlib reader establishes no namespace, opens no source on import, performs no
extraction or writes, and reads only the uniquely selected E-INF member and its
same-root manifest/checksum metadata. A caller-supplied whole-archive pin is
mandatory; production orchestration must independently enforce the reviewed
raw-v5 pin below. Other exact pins support explicitly unqualified fixtures.

Parser bounds are resource admission limits, not identity-capability limits.
The manifest's generic public contract does not specify count-key aliases, so
none are guessed. Row count is observed, never used as scientific authority.
"""
from __future__ import annotations

from hashlib import sha256
import json
import lzma
import os
from pathlib import Path
import re
import stat
from typing import BinaryIO
import unicodedata
import zipfile
import zlib


SCHEMA = "alice-personality-raw-inference-lineage-v1"
STATE = "DERIVED_UNQUALIFIED"
REVIEWED_RAW_V5_ARCHIVE_SHA256 = "5d122894348692900c2ef7f1e22198464b01ecd2df34fdb3a7e4e42e77f99911"
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 4096
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_PROPOSAL_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 1024 * 1024
MAX_PROPOSAL_ROWS = 100_000
_LOGICAL_NAMES = ("einf_proposals.jsonl", "generation_manifest.json", "SHA256SUMS.txt")
_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")
_ID = re.compile(r"[^\s\x00-\x1f\x7f]{1,512}\Z")
_CHECKSUM_LINE = re.compile(r"([0-9a-fA-F]{64}) ([ *])(.+)\Z")
_WINDOWS_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)


class RawInferenceLineageError(ValueError):
    """Fixed sanitized admission failure; source payloads are never reported."""


def _fail(reason: str) -> None:
    raise RawInferenceLineageError(reason)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        _fail("invalid_expected_archive_sha256")
    return value.lower()


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    _fail("nonfinite_json")


def _finite_float(value: str) -> float:
    number = float(value)
    if number == float("inf") or number == -float("inf"):
        _fail("nonfinite_json")
    return number


def _json(data: bytes) -> object:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=_no_constant, parse_float=_finite_float)
    except RawInferenceLineageError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        _fail("invalid_json")


def _member_path(name: object) -> str:
    if type(name) is not str or not name or len(name) > 4096 \
            or unicodedata.normalize("NFC", name) != name or "\\" in name or ":" in name \
            or name.startswith("/") or any(unicodedata.category(char) in ("Cc", "Cs") for char in name):
        _fail("unsafe_archive_member_path")
    parts = name.split("/")
    if any(not part or part in (".", "..") or len(part) > 255
           or part.endswith((".", " ")) or _WINDOWS_RESERVED.fullmatch(part) for part in parts):
        _fail("unsafe_archive_member_path")
    return name


def _inventory(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries = archive.infolist()
    if not entries or len(entries) > MAX_ARCHIVE_ENTRIES:
        _fail("archive_inventory_bound")
    seen = {}
    folded = set()
    namespace_paths = {}
    files = {}
    for entry in entries:
        # orig_filename retains a NUL-bearing name that ZipInfo.filename clips.
        original = entry.orig_filename
        if original != entry.filename:
            _fail("unsafe_archive_member_path")
        directory = entry.is_dir()
        name = _member_path(original[:-1] if directory else original)
        key = name.casefold()
        if name in seen or key in folded:
            _fail("duplicate_or_aliased_archive_member")
        seen[name] = directory
        folded.add(key)
        parts = name.split("/")
        for end in range(1, len(parts) + 1):
            namespace_path = "/".join(parts[:end])
            canonical_key = namespace_path.casefold()
            if canonical_key in namespace_paths and namespace_paths[canonical_key] != namespace_path:
                _fail("duplicate_or_aliased_archive_member")
            namespace_paths[canonical_key] = namespace_path
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if mode not in ((0, stat.S_IFDIR) if directory else (0, stat.S_IFREG)):
            _fail("nonregular_archive_member")
        if not directory and entry.external_attr & 0x10:
            _fail("archive_member_type_conflict")
        if entry.flag_bits & (0x1 | 0x40 | 0x2000):
            _fail("encrypted_archive_member")
        if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
                                       zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
            _fail("unsupported_archive_compression")
        if entry.file_size < 0 or entry.compress_size < 0:
            _fail("invalid_archive_member_size")
        if directory:
            if entry.file_size != 0:
                _fail("directory_archive_member_has_payload")
        else:
            files[name] = entry
    for name in seen:
        parts = name.split("/")
        for end in range(1, len(parts)):
            ancestor = "/".join(parts[:end])
            if ancestor in seen and not seen[ancestor]:
                _fail("archive_file_directory_alias")
            if ancestor.casefold() in folded and ancestor not in seen:
                _fail("archive_file_directory_alias")
    return files


def _selected(files: dict[str, zipfile.ZipInfo]) -> tuple[str, str, str, str]:
    selected = []
    for logical in _LOGICAL_NAMES:
        matches = [name for name in files if name.rsplit("/", 1)[-1] == logical]
        if len(matches) != 1:
            _fail("missing_or_ambiguous_logical_member")
        selected.append(matches[0])
    roots = [name.rsplit("/", 1)[0] if "/" in name else "" for name in selected]
    if len(set(roots)) != 1:
        _fail("logical_member_root_mismatch")
    return roots[0], *selected


def _read_member(archive: zipfile.ZipFile, entry: zipfile.ZipInfo, limit: int) -> bytes:
    if entry.file_size > limit:
        _fail("selected_member_resource_bound")
    with archive.open(entry, "r") as member:
        data = member.read(limit + 1)
        if len(data) > limit or len(data) != entry.file_size or member.read(1):
            _fail("selected_member_resource_bound")
    return data


def _ledger(data: bytes, files: dict[str, zipfile.ZipInfo], root: str,
            ledger_name: str) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        _fail("invalid_checksum_ledger")
    result = {}
    for line in text.splitlines():
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            _fail("invalid_checksum_ledger")
        digest, _, supplied = match.groups()
        supplied = _member_path(supplied)
        # Permit the two exact ledger conventions, never fuzzy prefix removal.
        relative = f"{root}/{supplied}" if root else supplied
        candidates = {name for name in (supplied, relative) if name in files}
        if len(candidates) != 1:
            _fail("checksum_member_missing_or_ambiguous")
        name = candidates.pop()
        if (root and not name.startswith(root + "/")) or name == ledger_name:
            _fail("checksum_member_namespace_conflict")
        if name in result:
            _fail("duplicate_checksum_member")
        result[name] = digest.lower()
    if not result:
        _fail("empty_checksum_ledger")
    return result


def _proposal_ids(data: bytes) -> list[str]:
    ids = set()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines or len(lines) > MAX_PROPOSAL_ROWS:
        _fail("proposal_row_count_bound")
    for line in lines:
        if not line or len(line) > MAX_JSONL_LINE_BYTES:
            _fail("proposal_line_resource_bound")
        row = _json(line)
        if type(row) is not dict or row.get("provenance_class") != "E-INF" \
                or type(row.get("provenance_class")) is not str:
            _fail("invalid_proposal_namespace")
        proposal_id = row.get("proposal_id")
        if type(proposal_id) is not str or not _ID.fullmatch(proposal_id) \
                or any(unicodedata.category(char) in ("Cc", "Cs") for char in proposal_id):
            _fail("invalid_original_proposal_id")
        if proposal_id in ids:
            _fail("duplicate_original_proposal_id")
        ids.add(proposal_id)
    return sorted(ids)


def _file_state(value: os.stat_result) -> tuple:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _file_identity(value: os.stat_result) -> tuple:
    # Windows stat/fstat expose different ctime semantics on some Python builds.
    # Compare shared identity across APIs, then each API's full state to itself.
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _archive_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        _fail("invalid_archive_path")
    path = Path(value).absolute()
    if ".." in path.parts:
        _fail("unsafe_archive_path")
    for part in (path, *path.parents):
        metadata = part.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) \
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            _fail("linked_archive_path")
    if not stat.S_ISREG(path.lstat().st_mode):
        _fail("nonregular_source_archive")
    return path.resolve(strict=True)


def _hash_stream(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = sha256()
    count = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        count += len(block)
        if count > MAX_ARCHIVE_BYTES:
            _fail("source_archive_resource_bound")
        digest.update(block)
    return digest.hexdigest()


def derive_raw_inference_registry(raw_archive_path: str | Path, *,
                                  expected_archive_sha256: str) -> tuple[dict, dict]:
    """Return pinned original proposal IDs and a sanitized unqualified receipt.

    There is no privacy bypass/namespace fixture flag: the caller must admit
    its environment before this function. Supplying a fixture pin cannot grant
    source acceptance, historical truth, qualification or gradient authority.
    Receipt paths are archive logical metadata only; IDs exist only in registry.
    """
    expected = _digest(expected_archive_sha256)
    try:
        path = _archive_path(raw_archive_path)
        before = path.stat()
        if before.st_size > MAX_ARCHIVE_BYTES:
            _fail("source_archive_resource_bound")
        code_sha = sha256(Path(__file__).read_bytes()).hexdigest()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                             | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            descriptor_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(descriptor_before.st_mode) \
                    or _file_identity(descriptor_before) != _file_identity(before):
                _fail("source_archive_changed")
            if _hash_stream(stream) != expected:
                _fail("source_archive_sha256_mismatch")
            with zipfile.ZipFile(stream, "r") as archive:
                files = _inventory(archive)
                root, proposal_name, manifest_name, ledger_name = _selected(files)
                ledger_bytes = _read_member(archive, files[ledger_name], MAX_METADATA_BYTES)
                checksums = _ledger(ledger_bytes, files, root, ledger_name)
                if proposal_name not in checksums or manifest_name not in checksums:
                    _fail("required_member_missing_from_checksum_ledger")
                manifest_bytes = _read_member(archive, files[manifest_name], MAX_METADATA_BYTES)
                if sha256(manifest_bytes).hexdigest() != checksums[manifest_name]:
                    _fail("manifest_checksum_mismatch")
                if type(_json(manifest_bytes)) is not dict:
                    _fail("invalid_generation_manifest")
                proposal_bytes = _read_member(archive, files[proposal_name], MAX_PROPOSAL_BYTES)
                member_sha = sha256(proposal_bytes).hexdigest()
                if member_sha != checksums[proposal_name]:
                    _fail("proposal_member_checksum_mismatch")
                registry = {"EINF": _proposal_ids(proposal_bytes)}
            if _hash_stream(stream) != expected \
                    or _file_state(os.fstat(stream.fileno())) != _file_state(descriptor_before) \
                    or _file_state(_archive_path(path).stat()) != _file_state(before):
                _fail("source_archive_changed")
        if sha256(Path(__file__).read_bytes()).hexdigest() != code_sha:
            _fail("lineage_reader_code_changed")
        receipt = {"schema": SCHEMA, "state": STATE,
                   "source_archive_sha256": expected,
                   "source_member_path": proposal_name,
                   "source_member_sha256": member_sha,
                   "source_member_bytes": len(proposal_bytes),
                   "generation_manifest_path": manifest_name,
                   "generation_manifest_sha256": sha256(manifest_bytes).hexdigest(),
                   "checksum_ledger_path": ledger_name,
                   "checksum_ledger_sha256": sha256(ledger_bytes).hexdigest(),
                   "registry_sha256": sha256(_canonical(registry)).hexdigest(),
                   "raw_inference_count": len(registry["EINF"]),
                   "code_sha256": code_sha,
                   "mechanical_fixture_only": expected != REVIEWED_RAW_V5_ARCHIVE_SHA256,
                   "acceptance_authority": False, "private_gradient_authorized": False,
                   "historical_authority_granted": False, "training_authorized": False,
                   "behavior_qualification": None}
        receipt["receipt_sha256"] = sha256(_canonical(receipt)).hexdigest()
        return registry, receipt
    except RawInferenceLineageError:
        raise
    except (OSError, ValueError, UnicodeError, RuntimeError, KeyError, EOFError,
            zlib.error, lzma.LZMAError, zipfile.BadZipFile, zipfile.LargeZipFile,
            NotImplementedError, RecursionError):
        raise RawInferenceLineageError("raw_lineage_source_unreadable") from None
