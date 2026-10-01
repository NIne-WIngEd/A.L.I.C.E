#!/usr/bin/env python3
"""Bind source-only review packets to exact JSON byte slices in private custody.

This is an annotation aid, not a formation target, rights record or training
corpus. The blob contains only selected source-native values, never a complete
future-bearing structural file. The index preserves the original file digest
and exact offsets so a later author can verify each narrow evidence span.
"""

from __future__ import annotations

import argparse
from base64 import b64encode
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scripts.mfm.build_v16_source_review_packets import (
    _canonical, _check_view, _load, _source_path, build_packets,
)


SCHEMA = "mfm-v16-source-slice-index-v1"
WHITESPACE = b" \t\r\n"


def _space(raw: bytes, index: int) -> int:
    while index < len(raw) and raw[index] in WHITESPACE:
        index += 1
    return index


def _string_end(raw: bytes, index: int) -> int:
    if index >= len(raw) or raw[index] != 34:
        raise ValueError("expected JSON string")
    index += 1
    while index < len(raw):
        if raw[index] == 92:  # Backslash: the next byte cannot end the string.
            index += 2
        elif raw[index] == 34:
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated JSON string")


def _scan(raw: bytes, index: int, path: tuple[object, ...],
          locations: dict[str, tuple[int, int]]) -> int:
    """Walk parsed JSON byte positions; full document validity is checked below."""
    index = _space(raw, index)
    start = index
    if index >= len(raw):
        raise ValueError("incomplete JSON")
    marker = raw[index]
    if marker == 123:  # object
        index = _space(raw, index + 1)
        if index < len(raw) and raw[index] == 125:
            index += 1
        else:
            while True:
                end = _string_end(raw, index)
                key = json.loads(raw[index:end])
                index = _space(raw, end)
                if index >= len(raw) or raw[index] != 58:
                    raise ValueError("missing JSON colon")
                index = _scan(raw, index + 1, (*path, key), locations)
                index = _space(raw, index)
                if index < len(raw) and raw[index] == 125:
                    index += 1
                    break
                if index >= len(raw) or raw[index] != 44:
                    raise ValueError("missing JSON object comma")
                index = _space(raw, index + 1)
    elif marker == 91:  # array
        index = _space(raw, index + 1)
        if index < len(raw) and raw[index] == 93:
            index += 1
        else:
            ordinal = 0
            while True:
                index = _scan(raw, index, (*path, ordinal), locations)
                ordinal += 1
                index = _space(raw, index)
                if index < len(raw) and raw[index] == 93:
                    index += 1
                    break
                if index >= len(raw) or raw[index] != 44:
                    raise ValueError("missing JSON array comma")
                index = _space(raw, index + 1)
    elif marker == 34:
        index = _string_end(raw, index)
    else:
        while index < len(raw) and raw[index] not in b",}] \t\r\n":
            index += 1
        if index == start:
            raise ValueError("invalid JSON primitive")
    if path in (("facts",), ("anchor_window",)) or (
            len(path) == 2 and path[0] == "records" and type(path[1]) is int):
        pointer = "/" + "/".join(str(component) for component in path)
        if pointer in locations:
            raise ValueError("duplicate source-native JSON pointer")
        locations[pointer] = (start, index)
    return index


def source_slice_locations(raw: bytes) -> dict[str, tuple[int, int]]:
    """Return offsets into *original UTF-8 bytes*, never reserialized offsets."""
    document = _load(raw)  # Reject duplicate keys and malformed UTF-8/JSON.
    locations: dict[str, tuple[int, int]] = {}
    if _space(raw, _scan(raw, 0, (), locations)) != len(raw):
        raise ValueError("trailing source bytes")
    expected = ({f"/records/{index}" for index in range(len(document["records"]))}
                if "records" in document else {"/facts", "/anchor_window"})
    if set(locations) != expected:
        raise ValueError("source-native JSON pointers differ")
    for pointer, (start, end) in locations.items():
        key = (int(pointer.split("/")[-1]) if pointer.startswith("/records/")
               else pointer[1:])
        original = document["records"][key] if type(key) is int else document[key]
        if _load(raw[start:end]) != original:
            raise ValueError("source slice does not match parsed original")
    return locations


def _new_private_dir(output_dir: Path) -> Path:
    output = output_dir.absolute()
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("source slices belong in owner custody outside Git")
    if not output.parent.is_dir() or output.exists():
        raise ValueError("create a new output path beneath an existing private directory")
    temp = Path(tempfile.mkdtemp(prefix=".mfm-v16-slices-", dir=output.parent))
    os.chmod(temp, 0o700)
    return temp


def _private_file(path: Path):
    """Create a file with mode 0600 from its first observable instant."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    return os.fdopen(fd, "wb")


def _opaque_id(receipt_sha256: str, kind: str, original: str) -> str:
    digest = hmac.new(bytes.fromhex(receipt_sha256),
                      _canonical({"kind": kind, "original": original}),
                      sha256).hexdigest()
    return f"{kind}-{digest}"


def materialize(packet_path: Path, *, packet_sha256: str,
                dataset_dir: Path, inventory_path: Path,
                inventory_sha256: str, output_dir: Path) -> dict:
    """Rebuild pinned packets, then write exact fragments and a provenance index."""
    packet_raw = packet_path.read_bytes()
    if len(packet_sha256) != 64 or sha256(packet_raw).hexdigest() != packet_sha256:
        raise ValueError("source packet differs from pinned SHA-256")
    if not packet_raw or not packet_raw.endswith(b"\n"):
        raise ValueError("source packet must be nonempty canonical JSONL")
    packets = [_load(line) for line in packet_raw.splitlines()]
    if any(row.get("schema") != "mfm-v16-source-review-packet-v1" or
           row.get("status") != "candidate_unreviewed" or
           row.get("training_admitted") is not False or
           row.get("inventory_sha256") != inventory_sha256 for row in packets):
        raise ValueError("source packet has wrong unadmitted provenance")
    hosts = sorted({row["lineage"]["host_family"] for row in packets})
    days = sorted({row["window"]["end_day_index"] for row in packets})
    rebuilt = build_packets(dataset_dir, inventory_path,
                            expected_inventory_sha256=inventory_sha256,
                            hosts=tuple(hosts), days=tuple(days))
    if rebuilt != packet_raw:
        raise ValueError("source packet differs from original pinned reconstruction")

    root = dataset_dir.resolve()
    temp = _new_private_dir(output_dir)
    cached_files: dict[str, tuple[bytes, dict[str, tuple[int, int]]]] = {}
    packed: dict[tuple[str, str], dict] = {}
    counts = {"packets": 0, "source_references": 0, "unique_fragments": 0}
    try:
        with _private_file(temp / "fragments.bin") as blob, \
                _private_file(temp / "index.jsonl") as index:
            for packet in packets:
                sources = []
                through_day = packet["window"]["end_day_index"]
                for source in packet["sources"]:
                    relpath = source["source_path"]
                    if relpath not in cached_files:
                        path = _source_path(root, packet["lineage"]["host_family"], {
                            "source_type": source["source_type"], "path": relpath})
                        original = path.read_bytes()
                        if sha256(original).hexdigest() != source["source_file_sha256"]:
                            raise ValueError("original source SHA-256 differs")
                        cached_files[relpath] = (original, source_slice_locations(original))
                    original, locations = cached_files[relpath]
                    if sha256(original).hexdigest() != source["source_file_sha256"]:
                        raise ValueError("source parent digest differs across packet rows")
                    pointers = ([source["source_locator"], source["anchor_locator"]]
                                if source["source_type"] == "profile_ltm"
                                else [source["source_locator"]])
                    if source["source_type"] == "profile_ltm":
                        if (through_day != source["record"]["anchor_window"]["total_days"] or
                                pointers != ["/facts", "/anchor_window"]):
                            raise ValueError("profile appears before its whole-run horizon")
                    else:
                        day = source["observed_day_index"]
                        if not 1 <= day <= through_day or pointers != [f"/records/{day - 1}"]:
                            raise ValueError("source slice leaks a future day")
                    fragments = []
                    parsed = {}
                    for pointer in pointers:
                        if pointer not in locations:
                            raise ValueError("missing original source-native pointer")
                        start, end = locations[pointer]
                        raw = original[start:end]
                        unit = _load(raw)
                        _check_view(unit)
                        parsed[pointer] = unit
                        key = (relpath, pointer)
                        if key not in packed:
                            offset = blob.tell()
                            blob.write(raw)
                            packed[key] = {"blob_offset": offset, "byte_length": len(raw),
                                           "sha256": sha256(raw).hexdigest(),
                                           "original_file_byte_anchor": {
                                               "start": start, "end": end},
                                           "source_locator": pointer}
                        fragments.append(packed[key])
                    record = ({"facts": parsed["/facts"],
                               "anchor_window": parsed["/anchor_window"]}
                              if source["source_type"] == "profile_ltm" else
                              parsed[pointers[0]])
                    if (record != source["record"] or
                            sha256(_canonical(record)).hexdigest() != source["unit_sha256"]):
                        raise ValueError("original source slice differs from packet unit")
                    sources.append({"source_id": source["source_id"],
                                    "source_type": source["source_type"],
                                    "parent_source_path": relpath,
                                    "parent_source_file_sha256": source["source_file_sha256"],
                                    "unit_sha256": source["unit_sha256"],
                                    "fragments": fragments})
                row = {"schema": SCHEMA, "status": "source-only-unreviewed-unadmitted",
                       "training_admitted": False, "rights_status": "unverified",
                       "packet_id": packet["packet_id"],
                       "packet_sha256": packet_sha256,
                       "inventory_sha256": inventory_sha256,
                       "lineage": packet["lineage"], "window": packet["window"],
                       "sources": sources}
                index.write(_canonical(row) + b"\n")
                counts["packets"] += 1
                counts["source_references"] += len(sources)
            blob.flush()
            index.flush()
            os.fsync(blob.fileno())
            os.fsync(index.fileno())
        counts["unique_fragments"] = len(packed)
        counts["fragment_blob_sha256"] = sha256((temp / "fragments.bin").read_bytes()).hexdigest()
        counts["index_sha256"] = sha256((temp / "index.jsonl").read_bytes()).hexdigest()
        counts["schema"] = SCHEMA
        counts["training_admitted"] = False
        with _private_file(temp / "receipt.json") as receipt:
            receipt.write(_canonical(counts) + b"\n")
            receipt.flush()
            os.fsync(receipt.fileno())
        if output_dir.absolute().exists():
            raise ValueError("output directory appeared during materialization")
        os.rename(temp, output_dir.absolute())
        return counts
    except BaseException:
        shutil.rmtree(temp)
        raise


def isolated_packet(source_dir: Path, *, packet_id: str,
                    receipt_sha256: str) -> dict:
    """Return only one packet's exact fragments after checking external pins.

    A teacher caller must receive this result, not `fragments.bin` or an
    original structural file. The shared blob may hold later days; offsets
    outside the selected index row never enter the result.
    """
    directory = source_dir.resolve()
    receipt_raw = (directory / "receipt.json").read_bytes()
    if len(receipt_sha256) != 64 or sha256(receipt_raw).hexdigest() != receipt_sha256:
        raise ValueError("source slice receipt differs from external SHA-256 pin")
    receipt = _load(receipt_raw)
    if receipt.get("schema") != SCHEMA or receipt.get("training_admitted") is not False:
        raise ValueError("source slice receipt lacks unadmitted provenance")
    index_raw = (directory / "index.jsonl").read_bytes()
    blob = (directory / "fragments.bin").read_bytes()
    if (sha256(index_raw).hexdigest() != receipt.get("index_sha256") or
            sha256(blob).hexdigest() != receipt.get("fragment_blob_sha256")):
        raise ValueError("source slice index or fragment blob differs from pinned receipt")
    selected = []
    for line in index_raw.splitlines():
        row = _load(line)
        if row.get("packet_id") == packet_id:
            selected.append(row)
    if len(selected) != 1:
        raise ValueError("packet ID missing or duplicated in pinned source index")
    row = selected[0]
    if (row.get("schema") != SCHEMA or row.get("training_admitted") is not False or
            row.get("rights_status") != "unverified" or "target" in row):
        raise ValueError("selected packet has unexpected admission or target state")
    day = row["window"]["end_day_index"]
    sources = []
    for source in row["sources"]:
        if source["source_type"] != "profile_ltm" and (
                not source["source_id"].endswith(
                    f"/day-{int(source['source_id'].rsplit('day-', 1)[1]):02d}") or
                int(source["source_id"].rsplit("day-", 1)[1]) > day):
            raise ValueError("selected source is from a future day")
        fragments = []
        parsed = {}
        for fragment in source["fragments"]:
            offset, size = fragment["blob_offset"], fragment["byte_length"]
            if type(offset) is not int or type(size) is not int or offset < 0 or \
                    size < 1 or offset + size > len(blob):
                raise ValueError("source fragment offsets are invalid")
            raw = blob[offset:offset + size]
            if sha256(raw).hexdigest() != fragment["sha256"]:
                raise ValueError("source fragment SHA-256 differs")
            parsed[fragment["source_locator"]] = _load(raw)
            handle = hmac.new(bytes.fromhex(receipt_sha256), _canonical({
                "packet_id": packet_id, "source_id": source["source_id"],
                "source_locator": fragment["source_locator"],
                "sha256": fragment["sha256"]}), sha256).hexdigest()
            fragments.append({"slice_handle": handle,
                              "sha256": fragment["sha256"],
                              "payload_base64": b64encode(raw).decode("ascii")})
        if source["source_type"] == "profile_ltm":
            if set(parsed) != {"/facts", "/anchor_window"} or \
                    parsed["/anchor_window"]["total_days"] > day:
                raise ValueError("profile source is premature or incomplete")
            value = {"facts": parsed["/facts"],
                     "anchor_window": parsed["/anchor_window"]}
        else:
            if len(parsed) != 1:
                raise ValueError("daily source must have one source slice")
            value = next(iter(parsed.values()))
            if value["day_index"] > day:
                raise ValueError("selected source contents are from a future day")
        _check_view(value)
        if sha256(_canonical(value)).hexdigest() != source["unit_sha256"]:
            raise ValueError("selected source unit SHA-256 differs")
        sources.append({"source_id": _opaque_id(
                            receipt_sha256, "source", source["source_id"]),
                        "source_type": source["source_type"],
                        "fragments": fragments})
    return {"schema": "mfm-v16-isolated-source-packet-v1",
            "status": "source-only-unreviewed-unadmitted",
            "training_admitted": False, "rights_status": "unverified",
            "packet_id": _opaque_id(receipt_sha256, "packet", packet_id),
            "host_id": _opaque_id(receipt_sha256, "host", row["lineage"]["host_family"]),
            "window": row["window"],
            "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--packets-sha256", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.packets, packet_sha256=args.packets_sha256,
                                 dataset_dir=args.dataset_dir,
                                 inventory_path=args.inventory,
                                 inventory_sha256=args.inventory_sha256,
                                 output_dir=args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
