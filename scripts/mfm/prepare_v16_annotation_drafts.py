#!/usr/bin/env python3
"""Prepare unadmitted MFM 1.6 annotation drafts from verified source packets.

This deliberately creates neither a target nor an admissible corpus row. The
original source bytes must be inspected for narrow byte anchors and independent
review before a formation target can be frozen.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.formation_semantics_v16 import FULL_ROLE_ADJUDICATION_DIMENSIONS
from scripts.mfm.build_v16_source_review_packets import build_packets, _load


SCHEMA = "mfm-v16-annotation-draft-v1"


def prepare_drafts(packet_path: Path, *, packet_sha256: str,
                   dataset_dir: Path, inventory_path: Path,
                   inventory_sha256: str) -> bytes:
    """Rebuild the exact packet bytes before creating incomplete draft rows."""
    raw = packet_path.read_bytes()
    if sha256(raw).hexdigest() != packet_sha256 or len(packet_sha256) != 64:
        raise ValueError("source packet differs from pinned SHA-256")
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("source packets must be nonempty canonical JSONL")
    lines = raw.splitlines(keepends=True)
    packets = [_load(line) for line in lines]
    if any(packet.get("schema") != "mfm-v16-source-review-packet-v1" or
           packet.get("status") != "candidate_unreviewed" or
           packet.get("training_admitted") is not False or
           packet.get("inventory_sha256") != inventory_sha256
           for packet in packets):
        raise ValueError("source packet has wrong unadmitted provenance")
    hosts = sorted({packet["lineage"]["host_family"] for packet in packets})
    days = sorted({packet["window"]["end_day_index"] for packet in packets})
    regenerated = build_packets(dataset_dir, inventory_path,
                                expected_inventory_sha256=inventory_sha256,
                                hosts=tuple(hosts), days=tuple(days))
    if raw != regenerated:
        raise ValueError("source packet differs from original pinned source reconstruction")

    result = []
    for ordinal, (packet, line) in enumerate(zip(packets, lines, strict=True), 1):
        anchors = [{
            "source_id": source["source_id"],
            "source_type": source["source_type"],
            "source_path": source["source_path"],
            "source_file_sha256": source["source_file_sha256"],
            "source_locator": source["source_locator"],
            "unit_sha256": source["unit_sha256"],
            "original_file_byte_anchor": {"start": None, "end": None},
        } for source in packet["sources"]]
        for source, anchor in zip(packet["sources"], anchors, strict=True):
            if "anchor_locator" in source:
                anchor["anchor_locator"] = source["anchor_locator"]
        draft = {
            "schema": SCHEMA, "status": "unreviewed_unadmitted_annotation_draft",
            "training_admitted": False,
            "packet_id": packet["packet_id"],
            "provenance": {
                "packet_file_sha256": packet_sha256,
                "packet_row_number": ordinal,
                "packet_row_sha256": sha256(line).hexdigest(),
                "inventory_sha256": inventory_sha256,
                "lineage": packet["lineage"],
                "window": packet["window"],
            },
            "dimension_review": {
                dimension: "unknown" for dimension in sorted(FULL_ROLE_ADJUDICATION_DIMENSIONS)
            },
            "source_anchor_inventory": anchors,
        }
        result.append(json.dumps(draft, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8") + b"\n")
    return b"".join(result)


def write_private_new(path: Path, raw: bytes) -> None:
    """Publish once with restrictive permissions; never overwrite reviewed work."""
    output = path.absolute()
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("annotation drafts belong in owner custody outside the repository")
    if not output.parent.is_dir():
        raise ValueError("create the owner-controlled output directory first")
    temp_name = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".mfm-v16-draft-", dir=output.parent)
        with os.fdopen(fd, "wb") as file:
            os.fchmod(file.fileno(), 0o600)
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.link(temp_name, output)
    finally:
        if temp_name is not None:
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--packets-sha256", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = prepare_drafts(args.packets, packet_sha256=args.packets_sha256,
                         dataset_dir=args.dataset_dir, inventory_path=args.inventory,
                         inventory_sha256=args.inventory_sha256)
    write_private_new(args.output, raw)
    print(json.dumps({"schema": SCHEMA, "drafts": raw.count(b"\n"),
                      "output_sha256": sha256(raw).hexdigest(),
                      "training_admitted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
