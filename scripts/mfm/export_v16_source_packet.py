#!/usr/bin/env python3
"""Export one receipt-pinned MFM source packet, never the shared fragment blob."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from scripts.mfm.materialize_v16_source_slices import _canonical, isolated_packet
from scripts.mfm.prepare_v16_annotation_drafts import write_private_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-slices-dir", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = isolated_packet(args.source_slices_dir, packet_id=args.packet_id,
                          receipt_sha256=args.receipt_sha256)
    raw = _canonical(row) + b"\n"
    write_private_new(args.output, raw)
    print(json.dumps({"schema": row["schema"], "packet_id": row["packet_id"],
                      "sources": len(row["sources"]), "output_sha256": sha256(raw).hexdigest(),
                      "training_admitted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
