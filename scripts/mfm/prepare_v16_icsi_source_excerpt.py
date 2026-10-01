#!/usr/bin/env python3
"""Create a small, speaker-bound source excerpt from a pinned ICSI window.

The groups are derived presentation spans, not annotated turn boundaries or
formation targets. Each word retains its original ZIP-member byte anchor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

try:
    from .acquire_v16_icsi_source_windows import (
        ROOT, _write_private, canonical, WINDOW_SCHEMA,
    )
except ImportError:  # Direct CLI execution from scripts/mfm/.
    from acquire_v16_icsi_source_windows import (
        ROOT, _write_private, canonical, WINDOW_SCHEMA,
    )

SCHEMA = "mfm-v16-icsi-derived-source-excerpt-v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excerpt(windows_path: Path, receipt_path: Path, *, meeting_id: str,
            as_of_end_seconds: int, clip_start_seconds: int,
            clip_end_seconds: int, max_words_per_span: int = 60) -> dict:
    raw_receipt = receipt_path.read_bytes()
    receipt = json.loads(raw_receipt)
    if file_sha256(windows_path) != receipt["windows_sha256"]:
        raise ValueError("source windows do not match acquisition receipt")
    if not (0 <= clip_start_seconds < clip_end_seconds <= as_of_end_seconds):
        raise ValueError("clip is outside its as-of boundary")
    if clip_end_seconds - clip_start_seconds > 120:
        raise ValueError("source excerpt must remain a bounded two-minute clip")
    if max_words_per_span <= 0 or max_words_per_span > 60:
        raise ValueError("invalid word limit")
    matching = []
    with windows_path.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["meeting_id"] == meeting_id and
                    row["as_of_end_seconds"] == as_of_end_seconds):
                matching.append(row)
    if len(matching) != 1 or matching[0]["schema"] != WINDOW_SCHEMA:
        raise ValueError("source window not uniquely present")
    row = matching[0]
    if clip_start_seconds < row["interval_start_seconds"]:
        raise ValueError("clip would need a preceding source window")
    words = [w for w in row["observations"]
             if (Decimal(w["start_seconds"]) >= clip_start_seconds and
                 Decimal(w["end_seconds"]) <= clip_end_seconds)]
    by_channel = defaultdict(list)
    for word in words:
        by_channel[word["channel"], word["participant_id"]].append(word)
    spans = []
    for (channel, participant), channel_words in by_channel.items():
        channel_words.sort(key=lambda w: (Decimal(w["start_seconds"]),
                                          Decimal(w["end_seconds"]),
                                          w["source"]["byte_start"]))
        current = []

        def flush() -> None:
            if not current:
                return
            spans.append({"channel": channel, "participant_id": participant,
                          "start_seconds": current[0]["start_seconds"],
                          "end_seconds": max((w["end_seconds"] for w in current),
                                             key=Decimal),
                          "text": " ".join(w["text"] for w in current),
                          "words": list(current)})

        for word in channel_words:
            gap = (Decimal(word["start_seconds"]) -
                   Decimal(current[-1]["end_seconds"])) if current else Decimal(0)
            if current and (gap > Decimal("0.8") or len(current) >= max_words_per_span):
                flush()
                current = []
            current.append(word)
        flush()
    spans.sort(key=lambda s: (Decimal(s["start_seconds"]),
                              Decimal(s["end_seconds"]), s["channel"]))
    return {"schema": SCHEMA, "status": "source_only_unadmitted",
            "source_family": row["source_family"], "meeting_id": meeting_id,
            "window_id": row["window_id"], "as_of_end_seconds": as_of_end_seconds,
            "clip_start_seconds": clip_start_seconds,
            "clip_end_seconds": clip_end_seconds,
            "acquisition_receipt_sha256": sha256(raw_receipt).hexdigest(),
            "source_windows_sha256": receipt["windows_sha256"],
            "boundary_method": "derived per-channel gap <=0.8 s or max 60 words; not an NXT turn",
            "spans": spans}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--as-of-end-seconds", type=int, required=True)
    parser.add_argument("--clip-start-seconds", type=int, required=True)
    parser.add_argument("--clip-end-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = excerpt(args.windows, args.receipt, meeting_id=args.meeting_id,
                     as_of_end_seconds=args.as_of_end_seconds,
                     clip_start_seconds=args.clip_start_seconds,
                     clip_end_seconds=args.clip_end_seconds)
    if args.output.exists():
        raise ValueError("refusing to replace an existing source excerpt")
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("source excerpt belongs outside Git")
    _write_private(args.output, canonical(output) + b"\n")
    print(json.dumps({"spans": len(output["spans"]),
                      "words": sum(len(s["words"]) for s in output["spans"]),
                      "sha256": sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
