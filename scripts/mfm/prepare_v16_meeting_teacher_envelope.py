#!/usr/bin/env python3
"""Select exact speaker-bound AMI/ICSI transcript spans for a teacher envelope.

This accepts one already pinned source-only window, never a future meeting.
The displayed UTF-8 text is a derived transcription; the copied original
window retains every XML member/word byte pointer. No target or rights are
created here. The caller chooses indices after reading the source itself.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import stat

from scripts.mfm.prepare_v16_annotation_drafts import write_private_new
from scripts.mfm.prepare_v16_observed_history_teacher_case import _raw


def _load_pinned(path: Path, pin: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != pin:
        raise ValueError("source-only window differs from exact upstream pin")
    row = json.loads(raw)
    if not isinstance(row, dict) or row.get("training_admitted") is True:
        raise ValueError("source-only window has unsupported status")
    return raw, row


def prepare(*, source_window: Path, source_window_sha256: str,
            indices: tuple[int, ...], family: str, case_id: str,
            authorization_id: str, output_dir: Path) -> dict:
    raw, row = _load_pinned(source_window, source_window_sha256)
    if family == "ami":
        if (row.get("schema") != "mfm-ami-original-word-window-v1" or
                row.get("status") != "source_candidate_unreviewed" or
                not isinstance(row.get("derived_turns"), list)):
            raise ValueError("AMI source requires speaker-bound derived turns")
        spans = row["derived_turns"]
        meeting = row["meeting_id"]
        cutoff = row["as_of"]["window_end_seconds"]
        origin_family = "ami-manual-2017"
        speaker = lambda x: x["participant_id"]
        end = lambda x: x["end_seconds"]
        prefix = "ami"
        # A derived turn must bind to its original, byte-anchored word IDs.
        original_words = {word["word_id"] for word in row["words"]}
        if len(original_words) != len(row["words"]):
            raise ValueError("AMI source window repeats original word IDs")
        for span in spans:
            if not span["word_ids"] or not set(span["word_ids"]).issubset(original_words):
                raise ValueError("AMI derived turn lacks original word ancestry")
    elif family == "icsi":
        if (row.get("schema") != "mfm-v16-icsi-derived-source-excerpt-v1" or
                row.get("status") != "source_only_unadmitted" or
                not isinstance(row.get("spans"), list)):
            raise ValueError("ICSI source requires speaker-bound derived spans")
        spans = row["spans"]
        meeting = row["meeting_id"]
        # This excerpt ends before its parent 600-second window. Its own
        # cutoff is the selected clip end, never the unseen later 180 seconds.
        cutoff = row["clip_end_seconds"]
        origin_family = "icsi-core-nxt-2016"
        speaker = lambda x: x["participant_id"]
        end = lambda x: float(x["end_seconds"])
        prefix = "icsi"
        for span in spans:
            if not span["words"] or any(not word.get("source", {}).get("element_sha256")
                                          for word in span["words"]):
                raise ValueError("ICSI derived span lacks original word byte ancestry")
    else:
        raise ValueError("unsupported meeting source family")
    if not indices or len(set(indices)) != len(indices) or len(indices) > 32 or \
            any(isinstance(i, bool) or not isinstance(i, int) or
                i < 0 or i >= len(spans) for i in indices):
        raise ValueError("selected turn indices must be unique and bounded")
    if (output_dir.exists() or output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[2])):
        raise ValueError("new private source output must be outside Git")
    output_dir.mkdir(mode=0o700, parents=False)
    if stat.S_IMODE(output_dir.stat().st_mode) & 0o077:
        raise ValueError("private source directory must be 0700")
    write_private_new(output_dir / "source-window.json", raw)
    selected = []
    for i in indices:
        span = spans[i]
        participant = speaker(span)
        if not isinstance(participant, str) or not participant:
            raise ValueError("selected source span has no speaker identity")
        text = span["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("selected source span has no text")
        raw_text = text.encode("utf-8")
        name = f"source-{i:03d}.txt"
        write_private_new(output_dir / name, raw_text)
        ref = f"{prefix}-{meeting.lower()}-turn-{i:03d}"
        selected.append({"ref_id": ref, "role": "outside_source",
                         "modality": "text", "subject_ref": f"{prefix}-{participant.lower()}",
                         "speaker_ref": f"{prefix}-{participant.lower()}",
                         "source_item_ref": ref + "-derived-transcript",
                         "observed_at": None, "recorded_at": None,
                         "temporal_granularity": "instant",
                         "minimum_sensitivity": "public", "path": name,
                         "sha256": sha256(raw_text).hexdigest(),
                         "relative_time": {"timeline_ref": f"{prefix}-{meeting.lower()}",
                                           "end_seconds": end(span)}})
    envelope = {"schema": "mfm-v16-observed-history-envelope-v1",
                "case_id": case_id, "split": "train",
                "authorization_id": authorization_id,
                "host_id": f"{prefix}-source-observer",
                "self_id": "alice-assistant-self",
                "authority_id": f"{prefix}-public-history-authority",
                "source_family": origin_family,
                "as_of": {"timeline_ref": f"{prefix}-{meeting.lower()}",
                          "cutoff_seconds": cutoff},
                "source_window": {"path": "source-window.json",
                                  "sha256": source_window_sha256},
                "sources": selected, "targets": [],
                "training_admitted": False, "rights_status": "unverified"}
    rendered = _raw(envelope)
    write_private_new(output_dir / "envelope.json", rendered)
    return {"case_id": case_id, "envelope_sha256": sha256(rendered).hexdigest(),
            "source_window_sha256": source_window_sha256,
            "selected_turn_indices": list(indices), "training_admitted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-window", type=Path, required=True)
    parser.add_argument("--source-window-sha256", required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--family", choices=("ami", "icsi"), required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(source_window=args.source_window,
                             source_window_sha256=args.source_window_sha256,
                             indices=tuple(args.indices), family=args.family,
                             case_id=args.case_id,
                             authorization_id=args.authorization_id,
                             output_dir=args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
