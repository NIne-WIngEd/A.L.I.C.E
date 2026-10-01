#!/usr/bin/env python3
"""Inventory original AMI XML words and export bounded, unreviewed source windows.

The archive is the official 2017 manual annotation release. This program does
not read AMI abstracts, extractive summaries, decisions, or annotation labels.
It supplies no MFM formation target, split, rights signature, or model result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


ARCHIVE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ARCHIVE_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
LICENSE_SHA256 = "d52fd9c7c19eec9f0bec3107554c10937203e261259d73b01637a61501fec7f7"
MEETINGS_SHA256 = "8ab6cdcf03ed863e839418f31c7380392397a2eed892d73b9bc1b438c7ec520a"
SESSION_IDS = ("ES2003", "ES2005", "IS1000", "IS1001", "TS3003", "TS3004")
NITE_ID = "{http://nite.sourceforge.net/}id"
WORD_TAG = re.compile(rb"<w\s[^>]*>.*?</w>", re.DOTALL)
MEETING_ID = re.compile(r"(?:ES|IS|TS)\d{4}[abcd]\Z")
MAX_MEMBER_SIZE = 8_000_000


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _member(archive: zipfile.ZipFile, name: str) -> bytes:
    infos = [x for x in archive.infolist() if x.filename == name]
    if len(infos) != 1 or infos[0].file_size > MAX_MEMBER_SIZE:
        raise ValueError(f"missing, duplicated or oversized ZIP member: {name}")
    return archive.read(infos[0])


def original_words(raw: bytes, *, member: str) -> tuple[dict, ...]:
    """Bind each XML word and its decoded text to its original member byte span."""
    parsed = ET.fromstring(raw)
    elements = list(parsed.findall("w"))
    matches = list(WORD_TAG.finditer(raw))
    if len(elements) != len(matches) or not elements:
        raise ValueError(f"word-element byte parsing mismatch: {member}")
    seen = set()
    words = []
    for node, match in zip(elements, matches, strict=True):
        original = ET.fromstring(b'<root xmlns:nite="http://nite.sourceforge.net/">' +
                                 match.group() + b'</root>')[0]
        if original.attrib != node.attrib or original.text != node.text or list(original):
            raise ValueError(f"XML word or byte alignment mismatch: {member}")
        word_id = node.get(NITE_ID)
        if not word_id or word_id in seen:
            raise ValueError(f"missing or duplicate word ID: {member}")
        seen.add(word_id)
        if node.find("*") is not None:
            raise ValueError(f"nested word element: {member}")
        start, end = node.get("starttime"), node.get("endtime")
        try:
            start_seconds = float(start) if start is not None else None
            end_seconds = float(end) if end is not None else None
        except ValueError:
            start_seconds = end_seconds = None
        if (start_seconds is None or end_seconds is None or
                not math.isfinite(start_seconds) or not math.isfinite(end_seconds) or
                start_seconds < 0 or end_seconds < start_seconds):
            continue  # Explicitly counted in inventory. No inferred timing.
        span = match.group()
        words.append({"word_id": word_id, "start_seconds": start_seconds,
                      "end_seconds": end_seconds, "text": node.text or "",
                      "byte_start": match.start(), "byte_end": match.end(),
                      "raw_span_sha256": digest(span)})
    if not words:
        raise ValueError(f"no timed original words: {member}")
    return tuple(words)


def attributed_turns(words: list[dict]) -> list[dict]:
    """Readable speaker-preserving view; exact evidence remains in `words`."""
    per_speaker = defaultdict(list)
    for word in words:
        per_speaker[word["speaker"]].append(word)
    turns = []
    for speaker, tokens in sorted(per_speaker.items()):
        tokens.sort(key=lambda x: (x["start_seconds"], x["end_seconds"], x["word_id"]))
        current = []
        for token in tokens:
            if current and (token["start_seconds"] - current[-1]["end_seconds"] > 1.5 or
                            len(current) >= 48 or
                            current[-1]["text"] in {".", "?", "!"}):
                turns.append(_turn(current))
                current = []
            current.append(token)
        if current:
            turns.append(_turn(current))
    return sorted(turns, key=lambda x: (x["start_seconds"], x["speaker"],
                                      x["end_seconds"], x["word_ids"][0]))


def _turn(words: list[dict]) -> dict:
    first = words[0]
    text = " ".join(word["text"] for word in words)
    text = re.sub(r"\s+([,.?!:;])", r"\1", text)
    return {"speaker": first["speaker"],
            "participant_id": first["participant_id"], "role": first["role"],
            "start_seconds": min(w["start_seconds"] for w in words),
            "end_seconds": max(w["end_seconds"] for w in words),
            "word_ids": [w["word_id"] for w in words], "text": text}


def _meetings(archive: zipfile.ZipFile, sessions: tuple[str, ...]) -> tuple[dict, ...]:
    raw = _member(archive, "corpusResources/meetings.xml")
    if digest(raw) != MEETINGS_SHA256:
        raise ValueError("official meeting metadata differs from pinned bytes")
    all_meetings = {}
    for meeting in ET.fromstring(raw).findall("meeting"):
        name = meeting.get("observation", "")
        if MEETING_ID.fullmatch(name):
            if name in all_meetings:
                raise ValueError(f"duplicate meeting ID: {name}")
            all_meetings[name] = meeting
    result = []
    for session in sessions:
        for stage in "abcd":
            meeting_id = session + stage
            meeting = all_meetings.get(meeting_id)
            if meeting is None or meeting.get("type") != "scenario":
                raise ValueError(f"incomplete or non-scenario meeting: {meeting_id}")
            speakers = {}
            for speaker in meeting.findall("speaker"):
                agent = speaker.get("nxt_agent")
                if not agent or agent in speakers or not speaker.get("global_name"):
                    raise ValueError(f"missing/duplicate speaker: {meeting_id}")
                speakers[agent] = {"participant_id": speaker.get("global_name"),
                                   "role": speaker.get("role")}
            if len(speakers) != 4:
                raise ValueError(f"expected four speakers: {meeting_id}")
            duration = float(meeting.get("duration", "nan"))
            if not math.isfinite(duration) or duration < 180:
                raise ValueError(f"invalid meeting duration: {meeting_id}")
            result.append({"meeting_id": meeting_id, "session_id": session,
                           "stage": stage, "date_raw": meeting.get("dateOnly"),
                           "start_time_raw": meeting.get("startTime"),
                           "duration_seconds": duration, "speakers": speakers})
    return tuple(result)


def build(archive_path: Path, *, sessions: tuple[str, ...] = SESSION_IDS) -> tuple[bytes, bytes]:
    if (not sessions or len(set(sessions)) != len(sessions) or
            any(not re.fullmatch(r"(?:ES|IS|TS)\d{4}", x) for x in sessions)):
        raise ValueError("invalid or duplicated session")
    raw_archive = archive_path.read_bytes()
    if digest(raw_archive) != ARCHIVE_SHA256:
        raise ValueError("official AMI manual v1.6.2 archive SHA-256 differs")
    records = []
    windows = []
    with zipfile.ZipFile(archive_path) as archive:
        raw_license = _member(archive, "LICENCE.txt")
        if digest(raw_license) != LICENSE_SHA256 or b"CC BY\n4.0" not in raw_license:
            raise ValueError("archive license differs from reviewed CC BY 4.0 bytes")
        for meeting in _meetings(archive, sessions):
            meeting_id = meeting["meeting_id"]
            duration = meeting["duration_seconds"]
            sources = []
            all_words = []
            total_words = untimed = 0
            for agent, person in sorted(meeting["speakers"].items()):
                member = f"words/{meeting_id}.{agent}.words.xml"
                original = _member(archive, member)
                parsed = original_words(original, member=member)
                word_nodes = ET.fromstring(original).findall("w")
                total_words += len(word_nodes)
                untimed += len(word_nodes) - len(parsed)
                sources.append({"member": member, "sha256": digest(original),
                                "bytes": len(original), "timed_words": len(parsed),
                                "untimed_words": len(word_nodes) - len(parsed),
                                "participant_id": person["participant_id"],
                                "role": person["role"]})
                all_words.extend({**word, "speaker": agent,
                                  "participant_id": person["participant_id"],
                                  "role": person["role"], "member": member,
                                  "member_sha256": digest(original)} for word in parsed)
            records.append({**meeting, "source_members": sources,
                            "word_elements": total_words, "untimed_word_elements": untimed})
            ordered_words = sorted(all_words, key=lambda x: (
                x["start_seconds"], x["end_seconds"], x["speaker"], x["word_id"]))
            for fraction in (0.2, 0.5, 0.8):
                # Some AMI meetings have shorter word annotation than their
                # metadata duration. Choose a retrospective word-quantile
                # center, but expose only the bounded as-of span itself.
                center = ordered_words[math.floor((len(ordered_words) - 1) * fraction)]
                start = max(0, math.floor(center["start_seconds"] - 60))
                end = min(start + 120, duration)
                words = [w for w in ordered_words if start <= w["start_seconds"]
                         and w["end_seconds"] <= end]
                if not words:
                    raise ValueError(f"empty timed source window: {meeting_id}/{fraction}")
                turns = attributed_turns(words)
                rendered = "\n".join(
                    f'{turn["speaker"]} ({turn["participant_id"]}, {turn["role"]}) '
                    f'[{turn["start_seconds"]:.2f}-{turn["end_seconds"]:.2f}]: '
                    f'{turn["text"]}' for turn in turns)
                windows.append({"schema": "mfm-ami-original-word-window-v1",
                                "status": "source_candidate_unreviewed",
                                "training_admitted": False, "formation_targets": None,
                                "archive_sha256": ARCHIVE_SHA256,
                                "session_id": meeting["session_id"],
                                "meeting_id": meeting_id, "stage": meeting["stage"],
                                "meeting_date_raw": meeting["date_raw"],
                                "meeting_start_time_raw": meeting["start_time_raw"],
                                "as_of": {"meeting_stage": meeting["stage"],
                                          "window_end_seconds": end},
                                "window": {"start_seconds": start,
                                           "end_seconds": end,
                                           "anchor_fraction": fraction},
                                "words": words, "derived_turns": turns,
                                "derived_display": rendered,
                                "derived_display_sha256": digest(rendered.encode("utf-8")),
                                "derived_display_note": (
                                    "Punctuation and speaker-turn segmentation derived from "
                                    "the pinned original word XML; overlapping speakers are "
                                    "represented by attributed turns ordered by start time")})
    inventory = {"schema": "mfm-ami-source-inventory-v1",
                 "status": "source_candidate_unreviewed",
                 "training_admitted": False, "formation_targets": None,
                 "archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
                 "archive_bytes": len(raw_archive),
                 "license": "CC BY 4.0", "license_member": "LICENCE.txt",
                 "license_sha256": LICENSE_SHA256,
                 "metadata_member": "corpusResources/meetings.xml",
                 "metadata_sha256": MEETINGS_SHA256,
                 "generator_family": "ami-manual-annotations-v1.6.2",
                 "session_ids": list(sessions), "meeting_count": len(records),
                 "window_count": len(windows), "meetings": records}
    return canonical(inventory) + b"\n", b"".join(canonical(w) + b"\n" for w in windows)


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"existing output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("source payloads belong outside the repository")
    inventory, windows = build(args.archive)
    _write_new(args.output_dir / "ami-v16-source-inventory.json", inventory)
    _write_new(args.output_dir / "ami-v16-word-windows.jsonl", windows)
    print(json.dumps({"schema": "mfm-ami-source-acquisition-receipt-v1",
                      "inventory_sha256": digest(inventory),
                      "windows_sha256": digest(windows),
                      "meeting_count": 24, "window_count": windows.count(b"\n"),
                      "training_admitted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
