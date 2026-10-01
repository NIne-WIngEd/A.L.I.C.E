#!/usr/bin/env python3
"""Extract source-only, as-of ICSI transcript windows from the Edinburgh release.

No MFM target, review, rights attestation, split, or cross-meeting chronology is
created here. Output belongs in private custody outside the Git checkout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZipFile


URL = "https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip"
RIGHTS_URL = "https://groups.inf.ed.ac.uk/ami/icsi/license.shtml"
PINNED_SHA256 = "cf4860245b9ca9c9ed11a66e5a74cfea2a99a686a52f7580aa0a6bc2225895a9"
NITE_ID = "{http://nite.sourceforge.net/}id"
WORD = re.compile(rb"<w\b[^>]*>.*?</w>", re.DOTALL)
WINDOW_SCHEMA = "mfm-v16-icsi-source-window-v1"
RECEIPT_SCHEMA = "mfm-v16-icsi-source-acquisition-v1"
ROOT = Path(__file__).resolve().parents[2]


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def timed_words(member: str, raw: bytes) -> tuple[list[dict], Counter]:
    """Map every selected word to exact byte offsets in its original ZIP member."""
    root = ET.fromstring(raw)
    if root.tag != f"{NITE_ID.removesuffix('id')}root":
        raise ValueError(f"unexpected NXT root: {member}")
    elements = [element for element in root if element.tag == "w"]
    spans = list(WORD.finditer(raw))
    if len(elements) != len(spans):
        raise ValueError(f"cannot anchor each word to original XML bytes: {member}")
    member_sha = digest(raw)
    result = []
    skipped = Counter()
    ids = set()
    for element, match in zip(elements, spans, strict=True):
        # The isolated lexical element must parse to the same attributes/text.
        isolated = ET.fromstring(b'<root xmlns:nite="http://nite.sourceforge.net/">'
                                 + match.group() + b"</root>")[0]
        if element.attrib != isolated.attrib or element.text != isolated.text or len(element):
            raise ValueError(f"ambiguous XML word anchor: {member}")
        word_id = element.attrib.get(NITE_ID)
        if not word_id or word_id in ids:
            raise ValueError(f"duplicate/missing word ID: {member}")
        ids.add(word_id)
        try:
            start = Decimal(element.attrib["starttime"])
            end = Decimal(element.attrib["endtime"])
        except (KeyError, InvalidOperation):
            skipped["untimed"] += 1
            continue
        if not (start.is_finite() and end.is_finite() and 0 <= start <= end):
            skipped["invalid_time"] += 1
            continue
        text = element.text or ""
        if not text.strip():
            skipped["empty_text"] += 1
            continue
        result.append({
            "word_id": word_id, "start_seconds": str(start),
            "end_seconds": str(end), "text": text,
            "word_class": element.attrib.get("c"),
            "source": {"zip_member": member, "member_sha256": member_sha,
                       "byte_start": match.start(), "byte_end": match.end(),
                       "element_sha256": digest(match.group())},
        })
    return result, skipped


def participant_for_channel(raw: bytes, member: str) -> str | None:
    root = ET.fromstring(raw)
    found = {element.attrib["participant"] for element in root
             if element.tag == "dialogueact" and element.attrib.get("participant")}
    if len(found) > 1:
        raise ValueError(f"multiple participants in one channel: {member}")
    return next(iter(found), None)


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def acquire(zip_path: Path, output_dir: Path, *, window_seconds: int = 600,
            expected_sha256: str = PINNED_SHA256) -> dict:
    if window_seconds <= 0:
        raise ValueError("window duration must be positive")
    source = zip_path.read_bytes()
    if digest(source) != expected_sha256:
        raise ValueError("ICSI source archive differs from the pinned SHA-256")
    destination = output_dir.absolute()
    if destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("original ICSI material belongs outside Git")
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError("output must be a new directory below an existing private parent")
    with ZipFile(zip_path) as archive:
        metadata_name = "ICSI/ICSI-metadata.xml"
        licence_name = "ICSI/LICENCE.txt"
        names = set(archive.namelist())
        if metadata_name not in names or licence_name not in names:
            raise ValueError("core NXT metadata/licence absent")
        metadata = archive.read(metadata_name)
        licence = archive.read(licence_name)
        if b"Creative Commons Attribution 4.0" not in licence:
            raise ValueError("unrecognized core release licence")
        observations = ET.fromstring(metadata).find("observations")
        if observations is None:
            raise ValueError("ICSI observations absent")
        meetings = [e.attrib["name"] for e in observations]
        if len(meetings) != len(set(meetings)) or len(meetings) != 75:
            raise ValueError("unexpected ICSI v1.0 observation inventory")
        windows: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        skipped = Counter()
        members = []
        untimed_members = []
        mapped = 0
        for member in sorted(names):
            if not member.startswith("ICSI/Words/") or not member.endswith(".words.xml"):
                continue
            suffix = member.removeprefix("ICSI/Words/").removesuffix(".words.xml")
            meeting, sep, channel = suffix.partition(".")
            if meeting in meetings and suffix == f"{meeting}.":
                raw = archive.read(member)
                words, missed = timed_words(member, raw)
                if words:
                    raise ValueError(f"unnamed channel includes timed words: {member}")
                skipped.update(missed)
                untimed_members.append({"path": member, "sha256": digest(raw),
                                        "bytes": len(raw), "reason": "unnamed untimed channel"})
                continue
            if not sep or meeting not in meetings or not re.fullmatch(r"[A-J]", channel):
                raise ValueError(f"unknown transcript member: {member}")
            words_raw = archive.read(member)
            dialogue_name = f"ICSI/DialogueActs/{suffix}.dialogue-acts.xml"
            participant = None
            if dialogue_name in names:
                dialogue_raw = archive.read(dialogue_name)
                participant = participant_for_channel(dialogue_raw, dialogue_name)
                members.append({"path": dialogue_name, "sha256": digest(dialogue_raw),
                                "bytes": len(dialogue_raw)})
                if participant:
                    mapped += 1
            members.append({"path": member, "sha256": digest(words_raw),
                            "bytes": len(words_raw)})
            words, missed = timed_words(member, words_raw)
            skipped.update(missed)
            for word in words:
                word["channel"] = channel
                word["participant_id"] = participant  # Unknown stays unknown.
                # The word is visible only once its endtime has passed. The
                # half-open interval is a storage window, not a new time claim.
                end = Decimal(word["end_seconds"])
                index = max(0, (max(end, Decimal("0.000000001")) -
                                Decimal("0.000000001")) // window_seconds)
                windows[meeting][int(index)].append(word)

    rows = []
    for meeting in meetings:
        previous = None
        for index in sorted(windows[meeting]):
            end = (index + 1) * window_seconds
            observations = sorted(windows[meeting][index],
                                  key=lambda w: (Decimal(w["end_seconds"]),
                                                 Decimal(w["start_seconds"]),
                                                 w["channel"], w["word_id"]))
            if any(Decimal(w["end_seconds"]) > end for w in observations):
                raise ValueError("future word leaked into an earlier window")
            row_id = f"icsi-{meeting}-through-{end:06d}s"
            rows.append({"schema": WINDOW_SCHEMA, "status": "source_only_unadmitted",
                         "source_family": "ICSI-core-NXT-v1.0-Edinburgh",
                         "meeting_id": meeting, "window_id": row_id,
                         "previous_window_id": previous,
                         "interval_start_seconds": index * window_seconds,
                         "as_of_end_seconds": end,
                         "observations": observations})
            previous = row_id
    if not rows:
        raise ValueError("no timed transcript windows")
    raw_rows = b"".join(canonical(row) + b"\n" for row in rows)
    receipt = {"schema": RECEIPT_SCHEMA, "status": "source_only_unadmitted",
               "upstream_url": URL, "upstream_release": "ICSI core annotations v1.0 (2016-07-22)",
               "upstream_rights_url": RIGHTS_URL, "upstream_license": "CC BY 4.0",
               "archive_bytes": len(source), "archive_sha256": digest(source),
               "metadata_sha256": digest(metadata), "licence_sha256": digest(licence),
               "members": sorted(members, key=lambda m: m["path"]),
               "excluded_untimed_members": untimed_members,
               "meeting_ids": meetings, "window_seconds": window_seconds,
               "window_count": len(rows), "word_count": sum(len(row["observations"]) for row in rows),
               "mapped_channel_count": mapped, "skipped_word_count_by_reason": dict(skipped),
               "windows_sha256": digest(raw_rows),
               "scope": "Original timed orthographic words only. No audio, contributed annotations, "
                        "cross-meeting chronology, teacher answers, or steward rights attestation."}
    temp = Path(tempfile.mkdtemp(prefix=".icsi-acquire-", dir=destination.parent))
    os.chmod(temp, 0o700)
    try:
        _write_private(temp / "source_windows.jsonl", raw_rows)
        _write_private(temp / "acquisition_receipt.json", canonical(receipt) + b"\n")
        temp.rename(destination)
    except BaseException:
        for child in temp.iterdir():
            child.unlink()
        temp.rmdir()
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=int, default=600)
    args = parser.parse_args()
    result = acquire(args.zip, args.output_dir, window_seconds=args.window_seconds)
    print(json.dumps({key: result[key] for key in ("archive_sha256", "windows_sha256",
                                                 "window_count", "word_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
