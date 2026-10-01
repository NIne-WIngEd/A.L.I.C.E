#!/usr/bin/env python3
"""Make one as-of, cross-meeting AMI teacher candidate with visible chronology.

Original annotation windows are pinned individually. The model-visible text
is a deterministic wrapper around exact derived turn text, with meeting stage,
speaker and relative time. Quotes must remain within those exact source turns.
No source permission, reviewer judgment or memory write is conferred here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cognitive_kernel.formation_dataset_admission import _unique_json
from scripts.mfm.prepare_v16_annotation_drafts import write_private_new
from scripts.mfm import prepare_v16_observed_history_teacher_case as base


ENVELOPE_SCHEMA = "mfm-v16-ami-cross-stage-envelope-v1"
WINDOW_MANIFEST_SCHEMA = "mfm-v16-ami-cross-stage-window-manifest-v1"
STAGES = "abcd"


def _read_window(path: Path, expected: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    if base._digest(raw) != expected:
        raise ValueError("original AMI source window differs from external pin")
    row = _unique_json(raw, "AMI source window")
    if (row.get("schema") != "mfm-ami-original-word-window-v1" or
            row.get("status") != "source_candidate_unreviewed" or
            row.get("training_admitted") is not False or
            not isinstance(row.get("derived_turns"), list)):
        raise ValueError("source window is not an unadmitted AMI turn view")
    if row.get("meeting_id") != row.get("session_id", "") + row.get("stage", "") or \
            row.get("stage") not in STAGES:
        raise ValueError("AMI meeting does not follow one a-d session stage")
    return raw, row


def _wrapper(row: dict, turn: dict, index: int, *, latest: dict) -> bytes:
    text = turn["text"].encode("utf-8")
    lines = (
        "AMI DERIVED TRANSCRIPT; original word/XML pointers are in pinned source window",
        f"session: {row['session_id']}",
        f"meeting: {row['meeting_id']}",
        f"stage: {row['stage']}",
        f"as_of: {latest['meeting_id']}@{latest['as_of']['window_end_seconds']}s",
        f"speaker: {turn['participant_id']}",
        f"turn: {index:03d}",
        f"relative_seconds: {turn['start_seconds']}-{turn['end_seconds']}",
        f"original_turn_sha256: {sha256(text).hexdigest()}",
        "source_text:",
    )
    return ("\n".join(lines) + "\n").encode("utf-8") + text


def _read_manifest(envelope: dict, envelope_path: Path) -> tuple[dict, list[dict]]:
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        raise ValueError("cross-stage envelope has wrong schema")
    manifest_ref = envelope["source_window"]
    base._exact(manifest_ref, {"path", "sha256"}, "window manifest ref")
    manifest_path = (envelope_path.parent / manifest_ref["path"]).resolve()
    if not manifest_path.is_relative_to(envelope_path.parent.resolve()):
        raise ValueError("window manifest escapes private case directory")
    manifest = _unique_json(manifest_path.read_bytes(), "window manifest")
    if base._digest(manifest_path.read_bytes()) != manifest_ref["sha256"] or \
            manifest.get("schema") != WINDOW_MANIFEST_SCHEMA or \
            len(manifest.get("windows", [])) < 2:
        raise ValueError("cross-stage original window manifest differs")
    originals = []
    for ref in manifest["windows"]:
        base._exact(ref, {"path", "sha256", "meeting_id", "stage"}, "window ref")
        path = (envelope_path.parent / ref["path"]).resolve()
        if not path.is_relative_to(envelope_path.parent.resolve()):
            raise ValueError("original window escapes private case directory")
        _, row = _read_window(path, ref["sha256"])
        if ref["meeting_id"] != row["meeting_id"] or ref["stage"] != row["stage"]:
            raise ValueError("window manifest stage/meeting differs from original")
        originals.append(row)
    session = originals[0]["session_id"]
    if (any(row["session_id"] != session for row in originals) or
            len({row["meeting_id"] for row in originals}) != len(originals) or
            [STAGES.index(row["stage"]) for row in originals] != sorted(
                {STAGES.index(row["stage"]) for row in originals}) or
            len({row["archive_sha256"] for row in originals}) != 1):
        raise ValueError("window ancestry mixes teams, archive or stage order")
    latest = originals[-1]
    expected_as_of = {"session_ref": f"ami-{session.lower()}",
                      "meeting_id": latest["meeting_id"],
                      "stage": latest["stage"],
                      "cutoff_seconds": latest["as_of"]["window_end_seconds"]}
    if manifest.get("session_id") != session or \
            manifest.get("as_of") != expected_as_of or \
            envelope.get("as_of") != expected_as_of or \
            envelope.get("source_family") != "ami-manual-2017":
        raise ValueError("source selection includes later or foreign meeting stage")
    return manifest, originals


def _opened(envelope: dict, envelope_path: Path) -> list[tuple[dict, bytes]]:
    _, originals = _read_manifest(envelope, envelope_path)
    by_meeting = {row["meeting_id"].lower(): row for row in originals}
    latest = originals[-1]
    opened = []
    seen = set()
    last_position = (-1, -1.0)
    for item in envelope["sources"]:
        m = re.fullmatch(r"ami-([a-z0-9]+[a-d])-turn-([0-9]{3})", item["ref_id"])
        if not m or item["ref_id"] in seen or m.group(1) not in by_meeting:
            raise ValueError("source turn is absent, duplicated or from future meeting")
        seen.add(item["ref_id"])
        row = by_meeting[m.group(1)]
        index = int(m.group(2))
        if index >= len(row["derived_turns"]):
            raise ValueError("source turn exceeds selected original window")
        turn = row["derived_turns"][index]
        position = (STAGES.index(row["stage"]), turn["start_seconds"])
        if (position < last_position or
                turn["end_seconds"] > row["as_of"]["window_end_seconds"]):
            raise ValueError("source turn is out of stage order or beyond as-of cutoff")
        last_position = position
        if not turn["word_ids"] or not set(turn["word_ids"]).issubset(
                {w["word_id"] for w in row["words"]}):
            raise ValueError("derived turn has no exact original word ancestry")
        source_path = (envelope_path.parent / item["path"]).resolve()
        if not source_path.is_relative_to(envelope_path.parent.resolve()):
            raise ValueError("source payload escapes private case directory")
        raw = source_path.read_bytes()
        expected = _wrapper(row, turn, index, latest=latest)
        if (raw != expected or base._digest(raw) != item["sha256"] or
                item["role"] != "outside_source" or item["modality"] != "text" or
                item["observed_at"] is not None or item["recorded_at"] is not None or
                item["speaker_ref"] != f"ami-{turn['participant_id'].lower()}" or
                item["subject_ref"] != item["speaker_ref"] or
                item["source_item_ref"] != item["ref_id"] + "-derived-transcript" or
                item["relative_time"] != {
                    "timeline_ref": f"ami-{row['meeting_id'].lower()}",
                    "end_seconds": turn["end_seconds"]}):
            raise ValueError("model-visible stage/speaker/text differs from original turn")
        opened.append((item, raw))
    if not opened or len(opened) > 32 or \
            not {row["meeting_id"].lower() for row in originals}.issubset(
                {re.fullmatch(r"ami-([a-z0-9]+[a-d])-turn-[0-9]{3}", item["ref_id"]).group(1)
                 for item, _ in opened}):
        raise ValueError("cross-stage case must include each selected stage")
    return opened


def prepare(*, windows: tuple[tuple[Path, str], ...], selection: tuple[str, ...],
            case_id: str, authorization_id: str, output_dir: Path) -> dict:
    if output_dir.exists() or len(windows) < 2:
        raise ValueError("new private directory and multiple original windows required")
    rows = [_read_window(path, pin) for path, pin in windows]
    session = rows[0][1]["session_id"]
    stages = [STAGES.index(row["stage"]) for _, row in rows]
    if (any(row["session_id"] != session for _, row in rows) or
            len(set(stages)) != len(stages) or stages != sorted(stages) or
            len({row["archive_sha256"] for _, row in rows}) != 1):
        raise ValueError("selected original AMI windows do not form one ordered team history")
    output_dir.mkdir(mode=0o700, parents=False)
    if stat.S_IMODE(output_dir.stat().st_mode) & 0o077:
        raise ValueError("source custody directory must be 0700")
    refs = []
    by_meeting = {}
    for raw_index, ((path, pin), (raw, row)) in enumerate(zip(windows, rows, strict=True)):
        name = f"source-window-{raw_index:02d}.json"
        write_private_new(output_dir / name, raw)
        refs.append({"path": name, "sha256": pin,
                     "meeting_id": row["meeting_id"], "stage": row["stage"]})
        by_meeting[row["meeting_id"].lower()] = row
    latest = rows[-1][1]
    as_of = {"session_ref": f"ami-{session.lower()}",
             "meeting_id": latest["meeting_id"], "stage": latest["stage"],
             "cutoff_seconds": latest["as_of"]["window_end_seconds"]}
    manifest = {"schema": WINDOW_MANIFEST_SCHEMA,
                "session_id": session, "as_of": as_of, "windows": refs}
    manifest_raw = base._raw(manifest)
    write_private_new(output_dir / "source-window-manifest.json", manifest_raw)
    selected = []
    for identifier in selection:
        if not re.fullmatch(r"[a-z0-9]+[a-d]:[0-9]{1,3}", identifier):
            raise ValueError("selection must be meeting:turn-index")
        meeting, turn_index = identifier.split(":")
        if meeting not in by_meeting or int(turn_index) >= len(
                by_meeting[meeting]["derived_turns"]):
            raise ValueError("selected turn lies outside pinned source windows")
        row = by_meeting[meeting]
        index = int(turn_index)
        turn = row["derived_turns"][index]
        if turn["end_seconds"] > row["as_of"]["window_end_seconds"]:
            raise ValueError("selected turn lies beyond original window as-of cutoff")
        ref = f"ami-{meeting}-turn-{index:03d}"
        raw = _wrapper(row, turn, index, latest=latest)
        name = f"source-{meeting}-{index:03d}.txt"
        write_private_new(output_dir / name, raw)
        selected.append({"ref_id": ref, "role": "outside_source",
                         "modality": "text", "subject_ref": f"ami-{turn['participant_id'].lower()}",
                         "speaker_ref": f"ami-{turn['participant_id'].lower()}",
                         "source_item_ref": ref + "-derived-transcript",
                         "observed_at": None, "recorded_at": None,
                         "temporal_granularity": "instant",
                         "minimum_sensitivity": "public", "path": name,
                         "sha256": base._digest(raw),
                         "relative_time": {"timeline_ref": f"ami-{meeting}",
                                           "end_seconds": turn["end_seconds"]}})
    envelope = {"schema": ENVELOPE_SCHEMA, "case_id": case_id,
                "split": "train", "authorization_id": authorization_id,
                "host_id": "ami-source-observer", "self_id": "alice-assistant-self",
                "authority_id": "ami-public-history-authority",
                "source_family": "ami-manual-2017", "as_of": as_of,
                "source_window": {"path": "source-window-manifest.json",
                                  "sha256": base._digest(manifest_raw)},
                "sources": selected, "targets": [],
                "training_admitted": False, "rights_status": "unverified"}
    envelope_raw = base._raw(envelope)
    write_private_new(output_dir / "envelope.json", envelope_raw)
    _opened(envelope, output_dir / "envelope.json")
    view_raw = base._raw(base.source_view(envelope, [(i, (output_dir / i["path"]).read_bytes())
                                                     for i in selected]))
    write_private_new(output_dir / "view.json", view_raw)
    prompt = base.rendered_prompt(view_raw)
    write_private_new(output_dir / "prompt.txt", prompt)
    return {"case_id": case_id, "envelope_sha256": base._digest(envelope_raw),
            "view_sha256": base._digest(view_raw), "prompt_sha256": base._digest(prompt),
            "source_window_manifest_sha256": base._digest(manifest_raw),
            "training_admitted": False}


def convert(*, directory: Path, envelope_sha256: str,
            view_sha256: str, prompt_sha256: str,
            response_sha256: str) -> dict:
    envelope_path = directory / "envelope.json"
    envelope = base._load(envelope_path, envelope_sha256)
    opened = _opened(envelope, envelope_path)
    view = base._raw(base.source_view(envelope, opened))
    if (base._digest(view) != view_sha256 or
            (directory / "view.json").read_bytes() != view or
            base._digest(base.rendered_prompt(view)) != prompt_sha256 or
            (directory / "prompt.txt").read_bytes() != base.rendered_prompt(view)):
        raise ValueError("teacher prompt/view changed from exact selected history")
    response = base._load(directory / "teacher-response.json", response_sha256)
    case, provenance = base.convert(
        envelope, opened, response, envelope_sha256=envelope_sha256,
        response_sha256=response_sha256, prompt_sha256=prompt_sha256)
    provenance["schema"] = "mfm-v16-ami-cross-stage-teacher-provenance-v1"
    provenance["base_converter_sha256"] = provenance["converter_sha256"]
    provenance["converter_sha256"] = base._digest(Path(__file__).read_bytes())
    provenance["original_turn_sha256s"] = {
        item["ref_id"]: base._digest(raw.split(b"source_text:\n", 1)[1])
        for item, raw in opened}
    base.write_private_new(directory / "teacher-candidate.json", base._raw(case))
    base.write_private_new(directory / "teacher-provenance.json", base._raw(provenance))
    return {"case_id": case["case_id"], "case_sha256": provenance["case_sha256"],
            "provenance_sha256": base._digest(base._raw(provenance)),
            "training_admitted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--window", action="append", type=Path, required=True)
    prep.add_argument("--window-sha256", action="append", required=True)
    prep.add_argument("--selection", action="append", required=True)
    prep.add_argument("--case-id", required=True)
    prep.add_argument("--authorization-id", required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    conv = commands.add_parser("convert")
    conv.add_argument("--directory", type=Path, required=True)
    for name in ("envelope", "view", "prompt", "response"):
        conv.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if len(args.window) != len(args.window_sha256):
            raise ValueError("one SHA-256 pin per original window required")
        result = prepare(windows=tuple(zip(args.window, args.window_sha256, strict=True)),
                         selection=tuple(args.selection), case_id=args.case_id,
                         authorization_id=args.authorization_id,
                         output_dir=args.output_dir)
    else:
        result = convert(directory=args.directory,
                         envelope_sha256=args.envelope_sha256,
                         view_sha256=args.view_sha256,
                         prompt_sha256=args.prompt_sha256,
                         response_sha256=args.response_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
