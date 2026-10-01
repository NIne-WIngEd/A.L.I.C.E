#!/usr/bin/env python3
"""Replay frozen, unadmitted AMI cross-stage teacher cases against the original release.

This is evidence of exact byte ancestry and converter execution. It does not
issue source rights, authenticate participant consent, review semantic truth,
or qualify a model. Teacher-visible data remains in private custody.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cognitive_kernel.formation_dataset_admission import _unique_json
from scripts.mfm.prepare_v16_ami_longitudinal_teacher_case import _opened
from scripts.mfm.prepare_v16_observed_history_teacher_case import (
    _digest, _raw, convert as convert_base, rendered_prompt, source_view,
)

ARCHIVE_SHA = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
LICENSE_SHA = "d52fd9c7c19eec9f0bec3107554c10937203e261259d73b01637a61501fec7f7"
WINDOWS_SHA = "f414869926088e241f694db00f20862609e6992933e8713fda4a1733f659311d"
CONFIG = ROOT / "configs/mfm/v16_ami_cross_stage_teacher_expansion_20261002.json"


def _read(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if _digest(raw) != expected:
        raise ValueError(f"pinned file differs: {path}")
    return raw


def audit(*, bundle_root: Path, archive: Path, windows: Path,
          config: Path = CONFIG) -> dict:
    config_raw = config.read_bytes()
    spec = _unique_json(config_raw, "public teacher expansion config")
    if spec.get("schema") != "mfm-v16-ami-public-teacher-expansion-v1" or \
            spec.get("status") != "unadmitted-teacher-candidates" or \
            len(spec.get("cases", [])) < 2:
        raise ValueError("public teacher expansion config is incomplete")
    _read(archive, ARCHIVE_SHA)
    with ZipFile(archive) as z:
        if _digest(z.read("LICENCE.txt")) != LICENSE_SHA:
            raise ValueError("embedded AMI manual v1.6.2 license differs")
    source_rows: dict[str, dict] = {}
    hasher = sha256()
    with windows.open("rb") as stream:
        for line in stream:
            hasher.update(line)
            row = _unique_json(line, "AMI original window")
            key = f"{row['meeting_id']}@{row['as_of']['window_end_seconds']}"
            if key in source_rows:
                raise ValueError("duplicate AMI meeting/cutoff window")
            source_rows[key] = {"raw": line, "sha256": _digest(line)}
    if hasher.hexdigest() != WINDOWS_SHA:
        raise ValueError("AMI source-only windows differ from acquisition pin")
    seen_cases, seen_sessions = set(), set()
    for record in spec["cases"]:
        case_id = record["case_id"]
        if case_id in seen_cases or not case_id.startswith("ami-"):
            raise ValueError("duplicate or non-AMI case")
        seen_cases.add(case_id)
        directory = (bundle_root / case_id).resolve()
        if not directory.is_relative_to(bundle_root.resolve()):
            raise ValueError("case escapes private bundle root")
        envelope = _unique_json(_read(directory / "envelope.json", record["envelope_sha256"]),
                                "envelope")
        if envelope["case_id"] != case_id or envelope["split"] != "train" or \
                envelope["training_admitted"] is not False or \
                envelope["rights_status"] != "unverified" or \
                envelope["source_family"] != "ami-manual-2017":
            raise ValueError("public teacher case misstates source permissions or split")
        manifest = _unique_json(_read(directory / "source-window-manifest.json",
                                     envelope["source_window"]["sha256"]),
                                "source window manifest")
        if manifest["session_id"] in seen_sessions:
            raise ValueError("duplicate AMI session among expansion cases")
        seen_sessions.add(manifest["session_id"])
        for item in manifest["windows"]:
            raw = _read(directory / item["path"], item["sha256"])
            key = f"{item['meeting_id']}@{_unique_json(raw, 'window')['as_of']['window_end_seconds']}"
            if key not in source_rows or source_rows[key] != {"raw": raw, "sha256": item["sha256"]}:
                raise ValueError("case window differs from pinned source-only export")
        opened = _opened(envelope, directory / "envelope.json")
        view = _raw(source_view(envelope, opened))
        if _read(directory / "view.json", record["view_sha256"]) != view:
            raise ValueError("teacher view differs from opened source")
        prompt = rendered_prompt(view)
        if _read(directory / "prompt.txt", record["prompt_sha256"]) != prompt:
            raise ValueError("rendered teacher prompt differs")
        response = _unique_json(_read(directory / "teacher-response.json",
                                     record["response_sha256"]), "teacher response")
        replay_case, replay_provenance = convert_base(
            envelope, opened, response, envelope_sha256=record["envelope_sha256"],
            response_sha256=record["response_sha256"],
            prompt_sha256=record["prompt_sha256"])
        if _read(directory / "teacher-candidate.json", record["case_sha256"]) != _raw(replay_case):
            raise ValueError("candidate differs from replayed teacher response")
        provenance = _unique_json(_read(directory / "teacher-provenance.json",
                                       record["provenance_sha256"]), "teacher provenance")
        for field, expected in replay_provenance.items():
            if field in {"schema", "converter_sha256"}:
                continue
            if provenance.get(field) != expected:
                raise ValueError(f"teacher provenance differs: {field}")
        if (provenance.get("schema") != "mfm-v16-ami-cross-stage-teacher-provenance-v1" or
                provenance.get("base_converter_sha256") != replay_provenance["converter_sha256"] or
                provenance.get("converter_sha256") != _digest((ROOT / "scripts/mfm/prepare_v16_ami_longitudinal_teacher_case.py").read_bytes()) or
                provenance.get("training_admitted") is not False or
                provenance.get("independent_review") is not False or
                provenance.get("semantic_entailment_reviewed_independently") is not False or
                provenance.get("rights_status") != "unverified" or
                provenance.get("original_turn_sha256s") != {
                    item["ref_id"]: _digest(raw.split(b"source_text:\n", 1)[1])
                    for item, raw in opened}):
            raise ValueError("public teacher provenance claims unsupported review or rights")
    return {"schema": "mfm-v16-ami-public-teacher-audit-v1", "config_sha256": _digest(config_raw),
            "archive_sha256": ARCHIVE_SHA, "embedded_license_sha256": LICENSE_SHA,
            "source_windows_sha256": WINDOWS_SHA, "case_count": len(seen_cases),
            "source_family": "ami-manual-2017", "meeting_sessions": sorted(seen_sessions),
            "split": "train", "formation_training_admitted": False,
            "independent_target_review": False, "noncopyright_rights": "not_asserted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(audit(bundle_root=args.bundle_root, archive=args.archive,
                           windows=args.windows, config=args.config), sort_keys=True))


if __name__ == "__main__":
    main()
