#!/usr/bin/env python3
"""Issue exact-source, project-side CC BY 4.0 receipts for selected AMI text.

The receipt records a copyright license interpretation, with attribution
conditions. It is not a publisher signature, participant-consent record,
independent review, or authorization for unrelated media or releases.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cognitive_kernel.formation_dataset_admission import _unique_json
from scripts.mfm.audit_v16_ami_teacher_expansion import (
    ARCHIVE_SHA, CONFIG, LICENSE_SHA, WINDOWS_SHA, _read, audit,
)
from scripts.mfm.prepare_v16_observed_history_teacher_case import _digest, _raw

PUBLISHER = "https://groups.inf.ed.ac.uk/ami/download/"
LICENSE = "https://creativecommons.org/licenses/by/4.0/legalcode.en"


def _receipt(source_id: str, source_raw: bytes, host_id: str, *, issued_at: str) -> dict:
    return {
        "schema": "mfm-source-rights-v1", "source_id": source_id,
        "source_sha256": _digest(source_raw), "host_family": host_id,
        "issuer_id": "alice-agent-under-owner-direction",
        "authority_ref": "ami-manual-v162-ccby4-publisher-release",
        "issuer_method": "project-side-publisher-license-verification",
        "issued_at_utc": issued_at,
        "formation_training": True, "formation_evaluation": True,
        "model_distribution": True, "revoked": False,
        "permission_scope": "AMI manual annotations v1.6.2 orthographic text: copyright and similar rights licensed under CC BY 4.0, subject to attribution, license link and indication of modifications. Model distribution field records this copyright permission only.",
        "publisher_release_url": PUBLISHER,
        "license_url": LICENSE,
        "archive_sha256": ARCHIVE_SHA,
        "embedded_license_sha256": LICENSE_SHA,
        "source_windows_sha256": WINDOWS_SHA,
        "source_text_derivation": "selected attributed turns, bounded by original as-of window; stage/speaker/time headers added and marked as derived",
        "unassessed_rights": ["participant privacy or publicity", "third-party content outside the licensed release"],
        "not_asserted": ["publisher-issued private permission", "participant consent", "independent semantic target review", "product qualification"],
    }


def issue(*, bundle_root: Path, archive: Path, windows: Path,
          output_dir: Path, config: Path = CONFIG) -> dict:
    result = audit(bundle_root=bundle_root, archive=archive, windows=windows, config=config)
    if output_dir.exists() or output_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("rights output must be a new private directory outside Git")
    spec = _unique_json(config.read_bytes(), "public teacher config")
    issued_at = datetime.now(timezone.utc).isoformat()
    records = []
    for case in spec["cases"]:
        directory = bundle_root / case["case_id"]
        envelope = _unique_json(_read(directory / "envelope.json", case["envelope_sha256"]),
                                "source envelope")
        for item in envelope["sources"]:
            raw = _read(directory / item["path"], item["sha256"])
            records.append(_receipt(item["ref_id"], raw, envelope["host_id"],
                                    issued_at=issued_at))
    if len({r["source_id"] for r in records}) != len(records):
        raise ValueError("source ID duplicated in rights issuance")
    output_dir.mkdir(mode=0o700)
    created = []
    try:
        for receipt in records:
            path = output_dir / (receipt["source_id"] + ".json")
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(_raw(receipt))
            created.append(path)
        manifest = {
            "schema": "mfm-v16-public-license-issuance-v1",
            "scope": "project-side CC BY 4.0 copyright evidence for exact AMI manual v1.6.2 source text; no participant consent or independent target gold asserted",
            "issued_at_utc": issued_at,
            "source_count": len(created),
            "receipts": [{"path": p.name, "sha256": _digest(p.read_bytes())} for p in created],
            "teacher_case_audit": result,
        }
        fd = os.open(output_dir / "issuance-manifest.json",
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_raw(manifest))
        return {"source_count": len(created),
                "issuance_manifest_sha256": _digest((output_dir / "issuance-manifest.json").read_bytes()),
                "teacher_cases_admitted": False}
    except BaseException:
        for path in created:
            path.unlink()
        (output_dir / "issuance-manifest.json").unlink(missing_ok=True)
        output_dir.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(issue(bundle_root=args.bundle_root, archive=args.archive,
                           windows=args.windows, output_dir=args.output_dir),sort_keys=True))


if __name__ == "__main__":
    main()
