from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROHIBITED_TRAINING_ORIGINS = {
    "openai_service_output",
    "unknown_hosted_model_output",
    "unreviewed_hosted_model_output",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_curriculum_manifest(
    curriculum_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    curriculum_path = Path(curriculum_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("training_authorized") is not True:
        raise ValueError("curriculum manifest does not authorize training")
    if manifest.get("private_identity_data") is True:
        raise ValueError("private identity data is not allowed in N0 curriculum")

    origin = str(manifest.get("origin_type", "unknown_hosted_model_output"))
    if origin in PROHIBITED_TRAINING_ORIGINS:
        raise ValueError(f"curriculum origin is prohibited for this training path: {origin}")

    expected_hash = str(manifest.get("curriculum_sha256", ""))
    actual_hash = sha256_file(curriculum_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"curriculum hash mismatch: manifest={expected_hash!r} actual={actual_hash!r}"
        )

    rights = manifest.get("rights_or_license")
    if not rights:
        raise ValueError("curriculum manifest must record rights_or_license")
    if not manifest.get("terms_or_license_reviewed"):
        raise ValueError("curriculum terms/license review must be explicit")

    return manifest
