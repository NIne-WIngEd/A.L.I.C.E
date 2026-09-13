from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OWNER_AUTHORIZED_ORIGINS = {"owner_authorized_service_teacher"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: str | Path) -> str:
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


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

    origin = str(manifest.get("origin_type", "unknown"))
    if origin in OWNER_AUTHORIZED_ORIGINS:
        if manifest.get("owner_authorization_asserted") is not True:
            raise ValueError("owner-authorized teacher curriculum requires explicit owner authorization")
        if not manifest.get("actor"):
            raise ValueError("owner-authorized teacher curriculum must record actor")
        if not manifest.get("authorization_scope"):
            raise ValueError("owner-authorized teacher curriculum must record authorization_scope")

    expected_sha256 = str(manifest.get("curriculum_sha256", ""))
    expected_git_sha1 = str(manifest.get("curriculum_git_blob_sha1", ""))
    if expected_sha256:
        actual_sha256 = sha256_file(curriculum_path)
        if expected_sha256 != actual_sha256:
            raise ValueError(
                f"curriculum SHA256 mismatch: manifest={expected_sha256!r} actual={actual_sha256!r}"
            )
    elif expected_git_sha1:
        actual_git_sha1 = git_blob_sha1(curriculum_path)
        if expected_git_sha1 != actual_git_sha1:
            raise ValueError(
                f"curriculum git blob SHA1 mismatch: manifest={expected_git_sha1!r} actual={actual_git_sha1!r}"
            )
    else:
        raise ValueError("curriculum manifest must record curriculum_sha256 or curriculum_git_blob_sha1")

    provenance = manifest.get("provenance_or_authorization")
    if not provenance:
        raise ValueError("curriculum manifest must record provenance_or_authorization")

    return manifest
