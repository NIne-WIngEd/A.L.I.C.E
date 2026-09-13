from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OWNER_AUTHORIZED_ORIGINS = {"owner_authorized_service_teacher"}
ALLOWED_SPLITS = {"train", "dev", "test"}
REQUIRED_ROW_FIELDS = {
    "id",
    "competency",
    "split",
    "task",
    "prompt",
    "candidates",
    "preferred_indices",
    "source",
}


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


def validate_curriculum_rows(curriculum_path: str | Path) -> dict[str, Any]:
    path = Path(curriculum_path)
    ids: set[str] = set()
    split_counts = {split: 0 for split in sorted(ALLOWED_SPLITS)}
    competency_counts: dict[str, int] = {}
    rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on curriculum line {line_number}") from exc

            missing = REQUIRED_ROW_FIELDS.difference(row)
            if missing:
                raise ValueError(
                    f"curriculum line {line_number} missing fields: {sorted(missing)}"
                )

            row_id = str(row["id"]).strip()
            if not row_id:
                raise ValueError(f"curriculum line {line_number} has empty id")
            if row_id in ids:
                raise ValueError(f"duplicate curriculum id: {row_id}")
            ids.add(row_id)

            split = str(row["split"])
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"curriculum row {row_id} has unsupported split={split!r}")
            split_counts[split] += 1

            if str(row["task"]) != "candidate_ranking":
                raise ValueError(
                    f"curriculum row {row_id} task must be 'candidate_ranking' for the N0 ranker"
                )

            competency = str(row["competency"]).strip()
            if not competency:
                raise ValueError(f"curriculum row {row_id} has empty competency")
            competency_counts[competency] = competency_counts.get(competency, 0) + 1

            if not str(row["prompt"]).strip():
                raise ValueError(f"curriculum row {row_id} has empty prompt")
            if not str(row["source"]).strip():
                raise ValueError(f"curriculum row {row_id} has empty source")

            candidates = row["candidates"]
            preferred = row["preferred_indices"]
            if not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(f"curriculum row {row_id} must have at least two candidates")
            if any(not str(candidate).strip() for candidate in candidates):
                raise ValueError(f"curriculum row {row_id} contains an empty candidate")
            if not isinstance(preferred, list) or not preferred:
                raise ValueError(f"curriculum row {row_id} must have preferred_indices")

            try:
                preferred_ints = [int(index) for index in preferred]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"curriculum row {row_id} preferred_indices must be integers"
                ) from exc

            if len(set(preferred_ints)) != len(preferred_ints):
                raise ValueError(f"curriculum row {row_id} repeats a preferred index")
            if min(preferred_ints) < 0 or max(preferred_ints) >= len(candidates):
                raise ValueError(f"curriculum row {row_id} preferred index out of range")

            rows += 1

    if rows == 0:
        raise ValueError("curriculum contains no rows")
    if split_counts["train"] == 0:
        raise ValueError("curriculum contains no train rows")
    if split_counts["dev"] == 0:
        raise ValueError("curriculum contains no dev rows")

    return {
        "row_count": rows,
        "split_counts": split_counts,
        "competency_counts": dict(sorted(competency_counts.items())),
        "unique_ids": len(ids),
    }


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
