from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parser_registry import ParserRegistry, load_registry


EXTRACTION_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_pilot(vault_root: Path, pilot_name: str) -> tuple[Path, dict[str, Any]]:
    snapshot = (vault_root / "raw" / pilot_name).resolve(strict=True)
    expected_parent = (vault_root / "raw").resolve(strict=True)
    if not is_relative_to(snapshot, expected_parent):
        raise ValueError("Pilot snapshot is outside the vault raw directory")
    if snapshot.is_symlink() or _is_reparse_point(snapshot):
        raise ValueError("Pilot snapshot may not be a link or reparse point")

    manifest_path = snapshot / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pilot_name") != pilot_name:
        raise ValueError("Pilot manifest name does not match requested pilot")
    if not isinstance(manifest.get("items"), list):
        raise ValueError("Pilot manifest items are missing")
    return snapshot, manifest


def _resolve_object(snapshot: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Unsafe object path in pilot manifest")
    path = (snapshot / relative_path).resolve(strict=True)
    if not is_relative_to(path, snapshot):
        raise ValueError("Pilot object escapes snapshot directory")
    if path.is_symlink() or _is_reparse_point(path):
        raise ValueError("Pilot objects may not be links or reparse points")
    if not path.is_file():
        raise ValueError("Pilot object is not a file")
    return path


def _existing_is_valid(
    metadata_path: Path,
    text_path: Path,
    *,
    source_sha256: str,
    registry_digest: str,
    parser_id: str,
) -> bool:
    if not metadata_path.is_file() or not text_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("status") == "success"
        and metadata.get("source_sha256") == source_sha256
        and metadata.get("registry_digest") == registry_digest
        and metadata.get("parser_id") == parser_id
        and metadata.get("text_sha256") == sha256_file(text_path)
    )


def extract_pilot(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    registry_path: Path | None = None,
    resume: bool = True,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    registry = load_registry(registry_path)
    snapshot, manifest = _load_pilot(vault_root, pilot_name)

    repo_root = Path(__file__).resolve().parents[2]
    worker = repo_root / "scripts" / "extraction_worker.py"
    if not worker.is_file():
        raise FileNotFoundError(f"Extraction worker not found: {worker}")

    output_root = vault_root / "derived" / pilot_name / "extracted"
    text_root = output_root / "text"
    metadata_root = output_root / "metadata"
    run_root = vault_root / "manifests" / "extractions" / pilot_name
    exports = vault_root / "manifests" / "exports"
    temporary_root = vault_root / "temporary"
    for path in (
        text_root,
        metadata_root,
        run_root,
        exports,
        temporary_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in manifest["items"]:
        key = (
            str(item["sha256"]),
            int(item["size_bytes"]),
            str(item["object_path"]),
        )
        groups[key].append(dict(item))

    run_id = str(uuid.uuid4())
    started_at = utc_now()
    result_records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    truncated_count = 0

    for index, ((source_hash, source_size, object_relative), items) in enumerate(
        groups.items(),
        start=1,
    ):
        representative = items[0]
        family = str(representative["family"]).strip().lower()
        family_counts[family] += 1
        record: dict[str, Any] = {
            "source_sha256": source_hash,
            "source_size_bytes": source_size,
            "object_path": object_relative,
            "family": family,
            "item_count": len(items),
            "file_ids": [str(item["file_id"]) for item in items],
        }

        try:
            source = _resolve_object(snapshot, object_relative)
            if source.stat().st_size != source_size:
                raise RuntimeError("Pilot object size mismatch")
            if sha256_file(source) != source_hash:
                raise RuntimeError("Pilot object hash mismatch")

            spec = registry.select(family, source)
            text_path = text_root / f"{source_hash}.txt"
            metadata_path = metadata_root / f"{source_hash}.json"

            if resume and _existing_is_valid(
                metadata_path,
                text_path,
                source_sha256=source_hash,
                registry_digest=registry.digest,
                parser_id=spec.parser_id,
            ):
                existing = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                record.update(existing)
                record["status"] = "resumed"
                status_counts["resumed"] += 1
                if existing.get("truncated"):
                    truncated_count += 1
                result_records.append(record)
                continue

            with tempfile.TemporaryDirectory(
                prefix=f"alice-extract-{source_hash[:12]}-",
                dir=temporary_root,
            ) as temp:
                temp_root = Path(temp)
                request_path = temp_root / "request.json"
                response_path = temp_root / "response.json"
                temp_text_path = temp_root / "output.txt"
                request = {
                    "source_path": str(source),
                    "output_text_path": str(temp_text_path),
                    "expected_size": source_size,
                    "expected_sha256": source_hash,
                    "parser_spec": spec.to_dict(),
                }
                request_path.write_text(
                    json.dumps(request, indent=2),
                    encoding="utf-8",
                )

                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(repo_root / "src")
                environment["PYTHONNOUSERSITE"] = "1"
                environment["HTTP_PROXY"] = "http://127.0.0.1:9"
                environment["HTTPS_PROXY"] = "http://127.0.0.1:9"
                environment["ALL_PROXY"] = "http://127.0.0.1:9"
                environment["NO_PROXY"] = ""

                try:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            str(worker),
                            "--request",
                            str(request_path),
                            "--response",
                            str(response_path),
                        ],
                        cwd=temp_root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=spec.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"Parser timed out after {spec.timeout_seconds}s"
                    ) from exc

                if response_path.is_file():
                    response = json.loads(
                        response_path.read_text(encoding="utf-8")
                    )
                else:
                    response = {
                        "status": "error",
                        "error": "Worker produced no response",
                    }

                if completed.returncode != 0 or response.get("status") != "success":
                    stderr = (completed.stderr or "").strip()[-4000:]
                    error = str(response.get("error", "Worker failed"))
                    if stderr:
                        error = f"{error}; stderr={stderr}"
                    raise RuntimeError(error)

                if not temp_text_path.is_file():
                    raise RuntimeError("Worker produced no text output")
                if response["text_sha256"] != sha256_file(temp_text_path):
                    raise RuntimeError("Worker text hash mismatch")

                final_temp_text = text_path.with_name(
                    f".{text_path.name}.{run_id}.tmp"
                )
                shutil.copy2(temp_text_path, final_temp_text)
                os.replace(final_temp_text, text_path)
                try:
                    os.chmod(text_path, stat.S_IREAD)
                except OSError:
                    pass

                metadata = {
                    **response,
                    "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
                    "run_id": run_id,
                    "pilot_name": pilot_name,
                    "registry_id": registry.registry_id,
                    "registry_digest": registry.digest,
                    "parser_id": spec.parser_id,
                    "source_object_path": object_relative,
                    "text_path": str(text_path),
                    "extracted_at": utc_now(),
                    "source_modified": False,
                }
                _atomic_json(metadata_path, metadata)
                record.update(metadata)
                status_counts["success"] += 1
                if metadata.get("truncated"):
                    truncated_count += 1

        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            status_counts["error"] += 1

        result_records.append(record)
        print(
            f"Processed {index}/{len(groups)} unique pilot objects "
            f"({status_counts['success']} success, "
            f"{status_counts['resumed']} resumed, "
            f"{status_counts['error']} errors)"
        )

    index_path = run_root / f"extraction-index-{run_id}.json"
    run_manifest = {
        "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        "run_id": run_id,
        "pilot_name": pilot_name,
        "started_at": started_at,
        "completed_at": utc_now(),
        "registry_id": registry.registry_id,
        "registry_digest": registry.digest,
        "pilot_manifest_sha256": sha256_file(
            snapshot / "pilot-manifest.json"
        ),
        "records": result_records,
    }
    _atomic_json(index_path, run_manifest)

    summary = {
        "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        "run_id": run_id,
        "pilot_name": pilot_name,
        "approved_item_records": len(manifest["items"]),
        "unique_content_objects": len(groups),
        "successful_extractions": status_counts["success"],
        "resumed_extractions": status_counts["resumed"],
        "failed_extractions": status_counts["error"],
        "truncated_extractions": truncated_count,
        "family_counts": dict(family_counts),
        "registry_id": registry.registry_id,
        "registry_digest": registry.digest,
        "output_root": str(output_root),
        "run_manifest_path": str(index_path),
        "source_files_modified": False,
    }
    summary_path = (
        exports / f"pilot-extraction-summary-{run_id}.json"
    )
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)

    if fail_on_error and status_counts["error"]:
        raise RuntimeError(
            f"{status_counts['error']} pilot objects failed extraction"
        )
    return summary


def verify_pilot_extraction(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    registry_path: Path | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    registry = load_registry(registry_path)
    snapshot, manifest = _load_pilot(vault_root, pilot_name)
    output_root = vault_root / "derived" / pilot_name / "extracted"
    text_root = output_root / "text"
    metadata_root = output_root / "metadata"

    errors: list[str] = []
    unique_hashes = {
        str(item["sha256"]) for item in manifest["items"]
    }
    verified = 0
    truncated = 0

    for source_hash in sorted(unique_hashes):
        text_path = text_root / f"{source_hash}.txt"
        metadata_path = metadata_root / f"{source_hash}.json"
        if not text_path.is_file():
            errors.append(f"Missing text output for {source_hash}")
            continue
        if not metadata_path.is_file():
            errors.append(f"Missing metadata output for {source_hash}")
            continue

        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            errors.append(f"Invalid metadata JSON for {source_hash}")
            continue

        if metadata.get("status") != "success":
            errors.append(f"Non-success metadata for {source_hash}")
        if metadata.get("source_sha256") != source_hash:
            errors.append(f"Source hash mismatch in metadata for {source_hash}")
        if metadata.get("registry_digest") != registry.digest:
            errors.append(f"Registry digest mismatch for {source_hash}")
        if metadata.get("text_sha256") != sha256_file(text_path):
            errors.append(f"Text hash mismatch for {source_hash}")
        if metadata.get("truncated"):
            truncated += 1
        verified += 1

    result = {
        "verification_schema_version": 1,
        "pilot_name": pilot_name,
        "expected_unique_contents": len(unique_hashes),
        "verified_unique_contents": verified,
        "truncated_extractions": truncated,
        "error_count": len(errors),
        "errors": errors,
        "registry_digest": registry.digest,
        "ready_for_chunking": not errors,
    }
    return result
