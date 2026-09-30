"""Legacy Gemma-derivative research snapshot staging, superseded for MFM.

``stage`` compares every repository file to Hugging Face's metadata for the
pinned commit. ``verify`` is wholly local and rehashes every staged file. The
receipt travels with the snapshot. An offline verification at a new path uses
``snapshot_override`` and rehashes the copied bytes.

The receipt is content-addressed, not a signature. Keep its expected digest in
the run record or another independently controlled location.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha1, sha256
import json
from pathlib import Path
import re

from cognitive_kernel.formation_multimodal import GEMMA_4_12B_MODEL, GEMMA_4_12B_REVISION


RECEIPT_SCHEMA = "mfm-gemma4-staged-model-v1"
# Published on the exact pinned model.safetensors file page. This independent
# anchor prevents a self-rehashed receipt from silently changing the weights.
PINNED_WEIGHT_SHA256 = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ESSENTIAL_FILES = frozenset({
    "config.json", "generation_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "model.safetensors",
})


class StagedModelError(ValueError):
    """An upstream snapshot or a local sealed snapshot cannot be trusted."""


def _canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _regular_files(snapshot: Path) -> dict[str, Path]:
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise StagedModelError("snapshot path must be a real directory")
    files = {}
    for path in snapshot.rglob("*"):
        rel = path.relative_to(snapshot)
        # local_dir metadata belongs to huggingface_hub, not the model.
        if rel.parts[:2] == (".cache", "huggingface"):
            continue
        if path.is_symlink():
            raise StagedModelError(f"snapshot contains a symlink: {rel}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise StagedModelError(f"snapshot contains a nonregular file: {rel}")
        files[rel.as_posix()] = path
    if not _ESSENTIAL_FILES.issubset(files):
        raise StagedModelError(
            f"snapshot is missing essential files: {sorted(_ESSENTIAL_FILES - files.keys())}")
    return files


def _hash_file(path: Path, *, git_blob: bool = False) -> tuple[int, str, str | None]:
    size = path.stat().st_size
    digest = sha256()
    git = sha1(f"blob {size}\0".encode("ascii")) if git_blob else None
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            if git is not None:
                git.update(chunk)
    if path.stat().st_size != size:
        raise StagedModelError(f"file changed during hashing: {path}")
    return size, digest.hexdigest(), git.hexdigest() if git is not None else None


def _meta_value(obj: object, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _official_files(api, *, model: str, revision: str,
                    metadata_for_file=None) -> dict[str, dict]:
    info = api.model_info(model, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise StagedModelError("Hub resolved a different model commit")
    siblings = {entry.rfilename: entry for entry in info.siblings or ()}
    # Some Hub versions omit expanded blob metadata from model_info().
    if (not _meta_value(siblings.get("model.safetensors"), "lfs") or
            any(not _meta_value(entry, "lfs") and not _meta_value(entry, "blob_id")
                for entry in siblings.values())):
        for entry in api.list_repo_tree(model, revision=revision,
                                        recursive=True, expand=True):
            path = _meta_value(entry, "path")
            if path in siblings and _meta_value(entry, "size") is not None:
                siblings[path] = entry
    if not siblings or not _ESSENTIAL_FILES.issubset(siblings):
        raise StagedModelError("Hub metadata lacks essential pinned model files")
    official = {}
    for path, entry in siblings.items():
        lfs = _meta_value(entry, "lfs")
        lfs_sha256 = _meta_value(lfs, "sha256") if lfs else None
        blob_id = _meta_value(entry, "blob_id")
        size = _meta_value(entry, "size")
        if lfs_sha256 and _HEX64.fullmatch(lfs_sha256):
            check = {"kind": "content_sha256", "digest": lfs_sha256}
        else:
            # Xet-backed objects may omit LFS fields from a RepoSibling. The
            # exact-revision resolve endpoint supplies a content ETag and size.
            head = metadata_for_file(path) if metadata_for_file else None
            etag = _meta_value(head, "etag")
            head_commit = _meta_value(head, "commit_hash")
            head_size = _meta_value(head, "size")
            if head and (head_commit != revision or
                         not isinstance(head_size, int) or head_size < 0):
                raise StagedModelError(f"Hub HEAD differs from pinned commit: {path}")
            if head and size is not None and size != head_size:
                raise StagedModelError(f"Hub metadata size differs: {path}")
            if head:
                size = head_size
            if isinstance(etag, str) and _HEX64.fullmatch(etag):
                check = {"kind": "content_sha256", "digest": etag}
            elif isinstance(etag, str) and _HEX40.fullmatch(etag):
                check = {"kind": "git_blob_sha1", "digest": etag}
            elif blob_id and _HEX40.fullmatch(blob_id):
                check = {"kind": "git_blob_sha1", "digest": blob_id}
            elif path == "model.safetensors" and isinstance(size, int):
                check = {"kind": "content_sha256", "digest": PINNED_WEIGHT_SHA256}
            else:
                raise StagedModelError(f"Hub supplies no content hash for {path}")
        if path == "model.safetensors":
            if check["kind"] == "git_blob_sha1":
                # Xet may expose the SHA-1 of a *pointer*, not the actual
                # reconstructed weights. The independently pinned file SHA
                # checks the actual bytes after download.
                check = {"kind": "content_sha256", "digest": PINNED_WEIGHT_SHA256}
            elif check["digest"] != PINNED_WEIGHT_SHA256:
                raise StagedModelError("pinned weight SHA-256 disagrees with Hub")
        if not isinstance(size, int) or size < 0:
            raise StagedModelError(f"Hub supplies no size for {path}")
        official[path] = {"size": size, "upstream": check}
    return official


def stage_model(snapshot_dir: str | Path, receipt_path: str | Path, *,
                download: bool = False) -> dict:
    """Validate a local snapshot against the exact upstream commit and seal it."""
    snapshot = Path(snapshot_dir).expanduser().resolve()
    receipt_file = Path(receipt_path).expanduser().resolve()
    if receipt_file == snapshot or snapshot in receipt_file.parents:
        raise StagedModelError("receipt must be outside the snapshot directory")
    if receipt_file.exists():
        raise StagedModelError("receipt already exists; keep previous seals immutable")
    try:
        from huggingface_hub import (
            HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download,
        )
    except ImportError as exc:
        raise StagedModelError("install huggingface_hub on the CPU staging machine") from exc
    def metadata_for_file(path: str):
        return get_hf_file_metadata(hf_hub_url(
            repo_id=GEMMA_4_12B_MODEL, filename=path,
            revision=GEMMA_4_12B_REVISION))

    official = _official_files(HfApi(), model=GEMMA_4_12B_MODEL,
                               revision=GEMMA_4_12B_REVISION,
                               metadata_for_file=metadata_for_file)
    if download:
        snapshot_download(repo_id=GEMMA_4_12B_MODEL, revision=GEMMA_4_12B_REVISION,
                          local_dir=str(snapshot))
    files = _regular_files(snapshot)
    if set(files) != set(official):
        raise StagedModelError(
            f"snapshot differs from pinned Hub tree: missing={sorted(set(official)-set(files))}, "
            f"unexpected={sorted(set(files)-set(official))}")
    records = []
    for relative, path in sorted(files.items()):
        expected = official[relative]
        upstream = expected["upstream"]
        size, digest, git_blob = _hash_file(
            path, git_blob=upstream["kind"] == "git_blob_sha1")
        if size != expected["size"] or (
            (digest if upstream["kind"] == "content_sha256" else git_blob) != upstream["digest"]
        ):
            raise StagedModelError(f"pinned upstream content differs: {relative}")
        records.append({"path": relative, "size": size, "sha256": digest,
                        "upstream": upstream})
    receipt = {"schema": RECEIPT_SCHEMA, "model": GEMMA_4_12B_MODEL,
               "revision": GEMMA_4_12B_REVISION,
               "snapshot_path": str(snapshot), "files": records,
               "total_bytes": sum(row["size"] for row in records),
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    receipt["receipt_sha256"] = sha256(_canonical(receipt)).hexdigest()
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    with receipt_file.open("x", encoding="utf-8") as output:
        output.write(_canonical(receipt).decode("utf-8") + "\n")
    return receipt


def verify_staged_model(receipt_path: str | Path, expected_model: str,
                        expected_revision: str, rehash: bool = True,
                        snapshot_override: str | Path | None = None) -> tuple[Path, dict]:
    """Offline, fail-closed verification before any GPU weights are loaded."""
    receipt_file = Path(receipt_path).expanduser().resolve()
    receipt = json.loads(receipt_file.read_bytes())
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "model", "revision", "snapshot_path", "files", "total_bytes",
        "created_at_utc", "receipt_sha256",
    }:
        raise StagedModelError("invalid staged model receipt schema")
    actual_digest = receipt.pop("receipt_sha256")
    if not isinstance(actual_digest, str) or not _HEX64.fullmatch(actual_digest) or \
            sha256(_canonical(receipt)).hexdigest() != actual_digest:
        raise StagedModelError("staged model receipt digest mismatch")
    receipt["receipt_sha256"] = actual_digest
    if (receipt["schema"] != RECEIPT_SCHEMA or
            receipt["model"] != expected_model or
            receipt["revision"] != expected_revision or
            expected_model != GEMMA_4_12B_MODEL or
            expected_revision != GEMMA_4_12B_REVISION):
        raise StagedModelError("staged model does not match pinned MFM backbone")
    raw_path = receipt["snapshot_path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise StagedModelError("staged model path is missing")
    if snapshot_override is None and not Path(raw_path).is_absolute():
        raise StagedModelError("staged model path is not absolute")
    snapshot = Path(snapshot_override).expanduser() if snapshot_override else Path(raw_path)
    if not snapshot.is_absolute() or snapshot.resolve() != snapshot or snapshot.is_symlink():
        raise StagedModelError("snapshot path changed or contains a symlink")
    files = _regular_files(snapshot)
    listed = receipt["files"]
    if not isinstance(listed, list) or not listed or any(
        not isinstance(item, dict) or set(item) != {"path", "size", "sha256", "upstream"}
        for item in listed
    ):
        raise StagedModelError("invalid staged model file manifest")
    names = [item["path"] for item in listed]
    if names != sorted(files) or len(set(names)) != len(names):
        raise StagedModelError("staged model files were added, removed or reordered")
    if receipt_file == snapshot or snapshot in receipt_file.parents:
        raise StagedModelError("receipt must be outside the snapshot directory")
    total = 0
    weight_seen = False
    for row in listed:
        name, size, digest = row["path"], row["size"], row["sha256"]
        upstream = row["upstream"]
        if (not isinstance(size, int) or size < 0 or
                not isinstance(digest, str) or not _HEX64.fullmatch(digest) or
                not isinstance(upstream, dict) or set(upstream) != {"kind", "digest"} or
                upstream["kind"] not in {"content_sha256", "git_blob_sha1"}):
            raise StagedModelError(f"invalid file record: {name}")
        value = upstream["digest"]
        pattern = _HEX64 if upstream["kind"] == "content_sha256" else _HEX40
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise StagedModelError(f"invalid upstream digest: {name}")
        if name == "model.safetensors":
            weight_seen = True
            if (upstream != {"kind": "content_sha256", "digest": PINNED_WEIGHT_SHA256}
                    or digest != PINNED_WEIGHT_SHA256):
                raise StagedModelError("staged model weight digest differs from pinned anchor")
        if files[name].stat().st_size != size:
            raise StagedModelError(f"staged model file size changed: {name}")
        if rehash:
            actual_size, actual_sha, git_blob = _hash_file(
                files[name], git_blob=upstream["kind"] == "git_blob_sha1")
            if actual_size != size or actual_sha != digest or (
                (actual_sha if upstream["kind"] == "content_sha256" else git_blob) != value
            ):
                raise StagedModelError(f"staged model file changed: {name}")
        total += size
    if not weight_seen or total != receipt["total_bytes"]:
        raise StagedModelError("staged model total size changed")
    return snapshot, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="online comparison to the pinned Hub commit")
    stage.add_argument("--snapshot-dir", required=True, type=Path)
    stage.add_argument("--receipt", required=True, type=Path)
    stage.add_argument("--download", action="store_true",
                       help="download the full pinned snapshot before verification")
    stage.add_argument("--research-derivative-only", action="store_true",
                       help="explicit nonproduct research; never stage as an FBM personal MFM base")
    verify = commands.add_parser("verify", help="offline rehash of the sealed snapshot")
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument("--snapshot-dir", type=Path,
                        help="verified copy at a new absolute path; rehash every file")
    args = parser.parse_args()
    if args.command == "stage":
        if not args.research_derivative_only:
            parser.error("Gemma staging is superseded for personal MFM; see "
                         "docs/MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md")
        receipt = stage_model(args.snapshot_dir, args.receipt, download=args.download)
        result = {"receipt_sha256": receipt["receipt_sha256"],
                  "files": len(receipt["files"]), "total_bytes": receipt["total_bytes"],
                  "snapshot_path": receipt["snapshot_path"]}
    else:
        snapshot, receipt = verify_staged_model(args.receipt, GEMMA_4_12B_MODEL,
                                                GEMMA_4_12B_REVISION,
                                                snapshot_override=args.snapshot_dir)
        result = {"receipt_sha256": receipt["receipt_sha256"],
                  "files": len(receipt["files"]), "snapshot_path": str(snapshot),
                  "verified": True}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
