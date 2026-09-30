"""Acquire one exact-revision Hugging Face dataset shard for public MFM review.

No source is admitted by a download. This step only proves the local bytes
match the dataset repository's pinned LFS SHA-256 or Git blob ID. The caller
chooses which files belong to the corpus; there is no implicit row/file cap.
"""

from __future__ import annotations

import argparse
from hashlib import sha1
import json
import os
from pathlib import Path

from scripts.mfm.native_public_corpus import _digest, _json_bytes, _safe_relative, _source


ACQUISITION_SCHEMA = "mfm-native-hf-exact-file-acquisition-v1"


def _git_blob_id(path: Path) -> str:
    value = sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch_exact_file(*, candidate_file: Path, candidate_sha: str,
                     n0_file: Path, n0_sha: str, candidate_id: str,
                     upstream_repo_path: str, output: Path,
                     api=None, downloader=None) -> dict:
    candidate, source = _source(candidate_file, candidate_sha, n0_file,
                                n0_sha, candidate_id)
    if candidate.get("dataset_repo_id") != source["repo_id"]:
        raise ValueError("candidate/N0 repository mismatch")
    if api is None or downloader is None:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise RuntimeError("CPU acquisition requires huggingface_hub") from exc
        api = api or HfApi()
        downloader = downloader or hf_hub_download
    repo = source["repo_id"]
    revision = source["revision"]
    if str(api.dataset_info(repo_id=repo, revision=revision).sha) != revision:
        raise ValueError("Hub did not resolve the exact pinned dataset commit")
    entries = [entry for entry in api.get_paths_info(repo, paths=[upstream_repo_path],
                                                    revision=revision, repo_type="dataset",
                                                    expand=True)
               if getattr(entry, "path", None) == upstream_repo_path]
    if len(entries) != 1:
        raise ValueError("file missing/ambiguous in pinned dataset commit")
    entry = entries[0]
    lfs = getattr(entry, "lfs", None)
    expected_lfs = getattr(lfs, "sha256", None) if lfs is not None else None
    expected_blob = getattr(entry, "blob_id", None)
    if expected_lfs is not None:
        if (len(expected_lfs) != 64 or
                any(ch not in "0123456789abcdef" for ch in expected_lfs)):
            raise ValueError("pinned LFS object has no verifiable SHA-256")
    elif (not isinstance(expected_blob, str) or len(expected_blob) != 40 or
          any(ch not in "0123456789abcdef" for ch in expected_blob)):
        raise ValueError("pinned regular Git blob has no verifiable object ID")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = _safe_relative(output, upstream_repo_path)
    raw = Path(downloader(repo_id=repo, filename=upstream_repo_path,
                          repo_type="dataset", revision=revision,
                          local_dir=str(output), local_dir_use_symlinks=False)).resolve()
    if not raw.is_relative_to(output) or raw != destination:
        raise ValueError("Hub download escaped exact local dataset path")
    actual_sha = _digest(raw)
    if expected_lfs is not None:
        if actual_sha != expected_lfs or raw.stat().st_size != getattr(lfs, "size", None):
            raise ValueError("download differs from pinned LFS object")
    elif _git_blob_id(raw) != expected_blob:
        raise ValueError("download differs from pinned Git blob")
    receipt = {"schema": ACQUISITION_SCHEMA,
               "status": "bytes_verified_no_training_admission",
               "candidate_inventory_sha256": candidate_sha,
               "n0_manifest_sha256": n0_sha, "candidate_id": candidate_id,
               "dataset_repo_id": repo, "revision": revision,
               "upstream_repo_path": upstream_repo_path,
               "upstream_lfs_sha256": expected_lfs,
               "upstream_git_blob_id": expected_blob,
               "downloaded_sha256": actual_sha,
               "downloaded_bytes": raw.stat().st_size,
               "raw_origin_repo_metadata_verified": True,
               "rights_admitted": False, "training_admitted": False}
    receipt_path = output / "acquisition_receipt.json"
    payload = _json_bytes(receipt)
    if receipt_path.exists():
        if receipt_path.read_bytes() != payload:
            raise ValueError("existing acquisition receipt conflicts with pinned bytes")
    else:
        with receipt_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--n0-manifest", type=Path, required=True)
    parser.add_argument("--n0-sha256", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--upstream-repo-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = fetch_exact_file(candidate_file=args.candidate_inventory,
                               candidate_sha=args.candidate_sha256,
                               n0_file=args.n0_manifest, n0_sha=args.n0_sha256,
                               candidate_id=args.candidate_id,
                               upstream_repo_path=args.upstream_repo_path,
                               output=args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
