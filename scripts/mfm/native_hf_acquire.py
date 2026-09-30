"""Inventory and acquire exact-revision Hugging Face data for public MFM review.

No source is admitted by a download. This step only proves the local bytes
match the dataset repository's pinned LFS SHA-256 or Git blob ID. A complete,
reviewed repository inventory is required before fetching any selected shard.
There is no implicit row/file cap.
"""

from __future__ import annotations

import argparse
from hashlib import sha1
import json
import os
from pathlib import Path

from scripts.mfm.native_public_corpus import (
    _bound_json, _digest, _json_bytes, _safe_relative, _source,
)


ACQUISITION_SCHEMA = "mfm-native-hf-exact-file-acquisition-v1"
TREE_SCHEMA = "mfm-native-hf-repo-tree-v1"
SELECTION_SCHEMA = "mfm-native-hf-file-selection-v1"
FROZEN_SCHEMA = "mfm-native-hf-frozen-file-inventory-v1"
COVERAGE_LIST_SCHEMA = "mfm-native-hf-acquisition-receipt-list-v1"
COVERAGE_SCHEMA = "mfm-native-hf-complete-acquisition-coverage-v1"
DATA_SUFFIXES = (".parquet", ".jsonl", ".jsonl.gz")
REPO_METADATA = {".gitattributes", ".gitignore", ".hfignore", "README.md",
                 "LICENSE", "LICENSE.md", "dataset_infos.json"}


def _hex(value: object, length: int, label: str) -> str:
    if (not isinstance(value, str) or len(value) != length or
            any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} has no verifiable {length}-hex object ID")
    return value


def _path(value: object) -> str:
    if (not isinstance(value, str) or not value or value.startswith("/") or
            "\\" in value or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("invalid relative path in pinned repository tree")
    return value


def _object(entry) -> dict:
    path = _path(getattr(entry, "path", None))
    size = getattr(entry, "size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"file {path} has no verifiable size")
    blob = _hex(getattr(entry, "blob_id", None), 40, f"file {path} Git blob")
    lfs = getattr(entry, "lfs", None)
    lfs_sha = None
    if lfs is not None:
        lfs_sha = _hex(getattr(lfs, "sha256", None), 64, f"file {path} LFS SHA-256")
        if getattr(lfs, "size", None) != size:
            raise ValueError(f"file {path} LFS and repository sizes differ")
    if path.endswith(DATA_SUFFIXES):
        suggestion = "data_shard_candidate"
    elif path in REPO_METADATA:
        suggestion = "repository_metadata"
    else:
        suggestion = "manual_review_required"
    return {"path": path, "size": size, "git_blob_id": blob,
            "lfs_sha256": lfs_sha, "suggestion": suggestion}


def _write_frozen(path: Path, record: dict) -> None:
    """A conflicting existing manifest is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(record)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing manifest conflicts with pinned repository state")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())


def inventory_exact_repo(*, candidate_file: Path, candidate_sha: str,
                         n0_file: Path, n0_sha: str, candidate_id: str,
                         output: Path, api=None) -> dict:
    """Exhaust the Hub's paginated, recursive tree at the pinned 40-hex commit."""
    candidate, source = _source(candidate_file, candidate_sha, n0_file,
                                n0_sha, candidate_id)
    repo, revision = source["repo_id"], _hex(source["revision"], 40, "revision")
    if candidate.get("dataset_repo_id") != repo:
        raise ValueError("candidate/N0 repository mismatch")
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("CPU inventory requires huggingface_hub") from exc
        api = HfApi()
    if str(api.dataset_info(repo_id=repo, revision=revision).sha) != revision:
        raise ValueError("Hub did not resolve the exact pinned dataset commit")
    files, folders, seen = [], [], set()
    # HfApi.list_repo_tree transparently follows every page. Exhaust the iterator;
    # no caller-supplied limit, path prefix or page truncation is permitted.
    for entry in api.list_repo_tree(repo_id=repo, revision=revision,
                                    repo_type="dataset", recursive=True, expand=False):
        path = _path(getattr(entry, "path", None))
        if path in seen:
            raise ValueError(f"duplicate repository tree path {path}")
        seen.add(path)
        if hasattr(entry, "size"):
            files.append(_object(entry))
        elif hasattr(entry, "tree_id"):
            folders.append({"path": path,
                            "git_tree_id": _hex(entry.tree_id, 40, "directory Git tree")})
        else:
            raise ValueError(f"unknown repository tree entry {path}")
    if not files:
        raise ValueError("pinned dataset repository tree contains no files")
    if str(api.dataset_info(repo_id=repo, revision=revision).sha) != revision:
        raise ValueError("Hub dataset revision changed during inventory")
    record = {"schema": TREE_SCHEMA,
              "status": "complete_tree_candidate_no_training_admission",
              "candidate_inventory_sha256": candidate_sha,
              "n0_manifest_sha256": n0_sha, "candidate_id": candidate_id,
              "dataset_repo_id": repo, "revision": revision,
              "enumeration": "HfApi.list_repo_tree(recursive=True,expand=False) exhausted",
              "files": sorted(files, key=lambda row: row["path"]),
              "folders": sorted(folders, key=lambda row: row["path"]),
              "file_count": len(files),
              "total_file_bytes": sum(row["size"] for row in files),
              "rights_admitted": False, "training_admitted": False}
    _write_frozen(output, record)
    return record


def selection_template(tree: dict) -> dict:
    """Propose all obvious shards and metadata; require review of other files."""
    decisions = []
    for row in tree["files"]:
        suggestion = row["suggestion"]
        decision, reason = (
            ("include", "recognized data shard; confirm split and source terms")
            if suggestion == "data_shard_candidate" else
            ("exclude", "repository metadata, not corpus bytes")
            if suggestion == "repository_metadata" else
            ("review_required", "classify this file explicitly"))
        decisions.append({"path": row["path"], "decision": decision, "reason": reason})
    return {"schema": SELECTION_SCHEMA, "repo_tree_sha256": None,
            "review_record": None,
            "files": decisions}


def freeze_selection(*, tree_file: Path, tree_sha: str,
                     selection_file: Path, selection_sha: str, output: Path) -> dict:
    """Bind a reviewed decision for *every* file in the exact tree."""
    tree = _bound_json(tree_file, tree_sha)
    selection = _bound_json(selection_file, selection_sha)
    if (tree.get("schema") != TREE_SCHEMA or
            tree.get("status") != "complete_tree_candidate_no_training_admission" or
            tree.get("training_admitted") is not False or
            selection.get("schema") != SELECTION_SCHEMA or
            selection.get("repo_tree_sha256") != tree_sha):
        raise ValueError("tree/selection is not a bound unadmitted repository inventory")
    if (not isinstance(selection.get("review_record"), str) or
            not selection["review_record"].strip()):
        raise ValueError("selection requires an explicit source/file review record")
    files = tree.get("files")
    if (not isinstance(files, list) or len(files) != tree.get("file_count") or
            len({row["path"] for row in files}) != len(files) or
            files != sorted(files, key=lambda row: row["path"])):
        raise ValueError("repo tree inventory has duplicate, missing or unsorted files")
    file_paths = {row["path"] for row in files}
    decisions = selection.get("files")
    if not isinstance(decisions, list) or len(decisions) != len(files):
        raise ValueError("selection must account for every repository file")
    by_path = {}
    for row in decisions:
        path = _path(row.get("path"))
        if path in by_path or path not in file_paths:
            raise ValueError("selection has duplicate or unknown repository path")
        if row.get("decision") not in {"include", "exclude"}:
            raise ValueError("unreviewed or invalid file selection")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("each inclusion/exclusion needs an explicit reason")
        by_path[path] = {"decision": row["decision"], "reason": row["reason"].strip()}
    if set(by_path) != file_paths:
        raise ValueError("selection omits repository files")
    rows = []
    for row in files:
        path = row["path"]
        decision = by_path[path]
        if decision["decision"] == "include" and not path.endswith(DATA_SUFFIXES):
            raise ValueError(f"selected shard format unsupported for {path}")
        rows.append({**row, **decision})
    selected = [row for row in rows if row["decision"] == "include"]
    if not selected:
        raise ValueError("no data shards selected in complete repository inventory")
    frozen = {"schema": FROZEN_SCHEMA,
              "status": "file_selection_frozen_no_training_admission",
              "candidate_inventory_sha256": tree["candidate_inventory_sha256"],
              "n0_manifest_sha256": tree["n0_manifest_sha256"],
              "candidate_id": tree["candidate_id"],
              "dataset_repo_id": tree["dataset_repo_id"],
              "revision": tree["revision"],
              "repo_tree_sha256": tree_sha, "reviewed_selection_sha256": selection_sha,
              "selection_review_record": selection["review_record"].strip(),
              "files": rows, "folders": tree["folders"],
              "file_count": len(rows), "selected_data_count": len(selected),
              "selected_data_bytes": sum(row["size"] for row in selected),
              "rights_admitted": False, "training_admitted": False}
    _write_frozen(output, frozen)
    return frozen


def verify_complete_acquisitions(*, frozen_inventory_file: Path,
                                 frozen_inventory_sha: str, receipt_list: Path,
                                 receipt_list_sha: str, output: Path) -> dict:
    """Require exactly one byte-verified acquisition for every selected shard."""
    frozen = _bound_json(frozen_inventory_file, frozen_inventory_sha)
    bundle = _bound_json(receipt_list, receipt_list_sha)
    if (frozen.get("schema") != FROZEN_SCHEMA or
            frozen.get("status") != "file_selection_frozen_no_training_admission" or
            bundle.get("schema") != COVERAGE_LIST_SCHEMA or
            bundle.get("frozen_repo_inventory_sha256") != frozen_inventory_sha or
            not isinstance(bundle.get("receipts"), list)):
        raise ValueError("acquisition list is not bound to frozen repository selection")
    selected = {row["path"]: row for row in frozen["files"]
                if row["decision"] == "include"}
    if len(selected) != frozen["selected_data_count"]:
        raise ValueError("duplicate paths in frozen data selection")
    observed, seen_receipts = {}, set()
    for link in bundle["receipts"]:
        filename = _path(link["path"])
        if filename in seen_receipts:
            raise ValueError("same acquisition receipt listed twice")
        seen_receipts.add(filename)
        path = _safe_relative(receipt_list.parent.resolve(), filename)
        receipt = _bound_json(path, link["sha256"])
        shard = receipt.get("upstream_repo_path")
        if shard in observed or shard not in selected:
            raise ValueError("duplicate or unselected acquired shard")
        row = selected[shard]
        if (receipt.get("schema") != ACQUISITION_SCHEMA or
                receipt.get("status") != "bytes_verified_no_training_admission" or
                receipt.get("training_admitted") is not False or
                receipt.get("frozen_repo_inventory_sha256") != frozen_inventory_sha or
                receipt.get("candidate_inventory_sha256") !=
                frozen["candidate_inventory_sha256"] or
                receipt.get("candidate_id") != frozen["candidate_id"] or
                receipt.get("dataset_repo_id") != frozen["dataset_repo_id"] or
                receipt.get("revision") != frozen["revision"] or
                receipt.get("downloaded_bytes") != row["size"] or
                receipt.get("upstream_lfs_sha256") != row["lfs_sha256"] or
                receipt.get("upstream_git_blob_id") != row["git_blob_id"]):
            raise ValueError("acquisition receipt differs from selected repository shard")
        raw = _safe_relative(path.parent.resolve(), shard)
        if raw.stat().st_size != row["size"] or _digest(raw) != receipt["downloaded_sha256"]:
            raise ValueError("acquired shard bytes differ from acquisition receipt")
        if row["lfs_sha256"]:
            if receipt["downloaded_sha256"] != row["lfs_sha256"]:
                raise ValueError("acquired shard differs from pinned LFS SHA-256")
        elif _git_blob_id(raw) != row["git_blob_id"]:
            raise ValueError("acquired shard differs from pinned Git blob")
        observed[shard] = {"path": filename, "sha256": link["sha256"]}
    if set(observed) != set(selected):
        missing = sorted(set(selected) - set(observed))
        raise ValueError(f"acquisition receipt list omits {len(missing)} selected shards")
    record = {"schema": COVERAGE_SCHEMA,
              "status": "all_selected_upstream_bytes_verified_no_training_admission",
              "frozen_repo_inventory_sha256": frozen_inventory_sha,
              "receipt_list_sha256": receipt_list_sha,
              "candidate_id": frozen["candidate_id"],
              "revision": frozen["revision"],
              "shards": [{"upstream_repo_path": path, **observed[path]}
                         for path in sorted(observed)],
              "shard_count": len(observed),
              "selected_data_bytes": frozen["selected_data_bytes"],
              "rights_admitted": False, "training_admitted": False}
    _write_frozen(output, record)
    return record


def _git_blob_id(path: Path) -> str:
    value = sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch_exact_file(*, candidate_file: Path, candidate_sha: str,
                     n0_file: Path, n0_sha: str, candidate_id: str,
                     upstream_repo_path: str, output: Path,
                     frozen_inventory_file: Path, frozen_inventory_sha: str,
                     api=None, downloader=None) -> dict:
    candidate, source = _source(candidate_file, candidate_sha, n0_file,
                                n0_sha, candidate_id)
    if candidate.get("dataset_repo_id") != source["repo_id"]:
        raise ValueError("candidate/N0 repository mismatch")
    frozen = _bound_json(frozen_inventory_file, frozen_inventory_sha)
    if (frozen.get("schema") != FROZEN_SCHEMA or
            frozen.get("status") != "file_selection_frozen_no_training_admission" or
            frozen.get("candidate_inventory_sha256") != candidate_sha or
            frozen.get("n0_manifest_sha256") != n0_sha or
            frozen.get("candidate_id") != candidate_id or
            frozen.get("dataset_repo_id") != source["repo_id"] or
            frozen.get("revision") != source["revision"] or
            frozen.get("training_admitted") is not False):
        raise ValueError("file is not bound to a frozen source-wide selection")
    file_rows = frozen.get("files")
    if (not isinstance(file_rows, list) or
            len(file_rows) != frozen.get("file_count") or
            len({row["path"] for row in file_rows}) != len(file_rows) or
            sum(row["decision"] == "include" for row in file_rows) !=
            frozen.get("selected_data_count")):
        raise ValueError("frozen repository file inventory is incomplete")
    chosen = [row for row in file_rows if row["path"] == upstream_repo_path and
              row["decision"] == "include"]
    if len(chosen) != 1 or not upstream_repo_path.endswith(DATA_SUFFIXES):
        raise ValueError("file is not included in frozen data shard selection")
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
    pinned = _object(entry)
    selected = chosen[0]
    for field in ("path", "size", "lfs_sha256", "git_blob_id"):
        if selected.get(field) != pinned[field]:
            raise ValueError("Hub shard metadata differs from frozen repo tree")
    expected_lfs, expected_blob = pinned["lfs_sha256"], pinned["git_blob_id"]
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
        if actual_sha != expected_lfs or raw.stat().st_size != selected["size"]:
            raise ValueError("download differs from pinned LFS object")
    elif raw.stat().st_size != selected["size"] or _git_blob_id(raw) != expected_blob:
        raise ValueError("download differs from pinned Git blob")
    receipt = {"schema": ACQUISITION_SCHEMA,
               "status": "bytes_verified_no_training_admission",
               "candidate_inventory_sha256": candidate_sha,
               "frozen_repo_inventory_sha256": frozen_inventory_sha,
               "repo_tree_sha256": frozen["repo_tree_sha256"],
               "reviewed_selection_sha256": frozen["reviewed_selection_sha256"],
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
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="exhaust and freeze a pinned Hub tree")
    fetch = commands.add_parser("fetch", help="fetch an included shard from a frozen tree")
    freeze = commands.add_parser("freeze", help="bind an explicit decision for every repo file")
    coverage = commands.add_parser("coverage", help="prove all included shards were acquired")
    for command in (inventory, fetch):
        command.add_argument("--candidate-inventory", type=Path, required=True)
        command.add_argument("--candidate-sha256", required=True)
        command.add_argument("--n0-manifest", type=Path, required=True)
        command.add_argument("--n0-sha256", required=True)
        command.add_argument("--candidate-id", required=True)
        command.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--selection-template-output", type=Path, required=True)
    freeze.add_argument("--repo-tree", type=Path, required=True)
    freeze.add_argument("--repo-tree-sha256", required=True)
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--selection-sha256", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    coverage.add_argument("--frozen-inventory", type=Path, required=True)
    coverage.add_argument("--frozen-inventory-sha256", required=True)
    coverage.add_argument("--receipt-list", type=Path, required=True)
    coverage.add_argument("--receipt-list-sha256", required=True)
    coverage.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--frozen-inventory", type=Path, required=True)
    fetch.add_argument("--frozen-inventory-sha256", required=True)
    fetch.add_argument("--upstream-repo-path", required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        record = inventory_exact_repo(candidate_file=args.candidate_inventory,
                                      candidate_sha=args.candidate_sha256,
                                      n0_file=args.n0_manifest, n0_sha=args.n0_sha256,
                                      candidate_id=args.candidate_id, output=args.output)
        template = selection_template(record)
        template["repo_tree_sha256"] = _digest(args.output)
        _write_frozen(args.selection_template_output, template)
        result = {"repo_tree_sha256": _digest(args.output),
                  "selection_template_sha256": _digest(args.selection_template_output),
                  "file_count": record["file_count"],
                  "status": record["status"]}
    elif args.command == "freeze":
        record = freeze_selection(tree_file=args.repo_tree,
                                  tree_sha=args.repo_tree_sha256,
                                  selection_file=args.selection,
                                  selection_sha=args.selection_sha256,
                                  output=args.output)
        result = {"frozen_inventory_sha256": _digest(args.output),
                  "selected_data_count": record["selected_data_count"],
                  "selected_data_bytes": record["selected_data_bytes"],
                  "status": record["status"]}
    elif args.command == "coverage":
        record = verify_complete_acquisitions(
            frozen_inventory_file=args.frozen_inventory,
            frozen_inventory_sha=args.frozen_inventory_sha256,
            receipt_list=args.receipt_list,
            receipt_list_sha=args.receipt_list_sha256, output=args.output)
        result = {"coverage_sha256": _digest(args.output),
                  "shard_count": record["shard_count"], "status": record["status"]}
    else:
        result = fetch_exact_file(candidate_file=args.candidate_inventory,
                                  candidate_sha=args.candidate_sha256,
                                  n0_file=args.n0_manifest, n0_sha=args.n0_sha256,
                                  candidate_id=args.candidate_id,
                                  frozen_inventory_file=args.frozen_inventory,
                                  frozen_inventory_sha=args.frozen_inventory_sha256,
                                  upstream_repo_path=args.upstream_repo_path,
                                  output=args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
