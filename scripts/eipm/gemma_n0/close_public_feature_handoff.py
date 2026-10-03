"""Allocated CPU closure of existing public features; no model or training.

The new package copies cache/evidence/code only, never the 24GB checkpoint.
Producer retokenization/source verification is delegated to the frozen exporter;
the consumer imports closed bytes without a tokenizer or publisher model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
handoff = producer = None


def modules() -> None:
    global handoff, producer
    if handoff is None:
        from src.alice_personality.gemma_n0 import public_feature_handoff
        from src.alice_personality.gemma_n0 import public_semantic_experiment
        handoff, producer = public_feature_handoff, public_semantic_experiment

# Operational allocation for small closure documents/filesystem overhead, not
# a data/capability ceiling. Exact existing file sizes determine the main cost.
METADATA_SLACK_BYTES = 16 * 1024 * 1024
# Host-side inventory admission limits for this two-wheel overlay only. Reject
# before streaming any hashes; larger installations need allocated inspection.
MAX_OVERLAY_FILES = 5000
MAX_OVERLAY_BYTES = 128 * 1024 * 1024


class LaunchError(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise LaunchError(reason)


def within(raw: str | Path, root: Path, *, directory: bool = False) -> Path:
    modules()
    path = handoff._regular(raw, directory=directory)
    require(path == root or root in path.parents, "input escapes explicit compute root")
    return path


def overlay_regular(raw: str | Path, *, directory: bool = False) -> Path:
    path = Path(raw)
    require(path.is_absolute() and not any(p.is_symlink() or
        (hasattr(p, "is_junction") and p.is_junction()) for p in (path, *path.parents)),
        "overlay paths cannot traverse links")
    require((path.is_dir() if directory else path.is_file()) and path == path.resolve(strict=True),
            "overlay must contain exact regular files/directories")
    return path


def overlay_file_hash(path: Path) -> str:
    before = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    require(identity(before) == identity(after), "overlay changed during hashing")
    return digest.hexdigest()


def overlay_inventory(directory: str | Path) -> dict:
    """Read-only pin of every relative file and directory; reject links/aliases."""
    root = overlay_regular(directory, directory=True)
    files, directories, namespaces, candidates, total = [], [], {}, [], 0
    def register(relative):
        parts = relative.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            folded = prefix.casefold()
            previous = namespaces.get(folded)
            require(previous is None or previous == prefix, "overlay contains aliased namespace paths")
            namespaces[folded] = prefix
    for parent, names, leaves in os.walk(root, followlinks=False):
        for name in sorted(names):
            path = overlay_regular(Path(parent) / name, directory=True)
            relative = path.relative_to(root).as_posix()
            register(relative)
            require(relative not in directories, "overlay contains duplicate directories")
            directories.append(relative)
            require(len(directories) <= MAX_OVERLAY_FILES, "overlay directory scan budget exceeded")
        for name in sorted(leaves):
            path = overlay_regular(Path(parent) / name)
            relative = path.relative_to(root).as_posix()
            register(relative)
            require(not any(candidate[0] == relative for candidate in candidates),
                    "overlay contains duplicate files")
            size = path.stat().st_size
            total += size
            candidates.append((relative, path, size))
            require(len(candidates) <= MAX_OVERLAY_FILES and total <= MAX_OVERLAY_BYTES,
                    "overlay host scan budget exceeded before hashes")
    require(bool(candidates), "existing runtime overlay is empty")
    for relative, path, size in sorted(candidates):
        require(path.stat().st_size == size, "overlay size changed before hashing")
        files.append({"path": relative, "size": size, "sha256": overlay_file_hash(path)})
    body = {"schema": "alice-public-runtime-overlay-inventory-v1",
            "directories": sorted(directories), "files": sorted(files, key=lambda row: row["path"])}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                           allow_nan=False).encode("utf-8")
    return {"sha256": sha256(canonical).hexdigest(),
            "file_count": len(files), "byte_count": sum(row["size"] for row in files)}


@dataclass(frozen=True)
class ExportInputs:
    experiment: Path
    experiment_sha256: str
    plan: Path
    plan_sha256: str
    cache: Path
    overlay: Path
    overlay_sha256: str
    exact_copied_input_bytes: int
    logical_feature_bytes: int
    complete_unique_inputs: int
    files: tuple[tuple[Path, str], ...]
    producer_runtime: dict


def inspect_inputs(*, experiment_path: str | Path, experiment_sha256: str,
                   plan_path: str | Path, plan_sha256: str, cache_directory: str | Path,
                   compute_root: str | Path, overlay_directory: str | Path,
                   overlay_sha256: str, producer_paths: dict | None = None,
                   own_paths: dict | None = None) -> ExportInputs:
    """Custody/resource preflight only; shared exporter proves source/tensor semantics.

    Optional path maps are isolated fixture seams, never exposed by the CLI.
    This preflight does not load tensor payloads or grant source acceptance.
    """
    modules()
    root = handoff._regular(compute_root, directory=True)
    exp = within(experiment_path, root)
    plan_file = within(plan_path, root)
    cache = within(cache_directory, root, directory=True)
    overlay = within(overlay_directory, root, directory=True)
    require(overlay == exp.parent.parent / "overlay", "reuse the original experiment overlay only")
    require(overlay_inventory(overlay)["sha256"] == handoff._pin(overlay_sha256),
            "existing overlay bytes differ from external pin")
    experiment = handoff._read(exp, handoff._pin(experiment_sha256))
    plan = handoff._read(plan_file, handoff._pin(plan_sha256))
    handoff._verify_seal(experiment)
    handoff._verify_seal(plan)
    require(experiment.get("state") == "PUBLIC_SEMANTIC_EXPERIMENT_UNQUALIFIED"
            and plan.get("state") == "PRECOMMITTED_PUBLIC_READOUT_UNQUALIFIED",
            "completed original public experiment/plan required")
    for field in ("private_identity_data", "final_payload_opened", "n0_approved"):
        require(experiment.get(field) is False and plan.get(field) is False,
                "only public TRAIN/DEV unqualified evidence is eligible")
    for field in ("upstream_gradient", "upstream_tensor_mutation", "personality_qualified"):
        require(experiment.get(field) is False, "historical frozen/unqualified boundary differs")
    require(experiment["binding"]["plan_sha256"] == plan_sha256
            and experiment["binding"]["plan_content_sha256"] == plan["receipt_sha256"],
            "original experiment/plan byte links differ")
    prepared_path = within(plan["preparation_path"], root)
    source_path = within(plan["source_admission_path"], root)
    prepared = handoff._read(prepared_path, plan["preparation_file_sha256"])
    source = handoff._read(source_path, plan["source_admission_file_sha256"])
    for value in (prepared, source):
        handoff._verify_seal(value)
    require(prepared["repository"] == handoff.MODEL and prepared["revision"] == handoff.REVISION
            and prepared["role"] == "personality" and prepared["state"] == "PREPARED_UNQUALIFIED",
            "wrong exact personality source/preparation")
    require(prepared["receipt_sha256"] == plan["preparation_sha256"]
            and source["receipt_sha256"] == plan["source_admission_sha256"],
            "source ancestry differs")
    require(source["private_identity_data"] is False
            and source["statistics"]["final_payload_opened"] is False,
            "source admission is not public TRAIN/DEV only")
    # Check redirects before the shared verifier follows any original reference.
    within(prepared["snapshot_path"], root, directory=True)
    for interface in prepared.get("interfaces", {}).values():
        within(interface["path"], root)
    for row in source["files"]:
        within(row["path"], root)
    clone_file = within(prepared["clone_receipt_path"], root)
    clone = handoff._read(clone_file)
    handoff._verify_seal(clone)
    require(clone["receipt_sha256"] == prepared["clone_receipt_sha256"], "clone ancestry differs")
    budget_file = within(experiment["pre_forward_feature_budget"]["path"], root)
    budget = handoff._read(budget_file, experiment["pre_forward_feature_budget"]["sha256"])
    handoff._verify_seal(budget)
    paths = producer.CODE_PATHS if producer_paths is None else producer_paths
    require(set(paths) == handoff._PRODUCER_NAMES, "exact six historical producer files required")
    declared = plan["code"]
    require(type(declared) is list and len(declared) == len(paths)
            and {row["name"] for row in declared} == set(paths), "historical code inventory differs")
    members = {exp, plan_file, prepared_path, source_path, clone_file, budget_file}
    for row in declared:
        path = within(paths[row["name"]], root)
        require(handoff._hash(path) == handoff._pin(row["sha256"]), "historical producer code differs")
        members.add(path)
    for path in (handoff._OWN_CODE if own_paths is None else own_paths).values():
        members.add(within(path, root))
    records = experiment["cache"]
    require(type(records) is list and len(records) > 0, "completed cache is missing")
    keys, cache_members = set(), set()
    logical = 0
    for row in records:
        key = handoff._pin(row["key"])
        require(key not in keys, "duplicate cache input")
        keys.add(key)
        require(type(row["logical_feature_bytes"]) is int and row["logical_feature_bytes"] > 0,
                "invalid logical feature bytes")
        logical += row["logical_feature_bytes"]
        for kind, suffix in (("tensor", ".pt"), ("metadata", ".json")):
            path = within(row[kind + "_path"], root)
            require(path.parent == cache and path.name == key + suffix,
                    "cache escaped original explicit directory")
            require(handoff._hash(path) == handoff._pin(row[kind + "_sha256"]), "cache bytes differ")
            cache_members.add(path)
    require(set(cache.iterdir()) == cache_members, "original cache has missing/extra members")
    require(type(budget["estimated_logical_feature_cache_bytes"]) is int
            and budget["estimated_logical_feature_cache_bytes"] == logical
            and type(plan["unique_complete_feature_inputs"]) is int
            and plan["unique_complete_feature_inputs"] == len(records), "cache/budget coverage differs")
    members |= cache_members
    files = tuple((path, handoff._hash(path)) for path in sorted(members))
    return ExportInputs(exp, experiment_sha256, plan_file, plan_sha256, cache, overlay,
                        overlay_sha256, sum(path.stat().st_size for path in members),
                        logical, len(records), files, plan["runtime"])


def storage_gate(inputs: ExportInputs, run_root: Path) -> dict:
    modules()
    root = handoff._regular(run_root, directory=True)
    require(not any(root.iterdir()), "handoff run requires a fresh empty directory")
    required = inputs.exact_copied_input_bytes + METADATA_SLACK_BYTES
    free = shutil.disk_usage(root).free
    require(free >= required, "insufficient filesystem free bytes for create-only package")
    return {"schema": "alice-public-feature-handoff-storage-preflight-v1",
            "exact_copied_input_bytes": inputs.exact_copied_input_bytes,
            "logical_feature_bytes": inputs.logical_feature_bytes,
            "metadata_and_filesystem_slack_bytes": METADATA_SLACK_BYTES,
            "required_free_bytes": required, "observed_filesystem_free_bytes": free,
            "filesystem_free_is_not_project_quota_proof": True,
            "checkpoint_bytes_copied": 0, "existing_cache_modified": False}


def verify_inputs_unchanged(inputs: ExportInputs) -> None:
    modules()
    cache = handoff._regular(inputs.cache, directory=True)
    expected_cache = {path for path, _ in inputs.files if path.parent == cache}
    require(set(cache.iterdir()) == expected_cache, "original cache membership changed during closure")
    require(all(handoff._hash(handoff._regular(path)) == digest for path, digest in inputs.files),
            "original input/code bytes changed during closure")
    require(overlay_inventory(inputs.overlay)["sha256"] == inputs.overlay_sha256,
            "runtime overlay changed during closure")


def git_state(repo: Path, revision: str) -> None:
    def git(*args):
        return subprocess.check_output(["git", "--git-dir=" + str(repo / ".git"),
            "--work-tree=" + str(repo), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == revision and not git("status", "--porcelain"),
            "exact clean launch code revision required")


def process_identity(host_uid: int, host_gid: int, *, status: Path = Path("/proc/self/status")) -> dict:
    """Record public P2 guest versus kernel IDs; owner proof is outside udocker.

    The public stage does not enter the private workload's outer USER namespace.
    No single-caller UID-map or private network assertion belongs here.
    """
    require(type(host_uid) is int and host_uid > 0
            and type(host_gid) is int and host_gid >= 0, "actual non-root host caller required")
    fields = {}
    for line in status.read_text().splitlines():
        if line.startswith(("Uid:", "Gid:")):
            key, values = line.split(":", 1)
            parsed = [int(field) for field in values.split()]
            require(len(parsed) == 4 and all(field >= 0 for field in parsed), "kernel ID fields differ")
            fields[key] = parsed
    require(set(fields) == {"Uid", "Gid"}, "kernel identity metadata unavailable")
    return {"host_uid": host_uid, "host_gid": host_gid,
            "kernel_status_uid_fields": fields["Uid"], "kernel_status_gid_fields": fields["Gid"],
            "owner_proof": "outer Slurm owner check; guest IDs are recorded only",
            "private_namespace_claim": False}


def close_handoff(inputs: ExportInputs, run_root: Path, *, context: dict) -> dict:
    modules()
    started = time.monotonic()
    storage = storage_gate(inputs, run_root)
    handoff._write(run_root / "storage-preflight.json", handoff._seal(storage))
    pins = handoff.export_public_features(experiment_path=inputs.experiment,
        expected_experiment_sha256=inputs.experiment_sha256, plan_path=inputs.plan,
        cache_directory=inputs.cache, output_directory=run_root / "package")
    imported = handoff.import_public_features(run_root / "package",
        expected_manifest_sha256=pins["manifest_sha256"],
        expected_closure_sha256=pins["closure_sha256"])
    require(imported.binding["consumer_retokenized"] is False
            and imported.binding["publisher_checkpoint_loaded"] is False
            and imported.binding["n0_approved"] is False
            and imported.binding["mechanics_only"] is False, "consumer boundary differs")
    verify_inputs_unchanged(inputs)
    # Torch/Transformers are already imported by producer verification, never
    # imported by the consumer. Checking module versions adds no model loading.
    actual = {"backend": "transformers", "dtype": "bfloat16",
        "torch_version": str(sys.modules["torch"].__version__),
        "transformers_version": str(sys.modules["transformers"].__version__)}
    require(actual == inputs.producer_runtime, "actual source runtime differs")
    if "code_revision" in context:
        git_state(REPO_ROOT, context["code_revision"])
    handoff.write_import_receipt(imported, run_root / "import-admission.json")
    import resource
    result = handoff._seal({"schema": "alice-public-feature-handoff-run-v1",
        "state": "PUBLIC_FEATURE_HANDOFF_VERIFIED_UNQUALIFIED", "context": context,
        "experiment_file_sha256": inputs.experiment_sha256, "plan_file_sha256": inputs.plan_sha256,
        "overlay_inventory_sha256": inputs.overlay_sha256, "producer_runtime": actual,
        "portable_file_pins": pins, "complete_unique_inputs": inputs.complete_unique_inputs,
        "storage_preflight": storage,
        "expanded_package_bytes": sum(path.stat().st_size for path in (run_root / "package").rglob("*") if path.is_file()),
        "import_admission_file_sha256": handoff._hash(run_root / "import-admission.json"),
        "elapsed_seconds": time.monotonic() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_measurement": "Linux process peak RSS, not Slurm allocation or job peak",
        "model_constructed": False, "consumer_retokenized": False, "training_run": False,
        "private_identity_data": False, "final_payload_opened": False,
        "upstream_tensor_mutation": False, "n0_approved": False,
        "personality_qualified": False, "qualification": "UNQUALIFIED"})
    handoff._write(run_root / "handoff-run.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("overlay-pin", help="read-only canonical byte inventory")
    inventory.add_argument("--overlay", type=Path, required=True)
    run = commands.add_parser("close")
    for name in ("experiment", "plan", "cache-directory", "compute-root", "overlay", "run-root"):
        run.add_argument("--" + name, type=Path, required=True)
    for name in ("experiment-sha256", "plan-sha256", "overlay-sha256", "code-revision"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--host-uid", type=int, required=True)
    run.add_argument("--host-gid", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "overlay-pin":
            print(json.dumps(overlay_inventory(args.overlay), sort_keys=True))
            return 0
        modules()
        require(sys.platform == "linux", "allocated Linux execution required")
        identity = process_identity(args.host_uid, args.host_gid)
        job = os.environ.get("SLURM_JOB_ID", "")
        require(re.fullmatch(r"[0-9]+", job) is not None
                and re.fullmatch(r"[0-9a-f]{40}", args.code_revision) is not None,
                "actual allocation and exact launch revision required")
        compute = handoff._regular(args.compute_root, directory=True)
        run_root = within(args.run_root, compute, directory=True)
        require(run_root == compute / "rayan-personality/runs" / ("public-feature-handoff-" + job),
                "fresh role-owned allocated output required")
        require(compute.stat().st_mode & 0o777 == 0o700
                and run_root.stat().st_mode & 0o777 == 0o700, "owner/private directory permissions required")
        require(REPO_ROOT == within(REPO_ROOT, compute, directory=True), "code must stay in owner compute root")
        git_state(REPO_ROOT, args.code_revision)
        inputs = inspect_inputs(experiment_path=args.experiment, experiment_sha256=args.experiment_sha256,
            plan_path=args.plan, plan_sha256=args.plan_sha256, cache_directory=args.cache_directory,
            compute_root=compute, overlay_directory=args.overlay, overlay_sha256=args.overlay_sha256)
        result = close_handoff(inputs, run_root, context={"host_owner": "mxrayan",
            "effective_uid": os.getuid(), "effective_gid": os.getgid(), "process_identity": identity,
            "slurm_job_id": job, "node": socket.gethostname(), "code_revision": args.code_revision,
            "launcher_file_sha256": handoff._hash(Path(__file__).resolve())})
        git_state(REPO_ROOT, args.code_revision)
        print(json.dumps(result, sort_keys=True))
    except (LaunchError, OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
