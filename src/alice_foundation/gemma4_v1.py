"""Offline custody and provenance for the shared Fable V1 Gemma 4 source.

This module does not download weights, import model code, train, or infer
behavior. It can stage one explicitly supplied tensor payload into a separate
copy, leaving the source unchanged. The resulting artifact is not a finished
or behaviorally qualified personal model.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4


MODEL = "google/gemma-4-12B"  # pretrained; never silently substitute -it
REVISION = "023679ed352de9bb66cc873c9009ce3482585c08"
SOURCE_SCHEMA = "alice-gemma4-v1-source-v1"
CLONE_SCHEMA = "alice-gemma4-v1-clone-v1"
STAGE_SCHEMA = "alice-gemma4-v1-tensor-stage-v1"
DERIVATIVE_SCHEMA = "alice-gemma4-v1-derivative-v1"
OPERATION_SCHEMA = "alice-gemma4-v1-transform-operation-v1"
PINNED_FILES = {
    ".gitattributes": (1624, "484fac0cb8b057eefe1992c8b72ac6e7438c7d17bd60c0e278b401c2190f7e72"),
    "README.md": (28340, "131e26f7f1fa69445dc4b0ab98a2251811f8c3128426bf09fcef6ac5ee16e7e4"),
    "config.json": (4383, "14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098"),
    "generation_config.json": (233, "02b56bd11e1cd1e363e701a85a2fd7fbaa2992ec3358c1cd7cc44ead7208f505"),
    "model.safetensors": (23919549408, "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a"),
    "processor_config.json": (1382, "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9"),
    "tokenizer.json": (32170070, "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6"),
    "tokenizer_config.json": (888, "522a38334973725dba8f7c645195b19dda0c284f403f43273f77837679ba2eab"),
}
_ROLE = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")


class FoundationError(ValueError):
    """A source, copy or derivative fails the pinned provenance contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest_file(path: Path) -> tuple[int, str]:
    before = path.stat()
    h = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after):
        raise FoundationError(f"file changed during verification: {path.name}")
    return after.st_size, h.hexdigest()


def _absolute_directory(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        raise FoundationError("snapshot path must be absolute and not a symlink")
    if not raw.is_dir():
        raise FoundationError(f"snapshot directory is missing: {raw}")
    return raw.resolve(strict=True)


def _list_regular_files(snapshot: Path) -> dict[str, Path]:
    found = {}
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot).as_posix()
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise FoundationError(f"snapshot contains nonregular entry: {relative}")
        if path.is_file():
            found[relative] = path
    return found


def _receipt(schema: str, **fields: object) -> dict:
    receipt = {"schema": schema, **fields}
    receipt["receipt_sha256"] = sha256(_canonical(receipt)).hexdigest()
    return receipt


def _write_receipt(receipt: dict, output: str | Path, *forbidden_dirs: Path) -> None:
    output = Path(output).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise FoundationError("receipt path must be absolute and not a symlink")
    resolved = output.resolve(strict=False)
    if any(resolved == root or root in resolved.parents for root in forbidden_dirs):
        raise FoundationError("receipt must live outside source and derivative snapshots")
    # Receipts are append-only evidence: never silently replace one.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(receipt).decode("utf-8") + "\n")


def verify_source(snapshot: str | Path) -> dict:
    """Fully rehash all eight pinned publisher files; return a sealed receipt."""
    source = _absolute_directory(snapshot)
    found = _list_regular_files(source)
    if set(found) != set(PINNED_FILES):
        raise FoundationError(
            f"source file set mismatch: missing={sorted(set(PINNED_FILES)-set(found))}, "
            f"unexpected={sorted(set(found)-set(PINNED_FILES))}")
    files = []
    for name, (size, digest) in sorted(PINNED_FILES.items()):
        if found[name].stat().st_size != size or _digest_file(found[name]) != (size, digest):
            raise FoundationError(f"publisher content mismatch: {name}")
        files.append({"path": name, "size": size, "sha256": digest})
    config_bytes = found["config.json"].read_bytes()
    if (len(config_bytes), sha256(config_bytes).hexdigest()) != PINNED_FILES["config.json"]:
        raise FoundationError("publisher configuration changed during verification")
    config = json.loads(config_bytes)
    if config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"] or \
            config.get("model_type") != "gemma4_unified":
        raise FoundationError("publisher configuration does not match Gemma 4 Unified")
    return _receipt(SOURCE_SCHEMA, repository=MODEL, revision=REVISION,
                    snapshot_path=str(source), files=files,
                    meaning="Exact publisher bytes only; no behavior or role qualification")


def seal_source(snapshot: str | Path, receipt_path: str | Path) -> dict:
    receipt = verify_source(snapshot)
    _write_receipt(receipt, receipt_path, Path(receipt["snapshot_path"]))
    return receipt


def read_receipt(path: str | Path, *, schemas: set[str] | None = None) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "receipt_sha256" not in raw:
        raise FoundationError("invalid receipt")
    digest = raw.pop("receipt_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or \
            sha256(_canonical(raw)).hexdigest() != digest:
        raise FoundationError("receipt digest mismatch")
    raw["receipt_sha256"] = digest
    if schemas is not None and raw.get("schema") not in schemas:
        raise FoundationError("unexpected receipt schema")
    if raw.get("repository") != MODEL or raw.get("revision") != REVISION:
        raise FoundationError("receipt is not for the pinned pretrained publisher source")
    return raw


def _verify_receipted_snapshot(receipt: dict) -> tuple[Path, dict[str, dict]]:
    """Rehash the parent's current bytes; a receipt does not freeze a directory."""
    if not isinstance(receipt.get("snapshot_path"), str):
        raise FoundationError("receipt lacks an absolute snapshot path")
    snapshot = _absolute_directory(receipt["snapshot_path"])
    if str(snapshot) != receipt["snapshot_path"]:
        raise FoundationError("receipt snapshot path changed")
    rows = receipt.get("files")
    if not isinstance(rows, list) or not rows:
        raise FoundationError("receipt lacks a file inventory")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
            raise FoundationError("invalid receipt file inventory")
        name = row["path"]
        if not isinstance(name, str) or not name or name.startswith("/") or \
                ".." in Path(name).parts or name in indexed or \
                type(row["size"]) is not int or row["size"] < 0 or \
                not isinstance(row["sha256"], str) or \
                not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise FoundationError("invalid receipt file entry")
        indexed[name] = row
    found = _list_regular_files(snapshot)
    if set(found) != set(indexed):
        raise FoundationError("parent snapshot file set changed since receipt")
    for name, path in found.items():
        if _digest_file(path) != (indexed[name]["size"], indexed[name]["sha256"]):
            raise FoundationError(f"parent snapshot changed since receipt: {name}")
    if receipt["schema"] == CLONE_SCHEMA and (set(indexed) != set(PINNED_FILES) or any(
            indexed.get(name) != {"path": name, "size": size, "sha256": digest}
            for name, (size, digest) in PINNED_FILES.items())):
        raise FoundationError("clone receipt does not match pinned publisher bytes")
    return snapshot, indexed


def verify_derivative(receipt_path: str | Path, *, snapshot: str | Path | None = None,
                      expected_role: str | None = None) -> dict:
    """Admit the exact modified artifact bytes, without claiming role quality.

    This checks an artifact before it is passed to a role trainer. Receipt
    digests detect accidental changes; they are not third-party signatures.
    """
    receipt = read_receipt(receipt_path, schemas={DERIVATIVE_SCHEMA})
    if expected_role is not None and receipt.get("role") != expected_role:
        raise FoundationError("prepared base has the wrong role")
    _validate_role(receipt.get("role", ""))
    if receipt.get("qualification") != "unqualified":
        raise FoundationError("unexpected derivative qualification")
    if receipt.get("upstream_weight_sha256") != PINNED_FILES["model.safetensors"][1]:
        raise FoundationError("upstream weight ancestry is not pinned")
    if not isinstance(receipt.get("parent_receipt_sha256"), str) or \
            not re.fullmatch(r"[0-9a-f]{64}", receipt["parent_receipt_sha256"]):
        raise FoundationError("derivative lacks parent receipt digest")
    if snapshot is not None and _absolute_directory(snapshot) != \
            _absolute_directory(receipt["snapshot_path"]):
        raise FoundationError("prepared base path differs from derivative receipt")
    directory, files = _verify_receipted_snapshot(receipt)
    for notice in ("README.md", ".gitattributes"):
        size, digest = PINNED_FILES[notice]
        if files.get(notice) != {"path": notice, "size": size, "sha256": digest}:
            raise FoundationError(f"publisher notice changed or missing: {notice}")
    weight = files.get("model.safetensors")
    if weight is None or weight["sha256"] == PINNED_FILES["model.safetensors"][1]:
        raise FoundationError("prepared base has pristine publisher weights")
    changed = receipt.get("changed_weight_files")
    if not isinstance(changed, list) or any(not isinstance(value, str) for value in changed) or \
            "model.safetensors" not in changed:
        raise FoundationError("derivative does not record a changed base tensor file")
    from .gemma4_inventory import InventoryError, inspect_checkpoint
    try:
        inspect_checkpoint(directory, model=MODEL, revision=REVISION,
                           hash_weights=False)
    except InventoryError as exc:
        raise FoundationError(f"prepared base structure invalid: {exc}") from exc
    return receipt


def _validate_role(role: str) -> None:
    if not _ROLE.fullmatch(role):
        raise FoundationError("role must be a lowercase slug of 2 to 64 characters")


def clone_verified_source(source: str | Path, destination: str | Path,
                          receipt_path: str | Path, *, role: str) -> dict:
    """Copy exact source into a role-local work directory and record its ancestry.

    The source is never modified. Treat the clone as immutable after receipt.
    It is a licensed starting checkpoint,
    not a trained role model or evidence of suppressed inherited behavior.
    """
    _validate_role(role)
    parent = verify_source(source)
    origin = Path(parent["snapshot_path"])
    target = Path(destination).expanduser()
    receipt_target = Path(receipt_path).expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists():
        raise FoundationError("destination must be a new absolute nonsymlink path")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise FoundationError("destination parent must be an existing nonsymlink directory")
    target = target.resolve(strict=False)
    if target == origin or origin in target.parents or target in origin.parents:
        raise FoundationError("source and destination may not nest")
    if receipt_target.exists() or receipt_target.is_symlink():
        raise FoundationError("receipt already exists")
    if shutil.disk_usage(target.parent).free < sum(size for size, _ in PINNED_FILES.values()):
        raise FoundationError("insufficient free space for the complete source clone")
    staging = target.parent / f".{target.name}.copy-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        for row in parent["files"]:
            name = row["path"]
            with (origin / name).open("rb") as input_stream, (staging / name).open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            if _digest_file(staging / name) != (row["size"], row["sha256"]):
                raise FoundationError(f"cloned file differs from pinned source: {name}")
        if target.exists():
            raise FoundationError("destination appeared during clone")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = _receipt(CLONE_SCHEMA, repository=MODEL, revision=REVISION,
                       role=role, snapshot_path=str(target),
                       parent_source_receipt_sha256=parent["receipt_sha256"],
                       files=parent["files"],
                       meaning="Verified editable copy of licensed source; no role qualification")
    try:
        _write_receipt(receipt, receipt_target, origin, target)
    except BaseException:
        shutil.rmtree(target)
        raise
    return receipt


def _copy_exact(source, destination, count: int, *digests) -> None:
    remaining = count
    while remaining:
        block = source.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise FoundationError("source or replacement ended during tensor patch")
        for digest in digests:
            digest.update(block)
        if destination is not None:
            destination.write(block)
        remaining -= len(block)


def _patch_operation(operation: object, parent_digest: str) -> dict:
    required = {"schema", "parent_receipt_sha256", "transform_id", "implementation_ref",
                "parameters", "intent", "evaluation_receipt"}
    if not isinstance(operation, dict) or set(operation) != required or \
            operation["schema"] != OPERATION_SCHEMA or \
            operation["parent_receipt_sha256"] != parent_digest or \
            not isinstance(operation["transform_id"], str) or \
            not _ROLE.fullmatch(operation["transform_id"]) or \
            not isinstance(operation["implementation_ref"], str) or \
            not operation["implementation_ref"].strip() or \
            not isinstance(operation["intent"], str) or \
            not operation["intent"].strip() or \
            operation["evaluation_receipt"] is not None:
        raise FoundationError("invalid or unrelated tensor patch operation")
    parameters = operation["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != {
            "tensor_name", "expected_tensor_sha256", "replacement_sha256", "replacement_size"} or \
            not isinstance(parameters["tensor_name"], str) or \
            any(not isinstance(parameters[field], str) or
                not re.fullmatch(r"[0-9a-f]{64}", parameters[field])
                for field in ("expected_tensor_sha256", "replacement_sha256")) or \
            type(parameters["replacement_size"]) is not int or \
            parameters["replacement_size"] <= 0 or \
            parameters["replacement_sha256"] == parameters["expected_tensor_sha256"]:
        raise FoundationError("tensor patch requires exact changed payload hashes and size")
    return parameters


def _compare_exact(left, right, count: int) -> None:
    remaining = count
    while remaining:
        a = left.read(min(8 * 1024 * 1024, remaining))
        b = right.read(len(a))
        if not a or a != b:
            raise FoundationError("candidate changed undeclared bytes outside named tensor payload")
        remaining -= len(a)


def stage_tensor_replacement(parent_receipt_path: str | Path,
                             replacement_payload: str | Path,
                             candidate_destination: str | Path,
                             operation_manifest_path: str | Path,
                             stage_receipt_path: str | Path) -> dict:
    """Stream exactly one named, same-shape tensor payload into a new candidate.

    Another process must produce and justify the replacement bytes. This tool
    checks expected before/after payload hashes, the entire parent file and
    candidate structure. It supplies no tensor choice or replacement values.
    Pass the candidate, operation manifest and stage receipt to
    materialize_transform next.
    """
    parent = read_receipt(parent_receipt_path, schemas={CLONE_SCHEMA, DERIVATIVE_SCHEMA})
    origin, parent_files = _verify_receipted_snapshot(parent)
    manifest_bytes = Path(operation_manifest_path).read_bytes()
    operation = json.loads(manifest_bytes)
    parameters = _patch_operation(operation, parent["receipt_sha256"])
    target = Path(candidate_destination).expanduser()
    payload = Path(replacement_payload).expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists() or \
            not target.parent.is_dir() or target.parent.is_symlink():
        raise FoundationError("candidate must be a new absolute directory")
    if not payload.is_absolute() or payload.is_symlink() or not payload.is_file():
        raise FoundationError("replacement must be an absolute regular nonsymlink file")
    target = target.resolve(strict=False)
    payload = payload.resolve(strict=True)
    stage_receipt_target = Path(stage_receipt_path).expanduser()
    if not stage_receipt_target.is_absolute() or stage_receipt_target.is_symlink() or \
            stage_receipt_target.exists() or \
            any(stage_receipt_target.resolve(strict=False) == root or
                root in stage_receipt_target.resolve(strict=False).parents
                for root in (origin, target)):
        raise FoundationError("stage receipt must be a new path outside checkpoint snapshots")
    if target == origin or target in origin.parents or origin in target.parents or \
            origin == payload or origin in payload.parents or \
            target == payload or target in payload.parents:
        raise FoundationError("candidate, replacement and parent may not nest")
    weight = origin / "model.safetensors"
    from .gemma4_inventory import (_read_header, _validate_tensors, InventoryError,
                                    inspect_checkpoint)
    try:
        header, data_start, size, _ = _read_header(weight)
        tensors = _validate_tensors(header, size - data_start)
    except InventoryError as exc:
        raise FoundationError(f"parent tensor structure invalid: {exc}") from exc
    match = [row for row in tensors if row["name"] == parameters["tensor_name"]]
    if len(match) != 1 or match[0]["bytes"] != parameters["replacement_size"]:
        raise FoundationError("tensor missing or replacement size differs from its shape")
    if _digest_file(payload) != (match[0]["bytes"], parameters["replacement_sha256"]):
        raise FoundationError("replacement payload hash or size mismatch")
    if shutil.disk_usage(target.parent).free < sum(row["size"] for row in parent_files.values()):
        raise FoundationError("insufficient free space for staged tensor replacement")
    staging = target.parent / f".{target.name}.patch-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        first = data_start + match[0]["data_offsets"][0]
        count = match[0]["bytes"]
        source_digest = sha256()
        old_digest = sha256()
        new_digest = sha256()
        with weight.open("rb") as source, \
                (staging / "model.safetensors").open("xb") as output, \
                payload.open("rb") as replacement:
            _copy_exact(source, output, first, source_digest)
            _copy_exact(source, None, count, source_digest, old_digest)
            _copy_exact(replacement, output, count, new_digest)
            _copy_exact(source, output, size - first - count, source_digest)
            if source.read(1) or replacement.read(1):
                raise FoundationError("source or replacement grew during tensor patch")
        if source_digest.hexdigest() != parent_files["model.safetensors"]["sha256"] or \
                old_digest.hexdigest() != parameters["expected_tensor_sha256"] or \
                new_digest.hexdigest() != parameters["replacement_sha256"]:
            raise FoundationError("parent or tensor payload changed during patch")
        for name, row in sorted(parent_files.items()):
            if name == "model.safetensors":
                continue
            output = staging / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with (origin / name).open("rb") as source, output.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if _digest_file(output) != (row["size"], row["sha256"]):
                raise FoundationError(f"parent file changed during patch: {name}")
        try:
            inspect_checkpoint(staging, model=MODEL, revision=REVISION,
                               hash_weights=False)
        except InventoryError as exc:
            raise FoundationError(f"staged tensor structure invalid: {exc}") from exc
        candidate_size, candidate_digest = _digest_file(staging / "model.safetensors")
        if candidate_size != size or candidate_digest == parent_files["model.safetensors"]["sha256"]:
            raise FoundationError("tensor patch yielded unchanged or truncated checkpoint")
        if target.exists():
            raise FoundationError("candidate destination appeared during patch")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = _receipt(STAGE_SCHEMA, repository=MODEL, revision=REVISION,
                       role=parent["role"], candidate_path=str(target),
                       parent_receipt_sha256=parent["receipt_sha256"],
                       operation_manifest_sha256=sha256(manifest_bytes).hexdigest(),
                       tensor_name=parameters["tensor_name"],
                       parent_weight_sha256=parent_files["model.safetensors"]["sha256"],
                       expected_tensor_sha256=old_digest.hexdigest(),
                       replacement_sha256=new_digest.hexdigest(),
                       replacement_size=count,
                       candidate_weight_sha256=candidate_digest,
                       qualification="unqualified")
    try:
        _write_receipt(receipt, stage_receipt_target, origin, target)
    except BaseException:
        shutil.rmtree(target)
        raise
    return receipt


def materialize_transform(parent_receipt_path: str | Path,
                          transformed_snapshot: str | Path, destination: str | Path,
                          operation_manifest_path: str | Path,
                          receipt_path: str | Path, *,
                          stage_receipt_path: str | Path) -> dict:
    """Copy an externally transformed candidate into a verified role artifact.

    Accept only a staged, explicit one-tensor same-shape payload replacement.
    Independently compare all other bytes against the immutable parent. The
    receipt establishes no behavioral efficacy or role qualification.
    """
    parent = read_receipt(parent_receipt_path, schemas={CLONE_SCHEMA, DERIVATIVE_SCHEMA})
    _validate_role(parent.get("role", ""))
    origin, parent_files = _verify_receipted_snapshot(parent)
    candidate = _absolute_directory(transformed_snapshot)
    target = Path(destination).expanduser()
    receipt_target = Path(receipt_path).expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists() or \
            not target.parent.is_dir() or target.parent.is_symlink():
        raise FoundationError("destination must be a new absolute directory under an existing parent")
    target = target.resolve(strict=False)
    if any(a == b or a in b.parents or b in a.parents for a, b in
           ((origin, candidate), (origin, target), (candidate, target))):
        raise FoundationError("parent, candidate and destination may not nest")
    if receipt_target.exists() or receipt_target.is_symlink():
        raise FoundationError("receipt already exists")
    if not receipt_target.is_absolute() or receipt_target.is_symlink() or \
            any(receipt_target.resolve(strict=False) == root or root in
                receipt_target.resolve(strict=False).parents
                for root in (origin, candidate, target)):
        raise FoundationError("receipt must be outside checkpoint snapshots")
    operation_bytes = Path(operation_manifest_path).read_bytes()
    operation = json.loads(operation_bytes)
    parameters = _patch_operation(operation, parent["receipt_sha256"])
    stage = read_receipt(stage_receipt_path, schemas={STAGE_SCHEMA})
    if stage.get("parent_receipt_sha256") != parent["receipt_sha256"] or \
            stage.get("candidate_path") != str(candidate) or \
            stage.get("role") != parent["role"] or \
            stage.get("qualification") != "unqualified" or \
            stage.get("operation_manifest_sha256") != sha256(operation_bytes).hexdigest() or \
            stage.get("tensor_name") != parameters["tensor_name"] or \
            stage.get("expected_tensor_sha256") != parameters["expected_tensor_sha256"] or \
            stage.get("replacement_sha256") != parameters["replacement_sha256"] or \
            stage.get("replacement_size") != parameters["replacement_size"] or \
            stage.get("parent_weight_sha256") != parent_files["model.safetensors"]["sha256"]:
        raise FoundationError("stage receipt does not bind this parent, candidate and patch")
    found = _list_regular_files(candidate)
    if set(found) != set(parent_files) or "model.safetensors" not in found:
        raise FoundationError("candidate changed undeclared nonweight files or file set")
    # Preserve the publisher license and attribution in every materialized base.
    for notice in ("README.md", ".gitattributes"):
        size, digest = PINNED_FILES[notice]
        if notice not in found or _digest_file(found[notice]) != (size, digest):
            raise FoundationError(f"publisher notice changed or missing: {notice}")
    from .gemma4_inventory import InventoryError, inspect_checkpoint
    try:
        inspect_checkpoint(candidate, model=MODEL, revision=REVISION,
                           hash_weights=False)
    except InventoryError as exc:
        raise FoundationError(f"derivative structure invalid: {exc}") from exc
    files = []
    for name, path in sorted(found.items()):
        size, digest = _digest_file(path)
        files.append({"path": name, "size": size, "sha256": digest})
    by_name = {row["path"]: row for row in files}
    if any(by_name[name] != row for name, row in parent_files.items()
           if name != "model.safetensors"):
        raise FoundationError("candidate changed undeclared nonweight file bytes")
    if by_name["model.safetensors"]["sha256"] != stage.get("candidate_weight_sha256"):
        raise FoundationError("candidate weight digest differs from stage receipt")
    if by_name["model.safetensors"]["sha256"] == PINNED_FILES["model.safetensors"][1]:
        raise FoundationError("derivative has pristine publisher weights")
    from .gemma4_inventory import _read_header, _validate_tensors
    original_weight = origin / "model.safetensors"
    candidate_weight = candidate / "model.safetensors"
    try:
        old_header, old_data_start, old_size, old_header_sha = _read_header(original_weight)
        new_header, new_data_start, new_size, new_header_sha = _read_header(candidate_weight)
        tensors = _validate_tensors(old_header, old_size - old_data_start)
    except InventoryError as exc:
        raise FoundationError(f"tensor patch structure invalid: {exc}") from exc
    if old_size != new_size or old_data_start != new_data_start or \
            old_header_sha != new_header_sha or old_header != new_header:
        raise FoundationError("candidate changed undeclared tensor header or checkpoint length")
    match = [row for row in tensors if row["name"] == parameters["tensor_name"]]
    if len(match) != 1 or match[0]["bytes"] != parameters["replacement_size"]:
        raise FoundationError("declared tensor is missing or replacement length differs")
    start = old_data_start + match[0]["data_offsets"][0]
    count = match[0]["bytes"]
    before_digest, after_digest = sha256(), sha256()
    with original_weight.open("rb") as old_stream, candidate_weight.open("rb") as new_stream:
        _compare_exact(old_stream, new_stream, start)
        _copy_exact(old_stream, None, count, before_digest)
        _copy_exact(new_stream, None, count, after_digest)
        _compare_exact(old_stream, new_stream, old_size - start - count)
        if old_stream.read(1) or new_stream.read(1):
            raise FoundationError("candidate has undeclared trailing weight bytes")
    if before_digest.hexdigest() != parameters["expected_tensor_sha256"] or \
            after_digest.hexdigest() != parameters["replacement_sha256"]:
        raise FoundationError("named tensor payload differs from declared before/after hashes")
    if shutil.disk_usage(target.parent).free < sum(row["size"] for row in files):
        raise FoundationError("insufficient free space for transformed artifact")
    staging = target.parent / f".{target.name}.copy-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        for row in files:
            output = staging / row["path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            with found[row["path"]].open("rb") as input_stream, output.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            if _digest_file(output) != (row["size"], row["sha256"]):
                raise FoundationError(f"candidate changed during materialization: {row['path']}")
        if target.exists():
            raise FoundationError("destination appeared during materialization")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = _receipt(DERIVATIVE_SCHEMA, repository=MODEL, revision=REVISION,
                       role=parent["role"], snapshot_path=str(target),
                       parent_receipt_sha256=parent["receipt_sha256"],
                       upstream_weight_sha256=PINNED_FILES["model.safetensors"][1],
                       transform_id=operation["transform_id"],
                       operation_manifest_sha256=sha256(operation_bytes).hexdigest(),
                       stage_receipt_sha256=stage["receipt_sha256"],
                       changed_weight_files=["model.safetensors"],
                       changed_tensor=parameters["tensor_name"],
                       old_tensor_sha256=before_digest.hexdigest(),
                       new_tensor_sha256=after_digest.hexdigest(), files=files,
                       qualification="unqualified",
                       meaning="Changed weight bytes and ancestry only; behavioral efficacy unmeasured")
    try:
        _write_receipt(receipt, receipt_target, origin, candidate, target)
    except BaseException:
        shutil.rmtree(target)
        raise
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-source")
    verify.add_argument("snapshot", type=Path)
    verify.add_argument("receipt", type=Path)
    clone = commands.add_parser("clone")
    clone.add_argument("source", type=Path)
    clone.add_argument("destination", type=Path)
    clone.add_argument("receipt", type=Path)
    clone.add_argument("--role", required=True)
    transform = commands.add_parser("materialize-transform")
    transform.add_argument("parent_receipt", type=Path)
    transform.add_argument("candidate", type=Path)
    transform.add_argument("destination", type=Path)
    transform.add_argument("operation_manifest", type=Path)
    transform.add_argument("receipt", type=Path)
    transform.add_argument("--stage-receipt", required=True, type=Path)
    derivative = commands.add_parser("verify-derivative")
    derivative.add_argument("receipt", type=Path)
    derivative.add_argument("--snapshot", type=Path)
    derivative.add_argument("--role")
    patch = commands.add_parser("stage-tensor-replacement")
    patch.add_argument("parent_receipt", type=Path)
    patch.add_argument("replacement_payload", type=Path)
    patch.add_argument("candidate", type=Path)
    patch.add_argument("operation_manifest", type=Path)
    patch.add_argument("stage_receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-source":
            result = seal_source(args.snapshot, args.receipt)
        elif args.command == "clone":
            result = clone_verified_source(args.source, args.destination,
                                           args.receipt, role=args.role)
        elif args.command == "materialize-transform":
            result = materialize_transform(args.parent_receipt, args.candidate,
                                           args.destination, args.operation_manifest,
                                           args.receipt, stage_receipt_path=args.stage_receipt)
        elif args.command == "stage-tensor-replacement":
            result = stage_tensor_replacement(args.parent_receipt, args.replacement_payload,
                                              args.candidate, args.operation_manifest,
                                              args.stage_receipt)
        else:
            result = verify_derivative(args.receipt, snapshot=args.snapshot,
                                       expected_role=args.role)
    except (FoundationError, FileExistsError, FileNotFoundError,
            PermissionError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.command == "stage-tensor-replacement":
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps({"schema": result["schema"],
                          "receipt_sha256": result["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
