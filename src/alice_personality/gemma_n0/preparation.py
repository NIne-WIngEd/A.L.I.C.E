"""Offline preparation and admission for the unchanged personality N0 base.

Preparation binds a personality-role clone, this feature-only implementation,
and explicitly selected runtime versions. It neither downloads nor trains a
model, and PREPARED_UNQUALIFIED never means personality behavior is approved.
Receipts detect changed bytes; they are not signatures or filesystem locks.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Mapping

from src.alice_foundation import gemma4_v1 as foundation


SCHEMA = "alice-personality-gemma-n0-preparation-v2"
ROLE = "personality"
STATE = "PREPARED_UNQUALIFIED"
IMPLEMENTATION_PATHS = {
    "preparation.py": Path(__file__),
    "backbone.py": Path(__file__).with_name("backbone.py"),
}
FEATURE_SCOPE = {
    "input": "source_token_ids_and_attention_mask_with_optional_processor_media",
    "output": "token_aligned_all_text_hidden_states_and_final_state",
    "hidden_state_layout": "embedding_then_decoder_layers_with_final_normalized_state",
    "all_hidden_states_required": True,
    "personality_judgments": False,
    "text_generation": False,
    "personality_authority": ["N1", "N2", "N3", "EIPM"],
    "upstream_tensor_mutation": False,
}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CONTRACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_RUNTIME_KEYS = {"backend", "dtype", "torch_version", "transformers_version"}
_RECEIPT_KEYS = {
    "schema", "repository", "revision", "role", "state", "snapshot_path",
    "clone_receipt_path", "clone_receipt_sha256", "files", "implementation",
    "runtime", "interfaces", "feature_scope", "model_geometry", "behavior_qualification",
    "receipt_sha256",
}


class PreparationError(ValueError):
    """The preparation or current artifact does not satisfy N0 admission."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _regular_file(path: str | Path, *, label: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise PreparationError(f"{label} must name a file path")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise PreparationError(f"{label} must be an absolute regular nonsymlink file")
    return candidate.resolve(strict=True)


def _file_digest(path: Path) -> str:
    before = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise PreparationError(f"file changed during preparation: {path.name}")
    return digest.hexdigest()


def _runtime(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != _RUNTIME_KEYS:
        raise PreparationError("runtime requires backend, dtype and exact torch/transformers versions")
    if value["backend"] != "transformers" or not isinstance(value["dtype"], str) \
            or value["dtype"] not in {
            "bfloat16", "float32", "float16"}:
        raise PreparationError("unsupported runtime backend or dtype")
    for key in ("torch_version", "transformers_version"):
        version = value[key]
        if not isinstance(version, str) or not version.strip() or version != version.strip() \
                or any(char.isspace() for char in version):
            raise PreparationError(f"{key} must name one exact nonempty version")
    return dict(value)


def _implementation() -> list[dict]:
    if set(IMPLEMENTATION_PATHS) != {"preparation.py", "backbone.py"}:
        raise PreparationError("personality preparation requires both implementation files")
    return [{"path": name, "sha256": _file_digest(
        _regular_file(path, label=f"implementation {name}"))}
        for name, path in sorted(IMPLEMENTATION_PATHS.items())]


def _model_geometry(snapshot: str | Path) -> dict:
    """Resolve the complete text-state geometry from verified publisher config.

    No custom-backbone dimension or Transformers default substitutes for a
    missing config field. The clone verifier is responsible for custody of
    these config bytes; this binds their representation meaning as well.
    """
    config_path = _regular_file(Path(snapshot) / "config.json", label="publisher config")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PreparationError("invalid publisher configuration JSON") from exc
    if not isinstance(config, dict) or config.get("model_type") != "gemma4_unified":
        raise PreparationError("publisher configuration lacks the Gemma 4 unified schema")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "gemma4_unified_text":
        raise PreparationError("publisher configuration lacks the Gemma 4 unified text schema")
    geometry = {"model_type": config["model_type"], "text_model_type": text["model_type"]}
    for field in ("hidden_size", "num_hidden_layers", "max_position_embeddings"):
        value = text.get(field)
        if type(value) is not int or value < 1:
            raise PreparationError(f"publisher text configuration requires explicit positive {field}")
        geometry[field] = value
    # Transformers 5.17 text capture returns the layer-zero embedding input
    # and every decoder layer, tying the last entry to normalized final state.
    geometry["hidden_state_count"] = geometry["num_hidden_layers"] + 1
    return geometry


def _contract(value: Mapping | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"contract_id", "path"} \
            or not isinstance(value["contract_id"], str) \
            or not _CONTRACT_ID.fullmatch(value["contract_id"]):
        raise PreparationError("resolved interface binding requires contract_id and path")
    path = _regular_file(value["path"], label="interface contract")
    return {"contract_id": value["contract_id"], "path": str(path),
            "sha256": _file_digest(path)}


def _clone(path: str | Path, *, snapshot: str | Path | None = None) -> tuple[Path, dict]:
    clone_path = _regular_file(path, label="clone receipt")
    try:
        # Reject derivatives before invoking any derivative inventory machinery.
        foundation.read_receipt(clone_path, schemas={foundation.CLONE_SCHEMA})
        clone = foundation.verify_clone(clone_path, snapshot=snapshot, expected_role=ROLE)
    except (foundation.FoundationError, OSError, ValueError, TypeError) as exc:
        raise PreparationError(f"personality clone admission failed: {exc}") from exc
    if clone.get("repository") != foundation.MODEL or clone.get("revision") != foundation.REVISION:
        raise PreparationError("personality clone does not match the exact publisher pin")
    return clone_path, clone


def _new_receipt_path(path: str | Path, snapshot: Path) -> Path:
    output = Path(path).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise PreparationError("preparation receipt must use an absolute nonsymlink path")
    output = output.resolve(strict=False)
    if output == snapshot or snapshot in output.parents:
        raise PreparationError("preparation receipt must live outside the checkpoint snapshot")
    if output.exists():
        raise PreparationError("preparation receipt already exists; use a new artifact path")
    if not output.parent.is_dir():
        raise PreparationError("preparation receipt parent directory is missing")
    return output


def prepare(clone_receipt_path: str | Path, output_receipt_path: str | Path, *,
            snapshot: str | Path | None = None, runtime: dict | None = None,
            input_contract: Mapping | None = None,
            output_contract: Mapping | None = None) -> dict:
    """Write one append-only, unqualified receipt for a feature-only N0 clone.

    Runtime versions must be supplied explicitly; this offline operation does
    not establish that those packages can load or execute the actual model.
    Optional interface bindings hash existing resolved contracts, without
    interpreting their contents or inventing an ACFP/IDP contract.
    """
    selected_runtime = _runtime(runtime)
    clone_path, clone = _clone(clone_receipt_path, snapshot=snapshot)
    output = _new_receipt_path(output_receipt_path, Path(clone["snapshot_path"]))
    receipt = {
        "schema": SCHEMA,
        "repository": foundation.MODEL,
        "revision": foundation.REVISION,
        "role": ROLE,
        "state": STATE,
        "snapshot_path": clone["snapshot_path"],
        "clone_receipt_path": str(clone_path),
        "clone_receipt_sha256": clone["receipt_sha256"],
        "files": clone["files"],
        "implementation": _implementation(),
        "runtime": selected_runtime,
        "interfaces": {"input": _contract(input_contract), "output": _contract(output_contract)},
        "feature_scope": json.loads(_canonical(FEATURE_SCOPE)),
        "model_geometry": _model_geometry(clone["snapshot_path"]),
        "behavior_qualification": None,
    }
    receipt["receipt_sha256"] = sha256(_canonical(receipt)).hexdigest()
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(receipt).decode("utf-8") + "\n")
    except FileExistsError as exc:
        raise PreparationError("preparation receipt already exists; use a new artifact path") from exc
    return receipt


def verify_prepared(prepared_receipt_path: str | Path, *,
                    snapshot: str | Path | None = None,
                    expected_runtime: dict | None = None) -> dict:
    """Recheck clone bytes, code and interface bindings before model loading.

    Passing proves custody and declared runtime consistency only. A loader
    must also compare its actual runtime; supplied expected_runtime makes that
    exact comparison here. This function never approves personality behavior.
    """
    path = _regular_file(prepared_receipt_path, label="preparation receipt")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PreparationError("invalid preparation receipt JSON") from exc
    if not isinstance(raw, dict) or set(raw) != _RECEIPT_KEYS:
        raise PreparationError("invalid preparation receipt fields")
    supplied_digest = raw.pop("receipt_sha256")
    try:
        actual_digest = sha256(_canonical(raw)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise PreparationError("invalid preparation receipt values") from exc
    if not isinstance(supplied_digest, str) or not _DIGEST.fullmatch(supplied_digest) \
            or actual_digest != supplied_digest:
        raise PreparationError("preparation receipt digest mismatch")
    raw["receipt_sha256"] = supplied_digest
    if raw["schema"] != SCHEMA or raw["role"] != ROLE:
        raise PreparationError("preparation receipt has the wrong schema or role")
    if raw["repository"] != foundation.MODEL or raw["revision"] != foundation.REVISION:
        raise PreparationError("preparation receipt does not match the exact publisher pin")
    if raw["state"] != STATE or raw["behavior_qualification"] is not None \
            or _canonical(raw["feature_scope"]) != _canonical(FEATURE_SCOPE):
        raise PreparationError("preparation cannot assert personality or behavioral qualification")
    selected_runtime = _runtime(raw["runtime"])
    if expected_runtime is not None and selected_runtime != _runtime(expected_runtime):
        raise PreparationError("loaded runtime does not match the prepared runtime")
    clone_path, clone = _clone(raw["clone_receipt_path"], snapshot=snapshot)
    if str(clone_path) != raw["clone_receipt_path"] \
            or clone["snapshot_path"] != raw["snapshot_path"] \
            or clone["receipt_sha256"] != raw["clone_receipt_sha256"] \
            or clone["files"] != raw["files"]:
        raise PreparationError("preparation clone binding changed")
    checkpoint = Path(clone["snapshot_path"])
    if path == checkpoint or checkpoint in path.parents:
        raise PreparationError("preparation receipt must live outside the checkpoint snapshot")
    if _canonical(raw["model_geometry"]) != _canonical(_model_geometry(checkpoint)):
        raise PreparationError("prepared text representation geometry differs from publisher config")
    if raw["implementation"] != _implementation():
        raise PreparationError("personality implementation changed since preparation")
    interfaces = raw["interfaces"]
    if not isinstance(interfaces, dict) or set(interfaces) != {"input", "output"}:
        raise PreparationError("invalid preparation interface bindings")
    for binding in interfaces.values():
        if binding is None:
            continue
        if not isinstance(binding, dict) or set(binding) != {"contract_id", "path", "sha256"} \
                or binding != _contract({"contract_id": binding["contract_id"],
                                         "path": binding["path"]}):
            raise PreparationError("interface contract changed since preparation")
    return raw


def _binding_args(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(f"--{name}-contract-id")
    parser.add_argument(f"--{name}-contract-path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="prepare an unchanged personality role clone")
    create.add_argument("clone_receipt")
    create.add_argument("output_receipt")
    create.add_argument("--snapshot")
    create.add_argument("--runtime-json", required=True,
                        help="JSON file naming exact backend, dtype and package versions")
    for name in ("input", "output"):
        _binding_args(create, name)
    verify = commands.add_parser("verify-prepared", help="rehash preparation bindings")
    verify.add_argument("prepared_receipt")
    verify.add_argument("--snapshot")
    verify.add_argument("--runtime-json", help="compare with the actual selected runtime JSON")
    args = parser.parse_args(argv)
    try:
        runtime = json.loads(Path(args.runtime_json).read_text(encoding="utf-8")) \
            if args.runtime_json else None
        if args.command == "prepare":
            bindings = {}
            for name in ("input", "output"):
                identity = getattr(args, f"{name}_contract_id")
                path = getattr(args, f"{name}_contract_path")
                if bool(identity) != bool(path):
                    raise PreparationError(f"{name} contract requires both ID and path")
                bindings[f"{name}_contract"] = {"contract_id": identity, "path": path} \
                    if identity else None
            receipt = prepare(args.clone_receipt, args.output_receipt,
                              snapshot=args.snapshot, runtime=runtime, **bindings)
        else:
            receipt = verify_prepared(args.prepared_receipt, snapshot=args.snapshot,
                                      expected_runtime=runtime)
        print(json.dumps({"schema": receipt["schema"], "state": receipt["state"],
                          "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
        return 0
    except (PreparationError, OSError, ValueError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
