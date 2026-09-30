"""Read-only, bounded structural inventory of a Gemma 4 safetensors checkpoint.

This tool never imports model code, loads a model, changes weights, or infers a
behavior from a tensor name. Name-based modality/module groups are inventory
labels only. Behavioral influence requires separate intervention and evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any


SCHEMA = "alice-gemma4-v1-checkpoint-inventory-v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
HASH_CHUNK_BYTES = 8 * 1024 * 1024
DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "U16": 16, "I16": 16,
    "U32": 32, "I32": 32, "U64": 64, "I64": 64,
    "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3FN": 8, "F8_E4M3FNUZ": 8, "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8, "F8_E8M0": 8,
}
CONFIG_DTYPE = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
LAYER_RE = re.compile(r"^(.*\.layers)\.(\d+)\.(.+)$")


class InventoryError(ValueError):
    """The local checkpoint or config failed structural validation."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise InventoryError(f"non-finite JSON constant: {value}")


def _json_bytes(content: bytes) -> Any:
    try:
        return json.loads(content, object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError("invalid UTF-8 JSON") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise InventoryError(f"not a regular file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> tuple[dict[str, Any], int, int, str]:
    size = path.stat().st_size
    with path.open("rb") as source:
        length_bytes = source.read(8)
        if len(length_bytes) != 8:
            raise InventoryError("safetensors header length is missing")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if not 0 < header_length <= MAX_HEADER_BYTES or header_length + 8 > size:
            raise InventoryError("safetensors header length is invalid or incomplete")
        header_bytes = source.read(header_length)
        if len(header_bytes) != header_length:
            raise InventoryError("safetensors header is truncated")
    header = _json_bytes(header_bytes)
    if not isinstance(header, dict) or not header:
        raise InventoryError("safetensors header must be a nonempty JSON object")
    has_metadata = "__metadata__" in header
    metadata = header.pop("__metadata__", None)
    if has_metadata and (not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items())):
        raise InventoryError("safetensors metadata must contain string values")
    if not header:
        raise InventoryError("safetensors checkpoint has no tensors")
    return header, header_length + 8, size, sha256(length_bytes + header_bytes).hexdigest()


def _validate_tensors(header: dict[str, Any], data_bytes: int) -> list[dict[str, Any]]:
    tensors = []
    for name, spec in header.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict) or \
                set(spec) != {"dtype", "shape", "data_offsets"}:
            raise InventoryError(f"invalid tensor entry: {name!r}")
        dtype, shape, offsets = spec["dtype"], spec["shape"], spec["data_offsets"]
        if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
            raise InventoryError(f"unsupported dtype for {name}: {dtype!r}")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise InventoryError(f"invalid shape for {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
                type(offset) is not int or offset < 0 for offset in offsets):
            raise InventoryError(f"invalid offsets for {name}")
        start, end = offsets
        elements = math.prod(shape)
        byte_count = (elements * DTYPE_BITS[dtype] + 7) // 8
        if end < start or end - start != byte_count or end > data_bytes:
            raise InventoryError(f"offset or shape byte count differs for {name}")
        tensors.append({"name": name, "dtype": dtype, "shape": shape,
                        "elements": elements, "data_offsets": offsets,
                        "bytes": byte_count})
    cursor = 0
    for record in sorted(tensors, key=lambda row: (row["data_offsets"][0],
                                                     row["data_offsets"][1], row["name"])):
        start, end = record["data_offsets"]
        if start != cursor:
            raise InventoryError(f"tensor data has an overlap or gap near {record['name']}")
        cursor = end
    if cursor != data_bytes:
        raise InventoryError("tensor data has unlisted trailing bytes")
    return sorted(tensors, key=lambda row: row["name"])


def _config_int(config: dict[str, Any], field: str) -> int | None:
    value = config.get(field)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise InventoryError(f"config {field} must be a positive integer")
    return value


def _validate_config(config: dict[str, Any], tensors: list[dict[str, Any]]) -> list[str]:
    if config.get("model_type") != "gemma4_unified":
        raise InventoryError("config is not a Gemma 4 unified checkpoint")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise InventoryError("config has no text_config")
    by_name = {record["name"]: record for record in tensors}
    checks = []
    hidden = _config_int(text, "hidden_size")
    vocab = _config_int(text, "vocab_size")
    embed = by_name.get("model.language_model.embed_tokens.weight")
    if not embed or hidden is None or vocab is None or embed["shape"] != [vocab, hidden]:
        raise InventoryError("token embedding shape differs from text config")
    checks.append("text embedding shape matches vocabulary and hidden size")
    norm = by_name.get("model.language_model.norm.weight")
    if norm is None or norm["shape"] != [hidden]:
        raise InventoryError("final language norm shape differs from hidden size")
    checks.append("final language norm shape matches hidden size")
    expected_layers = _config_int(text, "num_hidden_layers")
    layer_types = text.get("layer_types")
    if expected_layers is None or (layer_types is not None and (
            not isinstance(layer_types, list) or len(layer_types) != expected_layers)):
        raise InventoryError("config layer count or layer_types is invalid")
    layer_numbers = {int(match.group(2)) for row in tensors
                     if (match := LAYER_RE.match(row["name"])) and
                     match.group(1) == "model.language_model.layers"}
    if layer_numbers != set(range(expected_layers)):
        raise InventoryError("language layer indices differ from config")
    checks.append("language layer indices match config")
    dtype = CONFIG_DTYPE.get(config.get("dtype"))
    if dtype and any(row["dtype"] != dtype for row in tensors):
        raise InventoryError("tensor dtypes differ from config dtype")
    if dtype:
        checks.append("tensor dtype matches config")
    audio = config.get("audio_config")
    projection = by_name.get("model.embed_audio.embedding_projection.weight")
    if isinstance(audio, dict) and projection is not None:
        audio_hidden = _config_int(audio, "hidden_size")
        if audio_hidden and projection["shape"] != [hidden, audio_hidden]:
            raise InventoryError("audio projection shape differs from config")
        checks.append("audio projection matches configured dimensions")
    vision = config.get("vision_config")
    projection = by_name.get("model.embed_vision.embedding_projection.weight")
    if isinstance(vision, dict) and projection is not None:
        mm_dim = _config_int(vision, "mm_embed_dim")
        if mm_dim and projection["shape"] != [hidden, mm_dim]:
            raise InventoryError("vision projection shape differs from config")
        checks.append("vision projection matches configured dimensions")
    return checks


def _modality(name: str) -> str:
    # Labels describe prefixes, not learned functions or behavioral influence.
    if ".vision_" in name or ".embed_vision." in name:
        return "vision_named"
    if ".audio_" in name or ".embed_audio." in name:
        return "audio_named"
    if ".language_model." in name:
        return "language_named"
    return "other_named"


def _module_and_layer(name: str) -> tuple[str, str | None]:
    match = LAYER_RE.match(name)
    if match:
        module = f"{match.group(1)}.*.{match.group(3).rsplit('.', 1)[0]}" \
            if "." in match.group(3) else f"{match.group(1)}.*"
        return module, f"{match.group(1)}.{match.group(2)}"
    return name.rsplit(".", 1)[0], None


def _aggregate(tensors: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    layer_types = config["text_config"].get("layer_types")
    for row in tensors:
        module, layer = _module_and_layer(row["name"])
        modality = _modality(row["name"])
        row["name_modality"] = modality
        row["module_template"] = module
        row["layer"] = layer
        for group_name, key in (("name_modalities", modality), ("module_templates", module),
                                ("layers", layer)):
            if key is None:
                continue
            group = groups[group_name].setdefault(key, {"tensors": 0, "elements": 0,
                                                         "bytes": 0, "dtypes": Counter()})
            group["tensors"] += 1
            group["elements"] += row["elements"]
            group["bytes"] += row["bytes"]
            group["dtypes"][row["dtype"]] += 1
    if isinstance(layer_types, list):
        for layer_name, group in groups["layers"].items():
            if layer_name.startswith("model.language_model.layers."):
                group["configured_attention_type"] = layer_types[int(layer_name.rsplit(".", 1)[1])]
    return {key: dict(sorted(records.items())) for key, records in sorted(groups.items())}


def _bf16_stats(path: Path, data_start: int, tensors: list[dict[str, Any]],
                tensor_count: int, elements_per_tensor: int) -> list[dict[str, Any]]:
    choices = [row for row in tensors if row["dtype"] == "BF16" and row["elements"]]
    if not choices or not tensor_count:
        return []
    # Spread choices across sorted names and elements across three windows.
    indices = {(index * (len(choices) - 1)) // max(1, tensor_count - 1)
               for index in range(min(tensor_count, len(choices)))}
    sampled = []
    with path.open("rb") as source:
        for index in sorted(indices):
            row = choices[index]
            count = min(elements_per_tensor, row["elements"])
            window_sizes = [count // 3, count // 3, count - 2 * (count // 3)]
            windows = [0, max(0, (row["elements"] - window_sizes[1]) // 2),
                       row["elements"] - window_sizes[2]]
            values = []
            for start_element, window_size in zip(windows, window_sizes):
                if not window_size:
                    continue
                source.seek(data_start + row["data_offsets"][0] + start_element * 2)
                data = source.read(window_size * 2)
                if len(data) != window_size * 2:
                    raise InventoryError(f"BF16 tensor changed during sampling: {row['name']}")
                values.extend(struct.unpack("<f", struct.pack("<I", word << 16))[0]
                              for (word,) in struct.iter_unpack("<H", data))
            finite = [value for value in values if math.isfinite(value)]
            mean = sum(finite) / len(finite) if finite else None
            sampled.append({"name": row["name"], "elements_sampled": len(values),
                            "finite": len(finite), "zero": sum(x == 0 for x in values),
                            "nonfinite": len(values) - len(finite),
                            "minimum_finite": min(finite) if finite else None,
                            "maximum_finite": max(finite) if finite else None,
                            "mean_finite": mean})
    return sampled


def inspect_checkpoint(snapshot: str | Path, *, model: str = "google/gemma-4-12B",
                       revision: str | None = None, expected_sha256: str | None = None,
                       hash_weights: bool = True, sample_bf16_tensors: int = 0,
                       sample_elements: int = 192) -> dict[str, Any]:
    """Validate and inventory local files; no model code is imported or run."""
    snapshot = Path(snapshot).expanduser().resolve()
    weights = snapshot / "model.safetensors"
    config_path = snapshot / "config.json"
    for path in (weights, config_path):
        if not path.is_file():
            raise InventoryError(f"required checkpoint file is missing: {path.name}")
    if expected_sha256 and (not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or
                            not hash_weights):
        raise InventoryError("expected SHA-256 requires a full file hash")
    if sample_bf16_tensors < 0 or sample_bf16_tensors > 64 or \
            not 1 <= sample_elements <= 1024:
        raise InventoryError("BF16 sampling limits exceeded")
    before = (_file_identity(weights), _file_identity(config_path))
    config_bytes = config_path.read_bytes()
    config = _json_bytes(config_bytes)
    if not isinstance(config, dict):
        raise InventoryError("config must be a JSON object")
    header, data_start, file_size, header_digest = _read_header(weights)
    tensors = _validate_tensors(header, file_size - data_start)
    checks = _validate_config(config, tensors)
    digest = _sha256_file(weights) if hash_weights else None
    if expected_sha256 and digest != expected_sha256:
        raise InventoryError("full weights SHA-256 differs from expected upstream hash")
    sampled = _bf16_stats(weights, data_start, tensors, sample_bf16_tensors,
                          sample_elements)
    after = (_file_identity(weights), _file_identity(config_path))
    if before != after:
        raise InventoryError("checkpoint changed during inspection")
    result = {
        "schema": SCHEMA,
        "provenance": {"declared_model": model, "declared_revision": revision,
                       "expected_upstream_sha256": expected_sha256,
                       "full_weight_sha256": digest, "full_weight_hash_performed": hash_weights,
                       "config_sha256": sha256(config_bytes).hexdigest(),
                       "safetensors_header_sha256": header_digest,
                       "weights_bytes": file_size, "header_bytes": data_start},
        "config_checks": checks,
        "summary": {"tensor_count": len(tensors),
                    "tensor_bytes": sum(row["bytes"] for row in tensors),
                    "elements": sum(row["elements"] for row in tensors),
                    "dtypes": dict(sorted(Counter(row["dtype"] for row in tensors).items()))},
        "groups": _aggregate(tensors, config), "tensors": tensors,
        "bf16_samples": sampled,
        "interpretation_limit": "Structural/name groups and numeric samples do not identify or remove learned behavior.",
    }
    result["manifest_sha256"] = sha256(_canonical(result)).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="google/gemma-4-12B")
    parser.add_argument("--revision")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--skip-full-hash", action="store_true")
    parser.add_argument("--sample-bf16-tensors", type=int, default=0)
    parser.add_argument("--sample-elements", type=int, default=192)
    args = parser.parse_args(argv)
    try:
        snapshot = args.snapshot.expanduser().resolve()
        output = args.output.expanduser().resolve()
        if output == snapshot or snapshot in output.parents:
            raise InventoryError("output must be outside the checkpoint snapshot")
        result = inspect_checkpoint(snapshot, model=args.model,
                                    revision=args.revision,
                                    expected_sha256=args.expected_sha256,
                                    hash_weights=not args.skip_full_hash,
                                    sample_bf16_tensors=args.sample_bf16_tensors,
                                    sample_elements=args.sample_elements)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as target:
            target.write(_canonical(result).decode("utf-8") + "\n")
    except (InventoryError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
