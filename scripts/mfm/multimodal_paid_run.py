"""CPU-verifiable bindings for a paid, distributed MFM weight run."""

from __future__ import annotations

from hashlib import sha256
from itertools import chain
import json
import os
from pathlib import Path
import re
import time

from cognitive_kernel.canonical import CognitiveKernelContractError


PREFLIGHT_SCHEMA = "mfm-gemma4-processor-preflight-v1"
RUN_SCHEMA = "mfm-gemma4-distributed-run-v1"


def digest_record(record: dict) -> str:
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")).hexdigest()


def sealed_record(record: dict) -> dict:
    return {**record, "record_sha256": digest_record(record)}


def read_sealed(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CognitiveKernelContractError(f"cannot read sealed receipt: {path}") from exc
    if not isinstance(record, dict):
        raise CognitiveKernelContractError("sealed receipt must be an object")
    digest = record.pop("record_sha256", None)
    if digest != digest_record(record):
        raise CognitiveKernelContractError("sealed receipt digest differs")
    return {**record, "record_sha256": digest}


def write_sealed(path: Path, record: dict) -> dict:
    output = sealed_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(output, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return output


def source_fingerprint() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    return {name: sha256(path.read_bytes()).hexdigest() for name, path in {
        "multimodal_trainer": source / "train_multimodal_formation.py",
        "paid_run_contract": source / "multimodal_paid_run.py",
        "model_staging": source / "stage_gemma4_model.py",
        "artifact_receipt": source / "train_formation_model.py",
        "formation_learning": source.parents[1] / "src/cognitive_kernel/formation_learning.py",
        "corpus_admission": source.parents[1] / "src/cognitive_kernel/formation_dataset_admission.py",
        "formation_contracts": source.parents[1] / "src/cognitive_kernel/formation_contracts.py",
        "canonical_contract": source.parents[1] / "src/cognitive_kernel/canonical.py",
        "multimodal_prompt": source.parents[1] / "src/cognitive_kernel/formation_multimodal.py",
    }.items()}


def require_preflight(path: Path, expected: dict) -> dict:
    record = read_sealed(path)
    if record.get("schema") != PREFLIGHT_SCHEMA:
        raise CognitiveKernelContractError("wrong processor preflight schema")
    for key, value in expected.items():
        if record.get(key) != value:
            raise CognitiveKernelContractError(f"processor preflight differs: {key}")
    if record.get("source_fingerprint") != source_fingerprint():
        raise CognitiveKernelContractError("processor preflight source code differs")
    indices = record.get("probe_indices")
    if (not isinstance(indices, list) or not indices or
            any(not isinstance(i, int) or i < 0 or i >= record["train_cases"] for i in indices) or
            len(indices) != len(set(indices)) or
            record.get("longest_case_index") not in indices):
        raise CognitiveKernelContractError("processor preflight probe selection invalid")
    return record


def probe_indices(lengths: list[tuple[int, int]], count: int = 64) -> list[int]:
    """Include the four longest cases and deterministic coverage across the corpus."""
    if not lengths:
        raise CognitiveKernelContractError("empty processor preflight")
    longest = sorted(lengths, key=lambda row: (-row[1], row[0]))[:4]
    total = len(lengths)
    chosen = [index for index, _ in longest]
    selected = set(chosen)
    remaining = max(0, min(count, total) - len(chosen))
    spaced = ((n * total) // max(1, remaining) for n in range(remaining))
    for index in chain(spaced, range(total)):
        if len(chosen) == min(count, total):
            break
        if index not in selected:
            chosen.append(index)
            selected.add(index)
    return chosen


def require_zero3(path: Path) -> str:
    try:
        raw = path.read_bytes()
        config = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise CognitiveKernelContractError("cannot read DeepSpeed JSON") from exc
    zero = config.get("zero_optimization", {}) if isinstance(config, dict) else {}
    if not isinstance(zero, dict) or zero.get("stage") != 3:
        raise CognitiveKernelContractError("full-weight distributed MFM requires ZeRO stage 3")
    if zero.get("stage3_gather_16bit_weights_on_model_save") is not True:
        raise CognitiveKernelContractError("ZeRO-3 export must gather complete 16-bit weights")
    if config.get("bf16", {}).get("enabled") not in (True, "auto"):
        raise CognitiveKernelContractError("ZeRO-3 route requires BF16")
    for key in ("train_batch_size", "train_micro_batch_size_per_gpu",
                "gradient_accumulation_steps", "gradient_clipping"):
        if config.get(key) != "auto":
            raise CognitiveKernelContractError(f"DeepSpeed {key} must track TrainingArguments")
    if config.get("fp16", {}).get("enabled") is True:
        raise CognitiveKernelContractError("cannot enable FP16 and BF16 together")
    if "offload_param" in zero or "offload_optimizer" in zero:
        raise CognitiveKernelContractError("offload requires a separately qualified config")
    return sha256(raw).hexdigest()


def require_complete_weight_export(root: Path, *, min_bytes: int = 20_000_000_000,
                                   min_parameters: int = 10_000_000_000) -> dict:
    """Reject a partial or invalid ZeRO-3 safetensors export before issuing a receipt."""
    if not (root / "config.json").is_file():
        raise CognitiveKernelContractError("model export lacks config.json")
    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.is_file() and not index.exists() and not single.is_symlink():
        files = [single]
        mapping = None
    elif index.is_file() and not single.exists():
        try:
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CognitiveKernelContractError("model export has an invalid weight index") from exc
        if not isinstance(mapping, dict) or len(mapping) < 100:
            raise CognitiveKernelContractError("model export weight index is incomplete")
        names = set(mapping.values())
        if (any(not isinstance(name, str) or not re.fullmatch(
                r"model-[0-9]+-of-[0-9]+\.safetensors", name) for name in names)):
            raise CognitiveKernelContractError("model export has unsafe shard names")
        files = [root / name for name in sorted(names)]
        if any(not path.is_file() or path.is_symlink() for path in files):
            raise CognitiveKernelContractError("model export has a missing weight shard")
    else:
        raise CognitiveKernelContractError("model export lacks complete safetensors weights")
    size = sum(path.stat().st_size for path in files)
    if size < min_bytes:
        raise CognitiveKernelContractError("model export is smaller than the pinned 12B weights")
    tensor_names = set()
    parameters = 0
    dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
                   "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8,
                   "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
    for file in files:
        with file.open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            if header_size < 2 or header_size > 200_000_000:
                raise CognitiveKernelContractError("model export safetensors header is invalid")
            try:
                header = json.loads(stream.read(header_size))
            except (ValueError, UnicodeDecodeError) as exc:
                raise CognitiveKernelContractError("model export safetensors header is invalid") from exc
        if not isinstance(header, dict):
            raise CognitiveKernelContractError("model export safetensors header is invalid")
        spans = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in tensor_names or not isinstance(tensor, dict):
                raise CognitiveKernelContractError("model export has duplicate/invalid tensors")
            try:
                shape = tensor["shape"]
                start, end = tensor["data_offsets"]
                width = dtype_bytes[tensor["dtype"]]
                if (not isinstance(shape, list) or
                        any(not isinstance(dim, int) or dim < 1 for dim in shape)):
                    raise ValueError("invalid shape")
                count = 1
                for dim in shape:
                    count *= dim
                if (not isinstance(start, int) or start < 0 or
                        not isinstance(end, int) or end > file.stat().st_size - 8 - header_size or
                        end - start != count * width):
                    raise ValueError("invalid tensor offsets")
            except (KeyError, TypeError, ValueError) as exc:
                raise CognitiveKernelContractError("model export has invalid tensor metadata") from exc
            tensor_names.add(name)
            parameters += count
            spans.append((start, end))
        if any(left[1] > right[0] for left, right in zip(sorted(spans), sorted(spans)[1:])):
            raise CognitiveKernelContractError("model export has overlapping tensors")
    if mapping is not None and set(mapping) != tensor_names:
        raise CognitiveKernelContractError("model export index does not match tensors")
    if parameters < min_parameters:
        raise CognitiveKernelContractError("model export lacks the pinned 12B parameter count")
    return {"weight_files": [path.name for path in files], "weight_bytes": size,
            "weight_tensors": len(tensor_names), "parameters": parameters}


def bind_run(root: Path, record: dict, resume: Path | None) -> None:
    """Refuse an occupied root or a checkpoint from another frozen run."""
    manifest = root / "run-manifest.json"
    if resume is None:
        if root.exists() and any(root.iterdir()):
            raise CognitiveKernelContractError("new run output directory is occupied")
        write_sealed(manifest, record)
        return
    if not manifest.is_file() or read_sealed(manifest) != sealed_record(record):
        raise CognitiveKernelContractError("resume run manifest differs")
    if (root / "artifact-receipt.json").exists():
        raise CognitiveKernelContractError("completed artifact cannot resume")
    checkpoint_root = (root / "checkpoints").resolve()
    if (not re.fullmatch(r"checkpoint-[1-9][0-9]*", resume.name) or
            resume.resolve().parent != checkpoint_root or
            not (resume / "trainer_state.json").is_file()):
        raise CognitiveKernelContractError("resume requires a checkpoint in this exact run")


def await_run_binding(root: Path, record: dict, timeout_seconds: float = 30) -> None:
    manifest = root / "run-manifest.json"
    deadline = time.monotonic() + timeout_seconds
    while not manifest.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not manifest.exists() or read_sealed(manifest) != sealed_record(record):
        raise CognitiveKernelContractError("distributed run manifest missing or differs")
