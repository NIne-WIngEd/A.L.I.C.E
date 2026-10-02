"""Public, features-only forward diagnostics; never N0/personality approval.

The only input texts are the hash-pinned public fictional fixture shipped with
this script. No arbitrary text, private corpus, chat template or generation is
accepted. A successful receipt describes shapes and resources, not competence
or removal of pretrained priors. It is always unqualified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PLAN_PATH = REPO_ROOT / "evaluation/eipm/gemma_n0/public_probe_plan_v1.json"
PLAN_SHA256 = "ee90660581c471adb53a5541880917d4f4951ad51181063e377c2dcf1795a123"
PLAN_SCHEMA = "alice-personality-gemma-n0-public-probe-plan-v1"
RECEIPT_SCHEMA = "alice-personality-gemma-n0-public-probe-receipt-v2"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z][a-z0-9-]{0,95}\Z")


class ProbeError(ValueError):
    """A diagnostic violated its public-input, source or feature boundary."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise ProbeError("input must be an absolute regular nonsymlink file")
    return candidate.resolve(strict=True)


def _digest(path: Path) -> str:
    before = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after):
        raise ProbeError("input changed while hashing")
    return digest.hexdigest()


def load_public_plan(path: str | Path = PLAN_PATH) -> dict:
    """Admit only the exact published fixture; a 'public' label grants nothing."""
    source = _file(path)
    if source.stat().st_size > 256 * 1024:
        raise ProbeError("public plan exceeds the finite fixture bound")
    payload = source.read_bytes()
    if sha256(payload).hexdigest() != PLAN_SHA256:
        raise ProbeError("public plan differs from the immutable public fixture")
    plan = json.loads(payload)
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("qualification") != "unqualified"
            or plan.get("source_class") != "public_fictional_development_fixture"):
        raise ProbeError("invalid public probe scope")
    contract = plan["fixture_contract"]
    if (contract["chat_template"] is not False or contract["generation"] is not False
            or contract["truncation"] is not False or contract["raw_text_only"] is not True
            or contract["authoritative_identity_targets"] is not False):
        raise ProbeError("public probe must stay inside the raw-text feature boundary")
    rows = plan["examples"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= contract["maximum_examples"]:
        raise ProbeError("public probe example count is outside its finite bound")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(contract["required_fields"]):
            raise ProbeError("public example fields differ from the fixture contract")
        if (not isinstance(row["probe_id"], str) or not _ID.fullmatch(row["probe_id"])
                or row["probe_id"] in seen):
            raise ProbeError("public probe IDs must be canonical and unique")
        seen.add(row["probe_id"])
        if (not isinstance(row["text"], str) or not row["text"].strip()
                or len(row["text"].encode("utf-8")) > contract["maximum_utf8_bytes_per_example"]):
            raise ProbeError("public text is empty or exceeds the finite fixture bound")
    return plan


def _code_binding() -> dict:
    paths = [Path(__file__), REPO_ROOT / "src/alice_personality/gemma_n0/backbone.py",
             REPO_ROOT / "src/alice_personality/gemma_n0/preparation.py"]
    files = [{"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": _digest(_file(path))}
             for path in paths]
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10, check=True)
        revision = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"repository_commit_context": revision, "current_files": files,
            "binding": "Current file hashes, including uncommitted code; commit alone is insufficient"}


def _verified_preparation(path: Path) -> dict:
    from src.alice_personality.gemma_n0.preparation import verify_prepared
    return verify_prepared(path)


@dataclass(frozen=True)
class _Runtime:
    torch: Any
    processor_factory: Any
    backbone_loader: Any
    transformers_version: str


def _load_runtime() -> _Runtime:
    import torch
    import transformers
    from src.alice_personality.gemma_n0.backbone import load_prepared_gemma_n0
    return _Runtime(torch, transformers.AutoProcessor, load_prepared_gemma_n0,
                    str(transformers.__version__))


def _new_output(path: str | Path, snapshot: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ProbeError("receipt must use an absolute nonsymlink path")
    candidate = candidate.resolve(strict=False)
    if candidate == snapshot or snapshot in candidate.parents:
        raise ProbeError("receipt must remain outside the prepared model snapshot")
    if candidate.exists():
        raise ProbeError("receipt already exists; use a new append-only artifact path")
    if not candidate.parent.is_dir():
        raise ProbeError("receipt parent directory is missing")
    return candidate


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ProbeError(f"{name} must be a positive integer")
    return value


def _tokenize(tokenizer: Any, texts: list[str], *, padding: bool, torch: Any) -> dict:
    encoded = tokenizer(texts, padding=padding, truncation=False,
                        add_special_tokens=True, return_attention_mask=True,
                        return_token_type_ids=False, return_tensors="pt")
    if set(encoded) != {"input_ids", "attention_mask"}:
        raise ProbeError("raw text tokenizer returned unexpected non-source inputs")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if (not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor)
            or ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != len(texts)
            or ids.shape[1] < 1 or not bool(((mask == 0) | (mask == 1)).all())
            or not bool(mask.bool().any(dim=1).all())):
        raise ProbeError("raw text tokenizer did not preserve a complete source batch")
    return {"input_ids": ids, "attention_mask": mask}


def _complete_feature_metadata(features: Any, *, torch: Any, ids_shape: tuple,
                               source_mask: Any, hidden_size: int,
                               expected_state_count: int) -> dict:
    """Validate every retained token state without publishing feature values."""
    states = getattr(features, "hidden_states", None)
    returned_mask = getattr(features, "attention_mask", None)
    layer_states = getattr(features, "all_hidden_states", None)
    expected_shape = (*ids_shape, hidden_size)
    if (not isinstance(returned_mask, torch.Tensor) or returned_mask.dtype != torch.bool
            or returned_mask.requires_grad or returned_mask.grad_fn is not None
            or tuple(returned_mask.shape) != tuple(source_mask.shape)
            or not torch.equal(returned_mask.cpu(), source_mask.bool().cpu())):
        raise ProbeError("feature result violated complete finite detached source alignment")
    if not isinstance(layer_states, tuple) or len(layer_states) != expected_state_count:
        raise ProbeError("feature result lacks the complete model-declared hidden-state layer chain")
    if not isinstance(states, torch.Tensor) or tuple(states.shape) != expected_shape:
        raise ProbeError("final feature result violated complete finite detached source alignment")
    metadata = []
    for index, layer in enumerate(layer_states):
        if (not isinstance(layer, torch.Tensor) or tuple(layer.shape) != expected_shape
                or not layer.is_floating_point() or not bool(torch.isfinite(layer).all())
                or layer.requires_grad or layer.grad_fn is not None
                or layer.device != states.device or layer.dtype != states.dtype
                or returned_mask.device != layer.device):
            raise ProbeError(f"hidden state {index} violated complete finite detached source alignment")
        metadata.append({"hidden_state_index": index,
                         "transformer_layer_index": None if index == 0 else index - 1,
                         "feature_shape": list(layer.shape), "feature_dtype": str(layer.dtype),
                         "feature_device": str(layer.device),
                         "feature_tensor_bytes": layer.numel() * layer.element_size()})
    if (not states.is_floating_point() or not bool(torch.isfinite(states).all())
            or states.requires_grad or states.grad_fn is not None
            or not torch.equal(states, layer_states[-1])):
        raise ProbeError("final features disagree with the last detached hidden state")
    return {"feature_shape": list(states.shape), "feature_dtype": str(states.dtype),
            "feature_device": str(states.device),
            "feature_tensor_bytes": states.numel() * states.element_size(),
            "expected_hidden_state_count": expected_state_count,
            "observed_hidden_state_count": len(layer_states),
            "observed_transformer_layer_count": len(layer_states) - 1,
            "hidden_state_count_includes_embedding": True,
            "layers": metadata,
            "all_layer_feature_tensor_bytes": sum(item["feature_tensor_bytes"] for item in metadata),
            "tensor_byte_measurement": "Logical element bytes summed once per hidden state; final-state metadata is not added again; not process peak memory",
            "all_features_finite": True, "all_features_detached": True,
            "all_features_source_aligned": True}


def run_probe(*, preparation_receipt: str | Path, output_receipt: str | Path,
              public_plan: str | Path = PLAN_PATH, device: str = "cpu", batch_size: int = 1,
              maximum_source_tokens: int | None = None) -> dict:
    """Run the public diagnostic and create one source-bound unqualified receipt."""
    _positive(batch_size, "batch size")
    if maximum_source_tokens is not None:
        _positive(maximum_source_tokens, "maximum source tokens")
    plan = load_public_plan(public_plan)
    plan_path = _file(public_plan)
    prepared_path = _file(preparation_receipt)
    prepared_bytes_sha256 = _digest(prepared_path)
    prepared = _verified_preparation(prepared_path)  # Before importing model packages.
    if not _DIGEST.fullmatch(prepared.get("receipt_sha256", "")):
        raise ProbeError("prepared receipt lacks its exact content binding")
    snapshot = Path(prepared["snapshot_path"])
    output = _new_output(output_receipt, snapshot)
    code_before = _code_binding()
    runtime = _load_runtime()
    torch = runtime.torch
    actual_runtime = {"backend": "transformers", "dtype": prepared["runtime"]["dtype"],
                      "torch_version": str(torch.__version__),
                      "transformers_version": runtime.transformers_version}
    if actual_runtime != prepared["runtime"]:
        raise ProbeError("installed runtime differs from the prepared runtime")
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ProbeError("diagnostic device must be CPU or CUDA")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ProbeError("requested CUDA diagnostic device is unavailable")
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.perf_counter()
    processor = runtime.processor_factory.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        raise ProbeError("prepared processor has no local raw-text tokenizer")
    backbone = runtime.backbone_loader(prepared_path, device=str(selected_device),
                                       dtype=prepared["runtime"]["dtype"])
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    load_ms = (time.perf_counter() - started) * 1000
    bound = _positive(backbone.max_source_tokens, "prepared source-token budget")
    if maximum_source_tokens is not None:
        if maximum_source_tokens > bound:
            raise ProbeError("requested source budget exceeds the prepared model budget")
        bound = maximum_source_tokens
    hidden_size = _positive(backbone.hidden_size, "prepared hidden size")
    transformer_layer_count = _positive(getattr(backbone, "num_hidden_layers", None),
                                        "prepared transformer layer count")
    expected_state_count = transformer_layer_count + 1  # Embedding + all transformer outputs.
    if _positive(getattr(backbone, "hidden_state_count", None),
                 "prepared hidden-state count") != expected_state_count:
        raise ProbeError("prepared hidden-state count differs from embedding plus transformer layers")

    # Validate every full sequence before forwarding any example. Never ask a
    # tokenizer to discard the suffix, even when an input cannot fit.
    individual = []
    for row in plan["examples"]:
        encoded = _tokenize(tokenizer, [row["text"]], padding=False, torch=torch)
        if encoded["input_ids"].shape[1] > bound:
            raise ProbeError(f"public probe {row['probe_id']} exceeds source budget; truncation is forbidden")
        individual.append(encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist())

    batches = []
    forward_total_ms = 0.0
    for offset in range(0, len(plan["examples"]), batch_size):
        rows = plan["examples"][offset:offset + batch_size]
        encoded = _tokenize(tokenizer, [row["text"] for row in rows], padding=True, torch=torch)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if ids.shape[1] > bound:
            raise ProbeError("padded public batch exceeds source budget; truncation is forbidden")
        for index in range(len(rows)):
            if ids[index][mask[index].bool()].tolist() != individual[offset + index]:
                raise ProbeError("batch tokenizer altered or truncated the complete public source")
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        started = time.perf_counter()
        features = backbone.extract_features(input_ids=ids, attention_mask=mask)
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        feature_metadata = _complete_feature_metadata(
            features, torch=torch, ids_shape=tuple(ids.shape), source_mask=mask,
            hidden_size=hidden_size, expected_state_count=expected_state_count)
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ProbeError("nonfinite diagnostic timing")
        forward_total_ms += elapsed_ms
        batches.append({"probe_ids": [row["probe_id"] for row in rows],
                        "input_shape": list(ids.shape), **feature_metadata,
                        "source_token_counts": mask.sum(dim=1).tolist(),
                        "input_ids_sha256": sha256(_canonical(ids.tolist())).hexdigest(),
                        "attention_mask_sha256": sha256(_canonical(mask.tolist())).hexdigest(),
                        "forward_ms": elapsed_ms})
        del features, encoded, ids, mask

    memory = {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
              "cpu_process_peak_bytes": None,
              "cpu_process_peak_status": "not_measured; tensor byte counts are separate"}
    if selected_device.type == "cuda":
        memory.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(selected_device)),
                      cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(selected_device)))
    if (_digest(prepared_path) != prepared_bytes_sha256
            or _verified_preparation(prepared_path) != prepared
            or _digest(plan_path) != PLAN_SHA256 or _code_binding() != code_before):
        raise ProbeError("source, prepared contract, public plan or code changed during the diagnostic")
    receipt = {
        "schema": RECEIPT_SCHEMA, "qualification": "unqualified",
        "meaning": "Public forward diagnostic only; no competence, neutrality or personality pass",
        "public_plan_id": plan["plan_id"], "public_plan_sha256": PLAN_SHA256,
        "source_class": plan["source_class"],
        "prepared_receipt_sha256": prepared["receipt_sha256"],
        "prepared_receipt_file_sha256": prepared_bytes_sha256,
        "source_repository": prepared["repository"], "source_revision": prepared["revision"],
        "source_files": prepared["files"], "code": code_before,
        "runtime": actual_runtime,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "selected_device": str(selected_device), "torch_cuda": torch.version.cuda},
        "raw_text_tokenization": {"chat_template": False, "truncation": False,
                                  "generation": False, "maximum_source_tokens": bound},
        "example_count": len(plan["examples"]), "batch_size": batch_size,
        "expected_hidden_state_count": expected_state_count,
        "expected_transformer_layer_count": transformer_layer_count,
        "observed_hidden_state_counts": sorted({batch["observed_hidden_state_count"] for batch in batches}),
        "load_ms": load_ms, "forward_total_ms": forward_total_ms,
        "batches": batches, "memory": memory,
        "hidden_states_retained": False, "behavioral_scores": None,
    }
    receipt["receipt_sha256"] = sha256(_canonical(receipt)).hexdigest()
    payload = _canonical(receipt) + b"\n"  # Reject NaN/Infinity before creating output.
    _new_output(output, snapshot)
    try:
        with output.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise ProbeError("receipt already exists; append-only evidence cannot be replaced") from exc
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preparation_receipt", type=Path)
    parser.add_argument("output_receipt", type=Path)
    parser.add_argument("--public-plan", type=Path, default=PLAN_PATH,
                        help="Only byte-identical copies of the shipped public fixture are admitted")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--maximum-source-tokens", type=int)
    args = parser.parse_args(argv)
    try:
        receipt = run_probe(**vars(args))
    except (ProbeError, OSError, ValueError, TypeError, ImportError) as exc:
        print(f"Public diagnostic refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": receipt["schema"], "qualification": "unqualified",
                      "example_count": receipt["example_count"],
                      "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
