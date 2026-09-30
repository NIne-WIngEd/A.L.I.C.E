"""Offline, deterministic text baseline for the exact pretrained Gemma 4 12B.

This is a measurement runner, not an MFM trainer or a qualified product model.
It accepts only explicitly classified public synthetic prompts. Image/audio/video
attachments are recorded as unexercised; they are never silently treated as text.
The publisher snapshot is fully rehashed against independently pinned bytes before
the local Transformers loader sees any model files.

Example on a network-isolated GPU host with the dependencies already installed::

    python scripts/mfm/run_gemma4_base_behavior.py \
      --snapshot /models/gemma-4-12B-pretrained \
      --receipt /receipts/gemma-4-12B-pretrained.json \
      --prompts /cases/public-synthetic.jsonl \
      --output /results/gemma-4-12B-base.jsonl

The host needs enough GPU memory for roughly 24 GB of BF16 weights plus runtime
and context memory. A 48 GB GPU or suitable multi-GPU host is the practical
starting point; actual fit must be measured. The runtime needs compatible
PyTorch, Accelerate, and a Transformers release supporting gemma4_unified.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re

if __package__:
    from .verify_gemma4_pretrained import FILES, REPO, REVISION, SCHEMA, verify
else:
    from verify_gemma4_pretrained import FILES, REPO, REVISION, SCHEMA, verify


OUTPUT_SCHEMA = "mfm-gemma4-base-generation-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class BaselineError(ValueError):
    """An input, receipt, model runtime, or output failed closed."""


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_receipt(snapshot: Path, receipt_path: Path) -> dict:
    """Check the receipt's digest and rehash every upstream file independently."""
    snapshot = snapshot.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    if receipt_path == snapshot or snapshot in receipt_path.parents:
        raise BaselineError("receipt must be outside the source snapshot")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {"schema", "repository", "revision", "snapshot_path", "files",
                "verified_at_utc", "meaning", "receipt_sha256"}
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise BaselineError("source receipt schema differs")
    receipt_digest = receipt["receipt_sha256"]
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    publisher_receipt_bytes = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if not isinstance(receipt_digest, str) or not _HEX64.fullmatch(receipt_digest) or \
            sha256(publisher_receipt_bytes).hexdigest() != receipt_digest:
        raise BaselineError("source receipt digest differs")
    if (receipt["schema"], receipt["repository"], receipt["revision"],
            receipt["snapshot_path"]) != (SCHEMA, REPO, REVISION, str(snapshot)):
        raise BaselineError("receipt does not identify this exact local pretrained snapshot")
    current = verify(snapshot)
    if receipt["files"] != current["files"] or receipt["meaning"] != current["meaning"]:
        raise BaselineError("source bytes or source receipt differ from pinned pretrained publisher")
    expected_weight = FILES["model.safetensors"][1]
    if next(row for row in current["files"] if row["path"] == "model.safetensors")["sha256"] != expected_weight:
        raise BaselineError("pretrained weight digest differs")
    return receipt


def _public_prompts(path: Path) -> tuple[list[dict], str]:
    path = path.expanduser().resolve(strict=True)
    prompts = []
    seen = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineError(f"invalid prompt JSON at line {line_number}") from exc
            if not isinstance(row, dict) or row.get("classification") != "public_synthetic":
                raise BaselineError(f"line {line_number} lacks public_synthetic classification")
            case_id, prompt = row.get("case_id"), row.get("prompt")
            if not isinstance(case_id, str) or not case_id or case_id in seen or \
                    not isinstance(prompt, str) or not prompt or len(prompt) > 100_000:
                raise BaselineError(f"invalid or duplicate case/prompt at line {line_number}")
            digest = row.get("context_digest")
            if digest is not None and (not isinstance(digest, str) or
                                       not _HEX64.fullmatch(digest)):
                raise BaselineError(f"invalid context digest at line {line_number}")
            attachments = row.get("attachments", [])
            if not isinstance(attachments, list):
                raise BaselineError(f"invalid attachments at line {line_number}")
            seen.add(case_id)
            prompts.append({"case_id": case_id, "prompt": prompt,
                            "context_digest": digest, "has_attachments": bool(attachments)})
    if not prompts:
        raise BaselineError("prompt file has no public synthetic cases")
    return prompts, _file_digest(path)


def _load_local_model(snapshot: Path):
    # Set these before any HF or torch import. Network isolation by the host is
    # still needed to enforce the no-egress property against arbitrary code.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise BaselineError("install compatible torch, accelerate, and transformers on the GPU host") from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise BaselineError("a CUDA GPU with BF16 support is required for this BF16 baseline")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    processor = AutoProcessor.from_pretrained(str(snapshot), **kwargs)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(snapshot), dtype=torch.bfloat16, device_map="auto",
        use_safetensors=True, attn_implementation="eager", **kwargs)
    if type(processor).__name__ != "Gemma4UnifiedProcessor" or \
            type(model).__name__ != "Gemma4UnifiedForConditionalGeneration":
        raise BaselineError("the local loader resolved an unexpected processor or model class")
    placement = getattr(model, "hf_device_map", {})
    if not placement or any(str(place) == "cpu" or str(place) == "disk"
                            for place in placement.values()):
        raise BaselineError("model was offloaded to CPU/disk; use sufficient GPU memory")
    model.eval()
    return torch, processor, model


def run(snapshot: Path, receipt_path: Path, prompts_path: Path, output: Path,
        *, max_new_tokens: int, seed: int) -> dict:
    if not 1 <= max_new_tokens <= 1024 or not 0 <= seed <= 2**32 - 1:
        raise BaselineError("max_new_tokens or seed is outside the bounded range")
    snapshot = snapshot.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if output.exists() or output == snapshot or snapshot in output.parents:
        raise BaselineError("output exists or is inside the immutable source snapshot")
    if output == prompts_path.expanduser().resolve(strict=True) or \
            output == receipt_path.expanduser().resolve(strict=True):
        raise BaselineError("output overlaps an input file")
    receipt = _source_receipt(snapshot, receipt_path)
    cases, prompts_digest = _public_prompts(prompts_path)
    torch, processor, model = _load_local_model(snapshot)
    device = next(model.parameters()).device
    weight_digest = FILES["model.safetensors"][1]
    generation = {"do_sample": False, "num_beams": 1,
                  "max_new_tokens": max_new_tokens, "seed": seed,
                  "attention_implementation": "eager", "dtype": "bfloat16"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".partial-{os.getpid()}")
    if temp.exists():
        raise BaselineError("temporary output already exists")
    count = 0
    try:
        with temp.open("x", encoding="utf-8") as destination:
            for case in cases:
                row = {"schema": OUTPUT_SCHEMA,
                       "case_id": case["case_id"],
                       "context_digest": case["context_digest"],
                       "prompt_set_sha256": prompts_digest,
                       "model_receipt": receipt["receipt_sha256"],
                       "model_artifact_digest": weight_digest,
                       "source_repository": REPO, "source_revision": REVISION,
                       "generation": generation, "processed_modalities": ["text"]}
                if case["has_attachments"]:
                    row.update({"status": "unexercised", "output_text": None,
                                "generated_token_ids": [],
                                "reason": "non-text attachment unsupported by this baseline runner"})
                else:
                    # Derive an independent seed per case so input order changes
                    # do not change its output. Greedy decoding is also fixed.
                    case_seed = int.from_bytes(sha256(
                        f"{seed}:{case['case_id']}".encode()).digest()[:4], "big")
                    torch.manual_seed(case_seed)
                    torch.cuda.manual_seed_all(case_seed)
                    encoded = processor(text=case["prompt"], return_tensors="pt")
                    encoded = {key: value.to(device) if hasattr(value, "to") else value
                               for key, value in encoded.items()}
                    if "input_ids" not in encoded:
                        raise BaselineError("processor produced no text token ids")
                    with torch.inference_mode():
                        tokens = model.generate(**encoded, do_sample=False, num_beams=1,
                                                max_new_tokens=max_new_tokens)
                    suffix = tokens[0, encoded["input_ids"].shape[-1]:].tolist()
                    row.update({"status": "generated", "output_text": processor.tokenizer.decode(
                        suffix, skip_special_tokens=True),
                        "raw_output_text": processor.tokenizer.decode(
                            suffix, skip_special_tokens=False),
                        "generated_token_ids": suffix, "case_seed": case_seed})
                destination.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                destination.flush()
                count += 1
        if output.exists():
            raise BaselineError("output appeared during generation")
        temp.rename(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return {"output": str(output), "cases": count,
            "source_revision": REVISION, "source_weight_sha256": weight_digest,
            "source_receipt_sha256": receipt["receipt_sha256"],
            "prompt_set_sha256": prompts_digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        result = run(args.snapshot, args.receipt, args.prompts, args.output,
                     max_new_tokens=args.max_new_tokens, seed=args.seed)
    except (BaselineError, OSError, ValueError) as exc:
        parser.exit(2, f"baseline failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
