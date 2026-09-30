"""Offline public diagnostic for a verified MFM-role Gemma base.

The pristine publisher checkpoint is the untouched control run by
``run_gemma4_base_behavior.py``. This runner accepts a separately receipted
MFM-role clone or an evidence-backed modified derivative, verified against the
pinned publisher ancestry. Identical outputs from a clone and source are an
expected result; a changed hash is never required to start MFM training.
Neither run qualifies a base.

Stage dependencies and the complete prepared artifact locally. Network
isolation can be required by flag; the default public-only route remains
usable on free GPU platforms with an active network interface. No private
source is accepted.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import sys

if __package__:
    from .run_gemma4_base_behavior import (
        BaselineError, OUTPUT_SCHEMA, _load_local_model, _public_prompts,
    )
    from .verify_gemma4_pretrained import FILES, REPO, REVISION
else:
    from run_gemma4_base_behavior import (
        BaselineError, OUTPUT_SCHEMA, _load_local_model, _public_prompts,
    )
    from verify_gemma4_pretrained import FILES, REPO, REVISION


FROZEN_PUBLIC_PROMPTS_SHA256 = "28afd86e40f3d104988560de5cdb1c388b15cbb51d00e1dc1949f2bf3a96127f"


def _configure_offline_process(*, require_network_isolation: bool) -> str:
    """Disable library network paths and record, or require, a local NIC boundary.

    This route accepts only public synthetic prompts. Local-only flags do not
    establish OS-enforced egress denial and are never a private-data receipt.
    """
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true"})
    if not sys.platform.startswith("linux"):
        if require_network_isolation:
            raise BaselineError("network-isolated diagnostic requires Linux")
        return "network_interfaces_unchecked_public_only"
    try:
        interfaces = {name for _, name in socket.if_nameindex()}
    except OSError as exc:
        if require_network_isolation:
            raise BaselineError("could not inspect network interfaces") from exc
        return "network_interfaces_unchecked_public_only"
    if interfaces == {"lo"}:
        return "loopback_only_interface_check_unix_proxy_unverified"
    if require_network_isolation:
        raise BaselineError("prepared-base diagnostic needs a loopback-only network namespace")
    return "network_interfaces_present_public_only"


def _prepared_receipt(snapshot: Path, receipt_path: Path) -> tuple[dict, str]:
    try:
        from alice_foundation.gemma4_v1 import verify_role_base
    except ImportError as exc:
        raise BaselineError("install the pinned first-party alice_foundation package") from exc
    receipt = verify_role_base(receipt_path, snapshot=snapshot, expected_role="mfm")
    if (receipt.get("repository"), receipt.get("revision"), receipt.get("role")) != \
            (REPO, REVISION, "mfm"):
        raise BaselineError("prepared base lineage or status differs")
    schema = receipt.get("schema")
    source_weight = FILES["model.safetensors"][1]
    weight = next((row for row in receipt.get("files", [])
                   if row.get("path") == "model.safetensors"), None)
    digest = weight.get("sha256") if weight else None
    if schema == "alice-gemma4-v1-clone-v1":
        if digest != source_weight or \
                not isinstance(receipt.get("parent_source_receipt_sha256"), str):
            raise BaselineError("prepared role clone lacks pinned source ancestry")
    elif schema == "alice-gemma4-v1-derivative-v1":
        if (receipt.get("qualification") != "unqualified" or
                receipt.get("upstream_weight_sha256") != source_weight or
                not isinstance(digest, str) or len(digest) != 64 or digest == source_weight):
            raise BaselineError("prepared derivative lacks changed pinned ancestry")
    else:
        raise BaselineError("prepared base receipt is not a verified MFM role copy")
    return receipt, digest


def run(snapshot: Path, receipt_path: Path, prompts_path: Path, output: Path,
        *, max_new_tokens: int, seed: int,
        require_network_isolation: bool = False) -> dict:
    if not 1 <= max_new_tokens <= 1024 or not 0 <= seed <= 2**32 - 1:
        raise BaselineError("max_new_tokens or seed is outside the bounded range")
    snapshot = snapshot.expanduser().resolve(strict=True)
    prompts_path = prompts_path.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists() or output == snapshot or snapshot in output.parents or \
            output in {prompts_path, receipt_path}:
        raise BaselineError("output exists or overlaps immutable model/input")
    network_boundary = _configure_offline_process(
        require_network_isolation=require_network_isolation)
    receipt, weight_digest = _prepared_receipt(snapshot, receipt_path)
    cases, prompts_digest = _public_prompts(prompts_path)
    if prompts_digest != FROZEN_PUBLIC_PROMPTS_SHA256:
        raise BaselineError("prompts differ from the frozen public synthetic diagnostic")
    torch, processor, model = _load_local_model(snapshot)
    device = next(model.parameters()).device
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
                row = {"schema": OUTPUT_SCHEMA, "case_id": case["case_id"],
                       "context_digest": case["context_digest"],
                       "prompt_set_sha256": prompts_digest,
                       "model_receipt": receipt["receipt_sha256"],
                       "model_artifact_digest": weight_digest,
                       "source_repository": REPO, "source_revision": REVISION,
                       "base_source_sha256": FILES["model.safetensors"][1],
                       "prepared_base_parent_sha256": FILES["model.safetensors"][1],
                       "prepared_base_sha256": weight_digest,
                       "prepared_base_receipt_sha256": receipt["receipt_sha256"],
                       "prepared_base_qualification": "unqualified",
                       "prepared_base_kind": receipt["schema"],
                       "generation": generation, "processed_modalities": ["text"],
                       "network_boundary": network_boundary}
                if case["has_attachments"]:
                    row.update({"status": "unexercised", "output_text": None,
                                "generated_token_ids": [],
                                "reason": "non-text attachment unsupported by this text diagnostic"})
                else:
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
            "source_revision": REVISION, "source_weight_sha256": FILES["model.safetensors"][1],
            "prepared_base_sha256": weight_digest,
            "prepared_base_receipt_sha256": receipt["receipt_sha256"],
            "prompt_set_sha256": prompts_digest, "qualification_claim": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-network-isolation", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.snapshot, args.receipt, args.prompts, args.output,
                     max_new_tokens=args.max_new_tokens, seed=args.seed,
                     require_network_isolation=args.require_network_isolation)
    except (BaselineError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"prepared-base diagnostic failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
