"""Execute the separately versioned v1.6 formation decoder and seeded control.

Only a verified local role base supplies source representations. The Gemma
language head cannot answer for MFM. Raw invalid outputs are preserved. A
bounded probe remains diagnostic even if it produces a grounded proposal;
this command never promotes a proposal into authoritative memory.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from cognitive_kernel.canonical import CognitiveKernelContractError, require_identifier
from cognitive_kernel.formation_semantics_v16 import (
    bundle_v16_from_output, context_v16_from_record,
    validate_formation_grounding_v16,
)

from . import run_v1_formation_specialist as shared_inference
from . import train_v16_formation_specialist as training


def _config(record: dict):
    return shared_inference._config(record)


def verify_artifacts(component_dir: Path, prepared_base_dir: Path,
                     prepared_base_receipt: Path, preflight_receipt: Path, *,
                     control: str) -> tuple[dict, dict, dict, Path]:
    """Rehash the prepared base, versioned processor, run, and both controls."""
    if control not in ("trained", "seeded-untrained"):
        raise CognitiveKernelContractError("unknown v1.6 specialist control")
    prepared = training.shared._prepared_base(SimpleNamespace(
        prepared_base_dir=prepared_base_dir,
        prepared_base_receipt=prepared_base_receipt))
    preflight = training.shared._read_sealed(preflight_receipt,
                                              training.PREFLIGHT_SCHEMA)
    run = training.shared._read_sealed(component_dir / "run.json", training.RUN_SCHEMA)
    component = training.shared._read_sealed(component_dir / "formation-component.json",
                                              training.ARTIFACT_SCHEMA)
    config = _config(component.get("specialist_config"))
    seed = training._verify_seed_control(component_dir, run["record_sha256"], config)
    expected = {
        "objective": training.OBJECTIVE_VERSION_V16,
        "output_schema": "mfm-formation-output-v1.6",
        "context_schema": "mfm-formation-context-v1.6",
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "prepared_base_parent_weight_sha256": training.shared.SOURCE_WEIGHT_SHA256,
        "prepared_base_kind": training.shared._prepared_kind(prepared),
        "foundation_commit": training.shared.FOUNDATION_COMMIT,
        "foundation_verifier_sha256": training.shared.FOUNDATION_VERIFIER_SHA256,
        "foundation_inventory_sha256": training.shared.FOUNDATION_INVENTORY_SHA256,
        "source_template_sha256": sha256(training.shared.SOURCE_TEMPLATE.encode()).hexdigest(),
        "source_instruction_sha256": sha256(training.SOURCE_INSTRUCTION.encode()).hexdigest(),
        "source_builder_sha256": training.shared._digest(Path(
            __import__(training.formation_context_media_messages.__module__,
                       fromlist=["__file__"]).__file__)),
        "codec_sha256": training._codec_sha256(),
        "semantics_sha256": training._semantics_sha256(),
        "decoder_implementation_sha256": training._decoder_sha256(),
        "trainer_sha256": training.shared._digest(Path(training.__file__)),
    }
    if any(preflight.get(key) != value for key, value in expected.items()):
        raise CognitiveKernelContractError("v1.6 processor or objective lineage differs")
    prepared_weight = next((row["sha256"] for row in prepared["files"]
                            if row["path"] == "model.safetensors"), None)
    if preflight.get("prepared_base_weight_sha256") != prepared_weight or \
            run.get("objective") != training.OBJECTIVE_VERSION_V16 or \
            component.get("objective") != training.OBJECTIVE_VERSION_V16 or \
            run.get("preflight_sha256") != preflight["record_sha256"] or \
            run.get("prepared_base_receipt_sha256") != prepared["receipt_sha256"] or \
            run.get("corpus_sha256") != preflight.get("corpus_sha256") or \
            component.get("run_manifest_sha256") != run["record_sha256"] or \
            component.get("prepared_base_receipt_sha256") != prepared["receipt_sha256"] or \
            component.get("prepared_base_sha256") != prepared_weight or \
            component.get("prepared_base_parent_sha256") != training.shared.SOURCE_WEIGHT_SHA256 or \
            component.get("prepared_base_kind") != preflight.get("prepared_base_kind") or \
            component.get("specialist_config") != run.get("specialist_config") or \
            component.get("seed_control_sha256") != seed["specialist_sha256"] or \
            preflight.get("specialist_heads") != config.heads or \
            preflight.get("specialist_layers") != config.layers or \
            preflight.get("max_target_tokens") != config.max_target_tokens or \
            type(preflight.get("max_source_tokens")) is not int or \
            preflight["max_source_tokens"] < 1 or \
            run.get("transformers_version") != preflight.get("transformers_version") or \
            run.get("qualified_for_product") is not False or \
            component.get("qualified_for_product") is not False or \
            run.get("probe_only") is not True or \
            component.get("probe_only") is not True or \
            component.get("optimizer_steps") != 1:
        raise CognitiveKernelContractError("v1.6 component, run or preflight lineage differs")
    trained_path = component_dir / "formation-specialist.safetensors"
    if training.shared._digest(trained_path) != component.get("formation_component_sha256"):
        raise CognitiveKernelContractError("v1.6 specialist weight bytes differ")
    if component["formation_component_sha256"] == seed["specialist_sha256"]:
        raise CognitiveKernelContractError("v1.6 probe specialist is identical to seeded control")
    selected = trained_path if control == "trained" else component_dir / "seed-control.safetensors"
    if control == "seeded-untrained" and \
            training.shared._digest(selected) != seed["specialist_sha256"]:
        raise CognitiveKernelContractError("v1.6 seeded-control weight bytes differ")
    return prepared, preflight, component, selected


def read_input(row: dict):
    if not isinstance(row, dict) or set(row) != {
            "case_id", "context", "context_digest", "opened_sources"}:
        raise CognitiveKernelContractError("v1.6 inference input fields differ")
    if require_identifier(row["case_id"], "case_id") != row["case_id"]:
        raise CognitiveKernelContractError("v1.6 case ID is not canonical")
    context = context_v16_from_record(row["context"])
    if context.content_digest() != row["context_digest"]:
        raise CognitiveKernelContractError("v1.6 inference context digest differs")
    if not isinstance(row["opened_sources"], list):
        raise CognitiveKernelContractError("v1.6 opened sources must be an array")
    opened = []
    for entry in row["opened_sources"]:
        if not isinstance(entry, dict) or set(entry) != {"ref_id", "payload_base64"}:
            raise CognitiveKernelContractError("v1.6 exact source entry differs")
        try:
            raw = base64.b64decode(entry["payload_base64"], validate=True)
        except (binascii.Error, TypeError, ValueError) as exc:
            raise CognitiveKernelContractError("v1.6 exact source encoding differs") from exc
        if base64.b64encode(raw).decode("ascii") != entry["payload_base64"]:
            raise CognitiveKernelContractError("v1.6 source encoding is not canonical")
        opened.append((entry["ref_id"], raw))
    opened = tuple(opened)
    if tuple(ref for ref, _ in opened) != tuple(
            ref.ref_id for ref in context.base.evidence):
        raise CognitiveKernelContractError("v1.6 source order differs")
    if any(sha256(raw).hexdigest() != evidence.content_digest
           for (_, raw), evidence in zip(opened, context.base.evidence)):
        raise CognitiveKernelContractError("v1.6 exact source digest differs")
    return row["case_id"], context, opened


def generate_case(*, processor, base, specialist, context, opened_sources,
                  case_id: str, component_sha256: str, inference_run_id: str,
                  max_source_tokens: int, max_new_tokens: int) -> dict:
    import torch

    source = training.source_batch_v16(
        processor, context, opened_sources, max_source_tokens, case_id=case_id)
    device = next(specialist.parameters()).device
    payload = {name: value.to(device) for name, value in source.items()}
    with torch.inference_mode():
        states = base.model(**payload, use_cache=False,
                            return_dict=True).last_hidden_state
    if states.shape[:2] != payload["attention_mask"].shape:
        raise CognitiveKernelContractError("v1.6 prepared states do not align to source")
    token_ids, ended = shared_inference.greedy_tokens(
        specialist, base_states=states, source_mask=payload["attention_mask"],
        max_new_tokens=max_new_tokens)
    raw = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
    result = {"output_text": raw, "raw_output_text": raw,
              "generated_token_ids": token_ids, "eos_observed": ended,
              "validation_status": "invalid", "validation_error": None}
    if not ended:
        result["validation_error"] = "no EOS inside declared v1.6 target budget"
        return result
    try:
        body = json.loads(raw)
        bundle = bundle_v16_from_output(
            context, body, artifact_sha256=component_sha256,
            inference_run_id=inference_run_id)
        validate_formation_grounding_v16(context, bundle, opened_sources)
    except (ValueError, TypeError, KeyError, CognitiveKernelContractError) as exc:
        result["validation_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["validation_status"] = "grounded_probe_proposal_only"
    return result


def run(args: argparse.Namespace) -> None:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true"})
    training.shared._require_private_network_isolation()
    if require_identifier(args.inference_run_id, "inference_run_id") != args.inference_run_id or \
            len(args.inference_run_id) > 240:
        raise CognitiveKernelContractError("invalid v1.6 inference run ID")
    import torch
    from safetensors.torch import load_file
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist

    prepared, preflight, component, weight_path = verify_artifacts(
        args.component_dir, args.prepared_base_dir, args.prepared_base_receipt,
        args.preflight_receipt, control=args.control)
    config = _config(component["specialist_config"])
    if transformers.__version__ != preflight["transformers_version"] or \
            not 1 <= args.max_new_tokens <= config.max_target_tokens:
        raise CognitiveKernelContractError("v1.6 processor version or target budget differs")
    processor = AutoProcessor.from_pretrained(
        prepared["snapshot_path"], trust_remote_code=False, local_files_only=True)
    if (len(processor.tokenizer) != config.vocabulary_size or
            processor.tokenizer.bos_token_id != config.start_token_id or
            processor.tokenizer.eos_token_id != config.end_token_id or
            processor.tokenizer.pad_token_id != config.pad_token_id):
        raise CognitiveKernelContractError("v1.6 processor tokenizer differs")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForMultimodalLM.from_pretrained(
        prepared["snapshot_path"], trust_remote_code=False,
        local_files_only=True, use_safetensors=True, dtype=dtype)
    if getattr(base.config, "model_type", None) != "gemma4_unified" or \
            config.base_hidden_size != base.config.text_config.hidden_size:
        raise CognitiveKernelContractError("v1.6 prepared base geometry differs")
    base.to(device).requires_grad_(False).eval()
    specialist = FormationSpecialist(config).to(device)
    specialist.load_state_dict(load_file(str(weight_path)), strict=True)
    specialist.eval()
    if args.output_jsonl.exists() or args.output_jsonl.resolve() == args.input_jsonl.resolve():
        raise CognitiveKernelContractError("v1.6 output must be a new distinct file")
    input_sha = training.shared._digest(args.input_jsonl)
    selected_sha = training.shared._digest(weight_path)
    receipt = component["record_sha256"] if args.control == "trained" else \
        training.shared._read_sealed(args.component_dir / "seed-control.json",
                                     training.SEED_SCHEMA)["record_sha256"]
    temporary = args.output_jsonl.with_name(f".{args.output_jsonl.name}.partial-{uuid4().hex}")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output, \
                args.input_jsonl.open("r", encoding="utf-8") as source:
            seen = set()
            for number, line in enumerate(source, 1):
                case_id, context, opened = read_input(json.loads(line))
                if case_id in seen:
                    raise CognitiveKernelContractError("v1.6 duplicate inference case")
                seen.add(case_id)
                generated = generate_case(
                    processor=processor, base=base, specialist=specialist,
                    context=context, opened_sources=opened, case_id=case_id,
                    component_sha256=selected_sha,
                    inference_run_id=f"{args.inference_run_id}-{number}",
                    max_source_tokens=preflight["max_source_tokens"],
                    max_new_tokens=args.max_new_tokens)
                row = {
                    "case_id": case_id, "status": "probe_generated",
                    "context_digest": context.content_digest(),
                    "processed_modalities": sorted({
                        item.modality for item in context.base.evidence}),
                    "model_artifact_digest": selected_sha,
                    "model_receipt": receipt,
                    "formation_component_sha256": selected_sha,
                    "control_kind": args.control,
                    "probe_only": True, "qualified_for_product": False,
                    "source_repository": prepared["repository"],
                    "source_revision": prepared["revision"],
                    "base_source_sha256": training.shared.SOURCE_WEIGHT_SHA256,
                    "prepared_base_sha256": component["prepared_base_sha256"],
                    "prepared_base_parent_sha256": component["prepared_base_parent_sha256"],
                    "prepared_base_receipt_sha256": prepared["receipt_sha256"],
                    "prompt_set_sha256": input_sha,
                    "generation": {"do_sample": False,
                                   "max_new_tokens": args.max_new_tokens,
                                   "decoder": "v1.6-specialist-greedy-bos-eos"},
                    "inference_runner_sha256": training.shared._digest(Path(__file__)),
                    **generated,
                }
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.link(temporary, args.output_jsonl)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--prepared-base-dir", type=Path, required=True)
    parser.add_argument("--prepared-base-receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--control", choices=("trained", "seeded-untrained"), required=True)
    parser.add_argument("--inference-run-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
