"""Run the first-party V1 formation decoder over a verified local MFM role base.

The prepared Gemma language head is never used for generation. Input rows hold
canonical context metadata and exact opened source bytes (base64). Generated
JSON is retained verbatim even when invalid. This is an evaluation/runtime
path; a completed run does not qualify the model or authorize memory writes.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from cognitive_kernel.canonical import CognitiveKernelContractError, require_identifier
from cognitive_kernel.formation_contracts import validate_formation_grounding
from cognitive_kernel.formation_learning import bundle_from_output, context_from_record

from . import train_v1_formation_specialist as training


def verify_artifacts(component_dir: Path, prepared_base_dir: Path,
                     prepared_base_receipt: Path, preflight_receipt: Path, *,
                     control: str) -> tuple[dict, dict, dict, Path]:
    """Rehash every ancestor and bind the selected decoder to sealed run data."""
    if control not in ("trained", "seeded-untrained"):
        raise CognitiveKernelContractError("unknown specialist control")
    prepared = training._prepared_base(SimpleNamespace(
        prepared_base_dir=prepared_base_dir,
        prepared_base_receipt=prepared_base_receipt))
    run = training._read_sealed(component_dir / "run.json", training.RUN_SCHEMA)
    preflight = training._read_sealed(preflight_receipt, training.PREFLIGHT_SCHEMA)
    component = training._read_sealed(component_dir / "formation-component.json",
                                      training.ARTIFACT_SCHEMA)
    config = component["specialist_config"]
    seed = training._verify_seed_control(component_dir, run["record_sha256"],
                                          _config(config))
    if run.get("probe_only") is not False or component.get("probe_only") is not False:
        raise CognitiveKernelContractError("hardware probe cannot be evaluated as trained MFM")
    if (run.get("preflight_sha256") != preflight["record_sha256"] or
            preflight.get("objective") != training.OBJECTIVE_VERSION or
            run.get("prepared_base_receipt_sha256") != prepared["receipt_sha256"] or
            component.get("run_manifest_sha256") != run["record_sha256"] or
            component.get("prepared_base_receipt_sha256") != prepared["receipt_sha256"] or
            run.get("corpus_sha256") != preflight.get("corpus_sha256") or
            component.get("specialist_config") != run.get("specialist_config") or
            component.get("prepared_base_sha256") != preflight.get("prepared_base_weight_sha256") or
            component.get("prepared_base_parent_sha256") != preflight.get("prepared_base_parent_weight_sha256") or
            component.get("prepared_base_kind") != preflight.get("prepared_base_kind") or
            component.get("seed_control_sha256") != seed["specialist_sha256"] or
            type(component.get("optimizer_steps")) is not int or
            component["optimizer_steps"] < 1 or
            preflight.get("specialist_heads") != config["heads"] or
            preflight.get("specialist_layers") != config["layers"] or
            preflight.get("max_target_tokens") != config["max_target_tokens"] or
            type(preflight.get("max_source_tokens")) is not int or
            preflight["max_source_tokens"] < 1 or
            preflight.get("prepared_base_receipt_sha256") != prepared["receipt_sha256"] or
            preflight.get("foundation_commit") != training.FOUNDATION_COMMIT or
            preflight.get("foundation_verifier_sha256") != training.FOUNDATION_VERIFIER_SHA256 or
            preflight.get("foundation_inventory_sha256") != training.FOUNDATION_INVENTORY_SHA256 or
            preflight.get("source_template_sha256") != sha256(training.SOURCE_TEMPLATE.encode()).hexdigest() or
            preflight.get("source_builder_sha256") != training._digest(Path(
                __import__(training.formation_context_media_messages.__module__,
                           fromlist=["__file__"]).__file__)) or
            preflight.get("trainer_sha256") != training._digest(Path(training.__file__))):
        raise CognitiveKernelContractError("component, run, preflight or prepared base lineage differs")
    weight = next((row["sha256"] for row in prepared["files"]
                   if row["path"] == "model.safetensors"), None)
    if weight != component["prepared_base_sha256"] or \
            component["prepared_base_parent_sha256"] != training.SOURCE_WEIGHT_SHA256:
        raise CognitiveKernelContractError("prepared base weight lineage differs")
    trained_file = component_dir / "formation-specialist.safetensors"
    if training._digest(trained_file) != component.get("formation_component_sha256"):
        raise CognitiveKernelContractError("specialist weight bytes differ from sealed component")
    weight_file = trained_file if control == "trained" else component_dir / "seed-control.safetensors"
    if control == "seeded-untrained" and training._digest(weight_file) != seed["specialist_sha256"]:
        raise CognitiveKernelContractError("seeded control weight bytes differ")
    return prepared, preflight, component, weight_file


def _config(record: dict):
    from cognitive_kernel.formation_v1_specialist import SpecialistConfig

    if not isinstance(record, dict) or set(record) != set(SpecialistConfig.__dataclass_fields__):
        raise CognitiveKernelContractError("specialist configuration fields differ")
    config = SpecialistConfig(**record)
    config.validate()
    if config.record() != record:
        raise CognitiveKernelContractError("specialist configuration is not canonical")
    return config


def read_input(row: dict) -> tuple[str, object, tuple[tuple[str, bytes], ...]]:
    if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
        raise CognitiveKernelContractError("inference case needs a case ID")
    context = context_from_record(row.get("context"))
    if row.get("context_digest") != context.content_digest():
        raise CognitiveKernelContractError("inference context digest differs")
    source_rows = row.get("opened_sources")
    if not isinstance(source_rows, list):
        raise CognitiveKernelContractError("inference case needs exact opened source bytes")
    opened = []
    for item in source_rows:
        if not isinstance(item, dict) or set(item) != {"ref_id", "payload_base64"}:
            raise CognitiveKernelContractError("source bytes record differs")
        try:
            raw = base64.b64decode(item["payload_base64"], validate=True)
        except (binascii.Error, TypeError, ValueError) as exc:
            raise CognitiveKernelContractError("invalid opened source encoding") from exc
        opened.append((item["ref_id"], raw))
    opened = tuple(opened)
    if tuple(key for key, _ in opened) != tuple(ref.ref_id for ref in context.evidence):
        raise CognitiveKernelContractError("inference source order differs from canonical evidence")
    if any(sha256(raw).hexdigest() != ref.content_digest
           for (_, raw), ref in zip(opened, context.evidence)):
        raise CognitiveKernelContractError("inference source digest differs")
    return row["case_id"], context, opened


def greedy_tokens(specialist, *, base_states, source_mask, max_new_tokens: int) -> tuple[list[int], bool]:
    """Deterministic BOS→EOS decode with a hard cap; no full-prefix LM projection."""
    import torch

    config = specialist.config
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= config.max_target_tokens:
        raise CognitiveKernelContractError("invalid specialist decode budget")
    prefix = [config.start_token_id]
    generated = []
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if base_states.device.type == "cuda" else nullcontext())
    with torch.inference_mode(), autocast:
        for _ in range(max_new_tokens):
            inputs = torch.tensor([prefix], dtype=torch.long, device=base_states.device)
            logits = specialist.next_token_logits(
                base_states=base_states, source_mask=source_mask, input_ids=inputs)
            next_id = int(torch.argmax(logits[0]).item())
            if next_id == config.end_token_id:
                return generated, True
            generated.append(next_id)
            prefix.append(next_id)
    return generated, False


def generate_case(*, processor, base, specialist, context, opened_sources,
                  case_id: str, component_sha256: str, inference_run_id: str,
                  max_source_tokens: int, max_new_tokens: int) -> dict:
    import torch

    source = training.source_batch(processor, context, opened_sources,
                                   max_source_tokens, case_id=case_id)
    device = next(specialist.parameters()).device
    payload = {name: value.to(device) for name, value in source.items()}
    with torch.inference_mode():
        states = base.model(**payload, use_cache=False,
                            return_dict=True).last_hidden_state
    if states.shape[:2] != payload["attention_mask"].shape:
        raise CognitiveKernelContractError("prepared base states do not align to source tokens")
    token_ids, ended = greedy_tokens(specialist, base_states=states,
                                      source_mask=payload["attention_mask"],
                                      max_new_tokens=max_new_tokens)
    raw = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
    result = {"output_text": raw, "raw_output_text": raw,
              "generated_token_ids": token_ids, "eos_observed": ended,
              "validation_status": "invalid", "validation_error": None}
    if not ended:
        result["validation_error"] = "no EOS inside the declared target budget"
        return result
    try:
        body = json.loads(raw)
        bundle = bundle_from_output(context, body,
                                    artifact_sha256=component_sha256,
                                    inference_run_id=inference_run_id)
        validate_formation_grounding(context, bundle, opened_sources)
    except (ValueError, TypeError, KeyError, CognitiveKernelContractError) as exc:
        result["validation_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["validation_status"] = "grounded_proposal_only"
    return result


def run(args: argparse.Namespace) -> None:
    # The boundary is established before opening any private corpus bytes.
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true"})
    training._require_private_network_isolation()
    if (require_identifier(args.inference_run_id, "inference_run_id") !=
            args.inference_run_id or len(args.inference_run_id) > 240):
        raise CognitiveKernelContractError("inference run ID must be canonical and allow case suffixes")
    import torch
    from safetensors.torch import load_file
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist

    prepared, preflight, component, weight_file = verify_artifacts(
        args.component_dir, args.prepared_base_dir, args.prepared_base_receipt,
        args.preflight_receipt,
        control=args.control)
    config = _config(component["specialist_config"])
    if (transformers.__version__ != preflight.get("transformers_version") or
            preflight.get("transformers_version") != training._read_sealed(
                args.component_dir / "run.json", training.RUN_SCHEMA).get("transformers_version")):
        raise CognitiveKernelContractError("inference processor version differs from frozen run")
    if args.max_new_tokens > config.max_target_tokens:
        raise CognitiveKernelContractError("decode budget exceeds trained position budget")
    processor = AutoProcessor.from_pretrained(prepared["snapshot_path"],
                                              trust_remote_code=False, local_files_only=True)
    if (len(processor.tokenizer) != config.vocabulary_size or
            processor.tokenizer.bos_token_id != config.start_token_id or
            processor.tokenizer.eos_token_id != config.end_token_id or
            processor.tokenizer.pad_token_id != config.pad_token_id):
        raise CognitiveKernelContractError("processor tokenizer differs from trained specialist")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForMultimodalLM.from_pretrained(
        prepared["snapshot_path"], trust_remote_code=False,
        local_files_only=True, use_safetensors=True, dtype=dtype)
    if getattr(base.config, "model_type", None) != "gemma4_unified" or \
            config.base_hidden_size != base.config.text_config.hidden_size:
        raise CognitiveKernelContractError("prepared base hidden geometry differs")
    base.to(device).requires_grad_(False).eval()
    specialist = FormationSpecialist(config).to(device)
    specialist.load_state_dict(load_file(str(weight_file)), strict=True)
    specialist.eval()
    if args.output_jsonl.exists() or args.output_jsonl.resolve() == args.input_jsonl.resolve():
        raise CognitiveKernelContractError("inference output must be a new file distinct from input")
    input_sha = training._digest(args.input_jsonl)
    selected_digest = training._digest(weight_file)
    expected_digest = component["formation_component_sha256"] if args.control == "trained" \
        else training._read_sealed(args.component_dir / "seed-control.json",
                                   training.SEED_CONTROL_SCHEMA)["specialist_sha256"]
    if selected_digest != expected_digest:
        raise CognitiveKernelContractError("loaded specialist bytes differ from sealed run")
    # Preserve private outputs under owner custody. Do not send them to the Hub.
    temporary = args.output_jsonl.with_name(
        f".{args.output_jsonl.name}.partial-{uuid4().hex}")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output, \
                args.input_jsonl.open("r", encoding="utf-8") as source:
            case_ids = set()
            for line_no, line in enumerate(source, 1):
                row = json.loads(line)
                case_id, context, opened = read_input(row)
                if case_id in case_ids:
                    raise CognitiveKernelContractError(f"duplicate inference case {case_id}")
                case_ids.add(case_id)
                inference_id = f"{args.inference_run_id}-{line_no}"
                generated = generate_case(
                    processor=processor, base=base, specialist=specialist, context=context,
                    opened_sources=opened, case_id=case_id,
                    component_sha256=selected_digest, inference_run_id=inference_id,
                    max_source_tokens=preflight["max_source_tokens"],
                    max_new_tokens=args.max_new_tokens)
                result = {
                    "case_id": case_id, "status": "generated",
                    "context_digest": context.content_digest(),
                    "processed_modalities": sorted({ref.modality for ref in context.evidence}),
                    "model_artifact_digest": selected_digest,
                    "model_receipt": component["record_sha256"] if args.control == "trained"
                                     else training._read_sealed(args.component_dir / "seed-control.json",
                                                                training.SEED_CONTROL_SCHEMA)["record_sha256"],
                    "formation_component_sha256": selected_digest,
                    "control_kind": args.control,
                    "source_repository": prepared["repository"],
                    "source_revision": prepared["revision"],
                    "base_source_sha256": training.SOURCE_WEIGHT_SHA256,
                    "prepared_base_sha256": component["prepared_base_sha256"],
                    "prepared_base_parent_sha256": component["prepared_base_parent_sha256"],
                    "prepared_base_receipt_sha256": prepared["receipt_sha256"],
                    "prompt_set_sha256": input_sha,
                    "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                                   "decoder": "specialist-greedy-bos-eos-v1"},
                    **generated,
                }
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        # Same-directory hard link atomically publishes only a completed file,
        # and refuses to replace an output another process created meanwhile.
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
