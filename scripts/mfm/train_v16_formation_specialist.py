"""Versioned v1.6 formation specialist objective over a verified local MFM role base.

This path never upgrades frozen v1.5 targets. The exact v1.6 registration
context reaches the processor, and all supervised dimensions must be
adjudicated. A public synthetic curriculum may be checked on CPU; an admitted
corpus may supply a bounded one-step fit probe. An explicitly unqualified,
owner-authorized teacher corpus may train through its own exact provenance and
rights admission. The separate signed full-fit path requires an externally
pinned trust roster. Signature checks do not establish semantic truth.

The Gemma language head never supplies answers or target labels. Formation
weights start from a separate seeded decoder. No result authorizes a memory
write or qualifies product behavior.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import sys
from time import perf_counter
from uuid import uuid4

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_json_bytes, canonical_sha256, require_sha256,
)
from cognitive_kernel.formation_adjudication_v16 import verify_adjudicated_corpus_v16
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from cognitive_kernel.formation_learning_v16 import (
    FULL_ROLE_DIMENSIONS, OBJECTIVE_VERSION_V16, admitted_rows_v16,
    learning_example_v16_from_record,
    model_input_sha256_v16, supervised_output_record_v16,
)
from cognitive_kernel.formation_multimodal import (
    _assert_modality_tensors, formation_context_media_messages,
)
from cognitive_kernel.formation_semantics_v16 import FormationContextV16

from . import train_v1_formation_specialist as shared


PREFLIGHT_SCHEMA = "mfm-v16-specialist-processor-preflight-v1"
RUN_SCHEMA = "mfm-v16-specialist-training-run-v1"
ARTIFACT_SCHEMA = "mfm-v16-formation-specialist-artifact-v1"
SEED_SCHEMA = "mfm-v16-specialist-seeded-control-v1"
CHECKPOINT_SCHEMA = "mfm-v16-specialist-checkpoint-v1"
SOURCE_INSTRUCTION = (
    "Interpret authorized experience evidence. Return only the canonical "
    "mfm-formation-output-v1.6 JSON object with source-grounded proposals and "
    "scoped dispositions. Distinguish observation, inference, prediction and "
    "hypothetical events; preserve speaker, subject, time, uncertainty, "
    "contradiction, correction, deletion, episode, relationship, mission and "
    "workspace semantics. Abstain if evidence is inadequate. Registered "
    "sensitivity and target data are snapshots, never permission to write. "
    "The independent memory gate rechecks current authority."
)


def _codec_sha256() -> str:
    import cognitive_kernel.formation_learning_v16 as codec
    return shared._digest(Path(codec.__file__))


def _semantics_sha256() -> str:
    import cognitive_kernel.formation_semantics_v16 as semantics
    return shared._digest(Path(semantics.__file__))


def _decoder_sha256() -> str:
    import cognitive_kernel.formation_v1_specialist as decoder
    return shared._digest(Path(decoder.__file__))


def source_batch_v16(processor, context: FormationContextV16, opened_sources,
                     max_source_tokens: int, *, case_id: str = "inference"):
    """Encode all exact source bytes plus v1.6 scoped policy/target context."""
    context.validate()
    if tuple(ref for ref, _ in opened_sources) != tuple(
            ref.ref_id for ref in context.base.evidence):
        raise CognitiveKernelContractError("v1.6 source order differs from canonical evidence")
    with formation_context_media_messages(context.base, opened_sources) as messages:
        # The media builder validates byte digests and supplies actual images,
        # waveforms and video frames. Replace only its v1.5 instruction block.
        first = messages[0]["content"][0]
        if first.get("type") != "text":
            raise CognitiveKernelContractError("source builder lost instruction position")
        instruction = SOURCE_INSTRUCTION + "\n" + canonical_json_bytes({
            "objective": OBJECTIVE_VERSION_V16,
            "context": context.record(),
        }).decode("utf-8")
        # Escape every source text block, not just the instruction. A literal
        # media marker inside evidence must never become a processor token.
        versioned = [{**message, "content": [
            {**item, "text": "json_text=" + shared._escaped_text(
                instruction if message_index == 0 and item_index == 0
                else item["text"])}
            if item["type"] == "text" else item
            for item_index, item in enumerate(message["content"])]}
            for message_index, message in enumerate(messages)]
        payload = processor.apply_chat_template(
            versioned, chat_template=shared.SOURCE_TEMPLATE,
            tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors="pt", processor_kwargs={"do_sample_frames": False})
        _assert_modality_tensors(context.base, payload)
    selected = {name: value for name, value in payload.items()
                if hasattr(value, "shape") and name not in {
                    "num_soft_tokens_per_image", "num_soft_tokens_per_video"}}
    tokens = selected.get("input_ids")
    if tokens is None or tokens.ndim != 2 or tokens.shape[0] != 1 or \
            not 0 < tokens.shape[1] <= max_source_tokens:
        raise CognitiveKernelContractError(
            f"case {case_id} exceeds full v1.6 source budget; do not truncate")
    if "attention_mask" not in selected or selected["attention_mask"].shape != tokens.shape:
        raise CognitiveKernelContractError("processor omitted aligned attention mask")
    return selected


def _target_ids(tokenizer, example, maximum: int) -> tuple[list[int], list[int]]:
    answer = canonical_json_bytes(supervised_output_record_v16(example)).decode("utf-8")
    body = tokenizer.encode(answer, add_special_tokens=False)
    if not body or len(body) + 1 > maximum:
        raise CognitiveKernelContractError(
            f"case {example.case_id} exceeds full v1.6 target budget; do not truncate")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise CognitiveKernelContractError("prepared tokenizer lacks BOS/EOS")
    return [tokenizer.bos_token_id, *body], [*body, tokenizer.eos_token_id]


def _public_examples(path: Path, expected_sha256: str, owner_authorization_ref: str):
    """CPU checks only; caller explicitly attests the curriculum is public."""
    if not owner_authorization_ref:
        raise CognitiveKernelContractError("public synthetic data needs owner authorization")
    if shared._digest(path) != require_sha256(expected_sha256, "input_sha256"):
        raise CognitiveKernelContractError("v1.6 curriculum bytes differ from frozen digest")
    seen_cases, seen_inputs = set(), set()
    examples = {"train": [], "development": []}
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("authorization_id") != owner_authorization_ref:
                raise CognitiveKernelContractError("v1.6 curriculum lacks owner authorization")
            if row.get("split") not in examples:
                raise CognitiveKernelContractError("public synthetic CPU input cannot open FINAL")
            example = learning_example_v16_from_record(row, split=row["split"])
            supervised_output_record_v16(example)
            fingerprint = model_input_sha256_v16(example)
            if example.case_id in seen_cases or fingerprint in seen_inputs:
                raise CognitiveKernelContractError("duplicate v1.6 case or source input")
            seen_cases.add(example.case_id)
            seen_inputs.add(fingerprint)
            examples[example.split].append(example)
    if not examples["train"]:
        raise CognitiveKernelContractError("v1.6 curriculum is empty")
    return tuple(examples["train"]), tuple(examples["development"])


def _require_signed_fit_coverage(train, development, *, policy="signed") -> None:
    """Demand observed positives in both visible splits, never infer FINAL labels.

    This is a minimal GPU admission guard, not proof of diversity, target
    correctness, learning, or full product capability.
    """
    missing = {}
    for split, examples in (("train", train), ("development", development)):
        positives = {name for example in examples for name, state in
                     example.adjudications if state == "present"}
        absent = sorted(FULL_ROLE_DIMENSIONS - positives)
        if absent:
            missing[split] = absent
    if missing:
        raise CognitiveKernelContractError(
            f"{policy} v1.6 fit lacks positive supervised dimensions: " +
            json.dumps(missing, sort_keys=True, separators=(",", ":")))


def _examples(args):
    require_sha256(args.input_sha256, "input_sha256")
    teacher_fit = bool(getattr(args, "teacher_fit", False))
    teacher_manifest = getattr(args, "teacher_training_manifest", None)
    if teacher_fit:
        if teacher_manifest is None or args.owner_authorization_ref is None or \
                getattr(args, "full_fit", False) or args.admitted_manifest is not None or \
                args.public_synthetic_curriculum is not None:
            raise CognitiveKernelContractError("teacher fit needs its own pinned manifest and owner authorization")
        admission = admit_formation_corpus(
            teacher_manifest, expected_sha256=args.input_sha256,
            teacher_training=True, owner_authorization_ref=args.owner_authorization_ref)
        train = tuple(admitted_rows_v16(admission, split="train"))
        development = tuple(admitted_rows_v16(admission, split="development"))
        seen = set()
        for item in (*train, *development):
            supervised_output_record_v16(item)
            fingerprint = model_input_sha256_v16(item)
            if fingerprint in seen:
                raise CognitiveKernelContractError("teacher input repeats across splits")
            seen.add(fingerprint)
        _require_signed_fit_coverage(train, development, policy="teacher")
        return train, development, "owner-authorized-teacher-fit-diagnostic-development-unqualified"
    if getattr(args, "full_fit", False) and args.public_synthetic_curriculum is not None:
        raise CognitiveKernelContractError("synthetic curriculum cannot supply full fit")
    if getattr(args, "full_fit", False):
        if args.admitted_manifest is None or getattr(args, "trust_roster", None) is None or \
                getattr(args, "trust_roster_sha256", None) is None:
            raise CognitiveKernelContractError("full fit needs manifest and externally pinned roster")
        require_sha256(args.trust_roster_sha256, "trust_roster_sha256")
    if args.public_synthetic_curriculum is not None:
        if args.mode == "train":
            raise CognitiveKernelContractError("synthetic curriculum cannot supply a GPU fit")
        train, development = _public_examples(
            args.public_synthetic_curriculum, args.input_sha256,
            args.owner_authorization_ref)
        return train, development, "owner-attested-public-synthetic-cpu-only-unqualified"
    if args.owner_authorization_ref:
        raise CognitiveKernelContractError("admitted corpus does not use synthetic authorization")
    admission = admit_formation_corpus(args.admitted_manifest,
                                        expected_sha256=args.input_sha256)
    if getattr(args, "full_fit", False):
        # Base admission reads train/development bytes to hash and check
        # rights. Verify signed review before turning those bytes into any
        # optimizer examples. FINAL stays metadata-only throughout.
        signed = verify_adjudicated_corpus_v16(
            admission, args.admitted_manifest, args.trust_roster,
            expected_roster_sha256=args.trust_roster_sha256)
        if (signed.get("corpus_manifest_sha256") != args.input_sha256 or
                signed.get("externally_pinned_roster_sha256") != args.trust_roster_sha256 or
                signed.get("rights_and_review_signatures_verified") is not True or
                signed.get("final_payloads_opened") is not False):
            raise CognitiveKernelContractError("v1.6 signed corpus review receipt differs")
        args.signed_review_receipt_sha256 = canonical_sha256(signed)
    train = tuple(admitted_rows_v16(admission, split="train"))
    development = tuple(admitted_rows_v16(admission, split="development"))
    seen = set()
    for item in (*train, *development):
        supervised_output_record_v16(item)
        fingerprint = model_input_sha256_v16(item)
        if fingerprint in seen:
            raise CognitiveKernelContractError("v1.6 input repeats across train/development")
        seen.add(fingerprint)
    if not train or not development:
        raise CognitiveKernelContractError("v1.6 admission needs train and development")
    if getattr(args, "full_fit", False):
        _require_signed_fit_coverage(train, development)
    # CorpusAdmission audits FINAL's manifest metadata but never opens FINAL.
    status = ("admitted-signed-review-final-sealed-unqualified"
              if getattr(args, "full_fit", False) else
              "admitted-structural-final-sealed-unqualified")
    return train, development, status


def _binding(args, prepared, status: str, train_count: int, development_count: int,
             transformers_version: str) -> dict:
    weight = next((row["sha256"] for row in prepared["files"]
                   if row["path"] == "model.safetensors"), None)
    return {
        "objective": OBJECTIVE_VERSION_V16,
        "output_schema": "mfm-formation-output-v1.6",
        "context_schema": "mfm-formation-context-v1.6",
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "prepared_base_weight_sha256": weight,
        "prepared_base_parent_weight_sha256": shared.SOURCE_WEIGHT_SHA256,
        "prepared_base_kind": shared._prepared_kind(prepared),
        "foundation_commit": shared.FOUNDATION_COMMIT,
        "foundation_verifier_sha256": shared.FOUNDATION_VERIFIER_SHA256,
        "foundation_inventory_sha256": shared.FOUNDATION_INVENTORY_SHA256,
        "corpus_sha256": args.input_sha256, "corpus_status": status,
        "full_fit": bool(getattr(args, "full_fit", False)),
        "teacher_fit": bool(getattr(args, "teacher_fit", False)),
        "trust_roster_sha256": getattr(args, "trust_roster_sha256", None),
        "signed_review_receipt_sha256": getattr(
            args, "signed_review_receipt_sha256", None),
        "train_cases": train_count, "development_cases": development_count,
        "owner_authorization_ref": args.owner_authorization_ref,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
        "transformers_version": transformers_version,
        "source_template_sha256": sha256(shared.SOURCE_TEMPLATE.encode()).hexdigest(),
        "source_builder_sha256": shared._digest(Path(sys.modules[
            formation_context_media_messages.__module__].__file__)),
        "source_instruction_sha256": sha256(SOURCE_INSTRUCTION.encode()).hexdigest(),
        "codec_sha256": _codec_sha256(),
        "semantics_sha256": _semantics_sha256(),
        "decoder_implementation_sha256": _decoder_sha256(),
        "trainer_sha256": shared._digest(Path(__file__)),
    }


def _processor_preflight(args, train, development, status, prepared):
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(prepared["snapshot_path"],
                                              trust_remote_code=False, local_files_only=True)
    if not all(hasattr(processor, name) for name in (
            "apply_chat_template", "tokenizer", "image_token", "audio_token", "video_token")):
        raise CognitiveKernelContractError("prepared processor lacks Gemma 4 multimodal inputs")
    config = json.loads((Path(prepared["snapshot_path"]) / "config.json").read_bytes())
    vocabulary_size = config["text_config"]["vocab_size"]
    binding = _binding(args, prepared, status, len(train), len(development),
                       transformers.__version__)
    if args.mode == "processor-preflight":
        longest_source = longest_target = peak_pairs = 0
        train_source = train_target = 0
        stress_source = stress_target = stress_joint = None
        modalities = set()
        for index, example in enumerate((*train, *development)):
            encoded = source_batch_v16(processor, example.context, example.opened_sources,
                                       args.max_source_tokens, case_id=example.case_id)
            if int(encoded["input_ids"].max()) >= vocabulary_size:
                raise CognitiveKernelContractError("processor token exceeds prepared embeddings")
            inputs, _ = _target_ids(processor.tokenizer, example, args.max_target_tokens)
            length = encoded["input_ids"].shape[1]
            longest_source = max(longest_source, length)
            longest_target = max(longest_target, len(inputs))
            if index < len(train):
                pairs = length * len(inputs) * args.specialist_heads * args.specialist_layers
                if length > train_source:
                    train_source, stress_source = length, example.case_id
                if len(inputs) > train_target:
                    train_target, stress_target = len(inputs), example.case_id
                if pairs > peak_pairs:
                    peak_pairs, stress_joint = pairs, example.case_id
            modalities.update(ref.modality for ref in example.context.base.evidence)
        receipt = shared._write_new(args.preflight_receipt, {
            "schema": PREFLIGHT_SCHEMA, **binding,
            "longest_source_tokens": longest_source,
            "longest_target_tokens": longest_target,
            "peak_train_attention_pairs": peak_pairs,
            "stress_train_case_ids": list(dict.fromkeys((
                stress_source, stress_target, stress_joint))),
            "specialist_heads": args.specialist_heads,
            "specialist_layers": args.specialist_layers,
            "processed_modalities": sorted(modalities),
            "scope": "CPU v1.6 processor/data contract only; no fit or independent qualification",
        })
        return processor, receipt
    receipt = shared._read_sealed(args.preflight_receipt, PREFLIGHT_SCHEMA)
    if any(receipt.get(name) != value for name, value in binding.items()):
        raise CognitiveKernelContractError("v1.6 processor preflight binding differs")
    if receipt["longest_source_tokens"] > args.max_source_tokens or \
            receipt["longest_target_tokens"] > args.max_target_tokens or \
            receipt["specialist_heads"] != args.specialist_heads or \
            receipt["specialist_layers"] != args.specialist_layers:
        raise CognitiveKernelContractError("v1.6 processor preflight budgets differ")
    return processor, receipt


def _seed_control(output_dir: Path, specialist, run_digest: str, config):
    from safetensors.torch import save_file
    path = output_dir / "seed-control.safetensors"
    save_file({key: value.detach().cpu().contiguous()
               for key, value in specialist.state_dict().items()}, str(path))
    return shared._write_new(output_dir / "seed-control.json", {
        "schema": SEED_SCHEMA, "specialist_sha256": shared._digest(path),
        "run_manifest_sha256": run_digest, "specialist_config": config.record(),
        "optimizer_steps": 0,
        "meaning": "same-seed untrained v1.6 specialist negative control",
    })


def _verify_seed_control(output_dir: Path, run_digest: str, config):
    receipt = shared._read_sealed(output_dir / "seed-control.json", SEED_SCHEMA)
    if receipt.get("run_manifest_sha256") != run_digest or \
            receipt.get("specialist_config") != config.record() or \
            receipt.get("optimizer_steps") != 0 or \
            receipt.get("specialist_sha256") != shared._digest(
                output_dir / "seed-control.safetensors"):
        raise CognitiveKernelContractError("v1.6 seeded control differs from run")
    return receipt


def _device_plan(args, *, cuda_devices: int | None = None) -> dict[str, str]:
    """Bind explicit CUDA role placement; never silently select another GPU."""
    base = getattr(args, "base_device", "cuda:0")
    specialist = getattr(args, "specialist_device", "cuda:0")
    for name, value in (("base", base), ("specialist", specialist)):
        if not isinstance(value, str) or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) is None:
            raise CognitiveKernelContractError(f"v1.6 {name} device needs an explicit CUDA ordinal")
        if cuda_devices is not None and int(value.split(":", 1)[1]) >= cuda_devices:
            raise CognitiveKernelContractError(f"v1.6 {name} GPU is unavailable")
    return {"base": base, "specialist": specialist,
            "strategy": ("frozen-base-role-split" if base != specialist else "single-gpu")}


def _specialist_sources(states, source_mask, device):
    """Move detached frozen representations, not the base's computation graph."""
    if states.shape[:2] != source_mask.shape:
        raise CognitiveKernelContractError("prepared states do not align to source")
    return states.detach().to(device), source_mask.to(device)


def _probe_hardware(torch, devices, *, model_load_seconds: float,
                    optimizer_step_seconds: float) -> dict:
    """Observed CUDA peaks since the stress step began, not a capacity estimate."""
    return {"model_load_seconds": model_load_seconds,
            "optimizer_step_seconds": optimizer_step_seconds,
            "devices": {name: {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(name),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(name),
                "total_bytes": torch.cuda.get_device_properties(name).total_memory,
            } for name in devices}}


def _checkpoint(output_dir: Path, *, specialist, optimizer, run_digest: str,
                epoch: int, next_case: int, step: int):
    import torch
    from safetensors.torch import save_file
    import shutil

    staging = output_dir / f".checkpoint-{uuid4().hex}"
    staging.mkdir()
    try:
        save_file({name: value.detach().cpu().contiguous()
                   for name, value in specialist.state_dict().items()},
                  str(staging / "specialist.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.random.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all(),
                    "run_manifest_sha256": run_digest, "epoch": epoch,
                    "next_case": next_case, "step": step}, staging / "optimizer.pt")
        shared._write_new(staging / "checkpoint.json", {
            "schema": CHECKPOINT_SCHEMA,
            "specialist_sha256": shared._digest(staging / "specialist.safetensors"),
            "optimizer_sha256": shared._digest(staging / "optimizer.pt"),
            "run_manifest_sha256": run_digest,
            "epoch": epoch, "next_case": next_case, "step": step,
        })
        staging.rename(output_dir / f"checkpoint-{step:08d}")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _verify_checkpoint(checkpoint_dir: Path, run_digest: str):
    receipt = shared._read_sealed(checkpoint_dir / "checkpoint.json", CHECKPOINT_SCHEMA)
    if receipt.get("run_manifest_sha256") != run_digest or \
            receipt.get("specialist_sha256") != shared._digest(
                checkpoint_dir / "specialist.safetensors") or \
            receipt.get("optimizer_sha256") != shared._digest(
                checkpoint_dir / "optimizer.pt"):
        raise CognitiveKernelContractError("v1.6 checkpoint differs from run")
    return receipt


def _resume(checkpoint_dir, output_dir, specialist, optimizer, run_digest):
    import torch
    from safetensors.torch import load_file
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    if checkpoint_dir.parent != output_dir.resolve(strict=True) or \
            not checkpoint_dir.name.startswith("checkpoint-"):
        raise CognitiveKernelContractError("v1.6 checkpoint is outside run")
    receipt = _verify_checkpoint(checkpoint_dir, run_digest)
    state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
    for field in ("run_manifest_sha256", "epoch", "next_case", "step"):
        if state[field] != receipt[field]:
            raise CognitiveKernelContractError(f"v1.6 resume differs: {field}")
    specialist.load_state_dict(load_file(str(checkpoint_dir / "specialist.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    torch.random.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    return receipt["epoch"], receipt["next_case"], receipt["step"]


def _run_manifest_v16(args, prepared, preflight, config, torch_version: str) -> dict:
    """Bind the selected admission into the immutable run and resume digest."""
    if getattr(args, "full_fit", False):
        roster = getattr(args, "trust_roster_sha256", None)
        review = getattr(args, "signed_review_receipt_sha256", None)
        if roster is None or review is None:
            raise CognitiveKernelContractError("full fit has no verified review binding")
        require_sha256(roster, "trust_roster_sha256")
        require_sha256(review, "signed_review_receipt_sha256")
    return {
        "schema": RUN_SCHEMA, "objective": OBJECTIVE_VERSION_V16,
        "preflight_sha256": preflight["record_sha256"],
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "corpus_sha256": args.input_sha256,
        "full_fit": bool(getattr(args, "full_fit", False)),
        "teacher_fit": bool(getattr(args, "teacher_fit", False)),
        "trust_roster_sha256": getattr(args, "trust_roster_sha256", None),
        "signed_review_receipt_sha256": getattr(
            args, "signed_review_receipt_sha256", None),
        "specialist_config": config.record(), "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation, "seed": args.seed,
        "max_cross_attention_pairs": args.max_cross_attention_pairs,
        "device_placement": _device_plan(args),
        "torch_version": torch_version,
        "transformers_version": preflight["transformers_version"],
        "probe_only": args.probe_only, "qualified_for_product": False,
        "weight_lineage": shared._prepared_kind(prepared) + "+fresh-v16-formation-weights",
    }


def _run_training(args, train, development, status, prepared, processor, preflight):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig

    stress = preflight["stress_train_case_ids"]
    if args.probe_only and (args.gradient_accumulation < len(stress) or
                            len(train) < args.gradient_accumulation):
        raise CognitiveKernelContractError(
            "probe needs stress examples in one full accumulation step")
    if not torch.cuda.is_available():
        raise CognitiveKernelContractError("v1.6 specialist training requires BF16 CUDA")
    placement = _device_plan(args, cuda_devices=torch.cuda.device_count())
    for name in {placement["base"], placement["specialist"]}:
        with torch.cuda.device(name):
            if not torch.cuda.is_bf16_supported():
                raise CognitiveKernelContractError(f"v1.6 {name} lacks BF16 CUDA")
    if args.max_cross_attention_pairs is None or \
            preflight["peak_train_attention_pairs"] > args.max_cross_attention_pairs:
        raise CognitiveKernelContractError("v1.6 probe exceeds declared cross-attention cap")
    base_device = torch.device(placement["base"])
    specialist_device = torch.device(placement["specialist"])
    devices = tuple(sorted({placement["base"], placement["specialist"]}))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    load_start = perf_counter()
    base_kwargs = dict(trust_remote_code=False, local_files_only=True,
                       use_safetensors=True, dtype=torch.bfloat16)
    if placement["strategy"] == "frozen-base-role-split":
        # Stream the frozen base directly onto its assigned GPU. The specialist
        # and optimizer are constructed separately on the other GPU below.
        base_kwargs.update(device_map={"": placement["base"]}, low_cpu_mem_usage=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        prepared["snapshot_path"], **base_kwargs)
    if getattr(base.config, "model_type", None) != "gemma4_unified" or \
            not hasattr(base, "model"):
        raise CognitiveKernelContractError("prepared base has no Gemma 4 representation path")
    if placement["strategy"] == "single-gpu":
        base.to(base_device)
    elif any(parameter.device != base_device for parameter in base.parameters()):
        raise CognitiveKernelContractError("v1.6 frozen base was offloaded or mis-placed")
    base.requires_grad_(False).eval()
    config = SpecialistConfig(
        base_hidden_size=base.config.text_config.hidden_size,
        vocabulary_size=len(processor.tokenizer), width=args.specialist_width,
        layers=args.specialist_layers, heads=args.specialist_heads,
        max_target_tokens=args.max_target_tokens,
        pad_token_id=processor.tokenizer.pad_token_id,
        start_token_id=processor.tokenizer.bos_token_id,
        end_token_id=processor.tokenizer.eos_token_id,
        logit_chunk_tokens=args.logit_chunk_tokens)
    specialist = FormationSpecialist(config).to(specialist_device)
    optimizer = torch.optim.AdamW(specialist.parameters(), lr=args.learning_rate)
    for name in devices:
        torch.cuda.synchronize(name)
    model_load_seconds = perf_counter() - load_start
    run = _run_manifest_v16(args, prepared, preflight, config, torch.__version__)
    digest = shared._record_hash(run)
    if args.resume_checkpoint:
        if args.probe_only or not args.output_dir.is_dir() or (
                args.output_dir / "formation-component.json").exists():
            raise CognitiveKernelContractError("resume needs incomplete matching v1.6 run")
        previous = shared._read_sealed(args.output_dir / "run.json", RUN_SCHEMA)
        if previous["record_sha256"] != digest:
            raise CognitiveKernelContractError("v1.6 resume run configuration differs")
        _verify_seed_control(args.output_dir, digest, config)
        start_epoch, start_case, steps = _resume(
            args.resume_checkpoint, args.output_dir, specialist, optimizer, digest)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shared._write_new(args.output_dir / "run.json", run)
        _seed_control(args.output_dir, specialist, digest, config)
        start_epoch = start_case = steps = 0
    losses = []
    probe_hardware = None
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        if args.probe_only:
            order.sort(key=lambda index: stress.index(train[index].case_id)
                       if train[index].case_id in stress else len(stress))
        if epoch == start_epoch:
            if not 0 <= start_case <= len(order):
                raise CognitiveKernelContractError("v1.6 resume cursor is invalid")
            order = order[start_case:]
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        if args.probe_only:
            for name in devices:
                torch.cuda.synchronize(name)
                torch.cuda.reset_peak_memory_stats(name)
            step_start = perf_counter()
        for index, case_index in enumerate(order):
            example = train[case_index]
            encoded = source_batch_v16(processor, example.context, example.opened_sources,
                                       args.max_source_tokens, case_id=example.case_id)
            inputs, labels = _target_ids(processor.tokenizer, example, args.max_target_tokens)
            payload = {key: value.to(base_device) for key, value in encoded.items()}
            with torch.no_grad():
                states = base.model(**payload, use_cache=False,
                                    return_dict=True).last_hidden_state
            states, source_mask = _specialist_sources(
                states, payload["attention_mask"], specialist_device)
            ids = torch.tensor([inputs], device=specialist_device)
            target = torch.tensor([labels], device=specialist_device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = specialist(base_states=states,
                                     source_mask=source_mask,
                                     input_ids=ids, labels=target)
            if not torch.isfinite(loss):
                raise CognitiveKernelContractError("v1.6 training loss is nonfinite")
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            pending += 1
            if pending == args.gradient_accumulation or index == len(order) - 1:
                if pending != args.gradient_accumulation:
                    for parameter in specialist.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(args.gradient_accumulation / pending)
                torch.nn.utils.clip_grad_norm_(specialist.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                steps += 1
                if args.probe_only:
                    for name in devices:
                        torch.cuda.synchronize(name)
                    probe_hardware = _probe_hardware(
                        torch, devices, model_load_seconds=model_load_seconds,
                        optimizer_step_seconds=perf_counter() - step_start)
                if steps % args.save_every_steps == 0 or args.probe_only:
                    _checkpoint(args.output_dir, specialist=specialist,
                                optimizer=optimizer, run_digest=digest,
                                epoch=epoch,
                                next_case=(start_case + index + 1 if epoch == start_epoch
                                           else index + 1), step=steps)
                if args.probe_only:
                    _verify_checkpoint(args.output_dir / f"checkpoint-{steps:08d}", digest)
                    break
        if args.probe_only:
            break
    if steps < 1:
        raise CognitiveKernelContractError("v1.6 training produced no optimizer step")
    final = args.output_dir / "formation-specialist.safetensors"
    save_file({key: value.detach().cpu().contiguous()
               for key, value in specialist.state_dict().items()}, str(final))
    seed = _verify_seed_control(args.output_dir, digest, config)
    if shared._digest(final) == seed["specialist_sha256"]:
        raise CognitiveKernelContractError("v1.6 probe did not change specialist weights")
    component = shared._write_new(args.output_dir / "formation-component.json", {
        "schema": ARTIFACT_SCHEMA, "objective": OBJECTIVE_VERSION_V16,
        "formation_component_sha256": shared._digest(final),
        "prepared_base_sha256": preflight["prepared_base_weight_sha256"],
        "prepared_base_parent_sha256": preflight["prepared_base_parent_weight_sha256"],
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "prepared_base_kind": preflight["prepared_base_kind"],
        "run_manifest_sha256": digest,
        "full_fit": bool(getattr(args, "full_fit", False)),
        "teacher_fit": bool(getattr(args, "teacher_fit", False)),
        "trust_roster_sha256": getattr(args, "trust_roster_sha256", None),
        "signed_review_receipt_sha256": getattr(
            args, "signed_review_receipt_sha256", None),
        "seed_control_sha256": seed["specialist_sha256"],
        "device_placement": placement,
        "specialist_config": config.record(), "optimizer_steps": steps,
        "probe_hardware": probe_hardware,
        "mean_training_loss_for_current_segment": (
            sum(losses) / len(losses) if losses else None),
        "development_cases": len(development), "development_evaluated": False,
        "probe_only": args.probe_only, "qualified_for_product": False,
        "meaning": "v1.6 fit without independent behavior or FINAL qualification",
    })
    return {"component_sha256": component["record_sha256"],
            "optimizer_steps": steps, "probe_only": args.probe_only,
            "probe_hardware": probe_hardware,
            "qualified_for_product": False}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("data-preflight", "processor-preflight", "train"))
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--admitted-manifest", type=Path)
    corpus.add_argument("--public-synthetic-curriculum", type=Path)
    corpus.add_argument("--teacher-training-manifest", type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--owner-authorization-ref")
    parser.add_argument("--prepared-base-dir", type=Path)
    parser.add_argument("--prepared-base-receipt", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-source-tokens", type=int, default=32768)
    parser.add_argument("--max-target-tokens", type=int, default=8192)
    parser.add_argument("--specialist-width", type=int, default=768)
    parser.add_argument("--specialist-layers", type=int, default=6)
    parser.add_argument("--specialist-heads", type=int, default=12)
    parser.add_argument("--logit-chunk-tokens", type=int, default=64)
    parser.add_argument("--max-cross-attention-pairs", type=int)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--base-device", default="cuda:0",
                        help="explicit CUDA device for the frozen Gemma representation base")
    parser.add_argument("--specialist-device", default="cuda:0",
                        help="explicit CUDA device for the trainable formation specialist")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--full-fit", action="store_true",
                        help="require signed admission with an externally pinned roster")
    parser.add_argument("--teacher-fit", action="store_true",
                        help="owner-authorized training-only fit; never independent qualification")
    parser.add_argument("--trust-roster", type=Path,
                        help="steward-controlled rights/reviewer trust roster")
    parser.add_argument("--trust-roster-sha256",
                        help="exact roster hash pinned independently of the corpus")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if any(type(value) is not int or value < 1 for value in (
            args.max_source_tokens, args.max_target_tokens, args.specialist_width,
            args.specialist_layers, args.specialist_heads, args.logit_chunk_tokens,
            args.epochs, args.gradient_accumulation, args.save_every_steps)) or \
            args.learning_rate <= 0:
        raise CognitiveKernelContractError("invalid v1.6 training sizes")
    if args.max_cross_attention_pairs is not None and args.max_cross_attention_pairs < 1:
        raise CognitiveKernelContractError("invalid v1.6 cross-attention cap")
    if args.probe_only and args.mode != "train":
        raise CognitiveKernelContractError("probe-only applies only to training")
    full_fit = bool(getattr(args, "full_fit", False))
    teacher_fit = bool(getattr(args, "teacher_fit", False))
    teacher_manifest = getattr(args, "teacher_training_manifest", None)
    roster = getattr(args, "trust_roster", None)
    roster_sha = getattr(args, "trust_roster_sha256", None)
    if full_fit and args.probe_only:
        raise CognitiveKernelContractError("full fit and one-step probe are separate routes")
    if full_fit and teacher_fit:
        raise CognitiveKernelContractError("signed and teacher fit are separate routes")
    if (teacher_fit and (teacher_manifest is None or
                         getattr(args, "admitted_manifest", None) is not None or
                         args.public_synthetic_curriculum is not None or
                         not getattr(args, "owner_authorization_ref", None))) or \
            (teacher_manifest is not None and not teacher_fit):
        raise CognitiveKernelContractError("teacher fit needs its own pinned manifest and owner authorization")
    if full_fit and (args.public_synthetic_curriculum is not None or
                     getattr(args, "admitted_manifest", None) is None):
        raise CognitiveKernelContractError("v1.6 full fit requires an admitted corpus")
    if full_fit:
        if roster is None or roster_sha is None:
            raise CognitiveKernelContractError(
                "v1.6 full fit needs an externally pinned signed trust roster")
        require_sha256(roster_sha, "trust_roster_sha256")
    elif roster is not None or roster_sha is not None:
        raise CognitiveKernelContractError("trust roster flags require explicit full fit")
    if args.resume_checkpoint and (args.mode != "train" or args.probe_only):
        raise CognitiveKernelContractError("resume applies only to full v1.6 training")
    if args.mode == "train" and not (args.probe_only or full_fit or teacher_fit):
        raise CognitiveKernelContractError(
            "full v1.6 fit needs independent adjudication, signed admission and external roster")
    if args.mode == "train" and args.public_synthetic_curriculum:
        raise CognitiveKernelContractError("synthetic curriculum cannot supply a GPU fit")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true"})
    # Explicitly public/synthetic CPU tests may run on a connected compute
    # node. All admitted data and every GPU path require a loopback-only NS.
    if args.admitted_manifest is not None or teacher_manifest is not None or args.mode == "train":
        shared._require_private_network_isolation()
    train, development, status = _examples(args)
    if args.mode == "data-preflight":
        print(json.dumps({"objective": OBJECTIVE_VERSION_V16,
                          "corpus_status": status, "train_cases": len(train),
                          "development_cases": len(development),
                          "input_sha256": args.input_sha256,
                          "qualified_for_product": False}, sort_keys=True))
        return
    if any(item is None for item in (
            args.prepared_base_dir, args.prepared_base_receipt, args.preflight_receipt)):
        raise CognitiveKernelContractError("processor requires prepared base and receipt")
    prepared = shared._prepared_base(args)
    processor, preflight = _processor_preflight(args, train, development,
                                                status, prepared)
    if args.mode == "processor-preflight":
        print(json.dumps({"preflight_sha256": preflight["record_sha256"],
                          "longest_source_tokens": preflight["longest_source_tokens"],
                          "longest_target_tokens": preflight["longest_target_tokens"],
                          "qualified_for_product": False}, sort_keys=True))
        return
    if args.output_dir is None or (args.output_dir.exists() and not args.resume_checkpoint) or \
            (args.resume_checkpoint and not args.output_dir.is_dir()):
        raise CognitiveKernelContractError("v1.6 training needs new directory or exact resume")
    print(json.dumps(_run_training(args, train, development, status, prepared,
                                   processor, preflight), sort_keys=True))


if __name__ == "__main__":
    main()
