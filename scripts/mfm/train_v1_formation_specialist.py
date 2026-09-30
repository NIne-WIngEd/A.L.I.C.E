"""V1 MFM specialist: verified local Gemma representations, fresh formation weights.

The immutable 23,919,549,408-byte publisher checkpoint is only an ancestor.
This command requires a separately materialized, verified MFM-role clone or
an evidence-backed modified derivative. It never admits a bare publisher
snapshot. Gemma's LM head never produces the specialist's answer or its loss.
The independent decoder learns the existing source-linked proposal/disposition
contract from admitted or explicitly authorized training examples. Its output
is a proposal; the independent Claim gate alone may authorize memory writes.

CPU data admission can run without Torch. Processor preflight requires the
prepared base, Torch, Transformers and media decoders but no GPU. Training
requires its exact processor preflight receipt and a BF16 CUDA allocation.
This is an executable training path, not a qualified product model.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import socket
import sys
from uuid import uuid4

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes, require_sha256
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from cognitive_kernel.formation_learning import (
    OBJECTIVE_VERSION, admitted_rows, curriculum_rows, mixture_rows, output_record,
)
from cognitive_kernel.formation_multimodal import (
    _assert_modality_tensors, formation_media_messages,
)


PREFLIGHT_SCHEMA = "mfm-v1-specialist-processor-preflight-v1"
ARTIFACT_SCHEMA = "mfm-v1-formation-specialist-artifact-v1"
RUN_SCHEMA = "mfm-v1-specialist-training-run-v1"
FOUNDATION_COMMIT = "3e1328410dac52ba18be243cb9f7144e7a3964d1"
FOUNDATION_VERIFIER_SHA256 = "e6b36eb921d34b067703cf77adc81051923dc2b9bf82866f75176c9ebeb53e9f"
FOUNDATION_INVENTORY_SHA256 = "be1cd95b5db1544f3750fb278973aa88d0361633dc4fe5a53649d9d98c029737"
SOURCE_WEIGHT_SHA256 = "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a"
CLONE_SCHEMA = "alice-gemma4-v1-clone-v1"
DERIVATIVE_SCHEMA = "alice-gemma4-v1-derivative-v1"
# The non-instruction-tuned publisher tokenizer has no chat template. This
# first-party template renders a single source packet for the processor and
# keeps native media placeholders. Its exact bytes are bound to preflight.
SOURCE_TEMPLATE = (
    "{% for message in messages %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}\n"
    "{% elif item['type'] == 'image' %}<|image|>\n"
    "{% elif item['type'] == 'audio' %}<|audio|>\n"
    "{% elif item['type'] == 'video' %}<|video|>\n"
    "{% else %}{{ raise_exception('unsupported MFM media type') }}{% endif %}"
    "{% endfor %}{% endfor %}"
)


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _record_hash(row: dict) -> str:
    return sha256(canonical_json_bytes(row)).hexdigest()


def _seal(row: dict) -> dict:
    return {**row, "record_sha256": _record_hash(row)}


def _read_sealed(path: Path, schema: str) -> dict:
    row = json.loads(path.read_bytes())
    if not isinstance(row, dict) or row.get("schema") != schema:
        raise CognitiveKernelContractError("unexpected specialist receipt schema")
    digest = row.pop("record_sha256", None)
    if digest != _record_hash(row):
        raise CognitiveKernelContractError("specialist receipt digest differs")
    row["record_sha256"] = digest
    return row


def _write_new(path: Path, row: dict) -> dict:
    receipt = _seal(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(receipt).decode("utf-8") + "\n")
    return receipt


def _examples(args: argparse.Namespace):
    require_sha256(args.input_sha256, "input_sha256")
    if args.admitted_manifest:
        if args.owner_authorization_ref:
            raise CognitiveKernelContractError("admitted corpus does not take synthetic authorization")
        admission = admit_formation_corpus(args.admitted_manifest,
                                           expected_sha256=args.input_sha256)
        train = tuple(admitted_rows(admission, split="train"))
        development = tuple(admitted_rows(admission, split="development"))
        status = "admitted-train-development-final-sealed"
    else:
        if not args.owner_authorization_ref:
            raise CognitiveKernelContractError("synthetic curriculum requires owner authorization")
        if args.curriculum_manifest:
            train = tuple(mixture_rows(args.curriculum_manifest,
                                       expected_sha256=args.input_sha256,
                                       owner_authorization_ref=args.owner_authorization_ref))
        else:
            train = tuple(curriculum_rows(args.curriculum,
                                          expected_sha256=args.input_sha256,
                                          owner_authorization_ref=args.owner_authorization_ref))
        development = ()
        status = "owner-authorized-training-only-unqualified"
    if not train:
        raise CognitiveKernelContractError("no MFM training examples")
    for example in (*train, *development):
        example.validate()
    return train, development, status


def _require_private_network_isolation() -> None:
    """Fail before private bytes are opened unless this namespace is loopback-only.

    This is a local precondition, not a proof against a compromised host or
    same-host Unix socket proxy. The operator must still supply an isolated
    process/container with private output custody.
    """
    if not sys.platform.startswith("linux"):
        raise CognitiveKernelContractError("private MFM admission needs a Linux isolated network namespace")
    try:
        interfaces = {name for _, name in socket.if_nameindex()}
    except OSError as exc:
        raise CognitiveKernelContractError(
            "cannot verify a loopback-only namespace before opening MFM sources") from exc
    if interfaces != {"lo"}:
        raise CognitiveKernelContractError(
            "private MFM evidence requires a loopback-only network namespace before opening data")


def _prepared_base(args: argparse.Namespace) -> dict:
    # The foundation code is versioned in the shared Gemma branch. In the
    # product it is an installed first-party package; no Hub lookup occurs.
    try:
        from alice_foundation import gemma4_v1 as foundation
        from alice_foundation import gemma4_inventory
    except ImportError as exc:
        raise CognitiveKernelContractError(
            "install/include alice_foundation from the pinned V1 foundation branch") from exc
    if (_digest(Path(foundation.__file__)) != FOUNDATION_VERIFIER_SHA256 or
            _digest(Path(gemma4_inventory.__file__)) != FOUNDATION_INVENTORY_SHA256):
        raise CognitiveKernelContractError("installed foundation verifier differs from pinned V1 commit")
    receipt = foundation.verify_role_base(
        args.prepared_base_receipt, snapshot=args.prepared_base_dir, expected_role="mfm")
    if receipt.get("role") != "mfm" or receipt.get("schema") not in \
            (CLONE_SCHEMA, DERIVATIVE_SCHEMA):
        raise CognitiveKernelContractError("prepared base is not a verified MFM role copy")
    weight = next((row.get("sha256") for row in receipt.get("files", [])
                   if row.get("path") == "model.safetensors"), None)
    if receipt["schema"] == CLONE_SCHEMA:
        if weight != SOURCE_WEIGHT_SHA256 or \
                not isinstance(receipt.get("parent_source_receipt_sha256"), str):
            raise CognitiveKernelContractError("MFM clone lacks exact source ancestry")
    elif receipt.get("qualification") != "unqualified" or \
            receipt.get("upstream_weight_sha256") != SOURCE_WEIGHT_SHA256 or \
            weight == SOURCE_WEIGHT_SHA256:
        raise CognitiveKernelContractError("MFM derivative lacks changed pinned ancestry")
    return receipt


def _prepared_kind(prepared: dict) -> str:
    if prepared["schema"] == CLONE_SCHEMA:
        return "licensed-verified-mfm-role-clone"
    if prepared["schema"] == DERIVATIVE_SCHEMA:
        return "licensed-verified-mfm-role-derivative"
    raise CognitiveKernelContractError("unsupported prepared base receipt schema")


def _target_ids(tokenizer, example, max_target_tokens: int) -> tuple[list[int], list[int]]:
    # This canonical JSON includes narrow source anchors, subject, temporal
    # validity, uncertainty and scoped dispositions. No assistant response
    # from the external base supplies any target token.
    answer = canonical_json_bytes(output_record(example.target)).decode("utf-8")
    body = tokenizer.encode(answer, add_special_tokens=False)
    if not body or len(body) + 1 > max_target_tokens:
        raise CognitiveKernelContractError(
            f"case {example.case_id} exceeds full specialist target budget; do not truncate")
    start = tokenizer.bos_token_id
    end = tokenizer.eos_token_id
    if start is None or end is None:
        raise CognitiveKernelContractError("prepared tokenizer lacks BOS/EOS")
    return [start, *body], [*body, end]


def _escaped_text(value: str) -> str:
    """Lossless JSON string rendering with no accidental Gemma media marker."""
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c")


def _source_batch(processor, example, max_source_tokens: int):
    with formation_media_messages(example) as messages:
        # Preserve original bytes in custody and grounding while rendering
        # source text as a reversible JSON string. A literal "<|image|>" or
        # an original "\\u003c" cannot become a processor media placeholder.
        rendered = [{**message, "content": [
            {**item, "text": "json_text=" + _escaped_text(item["text"])}
            if item["type"] == "text" else item
            for item in message["content"]]}
            for message in messages]
        payload = processor.apply_chat_template(
            rendered, chat_template=SOURCE_TEMPLATE,
            tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", do_sample_frames=False)
        _assert_modality_tensors(example.context, payload)
    selected = {key: value for key, value in payload.items()
                if hasattr(value, "shape") and key not in {
                    "num_soft_tokens_per_image", "num_soft_tokens_per_video"}}
    tokens = selected.get("input_ids")
    if tokens is None or tokens.ndim != 2 or tokens.shape[0] != 1 or \
            not 0 < tokens.shape[1] <= max_source_tokens:
        raise CognitiveKernelContractError(
            f"case {example.case_id} exceeds full source budget or lacks tokens; do not truncate")
    if "attention_mask" not in selected or selected["attention_mask"].shape != tokens.shape:
        raise CognitiveKernelContractError("prepared processor omitted aligned attention mask")
    return selected


def _binding(args, prepared: dict, status: str, train_count: int, dev_count: int,
             transformers_version: str) -> dict:
    weight = next((row["sha256"] for row in prepared["files"]
                   if row["path"] == "model.safetensors"), None)
    return {"objective": OBJECTIVE_VERSION,
            "prepared_base_receipt_sha256": prepared["receipt_sha256"],
            "prepared_base_weight_sha256": weight,
            "prepared_base_parent_weight_sha256": SOURCE_WEIGHT_SHA256,
            "prepared_base_kind": _prepared_kind(prepared),
            "foundation_commit": FOUNDATION_COMMIT,
            "foundation_verifier_sha256": FOUNDATION_VERIFIER_SHA256,
            "foundation_inventory_sha256": FOUNDATION_INVENTORY_SHA256,
            "corpus_sha256": args.input_sha256, "corpus_status": status,
            "train_cases": train_count, "development_cases": dev_count,
            "owner_authorization_ref": args.owner_authorization_ref,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "transformers_version": transformers_version,
            "source_template_sha256": sha256(SOURCE_TEMPLATE.encode("utf-8")).hexdigest(),
            "source_builder_sha256": _digest(Path(sys.modules[formation_media_messages.__module__].__file__)),
            "trainer_sha256": _digest(Path(__file__))}


def _processor_preflight(args, train, development, status, prepared):
    import transformers
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        prepared["snapshot_path"], trust_remote_code=False, local_files_only=True)
    if not all(hasattr(processor, key) for key in
               ("apply_chat_template", "tokenizer", "image_token", "audio_token", "video_token")):
        raise CognitiveKernelContractError("prepared processor lacks Gemma 4 multimodal inputs")
    model_config = json.loads((Path(prepared["snapshot_path"]) / "config.json").read_bytes())
    source_vocabulary_size = model_config["text_config"]["vocab_size"]
    binding = _binding(args, prepared, status, len(train), len(development),
                       transformers.__version__)
    if args.mode == "processor-preflight":
        longest_source = longest_target = peak_attention_pairs = 0
        stress_source = stress_target = stress_pairs = None
        longest_train_source = longest_train_target = 0
        modalities = set()
        for case_index, example in enumerate((*train, *development)):
            encoded = _source_batch(processor, example, args.max_source_tokens)
            if int(encoded["input_ids"].max()) >= source_vocabulary_size:
                raise CognitiveKernelContractError(
                    f"case {example.case_id} contains a processor token beyond prepared base embeddings")
            inputs, _ = _target_ids(processor.tokenizer, example, args.max_target_tokens)
            length = encoded["input_ids"].shape[1]
            longest_source = max(longest_source, length)
            longest_target = max(longest_target, len(inputs))
            pairs = length * len(inputs) * args.specialist_heads * args.specialist_layers
            if case_index < len(train):
                if length > longest_train_source:
                    longest_train_source, stress_source = length, example.case_id
                if len(inputs) > longest_train_target:
                    longest_train_target, stress_target = len(inputs), example.case_id
                if pairs > peak_attention_pairs:
                    peak_attention_pairs, stress_pairs = pairs, example.case_id
            modalities.update(ref.modality for ref in example.context.evidence)
        receipt = _write_new(args.preflight_receipt, {
            "schema": PREFLIGHT_SCHEMA, **binding,
            "longest_source_tokens": longest_source,
            "longest_target_tokens": longest_target,
            "peak_train_attention_pairs": peak_attention_pairs,
            "stress_train_case_ids": list(dict.fromkeys((
                stress_source, stress_target, stress_pairs))),
            "specialist_heads": args.specialist_heads,
            "specialist_layers": args.specialist_layers,
            "processed_modalities": sorted(modalities),
            "scope": "CPU processor and data contract only; no weight fit or role qualification"})
        return processor, receipt
    receipt = _read_sealed(args.preflight_receipt, PREFLIGHT_SCHEMA)
    for name, value in binding.items():
        if receipt.get(name) != value:
            raise CognitiveKernelContractError(f"processor preflight changed: {name}")
    if receipt["longest_source_tokens"] > args.max_source_tokens or \
            receipt["longest_target_tokens"] > args.max_target_tokens:
        raise CognitiveKernelContractError("preflight exceeds selected budgets")
    if (receipt["specialist_heads"] != args.specialist_heads or
            receipt["specialist_layers"] != args.specialist_layers):
        raise CognitiveKernelContractError("preflight differs from decoder attention plan")
    return processor, receipt


def _checkpoint(output_dir: Path, *, specialist, optimizer, run_digest: str,
                epoch: int, next_case: int, step: int) -> None:
    import torch
    from safetensors.torch import save_file
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
        _write_new(staging / "checkpoint.json", {
            "schema": "mfm-v1-specialist-checkpoint-v1",
            "specialist_sha256": _digest(staging / "specialist.safetensors"),
            "optimizer_sha256": _digest(staging / "optimizer.pt"),
            "run_manifest_sha256": run_digest,
            "epoch": epoch, "next_case": next_case, "step": step})
        target = output_dir / f"checkpoint-{step:08d}"
        if target.exists():
            raise CognitiveKernelContractError("checkpoint step already exists")
        staging.rename(target)
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)


def _verify_probe_replay(checkpoint_dir: Path, specialist, config, run_digest: str) -> None:
    """Reload fresh specialist and optimizer, then perform a tiny continuation."""
    import torch
    from safetensors.torch import load_file
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist
    receipt = _read_sealed(checkpoint_dir / "checkpoint.json",
                           "mfm-v1-specialist-checkpoint-v1")
    if receipt["run_manifest_sha256"] != run_digest or \
            receipt["specialist_sha256"] != _digest(checkpoint_dir / "specialist.safetensors") or \
            receipt["optimizer_sha256"] != _digest(checkpoint_dir / "optimizer.pt"):
        raise CognitiveKernelContractError("probe checkpoint replay bytes or run binding differ")
    state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
    if state["run_manifest_sha256"] != run_digest or state["step"] != receipt["step"]:
        raise CognitiveKernelContractError("probe optimizer replay has a different step")
    device = next(specialist.parameters()).device
    restored = FormationSpecialist(config).to(device)
    restored.load_state_dict(load_file(str(checkpoint_dir / "specialist.safetensors"),
                                       device=str(device)))
    for name, value in specialist.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise CognitiveKernelContractError("probe specialist weights did not replay exactly")
    continued = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    continued.load_state_dict(state["optimizer"])
    prior = restored.source_projection.weight.detach().clone()
    toy_states = torch.ones(1, 3, config.base_hidden_size, device=device)
    toy_mask = torch.ones(1, 3, dtype=torch.long, device=device)
    toy_ids = torch.tensor([[config.start_token_id, config.end_token_id]], device=device)
    toy_labels = torch.tensor([[config.end_token_id, config.end_token_id]], device=device)
    _, loss = restored(base_states=toy_states, source_mask=toy_mask,
                       input_ids=toy_ids, labels=toy_labels)
    loss.backward()
    continued.step()
    if not torch.isfinite(loss) or torch.equal(prior, restored.source_projection.weight):
        raise CognitiveKernelContractError("probe checkpoint could not continue a backward step")


def _resume(checkpoint_dir: Path, output_dir: Path, specialist,
            optimizer, run_digest: str) -> tuple[int, int, int]:
    """Resume only a completed optimizer boundary with exact run/file hashes."""
    import torch
    from safetensors.torch import load_file
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    if checkpoint_dir.parent != output_dir.resolve(strict=True) or \
            not checkpoint_dir.name.startswith("checkpoint-"):
        raise CognitiveKernelContractError("resume checkpoint is outside this specialist run")
    receipt = _read_sealed(checkpoint_dir / "checkpoint.json",
                           "mfm-v1-specialist-checkpoint-v1")
    if receipt["run_manifest_sha256"] != run_digest or \
            receipt["specialist_sha256"] != _digest(checkpoint_dir / "specialist.safetensors") or \
            receipt["optimizer_sha256"] != _digest(checkpoint_dir / "optimizer.pt"):
        raise CognitiveKernelContractError("resume checkpoint differs from frozen run")
    state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
    for field in ("run_manifest_sha256", "epoch", "next_case", "step"):
        if state[field] != receipt[field]:
            raise CognitiveKernelContractError(f"resume state differs: {field}")
    specialist.load_state_dict(load_file(str(checkpoint_dir / "specialist.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    torch.random.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    return receipt["epoch"], receipt["next_case"], receipt["step"]


def _run_training(args, train, development, status, prepared, processor, preflight):
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForMultimodalLM
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise CognitiveKernelContractError("specialist training requires BF16 CUDA GPU")
    if args.max_cross_attention_pairs is None or \
            preflight["peak_train_attention_pairs"] > args.max_cross_attention_pairs:
        raise CognitiveKernelContractError(
            "source/target cross attention exceeds the declared probe pair cap")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForMultimodalLM.from_pretrained(
        prepared["snapshot_path"], trust_remote_code=False, local_files_only=True,
        use_safetensors=True, dtype=torch.bfloat16)
    if getattr(model.config, "model_type", None) != "gemma4_unified":
        raise CognitiveKernelContractError("prepared base is not Gemma 4 Unified")
    model.to(device).requires_grad_(False).eval()
    if not hasattr(model, "model"):
        raise CognitiveKernelContractError("prepared model has no representation path")
    config = SpecialistConfig(
        base_hidden_size=model.config.text_config.hidden_size,
        vocabulary_size=len(processor.tokenizer), width=args.specialist_width,
        layers=args.specialist_layers, heads=args.specialist_heads,
        max_target_tokens=args.max_target_tokens,
        pad_token_id=processor.tokenizer.pad_token_id,
        start_token_id=processor.tokenizer.bos_token_id,
        end_token_id=processor.tokenizer.eos_token_id,
        logit_chunk_tokens=args.logit_chunk_tokens)
    specialist = FormationSpecialist(config).to(device)
    optimizer = torch.optim.AdamW(specialist.parameters(), lr=args.learning_rate)
    run = {"schema": RUN_SCHEMA, "preflight_sha256": preflight["record_sha256"],
           "prepared_base_receipt_sha256": prepared["receipt_sha256"],
           "corpus_sha256": args.input_sha256,
           "specialist_config": config.record(), "epochs": args.epochs,
           "learning_rate": args.learning_rate,
           "gradient_accumulation": args.gradient_accumulation, "seed": args.seed,
           "max_cross_attention_pairs": args.max_cross_attention_pairs,
           "torch_version": torch.__version__, "transformers_version": preflight["transformers_version"],
           "probe_only": args.probe_only,
           "weight_lineage": _prepared_kind(prepared) + "-plus-fresh-formation-weights",
           "qualified_for_product": False}
    manifest_digest = _record_hash(run)
    if args.resume_checkpoint:
        if not args.output_dir.is_dir() or (args.output_dir / "formation-component.json").exists():
            raise CognitiveKernelContractError("resume needs an incomplete existing output directory")
        previous = _read_sealed(args.output_dir / "run.json", RUN_SCHEMA)
        if previous["record_sha256"] != manifest_digest:
            raise CognitiveKernelContractError("resume configuration differs from frozen run")
        start_epoch, start_case, global_step = _resume(
            args.resume_checkpoint, args.output_dir, specialist, optimizer, manifest_digest)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_new(args.output_dir / "run.json", run)
        start_epoch = start_case = global_step = 0
    loss_log = []
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        if args.probe_only:
            stress = preflight["stress_train_case_ids"]
            if args.gradient_accumulation < len(stress) or len(order) < args.gradient_accumulation:
                raise CognitiveKernelContractError(
                    "probe must include source, target and joint stress in a full accumulation step")
            order.sort(key=lambda index: stress.index(train[index].case_id)
                       if train[index].case_id in stress else len(stress))
        if epoch == start_epoch:
            if not 0 <= start_case <= len(order):
                raise CognitiveKernelContractError("resume case cursor differs from this corpus")
            order = order[start_case:]
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for index, item in enumerate(order):
            example = train[item]
            encoded = _source_batch(processor, example, args.max_source_tokens)
            inputs, labels = _target_ids(processor.tokenizer, example, args.max_target_tokens)
            payload = {key: value.to(device) for key, value in encoded.items()}
            # no_grad produces ordinary detached tensors. inference_mode
            # tensors cannot be saved for the specialist's backward pass.
            with torch.no_grad():
                states = model.model(**payload, use_cache=False,
                                     return_dict=True).last_hidden_state
            if states.shape[:2] != payload["attention_mask"].shape:
                raise CognitiveKernelContractError("base states do not align to source tokens")
            decoder_inputs = torch.tensor([inputs], device=device)
            target_labels = torch.tensor([labels], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = specialist(base_states=states,
                                     source_mask=payload["attention_mask"],
                                     input_ids=decoder_inputs, labels=target_labels)
            if not torch.isfinite(loss):
                raise CognitiveKernelContractError("specialist loss is nonfinite")
            (loss / args.gradient_accumulation).backward()
            loss_log.append(float(loss.detach().cpu()))
            pending += 1
            if pending == args.gradient_accumulation or index == len(order) - 1:
                if pending != args.gradient_accumulation:
                    # Preserve an average rather than downweighting the short
                    # final group of cases.
                    for parameter in specialist.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(args.gradient_accumulation / pending)
                torch.nn.utils.clip_grad_norm_(specialist.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                global_step += 1
                if global_step % args.save_every_steps == 0 or args.probe_only:
                    _checkpoint(args.output_dir, specialist=specialist,
                                optimizer=optimizer, run_digest=manifest_digest,
                                epoch=epoch,
                                next_case=(start_case + index + 1 if epoch == start_epoch else index + 1),
                                step=global_step)
                if args.probe_only:
                    break
        if args.probe_only:
            break
    if args.probe_only:
        _verify_probe_replay(args.output_dir / f"checkpoint-{global_step:08d}",
                             specialist, config, manifest_digest)
    final = args.output_dir / "formation-specialist.safetensors"
    save_file({name: value.detach().cpu().contiguous()
               for name, value in specialist.state_dict().items()}, str(final))
    _write_new(args.output_dir / "formation-component.json", {
        "schema": ARTIFACT_SCHEMA, "formation_component_sha256": _digest(final),
        "prepared_base_sha256": preflight["prepared_base_weight_sha256"],
        "prepared_base_parent_sha256": preflight["prepared_base_parent_weight_sha256"],
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "prepared_base_kind": preflight["prepared_base_kind"],
        "run_manifest_sha256": manifest_digest,
        "specialist_config": config.record(),
        "optimizer_steps": global_step,
        "mean_training_loss_for_current_segment": (
            sum(loss_log) / len(loss_log) if loss_log else None),
        "development_cases": len(development),
        "development_evaluated": False,
        "probe_only": args.probe_only, "qualified_for_product": False,
        "meaning": "Distinct learned formation component; independent behavior and FINAL unmeasured"})
    return {"formation_component_sha256": _digest(final),
            "optimizer_steps": global_step, "probe_only": args.probe_only,
            "qualified_for_product": False}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("data-preflight", "processor-preflight", "train"))
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--curriculum", type=Path)
    corpus.add_argument("--curriculum-manifest", type=Path)
    corpus.add_argument("--admitted-manifest", type=Path)
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
    parser.add_argument("--max-cross-attention-pairs", type=int,
                        help="explicit probe admission cap for source*target*heads*layers; no truncation or fit claim")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if any(type(value) is not int or value < 1 for value in (
            args.max_source_tokens, args.max_target_tokens, args.specialist_width,
            args.specialist_layers, args.specialist_heads, args.logit_chunk_tokens, args.epochs,
            args.gradient_accumulation, args.save_every_steps)) or args.learning_rate <= 0:
        raise CognitiveKernelContractError("invalid specialist training sizes or learning rate")
    if args.max_cross_attention_pairs is not None and args.max_cross_attention_pairs < 1:
        raise CognitiveKernelContractError("invalid cross-attention capacity bound")
    if args.probe_only and args.mode != "train":
        raise CognitiveKernelContractError("probe-only is a training mode")
    if args.resume_checkpoint and (args.mode != "train" or args.probe_only):
        raise CognitiveKernelContractError("resume applies only to the full training run")
    # Data admission opens exact source and target bytes. Establish the process
    # boundary before this point, including the CPU-only admission mode.
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "DO_NOT_TRACK": "1", "WANDB_DISABLED": "true"})
    # Owner-authorized synthetic curricula can carry real private source
    # bytes too. Apply the same boundary to every corpus route.
    _require_private_network_isolation()
    train, development, status = _examples(args)
    if args.mode == "data-preflight":
        print(json.dumps({"objective": OBJECTIVE_VERSION, "corpus_status": status,
                          "train_cases": len(train), "development_cases": len(development),
                          "input_sha256": args.input_sha256, "qualified_for_product": False},
                         sort_keys=True))
        return
    if any(value is None for value in (args.prepared_base_dir,
                                        args.prepared_base_receipt, args.preflight_receipt)):
        raise CognitiveKernelContractError("processor and training require prepared base and receipt")
    # No network path is used by our loader, but the host must also enforce
    # outbound denial before it ever opens private examples.
    prepared = _prepared_base(args)
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
        raise CognitiveKernelContractError("train needs a new directory or matching resume checkpoint")
    print(json.dumps(_run_training(args, train, development, status, prepared,
                                   processor, preflight), sort_keys=True))


if __name__ == "__main__":
    main()
