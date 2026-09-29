"""Train MFM weights with exact text, image, audio and video source inputs.

This uses the official Gemma 4 12B Unified processor, a pinned open-weight
checkpoint, source-only loss masking, and the existing MFM curriculum/admission
contracts. It does not open FINAL. Preflight processes the actual media without
loading model weights. GPU training requires Torch, Transformers and ffmpeg.

Example: PYTHONPATH=src python -m scripts.mfm.train_multimodal_formation \
    --curriculum /path/train.jsonl --input-sha256 SHA256 \
    --owner-authorization-ref owner_authorized_service_teacher \
    --output-dir /path/run --preflight-receipt /path/processor.json \
    --max-sequence-tokens 32768 --preflight-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from cognitive_kernel.formation_learning import admitted_rows, curriculum_rows, mixture_rows
from cognitive_kernel.formation_multimodal import (
    GEMMA_4_12B_MODEL, GEMMA_4_12B_REVISION, MULTIMODAL_OBJECTIVE,
    MultimodalFormationCandidate, formation_media_messages, supervised_multimodal_batch,
)
from scripts.mfm.train_formation_model import verify_artifact_receipt, write_artifact_receipt
from scripts.mfm.multimodal_paid_run import (
    PREFLIGHT_SCHEMA, RUN_SCHEMA, await_run_binding, bind_run, digest_record, probe_indices,
    read_sealed, require_complete_weight_export, require_preflight, require_zero3,
    source_fingerprint, write_sealed,
)
from scripts.mfm.stage_gemma4_model import verify_staged_model


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--curriculum", type=Path)
    group.add_argument("--curriculum-manifest", type=Path)
    group.add_argument("--admitted-manifest", type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--owner-authorization-ref")
    parser.add_argument("--model-revision", default=GEMMA_4_12B_REVISION,
                        help="exact 40-hex google/gemma-4-12B-it git commit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-tokens", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path,
                        help="CPU processor receipt; required before any paid GPU run")
    parser.add_argument("--staged-model-receipt", type=Path,
                        help="CPU-verified exact model snapshot; required before paid GPU work")
    parser.add_argument("--staged-model-dir", type=Path,
                        help="relocated local snapshot; verify its bytes against staging receipt")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--deepspeed-config", type=Path)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--expected-gpu-name", default="A100")
    parser.add_argument("--min-vram-gib", type=float, default=75)
    parser.add_argument("--min-free-disk-gib", type=float, default=750,
                        help="space for multiple full ZeRO optimizer checkpoints and export")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--probe-only", action="store_true",
                        help="one full optimizer step over bound processor-selected stress cases")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="0 updates full weights; positive trains and merges an adapter")
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    return parser.parse_args()


def _dataset(args: argparse.Namespace):
    if args.admitted_manifest:
        if args.owner_authorization_ref:
            raise CognitiveKernelContractError("admitted corpus does not take synthetic authorization")
        corpus = admit_formation_corpus(args.admitted_manifest,
                                        expected_sha256=args.input_sha256)
        return (tuple(admitted_rows(corpus, split="train")),
                tuple(admitted_rows(corpus, split="development")),
                "admitted-train-development-final-sealed")
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
    return train, (), "owner-authorized-training-only-unqualified"


def require_rank_receipt(path: Path, rank: int, world_size: int,
                         manifest_sha256: str, optimizer_steps: int) -> dict:
    receipt = read_sealed(path)
    if (receipt.get("rank") != rank or receipt.get("world_size") != world_size or
            receipt.get("run_manifest_sha256") != manifest_sha256 or
            receipt.get("optimizer_steps") != optimizer_steps):
        raise CognitiveKernelContractError("GPU rank receipt is missing or from another run")
    return receipt


def main() -> None:
    args = arguments()
    require_sha256(args.input_sha256, "input_sha256")
    if (len(args.model_revision) != 40 or
            any(ch not in "0123456789abcdef" for ch in args.model_revision)):
        raise CognitiveKernelContractError("model revision must be an exact 40-hex git commit")
    if (args.max_sequence_tokens < 1 or args.epochs <= 0 or
            args.gradient_accumulation < 1 or args.learning_rate <= 0 or
            args.lora_rank < 0 or args.max_new_tokens < 1 or args.save_steps < 1 or
            args.expected_world_size < 1 or args.min_vram_gib <= 0 or
            args.min_free_disk_gib <= 0):
        raise CognitiveKernelContractError("training hyperparameters must be positive")
    if args.preflight_receipt is None:
        raise CognitiveKernelContractError("exact processor preflight receipt is required")
    if not args.preflight_only and args.staged_model_receipt is None:
        raise CognitiveKernelContractError("paid run requires a verified staged model receipt")
    if args.staged_model_dir is not None and args.staged_model_receipt is None:
        raise CognitiveKernelContractError("relocated snapshot requires a staging receipt")
    if args.preflight_only and (args.probe_only or args.resume_from_checkpoint):
        raise CognitiveKernelContractError("processor preflight cannot train or resume")
    if not args.preflight_only and (args.precision != "bf16" or args.lora_rank != 0 or
                                    args.deepspeed_config is None):
        raise CognitiveKernelContractError(
            "paid full-weight route requires BF16 ZeRO-3; LoRA needs separate qualification")
    deepspeed_digest = (require_zero3(args.deepspeed_config)
                        if args.deepspeed_config and not args.preflight_only else None)
    train, dev, status = _dataset(args)
    if not train:
        raise CognitiveKernelContractError("no training examples")

    # The CPU processor pass is sealed against the exact input, revision,
    # dependency version and prompt code. Never repeat it on paid GPUs.
    if args.preflight_only:
        for example in (*train, *dev):
            with formation_media_messages(example):
                pass
        # A CPU compute node can process the pinned snapshot without outbound
        # Hub access after the snapshot has been staged and verified elsewhere.
        if args.staged_model_receipt:
            snapshot_path, staged_model = verify_staged_model(
                args.staged_model_receipt, GEMMA_4_12B_MODEL, args.model_revision,
                snapshot_override=args.staged_model_dir)
        else:
            snapshot_path = staged_model = None
    import torch
    import transformers
    binding = {"objective": MULTIMODAL_OBJECTIVE, "corpus_status": status,
               "model": GEMMA_4_12B_MODEL, "model_revision": args.model_revision,
               "input_sha256": args.input_sha256, "train_cases": len(train),
               "development_cases": len(dev), "max_sequence_tokens": args.max_sequence_tokens,
               "max_new_tokens": args.max_new_tokens,
               "owner_authorization_ref": args.owner_authorization_ref,
               "transformers_version": transformers.__version__}
    if not args.preflight_only:
        summary = require_preflight(args.preflight_receipt, binding)
        if args.probe_only and (len(summary["probe_indices"]) !=
                                args.expected_world_size * args.gradient_accumulation):
            raise CognitiveKernelContractError(
                "probe selection must supply one full accumulation step across all ranks")
        if args.probe_only and args.resume_from_checkpoint:
            raise CognitiveKernelContractError("hardware probe must start in a fresh output directory")
    if not args.preflight_only:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if (world_size != args.expected_world_size or not 0 <= rank < world_size or
                not 0 <= local_rank < world_size):
            raise CognitiveKernelContractError("torchrun world size or rank differs from paid allocation")
        if not torch.cuda.is_available():
            raise CognitiveKernelContractError("multimodal weight training needs an available GPU")
        if torch.cuda.device_count() != args.expected_world_size:
            raise CognitiveKernelContractError("visible GPU count differs from paid allocation")
        for device in range(args.expected_world_size):
            properties = torch.cuda.get_device_properties(device)
            if (args.expected_gpu_name.lower() not in properties.name.lower() or
                    properties.total_memory < args.min_vram_gib * 1024 ** 3):
                raise CognitiveKernelContractError("actual GPU type or VRAM differs from paid allocation")
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise CognitiveKernelContractError("bf16 training needs a bf16-capable GPU")
        disk_root = args.output_dir.parent.resolve()
        if not disk_root.is_dir() or shutil.disk_usage(disk_root).free < args.min_free_disk_gib * 1024 ** 3:
            raise CognitiveKernelContractError("output volume lacks free space for ZeRO checkpoints")
        snapshot_path, staged_model = verify_staged_model(
            args.staged_model_receipt, GEMMA_4_12B_MODEL, args.model_revision,
            rehash=rank == 0, snapshot_override=args.staged_model_dir)
    from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(
        snapshot_path if snapshot_path else GEMMA_4_12B_MODEL,
        **({"revision": args.model_revision} if snapshot_path is None else {}),
        trust_remote_code=False, local_files_only=snapshot_path is not None)
    if getattr(processor, "audio_seq_length", 0) < 750 or not hasattr(processor, "video_processor"):
        raise CognitiveKernelContractError("checkpoint lacks Gemma 4 Unified audio/video processor")
    if args.preflight_only:
        maximum = longest_prompt = longest_answer = longest_index = 0
        longest_train = -1
        lengths = []
        for index, example in enumerate((*train, *dev)):
            batch = supervised_multimodal_batch(processor, example, args.max_sequence_tokens)
            length = int(batch["input_ids"].shape[-1])
            maximum = max(maximum, length)
            if index < len(train):
                lengths.append((index, length))
                if length > longest_train:
                    longest_train, longest_index = length, index
            prompt_length = int((batch["labels"] == -100).sum().item())
            longest_prompt = max(longest_prompt, prompt_length)
            longest_answer = max(longest_answer, int(batch["labels"].numel()) - prompt_length)
        if longest_answer > args.max_new_tokens:
            raise CognitiveKernelContractError(
                f"longest formation answer needs {longest_answer} tokens; generation budget is "
                f"{args.max_new_tokens}")
        if longest_prompt + args.max_new_tokens > args.max_sequence_tokens:
            raise CognitiveKernelContractError(
                "longest multimodal context plus generation budget exceeds model token budget")
        receipt = write_sealed(args.preflight_receipt, {
            **binding, "schema": PREFLIGHT_SCHEMA, "source_fingerprint": source_fingerprint(),
            "longest_processed_tokens": maximum, "longest_prompt_tokens": longest_prompt,
            "longest_answer_tokens": longest_answer, "longest_case_index": longest_index,
            "probe_indices": probe_indices(lengths),
            "processor_snapshot_sha256": staged_model["receipt_sha256"]
            if staged_model else None,
            "modality_counts": {modality: sum(
                ref.modality == modality for case in (*train, *dev)
                for ref in case.context.evidence)
                for modality in ("text", "code", "structured", "image", "audio", "video")},
        })
        print(json.dumps(receipt, sort_keys=True))
        return
    run_record = {"schema": RUN_SCHEMA, "model": GEMMA_4_12B_MODEL,
                  "model_revision": args.model_revision, "input_sha256": args.input_sha256,
                  "preflight_sha256": summary["record_sha256"],
                  "staged_model_sha256": staged_model["receipt_sha256"],
                  "deepspeed_config_sha256": deepspeed_digest,
                  "torch_version": torch.__version__,
                  "transformers_version": transformers.__version__,
                  "source_fingerprint": source_fingerprint(),
                  "owner_authorization_ref": args.owner_authorization_ref,
                  "max_sequence_tokens": args.max_sequence_tokens,
                  "max_new_tokens": args.max_new_tokens,
                  "epochs": args.epochs, "gradient_accumulation": args.gradient_accumulation,
                  "learning_rate": args.learning_rate, "seed": args.seed,
                  "save_steps": args.save_steps, "world_size": world_size,
                  "expected_gpu_name": args.expected_gpu_name,
                  "min_vram_gib": args.min_vram_gib,
                  "min_free_disk_gib": args.min_free_disk_gib,
                  "probe_only": args.probe_only, "precision": args.precision}
    run_manifest_sha256 = digest_record(run_record)
    if rank == 0 or args.resume_from_checkpoint:
        bind_run(args.output_dir, run_record, args.resume_from_checkpoint)
    else:
        await_run_binding(args.output_dir, run_record)

    # Hugging Face must see ZeRO-3 before from_pretrained. The loading path is
    # partitioned only when TrainingArguments initializes the integration first.
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs, max_steps=1 if args.probe_only else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, bf16=True, gradient_checkpointing=True,
        eval_strategy="epoch" if dev and not args.probe_only else "no",
        save_strategy="steps", save_steps=1 if args.probe_only else args.save_steps,
        save_total_limit=2, save_safetensors=True, remove_unused_columns=False,
        deepspeed=str(args.deepspeed_config), seed=args.seed, report_to="none")
    model = AutoModelForMultimodalLM.from_pretrained(
        snapshot_path, trust_remote_code=False,
        local_files_only=True, dtype=torch.bfloat16)
    limit = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None)
    if limit and summary["longest_processed_tokens"] > limit:
        raise CognitiveKernelContractError("case exceeds the base model position budget")
    model.config.use_cache = False

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return supervised_multimodal_batch(
                processor, self.items[index], args.max_sequence_tokens)

    def collate(rows):
        if len(rows) != 1:
            raise CognitiveKernelContractError("ragged multimodal inputs need batch size one")
        return rows[0]

    selected_train = ([train[index] for index in summary["probe_indices"]]
                      if args.probe_only else train)
    trainer = Trainer(
        model=model, train_dataset=Dataset(selected_train),
        eval_dataset=Dataset(dev) if dev and not args.probe_only else None,
        data_collator=collate, args=training_args)
    setup_peak = torch.cuda.max_memory_reserved(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    started = time.monotonic()
    outcome = trainer.train(resume_from_checkpoint=(
        str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None))
    torch.cuda.synchronize(local_rank)
    train_seconds = time.monotonic() - started
    training_peak = torch.cuda.max_memory_reserved(local_rank)
    final = args.output_dir / ("probe-export" if args.probe_only else "model")
    torch.cuda.reset_peak_memory_stats(local_rank)
    saving = time.monotonic()
    trainer.save_model(str(final))  # collective ZeRO-3 gather; every rank participates
    torch.cuda.synchronize(local_rank)
    save_seconds = time.monotonic() - saving
    rank_receipt = {"rank": rank, "world_size": world_size,
                    "run_manifest_sha256": run_manifest_sha256,
                    "gpu": torch.cuda.get_device_name(local_rank),
                    "total_vram_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
                    "setup_peak_reserved_bytes": setup_peak,
                    "train_peak_reserved_bytes": training_peak,
                    "save_peak_reserved_bytes": torch.cuda.max_memory_reserved(local_rank),
                    "train_seconds": train_seconds, "save_seconds": save_seconds,
                    "optimizer_steps": outcome.global_step,
                    "run_mode": "hardware-qualification" if args.probe_only else "training"}
    write_sealed(args.output_dir / f"gpu-rank-{rank}.json", rank_receipt)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank != 0:
        return
    processor.save_pretrained(final)
    weight_export = require_complete_weight_export(final)
    ranks = [require_rank_receipt(args.output_dir / f"gpu-rank-{n}.json", n, world_size,
                                  run_manifest_sha256, outcome.global_step)
             for n in range(world_size)]
    summary = {**summary, "precision": args.precision, "lora_rank": args.lora_rank,
                    "seed": args.seed, "epochs": args.epochs,
                    "gradient_accumulation": args.gradient_accumulation,
                    "learning_rate": args.learning_rate,
                    "optimizer": str(trainer.args.optim),
                    "deepspeed_config_sha256": deepspeed_digest,
                    "staged_model_sha256": staged_model["receipt_sha256"],
                    "transformers_version": __import__("transformers").__version__,
                    "torch_version": torch.__version__, "gpu_rank_receipts": ranks,
                    "weight_export": weight_export,
                    "run_mode": "hardware-qualification" if args.probe_only else "training",
                    "qualified_for_product": False,
                    "qualification_status": "training-only-awaits-independent-review"}
    digest = write_artifact_receipt(final, summary)
    print(json.dumps({**summary, "artifact_sha256": digest}, sort_keys=True))


def load_multimodal_candidate(artifact_dir: Path, *, inference_run_id: str,
                              max_input_tokens: int, max_new_tokens: int):
    """Reload only the content-addressed local artifact for governed inference."""
    artifact_dir = Path(artifact_dir)
    digest = verify_artifact_receipt(artifact_dir)
    if max_input_tokens < 1 or max_new_tokens < 1:
        raise CognitiveKernelContractError("generation budgets must be positive")
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        artifact_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(
        artifact_dir, local_files_only=True, trust_remote_code=False, device_map="auto")
    model.eval()
    return MultimodalFormationCandidate(model, processor, digest, inference_run_id,
                                        max_input_tokens, max_new_tokens)


if __name__ == "__main__":
    main()
