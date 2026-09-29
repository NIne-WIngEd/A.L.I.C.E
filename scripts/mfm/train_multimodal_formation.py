"""Train MFM weights with exact text, image, audio and video source inputs.

This uses the official Gemma 4 12B Unified processor, a pinned open-weight
checkpoint, source-only loss masking, and the existing MFM curriculum/admission
contracts. It does not open FINAL. Preflight processes the actual media without
loading model weights. GPU training requires Torch, Transformers and ffmpeg.

Example: PYTHONPATH=src python -m scripts.mfm.train_multimodal_formation \
    --curriculum /path/train.jsonl --input-sha256 SHA256 \
    --owner-authorization-ref owner_authorized_service_teacher \
    --output-dir /path/run --max-sequence-tokens 32768 --preflight-only
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from cognitive_kernel.formation_learning import admitted_rows, curriculum_rows, mixture_rows
from cognitive_kernel.formation_multimodal import (
    GEMMA_4_12B_MODEL, GEMMA_4_12B_REVISION, MULTIMODAL_OBJECTIVE,
    MultimodalFormationCandidate, formation_media_messages, supervised_multimodal_batch,
)
from scripts.mfm.train_formation_model import (
    evaluate_development, verify_artifact_receipt, write_artifact_receipt,
)


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
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--deepspeed-config", type=Path)
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


def main() -> None:
    args = arguments()
    require_sha256(args.input_sha256, "input_sha256")
    if (len(args.model_revision) != 40 or
            any(ch not in "0123456789abcdef" for ch in args.model_revision)):
        raise CognitiveKernelContractError("model revision must be an exact 40-hex git commit")
    if (args.max_sequence_tokens < 1 or args.epochs <= 0 or
            args.gradient_accumulation < 1 or args.learning_rate <= 0 or
            args.lora_rank < 0 or args.max_new_tokens < 1):
        raise CognitiveKernelContractError("training hyperparameters must be positive")
    deepspeed_digest = (sha256(args.deepspeed_config.read_bytes()).hexdigest()
                        if args.deepspeed_config else None)
    train, dev, status = _dataset(args)
    if not train:
        raise CognitiveKernelContractError("no training examples")

    # Validate source bytes before loading a 24 GB checkpoint. Media decoding
    # happens again for processor preflight; no private cached payload is kept.
    for example in (*train, *dev):
        with formation_media_messages(example):
            pass
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(
        GEMMA_4_12B_MODEL, revision=args.model_revision, trust_remote_code=False)
    if getattr(processor, "audio_seq_length", 0) < 750 or not hasattr(processor, "video_processor"):
        raise CognitiveKernelContractError("checkpoint lacks Gemma 4 Unified audio/video processor")
    maximum = 0
    longest_prompt = 0
    longest_answer = 0
    for example in (*train, *dev):
        batch = supervised_multimodal_batch(processor, example, args.max_sequence_tokens)
        maximum = max(maximum, batch["input_ids"].shape[-1])
        prompt_length = int((batch["labels"] == -100).sum().item())
        longest_prompt = max(longest_prompt, prompt_length)
        longest_answer = max(longest_answer,
                             int(batch["labels"].numel()) - prompt_length)
    if longest_answer > args.max_new_tokens:
        raise CognitiveKernelContractError(
            f"longest formation answer needs {longest_answer} tokens; generation budget is "
            f"{args.max_new_tokens}")
    if longest_prompt + args.max_new_tokens > args.max_sequence_tokens:
        raise CognitiveKernelContractError(
            "longest multimodal context plus generation budget exceeds model token budget")
    summary = {"objective": MULTIMODAL_OBJECTIVE, "corpus_status": status,
               "model": GEMMA_4_12B_MODEL, "model_revision": args.model_revision,
               "input_sha256": args.input_sha256, "train_cases": len(train),
               "development_cases": len(dev), "longest_processed_tokens": maximum,
               "longest_prompt_tokens": longest_prompt,
               "longest_answer_tokens": longest_answer,
               "max_new_tokens": args.max_new_tokens,
               "modality_counts": {modality: sum(
                   ref.modality == modality for case in (*train, *dev)
                   for ref in case.context.evidence)
                   for modality in ("text", "code", "structured", "image", "audio", "video")}}
    if args.preflight_only:
        print(json.dumps(summary, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise CognitiveKernelContractError("multimodal weight training needs an available GPU")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise CognitiveKernelContractError("bf16 training needs a bf16-capable GPU")

    model = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_4_12B_MODEL, revision=args.model_revision, trust_remote_code=False,
        dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
    limit = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None)
    if limit and maximum > limit:
        raise CognitiveKernelContractError("case exceeds the base model position budget")
    if args.lora_rank:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
            lora_alpha=2 * args.lora_rank, target_modules=["q_proj", "v_proj"]))
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model, train_dataset=Dataset(train),
        eval_dataset=Dataset(dev) if dev else None, data_collator=collate,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            num_train_epochs=args.epochs, per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            bf16=args.precision == "bf16", gradient_checkpointing=True,
            eval_strategy="epoch" if dev else "no", save_strategy="epoch",
            save_safetensors=True, remove_unused_columns=False,
            deepspeed=str(args.deepspeed_config) if args.deepspeed_config else None,
            seed=args.seed, report_to="none"))
    trainer.train()
    if dev:
        trainer.evaluate()
    final = args.output_dir / "model"
    final.mkdir(parents=True, exist_ok=True)
    trained = trainer.model.merge_and_unload() if args.lora_rank else trainer.model
    trained.save_pretrained(final, safe_serialization=True)
    processor.save_pretrained(final)
    summary.update({"precision": args.precision, "lora_rank": args.lora_rank,
                    "seed": args.seed, "epochs": args.epochs,
                    "gradient_accumulation": args.gradient_accumulation,
                    "learning_rate": args.learning_rate,
                    "optimizer": str(trainer.args.optim),
                    "deepspeed_config_sha256": deepspeed_digest,
                    "transformers_version": __import__("transformers").__version__,
                    "torch_version": torch.__version__})
    digest = write_artifact_receipt(final, summary)
    if dev:
        trained.eval()
        candidate = MultimodalFormationCandidate(
            trained, processor, verify_artifact_receipt(final),
            "multimodal-development", args.max_sequence_tokens - args.max_new_tokens,
            args.max_new_tokens)
        report = evaluate_development(candidate, dev)
        (args.output_dir / "development-diagnostics.json").write_text(
            json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
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
