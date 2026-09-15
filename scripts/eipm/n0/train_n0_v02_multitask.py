#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
from alice_personality.n0.training_schedule import (
    accelerated_scheduler_steps,
    validate_scheduler_horizon,
)
from alice_personality.n0.v02_model import AliceN0V02Model
from alice_personality.n0.v02_training import (
    TeacherMultitaskCollator,
    export_ranker_state,
    sha256_file,
    teacher_objective_losses,
    verify_public_corpus_v021,
    verify_teacher_registry,
    verify_tokenizer_v021,
)


EXACT_PARAMETERS = 136_594_435
OBJECTIVE_WEIGHTS = {
    "span_mlm": 0.70,
    "candidate_preference": 0.10,
    "principle_rationale_alignment": 0.10,
    "semantic_contrastive": 0.10,
}


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def cycle_next(iterator: Any, dataloader: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def save_checkpoint(
    *,
    accelerator: Any,
    model: Any,
    output_dir: Path,
    step: int,
    config_path: Path,
    tokenizer_dir: Path,
    corpus_dir: Path,
    source_config_path: Path,
    teacher_registry: Path,
    teacher_audit: Path,
    corpus_receipt: dict[str, Any],
    tokenizer_receipt: dict[str, Any],
    local_mlm_tokens: int,
    local_teacher_rows: int,
    objective_loss_sums: dict[str, float],
    objective_loss_counts: dict[str, int],
    sequence_length: int,
    mlm_grad_accum: int,
    teacher_batch_size: int,
    scheduler_total_steps: int,
    warmup_steps: int,
    mixed_precision: str,
    seed: int,
) -> None:
    from safetensors.torch import save_file

    accelerator.wait_for_everyone()
    checkpoint = output_dir / f"step-{step:08d}"
    accelerator.save_state(str(checkpoint / "accelerator_state"))

    token_tensor = torch.tensor(
        local_mlm_tokens,
        device=accelerator.device,
        dtype=torch.long,
    )
    teacher_tensor = torch.tensor(
        local_teacher_rows,
        device=accelerator.device,
        dtype=torch.long,
    )
    global_tokens = int(accelerator.reduce(token_tensor, reduction="sum").item())
    global_teacher_rows = int(
        accelerator.reduce(teacher_tensor, reduction="sum").item()
    )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        report = unwrapped.parameter_report()
        if int(report["total_parameters"]) != EXACT_PARAMETERS:
            raise RuntimeError(
                "v0.2 parameter count drifted: "
                f"expected={EXACT_PARAMETERS} observed={report['total_parameters']}"
            )

        mlm_dir = checkpoint / "mlm"
        mlm_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.mlm.save_pretrained(mlm_dir, safe_serialization=True)

        ranker_path = checkpoint / "ranker.safetensors"
        save_file(export_ranker_state(unwrapped), str(ranker_path))

        full_path = checkpoint / "alice_n0_v02.safetensors"
        full_state = {
            name: value.detach().cpu().contiguous()
            for name, value in unwrapped.state_dict().items()
        }
        save_file(full_state, str(full_path))
        del full_state

        average_losses = {
            name: objective_loss_sums[name] / max(objective_loss_counts[name], 1)
            for name in objective_loss_sums
        }
        receipt = {
            "schema": "alice.eipm.n0.v02-multitask-checkpoint-receipt.v0.1",
            "model_id": "alice-n0-semantic-v0.2",
            "step": step,
            "git_revision": git_revision(),
            "random_initialization_only": True,
            "v01_weights_used": False,
            "exact_parameter_count": report["total_parameters"],
            "trainable_parameter_count": report["trainable_parameters"],
            "auxiliary_head_parameters": report["auxiliary_head_parameters"],
            "config_sha256": sha256_file(config_path),
            "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
            "tokenizer_receipt_sha256": sha256_file(
                tokenizer_dir / "tokenizer_receipt.json"
            ),
            "tokenizer_parent_corpus_receipt_sha256": tokenizer_receipt.get(
                "corpus_receipt_sha256"
            ),
            "corpus_receipt_sha256": sha256_file(corpus_dir / "corpus_receipt.json"),
            "source_config_sha256": sha256_file(source_config_path),
            "corpus_source_count": len(corpus_receipt.get("sources", [])),
            "teacher_registry_sha256": sha256_file(teacher_registry),
            "teacher_audit_sha256": sha256_file(teacher_audit),
            "teacher_registered_rows": 1020,
            "teacher_competencies": 51,
            "teacher_coverage_gate_open": True,
            "sequence_length": sequence_length,
            "mlm_probability": 0.30,
            "mlm_grad_accum_per_optimizer_step": mlm_grad_accum,
            "teacher_batch_size_per_process": teacher_batch_size,
            "objective_weights": OBJECTIVE_WEIGHTS,
            "mean_objective_losses_since_launch": average_losses,
            "mlm_tokens_seen_total_across_processes": global_tokens,
            "teacher_rows_seen_total_across_processes": global_teacher_rows,
            "scheduler_total_steps_global": scheduler_total_steps,
            "warmup_steps_global": warmup_steps,
            "world_size": accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "seed": seed,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "private_identity_data": False,
            "private_identity_gradient": False,
            "private_identity_gradient_authorized": False,
            "model_training_performed": True,
            "training_stage": "N0_public_native_multitask",
            "mlm_artifact_sha256": directory_hashes(mlm_dir),
            "ranker_sha256": sha256_file(ranker_path),
            "full_model_sha256": sha256_file(full_path),
        }
        (checkpoint / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Governed native multi-objective training for alice-n0-semantic-v0.2."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--teacher-registry", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--mlm-micro-batch-size", type=int, default=1)
    parser.add_argument("--mlm-grad-accum", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--teacher-max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--scheduler-total-steps", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="fp16",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before N0 v0.2 training") from exc

    if args.sequence_length != 512:
        raise SystemExit(
            "first N0 v0.2 tranche is deliberately fixed at sequence_length=512"
        )
    if args.max_steps < 1 or args.save_every < 1 or args.mlm_grad_accum < 1:
        raise SystemExit("steps/save interval/MLM accumulation must be positive")
    validate_scheduler_horizon(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        scheduler_total_steps=args.scheduler_total_steps,
    )

    config_path = Path(args.config).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    source_config_path = Path(args.source_config).resolve()
    teacher_registry = Path(args.teacher_registry).resolve()
    teacher_audit = Path(args.teacher_audit).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[3]

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_n0_config(config_path)
    if args.sequence_length not in config.train_sequence_lengths:
        raise SystemExit(f"sequence length {args.sequence_length} not allowed by config")

    tokenizer_receipt = verify_tokenizer_v021(tokenizer_dir)
    corpus_receipt, corpus_paths = verify_public_corpus_v021(
        corpus_dir,
        source_config_path,
    )
    curriculum_paths, teacher_report = verify_teacher_registry(
        repo_root,
        teacher_registry,
        teacher_audit,
    )

    seed_everything(args.seed)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes != 2:
        raise SystemExit(
            "first N0 v0.2 tranche requires exactly two accelerator processes; "
            f"observed {accelerator.num_processes}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("first N0 v0.2 tranche requires CUDA")

    tokenizer = load_tokenizer(tokenizer_dir)
    if len(tokenizer) != config.vocab_size:
        raise SystemExit(
            f"tokenizer size {len(tokenizer)} != config vocab {config.vocab_size}"
        )

    model = AliceN0V02Model(config)
    report = model.parameter_report()
    if int(report["total_parameters"]) != EXACT_PARAMETERS:
        raise SystemExit(
            "exact v0.2 parameter gate failed: "
            f"expected={EXACT_PARAMETERS} observed={report['total_parameters']}"
        )
    if not args.no_gradient_checkpointing and hasattr(
        model.mlm,
        "gradient_checkpointing_enable",
    ):
        try:
            model.mlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.mlm.gradient_checkpointing_enable()

    mlm_dataset = PackedJSONLIterableDataset(
        paths=corpus_paths,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        split="train",
    )
    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=args.mlm_micro_batch_size,
        collate_fn=SpanMLMCollator(
            tokenizer=tokenizer,
            mlm_probability=config.mlm_probability,
            mean_span=config.mean_mask_span,
            max_span=config.max_mask_span,
            seed=args.seed + accelerator.process_index,
        ),
        num_workers=0,
        pin_memory=True,
    )

    teacher_dataset = CurriculumDataset(curriculum_paths, "train")
    if len(teacher_dataset) != 51 * 15:
        raise SystemExit(
            "expected exactly 765 public teacher train rows, "
            f"observed {len(teacher_dataset)}"
        )
    teacher_loader = DataLoader(
        teacher_dataset,
        batch_size=args.teacher_batch_size,
        shuffle=True,
        collate_fn=TeacherMultitaskCollator(tokenizer, args.teacher_max_length),
        num_workers=0,
        pin_memory=True,
    )

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bias" in name.lower() or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler_warmup_internal = accelerated_scheduler_steps(
        args.warmup_steps,
        num_processes=accelerator.num_processes,
        split_batches=accelerator.split_batches,
    )
    scheduler_total_internal = accelerated_scheduler_steps(
        args.scheduler_total_steps,
        num_processes=accelerator.num_processes,
        split_batches=accelerator.split_batches,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_warmup_internal,
        num_training_steps=scheduler_total_internal,
    )

    model, optimizer, mlm_loader, teacher_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        mlm_loader,
        teacher_loader,
        scheduler,
    )

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "status": "TRAIN_START",
                    "model_id": "alice-n0-semantic-v0.2",
                    "parameters": report,
                    "teacher_rows": int(teacher_report["registered_rows"]),
                    "teacher_train_rows": len(teacher_dataset),
                    "teacher_competencies": int(teacher_report["competency_count"]),
                    "corpus_sources": len(corpus_receipt["sources"]),
                    "sequence_length": args.sequence_length,
                    "mlm_grad_accum": args.mlm_grad_accum,
                    "objective_weights": OBJECTIVE_WEIGHTS,
                    "max_steps": args.max_steps,
                    "private_identity_gradient": False,
                },
                sort_keys=True,
            )
        )

    model.train()
    mlm_iterator = iter(mlm_loader)
    teacher_iterator = iter(teacher_loader)
    local_mlm_tokens = 0
    local_teacher_rows = 0
    loss_sums = {name: 0.0 for name in OBJECTIVE_WEIGHTS}
    loss_counts = {name: 0 for name in OBJECTIVE_WEIGHTS}

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        mlm_step_loss = 0.0

        for micro_step in range(args.mlm_grad_accum):
            mlm_batch, mlm_iterator = cycle_next(mlm_iterator, mlm_loader)
            local_mlm_tokens += int(mlm_batch["attention_mask"].sum().item())

            # Synchronize the final MLM microbatch before the teacher-only
            # forward. Otherwise accumulated MLM-head gradients would never be
            # reduced because that prediction head is intentionally unused by
            # the teacher objective.
            sync_context = (
                accelerator.no_sync(model)
                if micro_step < args.mlm_grad_accum - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                outputs = model(task="mlm", **mlm_batch)
                mlm_loss = outputs.loss
                accelerator.backward(
                    mlm_loss
                    * (OBJECTIVE_WEIGHTS["span_mlm"] / args.mlm_grad_accum)
                )
            mlm_step_loss += float(mlm_loss.detach().cpu())

        teacher_batch, teacher_iterator = cycle_next(
            teacher_iterator,
            teacher_loader,
        )
        local_teacher_rows += len(teacher_batch["ids"])
        preference_loss, alignment_loss, contrastive_loss = teacher_objective_losses(
            model,
            teacher_batch,
            temperature=0.05,
        )
        teacher_total = (
            OBJECTIVE_WEIGHTS["candidate_preference"] * preference_loss
            + OBJECTIVE_WEIGHTS["principle_rationale_alignment"] * alignment_loss
            + OBJECTIVE_WEIGHTS["semantic_contrastive"] * contrastive_loss
        )
        accelerator.backward(teacher_total)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses = {
            "span_mlm": mlm_step_loss / args.mlm_grad_accum,
            "candidate_preference": float(preference_loss.detach().cpu()),
            "principle_rationale_alignment": float(alignment_loss.detach().cpu()),
            "semantic_contrastive": float(contrastive_loss.detach().cpu()),
        }
        for name, value in losses.items():
            loss_sums[name] += value
            loss_counts[name] += 1

        if accelerator.is_main_process and (step == 1 or step % 10 == 0):
            print(
                json.dumps(
                    {
                        "step": step,
                        "lr": scheduler.get_last_lr()[0],
                        **{
                            f"loss_{name}": value
                            for name, value in losses.items()
                        },
                    },
                    sort_keys=True,
                )
            )

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                output_dir=output_dir,
                step=step,
                config_path=config_path,
                tokenizer_dir=tokenizer_dir,
                corpus_dir=corpus_dir,
                source_config_path=source_config_path,
                teacher_registry=teacher_registry,
                teacher_audit=teacher_audit,
                corpus_receipt=corpus_receipt,
                tokenizer_receipt=tokenizer_receipt,
                local_mlm_tokens=local_mlm_tokens,
                local_teacher_rows=local_teacher_rows,
                objective_loss_sums=loss_sums,
                objective_loss_counts=loss_counts,
                sequence_length=args.sequence_length,
                mlm_grad_accum=args.mlm_grad_accum,
                teacher_batch_size=args.teacher_batch_size,
                scheduler_total_steps=args.scheduler_total_steps,
                warmup_steps=args.warmup_steps,
                mixed_precision=args.mixed_precision,
                seed=args.seed,
            )

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "step": args.max_steps,
                    "private_identity_gradient": False,
                }
            )
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
