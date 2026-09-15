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
from alice_personality.n0.curriculum import validate_curriculum_manifest, validate_curriculum_rows
from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
from alice_personality.n0.training_schedule import accelerated_scheduler_steps
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
PARENT_STEP = 250
PARENT_CHECKPOINT_KEY = "step-00000250"
REPAIR_ROWS = 30
REPAIR_TRAIN_ROWS = 20
REPAIR_DEV_ROWS = 10
REPAIR_FAMILIES = {
    "ALIGN-01+TEMP-01",
    "SEM-04+RANK-03",
    "PRAG-03+SOC-03",
    "SEM-06+SOC-02",
    "VOICE-05+SOC-04",
}

LOSS_WEIGHTS = {
    "public_mlm_replay": 0.15,
    "targeted_repair_teacher": 0.55,
    "governed_teacher_replay": 0.30,
}
TEACHER_COMPONENT_WEIGHTS = {
    "candidate_preference": 0.50,
    "principle_rationale_alignment": 0.25,
    "semantic_contrastive": 0.25,
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


def cycle_next(iterator: Any, dataloader: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def teacher_total(losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    preference, alignment, contrastive = losses
    return (
        TEACHER_COMPONENT_WEIGHTS["candidate_preference"] * preference
        + TEACHER_COMPONENT_WEIGHTS["principle_rationale_alignment"] * alignment
        + TEACHER_COMPONENT_WEIGHTS["semantic_contrastive"] * contrastive
    )


def configure_selective_repair(model: AliceN0V02Model, top_layers: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    layers = getattr(model.backbone, "layers", None)
    if layers is None:
        raise RuntimeError("ModernBERT backbone no longer exposes .layers")
    if top_layers < 1 or top_layers > len(layers):
        raise ValueError(f"top_layers must be in [1, {len(layers)}]")

    for layer in layers[-top_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    for module in (
        model.preference_scorer,
        model.semantic_projection,
        model.rationale_projection,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.principle_scale.requires_grad = True
    model.principle_bias.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "top_backbone_layers_trainable": top_layers,
        "bottom_backbone_layers_frozen": len(layers) - top_layers,
    }


def verify_parent_checkpoint(parent_dir: Path, tokenizer_dir: Path) -> dict[str, Any]:
    receipt_path = parent_dir / "receipt.json"
    full_path = parent_dir / "alice_n0_v02.safetensors"
    ranker_path = parent_dir / "ranker.safetensors"
    mlm_dir = parent_dir / "mlm"
    if not receipt_path.is_file() or not full_path.is_file() or not ranker_path.is_file() or not mlm_dir.is_dir():
        raise ValueError("step-250 parent checkpoint is incomplete")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("model_id") != "alice-n0-semantic-v0.2" or int(receipt.get("step", -1)) != PARENT_STEP:
        raise ValueError("repair must start from the exact N0 v0.2 step-250 checkpoint")
    if receipt.get("private_identity_data") is not False or receipt.get("private_identity_gradient") is not False:
        raise ValueError("repair parent must be public N0 with no private identity gradient")
    if receipt.get("random_initialization_only") is not True or receipt.get("v01_weights_used") is not False:
        raise ValueError("unexpected parent lineage")
    if str(receipt.get("full_model_sha256")) != sha256_file(full_path):
        raise ValueError("parent full-model hash mismatch")
    if str(receipt.get("ranker_sha256")) != sha256_file(ranker_path):
        raise ValueError("parent ranker hash mismatch")
    if str(receipt.get("tokenizer_sha256")) != sha256_file(tokenizer_dir / "tokenizer.json"):
        raise ValueError("parent tokenizer hash mismatch")
    return receipt


def verify_repair_curriculum(curriculum_path: Path, manifest_path: Path) -> dict[str, Any]:
    summary = validate_curriculum_rows(curriculum_path)
    manifest = validate_curriculum_manifest(curriculum_path, manifest_path)
    if int(summary["row_count"]) != REPAIR_ROWS:
        raise ValueError(f"repair curriculum must have {REPAIR_ROWS} rows")
    if int(summary["split_counts"]["train"]) != REPAIR_TRAIN_ROWS:
        raise ValueError(f"repair curriculum must have {REPAIR_TRAIN_ROWS} train rows")
    if int(summary["split_counts"]["dev"]) != REPAIR_DEV_ROWS:
        raise ValueError(f"repair curriculum must have {REPAIR_DEV_ROWS} dev rows")
    if set(summary["competency_counts"]) != REPAIR_FAMILIES:
        raise ValueError("repair curriculum competency families drifted")
    if manifest.get("private_identity_gradient_authorized") is not False:
        raise ValueError("repair manifest must forbid private identity gradients")
    if manifest.get("additional_parameter_growth_authorized") is not False:
        raise ValueError("repair manifest must forbid parameter growth")
    if manifest.get("additional_context_growth_authorized") is not False:
        raise ValueError("repair manifest must forbid context growth")
    source_eval = manifest.get("source_evaluation", {})
    if source_eval.get("selected_parent_checkpoint") != PARENT_CHECKPOINT_KEY:
        raise ValueError("repair manifest does not select step-250 parent")
    return {"summary": summary, "manifest": manifest}


def save_checkpoint(
    *,
    accelerator: Any,
    model: Any,
    output_dir: Path,
    step: int,
    parent_dir: Path,
    parent_receipt: dict[str, Any],
    config_path: Path,
    tokenizer_dir: Path,
    corpus_dir: Path,
    source_config_path: Path,
    teacher_registry: Path,
    teacher_audit: Path,
    repair_curriculum: Path,
    repair_manifest: Path,
    trainability: dict[str, int],
    learning_rate: float,
    top_layers: int,
    seed: int,
    mixed_precision: str,
    max_steps: int,
    scheduler_warmup_internal: int,
    scheduler_total_internal: int,
    loss_sums: dict[str, float],
) -> None:
    from safetensors.torch import save_file

    accelerator.wait_for_everyone()
    checkpoint = output_dir / f"step-{step:08d}"
    accelerator.save_state(str(checkpoint / "accelerator_state"))

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        parameter_report = unwrapped.parameter_report()
        if int(parameter_report["total_parameters"]) != EXACT_PARAMETERS:
            raise RuntimeError("repair changed N0 parameter count")

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

        receipt = {
            "schema": "alice.eipm.n0.v02-targeted-repair-checkpoint.v0.1",
            "status": "TRAINED_NOT_PROMOTED",
            "model_id": "alice-n0-semantic-v0.2",
            "repair_step": step,
            "repair_max_steps": max_steps,
            "git_revision": git_revision(),
            "parent_checkpoint": str(parent_dir),
            "parent_step": PARENT_STEP,
            "parent_training_git_revision": parent_receipt.get("git_revision"),
            "parent_full_model_sha256": sha256_file(parent_dir / "alice_n0_v02.safetensors"),
            "parent_receipt_sha256": sha256_file(parent_dir / "receipt.json"),
            "config_sha256": sha256_file(config_path),
            "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
            "corpus_receipt_sha256": sha256_file(corpus_dir / "corpus_receipt.json"),
            "source_config_sha256": sha256_file(source_config_path),
            "teacher_registry_sha256": sha256_file(teacher_registry),
            "teacher_audit_sha256": sha256_file(teacher_audit),
            "repair_curriculum_sha256": sha256_file(repair_curriculum),
            "repair_manifest_sha256": sha256_file(repair_manifest),
            "repair_train_rows": REPAIR_TRAIN_ROWS,
            "repair_dev_rows_held_out": REPAIR_DEV_ROWS,
            "repair_failure_families": sorted(REPAIR_FAMILIES),
            "loss_weights": LOSS_WEIGHTS,
            "teacher_component_weights": TEACHER_COMPONENT_WEIGHTS,
            "mean_losses_since_launch": {
                key: value / max(step, 1) for key, value in loss_sums.items()
            },
            "learning_rate": learning_rate,
            "top_backbone_layers_trainable": top_layers,
            "trainability": trainability,
            "scheduler_warmup_internal": scheduler_warmup_internal,
            "scheduler_total_internal": scheduler_total_internal,
            "serving_graph_unchanged": True,
            "parameter_growth": 0,
            "context_length_growth": 0,
            "runtime_adapter_required": False,
            "seed": seed,
            "world_size": accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "private_identity_data": False,
            "private_identity_gradient": False,
            "private_identity_gradient_authorized": False,
            "model_training_performed": True,
            "promotion_authorized": False,
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
        description="Failure-driven selective repair of alice-n0-semantic-v0.2 from step 250."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--teacher-registry", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--repair-curriculum", required=True)
    parser.add_argument("--repair-manifest", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--teacher-max-length", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
        from safetensors.torch import load_file
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before targeted repair") from exc

    if args.sequence_length != 512:
        raise SystemExit("targeted repair deliberately keeps sequence_length=512")
    if args.max_steps < 1 or args.save_every < 1 or args.max_steps % args.save_every != 0:
        raise SystemExit("max_steps must be positive and divisible by save_every")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise SystemExit("warmup_steps must be non-negative and smaller than max_steps")
    if abs(sum(LOSS_WEIGHTS.values()) - 1.0) > 1e-9:
        raise SystemExit("repair loss weights must sum to 1.0")
    if abs(sum(TEACHER_COMPONENT_WEIGHTS.values()) - 1.0) > 1e-9:
        raise SystemExit("teacher component weights must sum to 1.0")

    config_path = Path(args.config).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    source_config_path = Path(args.source_config).resolve()
    teacher_registry = Path(args.teacher_registry).resolve()
    teacher_audit = Path(args.teacher_audit).resolve()
    repair_curriculum = Path(args.repair_curriculum).resolve()
    repair_manifest = Path(args.repair_manifest).resolve()
    parent_dir = Path(args.parent_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[3]

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty repair output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_n0_config(config_path)
    if args.sequence_length not in config.train_sequence_lengths:
        raise SystemExit("repair sequence length is not allowed by active config")

    verify_tokenizer_v021(tokenizer_dir)
    corpus_receipt, corpus_paths = verify_public_corpus_v021(corpus_dir, source_config_path)
    curriculum_paths, teacher_report = verify_teacher_registry(repo_root, teacher_registry, teacher_audit)
    repair_report = verify_repair_curriculum(repair_curriculum, repair_manifest)
    parent_receipt = verify_parent_checkpoint(parent_dir, tokenizer_dir)

    seed_everything(args.seed)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes != 2:
        raise SystemExit(f"targeted repair requires exactly two accelerator processes; observed={accelerator.num_processes}")
    if not torch.cuda.is_available():
        raise SystemExit("targeted repair requires CUDA")

    tokenizer = load_tokenizer(tokenizer_dir)
    if len(tokenizer) != config.vocab_size:
        raise SystemExit("tokenizer/config vocabulary mismatch")

    model = AliceN0V02Model(config)
    state = load_file(str(parent_dir / "alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"parent full-model state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if int(model.parameter_report()["total_parameters"]) != EXACT_PARAMETERS:
        raise SystemExit("N0 parameter count drifted before repair")
    trainability = configure_selective_repair(model, args.top_layers)

    repair_train = CurriculumDataset(repair_curriculum, "train")
    repair_dev = CurriculumDataset(repair_curriculum, "dev")
    if len(repair_train) != REPAIR_TRAIN_ROWS or len(repair_dev) != REPAIR_DEV_ROWS:
        raise SystemExit("repair split counts changed after validation")

    replay_train = CurriculumDataset(curriculum_paths, "train")
    if len(replay_train) != 765:
        raise SystemExit(f"governed replay bank expected 765 train rows, observed={len(replay_train)}")

    mlm_dataset = PackedJSONLIterableDataset(
        paths=corpus_paths,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        split="train",
        shuffle_seed=args.seed,
    )
    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=1,
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
    repair_loader = DataLoader(
        repair_train,
        batch_size=args.teacher_batch_size,
        shuffle=True,
        collate_fn=TeacherMultitaskCollator(tokenizer, args.teacher_max_length),
        num_workers=0,
        pin_memory=True,
    )
    replay_loader = DataLoader(
        replay_train,
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
        args.max_steps,
        num_processes=accelerator.num_processes,
        split_batches=accelerator.split_batches,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_warmup_internal,
        num_training_steps=scheduler_total_internal,
    )

    model, optimizer, mlm_loader, repair_loader, replay_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        mlm_loader,
        repair_loader,
        replay_loader,
        scheduler,
    )

    if accelerator.is_main_process:
        print(json.dumps({
            "status": "TARGETED_REPAIR_START",
            "model_id": "alice-n0-semantic-v0.2",
            "parent_step": PARENT_STEP,
            "parent_full_model_sha256": parent_receipt.get("full_model_sha256"),
            "repair_rows": repair_report["summary"]["row_count"],
            "repair_train_rows": len(repair_train),
            "repair_dev_rows_held_out": len(repair_dev),
            "governed_teacher_replay_rows": len(replay_train),
            "public_corpus_sources": len(corpus_receipt.get("sources", [])),
            "loss_weights": LOSS_WEIGHTS,
            "teacher_component_weights": TEACHER_COMPONENT_WEIGHTS,
            "trainability": trainability,
            "scheduler_warmup_internal": scheduler_warmup_internal,
            "scheduler_total_internal": scheduler_total_internal,
            "max_steps": args.max_steps,
            "serving_graph_unchanged": True,
            "private_identity_gradient": False,
        }, sort_keys=True))

    model.train()
    mlm_iterator = iter(mlm_loader)
    repair_iterator = iter(repair_loader)
    replay_iterator = iter(replay_loader)
    loss_sums = {
        "public_mlm": 0.0,
        "repair_preference": 0.0,
        "repair_alignment": 0.0,
        "repair_contrastive": 0.0,
        "replay_preference": 0.0,
        "replay_alignment": 0.0,
        "replay_contrastive": 0.0,
        "combined": 0.0,
    }

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        mlm_batch, mlm_iterator = cycle_next(mlm_iterator, mlm_loader)
        with accelerator.no_sync(model):
            mlm_outputs = model(task="mlm", **mlm_batch)
            mlm_loss = mlm_outputs.loss
            accelerator.backward(LOSS_WEIGHTS["public_mlm_replay"] * mlm_loss)

        repair_batch, repair_iterator = cycle_next(repair_iterator, repair_loader)
        with accelerator.no_sync(model):
            repair_losses = teacher_objective_losses(model, repair_batch, temperature=0.05)
            repair_total = teacher_total(repair_losses)
            accelerator.backward(LOSS_WEIGHTS["targeted_repair_teacher"] * repair_total)

        replay_batch, replay_iterator = cycle_next(replay_iterator, replay_loader)
        replay_losses = teacher_objective_losses(model, replay_batch, temperature=0.05)
        replay_total = teacher_total(replay_losses)
        accelerator.backward(LOSS_WEIGHTS["governed_teacher_replay"] * replay_total)

        accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        rp, ra, rc = repair_losses
        gp, ga, gc = replay_losses
        combined_value = (
            LOSS_WEIGHTS["public_mlm_replay"] * float(mlm_loss.detach().cpu())
            + LOSS_WEIGHTS["targeted_repair_teacher"] * float(repair_total.detach().cpu())
            + LOSS_WEIGHTS["governed_teacher_replay"] * float(replay_total.detach().cpu())
        )
        current = {
            "public_mlm": float(mlm_loss.detach().cpu()),
            "repair_preference": float(rp.detach().cpu()),
            "repair_alignment": float(ra.detach().cpu()),
            "repair_contrastive": float(rc.detach().cpu()),
            "replay_preference": float(gp.detach().cpu()),
            "replay_alignment": float(ga.detach().cpu()),
            "replay_contrastive": float(gc.detach().cpu()),
            "combined": combined_value,
        }
        for key, value in current.items():
            loss_sums[key] += value

        if accelerator.is_main_process and (step == 1 or step % 10 == 0):
            print(json.dumps({"step": step, "lr": scheduler.get_last_lr()[0], **current}, sort_keys=True))

        if step % args.save_every == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                output_dir=output_dir,
                step=step,
                parent_dir=parent_dir,
                parent_receipt=parent_receipt,
                config_path=config_path,
                tokenizer_dir=tokenizer_dir,
                corpus_dir=corpus_dir,
                source_config_path=source_config_path,
                teacher_registry=teacher_registry,
                teacher_audit=teacher_audit,
                repair_curriculum=repair_curriculum,
                repair_manifest=repair_manifest,
                trainability=trainability,
                learning_rate=args.learning_rate,
                top_layers=args.top_layers,
                seed=args.seed,
                mixed_precision=args.mixed_precision,
                max_steps=args.max_steps,
                scheduler_warmup_internal=scheduler_warmup_internal,
                scheduler_total_internal=scheduler_total_internal,
                loss_sums=loss_sums,
            )

    if accelerator.is_main_process:
        print(json.dumps({
            "status": "TARGETED_REPAIR_COMPLETE",
            "steps": args.max_steps,
            "saved_steps": list(range(args.save_every, args.max_steps + 1, args.save_every)),
            "promotion_authorized": False,
            "next_action": "evaluate_repair_checkpoints_on_repair_dev_teacher_dev_and_frozen_novel_challenge",
        }, sort_keys=True))


if __name__ == "__main__":
    main()
