#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum import sha256_file
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder
from alice_personality.n0.structured_state_objectives import structured_state_objective
from alice_personality.n0.v02_model import AliceN0V02Model


SEMANTIC_MODEL_ID = "alice-n0-semantic-v0.2"
EXPECTED_SEMANTIC_PARAMETERS = 136_594_435
EXPECTED_BRANCH_PARAMETERS = 1_656_064


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_structured_config(path: Path) -> StructuredStateConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    architecture = raw["architecture"]
    return StructuredStateConfig(
        semantic_size=int(architecture["semantic_size"]),
        state_size=int(architecture["state_size"]),
        num_layers=int(architecture["num_layers"]),
        num_heads=int(architecture["num_heads"]),
        feedforward_size=int(architecture["feedforward_size"]),
        dropout=float(architecture["dropout"]),
        max_fields=int(architecture["max_fields"]),
    )


def verify_inputs(
    *,
    semantic_checkpoint: Path,
    semantic_ratification: Path,
    curriculum: Path,
    curriculum_manifest: Path,
    structured_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    full_model = semantic_checkpoint / "alice_n0_v02.safetensors"
    receipt_path = semantic_checkpoint / "receipt.json"
    if not full_model.is_file() or not receipt_path.is_file():
        raise SystemExit("ratified semantic checkpoint is incomplete")

    ratification = json.loads(semantic_ratification.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(curriculum_manifest.read_text(encoding="utf-8"))
    structured_config = json.loads(structured_config_path.read_text(encoding="utf-8"))

    if ratification.get("semantic_base_ratified") is not True:
        raise SystemExit("semantic parent is not ratified")
    if ratification.get("selected_checkpoint") != "targeted-repair-v0.1/step-00000080":
        raise SystemExit("structured pilot must use ratified repair step080")
    expected_hash = str(ratification["checkpoint_hashes"]["full_model_sha256"])
    if sha256_file(full_model) != expected_hash:
        raise SystemExit("semantic checkpoint hash does not match ratification")
    if int(receipt.get("repair_step", -1)) != 80:
        raise SystemExit("semantic checkpoint receipt is not repair step080")
    if receipt.get("private_identity_gradient") is not False:
        raise SystemExit("semantic parent unexpectedly contains private identity gradient")

    if manifest.get("schema") != "alice.eipm.n0.v02-structured-state-curriculum.v0.2":
        raise SystemExit("structured curriculum schema mismatch")
    if manifest.get("status") != "COMPILED_NOT_ACTIVATED":
        raise SystemExit("structured curriculum has unexpected activation state")
    if manifest.get("semantic_target_mode") != "context_candidate_pair":
        raise SystemExit("structured curriculum semantic target mode mismatch")
    if manifest.get("rationale_target_separate") is not True:
        raise SystemExit("structured curriculum rationale target is not separate")
    if manifest.get("text_generated_by_compiler") is not False:
        raise SystemExit("structured curriculum unexpectedly contains compiler-generated text")
    if sha256_file(curriculum) != str(manifest.get("compiled_sha256")):
        raise SystemExit("structured curriculum hash mismatch")
    if int(manifest.get("competency_count", -1)) != 51:
        raise SystemExit("structured curriculum lost 51-competency coverage")

    boundary = structured_config.get("training_boundary", {})
    if boundary.get("public_structured_alignment_only") is not True:
        raise SystemExit("structured config does not restrict pilot to public alignment")
    if boundary.get("private_identity_gradient") is not False:
        raise SystemExit("structured config unexpectedly permits private identity gradient")
    if boundary.get("gradient_authorized") is not True:
        raise SystemExit("public structured-state gradient pilot is not authorized in config")

    return ratification, manifest, receipt


def encode_batches(
    *,
    model: AliceN0V02Model,
    tokenizer: Any,
    texts_a: list[str],
    texts_b: list[str] | None,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts_a), batch_size):
        end = start + batch_size
        a = texts_a[start:end]
        b = None if texts_b is None else texts_b[start:end]
        encoded = tokenizer(
            a,
            b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            pooled = model.encode(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
        chunks.append(pooled.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def build_cache(
    *,
    rows: list[dict[str, Any]],
    model: AliceN0V02Model,
    tokenizer: Any,
    device: torch.device,
    encode_batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    if any(len(row.get("fields", [])) != 2 for row in rows):
        raise SystemExit("v0.1 structured pilot expects exactly context and candidate fields")

    contexts = [str(row["fields"][0]["text"]) for row in rows]
    candidates = [str(row["fields"][1]["text"]) for row in rows]
    rationales = [str(row["rationale_text"]) for row in rows]

    for row, context, candidate in zip(rows, contexts, candidates):
        target = row.get("semantic_target", {})
        if target.get("mode") != "context_candidate_pair":
            raise SystemExit(f"bad semantic target mode for {row.get('id')}")
        if target.get("text_a") != context or target.get("text_b") != candidate:
            raise SystemExit(f"semantic target text drift for {row.get('id')}")

    context_vectors = encode_batches(
        model=model,
        tokenizer=tokenizer,
        texts_a=contexts,
        texts_b=None,
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    candidate_vectors = encode_batches(
        model=model,
        tokenizer=tokenizer,
        texts_a=candidates,
        texts_b=None,
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    semantic_targets = encode_batches(
        model=model,
        tokenizer=tokenizer,
        texts_a=contexts,
        texts_b=candidates,
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    rationale_targets = encode_batches(
        model=model,
        tokenizer=tokenizer,
        texts_a=rationales,
        texts_b=None,
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )

    field_semantic = torch.stack([context_vectors, candidate_vectors], dim=1)
    field_type_ids = torch.tensor(
        [[int(field["field_type_id"]) for field in row["fields"]] for row in rows],
        dtype=torch.long,
    )
    provenance_ids = torch.tensor(
        [[int(field["provenance_id"]) for field in row["fields"]] for row in rows],
        dtype=torch.long,
    )
    relation_role_ids = torch.tensor(
        [[int(field["relation_role_id"]) for field in row["fields"]] for row in rows],
        dtype=torch.long,
    )
    temporal_scope_ids = torch.tensor(
        [[int(field["temporal_scope_id"]) for field in row["fields"]] for row in rows],
        dtype=torch.long,
    )
    confidence = torch.tensor(
        [[[float(field["confidence"])] for field in row["fields"]] for row in rows],
        dtype=torch.float32,
    )
    missing_mask = torch.tensor(
        [[bool(field["missing"]) for field in row["fields"]] for row in rows],
        dtype=torch.bool,
    )
    valid_mask = torch.ones(len(rows), 2, dtype=torch.bool)
    labels = torch.tensor([float(row["compatibility_label"]) for row in rows], dtype=torch.float32)

    return {
        "field_semantic": field_semantic,
        "semantic_target": semantic_targets,
        "rationale_target": rationale_targets,
        "field_type_ids": field_type_ids,
        "provenance_ids": provenance_ids,
        "relation_role_ids": relation_role_ids,
        "temporal_scope_ids": temporal_scope_ids,
        "confidence": confidence,
        "missing_mask": missing_mask,
        "valid_mask": valid_mask,
        "labels": labels,
        "ids": [str(row["id"]) for row in rows],
        "source_teacher_ids": [str(row["source_teacher_id"]) for row in rows],
        "candidate_indices": [int(row["source_candidate_index"]) for row in rows],
        "competencies": [str(row["competency"]) for row in rows],
    }


class CachedStructuredDataset(Dataset):
    TENSOR_KEYS = (
        "field_semantic",
        "semantic_target",
        "rationale_target",
        "field_type_ids",
        "provenance_ids",
        "relation_role_ids",
        "temporal_scope_ids",
        "confidence",
        "missing_mask",
        "valid_mask",
        "labels",
    )

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.size = int(payload["labels"].shape[0])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: self.payload[key][index] for key in self.TENSOR_KEYS}
        item["index"] = index
        return item


def branch_forward(branch: StructuredStateEncoder, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return branch(
        semantic_values=batch["field_semantic"],
        field_type_ids=batch["field_type_ids"],
        provenance_ids=batch["provenance_ids"],
        relation_role_ids=batch["relation_role_ids"],
        temporal_scope_ids=batch["temporal_scope_ids"],
        confidence=batch["confidence"],
        missing_mask=batch["missing_mask"],
        valid_mask=batch["valid_mask"],
    )


def permute_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    order = torch.tensor([1, 0], device=batch["field_semantic"].device)
    permuted = dict(batch)
    for key in (
        "field_semantic",
        "field_type_ids",
        "provenance_ids",
        "relation_role_ids",
        "temporal_scope_ids",
        "confidence",
        "missing_mask",
        "valid_mask",
    ):
        permuted[key] = batch[key].index_select(1, order)
    return permuted


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compatibility_logits(pooled: torch.Tensor, rationale: torch.Tensor) -> torch.Tensor:
    return 5.0 * F.cosine_similarity(pooled, rationale, dim=-1)


def evaluate(
    *,
    branch: StructuredStateEncoder,
    dataset: CachedStructuredDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
    positive_weight: float,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    branch.eval()

    losses = defaultdict(float)
    examples = 0
    logits_all: list[float] = []
    labels_all: list[int] = []
    semantic_cosines: list[float] = []
    field_cosines: list[float] = []
    permutation_cosines: list[float] = []

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = branch_forward(branch, batch)
            permuted = branch_forward(branch, permute_batch(batch))
            result = structured_state_objective(
                pooled_state=outputs["pooled_state"],
                semantic_target=batch["semantic_target"],
                rationale_target=batch["rationale_target"],
                field_states=outputs["field_states"],
                field_semantic=batch["field_semantic"],
                valid_mask=batch["valid_mask"],
                permuted_pooled_state=permuted["pooled_state"],
                compatibility_labels=batch["labels"],
                compatibility_positive_weight=positive_weight,
            )
            n = int(batch["labels"].numel())
            examples += n
            for key, value in result.items():
                losses[key] += float(value.detach().cpu()) * n

            logits = compatibility_logits(outputs["pooled_state"], batch["rationale_target"])
            logits_all.extend(float(x) for x in logits.detach().cpu().tolist())
            labels_all.extend(int(x) for x in batch["labels"].detach().cpu().tolist())
            semantic_cosines.extend(
                float(x)
                for x in F.cosine_similarity(
                    outputs["pooled_state"], batch["semantic_target"], dim=-1
                ).detach().cpu().tolist()
            )
            field_cosines.extend(
                float(x)
                for x in F.cosine_similarity(
                    outputs["field_states"], batch["field_semantic"], dim=-1
                )[batch["valid_mask"]].detach().cpu().tolist()
            )
            permutation_cosines.extend(
                float(x)
                for x in F.cosine_similarity(
                    outputs["pooled_state"], permuted["pooled_state"], dim=-1
                ).detach().cpu().tolist()
            )

    predictions = [1 if value >= 0.0 else 0 for value in logits_all]
    tp = sum(1 for p, y in zip(predictions, labels_all) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(predictions, labels_all) if p == 0 and y == 0)
    positives = sum(labels_all)
    negatives = len(labels_all) - positives
    positive_recall = tp / max(positives, 1)
    negative_recall = tn / max(negatives, 1)
    balanced_accuracy = 0.5 * (positive_recall + negative_recall)
    accuracy = sum(int(p == y) for p, y in zip(predictions, labels_all)) / max(len(labels_all), 1)

    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for teacher_id, logit, label in zip(payload["source_teacher_ids"], logits_all, labels_all):
        grouped[str(teacher_id)].append((float(logit), int(label)))
    group_pass = 0
    for group in grouped.values():
        best = max(group, key=lambda item: item[0])
        group_pass += int(best[1] == 1)

    return {
        "examples": examples,
        "losses": {key: value / max(examples, 1) for key, value in sorted(losses.items())},
        "compatibility_accuracy": accuracy,
        "compatibility_positive_recall": positive_recall,
        "compatibility_negative_recall": negative_recall,
        "compatibility_balanced_accuracy": balanced_accuracy,
        "grouped_teacher_rows": len(grouped),
        "grouped_preferred_top1_accuracy": group_pass / max(len(grouped), 1),
        "mean_semantic_alignment_cosine": sum(semantic_cosines) / max(len(semantic_cosines), 1),
        "mean_field_preservation_cosine": sum(field_cosines) / max(len(field_cosines), 1),
        "mean_permutation_cosine": sum(permutation_cosines) / max(len(permutation_cosines), 1),
    }


def save_branch(
    *,
    branch: StructuredStateEncoder,
    output_root: Path,
    step: int,
    metrics: dict[str, Any],
    parent_hash: str,
    curriculum_hash: str,
    structured_config_hash: str,
    seed: int,
    learning_rate: float,
    positive_weight: float,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    checkpoint = output_root / f"step-{step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint / "structured_state.safetensors"
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in branch.state_dict().items()
    }
    save_file(state, str(state_path))
    receipt = {
        "schema": "alice.eipm.n0.v02-structured-state-pilot-checkpoint.v0.1",
        "status": "TRAINED_NOT_RATIFIED",
        "step": step,
        "git_revision": git_revision(),
        "semantic_parent_full_model_sha256": parent_hash,
        "structured_state_sha256": sha256_file(state_path),
        "curriculum_sha256": curriculum_hash,
        "structured_config_sha256": structured_config_hash,
        "seed": seed,
        "learning_rate": learning_rate,
        "compatibility_positive_weight": positive_weight,
        "branch_parameter_count": branch.parameter_report()["total_parameters"],
        "semantic_core_trainable_parameters": 0,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "metrics": metrics,
    }
    (checkpoint / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--semantic-ratification", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--save-every", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("structured-state pilot requires one CUDA device for semantic caching")
    device = torch.device("cuda")
    seed_everything(args.seed)

    config_path = Path(args.config).resolve()
    structured_config_path = Path(args.structured_config).resolve()
    semantic_ratification = Path(args.semantic_ratification).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    curriculum = Path(args.curriculum).resolve()
    curriculum_manifest = Path(args.curriculum_manifest).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    ratification, manifest, parent_receipt = verify_inputs(
        semantic_checkpoint=semantic_checkpoint,
        semantic_ratification=semantic_ratification,
        curriculum=curriculum,
        curriculum_manifest=curriculum_manifest,
        structured_config_path=structured_config_path,
    )

    n0_config = load_n0_config(config_path)
    tokenizer = load_tokenizer(tokenizer_dir)
    if len(tokenizer) != n0_config.vocab_size:
        raise SystemExit("tokenizer/config vocabulary mismatch")

    from safetensors.torch import load_file

    semantic_model = AliceN0V02Model(n0_config)
    state = load_file(str(semantic_checkpoint / "alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = semantic_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    semantic_report = semantic_model.parameter_report()
    if int(semantic_report["total_parameters"]) != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model.to(device).eval()

    rows = read_jsonl(curriculum)
    train_rows = [row for row in rows if row.get("split") == "train"]
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    if len(train_rows) != int(manifest["compiled_split_counts"]["train"]):
        raise SystemExit("train row count does not match manifest")
    if len(dev_rows) != int(manifest["compiled_split_counts"]["dev"]):
        raise SystemExit("dev row count does not match manifest")

    print("semantic_cache_start=true", flush=True)
    train_payload = build_cache(
        rows=train_rows,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    dev_payload = build_cache(
        rows=dev_rows,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    parent_hash = sha256_file(semantic_checkpoint / "alice_n0_v02.safetensors")
    cache_path = output_root / "semantic_cache.pt"
    torch.save(
        {
            "semantic_parent_full_model_sha256": parent_hash,
            "curriculum_sha256": sha256_file(curriculum),
            "train": train_payload,
            "dev": dev_payload,
        },
        cache_path,
    )
    print(f"semantic_cache_complete=true path={cache_path}", flush=True)
    del semantic_model, state
    torch.cuda.empty_cache()

    branch = StructuredStateEncoder(load_structured_config(structured_config_path)).to(device)
    report = branch.parameter_report()
    if int(report["total_parameters"]) != EXPECTED_BRANCH_PARAMETERS:
        raise SystemExit(f"structured branch parameter drift: {report}")

    train_dataset = CachedStructuredDataset(train_payload)
    dev_dataset = CachedStructuredDataset(dev_payload)
    positives = int(train_payload["labels"].sum().item())
    negatives = len(train_dataset) - positives
    if positives < 1 or negatives < 1:
        raise SystemExit("training split requires positive and negative compatibility labels")
    positive_weight = negatives / positives

    baseline = evaluate(
        branch=branch,
        dataset=dev_dataset,
        payload=dev_payload,
        device=device,
        batch_size=args.eval_batch_size,
        positive_weight=positive_weight,
    )
    (output_root / "random_baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("random_baseline=" + json.dumps(baseline, sort_keys=True), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        branch.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-6)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    iterator = iter(train_loader)
    running = defaultdict(float)
    checkpoint_receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        branch.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = branch_forward(branch, batch)
        permuted = branch_forward(branch, permute_batch(batch))
        result = structured_state_objective(
            pooled_state=outputs["pooled_state"],
            semantic_target=batch["semantic_target"],
            rationale_target=batch["rationale_target"],
            field_states=outputs["field_states"],
            field_semantic=batch["field_semantic"],
            valid_mask=batch["valid_mask"],
            permuted_pooled_state=permuted["pooled_state"],
            compatibility_labels=batch["labels"],
            compatibility_positive_weight=positive_weight,
        )
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        for key, value in result.items():
            running[key] += float(value.detach().cpu())

        if step % 20 == 0 or step == 1:
            means = {key: value / step for key, value in sorted(running.items())}
            print(
                json.dumps(
                    {"step": step, "lr": scheduler.get_last_lr()[0], "mean_train_losses": means},
                    sort_keys=True,
                ),
                flush=True,
            )

        if step % args.save_every == 0 or step == args.max_steps:
            metrics = evaluate(
                branch=branch,
                dataset=dev_dataset,
                payload=dev_payload,
                device=device,
                batch_size=args.eval_batch_size,
                positive_weight=positive_weight,
            )
            receipt = save_branch(
                branch=branch,
                output_root=output_root,
                step=step,
                metrics=metrics,
                parent_hash=parent_hash,
                curriculum_hash=sha256_file(curriculum),
                structured_config_hash=sha256_file(structured_config_path),
                seed=args.seed,
                learning_rate=args.learning_rate,
                positive_weight=positive_weight,
            )
            checkpoint_receipts[f"step-{step:08d}"] = receipt
            print("checkpoint=" + json.dumps(receipt, sort_keys=True), flush=True)

    def score(receipt: dict[str, Any]) -> tuple[float, float, float, int]:
        metrics = receipt["metrics"]
        return (
            float(metrics["grouped_preferred_top1_accuracy"]),
            float(metrics["compatibility_balanced_accuracy"]),
            float(metrics["mean_semantic_alignment_cosine"]),
            -int(receipt["step"]),
        )

    winner_key = max(checkpoint_receipts, key=lambda key: score(checkpoint_receipts[key]))
    winner = checkpoint_receipts[winner_key]
    wm = winner["metrics"]
    gate_pass = (
        float(wm["grouped_preferred_top1_accuracy"])
        >= float(baseline["grouped_preferred_top1_accuracy"]) + 0.05
        and float(wm["compatibility_balanced_accuracy"])
        >= float(baseline["compatibility_balanced_accuracy"]) + 0.05
        and float(wm["mean_semantic_alignment_cosine"])
        > float(baseline["mean_semantic_alignment_cosine"])
        and float(wm["mean_permutation_cosine"]) >= 0.999
    )
    comparison = {
        "schema": "alice.eipm.n0.v02-structured-state-pilot-comparison.v0.1",
        "status": "PASS" if gate_pass else "NEEDS_REVIEW",
        "semantic_parent": "targeted-repair-v0.1/step-00000080",
        "semantic_parent_full_model_sha256": parent_hash,
        "semantic_core_trainable_parameters": 0,
        "branch_parameter_count": EXPECTED_BRANCH_PARAMETERS,
        "curriculum_rows": len(rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "compatibility_positive_weight": positive_weight,
        "random_baseline": baseline,
        "checkpoints": checkpoint_receipts,
        "provisional_winner": winner_key,
        "pilot_gate_pass": gate_pass,
        "selection_policy": "grouped_preferred_top1_then_balanced_compatibility_then_semantic_alignment_then_earlier_checkpoint",
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": (
            "inspect_and_ratify_structured_branch_then_begin_graph_or_fusion_capability"
            if gate_pass
            else "inspect_failure_modes_before_any_additional_structured_gradient"
        ),
    }
    (output_root / "structured_state_pilot_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
