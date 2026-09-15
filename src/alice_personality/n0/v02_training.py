from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .curriculum import validate_curriculum_manifest, validate_curriculum_rows
from .ranker import listwise_preference_loss
from .v02_objectives import (
    multi_positive_contrastive_loss,
    principle_alignment_loss,
    validate_teacher_row_for_v02,
)

EXPECTED_CORPUS_SCHEMA = "alice.eipm.n0.derived-tokenizer-corpus-receipt.v0.2.1"
EXPECTED_SOURCE_SCHEMA = "alice.eipm.n0.public-corpus-sources.v0.2.1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_corpus_v021(
    corpus_dir: str | Path,
    source_config_path: str | Path,
) -> tuple[dict[str, Any], list[Path]]:
    root = Path(corpus_dir).resolve()
    source_config_path = Path(source_config_path).resolve()
    receipt_path = root / "corpus_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"missing corpus receipt: {receipt_path}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != EXPECTED_CORPUS_SCHEMA or receipt.get("status") != "PASS":
        raise ValueError("N0 v0.2 requires the passed derived v0.2.1 public corpus")
    for key in ("private_identity_data", "private_identity_gradient", "model_training_performed"):
        if receipt.get(key) is not False:
            raise ValueError(f"public corpus must declare {key}=false")

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if source_config.get("schema") != EXPECTED_SOURCE_SCHEMA:
        raise ValueError("unexpected v0.2.1 source-manifest schema")
    if source_config.get("status") != "activated_public_n0_v02":
        raise ValueError("v0.2.1 public source manifest is not activated")
    if receipt.get("source_config_sha256") != sha256_file(source_config_path):
        raise ValueError("corpus source-config hash mismatch")

    active = {str(row["source_id"]): row for row in source_config.get("sources", [])}
    observed = receipt.get("sources")
    if len(active) != 21 or not isinstance(observed, list) or len(observed) != 21:
        raise ValueError("v0.2.1 public corpus must contain all 21 activated sources")

    paths: list[Path] = []
    for source in observed:
        source_id = str(source.get("source_id", ""))
        spec = active.get(source_id)
        if spec is None:
            raise ValueError(f"receipt source absent from active manifest: {source_id}")
        if str(source.get("revision")) != str(spec.get("revision")):
            raise ValueError(f"source revision mismatch: {source_id}")
        if abs(float(source.get("target_share", 0.0)) - float(spec.get("target_share", 0.0))) > 1e-12:
            raise ValueError(f"source mixture-share mismatch: {source_id}")
        if float(source.get("fill_ratio", 0.0)) < 0.999:
            raise ValueError(f"source fill ratio below threshold: {source_id}")
        shards = source.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"source has no materialized shards: {source_id}")
        for shard in shards:
            rel = Path(str(shard["path"]))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe shard path: {rel}")
            path = (root / rel).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"missing or escaping corpus shard: {rel}")
            if path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"corpus shard size mismatch: {rel}")
            if sha256_file(path) != str(shard["sha256"]):
                raise ValueError(f"corpus shard hash mismatch: {rel}")
            paths.append(path)
    if not paths:
        raise ValueError("verified corpus produced no shard paths")
    return receipt, sorted(paths)


def verify_tokenizer_v021(tokenizer_dir: str | Path) -> dict[str, Any]:
    root = Path(tokenizer_dir).resolve()
    tokenizer_path = root / "tokenizer.json"
    receipt_path = root / "tokenizer_receipt.json"
    if not tokenizer_path.is_file() or not receipt_path.is_file():
        raise ValueError("v0.2.1 tokenizer or tokenizer receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("model_id") != "alice-n0-semantic-v0.2":
        raise ValueError("tokenizer model_id mismatch")
    if int(receipt.get("vocab_size_observed", 0)) != 48_000:
        raise ValueError("tokenizer vocab must be 48000")
    if receipt.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("tokenizer hash mismatch")
    for key in ("private_identity_data", "private_identity_gradient", "model_training_performed"):
        if receipt.get(key) is not False:
            raise ValueError(f"tokenizer must declare {key}=false")
    return receipt


def resolve_registry_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def verify_teacher_registry(
    repo_root: str | Path,
    registry_path: str | Path,
    audit_path: str | Path,
) -> tuple[list[Path], dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    registry_path = Path(registry_path).resolve()
    audit_path = Path(audit_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if audit.get("status") != "PASS" or audit.get("full_multitask_gate_open") is not True:
        raise ValueError("teacher-bank audit has not opened the full multitask gate")
    if int(audit.get("registered_rows", 0)) < 1020 or int(audit.get("competency_count", 0)) != 51:
        raise ValueError("teacher bank does not match the 1020-row / 51-competency floor")
    if audit.get("coverage_gate_ok") is not True:
        raise ValueError("teacher-bank coverage gate is not satisfied")
    if audit.get("private_identity_data") is not False or audit.get("private_identity_gradient_authorized") is not False:
        raise ValueError("N0 teacher bank must remain public/non-identity")

    curricula: list[Path] = []
    total = 0
    for shard in registry.get("shards", []):
        curriculum = resolve_registry_path(repo_root, str(shard["curriculum"]))
        manifest = resolve_registry_path(repo_root, str(shard["manifest"]))
        summary = validate_curriculum_rows(curriculum)
        authority = validate_curriculum_manifest(curriculum, manifest)
        if authority.get("private_identity_data") is not False:
            raise ValueError(f"private curriculum forbidden in N0: {curriculum}")
        if authority.get("private_identity_gradient_authorized") not in (False, None):
            raise ValueError(f"private gradient unexpectedly authorized: {manifest}")
        total += int(summary["row_count"])
        curricula.append(curriculum)
    if total != int(audit["registered_rows"]):
        raise ValueError(f"teacher registry rows {total} != audit rows {audit['registered_rows']}")
    return curricula, audit


class TeacherMultitaskCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        prompts: list[str] = []
        candidates: list[str] = []
        rationales: list[str] = []
        group_sizes: list[int] = []
        preferred_masks: list[torch.Tensor] = []
        principle_tags: list[str] = []
        ids: list[str] = []
        candidate_rationale_index: list[int] = []

        for row_index, row in enumerate(rows):
            validate_teacher_row_for_v02(row)
            row_candidates = [str(value) for value in row["candidates"]]
            preferred = {int(value) for value in row["preferred_indices"]}
            ids.append(str(row["id"]))
            group_sizes.append(len(row_candidates))
            preferred_masks.append(
                torch.tensor([index in preferred for index in range(len(row_candidates))], dtype=torch.bool)
            )
            prompts.extend([str(row["prompt"])] * len(row_candidates))
            candidates.extend(row_candidates)
            candidate_rationale_index.extend([row_index] * len(row_candidates))
            rationales.append(str(row["rationale"]))
            principle_tags.append(str(row.get("principle_tag") or row["competency"]))

        candidate = self.tokenizer(
            prompts,
            candidates,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        rationale = self.tokenizer(
            rationales,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "candidate_input_ids": candidate["input_ids"],
            "candidate_attention_mask": candidate["attention_mask"],
            "rationale_input_ids": rationale["input_ids"],
            "rationale_attention_mask": rationale["attention_mask"],
            "candidate_rationale_index": torch.tensor(candidate_rationale_index, dtype=torch.long),
            "group_sizes": group_sizes,
            "preferred_masks": preferred_masks,
            "principle_tags": principle_tags,
            "ids": ids,
        }


def teacher_objective_losses(
    model: Any,
    batch: dict[str, Any],
    *,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(
        task="teacher",
        candidate_input_ids=batch["candidate_input_ids"],
        candidate_attention_mask=batch["candidate_attention_mask"],
        rationale_input_ids=batch["rationale_input_ids"],
        rationale_attention_mask=batch["rationale_attention_mask"],
        candidate_rationale_index=batch["candidate_rationale_index"],
    )
    scores = outputs["scores"]
    semantic = outputs["semantic"]
    rationale = outputs["rationale"]
    alignment_logits = outputs["alignment_logits"]

    preference = listwise_preference_loss(scores, batch["group_sizes"], batch["preferred_masks"])

    labels: list[torch.Tensor] = []
    positive_indices: list[int] = []
    offset = 0
    for size, preferred in zip(batch["group_sizes"], batch["preferred_masks"]):
        labels.append(preferred.to(device=semantic.device, dtype=semantic.dtype))
        first_preferred = int(torch.nonzero(preferred, as_tuple=False)[0].item())
        positive_indices.append(offset + first_preferred)
        offset += size
    alignment = principle_alignment_loss(alignment_logits, torch.cat(labels, dim=0))

    selected = semantic[
        torch.tensor(positive_indices, device=semantic.device, dtype=torch.long)
    ]
    contrastive = multi_positive_contrastive_loss(
        selected,
        rationale,
        batch["principle_tags"],
        temperature=temperature,
    )
    return preference, alignment, contrastive


def export_ranker_state(model: Any) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in model.backbone.state_dict().items():
        state[f"backbone.{name}"] = value.detach().cpu().contiguous()
    for name, value in model.preference_scorer.state_dict().items():
        state[f"scorer.{name}"] = value.detach().cpu().contiguous()
    return state
