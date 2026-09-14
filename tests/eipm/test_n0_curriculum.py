from __future__ import annotations

import hashlib
import json

import pytest
import torch

from alice_personality.n0.curriculum import (
    git_blob_sha1,
    validate_curriculum_manifest,
    validate_curriculum_rows,
)
from alice_personality.n0.curriculum_data import CurriculumDataset, score_group
from alice_personality.n0.ranker import listwise_preference_loss


def valid_row(row_id: str = "x", split: str = "train") -> dict[str, object]:
    return {
        "id": row_id,
        "competency": "SEM-01",
        "split": split,
        "task": "candidate_ranking",
        "prompt": "Which candidate is supported?",
        "candidates": ["supported", "unsupported"],
        "preferred_indices": [0],
        "rationale": "fixture",
        "source": "test",
    }


def write_valid_curriculum(path) -> None:
    rows = [valid_row("train-1", "train"), valid_row("dev-1", "dev")]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_listwise_preference_loss_rewards_supported_tie() -> None:
    scores_good = torch.tensor([3.0, 3.0, -1.0])
    scores_bad = torch.tensor([-1.0, -1.0, 3.0])
    preferred = [torch.tensor([True, True, False])]
    good = listwise_preference_loss(scores_good, [3], preferred)
    bad = listwise_preference_loss(scores_bad, [3], preferred)
    assert good < bad


def test_score_group_accepts_supported_set_above_negative() -> None:
    result = score_group(
        torch.tensor([2.0, 1.5, -1.0]),
        torch.tensor([True, True, False]),
    )
    assert result["top_supported"] is True
    assert result["supported_set_separated"] is True
    assert result["separation_margin"] > 0


def test_score_group_flags_supported_alternative_below_negative() -> None:
    result = score_group(
        torch.tensor([2.0, -2.0, 1.0]),
        torch.tensor([True, True, False]),
    )
    assert result["top_supported"] is True
    assert result["supported_set_separated"] is False
    assert result["separation_margin"] < 0


def test_curriculum_rows_require_train_and_dev_and_unique_ids(tmp_path) -> None:
    curriculum = tmp_path / "curriculum.jsonl"
    write_valid_curriculum(curriculum)
    summary = validate_curriculum_rows(curriculum)
    assert summary["row_count"] == 2
    assert summary["split_counts"]["train"] == 1
    assert summary["split_counts"]["dev"] == 1

    duplicate = tmp_path / "duplicate.jsonl"
    row = valid_row("same", "train")
    duplicate.write_text(
        json.dumps(row) + "\n" + json.dumps({**row, "split": "dev"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate curriculum id"):
        validate_curriculum_rows(duplicate)


def test_curriculum_dataset_combines_governed_shards(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps(valid_row("train-a", "train"))
        + "\n"
        + json.dumps(valid_row("dev-a", "dev"))
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(valid_row("train-b", "train"))
        + "\n"
        + json.dumps(valid_row("dev-b", "dev"))
        + "\n",
        encoding="utf-8",
    )

    train = CurriculumDataset([first, second], "train")
    dev = CurriculumDataset([first, second], "dev")
    assert [row["id"] for row in train.rows] == ["train-a", "train-b"]
    assert [row["id"] for row in dev.rows] == ["dev-a", "dev-b"]


def test_curriculum_dataset_rejects_cross_shard_duplicate_ids(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(valid_row("same", "train")) + "\n", encoding="utf-8")
    second.write_text(json.dumps(valid_row("same", "dev")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate curriculum id across shards"):
        CurriculumDataset([first, second], "train")


def test_curriculum_manifest_allows_standard_authorized_origin(tmp_path) -> None:
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text('{"id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(curriculum.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "training_authorized": True,
                "private_identity_data": False,
                "origin_type": "self_hosted_permissive_model",
                "curriculum_sha256": digest,
                "provenance_or_authorization": "Apache-2.0-compatible local workflow",
            }
        ),
        encoding="utf-8",
    )
    loaded = validate_curriculum_manifest(curriculum, manifest_path)
    assert loaded["origin_type"] == "self_hosted_permissive_model"


def test_curriculum_manifest_allows_owner_authorized_sol_teacher(tmp_path) -> None:
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text('{"id":"x"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "training_authorized": True,
                "private_identity_data": False,
                "origin_type": "owner_authorized_service_teacher",
                "actor": "sol",
                "actor_instance": "GPT-5.6 Sol",
                "owner_authorization_asserted": True,
                "authorization_scope": "A.L.I.C.E./Fable model-development curriculum",
                "curriculum_git_blob_sha1": git_blob_sha1(curriculum),
                "provenance_or_authorization": "owner explicit project authorization",
            }
        ),
        encoding="utf-8",
    )
    loaded = validate_curriculum_manifest(curriculum, manifest_path)
    assert loaded["actor"] == "sol"


def test_curriculum_manifest_rejects_unasserted_teacher_authorization(tmp_path) -> None:
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text('{"id":"x"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "training_authorized": True,
                "private_identity_data": False,
                "origin_type": "owner_authorized_service_teacher",
                "actor": "sol",
                "owner_authorization_asserted": False,
                "authorization_scope": "A.L.I.C.E./Fable model-development curriculum",
                "curriculum_git_blob_sha1": git_blob_sha1(curriculum),
                "provenance_or_authorization": "missing active owner authorization",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit owner authorization"):
        validate_curriculum_manifest(curriculum, manifest_path)


def test_curriculum_manifest_rejects_missing_integrity_hash(tmp_path) -> None:
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text('{"id":"x"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "training_authorized": True,
                "private_identity_data": False,
                "origin_type": "owner_authored",
                "provenance_or_authorization": "owner-authored",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="curriculum_sha256 or curriculum_git_blob_sha1"):
        validate_curriculum_manifest(curriculum, manifest_path)
