from __future__ import annotations

import hashlib
import json

import pytest
import torch

from alice_personality.n0.curriculum import git_blob_sha1, validate_curriculum_manifest
from alice_personality.n0.ranker import listwise_preference_loss


def test_listwise_preference_loss_rewards_supported_tie() -> None:
    scores_good = torch.tensor([3.0, 3.0, -1.0])
    scores_bad = torch.tensor([-1.0, -1.0, 3.0])
    preferred = [torch.tensor([True, True, False])]
    good = listwise_preference_loss(scores_good, [3], preferred)
    bad = listwise_preference_loss(scores_bad, [3], preferred)
    assert good < bad


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
