from __future__ import annotations

import hashlib
import json

import pytest
import torch

from alice_personality.n0.curriculum import validate_curriculum_manifest
from alice_personality.n0.ranker import listwise_preference_loss


def test_listwise_preference_loss_rewards_supported_tie() -> None:
    scores_good = torch.tensor([3.0, 3.0, -1.0])
    scores_bad = torch.tensor([-1.0, -1.0, 3.0])
    preferred = [torch.tensor([True, True, False])]
    good = listwise_preference_loss(scores_good, [3], preferred)
    bad = listwise_preference_loss(scores_bad, [3], preferred)
    assert good < bad


def test_curriculum_manifest_requires_training_authority_and_hash(tmp_path) -> None:
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
                "rights_or_license": "Apache-2.0-compatible-output-workflow",
                "terms_or_license_reviewed": True,
            }
        ),
        encoding="utf-8",
    )
    loaded = validate_curriculum_manifest(curriculum, manifest_path)
    assert loaded["origin_type"] == "self_hosted_permissive_model"


def test_curriculum_manifest_rejects_openai_service_output(tmp_path) -> None:
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text('{"id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(curriculum.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "training_authorized": True,
                "private_identity_data": False,
                "origin_type": "openai_service_output",
                "curriculum_sha256": digest,
                "rights_or_license": "service-output",
                "terms_or_license_reviewed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prohibited"):
        validate_curriculum_manifest(curriculum, manifest_path)
