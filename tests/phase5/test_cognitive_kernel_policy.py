from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    load_cognitive_kernel_foundation_policy,
)

ROOT = Path(__file__).resolve().parents[2]


def test_policy_matches_ratified_product_identity_and_storage_sources() -> None:
    policy = load_cognitive_kernel_foundation_policy(
        repository_root=ROOT
    )
    assert policy.phase == "5"
    assert policy.milestone == "P5.0b"
    assert policy.shared_kernel_id == "personal-cognitive-kernel"
    assert set(policy.required_products) == {"alice", "friday"}
    assert policy.boundaries["metadata_only_contracts"] is True
    assert policy.boundaries["private_payload_allowed"] is False
    assert policy.capability_ceiling is False
    assert len(policy.digest) == 64


def test_policy_rejects_mutated_provenance_vocabulary(
    tmp_path: Path,
) -> None:
    source = (
        ROOT / "policies" / "cognitive_kernel_foundation_policy.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["allowed_provenance_types"].remove(
        "generated_reconstruction"
    )
    mutated = tmp_path / "mutated.json"
    mutated.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(CognitiveKernelContractError):
        load_cognitive_kernel_foundation_policy(
            mutated,
            repository_root=ROOT,
        )


def test_policy_rejects_runtime_capability_claims() -> None:
    policy = load_cognitive_kernel_foundation_policy(
        repository_root=ROOT
    )
    for field in (
        "persistent_store_implemented",
        "raw_buffer_implemented",
        "complete_ui_implemented",
        "autonomous_learning_implemented",
    ):
        assert policy.boundaries[field] is False
