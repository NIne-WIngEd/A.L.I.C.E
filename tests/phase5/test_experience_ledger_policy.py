from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    load_cognitive_kernel_experience_ledger_policy,
)

ROOT = Path(__file__).resolve().parents[2]


def test_experience_ledger_policy_loads() -> None:
    policy = load_cognitive_kernel_experience_ledger_policy(
        repository_root=ROOT
    )
    assert policy.version == "0.6.0"
    assert policy.milestone == "P5.1a"
    assert policy.required_contracts == (
        "experience_event",
        "storage_lifecycle",
    )
    assert policy.invariants["logically_append_only"] is True
    assert policy.invariants["raw_buffer_implemented"] is False
    assert policy.invariants["backup_restore_implemented"] is False
    assert policy.invariants["friday_product_source_implemented"] is False


def test_tampered_policy_boundary_is_rejected(tmp_path: Path) -> None:
    source = json.loads(
        (
            ROOT
            / "policies/cognitive_kernel_experience_ledger_policy.json"
        ).read_text(encoding="utf-8")
    )
    source["invariants"]["private_payload_allowed"] = True
    path = tmp_path / "ledger-policy.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(CognitiveKernelContractError, match="must be false"):
        load_cognitive_kernel_experience_ledger_policy(
            path=path,
            repository_root=ROOT,
        )
