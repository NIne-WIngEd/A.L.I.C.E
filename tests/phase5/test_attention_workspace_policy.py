import json
from pathlib import Path

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    load_cognitive_kernel_attention_workspace_policy,
)

ROOT = Path(__file__).resolve().parents[2]


def test_policy_binds_ratified_attention_and_workspace_capabilities():
    policy = load_cognitive_kernel_attention_workspace_policy(
        repository_root=ROOT
    )
    assert policy.version == "0.3.0"
    assert policy.phase == "5"
    assert policy.milestone == "P5.0d"
    assert policy.required_capabilities == (
        "attention_policy.v1",
        "workspace_projection.v1",
        "adaptive_compositor.v1",
        "host_window_override.v1",
    )
    assert policy.invariants["protected_interrupts_cannot_be_suppressed"] is True
    assert policy.invariants["remote_attention_manipulation_allowed"] is False
    assert policy.invariants["complete_ui_implemented"] is False
    assert policy.capability_ceiling is False


def test_policy_rejects_commercial_attention_or_ui_activation(tmp_path):
    source = ROOT / "policies" / "cognitive_kernel_attention_workspace_policy.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["invariants"]["remote_attention_manipulation_allowed"] = True
    payload["invariants"]["complete_ui_implemented"] = True
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CognitiveKernelContractError):
        load_cognitive_kernel_attention_workspace_policy(
            mutated,
            repository_root=ROOT,
        )
