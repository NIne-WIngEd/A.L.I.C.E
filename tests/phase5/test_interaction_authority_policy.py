import json
from pathlib import Path

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    load_cognitive_kernel_interaction_authority_policy,
)


def synthetic_repository(tmp_path: Path) -> Path:
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    source_root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (source_root / "policies" / "cognitive_kernel_interaction_authority_policy.json").read_text(encoding="utf-8")
    )
    (policy_root / "cognitive_kernel_interaction_authority_policy.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    (policy_root / "cognitive_kernel_attention_workspace_policy.json").write_text(
        json.dumps({"version": "0.3.0", "milestone": "P5.0d"}), encoding="utf-8"
    )
    (policy_root / "capability_parity_ledger.json").write_text(
        json.dumps({"capabilities": ["speaker_context.v1", "guest_session.v1", "guest_grant.v1"]}),
        encoding="utf-8",
    )
    (policy_root / "product_lines.json").write_text(
        json.dumps({"phase5_contracts": ["speaker_context", "guest_grant"]}),
        encoding="utf-8",
    )
    return tmp_path


def test_interaction_authority_policy_loads_with_exact_contracts(tmp_path):
    root = synthetic_repository(tmp_path)
    policy = load_cognitive_kernel_interaction_authority_policy(
        repository_root=root
    )
    assert policy.version == "0.4.0"
    assert policy.milestone == "P5.0e"
    assert policy.capability_ceiling is False


def test_policy_rejects_voice_authority_invariant_change(tmp_path):
    root = synthetic_repository(tmp_path)
    path = root / "policies" / "cognitive_kernel_interaction_authority_policy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["invariants"]["voice_alone_privileged_authority_forbidden"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CognitiveKernelContractError):
        load_cognitive_kernel_interaction_authority_policy(repository_root=root)
