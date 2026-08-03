import json

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    load_cognitive_kernel_raw_buffer_policy,
)


def test_raw_buffer_policy_loads_from_repository():
    policy = load_cognitive_kernel_raw_buffer_policy()
    assert policy.version == "0.7.0"
    assert policy.milestone == "P5.1b"
    assert policy.invariants["host_sealed_opaque_payloads"] is True
    assert policy.invariants["automatic_expiry_implemented"] is False


def test_raw_buffer_policy_rejects_key_custody(tmp_path):
    source = load_cognitive_kernel_raw_buffer_policy().source_path
    value = json.loads(source.read_text(encoding="utf-8"))
    value["invariants"]["kernel_key_custody"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CognitiveKernelContractError):
        load_cognitive_kernel_raw_buffer_policy(path)
