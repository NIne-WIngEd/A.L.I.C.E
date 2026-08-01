import pytest

from cognitive_kernel import CognitiveKernelContractError, HostWorkspaceOverride
from attention_workspace_helpers import digest, provenance, scope


def override(command, **kwargs):
    return HostWorkspaceOverride.create(
        override_key=f"override-{command}",
        scope=scope(),
        command=command,
        status="applied",
        issued_at="2026-08-01T10:00:00Z",
        reason_digest=digest(command),
        provenance=provenance(),
        **kwargs,
    )


def test_target_and_setting_overrides_are_contractible():
    assert override("pin", target_reference_id="node-1").command == "pin"
    assert override("set_visibility_limit", numeric_value=3).numeric_value == 3
    assert (
        override(
            "set_interruption_preference",
            setting_value="focus_only",
        ).setting_value
        == "focus_only"
    )
    assert override("restore_automatic_layout").target_reference_id is None


def test_override_command_argument_shapes_are_enforced():
    with pytest.raises(CognitiveKernelContractError):
        override("pin")
    with pytest.raises(CognitiveKernelContractError):
        override("set_visibility_limit", numeric_value=11)
    with pytest.raises(CognitiveKernelContractError):
        override(
            "set_interruption_preference",
            setting_value="maximize_engagement",
        )
    with pytest.raises(CognitiveKernelContractError):
        override("lock_layout", target_reference_id="node-1")


def test_expired_override_requires_a_future_expiry():
    with pytest.raises(CognitiveKernelContractError):
        HostWorkspaceOverride.create(
            override_key="expired",
            scope=scope(),
            command="lock_layout",
            status="expired",
            issued_at="2026-08-01T10:00:00Z",
            reason_digest=digest("expired"),
            provenance=provenance(),
        )
    receipt = HostWorkspaceOverride.create(
        override_key="expires",
        scope=scope(),
        command="lock_layout",
        status="expired",
        issued_at="2026-08-01T10:00:00Z",
        expires_at="2026-08-01T11:00:00Z",
        reason_digest=digest("expires"),
        provenance=provenance(),
    )
    assert receipt.status == "expired"
