import pytest

from cognitive_kernel import CognitiveKernelContractError, WorkspaceProjection
from attention_workspace_helpers import decision, entry, item, layout, scope


def test_attention_identity_and_projection_are_host_local():
    alice_entry = entry(key="same", reference="node-1", product="alice", host="host-a")
    friday_entry = entry(key="same", reference="node-1", product="friday", host="host-b")
    alice_decision = decision((alice_entry,), product="alice", host="host-a", limit=1)
    friday_decision = decision((friday_entry,), product="friday", host="host-b", limit=1)
    assert alice_decision.decision_id != friday_decision.decision_id

    friday_item = item(
        friday_entry,
        product="friday",
        host="host-b",
        role="primary",
    )
    friday_layout = layout(
        (friday_item,),
        product="friday",
        host="host-b",
    )
    with pytest.raises(CognitiveKernelContractError):
        WorkspaceProjection.create(
            projection_key="cross-host",
            scope=scope("alice", "host-a"),
            attention_decision=alice_decision,
            projected_at="2026-08-01T10:12:00Z",
            audience="host",
            layout=friday_layout,
            items=(friday_item,),
        )
