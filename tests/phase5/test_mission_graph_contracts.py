from dataclasses import replace
import pytest
from cognitive_kernel import CognitiveKernelContractError, Mission, MissionNode
from mission_graph_helpers import scope, provenance, digest

def mission(product="alice", host="host-a"):
    return Mission.create(mission_key="mission-1", scope=scope(product, host), created_at="2026-08-01T09:00:00Z", title_digest=digest("mission"), provenance=provenance(), policy_bindings=("mission-graph-policy",))

def node(m, key="root", node_type="mission", status="active", execution="running", updated="2026-08-01T09:00:00Z"):
    return MissionNode.create(node_key=key, mission_id=m.mission_id, scope=m.scope, node_type=node_type, status=status, execution_state=execution, visibility_state="foreground", created_at="2026-08-01T09:00:00Z", updated_at=updated, title_digest=digest(key), provenance=provenance())

def test_mission_and_node_are_tamper_evident():
    m=mission(); n=node(m)
    assert m.mission_id.startswith("mission-")
    assert n.node_id.startswith("mission-node-")
    with pytest.raises(CognitiveKernelContractError):
        replace(n, title_digest="0"*64).validate()

def test_node_identity_survives_valid_reopen_successor():
    m=mission(); completed=node(m, key="task", node_type="task", status="completed", execution="succeeded")
    reopened=MissionNode.create(node_key="task", mission_id=m.mission_id, scope=m.scope, node_type="task", status="active", execution_state="running", visibility_state="foreground", created_at=completed.created_at, updated_at="2026-08-01T10:00:00Z", title_digest=digest("task"), provenance=provenance())
    assert reopened.node_id == completed.node_id
    completed.assert_valid_successor(reopened)

def test_invalid_status_execution_pair_is_rejected():
    m=mission()
    with pytest.raises(CognitiveKernelContractError):
        node(m, status="completed", execution="running")
