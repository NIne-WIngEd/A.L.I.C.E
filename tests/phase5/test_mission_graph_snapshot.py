import pytest
from cognitive_kernel import CognitiveKernelContractError, Mission, MissionEdge, MissionGraphSnapshot, MissionNode
from mission_graph_helpers import scope, provenance, digest

def make_graph(product="alice", host="host-a"):
    s=scope(product,host); p=provenance()
    m=Mission.create(mission_key="m1", scope=s, created_at="2026-08-01T09:00:00Z", title_digest=digest("m"), provenance=p)
    root=MissionNode.create(node_key="root", mission_id=m.mission_id, scope=s, node_type="mission", status="active", execution_state="running", visibility_state="foreground", created_at="2026-08-01T09:00:00Z", updated_at="2026-08-01T09:00:00Z", title_digest=digest("root"), provenance=p)
    child=MissionNode.create(node_key="child", mission_id=m.mission_id, scope=s, node_type="task", status="ready", execution_state="queued", visibility_state="supporting", created_at="2026-08-01T09:01:00Z", updated_at="2026-08-01T09:01:00Z", title_digest=digest("child"), provenance=p)
    edge=MissionEdge.create(edge_key="root-child", mission_id=m.mission_id, scope=s, source_node_id=root.node_id, target_node_id=child.node_id, edge_type="parent_child", created_at="2026-08-01T09:01:00Z", provenance=p)
    graph=MissionGraphSnapshot.create(mission=m, root_node_id=root.node_id, nodes=(root,child), edges=(edge,), revision=1, snapshot_at="2026-08-01T09:02:00Z")
    return m,root,child,edge,graph

def test_rooted_graph_is_deterministic():
    *_, graph=make_graph()
    assert graph.snapshot_id.startswith("mission-snapshot-")
    assert len(graph.snapshot_sha256)==64

def test_detached_node_is_rejected():
    m,root,child,edge,_=make_graph()
    extra=MissionNode.create(node_key="extra", mission_id=m.mission_id, scope=m.scope, node_type="task", status="planned", execution_state="idle", visibility_state="background", created_at="2026-08-01T09:03:00Z", updated_at="2026-08-01T09:03:00Z", title_digest=digest("extra"), provenance=provenance())
    with pytest.raises(CognitiveKernelContractError):
        MissionGraphSnapshot.create(mission=m, root_node_id=root.node_id, nodes=(root,child,extra), edges=(edge,), revision=2, snapshot_at="2026-08-01T09:04:00Z")

def test_cross_host_node_is_rejected():
    m,root,child,edge,_=make_graph()
    other_scope=scope("alice","host-b")
    foreign=MissionNode.create(node_key="foreign", mission_id=m.mission_id, scope=other_scope, node_type="task", status="planned", execution_state="idle", visibility_state="background", created_at="2026-08-01T09:03:00Z", updated_at="2026-08-01T09:03:00Z", title_digest=digest("foreign"), provenance=provenance())
    with pytest.raises(CognitiveKernelContractError):
        MissionGraphSnapshot.create(mission=m, root_node_id=root.node_id, nodes=(root,child,foreign), edges=(edge,), revision=2, snapshot_at="2026-08-01T09:04:00Z")
