import pytest
from cognitive_kernel import CognitiveKernelContractError, RoutingDecision
from mission_graph_helpers import scope, provenance, digest

def test_all_six_routing_actions_are_contractible():
    base=dict(scope=scope(), decided_at="2026-08-01T09:00:00Z", rationale_digest=digest("why"), confidence=.7, provenance=provenance())
    cases=[
      dict(decision_key="d1",action="continue_current",current_mission_id="m1",current_node_id="n1"),
      dict(decision_key="d2",action="create_child",current_mission_id="m1",current_node_id="n1",target_mission_id="m1",target_node_id="n1"),
      dict(decision_key="d3",action="create_sibling",current_mission_id="m1",current_node_id="n1",target_mission_id="m1",target_node_id="parent1"),
      dict(decision_key="d4",action="reattach",current_mission_id="m1",current_node_id="n1",target_mission_id="m2",target_node_id="n2"),
      dict(decision_key="d5",action="create_mission",current_mission_id="m1",current_node_id="n1"),
      dict(decision_key="d6",action="control_command",current_mission_id="m1",current_node_id="n1",control_command_id="workspace.pin"),
    ]
    decisions=[RoutingDecision.create(**base,**case) for case in cases]
    assert {d.action for d in decisions}=={"continue_current","create_child","create_sibling","reattach","create_mission","control_command"}

def test_reattach_cannot_target_current_node():
    with pytest.raises(CognitiveKernelContractError):
        RoutingDecision.create(decision_key="bad",scope=scope(),action="reattach",decided_at="2026-08-01T09:00:00Z",current_mission_id="m1",current_node_id="n1",target_mission_id="m1",target_node_id="n1",rationale_digest=digest("why"),confidence=.5,provenance=provenance())
