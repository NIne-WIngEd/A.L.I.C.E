import json
from pathlib import Path
import pytest
from cognitive_kernel import CognitiveKernelContractError, load_cognitive_kernel_mission_graph_policy
ROOT=Path(__file__).resolve().parents[2]

def test_policy_binds_capabilities_and_non_runtime_boundaries():
    policy=load_cognitive_kernel_mission_graph_policy(repository_root=ROOT)
    assert policy.version=="0.2.0"
    assert policy.required_capabilities==("mission_graph.v1","semantic_router.v1","result_capsule.v1","traceback_engine.v1")
    assert policy.invariants["immutable_node_identity"] is True
    assert policy.invariants["persistence_implemented"] is False
    assert policy.capability_ceiling is False

def test_policy_rejects_removed_routing_action(tmp_path):
    source=ROOT/"policies"/"cognitive_kernel_mission_graph_policy.json"
    payload=json.loads(source.read_text(encoding="utf-8"))
    payload["allowed_routing_actions"].remove("reattach")
    mutated=tmp_path/"mutated.json"; mutated.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(CognitiveKernelContractError):
        load_cognitive_kernel_mission_graph_policy(mutated,repository_root=ROOT)
