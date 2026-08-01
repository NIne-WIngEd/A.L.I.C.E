import pytest
from cognitive_kernel import CognitiveKernelContractError, ResultCapsule, TracebackChain, TracebackTransition
from mission_graph_helpers import scope, provenance, digest

def capsule(product="alice",host="host-a"):
    return ResultCapsule.create(result_key="r1",scope=scope(product,host),mission_id="mission-1",node_id="node-1",produced_at="2026-08-01T09:00:00Z",status="succeeded",summary_digest=digest("summary"),output_reference_ids=("output-1",),evidence_reference_ids=("evidence-1",),source_event_ids=("experience-1",),provenance=provenance())

def transition(c,key,seq,source,target,action):
    return TracebackTransition.create(transition_key=key,scope=c.scope,capsule_id=c.capsule_id,mission_id=c.mission_id,sequence=seq,source_node_id=source,target_node_id=target,action=action,created_at=f"2026-08-01T09:0{seq}:00Z",rationale_digest=digest(key),evidence_reference_ids=("evidence-1",))

def test_result_capsule_is_metadata_only_and_tamper_evident():
    c=capsule(); record=c.metadata_record()
    assert "summary" not in record and "output" not in record
    assert record["output_reference_ids"]==["output-1"]

def test_success_requires_output_reference():
    with pytest.raises(CognitiveKernelContractError):
        ResultCapsule.create(result_key="bad",scope=scope(),mission_id="m1",node_id="n1",produced_at="2026-08-01T09:00:00Z",status="succeeded",summary_digest=digest("summary"),evidence_reference_ids=("e1",),provenance=provenance())

def test_ordered_traceback_chain_links_to_root():
    c=capsule()
    t0=transition(c,"t0",0,"node-1","parent-1","propagate_to_parent")
    t1=transition(c,"t1",1,"parent-1",None,"stop_at_mission_root")
    chain=TracebackChain.create(chain_key="chain-1",scope=c.scope,capsule_id=c.capsule_id,mission_id=c.mission_id,created_at="2026-08-01T09:02:00Z",status="applied",transitions=(t0,t1))
    assert chain.transitions[-1].action=="stop_at_mission_root"

def test_traceback_chain_rejects_cross_host_transition():
    c=capsule(); other=capsule("friday","host-b")
    t=transition(other,"t0",0,"node-1",None,"stop_at_mission_root")
    with pytest.raises(CognitiveKernelContractError):
        TracebackChain.create(chain_key="bad",scope=c.scope,capsule_id=c.capsule_id,mission_id=c.mission_id,created_at="2026-08-01T09:02:00Z",status="planned",transitions=(t,))
