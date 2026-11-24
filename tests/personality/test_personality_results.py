"""Actual native decision to durable result linkage, with no invented learning credit."""
from dataclasses import replace
from hashlib import sha256
import importlib
import importlib.util
import json
import pytest

from cognitive_kernel import (ResultCapsule, ProvenanceReference, open_experience_ledger,
                              DuplicateExperienceEventError)
from src.alice_personality.identity.codec import canonical
from cognitive_kernel.canonical import canonical_sha256
from tests.personality.test_personality_experience import decisions
from tests.personality import test_personality_experience as source


def api():
    name='cognitive_kernel.personality_results'
    assert importlib.util.find_spec(name) is not None, 'Missing personality result-to-Experience connection'
    return importlib.import_module(name)


@pytest.fixture
def recorded(decisions,tmp_path):
    repository=tmp_path/'repository';repository.mkdir()
    path=tmp_path/'vault/ledger.sqlite3'
    with open_experience_ledger(path,scope=source.scope(),repository_root=repository,
                               created_at='2026-10-25T06:19:00Z') as ledger:
        receipt=source.append(ledger,canonical(decisions[0]))
        event=ledger.load_event(receipt.entries[0].event_id)
        yield ledger,event,path,repository


def capsule(decision,**changes):
    args=dict(result_key='public-result',scope=source.scope(version='1.1.0'),mission_id='public-mission',
        node_id='public-node',produced_at='2026-10-25T06:21:00Z',status='partial',
        summary_digest=sha256(b'Observed mixed result, not a reward.').hexdigest(),
        output_reference_ids=('public-output',),evidence_reference_ids=('public-observation',),
        source_event_ids=(decision.event_id,),deletion_lineage=('public-observation-source',),
        policy_bindings=('public-result-policy',),
        provenance=ProvenanceReference.create(provenance_type='conflict_or_uncertain',
            source_reference_ids=('public-observation',),derivation_activity_id='public-result-review',
            responsible_component='public-result-owner',confidence=.35))
    args.update(changes)
    return ResultCapsule.create(**args)


def record(ledger,decision,result,**changes):
    payload=canonical(result.metadata_record())+b'\n'
    args=dict(decision_event_id=decision.event_id,expected_decision_event_sha256=decision.event_sha256,
        capsule=result,payload=payload,expected_payload_sha256=sha256(payload).hexdigest(),
        payload_reference='public-saved-result',committed_at='2026-10-25T06:22:00Z')
    args.update(changes)
    return api().record_personality_result(ledger,**args)


@pytest.mark.parametrize('status',['succeeded','partial','failed','cancelled','blocked'])
def test_reported_result_reopens_with_exact_original_and_complete_lineage(recorded,status):
    ledger,decision,path,repository=recorded
    result=capsule(decision,status=status)
    receipt=record(ledger,decision,result)
    linked=ledger.load_event(receipt.entries[0].event_id)
    assert linked.event_type=='personality.reported_result'
    assert linked.parent_event_ids==(decision.event_id,)
    assert linked.content_digest==sha256(canonical(result.metadata_record())+b'\n').hexdigest()
    assert linked.outcome_reference_ids==(result.capsule_id,)
    assert linked.provenance.provenance_type=='derived_inference'
    assert linked.provenance.model_id is linked.provenance.confidence is None
    assert 'personality-decision:'+decision.event_sha256 in linked.provenance.source_reference_ids
    assert 'result-capsule:'+result.capsule_sha256 in linked.provenance.source_reference_ids
    assert set(result.evidence_reference_ids+result.output_reference_ids)<=set(linked.provenance.source_reference_ids)
    assert set(linked.deletion_lineage)==set(decision.deletion_lineage+result.deletion_lineage)
    assert set(linked.policy_bindings)==set(decision.policy_bindings+result.policy_bindings)
    assert ledger.load_event(decision.event_id)==decision
    assert not decision.outcome_reference_ids
    assert ledger.verify_integrity().entry_count==2
    with pytest.raises(DuplicateExperienceEventError): record(ledger,decision,result,committed_at='2026-10-25T06:23:00Z')
    with open_experience_ledger(path,scope=source.scope(version='2.0.0'),repository_root=repository) as reopened:
        assert reopened.load_event(linked.event_id)==linked
        assert reopened.load_event(decision.event_id)==decision
        assert reopened.verify_integrity().valid
    assert 'Observed mixed result' not in json.dumps(linked.metadata_record())
    assert result.provenance.provenance_type=='conflict_or_uncertain' and result.status==status
    assert result.provenance.confidence==.35  # Fictional reporter estimate, never a teaching target.


@pytest.mark.parametrize('changed',['payload','capsule','decision_pin','missing_parent','wrong_kind','scope','source_link','time','deleted','quarantine'])
def test_invalid_association_cannot_append(recorded,changed):
    ledger,decision,_,_=recorded;result=capsule(decision);options={}
    if changed=='payload': options['payload']=canonical(result.metadata_record())+b'  '
    elif changed=='capsule': result=replace(result,status='succeeded')
    elif changed=='decision_pin': options['expected_decision_event_sha256']='a'*64
    elif changed=='missing_parent': options['decision_event_id']='missing-event'
    elif changed=='wrong_kind':
        linked=ledger.load_event(record(ledger,decision,result).entries[0].event_id)
        decision=linked;result=capsule(decision,result_key='second')
    elif changed=='scope': result=capsule(decision,scope=source.scope('another-owner'))
    elif changed=='source_link': result=capsule(decision,source_event_ids=('another-event',))
    elif changed=='time': result=capsule(decision,produced_at='2026-10-25T06:18:00Z')
    elif changed=='deleted': result=capsule(decision,storage_tier='deleted',output_reference_ids=())
    elif changed=='quarantine': result=capsule(decision,retention_class='quarantine')
    before=ledger.verify_integrity().entry_count
    with pytest.raises((ValueError,KeyError)): record(ledger,decision,result,**options)
    assert ledger.verify_integrity().entry_count==before


def test_exact_saved_payload_may_use_different_json_whitespace(recorded):
    ledger,decision,_,_=recorded;result=capsule(decision)
    payload=json.dumps(result.metadata_record(),indent=2).encode()
    record(ledger,decision,result,payload=payload,expected_payload_sha256=sha256(payload).hexdigest())
    assert ledger.verify_integrity().entry_count==2


@pytest.mark.parametrize('timestamp,allowed',[
    ('2026-10-25T05:21:00-01:00',True),  # 06:21UTC, after decision.
    ('2026-10-25T07:19:00+01:00',False), # 06:19UTC, before decision.
])
def test_valid_capsule_offset_timestamp_is_compared_as_an_instant(recorded,timestamp,allowed):
    ledger,decision,_,_=recorded
    # The existing validated ResultCapsule contract admits explicit offsets.
    result=replace(capsule(decision),produced_at=timestamp)
    result=replace(result,capsule_id='result-capsule-'+canonical_sha256(result.identity_record())[:32])
    result=replace(result,capsule_sha256=canonical_sha256(result.material_record()))
    result.validate()
    if allowed:
        event=ledger.load_event(record(ledger,decision,result).entries[0].event_id)
        assert event.occurred_at=='2026-10-25T06:21:00.000000Z'
    else:
        with pytest.raises(ValueError,match='precedes'):record(ledger,decision,result)
        assert ledger.verify_integrity().entry_count==1


@pytest.mark.parametrize('tier',['DELETED',' QUARANTINE '])
def test_valid_capsule_inactive_tier_cannot_be_reclassified_as_active(recorded,tier):
    ledger,decision,_,_=recorded
    result=replace(capsule(decision,output_reference_ids=()),storage_tier=tier)
    result=replace(result,capsule_sha256=canonical_sha256(result.material_record()))
    result.validate()
    with pytest.raises(ValueError,match='inactive'):record(ledger,decision,result)
    assert ledger.verify_integrity().entry_count==1
