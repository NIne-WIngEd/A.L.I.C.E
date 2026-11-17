"""Actual optimized native decision -> existing durable Experience metadata."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json

import pytest
import torch

from scripts.eipm.gemma_n0.governed_response import respond_with_governance
from src.alice_personality.identity.codec import canonical
from src.alice_personality.identity.consumer import RenderedResponse
from cognitive_kernel import (ProductHostScope, open_experience_ledger,
                              DuplicateExperienceEventError, ExperienceLedgerIsolationError)
from tests.personality import test_identity_consumer as fixtures


@pytest.fixture(scope='module')
def decisions():
    previous=torch.get_num_threads()
    fixture=fixtures.ConsumerTests();fixture.setUp()
    try:
        consumer=fixture.build(co_valid=True)
        natural=respond_with_governance(consumer,fixture.doc,fixture.produced,fixture.session)
        unresolved=respond_with_governance(consumer,fixture.doc,fixture.produced,fixture.session,
            renderer=lambda envelope:RenderedResponse('Different public words.',{},'public-renderer',None))
        assert natural['status']=='SELECTED' and unresolved['status']=='UNRESOLVED_RENDERER'
        yield natural,unresolved
    finally:
        fixture.doCleanups();torch.set_num_threads(previous)


def api():
    name='cognitive_kernel.personality_experience'
    assert importlib.util.find_spec(name) is not None,'Missing native decision-to-Experience connection'
    return importlib.import_module(name)


@pytest.fixture(autouse=True)
def existing_repository_root(tmp_path):
    (tmp_path/'repository').mkdir()


def scope(host='public-owner',version='1.0.0'):
    return ProductHostScope.create(product_id='alice',host_instance_id=host,
        schema_version=version,encryption_domain='public-fixture-domain')


def append(ledger,payload,**options):
    arguments=dict(payload=payload,expected_payload_sha256=sha256(payload).hexdigest(),
        payload_reference='public-payload-1',scope=scope(),invocation_id='public-invocation-1',
        occurred_at='2026-10-25T06:20:00Z',committed_at='2026-10-25T06:20:01Z',
        policy_bindings=('public-fixture-policy',),deletion_lineage=('public-source-1',))
    arguments.update(options)
    return api().record_personality_decision(ledger,**arguments)


def test_actual_native_packet_records_exact_lineage_and_reopens_without_payload_leak(decisions,tmp_path):
    response=decisions[0];payload=canonical(response)+b'\n'
    vault=tmp_path/'vault';vault.mkdir();artifact=vault/'response.json';artifact.write_bytes(payload)
    db=vault/'experience.sqlite3'
    with open_experience_ledger(db,scope=scope(),repository_root=tmp_path/'repository',
                               created_at='2026-10-25T06:19:00Z') as ledger:
        receipt=append(ledger,artifact.read_bytes())
        event=ledger.load_event(receipt.entries[0].event_id)
        assert event.content_digest==sha256(artifact.read_bytes()).hexdigest()
        assert event.event_type=='personality.development_decision'
        assert event.provenance.provenance_type=='derived_inference'
        assert event.provenance.confidence is None
        assert event.provenance.derivation_activity_id=='public-invocation-1'
        assert event.provenance.model_id=='personality-model:'+response['preference_selection']['model_state_sha256']
        assert event.payload_reference=='public-payload-1'
        assert event.deletion_lineage==('public-source-1',)
        assert event.outcome_reference_ids==()
        refs=event.provenance.source_reference_ids
        for field in ('document_sha256','frame_sha256','feature_binding_sha256','closed_session_sha256','packet_sha256'):
            assert 'personality-'+field.replace('_sha256','').replace('_','-')+':'+response['preference_selection'][field] in refs
        for pin in response['preference_selection']['checkpoint_manifest_sha256']:
            assert 'personality-checkpoint:'+pin in refs
        assert 'personality-response:'+response['receipt_sha256'] in refs
        metadata=json.dumps(event.metadata_record())
        assert response['response_text'] not in metadata and 'native_latent' not in metadata
        with pytest.raises(DuplicateExperienceEventError):append(ledger,payload)
        assert ledger.verify_integrity().entry_count==1
    with open_experience_ledger(db,scope=scope(),repository_root=tmp_path/'repository') as ledger:
        assert ledger.load_event(event.event_id)==event
        assert ledger.verify_integrity().valid
    assert response['response_text'].encode() not in db.read_bytes()
    assert artifact.read_bytes()==payload and canonical(response)+b'\n'==payload


def test_unresolved_realization_is_recorded_without_becoming_an_outcome(decisions,tmp_path):
    response=decisions[1];payload=canonical(response)
    with open_experience_ledger(tmp_path/'vault/ledger.sqlite3',scope=scope(),repository_root=tmp_path/'repository',
                               created_at='2026-10-25T06:19:00Z') as ledger:
        receipt=append(ledger,payload)
        event=ledger.load_event(receipt.entries[0].event_id)
        assert event.content_digest==sha256(payload).hexdigest()
        assert not event.outcome_reference_ids and not event.provenance.supersedes_record_ids
        assert response['status']=='UNRESOLVED_RENDERER'
        assert response['rendering']['text']=='Different public words.'
        assert response['rendering']['text']!=response['response_text']
        assert response['acceptance_authority'] is response['owner_fidelity_approved'] is False


def test_changed_payload_inner_seal_or_authority_cannot_write(decisions,tmp_path):
    response=decisions[0];payload=canonical(response)
    with open_experience_ledger(tmp_path/'vault/ledger.sqlite3',scope=scope(),repository_root=tmp_path/'repository',
                               created_at='2026-10-25T06:19:00Z') as ledger:
        api()
        with pytest.raises(ValueError,match='payload'):
            append(ledger,payload+b' ',expected_payload_sha256=sha256(payload).hexdigest())
        changed=deepcopy(response);changed['preference_selection']['model_state_sha256']='a'*64
        changed['receipt_sha256']=sha256(canonical({k:v for k,v in changed.items() if k!='receipt_sha256'})).hexdigest()
        with pytest.raises(ValueError,match='seal'):append(ledger,canonical(changed))
        changed=deepcopy(response);changed['owner_fidelity_approved']=True
        changed['receipt_sha256']=sha256(canonical({k:v for k,v in changed.items() if k!='receipt_sha256'})).hexdigest()
        with pytest.raises(ValueError,match='authority'):append(ledger,canonical(changed))
        assert ledger.verify_integrity().entry_count==0


def test_source_scope_must_match_destination_and_separate_invocations_remain_separate(decisions,tmp_path):
    payload=canonical(decisions[0])
    with open_experience_ledger(tmp_path/'vault/ledger.sqlite3',scope=scope(),repository_root=tmp_path/'repository',
                               created_at='2026-10-25T06:19:00Z') as ledger:
        api()
        with pytest.raises(ExperienceLedgerIsolationError,match='scope'):append(ledger,payload,scope=scope('different-owner'))
        one=append(ledger,payload)
        two=append(ledger,payload,invocation_id='public-invocation-2')
        assert one.entries[0].event_id!=two.entries[0].event_id
        three=append(ledger,payload,invocation_id='public-invocation-3',scope=scope(version='1.1.0'))
        assert three.entries[0].sequence==3
        assert ledger.verify_integrity().entry_count==3
        third_event=ledger.load_event(three.entries[0].event_id)
        original_identity=ledger.ledger_id
    with open_experience_ledger(tmp_path/'vault/ledger.sqlite3',scope=scope(version='2.0.0'),
                               repository_root=tmp_path/'repository') as ledger:
        assert ledger.ledger_id==original_identity
        assert ledger.load_event(third_event.event_id)==third_event
        assert third_event.scope.schema_version=='1.1.0'
        assert ledger.verify_integrity().entry_count==3
