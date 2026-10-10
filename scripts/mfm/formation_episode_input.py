"""Strict original Episode parts; supplied custody is not service authority."""
from base64 import b64encode
from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

from cognitive_kernel import projection_contracts as P,projection_prototype as Q
from . import formation_world_contracts as C,formation_caller_meaning as M

PARTS=('episode','full_content','summary_content','append_receipt','append_invocation','projection_profile')


def episode_input_binding():
    return C.seal({'schema':'mfm-episode-input-binding-v1','profile':'supplied-claim-episode-prefix-v1',
        'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'projection_contracts_sha256':sha256(Path(P.__file__).read_bytes()).hexdigest(),
        'projection_prototype_sha256':sha256(Path(Q.__file__).read_bytes()).hexdigest(),
        'original_parser_sha256':sha256(Path(C.__file__).read_bytes()).hexdigest(),
        'memory_parser_sha256':sha256(Path(M.__file__).read_bytes()).hexdigest(),
        'projection_schema_version':P.PROJECTION_CONTRACT_SCHEMA_VERSION,'authority_granted':False,'authenticated_service_claimed':False})


def _ref(ref):
    C.exact(ref,{'artifact_id','bytes','file_sha256','media_type'},'Episode artifact')
    C.identifier(ref['artifact_id']);C.integer(ref['bytes'],1);C.digest(ref['file_sha256'])
    C.require(ref['media_type'] in ('application/json','application/octet-stream'),'Episode media type differs')


def _original(raw,ref):
    _ref(ref)
    C.require(type(raw) is bytes and len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['file_sha256'],'Episode original bytes differ')
    value=C.parse_json(raw)
    C.require(type(value) is dict,'Episode original must be an object')
    return value


def _episode(value):
    keys={f.name for f in fields(P.EpisodeRecord)}|{'schema_version'}
    C.exact(value,keys,'original Episode')
    C.require(value['schema_version']==P.PROJECTION_CONTRACT_SCHEMA_VERSION,'Episode schema differs')
    args=deepcopy(value);args.pop('schema_version');args['envelope']=M._envelope(args['envelope'])
    for key in ('member_evidence_ids','member_claim_version_ids','member_candidate_ids','mission_node_ids','participant_ids'):
        C.require(type(args[key]) is list,'Episode member array differs');args[key]=tuple(args[key])
    result=P.EpisodeRecord(**args);result.validate()
    C.require(result.metadata_record()==value,'Episode metadata normalizes')
    return result


def _profile(value):
    C.exact(value,{f.name for f in fields(Q.ProjectionPrototypeProfile)}|{'schema_version'},'original Episode profile')
    C.require(value['schema_version']==Q.PROJECTION_PROTOTYPE_SCHEMA_VERSION,'Episode profile schema differs')
    args=deepcopy(value);args.pop('schema_version');args['scope']=C.scope(args['scope'])
    result=Q.ProjectionPrototypeProfile(**args);result.validate()
    C.require(result.metadata_record()==value,'Episode profile normalizes')
    return result


def episode_transition_metadata(metadata_bytes,receipt_bytes,invocation_bytes,profile_bytes,*,refs,scope,namespace,contract_binding,contract_binding_sha256):
    C.pin(contract_binding,contract_binding_sha256,'mfm-episode-input-binding-v1')
    C.require(contract_binding==episode_input_binding(),'normative Episode binding differs')
    C.exact(refs,set(PARTS),'complete Episode artifact inventory')
    for ref in refs.values():_ref(ref)
    metadata=_original(metadata_bytes,refs['episode']);episode=_episode(metadata)
    profile_value=_original(profile_bytes,refs['projection_profile']);profile=_profile(profile_value)
    receipt_value=_original(receipt_bytes,refs['append_receipt'])
    C.exact(receipt_value,{f.name for f in fields(Q.ProjectionAppendReceipt)},'Episode append receipt')
    receipt=Q.ProjectionAppendReceipt(**receipt_value);receipt.validate()
    C.require(receipt.metadata_record()==receipt_value,'Episode receipt normalizes')
    invocation=_original(invocation_bytes,refs['append_invocation'])
    C.exact(invocation,{'schema','episode_sha256','full_content_ref','summary_content_ref','profile_sha256',
        'idempotency_namespace','idempotency_key','request_digest','record_sha256'},'Episode invocation')
    C.pin(invocation,invocation['record_sha256'],'mfm-episode-append-invocation-v1')
    for key in ('idempotency_namespace','idempotency_key'):C.identifier(invocation[key])
    C.digest(invocation['request_digest'])
    C.require(episode.scope==scope and episode.envelope.authority_namespace_id==namespace and
        profile.scope==scope and profile.authority_namespace_id==namespace,'foreign Episode/profile')
    C.require(invocation['episode_sha256']==episode.episode_sha256 and invocation['profile_sha256']==profile.profile_sha256 and
        invocation['full_content_ref']==refs['full_content'] and invocation['summary_content_ref']==refs['summary_content'],
        'Episode invocation custody differs')
    C.require(refs['full_content']['file_sha256']==episode.full_content_digest and refs['summary_content']['file_sha256']==episode.summary_content_digest,
        'Episode declared content digest differs')
    C.require(receipt.operation=='append_episode' and receipt.record_id==episode.episode_id and receipt.generation==episode.generation and
        receipt.content_digest==episode.full_content_digest and receipt.created_at==episode.formed_at and
        receipt.idempotency_namespace==invocation['idempotency_namespace'] and receipt.idempotency_key==invocation['idempotency_key'] and
        receipt.request_digest==invocation['request_digest'],'Episode original receipt/invocation differs')
    return {'records':{'episode':metadata},'receipt':receipt_value,'invocation':invocation,'profile':profile_value,
        'artifacts':deepcopy(refs),'body_bytes_verified':False,'authenticated_service_claimed':False}


def episode_originals(metadata_bytes,full_bytes,summary_bytes,receipt_bytes,invocation_bytes,profile_bytes,**kwargs):
    result=episode_transition_metadata(metadata_bytes,receipt_bytes,invocation_bytes,profile_bytes,**kwargs)
    refs=kwargs['refs'];full=_original(full_bytes,refs['full_content']);summary=_original(summary_bytes,refs['summary_content'])
    result['records'].update(full_content=full,summary_content=summary)
    result['originals']={name:{'artifact':deepcopy(refs[name]),'base64':b64encode(raw).decode('ascii')}
        for name,raw in zip(('episode','full_content','summary_content'),(metadata_bytes,full_bytes,summary_bytes))}
    result['body_bytes_verified']=True
    return result
