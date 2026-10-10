"""Deliver causal originals through the frozen target-free caller interface.

Read adapters own current permission/registration. Digests authenticate no
service, establish no truth and grant no training rights.
"""
from base64 import b64decode, b64encode
from copy import deepcopy
from hashlib import sha256

from cognitive_kernel.formation_semantics_v16 import context_v16_from_record
from . import formation_world_contracts as C, formation_world_replay as replay
from . import formation_caller_input as caller, formation_caller_meaning as meaning


def _boundary(value, view):
    C.exact(value, {'scope','namespace','purpose','read_service_id','as_of','policy_sha256','rights_receipt_refs','lifecycle_receipt_ref','record_sha256'}, 'current read boundary')
    C.require(value['record_sha256']==C.canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'}), 'read boundary seal differs')
    C.require(value['scope']==view['scope'] and value['namespace']==view['namespace'] and
        value['purpose'] in ('memory_formation','formation_training','formation_evaluation'), 'read boundary scope/purpose differs')
    C.identifier(value['read_service_id']); C.timestamp(value['as_of']); C.digest(value['policy_sha256']); C.identifier(value['lifecycle_receipt_ref'])
    C.require(type(value['rights_receipt_refs']) is list and value['rights_receipt_refs'] and
        len(set(value['rights_receipt_refs']))==len(value['rights_receipt_refs']), 'explicit rights references required')
    for item in value['rights_receipt_refs']: C.identifier(item)


def _entry(slot, family, availability='present', body=None):
    raw=None if body is None else C.canonical_json_bytes(body)
    return {'slot':slot,'family':family,'availability':availability,'body_base64':None if raw is None else b64encode(raw).decode('ascii'),
        'body_bytes':None if raw is None else len(raw),'body_sha256':None if raw is None else sha256(raw).hexdigest()}


def world_caller_bundle(view, *, replay_sha256, input_request, input_request_sha256, read_adapter,
                        view_policy, view_policy_sha256, contract_binding, contract_binding_sha256, limits):
    C.binding(contract_binding,contract_binding_sha256)
    C.exact(view, {'schema','profile','scope','namespace','cut','world_prefix_sha256','experience_head','authority_id',
        'authority_generation','authority_source_position','current_claims','registry_metadata','sources','visibility','original_artifacts','replay_sha256'}, 'replay view')
    C.pin(view,replay_sha256,'mfm-world-replay-view-v1','replay_sha256')
    C.exact(input_request, {'schema','cut','context','opened_source_refs','prior_ref_ids','target_descriptions','read_boundary','record_sha256'}, 'independent world input')
    C.pin(input_request,input_request_sha256,'mfm-world-input-request-v1')
    C.require(input_request['cut']==view['cut'] and view['profile']=='supplied-claim-prefix-v1' and
        type(view['authority_generation']) is int and view['authority_generation']==1, 'world input cut/profile differs')
    C.integer(view['authority_source_position']); C.scope(view['scope']); C.identifier(view['namespace'])
    context=context_v16_from_record(input_request['context']); context.validate()
    C.require(context.record()==input_request['context'] and context.base.scope.metadata_record()==view['scope'] and
        context.base.authority_namespace_id==view['namespace'], 'world input context differs')
    for key in ('opened_source_refs','prior_ref_ids'):
        C.require(type(input_request[key]) is list and len(set(input_request[key]))==len(input_request[key]), 'input identity order differs')
        for ref_id in input_request[key]: C.identifier(ref_id)
    C.require(input_request['opened_source_refs']==[r.ref_id for r in context.base.evidence], 'complete explicit source order differs')
    boundary=input_request['read_boundary']; _boundary(boundary,view)
    def current_boundary():
        value=read_adapter.boundary_record(); _boundary(value,view)
        C.require(value==boundary, 'current policy/rights/lifecycle changed')
    current_boundary()
    sources={row['registration']['evidence']['ref_id']:row for row in view['sources']}
    claims={row['ref_id']:row for row in view['current_claims']}; metadata={row['ref_id']:row for row in view['registry_metadata']}
    C.require(len(sources)==len(view['sources']) and len(claims)==len(view['current_claims']) and
        len(metadata)==len(view['registry_metadata']), 'duplicate replay identities')
    opened=[]; source_rows=[]; consumed=0
    def account(ref):
        nonlocal consumed
        C.artifact_ref(ref,limits); C.require(consumed+ref['bytes']<=limits.max_opened_bytes, 'complete delivered bytes exceed explicit limit')
        consumed+=ref['bytes']
    for evidence in context.base.evidence:
        C.require(evidence.ref_id in sources and view['visibility'].get(evidence.ref_id)=='present', 'source outside visible cut')
        row=sources[evidence.ref_id]; reg=C.registration(row['registration'])
        C.require(reg.evidence==evidence, 'context does not bind original observation')
        account(row['source_artifact']); current_boundary(); raw=read_adapter.read_source(reg); current_boundary()
        C.require(type(raw) is bytes and len(raw)==row['source_artifact']['bytes'] and sha256(raw).hexdigest()==row['source_artifact']['file_sha256'], 'delivered source differs')
        opened.append((evidence.ref_id,raw)); source_rows.append({'ref_id':evidence.ref_id,'registration_sha256':reg.registration_sha256,'artifact':row['source_artifact']})
    descriptions=input_request['target_descriptions']; C.require(type(descriptions) is list, 'explicit descriptions required')
    targets={t.ref_id:t for t in context.targets}; desc={}
    for row in descriptions:
        C.exact(row, {'ref_id','kind','description'}, 'registered target description')
        C.require(row['ref_id'] in targets and row['ref_id'] not in desc and row['kind']==targets[row['ref_id']].kind and
            type(row['description']) is str and '\x00' not in row['description'], 'description exceeds supplied target authority')
        desc[row['ref_id']]=row
    prior_rows=[]; prior_bodies={}; prior_read=[]
    for ref_id in input_request['prior_ref_ids']:
        C.require(ref_id in claims or ref_id in metadata, 'prior outside supplied cut')
        status=view['visibility'].get(ref_id); C.require(status in caller.REGISTRY_STATUSES, 'prior lifecycle differs')
        if ref_id in claims:
            row=claims[ref_id]; records=row['records']; C.exact(records, {'identity','version','projection'}, 'original Claim triple')
            refs={k:row['originals'][k]['artifact'] for k in records}
            for key in records:
                original=row['originals'][key]; C.exact(original, {'artifact','base64'}, 'Claim original custody')
                raw=b64decode(original['base64'],validate=True)
                C.require(b64encode(raw).decode('ascii')==original['base64'] and len(raw)==refs[key]['bytes'] and
                    sha256(raw).hexdigest()==refs[key]['file_sha256'] and C.parse_json(raw)==records[key], 'replay Claim bytes differ')
            body=None
            if status=='present':
                for ref in refs.values(): account(ref)
                current_boundary(); received=read_adapter.read_claim_originals(ref_id,deepcopy(refs)); current_boundary()
                C.require(type(received) is tuple and len(received)==3, 'complete original Claim read required')
                for key,raw in zip(('identity','version','projection'),received):
                    C.require(type(raw) is bytes and raw==b64decode(row['originals'][key]['base64'],validate=True), 'current Claim read differs')
                body={'schema':'mfm-claim-snapshot-meaning-v1',**deepcopy(records)}
                prior_read.append({'ref_id':ref_id,'artifacts':refs})
            prior_bodies[ref_id]=body
            prior_rows.append({'ref_id':ref_id,'kind':'claim','record_sha256':records['version']['version_sha256'], 'status':status,'description':''})
        else:
            row=metadata[ref_id]; prior_rows.append({k:deepcopy(row[k]) for k in ('ref_id','kind','record_sha256','description')})
            prior_rows[-1]['status']=status; prior_bodies[ref_id]=None
    current_boundary()
    source_receipt=C.seal({'schema':'mfm-world-source-read-receipt-v1','boundary_sha256':boundary['record_sha256'],'sources':source_rows,
        'supplied_policy_evidence_only':True,'authenticated_service_claimed':False})
    prior_receipt=C.seal({'schema':'mfm-world-prior-read-receipt-v1','boundary_sha256':boundary['record_sha256'],'prior_reads':prior_read,
        'registry_records':prior_rows,'target_descriptions':descriptions,'supplied_policy_evidence_only':True,'authenticated_service_claimed':False})
    read_digest=C.canonical_sha256({'source':source_receipt['record_sha256'],'prior':prior_receipt['record_sha256']})
    generation='world-read-'+C.canonical_sha256({'prefix':view['world_prefix_sha256'],'request':input_request_sha256,'boundary':boundary})[:40]
    registry={'schema':caller.REGISTRY_SCHEMA,'host_scope':view['scope'],'namespace':view['namespace'],'generation':generation,'records':prior_rows,
        'records_sha256':C.canonical_sha256(prior_rows)}
    registry['registry_sha256']=C.canonical_sha256(registry)
    input_record=caller.caller_input_record(context,opened,registry)
    snapshot=C.seal({'schema':meaning.SNAPSHOT_SCHEMA,'scope':view['scope'],'namespace':view['namespace'],'generation':generation,
        'authority_generation':view['authority_generation'],'as_of':boundary['as_of'],'source_position':view['authority_source_position'],
        'read_service_id':boundary['read_service_id'],'read_receipt_sha256':read_digest,'policy_sha256':boundary['policy_sha256']})
    entries=[]
    for slot,candidate in enumerate(caller.selection_candidates(context,opened,registry)):
        origin=candidate['origin']; ref_id=candidate['ref_id']
        if origin=='opened_source': entry=_entry(slot,'opened-source-v1')
        elif origin=='source_subject': entry=_entry(slot,'source-subject-v1','metadata_only')
        elif origin=='context_target':
            if ref_id in desc:
                body={'schema':'mfm-registered-target-description-v1','scope':view['scope'],'namespace':view['namespace'],
                    **desc[ref_id],'read_receipt_sha256':read_digest}
                entry=_entry(slot,'registered-description-v1',body=body)
            else: entry=_entry(slot,'absent-body-v1','unavailable')
        elif prior_bodies[ref_id] is not None: entry=_entry(slot,'claim-snapshot-v1',body=prior_bodies[ref_id])
        else: entry=_entry(slot,'absent-body-v1','metadata_only' if view['visibility'][ref_id]=='present' else view['visibility'][ref_id])
        entries.append(entry)
    companion=meaning.caller_meaning_record(input_record,caller_sha256=input_record['record_sha256'],snapshot=snapshot,
        snapshot_sha256=snapshot['record_sha256'],candidate_meanings=entries,view_policy=view_policy,view_policy_sha256=view_policy_sha256)
    current_boundary()
    return C.seal({'schema':'mfm-world-caller-bundle-v1','cut':view['cut'],'replay_sha256':replay_sha256,
        'input_request_sha256':input_request_sha256,'original_artifact_pins':view['original_artifacts'],
        'source_read_receipt':source_receipt,'prior_read_receipt':prior_receipt,'caller':input_record,'meaning':companion}, 'bundle_sha256')


def verify_world_caller_bundle(bundle, *, bundle_sha256, world, world_sha256, cut, cut_sha256, input_request,
    input_request_sha256, read_artifact, read_adapter, limits, contract_binding, contract_binding_sha256, view_policy, view_policy_sha256):
    C.pin(bundle,bundle_sha256,'mfm-world-caller-bundle-v1','bundle_sha256')
    view=replay.replay_world_prefix(world,world_sha256=world_sha256,cut=cut,cut_sha256=cut_sha256,
        read_artifact=read_artifact,limits=limits,contract_binding=contract_binding,contract_binding_sha256=contract_binding_sha256)
    rebuilt=world_caller_bundle(view,replay_sha256=view['replay_sha256'],input_request=input_request,
        input_request_sha256=input_request_sha256,read_adapter=read_adapter,limits=limits,contract_binding=contract_binding,
        contract_binding_sha256=contract_binding_sha256,view_policy=view_policy,view_policy_sha256=view_policy_sha256)
    C.require(bundle==rebuilt, 'complete world caller reload differs')
    return rebuilt
