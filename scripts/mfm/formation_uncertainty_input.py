"""Existing registered uncertainty originals, never a local output allocation."""
from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from . import formation_task_content_input as T,formation_world_contracts as C


def uncertainty_input_binding():
    return C.seal({'schema':'mfm-uncertainty-input-binding-v1','profile':T.PROFILE,
        'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'task_contract':T.task_content_binding(),
        'authority_granted':False,'authenticated_service_claimed':False,'local_output_allocation':False})


def identity_registration(row,ref_id):
    C.exact(row,{'ref_id','kind','registration_payload_sha256','known_at_sequence','known_at_entry_sha256','world_record_sha256'},'uncertainty identity registration')
    C.require(row['ref_id']==ref_id and row['kind']=='uncertainty','uncertainty identity/kind differs')
    C.identifier(row['ref_id']);C.integer(row['known_at_sequence'],1)
    for key in ('registration_payload_sha256','known_at_entry_sha256','world_record_sha256'):C.digest(row[key])


def uncertainty_original(metadata_bytes,parts,*,refs,scope,namespace,contract_binding,contract_binding_sha256,full=True):
    C.pin(contract_binding,contract_binding_sha256,'mfm-uncertainty-input-binding-v1')
    C.require(contract_binding==uncertainty_input_binding(),'normative uncertainty binding differs')
    record=C.parse_json(T._bytes(metadata_bytes,refs['content-metadata']))
    C.require(T.common(record,scope,namespace)=='uncertainty','not an uncertainty original')
    C.identifier(record['uncertainty_key']);C.require(record['uncertainty_ref']==record['entity_id'],'uncertainty identity differs')
    identity_registration(record['identity_registration'],record['entity_id'])
    C.require(record['state'] in ('open','resolved','reopened','retired'),'uncertainty state differs')
    rows=T.owned_artifacts(record)
    C.require(refs=={'content-metadata':refs['content-metadata'],**{r['part_name']:r['artifact'] for r in rows}},'complete uncertainty inventory differs')
    C.require(type(parts) is dict and set(parts)==({r['part_name'] for r in rows} if full else set()),'complete uncertainty parts differ')
    T._rows(record['alternatives'],{'alternative_id','text','support_refs','counter_refs'},'alternative_id','alternatives')
    C.require(record['state'] not in ('open','reopened') or len(record['alternatives'])>=2,'open uncertainty requires alternatives')
    T._evidence(record['evidence_refs']);evidence_ids={r['ref_id'] for r in record['evidence_refs']}
    for row in record['alternatives']:
        for key in ('support_refs','counter_refs'):
            T.ids(row[key],key);C.require(set(row[key])<=evidence_ids,'alternative evidence missing')
    alternatives={r['alternative_id'] for r in record['alternatives']}
    T.array(record['resolution_observations'],'resolution observations')
    C.require(record['state']!='resolved' or bool(record['resolution_observations']),'resolved observation missing')
    observations=set()
    for row in record['resolution_observations']:
        C.exact(row,{'observation_ref','selected_alternative_ids','report','source_event_refs'},'resolution observation')
        T.evidence(row['observation_ref']);C.require(row['observation_ref']['availability']=='present','resolution observation original unavailable')
        ref_id=row['observation_ref']['ref_id']
        C.require(ref_id not in observations,'resolution reuses an authored observation');observations.add(ref_id)
        T.ids(row['selected_alternative_ids'],'selected alternatives');C.require(bool(row['selected_alternative_ids']) and set(row['selected_alternative_ids'])<=alternatives,'resolution alternatives missing')
        T._events(row['source_event_refs'])
    texts={};originals={'content-metadata':{'artifact':deepcopy(refs['content-metadata']),'base64':b64encode(metadata_bytes).decode('ascii')}}
    for row in rows:
        if not full:continue
        name=row['part_name'];raw=T._bytes(parts[name],row['artifact']);texts[name]=T._text(raw)
        originals[name]={'artifact':deepcopy(row['artifact']),'base64':b64encode(raw).decode('ascii')}
    return {'family':'uncertainty','record':record,'typed':{},'texts':texts,'artifacts':deepcopy(refs),'originals':originals,
        'dependencies':[],'body_bytes_verified':full,'causal_dependencies_verified':False,'authenticated_service_claimed':False}


def uncertainty_successor(previous,current):
    a=previous['record'];b=current['record'];C.require(previous['family']==current['family']=='uncertainty','cross-family uncertainty successor')
    for key in ('uncertainty_ref','uncertainty_key','entity_id','scope','namespace','subject_ref','created_at','identity_registration','question'):
        C.require(a[key]==b[key],'immutable uncertainty identity/question differs')
    C.require(b['content_version']==a['content_version']+1 and b['previous_content_sha256']==a['record_sha256'] and b['updated_at']>a['updated_at'],'uncertainty successor differs')
    allowed={'open':{'open','resolved','retired'},'resolved':{'resolved','reopened','retired'},'reopened':{'reopened','resolved','retired'},'retired':set()}
    C.require(b['state'] in allowed[a['state']],'uncertainty transition differs')
    C.require(b['resolution_observations'][:len(a['resolution_observations'])]==a['resolution_observations'],'historical resolution removed/reordered/changed')
    if a['state'] in ('open','reopened') and b['state']=='resolved':
        C.require(len(b['resolution_observations'])>len(a['resolution_observations']),'new authored resolution missing')
    old={r['alternative_id']:r for r in a['alternatives']};new={r['alternative_id']:r for r in b['alternatives']}
    referenced={ref for row in a['resolution_observations'] for ref in row['selected_alternative_ids']}
    C.require(referenced<=set(new),'historically selected alternative removed')
    for key in set(old)&set(new):C.require(old[key]['text']==new[key]['text'],'retained alternative text changed')
