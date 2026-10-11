"""Current one-part delivery beside frozen interfaces; supplied policy is not a service."""
from base64 import b64decode,b64encode
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from cognitive_kernel.formation_semantics_v16 import context_v16_from_record
from . import formation_world_contracts as C,formation_world_replay_v3 as R,formation_task_content_input as T
from . import formation_caller_input as F,formation_caller_meaning as M,formation_caller_meaning_v2 as E2,formation_caller_meaning_v3 as V,formation_episode_input as E
from .formation_world_caller_v2 import _entry


def _boundary(row,view):
    C.exact(row,{'scope','namespace','purpose','read_service_id','as_of','policy_sha256','rights_receipt_refs','lifecycle_receipt_ref','audience','record_sha256'},'delivery boundary')
    C.require(row['record_sha256']==C.canonical_sha256({k:v for k,v in row.items() if k!='record_sha256'}),'delivery boundary seal differs')
    C.require(row['scope']==view['scope'] and row['namespace']==view['namespace'] and row['purpose'] in ('formation_training','formation_evaluation','memory_formation') and row['audience'] in ('host','non_host'),'delivery scope/purpose/audience differs')
    C.timestamp(row['as_of']);C.identifier(row['read_service_id']);C.digest(row['policy_sha256']);C.identifier(row['lifecycle_receipt_ref']);T.ids(row['rights_receipt_refs'],'delivery rights');C.require(bool(row['rights_receipt_refs']),'delivery rights absent')


def _proof(view,receipt,expected,adapter,boundary,boundary_sha,limits):
    R.verification_boundary_original(boundary,boundary_sha,view['scope'],view['namespace'])
    def check():
        value=adapter.verification_boundary_record();R.verification_boundary_original(value,boundary_sha,view['scope'],view['namespace']);C.require(value==boundary,'verification policy changed before delivery')
    check();C.pin(receipt,expected,'mfm-world-verification-read-receipt-v3')
    C.exact(receipt,{'schema','world_sha256','cut_sha256','replay_sha256','mode','boundary_sha256','reads','supplied_policy_evidence_only','current_service_read_claimed','current_permission_established','authenticated_service_claimed','record_sha256'},'verification receipt')
    C.require(receipt['mode']=='current_verification_reads' and receipt['replay_sha256']==view['replay_sha256'] and receipt['cut_sha256']==view['cut']['record_sha256'] and
        receipt['boundary_sha256']==boundary_sha and receipt['current_permission_established'] is True and receipt['current_service_read_claimed'] is True and
        receipt['authenticated_service_claimed'] is False and receipt['supplied_policy_evidence_only'] is True,'actual supplied current verification required')
    C.digest(receipt['world_sha256']);C.require(type(receipt['reads']) is list,'verification reads malformed')
    for row in receipt['reads']:
        C.exact(row,{'artifact','pre_boundary_sha256','post_boundary_sha256','body_bytes','body_sha256'},'verification read')
        C.artifact_ref(row['artifact'],limits);C.require(row['pre_boundary_sha256']==boundary_sha==row['post_boundary_sha256'] and
            row['body_bytes']==row['artifact']['bytes'] and row['body_sha256']==row['artifact']['file_sha256'],'verification original differs')
    C.require([r['artifact'] for r in receipt['reads']]==view['original_artifacts'],'complete verification original inventory required')
    originals={};reads=[];total=0
    for row in view['historical_prefix_proof']['original_artifacts']:
        ref=row['artifact'];C.artifact_ref(ref,limits);total+=ref['bytes'];C.require(total<=limits.max_opened_bytes,'complete proof byte limit exceeded')
        check();raw=adapter.read_verification_artifact(deepcopy(ref));check()
        C.require(type(raw) is bytes and len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['file_sha256'] and raw==b64decode(row['base64'],validate=True),'current proof original differs')
        originals[ref['artifact_id']]=raw;reads.append({'artifact':ref,'pre_boundary_sha256':boundary_sha,'post_boundary_sha256':boundary_sha,'body_bytes':len(raw),'body_sha256':sha256(raw).hexdigest()})
    rebuilt,_=R.prefix_proof_replay_v3(view['historical_prefix_proof'],cut=view['cut']);C.require(rebuilt==view,'complete historical view differs');check()
    return originals,reads,check


def world_caller_bundle_v3(view,*,replay_sha256,verification_receipt,verification_receipt_sha256,input_request,input_request_sha256,
    verification_adapter,verification_boundary,verification_boundary_sha256,read_adapter,view_policy,view_policy_sha256,contract_binding,contract_binding_sha256,limits):
    R.binding_v3(contract_binding,contract_binding_sha256);C.pin(view,replay_sha256,'mfm-world-replay-view-v3','replay_sha256')
    C.pin(input_request,input_request_sha256,'mfm-world-input-request-v3')
    C.exact(input_request,{'schema','cut','context','opened_source_refs','prior_ref_ids','target_descriptions','read_boundary','target_content_refs','uncertainty_ref_ids','audience','record_sha256'},'world input v3')
    C.require(input_request['cut']==view['cut'] and view['profile']==T.PROFILE and view['supplied_governance_only'] is True and view['authenticated_service_claimed'] is False,'input profile/cut differs')
    context=context_v16_from_record(input_request['context']);context.validate();C.require(context.record()==input_request['context'] and context.base.scope.metadata_record()==view['scope'] and context.base.authority_namespace_id==view['namespace'],'context scope differs')
    registered={r['target']['ref_id']:r['target'] for r in view['registry_targets']}
    for target in context.targets:C.require(registered.get(target.ref_id)==target.record(),'target not in selected registration')
    for key in ('opened_source_refs','prior_ref_ids','uncertainty_ref_ids'):T.ids(input_request[key],key)
    C.require(input_request['opened_source_refs']==[r.ref_id for r in context.base.evidence],'complete source order differs')
    C.require(input_request['uncertainty_ref_ids']==[x for x in input_request['prior_ref_ids'] if x in input_request['uncertainty_ref_ids']],'uncertainty order differs')
    C.pin(view_policy,view_policy_sha256,'mfm-caller-meaning-view-policy-v3');C.require(view_policy==V.normative_view_policy_v3(),'normative policy differs')
    boundary=input_request['read_boundary'];_boundary(boundary,view);C.require(boundary['audience']==input_request['audience'],'audience differs')
    targets={};last=-1;ordinals={r.ref_id:i for i,r in enumerate(context.targets)}
    for row in input_request['target_content_refs']:
        source=V.target_row(view,row,context);C.require(row['ref_id'] not in targets and ordinals[row['ref_id']]>last,'target order differs');targets[row['ref_id']]=source;last=ordinals[row['ref_id']]
    sources={r['registration']['evidence']['ref_id']:r for r in view['sources']};claims={r['ref_id']:r for r in view['current_claims']};episodes={r['ref_id']:r for r in view['episodes']};metadata={r['ref_id']:r for r in view['registry_metadata']}
    for ref_id in input_request['prior_ref_ids']:C.require(ref_id in claims or ref_id in episodes or ref_id in metadata,'prior not supplied at cut')
    for ref_id in input_request['uncertainty_ref_ids']:C.require(ref_id in metadata and metadata[ref_id]['kind']=='uncertainty','uncertainty prior kind differs')
    descriptions={}
    for row in input_request['target_descriptions']:
        C.exact(row,{'ref_id','kind','description'},'target description');C.require(row['ref_id'] in ordinals and row['ref_id'] not in descriptions and row['kind']==context.targets[ordinals[row['ref_id']]].kind and type(row['description']) is str and '\x00' not in row['description'],'unsupported target description');descriptions[row['ref_id']]=row
    proof,proof_reads,check_proof=_proof(view,verification_receipt,verification_receipt_sha256,verification_adapter,verification_boundary,verification_boundary_sha256,limits)
    def current():
        check_proof();actual=read_adapter.delivery_boundary_record();_boundary(actual,view);C.require(actual==boundary,'current delivery policy changed')
    current();audit=[];consumed=0
    def read(kind,key,ref):
        nonlocal consumed
        C.artifact_ref(ref,limits);C.require(consumed+ref['bytes']<=limits.max_opened_bytes,'complete delivered byte limit exceeded')
        current();raw=getattr(read_adapter,'read_'+kind+'_part')(deepcopy(key),deepcopy(ref));current()
        C.require(type(raw) is bytes and len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['file_sha256'],'delivered original differs');consumed+=len(raw)
        audit.append({'family':kind,'key':deepcopy(key),'artifact':deepcopy(ref),'pre_boundary_sha256':boundary['record_sha256'],'post_boundary_sha256':boundary['record_sha256'],'body_bytes':len(raw),'body_sha256':sha256(raw).hexdigest()});return raw
    opened=[]
    for evidence in context.base.evidence:
        C.require(evidence.ref_id in sources and view['visibility'][evidence.ref_id]=='present','source not visible at cut');row=sources[evidence.ref_id]
        C.require(C.registration(row['registration']).evidence==evidence,'original source registration differs')
        opened.append((evidence.ref_id,read('source',row['registration'],row['source_artifact'])))
    prior_rows=[];bodies={};extras={};episode_rows=[]
    for ref_id in input_request['prior_ref_ids']:
        status=view['visibility'].get(ref_id);C.require(status in F.REGISTRY_STATUSES,'prior lifecycle differs');body=None
        if ref_id in claims:
            source=claims[ref_id];records=source['records'];refs={k:r['artifact'] for k,r in source['originals'].items()}
            if status=='present':
                raws={}
                for name in ('identity','version','projection'):
                    projection=records['projection'];version=records['version'];key={'family':'claim','claim_id':ref_id,'claim_version_id':version['claim_version_id'],
                        'version_sequence':version['version_sequence'],'projection_generation':projection['projection_generation'],'source_position':projection['source_position'],
                        'authority_generation':projection['authority_generation'],'authority_source_position':view['authority_source_position'],'part_name':name}
                    raws[name]=read('claim',key,refs[name]);C.require(raws[name]==b64decode(source['originals'][name]['base64'],validate=True),'Claim original differs')
                body={'schema':'mfm-claim-snapshot-meaning-v1',**deepcopy(records)}
            prior_rows.append({'ref_id':ref_id,'kind':'claim','record_sha256':records['version']['version_sha256'],'status':status,'description':''});bodies[ref_id]=('claim-snapshot-v1',body)
        elif ref_id in episodes:
            source=episodes[ref_id];records=source['records'];known={k:deepcopy(source[k]) for k in ('ref_id','known_at_sequence','known_at_entry_sha256','member_resolution','member_registrations')};known.update(episode_sha256=records['episode']['episode_sha256'],generation=records['episode']['generation']);episode_rows.append(known)
            if status=='present':
                C.require(records['episode']['envelope']['deletion_state']=='active' and records['episode']['episode_state'] in view_policy['episode_policy']['eligible_episode_states'],'inactive Episode')
                raws={name:read('episode',{'family':'episode','episode_id':ref_id,'generation':records['episode']['generation'],'part_name':name},source['artifacts'][name]) for name in E2.PARTS}
                remaining={name:proof[ref['artifact_id']] for name,ref in source['artifacts'].items() if name not in E2.PARTS};eb=E.episode_input_binding()
                complete=E.episode_originals(*(raws[name] for name in E2.PARTS),*(remaining[name] for name in ('append_receipt','append_invocation','projection_profile')),
                    refs=source['artifacts'],scope=C.scope(view['scope']),namespace=view['namespace'],contract_binding=eb,contract_binding_sha256=eb['record_sha256'])
                body={'schema':'mfm-episode-snapshot-meaning-v1',**complete['records']};extras[ref_id]={'original_parts':complete['originals'],'member_resolution':deepcopy(source['member_resolution'])}
            prior_rows.append({'ref_id':ref_id,'kind':'episode','record_sha256':records['episode']['episode_sha256'],'status':status,'description':''});bodies[ref_id]=(E2.EPISODE_FAMILY,body)
        else:
            row=metadata[ref_id];prior_rows.append({k:deepcopy(row[k]) for k in ('ref_id','kind','record_sha256','description') }|{'status':status});bodies[ref_id]=('absent-body-v1',None)
    def task_body(primary):
        closure=V.content_closure(view,primary) if primary['family']!='uncertainty' else [primary];items=[];custody=[]
        for source in closure:
            status=V.content_status(view,source,input_request['audience']);item={k:source[k] for k in ('family','entity_id','content_version')};item.update(availability=status,record=None,parts=None)
            refs={r['part_name']:r['artifact'] for r in source['commit']['original_artifacts']};raws={};parts={}
            if status=='present':
                for name,ref in refs.items():
                    if name in V.delivery_parts(source):raw=read('content',{'family':source['family'],'entity_id':source['entity_id'],'content_version':source['content_version'],'part_name':name},ref)
                    else:C.require(ref['artifact_id'] in proof,'verification-only original absent');raw=proof[ref['artifact_id']]
                    raws[name]=raw;parts[name]={'artifact':ref,'base64':b64encode(raw).decode('ascii')}
                C.require(C.parse_json(raws['content-metadata'])==source['record'],'current task metadata differs')
                item.update(record=deepcopy(source['record']),parts={name:C.parse_json(raws[name]) if refs[name]['media_type']=='application/json' else T._text(raws[name]) for name in V.visible_parts(source) if name!='content-metadata'})
            items.append(item);custody.append({'known_content_ref':source['known_content_ref'],'parts':parts})
        return {'schema':'mfm-'+primary['family']+'-snapshot-meaning-v1','contents':items},{'original_parts':custody,'observed_source_eligible':False}
    tasks={};derived_absence={}
    for ref_id,primary in targets.items():
        status=V.content_status(view,primary,input_request['audience'])
        if status=='present':tasks[ref_id]=(primary['family']+'-content-v1',*task_body(primary))
        else:derived_absence[ref_id]=status
    for ref_id in input_request['uncertainty_ref_ids']:
        binding=next((r for r in view['uncertainty_bindings'] if r['ref_id']==ref_id),None)
        if binding is not None and view['visibility'][ref_id]=='present':
            primary=next(r for r in view['input_contents'] if r['known_content_ref']==binding['content_ref'])
            status=V.content_status(view,primary,input_request['audience'])
            if status=='present':bodies[ref_id]=('uncertainty-content-v1',*task_body(primary))
            else:derived_absence[ref_id]=status
    current();delivery_receipt=C.seal({'schema':'mfm-world-delivery-read-receipt-v3','boundary_sha256':boundary['record_sha256'],'audience':boundary['audience'],'reads':audit,'supplied_policy_evidence_only':True,'authenticated_service_claimed':False})
    read_digest=delivery_receipt['record_sha256'];generation='world-read-'+C.canonical_sha256({'prefix':view['world_prefix_sha256'],'request':input_request_sha256,'boundary':boundary})[:40]
    registry={'schema':F.REGISTRY_SCHEMA,'host_scope':view['scope'],'namespace':view['namespace'],'generation':generation,'records':prior_rows,'records_sha256':C.canonical_sha256(prior_rows)};registry['registry_sha256']=C.canonical_sha256(registry)
    caller=F.caller_input_record(context,opened,registry)
    snapshot=C.seal({'schema':M.SNAPSHOT_SCHEMA,'scope':view['scope'],'namespace':view['namespace'],'generation':generation,'authority_generation':view['authority_generation'],'source_position':view['authority_source_position'],'as_of':boundary['as_of'],'read_service_id':boundary['read_service_id'],'read_receipt_sha256':read_digest,'policy_sha256':boundary['policy_sha256']})
    common={'scope':view['scope'],'namespace':view['namespace'],'cut':view['cut'],'historical_prefix_proof':view['historical_prefix_proof'],'read_service_id':boundary['read_service_id'],'read_receipt_sha256':read_digest,'policy_sha256':boundary['policy_sha256']}
    episode_snapshot=C.seal({'schema':E2.EPISODE_SNAPSHOT_SCHEMA,**common,'projection_profile_sha256':view['episode_profile']['profile_sha256'],'episodes':episode_rows})
    task_snapshot=C.seal({'schema':V.TASK_SNAPSHOT_SCHEMA,**common,'replay_sha256':view['replay_sha256'],'content_profile':view['input_content_profile'],
        'content_refs':[r['known_content_ref'] for r in view['input_contents']],'target_content_refs':input_request['target_content_refs'],'uncertainty_ref_ids':input_request['uncertainty_ref_ids'],'uncertainty_bindings':view['uncertainty_bindings'],'audience':input_request['audience']})
    entries=[]
    for slot,candidate in enumerate(F.selection_candidates(context,opened,registry)):
        ref_id=candidate['ref_id'];origin=candidate['origin']
        if origin=='opened_source':entry=_entry(slot,'opened-source-v1')
        elif origin=='source_subject':entry=_entry(slot,'source-subject-v1','metadata_only')
        elif origin=='context_target':
            if ref_id in tasks:family,body,extra=tasks[ref_id];entry=_entry(slot,family,body=body);entry.update(extra)
            elif ref_id in descriptions and ref_id not in targets:entry=_entry(slot,'registered-description-v1',body={'schema':'mfm-registered-target-description-v1','scope':view['scope'],'namespace':view['namespace'],**descriptions[ref_id],'read_receipt_sha256':read_digest})
            else:entry=_entry(slot,'absent-body-v1',derived_absence.get(ref_id,'unavailable'))
        else:
            values=bodies[ref_id];family,body=values[:2]
            if body is None:entry=_entry(slot,'absent-body-v1',derived_absence.get(ref_id,'metadata_only' if view['visibility'][ref_id]=='present' else view['visibility'][ref_id]))
            else:
                entry=_entry(slot,family,body=body)
                if len(values)>2:entry.update(values[2])
                elif ref_id in extras:entry.update(extras[ref_id])
        entries.append(entry)
    meaning=V.caller_meaning_v3_record(caller,caller_sha256=caller['record_sha256'],claim_snapshot=snapshot,claim_snapshot_sha256=snapshot['record_sha256'],
        episode_snapshot=episode_snapshot,episode_snapshot_sha256=episode_snapshot['record_sha256'],task_snapshot=task_snapshot,task_snapshot_sha256=task_snapshot['record_sha256'],
        candidate_meanings=entries,view_policy=view_policy,view_policy_sha256=view_policy_sha256);current()
    return C.seal({'schema':'mfm-world-caller-bundle-v3','code_binding':world_caller_code_binding_v3(),'cut':view['cut'],'replay_sha256':replay_sha256,'input_request_sha256':input_request_sha256,
        'original_artifact_pins':view['original_artifacts'],'verification_receipt':verification_receipt,'verification_recheck_reads':proof_reads,
        'delivery_read_receipt':delivery_receipt,'caller':caller,'meaning':meaning,'production_authenticated_adapter_claimed':False},'bundle_sha256')


def verify_world_caller_bundle_v3(bundle,*,bundle_sha256,world,world_sha256,cut,cut_sha256,**kwargs):
    C.pin(bundle,bundle_sha256,'mfm-world-caller-bundle-v3','bundle_sha256')
    view,vr=R.replay_world_prefix_v3(world,world_sha256=world_sha256,cut=cut,cut_sha256=cut_sha256,verification_mode='current_verification_reads',
        verification_adapter=kwargs['verification_adapter'],verification_boundary=kwargs['verification_boundary'],verification_boundary_sha256=kwargs['verification_boundary_sha256'],
        limits=kwargs['limits'],contract_binding=kwargs['contract_binding'],contract_binding_sha256=kwargs['contract_binding_sha256'])
    actual=world_caller_bundle_v3(view,**{**kwargs,'replay_sha256':view['replay_sha256'],'verification_receipt':vr,'verification_receipt_sha256':vr['record_sha256']})
    C.require(actual==bundle,'complete current caller reload differs');return actual


def world_caller_code_binding_v3():
    return {'schema':'mfm-world-caller-code-binding-v3','code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'caller_sha256':sha256(Path(F.__file__).read_bytes()).hexdigest(),
        'meaning_policy':V.normative_view_policy_v3(),'replay_binding':R.world_contract_binding_v3(),'one_part_callbacks_only':True,'authenticated_service_claimed':False}
