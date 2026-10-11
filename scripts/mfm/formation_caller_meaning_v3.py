"""Closed supplied Claim/Episode/task meaning. No teacher or authenticated rights."""
from base64 import b64decode,b64encode
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from . import formation_world_contracts as C,formation_caller_input as F,formation_caller_meaning as M,formation_caller_meaning_v2 as E2
from . import formation_world_replay_v3 as R,formation_task_content_input as T,formation_uncertainty_input as U,formation_episode_input as E

SCHEMA='mfm-formation-caller-meaning-v3'
PROFILE='text-causal-claim-episode-task-uncertainty-joint-v1'
TASK_SNAPSHOT_SCHEMA='mfm-task-semantic-snapshot-v1'
FAMILIES=('mission-content-v1','workspace-content-v1','uncertainty-content-v1')


def normative_view_policy_v3():
    return C.seal({'schema':'mfm-caller-meaning-view-policy-v3','profile':PROFILE,'families':[*M.FAMILIES,E2.EPISODE_FAMILY,*FAMILIES],
        'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'old_family_policy':M.normative_view_policy(),
        'episode_policy':E2.normative_view_policy_v2(),'task_binding':T.task_content_binding(),'uncertainty_binding':U.uncertainty_input_binding(),
        'historical_replay_binding':R.world_contract_binding_v3(),'teacher_input':False,'authority':False})


def content_key(row):return (row['family'],row['entity_id'],row['content_version'])


def delivery_parts(row):
    """Owned originals visible under this finite audience profile, never all proof bytes."""
    record=row['record'];family=row['family']
    names=['content-metadata',*[r['part_name'] for r in T.owned_artifacts(record)]]
    if family=='workspace':
        return ['content-metadata',record['projection_artifact']['artifact_id'],*[r['projected_body']['artifact_id'] for r in record['projected_items']]]
    return names


def visible_parts(row):
    names=delivery_parts(row)
    if row['family']=='result':
        opaque={r['artifact']['artifact_id'] for r in row['record']['output_artifacts'] if r['availability']=='present'}
        return [n for n in names if n not in opaque] # authenticated opaque bytes are not guessed text
    return names


def content_status(view,row,audience):
    status=next(r['status'] for r in view['content_visibility'] if (r['family'],r['entity_id'])==content_key(row)[:2])
    if status!='present':return status
    if audience not in row['record']['audiences']:return 'unavailable'
    if row['family']=='workspace' and row['validated']['typed']['projection']['audience']!=audience:return 'unavailable'
    return 'present'


def target_row(view,target,context):
    C.exact(target,{'ref_id','kind','family','entity_id','content_version','record_sha256'},'target content binding')
    C.require(target['kind']==target['family'] and target['kind'] in ('mission','workspace'),'task role differs')
    matches=[t for t in context.targets if t.ref_id==target['ref_id'] and t.kind==target['kind']]
    C.require(len(matches)==1,'task body not an independently requested context target')
    key=content_key(target);rows={content_key(r):r for r in view['input_contents']}
    C.require(key in rows and rows[key]['record']['record_sha256']==target['record_sha256'],'target original absent/different')
    C.require(any(content_key(r)==key for r in view['content_current']),'target content not selected current version')
    row=rows[key]
    if target['kind']=='mission':
        graph=row['validated']['typed']['graph'];C.require(target['ref_id']==graph['mission']['mission_id'] or target['ref_id'] in [n['node_id'] for n in graph['nodes']],'mission alias unsupported')
    else:C.require(target['ref_id']==row['validated']['typed']['projection']['projection_id'],'workspace alias unsupported')
    return row


def content_closure(view,primary):
    """Exact authored joins plus actor/observer records anchored to those same Attempts."""
    all_rows={content_key(r):r for r in view['input_contents']};selected={content_key(primary)}
    changed=True
    while changed:
        changed=False
        for key in list(selected):
            for ref in all_rows[key]['validated']['dependencies']:
                child=content_key(ref['content'])
                if child not in selected:selected.add(child);changed=True
        attempts={k for k in selected if k[0]=='attempt'}
        for key,row in all_rows.items():
            if row['family'] in ('result','outcome') and content_key(row['record']['attempt_ref']['content']) in attempts and key not in selected:
                selected.add(key);changed=True
    return [primary,*[r for r in view['input_contents'] if content_key(r) in selected and content_key(r)!=content_key(primary)]]


def _snapshot(row,expected,context,registry,claim_snapshot,episode_snapshot):
    C.pin(row,expected,TASK_SNAPSHOT_SCHEMA)
    C.exact(row,{'schema','scope','namespace','cut','replay_sha256','historical_prefix_proof','content_profile','content_refs','target_content_refs',
        'uncertainty_ref_ids','uncertainty_bindings','audience','read_service_id','read_receipt_sha256','policy_sha256','record_sha256'},'task snapshot')
    C.require(row['scope']==context.base.scope.metadata_record() and row['namespace']==context.base.authority_namespace_id,'task snapshot scope differs')
    C.require(row['audience'] in ('host','non_host'),'task audience differs')
    for key in ('read_service_id','read_receipt_sha256','policy_sha256'):C.require(row[key]==claim_snapshot[key] and row[key]==episode_snapshot[key],'same delivery boundary required')
    view,_=R.prefix_proof_replay_v3(row['historical_prefix_proof'],cut=row['cut'])
    C.require(view['replay_sha256']==row['replay_sha256'] and view['input_content_profile']==row['content_profile'] and
        row['content_refs']==[r['known_content_ref'] for r in view['input_contents']] and row['uncertainty_bindings']==view['uncertainty_bindings'],
        'task selected prefix/original inventory differs')
    C.require(claim_snapshot['authority_generation']==view['authority_generation'] and claim_snapshot['source_position']==view['authority_source_position'],'Claim/world counters differ')
    C.require(episode_snapshot['historical_prefix_proof']==row['historical_prefix_proof'] and episode_snapshot['cut']==row['cut'] and
        episode_snapshot['projection_profile_sha256']==view['episode_profile']['profile_sha256'],'Episode/task proof differs')
    known=[]
    for item in episode_snapshot['episodes']:
        source=next((r for r in view['episodes'] if r['ref_id']==item['ref_id']),None);C.require(source is not None,'Episode known-at absent')
        expected_row={k:deepcopy(source[k]) for k in ('ref_id','known_at_sequence','known_at_entry_sha256','member_resolution','member_registrations')}
        expected_row.update(episode_sha256=source['records']['episode']['episode_sha256'],generation=source['records']['episode']['generation'])
        C.require(item==expected_row,'Episode known-at originals differ');known.append(item['ref_id'])
    C.require(known==[r['ref_id'] for r in registry['records'] if r['kind']=='episode'],'complete Episode snapshot required')
    seen=set();ordinals={r.ref_id:i for i,r in enumerate(context.targets)};last=-1
    for target in row['target_content_refs']:
        target_row(view,target,context);C.require(target['ref_id'] not in seen and ordinals[target['ref_id']]>last,'task target order differs')
        seen.add(target['ref_id']);last=ordinals[target['ref_id']]
    T.ids(row['uncertainty_ref_ids'],'uncertainty selection');priors={r['ref_id']:r for r in registry['records']}
    C.require(all(ref in priors and priors[ref]['kind']=='uncertainty' for ref in row['uncertainty_ref_ids']),'uncertainty registry role differs')
    C.require(row['uncertainty_ref_ids']==[r['ref_id'] for r in registry['records'] if r['ref_id'] in row['uncertainty_ref_ids']],'uncertainty request order differs')
    return view


def _task_meaning(row,candidate,snapshot,view):
    C.exact(row,M._BODY_KEYS|{'original_parts','observed_source_eligible'},'task meaning')
    C.require(row['observed_source_eligible'] is False and row['availability']=='present','prior content cannot be observation authority')
    family=row['family'].split('-')[0]
    if family=='uncertainty':
        C.require(candidate['origin']=='readonly_registry' and candidate['kind']=='uncertainty' and candidate['ref_id'] in snapshot['uncertainty_ref_ids'],'uncertainty role differs')
        binding=next((r for r in view['uncertainty_bindings'] if r['ref_id']==candidate['ref_id']),None);C.require(binding is not None,'uncertainty material not bound')
        primary=next(r for r in view['input_contents'] if r['known_content_ref']==binding['content_ref'])
        C.require(candidate['provenance']['record']['record_sha256']==primary['record']['record_sha256'] and candidate['provenance']['record']['status']=='present','uncertainty current registry differs')
    else:
        C.require(candidate['origin']=='context_target' and candidate['kind']==family,'task candidate role differs')
        target=next((r for r in snapshot['target_content_refs'] if r['ref_id']==candidate['ref_id']),None);C.require(target is not None,'task target not requested')
        primary=next(r for r in view['input_contents'] if content_key(r)==content_key(target))
    C.require(content_status(view,primary,snapshot['audience'])=='present','task primary lifecycle/audience differs')
    originals=content_closure(view,primary) if family!='uncertainty' else [primary]
    body=M._body(row);C.exact(body,{'schema','contents'},'task body');C.require(body['schema']=='mfm-'+family+'-snapshot-meaning-v1','task body schema differs')
    C.require(type(row['original_parts']) is list and len(row['original_parts'])==len(originals) and len(body['contents'])==len(originals),'complete linked originals required')
    proof={r['artifact']['artifact_id']:r for r in view['historical_prefix_proof']['original_artifacts']}
    for source,custody,item in zip(originals,row['original_parts'],body['contents']):
        C.exact(custody,{'known_content_ref','parts'},'task original custody');C.require(custody['known_content_ref']==source['known_content_ref'],'task original known-at differs')
        C.exact(item,{'family','entity_id','content_version','availability','record','parts'},'task body original')
        status=content_status(view,source,snapshot['audience'])
        C.require({k:item[k] for k in ('family','entity_id','content_version')}=={k:source[k] for k in ('family','entity_id','content_version')} and item['availability']==status,'linked content identity/status differs')
        if status!='present':C.require(item['record'] is None and item['parts'] is None and custody['parts']=={},'denied linked original has body');continue
        refs={r['part_name']:r['artifact'] for r in source['commit']['original_artifacts']}
        C.require(set(custody['parts'])==set(refs),'task full original inventory differs');raws={}
        for name,part in custody['parts'].items():
            C.exact(part,{'artifact','base64'},'task original part');C.require(part['artifact']==refs[name],'task original descriptor differs')
            raw=b64decode(part['base64'],validate=True);C.require(b64encode(raw).decode('ascii')==part['base64'],'task original encoding differs');T._bytes(raw,refs[name]);raws[name]=raw
            if refs[name]['media_type']=='application/json':C.require(proof[refs[name]['artifact_id']]==part,'typed proof original differs')
        original=(U.uncertainty_original if source['family']=='uncertainty' else T.task_originals)(raws['content-metadata'],{k:v for k,v in raws.items() if k!='content-metadata'},
            refs=refs,scope=C.scope(snapshot['scope']),namespace=snapshot['namespace'],contract_binding=U.uncertainty_input_binding() if source['family']=='uncertainty' else T.task_content_binding(),
            contract_binding_sha256=(U.uncertainty_input_binding() if source['family']=='uncertainty' else T.task_content_binding())['record_sha256'])
        C.require(item['record']==original['record'],'task original metadata differs')
        visible={name:(C.parse_json(raws[name]) if refs[name]['media_type']=='application/json' else T._text(raws[name])) for name in visible_parts(source) if name!='content-metadata'}
        C.require(item['parts']==visible,'approved complete task projection differs')


def caller_meaning_v3_record(caller,*,caller_sha256,claim_snapshot,claim_snapshot_sha256,episode_snapshot,episode_snapshot_sha256,
    task_snapshot,task_snapshot_sha256,candidate_meanings,view_policy,view_policy_sha256):
    C.require(caller.get('record_sha256')==caller_sha256,'external caller differs');context,opened,registry=F.caller_input_from_record(caller)
    M._snapshot(claim_snapshot,claim_snapshot_sha256,context,registry);C.pin(episode_snapshot,episode_snapshot_sha256,E2.EPISODE_SNAPSHOT_SCHEMA)
    C.exact(episode_snapshot,{'schema','scope','namespace','projection_profile_sha256','cut','episodes','historical_prefix_proof','read_service_id','read_receipt_sha256','policy_sha256','record_sha256'},'Episode snapshot')
    view=_snapshot(task_snapshot,task_snapshot_sha256,context,registry,claim_snapshot,episode_snapshot)
    C.pin(view_policy,view_policy_sha256,'mfm-caller-meaning-view-policy-v3');C.require(view_policy==normative_view_policy_v3(),'normative v3 policy differs')
    candidates=F.selection_candidates(context,opened,registry);C.require(type(candidate_meanings) is list and len(candidate_meanings)==len(candidates),'complete meaning inventory required')
    for slot,(row,candidate) in enumerate(zip(candidate_meanings,candidates)):
        C.require(type(row.get('slot')) is int and row['slot']==slot,'typed meaning ordinal differs')
        requested=None
        if candidate['origin']=='context_target':
            target=next((r for r in task_snapshot['target_content_refs'] if r['ref_id']==candidate['ref_id']),None)
            if target is not None:requested=next(r for r in view['input_contents'] if content_key(r)==content_key(target))
        elif candidate['origin']=='readonly_registry' and candidate['ref_id'] in task_snapshot['uncertainty_ref_ids']:
            binding=next((r for r in view['uncertainty_bindings'] if r['ref_id']==candidate['ref_id']),None)
            if binding is not None:requested=next(r for r in view['input_contents'] if r['known_content_ref']==binding['content_ref'])
        if requested is not None:
            status=content_status(view,requested,task_snapshot['audience'])
            if candidate['origin']=='readonly_registry' and candidate['provenance']['record']['status']!='present':status=candidate['provenance']['record']['status']
            if status=='present':C.require(row['family']==requested['family']+'-content-v1','complete requested original body is missing')
            else:C.require(row['family']=='absent-body-v1' and row['availability']==status,'denied requested original availability differs')
        if row.get('family') in FAMILIES:_task_meaning(row,candidate,task_snapshot,view)
        elif row.get('family')==E2.EPISODE_FAMILY:E2._episode_meaning(row,candidate,episode_snapshot,context,candidates,view_policy['episode_policy'])
        else:M._meaning(row,candidate,claim_snapshot,context)
    return C.seal({'schema':SCHEMA,'caller_sha256':caller_sha256,'claim_snapshot':claim_snapshot,'episode_snapshot':episode_snapshot,'task_snapshot':task_snapshot,
        'candidate_inventory':candidates,'candidate_meanings':candidate_meanings,'view_policy':view_policy})


def caller_meaning_v3_from_record(record,caller,*,meaning_sha256,caller_sha256):
    C.pin(record,meaning_sha256,SCHEMA);C.exact(record,{'schema','caller_sha256','claim_snapshot','episode_snapshot','task_snapshot','candidate_inventory','candidate_meanings','view_policy','record_sha256'},'meaning v3')
    kwargs={k:record[k] for k in ('claim_snapshot','episode_snapshot','task_snapshot','candidate_meanings','view_policy')}
    for name in ('claim_snapshot','episode_snapshot','task_snapshot','view_policy'):kwargs[name+'_sha256']=record[name]['record_sha256']
    rebuilt=caller_meaning_v3_record(caller,caller_sha256=caller_sha256,**kwargs);C.require(rebuilt==record,'complete meaning v3 differs');return rebuilt


def semantic_view_v3(record,caller,*,meaning_sha256,caller_sha256):
    value=caller_meaning_v3_from_record(record,caller,meaning_sha256=meaning_sha256,caller_sha256=caller_sha256);result=[]
    for candidate,row in zip(value['candidate_inventory'],value['candidate_meanings']):
        item={'slot':row['slot'],'candidate':deepcopy(candidate),'availability':row['availability'],'family':row['family'],
            'provenance_class':'validated_original_prior' if row['family'] in (*FAMILIES,E2.EPISODE_FAMILY,'claim-snapshot-v1') else 'supplied_metadata',
            'content':M._body(row),'body_file_sha256':row['body_sha256'],'independent_observation':False}
        for key in ('original_parts','member_resolution'): 
            if key in row:item[key]=deepcopy(row[key])
        result.append(item)
    return result
