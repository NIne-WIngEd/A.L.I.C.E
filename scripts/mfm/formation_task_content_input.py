"""Closed supplied task originals. Typed validity, byte custody and knowledge differ."""
from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

from cognitive_kernel import canonical,contracts,mission,results,attention,workspace
from . import formation_world_contracts as C

PROFILE='supplied-claim-episode-task-uncertainty-prefix-v1'
SCHEMAS={'mission':'mfm-mission-original-content-v1','attempt':'mfm-task-attempt-input-v1',
    'result':'mfm-result-original-content-v1','outcome':'mfm-task-outcome-observation-input-v1',
    'workspace':'mfm-workspace-original-content-v1','uncertainty':'mfm-uncertainty-original-input-v1'}
COMMON={'schema','profile','scope','namespace','entity_id','content_version','previous_content_sha256','created_at','updated_at',
    'subject_ref','valid_from','valid_to','provenance','retention_class','storage_tier','deletion_lineage','policy_bindings','audiences','record_sha256'}
EXTRAS={'mission':{'graph_artifact','title_projection','mission_title','node_contents'},
    'attempt':{'attempt_id','mission_id','node_id','mission_content_ref','actor_ref','phase','instruction','expected_postconditions','related_plan_step_ids','source_event_refs'},
    'result':{'capsule_artifact','summary_projection','summary','mission_content_ref','attempt_ref','output_artifacts','outcome_observation_refs','traceback_artifacts','rationale_texts','evidence_refs','source_event_refs'},
    'outcome':{'observation_id','attempt_ref','observer_ref','observed_at','report','postcondition_reports','evidence_refs','source_event_refs','supersedes_observation_refs'},
    'workspace':{'projection_artifact','attention_artifact','override_artifacts','projected_items','override_reason_texts'},
    'uncertainty':{'uncertainty_ref','uncertainty_key','identity_registration','state','question','alternatives','evidence_refs','resolution_observations'}}
AVAILABILITY={'present','not_supplied','withheld','revoked','deleted'}


def task_content_binding():
    modules=(canonical,contracts,mission,results,attention,workspace,C)
    return C.seal({'schema':'mfm-task-content-binding-v1','profile':PROFILE,'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependencies':{m.__name__:sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules},
        'digest_profiles':['raw-utf8-sha256-v1','canonical-json-sha256-v1'],
        'authenticated_service_claimed':False,'authority_granted':False})


def array(value,name):
    C.require(type(value) is list,name+' must be an array');return value


def ids(value,name):
    array(value,name);C.require(len(set(value))==len(value),name+' duplicates')
    for item in value:C.identifier(item)
    return value


def _typed(cls,value):
    """Reconstruct supported original classes without a normalizing create shortcut."""
    derived={'priority_order'} if cls is attention.AttentionRankEntry else set()
    C.exact(value,{f.name for f in fields(cls)}|derived,'original '+cls.__name__)
    args=deepcopy(value)
    for key in derived:args.pop(key)
    nested={'mission':mission.Mission,'nodes':mission.MissionNode,'edges':mission.MissionEdge,
        'transitions':results.TracebackTransition,'entries':attention.AttentionRankEntry,
        'items':workspace.WorkspaceItemProjection,'layout':workspace.WorkspaceLayout}
    for f in fields(cls):
        key=f.name;annotation=str(f.type)
        if key=='scope':args[key]=C.scope(args[key])
        elif key=='provenance':args[key]=_typed(contracts.ProvenanceReference,args[key])
        elif key in nested:
            args[key]=tuple(_typed(nested[key],r) for r in array(args[key],key)) if key in ('nodes','edges','transitions','entries','items') else _typed(nested[key],args[key])
        elif 'tuple[' in annotation:args[key]=tuple(array(args[key],key))
        elif annotation in ('int',"<class 'int'>"):C.integer(args[key])
        elif annotation in ('bool',"<class 'bool'>"):C.require(type(args[key]) is bool,'original bool differs')
    try:
        result=cls(**args);result.validate()
        C.require(result.metadata_record()==value,'original '+cls.__name__+' normalizes')
        return result
    except (TypeError,AttributeError) as exc:
        raise canonical.CognitiveKernelContractError('malformed typed original '+cls.__name__) from exc


def artifact(ref,*,owned=True,media=None):
    C.exact(ref,{'artifact_id','bytes','file_sha256','media_type'},'content artifact')
    C.identifier(ref['artifact_id']);C.integer(ref['bytes'],1 if owned else 0);C.digest(ref['file_sha256'])
    C.require(ref['media_type'] in ('application/json','application/octet-stream'),'content media differs')
    if media is not None:C.require(ref['media_type']==media,'declared part media differs')
    return ref


def _bytes(raw,ref):
    artifact(ref);C.require(type(raw) is bytes and len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['file_sha256'],'content original bytes differ')
    return raw


def _text(raw):
    try:text=raw.decode('utf-8','strict')
    except UnicodeError as exc:raise canonical.CognitiveKernelContractError('invalid original text UTF-8') from exc
    C.require(bool(text) and '\x00' not in text,'empty/NUL original text');return text


def _experience(row):
    C.exact(row,{'event_id','event_sha256','known_at_sequence','entry_sha256'},'Experience ref')
    C.identifier(row['event_id']);C.digest(row['event_sha256']);C.integer(row['known_at_sequence'],1);C.digest(row['entry_sha256'])


def _events(rows):
    array(rows,'source events');seen=set()
    for row in rows:
        _experience(row);C.require(row['event_id'] not in seen,'duplicate event');seen.add(row['event_id'])


def known(row,family=None):
    C.exact(row,{'content','commit'},'known content');content=row['content']
    C.exact(content,{'family','entity_id','content_version','record_sha256'},'content ref')
    C.require(content['family'] in SCHEMAS and (family is None or content['family']==family),'content family differs')
    C.identifier(content['entity_id']);C.integer(content['content_version'],1);C.digest(content['record_sha256']);_experience(row['commit'])
    return row


def _knowns(rows,family):
    array(rows,'known refs');seen=set()
    for row in rows:
        known(row,family);key=canonical.canonical_sha256(row['content'])
        C.require(key not in seen,'duplicate known ref');seen.add(key)


def evidence(row):
    C.exact(row,{'ref_id','registration_sha256','known_at_entry_sha256','artifact','availability'},'evidence ref')
    C.identifier(row['ref_id']);C.require(row['availability'] in AVAILABILITY,'evidence availability differs')
    if row['availability']=='not_supplied':C.require(all(row[k] is None for k in ('registration_sha256','known_at_entry_sha256','artifact')),'absent evidence has bytes/proof')
    else:
        C.digest(row['registration_sha256']);C.digest(row['known_at_entry_sha256'])
        if row['availability']=='present':artifact(row['artifact'],owned=False)
        else:C.require(row['artifact'] is None,'denied evidence has bytes')


def _evidence(rows):
    array(rows,'evidence refs');seen=set()
    for row in rows:evidence(row);C.require(row['ref_id'] not in seen,'duplicate evidence');seen.add(row['ref_id'])


def common(record,scope,namespace):
    family=next((f for f,s in SCHEMAS.items() if record.get('schema')==s),None)
    C.require(family is not None,'unsupported content schema');C.exact(record,COMMON|EXTRAS[family],'content record')
    C.pin(record,record['record_sha256'],SCHEMAS[family]);C.require(record['profile']==PROFILE,'content profile differs')
    C.require(C.scope(record['scope'])==scope and record['namespace']==namespace,'foreign content scope/namespace')
    C.identifier(record['namespace']);C.identifier(record['entity_id']);C.integer(record['content_version'],1)
    if record['content_version']==1:C.require(record['previous_content_sha256'] is None,'initial predecessor differs')
    else:C.digest(record['previous_content_sha256'])
    for key in ('created_at','updated_at'):C.timestamp(record[key])
    C.require(record['updated_at']>=record['created_at'],'content time reverses')
    for key in ('valid_from','valid_to'):
        if record[key] is not None:C.timestamp(record[key])
    C.require(record['valid_from'] is None or record['valid_to'] is None or record['valid_from']<=record['valid_to'],'valid time reverses')
    if record['subject_ref'] is not None:C.identifier(record['subject_ref'])
    _typed(contracts.ProvenanceReference,record['provenance'])
    C.require(record['retention_class'] in contracts.RETENTION_CLASSES and record['storage_tier'] in contracts.STORAGE_TIERS,'lifecycle differs')
    C.require(record['retention_class']!='quarantine' or record['storage_tier']=='quarantine','quarantine tier differs')
    for key in ('deletion_lineage','policy_bindings'):ids(record[key],key)
    C.require(record['storage_tier']!='deleted' or bool(record['deletion_lineage']),'deleted lineage missing')
    ids(record['audiences'],'audiences');C.require(bool(record['audiences']) and set(record['audiences'])<={'host','non_host'},'audience differs')
    return family


def owned_artifacts(record):
    """Declared traversal; external evidence and dependency originals are not recaptured."""
    family=next((f for f,s in SCHEMAS.items() if record.get('schema')==s),None);C.require(family is not None,'unsupported content schema')
    rows=[]
    def add(ref,media):artifact(ref,media=media);rows.append({'part_name':ref['artifact_id'],'artifact':deepcopy(ref)})
    def text(ref):add(ref,'application/octet-stream')
    def obj(ref):add(ref,'application/json')
    if family=='mission':
        obj(record['graph_artifact']);text(record['mission_title'])
        for row in array(record['node_contents'],'nodes'):
            text(row['title'])
            for key in ('goals','constraints','plan_steps'):
                for item in array(row[key],key):text(item['text'])
    elif family=='attempt':
        text(record['instruction'])
        for row in array(record['expected_postconditions'],'conditions'):text(row['text'])
    elif family=='result':
        obj(record['capsule_artifact']);text(record['summary'])
        for row in array(record['output_artifacts'],'outputs'):
            C.exact(row,{'ref_id','availability','artifact'},'output ref');C.identifier(row['ref_id'])
            C.require(row['availability'] in AVAILABILITY,'output availability differs')
            if row['availability']=='present':add(row['artifact'],row['artifact']['media_type'])
            else:C.require(row['artifact'] is None,'absent output has bytes')
        for ref in array(record['traceback_artifacts'],'tracebacks'):obj(ref)
        for row in array(record['rationale_texts'],'rationales'):text(row['text'])
    elif family=='outcome':
        text(record['report'])
        for row in array(record['postcondition_reports'],'condition reports'):text(row['explanation'])
    elif family=='workspace':
        obj(record['projection_artifact']);obj(record['attention_artifact'])
        for ref in array(record['override_artifacts'],'overrides'):obj(ref)
        for row in array(record['projected_items'],'projected items'):obj(row['projected_body'])
        for row in array(record['override_reason_texts'],'override reasons'):text(row['text'])
    elif family=='uncertainty':
        text(record['question'])
        for row in array(record['alternatives'],'alternatives'):text(row['text'])
        for row in array(record['resolution_observations'],'resolutions'):text(row['report'])
    names=[r['part_name'] for r in rows]
    C.require(len(set(names))==len(names) and 'content-metadata' not in names,'duplicate/reserved original part')
    return rows


def _rows(rows,keys,id_key,name):
    array(rows,name);seen=set()
    for row in rows:
        C.exact(row,keys,name);C.identifier(row[id_key]);C.require(row[id_key] not in seen,name+' duplicate');seen.add(row[id_key])


def _dag(rows):
    _rows(rows,{'step_id','text','depends_on_step_ids'},'step_id','plan steps')
    graph={r['step_id']:ids(r['depends_on_step_ids'],'step dependencies') for r in rows};visiting=set();done=set()
    def walk(key):
        C.require(key in graph and key not in visiting,'missing/cyclic step dependency')
        if key in done:return
        visiting.add(key)
        for dep in graph[key]:walk(dep)
        visiting.remove(key);done.add(key)
    for key in graph:walk(key)


def _family(record,family,objects):
    typed={}
    if family=='mission':
        C.require(record['title_projection']=='raw-utf8-sha256-v1','unsupported title projection')
        graph=_typed(mission.MissionGraphSnapshot,objects[record['graph_artifact']['artifact_id']]);typed['graph']=graph.metadata_record()
        C.require(graph.mission.mission_id==record['entity_id'] and graph.mission.title_digest==record['mission_title']['file_sha256'],'mission identity/title differs')
        C.require(graph.mission.scope==C.scope(record['scope']),'foreign mission original')
        if record['content_version']==1:C.require(graph.revision==1 and graph.previous_snapshot_sha256 is None,'initial graph revision differs')
        C.require([r.get('node_id') for r in record['node_contents']]==[n.node_id for n in graph.nodes],'complete graph node order differs')
        for row,node in zip(record['node_contents'],graph.nodes):
            C.exact(row,{'node_id','node_sha256','title','goals','constraints','plan_steps','action_attempts'},'node contents')
            C.require(row['node_sha256']==node.node_sha256 and row['title']['file_sha256']==node.title_digest,'node original digest differs')
            for key in ('goals','constraints'):_rows(row[key],{'item_id','text'},'item_id',key)
            _dag(row['plan_steps']);_knowns(row['action_attempts'],'attempt')
    elif family=='attempt':
        for key in ('attempt_id','mission_id','node_id','actor_ref'):C.identifier(record[key])
        C.require(record['attempt_id']==record['entity_id'] and record['phase'] in ('planned','attempted','cancelled'),'attempt identity/phase differs')
        known(record['mission_content_ref'],'mission');_rows(record['expected_postconditions'],{'condition_id','text'},'condition_id','conditions')
        ids(record['related_plan_step_ids'],'related steps');_events(record['source_event_refs'])
    elif family=='result':
        C.require(record['summary_projection']=='raw-utf8-sha256-v1','unsupported summary projection')
        capsule=_typed(results.ResultCapsule,objects[record['capsule_artifact']['artifact_id']]);typed['capsule']=capsule.metadata_record()
        C.require(capsule.scope==C.scope(record['scope']) and capsule.capsule_id==record['entity_id'] and capsule.summary_digest==record['summary']['file_sha256'],'capsule identity/summary differs')
        known(record['mission_content_ref'],'mission');known(record['attempt_ref'],'attempt');_knowns(record['outcome_observation_refs'],'outcome')
        C.require([r['ref_id'] for r in record['output_artifacts']]==list(capsule.output_reference_ids),'capsule output roster differs')
        _evidence(record['evidence_refs']);_events(record['source_event_refs'])
        C.require([r['ref_id'] for r in record['evidence_refs']]==list(capsule.evidence_reference_ids) and
            [r['event_id'] for r in record['source_event_refs']]==list(capsule.source_event_ids),'capsule ancestry roster differs')
        chains=[_typed(results.TracebackChain,objects[ref['artifact_id']]) for ref in record['traceback_artifacts']];typed['tracebacks']=[x.metadata_record() for x in chains]
        transitions=[]
        for chain in chains:
            C.require(chain.scope==capsule.scope and chain.capsule_id==capsule.capsule_id and chain.mission_id==capsule.mission_id,'foreign traceback')
            transitions.extend(chain.transitions)
        _rows(record['rationale_texts'],{'transition_id','text'},'transition_id','rationales')
        C.require([r['transition_id'] for r in record['rationale_texts']]==[x.transition_id for x in transitions],'complete rationale roster differs')
        for row,transition in zip(record['rationale_texts'],transitions):C.require(row['text']['file_sha256']==transition.rationale_digest,'rationale digest differs')
    elif family=='outcome':
        for key in ('observation_id','observer_ref'):C.identifier(record[key])
        C.require(record['observation_id']==record['entity_id'] and record['content_version']==1,'observation identity/rewrite differs')
        C.timestamp(record['observed_at']);known(record['attempt_ref'],'attempt')
        _rows(record['postcondition_reports'],{'condition_id','reported_state','explanation'},'condition_id','condition reports')
        for row in record['postcondition_reports']:C.require(row['reported_state'] in ('met','unmet','unknown'),'unknown supplied reported state')
        _evidence(record['evidence_refs']);_events(record['source_event_refs']);_knowns(record['supersedes_observation_refs'],'outcome')
    elif family=='workspace':
        projection=_typed(workspace.WorkspaceProjection,objects[record['projection_artifact']['artifact_id']])
        decision=_typed(attention.AttentionDecision,objects[record['attention_artifact']['artifact_id']]);projection.validate(attention_decision=decision)
        C.require(projection.scope==C.scope(record['scope']) and projection.projection_id==record['entity_id'] and projection.audience in record['audiences'],'workspace identity/audience differs')
        typed.update(projection=projection.metadata_record(),attention=decision.metadata_record(),visible=[])
        C.require([r.get('item_id') for r in record['projected_items']]==[i.item_id for i in projection.items],'workspace item order differs')
        for row,item in zip(record['projected_items'],projection.items):
            C.exact(row,{'item_id','projected_body','projection_format','linked_content_refs'},'projected item')
            C.require(row['projection_format']=='canonical-json-sha256-v1' and row['projected_body']['file_sha256']==item.projected_metadata_digest,'projected body digest/profile differs')
            body=objects[row['projected_body']['artifact_id']];C.exact(body,{'schema','audience','item_id','display_title','display_text','visible_fields'},'visible body')
            C.require(body['schema']=='mfm-workspace-visible-item-body-v1' and body['audience']==projection.audience and body['item_id']==item.item_id,'visible body identity/audience differs')
            for key in ('display_title','display_text'):C.require(type(body[key]) is str and '\x00' not in body[key],'visible string differs')
            C.require(type(body['visible_fields']) is dict,'visible fields differ');typed['visible'].append(deepcopy(body))
            _knowns(row['linked_content_refs'],None)
            for ref in row['linked_content_refs']:C.require(ref['content']['family'] in ('mission','result'),'unsupported workspace content link')
        overrides=[_typed(attention.HostWorkspaceOverride,objects[r['artifact_id']]) for r in record['override_artifacts']]
        typed['overrides']=[x.metadata_record() for x in overrides]
        _rows(record['override_reason_texts'],{'override_id','text'},'override_id','override reasons')
        C.require([r['override_id'] for r in record['override_reason_texts']]==[x.override_id for x in overrides],'override reason roster differs')
        for row,override in zip(record['override_reason_texts'],overrides):
            C.require(override.scope==projection.scope and row['text']['file_sha256']==override.reason_digest,'override scope/reason differs')
    return typed


def _dependencies(record):
    rows=[]
    def walk(value):
        if type(value) is dict:
            if set(value)=={'content','commit'}:known(value);rows.append(deepcopy(value))
            else:
                for child in value.values():walk(child)
        elif type(value) is list:
            for child in value:walk(child)
    walk(record);return rows


def _validate(metadata_bytes,parts,*,refs,scope,namespace,contract_binding,contract_binding_sha256,full):
    C.pin(contract_binding,contract_binding_sha256,'mfm-task-content-binding-v1');C.require(contract_binding==task_content_binding(),'normative task binding differs')
    record=C.parse_json(_bytes(metadata_bytes,refs['content-metadata']));family=common(record,scope,namespace)
    C.require(family!='uncertainty','uncertainty requires its distinct validator')
    rows=owned_artifacts(record);expected={'content-metadata':refs['content-metadata'],**{r['part_name']:r['artifact'] for r in rows}}
    C.require(refs==expected,'complete content artifact inventory differs')
    required={r['part_name'] for r in rows if full or r['artifact']['media_type']=='application/json'}
    C.require(type(parts) is dict and set(parts)==required,'complete supplied original parts differ')
    objects={};texts={};originals={'content-metadata':{'artifact':deepcopy(refs['content-metadata']),'base64':b64encode(metadata_bytes).decode('ascii')}}
    for row in rows:
        name=row['part_name'];ref=row['artifact']
        if name not in parts:continue
        raw=_bytes(parts[name],ref);originals[name]={'artifact':deepcopy(ref),'base64':b64encode(raw).decode('ascii')}
        if ref['media_type']=='application/json':objects[name]=C.parse_json(raw)
        elif family=='result' and any(x['availability']=='present' and x['artifact']['artifact_id']==name for x in record['output_artifacts']):
            pass  # Arbitrary actor output is opaque; textual interpretation is not inferred.
        else:texts[name]=_text(raw)
    typed=_family(record,family,objects)
    return {'family':family,'record':record,'typed':typed,'texts':texts,'artifacts':deepcopy(refs),'originals':originals,
        'dependencies':_dependencies(record),'body_bytes_verified':full,'causal_dependencies_verified':False,'authenticated_service_claimed':False}


def task_content_metadata(metadata_bytes,metadata_parts,**kwargs):return _validate(metadata_bytes,metadata_parts,full=False,**kwargs)
def task_originals(metadata_bytes,parts,**kwargs):return _validate(metadata_bytes,parts,full=True,**kwargs)


def task_successor(previous,current):
    a=previous['record'];b=current['record']
    C.require(previous['family']==current['family'],'cross-family successor')
    for key in ('entity_id','scope','namespace','created_at','subject_ref'):C.require(a[key]==b[key],'content immutable identity differs')
    C.require(b['content_version']==a['content_version']+1 and b['previous_content_sha256']==a['record_sha256'] and b['updated_at']>a['updated_at'],'content successor counter/predecessor differs')
    family=previous['family']
    if family=='mission':
        ga=_typed(mission.MissionGraphSnapshot,previous['typed']['graph']);gb=_typed(mission.MissionGraphSnapshot,current['typed']['graph'])
        C.require(ga.mission==gb.mission and gb.revision==ga.revision+1 and gb.previous_snapshot_sha256==ga.snapshot_sha256,'graph historical successor differs')
        old={n.node_id:n for n in ga.nodes}
        for node in gb.nodes:
            if node.node_id in old and node!=old[node.node_id]:old[node.node_id].assert_valid_successor(node)
    elif family=='attempt':
        for key in ('attempt_id','mission_id','node_id','mission_content_ref','actor_ref','instruction','expected_postconditions'):C.require(a[key]==b[key],'attempt action redefinition')
        edges={'planned':{'planned','attempted','cancelled'},'attempted':{'attempted','cancelled'},'cancelled':set()}
        C.require(b['phase'] in edges[a['phase']],'attempt transition differs')
        C.require(b['source_event_refs'][:len(a['source_event_refs'])]==a['source_event_refs'],'attempt event ancestry removed')
    elif family=='result':C.require(previous['typed']['capsule']==current['typed']['capsule'],'immutable capsule rewrite')
    elif family in ('outcome','workspace'):C.require(False,'immutable observation/projection requires new identity')


def validate_dependency_originals(original,known_originals):
    """Pure cross-original agreement. Actual selected-prefix ledger proof belongs to R3."""
    record=original['record'];resolved={}
    for ref in original['dependencies']:
        value=ref['content'];key=(value['family'],value['entity_id'],value['content_version'])
        C.require(key in known_originals,'missing exact dependency original');other=known_originals[key]
        C.require(other['record']['record_sha256']==value['record_sha256'],'dependency original version differs');resolved[key]=other
    def get(ref):v=ref['content'];return resolved[(v['family'],v['entity_id'],v['content_version'])]
    result={'causal_dependencies_verified':False}
    family=original['family']
    if family=='attempt':
        source=get(record['mission_content_ref']);graph=source['typed']['graph'];nodes=source['record']['node_contents']
        C.require(graph['mission']['mission_id']==record['mission_id'] and record['node_id'] in [n['node_id'] for n in nodes],'attempt mission/node differs')
        steps=next(n for n in nodes if n['node_id']==record['node_id'])['plan_steps'];C.require(set(record['related_plan_step_ids'])<={s['step_id'] for s in steps},'foreign/later plan step')
    elif family=='outcome':
        source=get(record['attempt_ref'])['record'];C.require(record['observer_ref']!=source['actor_ref'],'actor assertion is not independent observer input')
        conditions=[r['condition_id'] for r in source['expected_postconditions']];reported=[r['condition_id'] for r in record['postcondition_reports']]
        C.require(reported==[k for k in conditions if k in reported],'foreign/reordered condition report')
        result['condition_availability']=[{'condition_id':k,'availability':'present' if k in reported else 'not_supplied'} for k in conditions]
    elif family=='result':
        source=get(record['attempt_ref'])['record'];capsule=original['typed']['capsule']
        C.require(source['mission_id']==capsule['mission_id'] and source['node_id']==capsule['node_id'] and source['mission_content_ref']==record['mission_content_ref'],'result rebased mission/attempt')
        for ref in record['outcome_observation_refs']:C.require(get(ref)['record']['attempt_ref']==record['attempt_ref'],'foreign outcome attachment')
    elif family=='mission':
        for node in record['node_contents']:
            for ref in node['action_attempts']:
                source=get(ref)['record'];C.require(source['mission_id']==record['entity_id'] and source['node_id']==node['node_id'],'foreign mission attempt')
    elif family=='workspace':
        for row,item in zip(record['projected_items'],original['typed']['projection']['items']):
            for ref in row['linked_content_refs']:
                source=get(ref)
                if source['family']=='mission':
                    node=next((n for n in source['typed']['graph']['nodes'] if n['node_id']==item['reference_id']),None)
                    C.require(node is not None and node['node_sha256']==item['state_digest'],'workspace mission state differs')
                else:C.require(source['record']['entity_id']==item['reference_id'] and source['typed']['capsule']['capsule_sha256']==item['state_digest'],'workspace result state differs')
    return result
