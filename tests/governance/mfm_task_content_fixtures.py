"""Public fictional typed originals; structural refs earn knowledge only in world fixtures."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

from cognitive_kernel import mission as G, results as R, attention as A, workspace as W
from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel.contracts import ProductHostScope, ProvenanceReference

TIME='2026-08-01T10:00:00.000000Z'
LATER='2026-08-01T11:00:00.000000Z'
from tests.governance.mfm_world_fixtures import SCOPE,NAMESPACE
PROVENANCE=ProvenanceReference.create(provenance_type='derived_inference',source_reference_ids=('fictional-input',),
    derivation_activity_id='fictional-builder',responsible_component='fictional-input')


def seal(value,key='record_sha256'):
    return {**deepcopy(value),key:canonical_sha256(value)}


def content_ref(record):
    return {k:record[k] for k in ('entity_id','content_version','record_sha256')}|{'family':record['schema'].split('-')[1]}


def structural_known(family,record,position=1):
    """Shape-only Task1 reference. World fixture replaces with actual ledger receipt."""
    return {'content':{'family':family,**{k:record[k] for k in ('entity_id','content_version','record_sha256')}},
        'commit':{'event_id':f'structural-{family}-{position}','event_sha256':'1'*64,'known_at_sequence':position,'entry_sha256':'2'*64}}


class TaskOriginals:
    def __init__(self):
        self.parts={};self.records={};self.refs={}
        title=self.text('mission-title','Café "A" \\ 😀 ana ana')
        node_title=self.text('node-title','Check the fictional lamp')
        self.mission=G.Mission.create(mission_key='lamp',scope=SCOPE,created_at=TIME,title_digest=title['file_sha256'],provenance=PROVENANCE)
        self.root=G.MissionNode.create(node_key='root',mission_id=self.mission.mission_id,scope=SCOPE,node_type='mission',
            status='active',execution_state='running',visibility_state='foreground',created_at=TIME,updated_at=TIME,
            title_digest=node_title['file_sha256'],provenance=PROVENANCE)
        self.graph=G.MissionGraphSnapshot.create(mission=self.mission,root_node_id=self.root.node_id,nodes=(self.root,),edges=(),revision=1,snapshot_at=TIME)
        graph=self.json('graph',self.graph.metadata_record())
        self.add('mission',self.envelope('mission',self.mission.mission_id)|{'graph_artifact':graph,'title_projection':'raw-utf8-sha256-v1',
            'mission_title':title,'node_contents':[{'node_id':self.root.node_id,'node_sha256':self.root.node_sha256,'title':node_title,
                'goals':[{'item_id':'goal-1','text':self.text('goal-1','Lamp produces light')}],'constraints':[],
                'plan_steps':[{'step_id':'step-1','text':self.text('step-1','Press the switch'),'depends_on_step_ids':[]}],'action_attempts':[]}]})
        self.add('attempt',self.envelope('attempt','attempt-1')|{'attempt_id':'attempt-1','mission_id':self.mission.mission_id,
            'node_id':self.root.node_id,'mission_content_ref':structural_known('mission',self.records['mission']),
            'actor_ref':'fictional-actor','phase':'planned','instruction':self.text('instruction','Press the switch'),
            'expected_postconditions':[{'condition_id':'light','text':self.text('condition-light','Lamp is lit')},
                {'condition_id':'heat','text':self.text('condition-heat','Lamp remains cool')}],
            'related_plan_step_ids':['step-1'],'source_event_refs':[]})
        summary=self.text('result-summary','Actor reports success; this is not an observation.')
        self.capsule=R.ResultCapsule.create(result_key='actor-result',scope=SCOPE,mission_id=self.mission.mission_id,node_id=self.root.node_id,
            produced_at=TIME,status='succeeded',summary_digest=summary['file_sha256'],output_reference_ids=('output-1',),provenance=PROVENANCE)
        rationale=self.text('rationale','Actor declared completion.')
        transition=R.TracebackTransition.create(transition_key='stop',scope=SCOPE,capsule_id=self.capsule.capsule_id,
            mission_id=self.mission.mission_id,sequence=0,source_node_id=self.root.node_id,target_node_id=None,
            action='stop_at_mission_root',created_at=TIME,rationale_digest=rationale['file_sha256'])
        chain=R.TracebackChain.create(chain_key='actor-chain',scope=SCOPE,capsule_id=self.capsule.capsule_id,mission_id=self.mission.mission_id,
            created_at=TIME,status='applied',transitions=(transition,))
        self.add('result',self.envelope('result',self.capsule.capsule_id)|{'capsule_artifact':self.json('capsule',self.capsule.metadata_record()),
            'summary_projection':'raw-utf8-sha256-v1','summary':summary,'mission_content_ref':structural_known('mission',self.records['mission']),
            'attempt_ref':structural_known('attempt',self.records['attempt'],2),'output_artifacts':[{'ref_id':'output-1','availability':'present',
                'artifact':self.text('output-1','Actor output')}],'outcome_observation_refs':[],
            'traceback_artifacts':[self.json('traceback',chain.metadata_record())],'rationale_texts':[{'transition_id':transition.transition_id,'text':rationale}],
            'evidence_refs':[],'source_event_refs':[]})
        self.add('outcome',self.envelope('outcome','observation-1')|{'observation_id':'observation-1','attempt_ref':structural_known('attempt',self.records['attempt'],2),
            'observer_ref':'fictional-observer','observed_at':TIME,'report':self.text('observer-report','The lamp stayed dark.'),
            'postcondition_reports':[{'condition_id':'light','reported_state':'unmet','explanation':self.text('unmet-explanation','No light was visible.')}],
            'evidence_refs':[],'source_event_refs':[],'supersedes_observation_refs':[]})
        self.workspace('host')

    def envelope(self,family,entity):
        names={'mission':'mfm-mission-original-content-v1','result':'mfm-result-original-content-v1','attempt':'mfm-task-attempt-input-v1',
            'outcome':'mfm-task-outcome-observation-input-v1','workspace':'mfm-workspace-original-content-v1','uncertainty':'mfm-uncertainty-original-input-v1'}
        return {'schema':names[family],'profile':'supplied-claim-episode-task-uncertainty-prefix-v1','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'entity_id':entity,'content_version':1,'previous_content_sha256':None,'created_at':TIME,'updated_at':TIME,'subject_ref':None,
            'valid_from':None,'valid_to':None,'provenance':PROVENANCE.metadata_record(),'retention_class':'active_project','storage_tier':'ledger',
            'deletion_lineage':[],'policy_bindings':[],'audiences':['host','non_host']}

    def blob(self,name,raw,media):
        assert name not in self.parts or self.parts[name]==raw
        self.parts[name]=raw
        return {'artifact_id':name,'bytes':len(raw),'file_sha256':sha256(raw).hexdigest(),'media_type':media}

    def text(self,name,text):return self.blob(name,text.encode('utf-8'),'application/octet-stream')
    def json(self,name,record):return self.blob(name,canonical_json_bytes(record),'application/json')

    def add(self,family,record):
        record=seal(record);self.records[family]=record;return record

    def workspace(self,audience):
        body={'schema':'mfm-workspace-visible-item-body-v1','audience':audience,'item_id':'pending','display_title':'Lamp',
            'display_text':'Actor report only','visible_fields':{'null':None,'bool':True,'int':1,'float':0.5,'array':[], 'object':{},'text':'ana ana'}}
        entry=A.AttentionRankEntry.create(entry_key='lamp',scope=SCOPE,reference_id=self.root.node_id,subject_type='mission_node',
            mission_id=self.mission.mission_id,node_id=self.root.node_id,capsule_id=None,observed_at=TIME,state_digest=self.root.node_sha256,
            priority_class='host_engaged',rank=1,score=.5,interruption_cost=.1,protected_interrupt_reason=None,host_override='none',
            selected=True,reason_codes=('host_engaged',),suppression_reason=None,policy_bindings=())
        non_host=audience=='non_host';entries=(entry,);overrides=[];reasons=[]
        if non_host:
            omitted=A.AttentionRankEntry.create(entry_key='restricted',scope=SCOPE,reference_id='restricted-node',subject_type='mission_node',
                mission_id=self.mission.mission_id,node_id='restricted-node',capsule_id=None,observed_at=TIME,state_digest='7'*64,
                priority_class='low_attention',rank=2,score=.1,interruption_cost=.2,protected_interrupt_reason=None,host_override='suppressed',
                selected=False,reason_codes=('host_suppressed',),suppression_reason='restricted',policy_bindings=())
            entries=(entry,omitted);reason=self.text('override-secret-reason','NEVER-MODEL-PRIVATE fictional reason')
            override=A.HostWorkspaceOverride.create(override_key='restrict',scope=SCOPE,command='background',status='applied',issued_at=TIME,
                reason_digest=reason['file_sha256'],provenance=PROVENANCE,target_reference_id='restricted-node')
            overrides=[self.json('override-private',override.metadata_record())];reasons=[{'override_id':override.override_id,'text':reason}]
        decision=A.AttentionDecision.create(decision_key='lamp-'+audience,scope=SCOPE,decided_at=TIME,visibility_limit=1,interruption_preference='allow',
            focus_mode='automatic',layout_stability_weight=.5,entries=entries,provenance=PROVENANCE)
        # Item identity excludes projected digest, allowing an authored body to bind its actual ID.
        item=W.WorkspaceItemProjection.create(item_key='lamp',scope=SCOPE,reference_id=self.root.node_id,item_type='mission_node',
            mission_id=self.mission.mission_id,node_id=self.root.node_id,capsule_id=None,attention_entry_id=entry.entry_id,attention_rank=1,
            role='primary',privacy_class='sensitive' if non_host else 'ordinary',redaction_state='metadata_only' if non_host else 'none',
            state_digest=self.root.node_sha256,projected_metadata_digest='1'*64)
        body['item_id']=item.item_id
        item=replace(item,projected_metadata_digest=canonical_sha256(body),item_sha256='0'*64)
        item=replace(item,item_sha256=canonical_sha256(item.material_record()));item.validate()
        layout=W.WorkspaceLayout.create(layout_key='lamp',scope=SCOPE,created_at=TIME,visible_count=1,total_candidate_count=len(entries),max_visible=10,
            layout_locked=False,stability_anchor_digest='3'*64,item_order=(item.item_id,),
            omitted_reference_digests=(sha256(b'restricted-node').hexdigest(),) if non_host else ())
        projection=W.WorkspaceProjection.create(projection_key='lamp-'+audience,scope=SCOPE,attention_decision=decision,
            projected_at=TIME,audience=audience,layout=layout,items=(item,))
        return self.add('workspace',self.envelope('workspace',projection.projection_id)|{'projection_artifact':self.json('workspace-'+audience,projection.metadata_record()),
            'attention_artifact':self.json('attention-'+audience,decision.metadata_record()),'override_artifacts':overrides,
            'projected_items':[{'item_id':item.item_id,'projected_body':self.json('visible-'+audience,body),'projection_format':'canonical-json-sha256-v1',
                'linked_content_refs':[structural_known('mission',self.records['mission'])]}],'override_reason_texts':reasons})

    def arguments(self,family,record=None):
        from scripts.mfm import formation_task_content_input as T
        record=record or self.records[family];raw=canonical_json_bytes(record)
        rows=T.owned_artifacts(record)
        refs={'content-metadata':{'artifact_id':'metadata-'+family,'bytes':len(raw),'file_sha256':sha256(raw).hexdigest(),'media_type':'application/json'},
            **{r['part_name']:r['artifact'] for r in rows}}
        binding=T.task_content_binding()
        return raw,{k:self.parts[k] for k in refs if k!='content-metadata'},dict(refs=refs,scope=SCOPE,namespace=NAMESPACE,
            contract_binding=binding,contract_binding_sha256=binding['record_sha256'])

    def originals(self,family,record=None):
        from scripts.mfm import formation_task_content_input as T
        raw,parts,kwargs=self.arguments(family,record);return T.task_originals(raw,parts,**kwargs)

    def uncertainty(self,entity='uncertainty-1'):
        identity={'ref_id':entity,'kind':'uncertainty','registration_payload_sha256':'3'*64,
            'known_at_sequence':1,'known_at_entry_sha256':'4'*64,'world_record_sha256':'5'*64}
        return self.add('uncertainty',self.envelope('uncertainty',entity)|{'uncertainty_ref':entity,'uncertainty_key':'lamp-explanation',
            'identity_registration':identity,'state':'open','question':self.text('question','Why is the lamp dark?'),
            'alternatives':[{'alternative_id':'switch','text':self.text('alt-switch','Switch not pressed'),'support_refs':[],'counter_refs':[]},
                {'alternative_id':'bulb','text':self.text('alt-bulb','Bulb broken'),'support_refs':[],'counter_refs':[]},
                {'alternative_id':'wire','text':self.text('alt-wire','Wire disconnected'),'support_refs':[],'counter_refs':[]}],
            'evidence_refs':[],'resolution_observations':[]})


class TaskWorld:
    """New disclosed v3 originals backed by real stores and one real Experience chain."""
    def __init__(self,*,binding=True,future_dependency=False,legacy_only=False,longitudinal=False,audience='host',wrong_resolution_ancestry=False,content_status=None,complete_trajectory=False):
        import json,tempfile
        from pathlib import Path
        from cognitive_kernel.ledger_store import open_experience_ledger
        from cognitive_kernel.experience import ExperienceEvent
        from scripts.mfm import formation_input_content_prototype as P,formation_task_content_input as T,formation_uncertainty_input as U
        from cognitive_kernel.formation_semantics_v16 import RegisteredFormationTarget
        from tests.governance.mfm_episode_fixtures import EpisodeWorld
        base=EpisodeWorld(status='present',register_participant=False);self.base=base;self.f=TaskOriginals();self.audience=audience
        if audience!='host':self.f.workspace(audience)
        self.artifacts=deepcopy(base.artifacts);self.rows=[];self.events=[];self.content_known={};self.calls=[];self.content_rows={};self.case_cuts={}
        self.provenance=deepcopy(base.provenance);self.authority_profile=base.authority_profile;self.episode_projection_profile=base.episode_projection_profile
        with tempfile.TemporaryDirectory(prefix='mfm-task-world-') as temporary:
            root=Path(temporary);repo=root/'repository';repo.mkdir()
            store=P.InputContentPrototypeStore(root/'content',scope=SCOPE,namespace=NAMESPACE,store_key='task-world',created_at=TIME,
                family_bindings={'task':T.task_content_binding(),'uncertainty':U.uncertainty_input_binding()})
            try:
                self.input_content_profile=self.add('task-profile',store.profile_record())
                with open_experience_ledger(root/'ledger.sqlite3',scope=SCOPE,repository_root=repo,created_at=TIME) as ledger:
                    self.ledger_profile={'schema':'experience-ledger-records-v1','ledger_schema_version':'1.0.0','ledger_id':ledger.ledger_id,
                        'scope_sha256':ledger.scope_digest,'created_at':TIME,'genesis_sha256':ledger.genesis_sha256}
                    self.genesis=canonical_sha256({'schema':'mfm-world-genesis-v3','world_id':'world-a','scope':SCOPE.metadata_record(),
                        'namespace':NAMESPACE,'ledger_profile':self.ledger_profile,'authority_profile':self.authority_profile,
                        'episode_projection_profile':self.episode_projection_profile,'input_content_profile':self.input_content_profile})
                    for row in base.rows:
                        original=json.loads(base.artifacts[row['artifact']['artifact_id']]);self.append(ledger,original['payload_kind'],original['payload'])
                    target=RegisteredFormationTarget(self.f.root.node_id,'mission',SCOPE,NAMESPACE,('source-0',))
                    self.append(ledger,'registry_target',{'target':target.record(),'registration_ref':'fictional-task-target','rights_refs':['fictional-rights']})
                    workspace_target=RegisteredFormationTarget(self.f.records['workspace']['entity_id'],'workspace',SCOPE,NAMESPACE,('source-0',))
                    self.append(ledger,'registry_target',{'target':workspace_target.record(),'registration_ref':'fictional-workspace-target','rights_refs':['fictional-rights']})
                    identity_payload={'ref_id':'uncertainty-1','kind':'uncertainty','record_sha256':'d'*64,'status':'present',
                        'description':'Original uncertainty identity only','authority_ref':'fictional-uncertainty-read'}
                    identity=self.append(ledger,'registry_metadata',identity_payload)
                    self.identity={'ref_id':'uncertainty-1','kind':'uncertainty','registration_payload_sha256':canonical_sha256(identity_payload),
                        'known_at_sequence':identity['sequence'],'known_at_entry_sha256':identity['entry_receipt']['entry_sha256'],'world_record_sha256':identity['record_sha256']}
                    self.commit(ledger,store,'mission',self.f.records['mission'])
                    attempt=deepcopy(self.f.records['attempt']);attempt['mission_content_ref']=self.content_known['mission'];self.commit(ledger,store,'attempt',attempt)
                    if longitudinal:
                        attempt=deepcopy(self.content_rows['attempt']['record']);attempt.update(content_version=2,previous_content_sha256=attempt['record_sha256'],updated_at=LATER,phase='attempted')
                        self.commit(ledger,store,'attempt',attempt)
                    previous_graph=self.f.graph
                    if complete_trajectory:
                        attached=deepcopy(self.content_rows['mission']['record']);attachment_time='2026-08-01T10:30:00.000000Z'
                        attached.update(content_version=2,previous_content_sha256=attached['record_sha256'],updated_at=attachment_time)
                        previous_graph=G.MissionGraphSnapshot.create(mission=self.f.mission,root_node_id=self.f.root.node_id,nodes=(self.f.root,),edges=(),revision=2,
                            snapshot_at=attachment_time,previous_snapshot_sha256=self.f.graph.snapshot_sha256)
                        attached['graph_artifact']=self.f.json('graph-attached',previous_graph.metadata_record());attached['node_contents'][0]['action_attempts']=[self.content_known['attempt']]
                        self.commit(ledger,store,'mission',attached)
                    result=deepcopy(self.f.records['result']);result['mission_content_ref']=self.content_rows['attempt']['record']['mission_content_ref'];result['attempt_ref']=self.content_known['attempt']
                    if future_dependency:result['attempt_ref']['commit']['known_at_sequence']=999
                    self.commit(ledger,store,'result',result)
                    self.case_cuts['actor']=len(self.rows)
                    outcome=deepcopy(self.f.records['outcome']);outcome['attempt_ref']=self.content_known['attempt'];self.commit(ledger,store,'outcome',outcome)
                    self.case_cuts['contrary']=len(self.rows)
                    if complete_trajectory:
                        prior=deepcopy(self.content_known['outcome']);correction=deepcopy(outcome)
                        correction.update(entity_id='observation-2',observation_id='observation-2',observed_at=LATER,updated_at=LATER,
                            report=self.f.text('observer-report-2','A later independent observation saw light.'),supersedes_observation_refs=[prior],
                            postcondition_reports=[{'condition_id':'light','reported_state':'met','explanation':self.f.text('met-explanation','Light was visible.')},
                                {'condition_id':'heat','reported_state':'unknown','explanation':self.f.text('unknown-explanation','Temperature was not measured.')}])
                        self.commit(ledger,store,'outcome',correction)
                    workspace=deepcopy(self.f.records['workspace']);workspace['projected_items'][0]['linked_content_refs']=[self.content_known['mission']]
                    self.commit(ledger,store,'workspace',workspace)
                    if longitudinal:
                        revised=deepcopy(self.content_rows['mission']['record']);revised.update(content_version=revised['content_version']+1,previous_content_sha256=revised['record_sha256'],updated_at=LATER)
                        node=self.f.root
                        if complete_trajectory:
                            node=G.MissionNode.create(**{k:v for k,v in self.f.root.__dict__.items() if k not in ('node_id','node_sha256')}
                                |{'status':'completed','execution_state':'succeeded','updated_at':LATER})
                            revised['node_contents'][0]['node_sha256']=node.node_sha256
                        graph=G.MissionGraphSnapshot.create(mission=self.f.mission,root_node_id=node.node_id,nodes=(node,),edges=(),revision=previous_graph.revision+1,
                            snapshot_at=LATER,previous_snapshot_sha256=previous_graph.snapshot_sha256)
                        revised['graph_artifact']=self.f.json('graph-2',graph.metadata_record());revised['node_contents'][0]['action_attempts']=[self.content_known['attempt']]
                        self.commit(ledger,store,'mission',revised)
                        self.case_cuts['revised']=len(self.rows)
                        if complete_trajectory:
                            reopened=deepcopy(self.content_rows['mission']['record']);next_time='2026-08-01T12:00:00.000000Z'
                            node=G.MissionNode.create(**{k:v for k,v in node.__dict__.items() if k not in ('node_id','node_sha256')}
                                |{'status':'active','execution_state':'running','updated_at':next_time})
                            reopened.update(content_version=reopened['content_version']+1,previous_content_sha256=reopened['record_sha256'],updated_at=next_time)
                            graph3=G.MissionGraphSnapshot.create(mission=self.f.mission,root_node_id=node.node_id,nodes=(node,),edges=(),revision=graph.revision+1,
                                snapshot_at=next_time,previous_snapshot_sha256=graph.snapshot_sha256)
                            reopened['graph_artifact']=self.f.json('graph-3',graph3.metadata_record());reopened['node_contents'][0]['node_sha256']=node.node_sha256
                            self.commit(ledger,store,'mission',reopened)
                    uncertainty=self.f.uncertainty();uncertainty['identity_registration']=self.identity;self.commit(ledger,store,'uncertainty',uncertainty)
                    self.before_binding=len(self.rows)
                    if binding:self.bind_uncertainty(ledger,self.content_known['uncertainty'],'d'*64)
                    if longitudinal and binding:
                        for version,state in ((2,'resolved'),(3,'reopened')):
                            uncertainty=deepcopy(self.content_rows['uncertainty']['record'])
                            uncertainty.update(content_version=version,previous_content_sha256=uncertainty['record_sha256'],updated_at=f'2026-08-01T{10+version:02}:00:00.000000Z',state=state)
                            if state=='resolved':
                                observation=self.resolution_observation(ledger,2)
                                if wrong_resolution_ancestry:
                                    first=__import__('json').loads(self.artifacts[self.rows[0]['artifact']['artifact_id']])
                                    observation['source_event_refs']=[{'event_id':first['event']['event_id'],'event_sha256':first['event']['event_sha256'],
                                        'known_at_sequence':1,'entry_sha256':first['entry_receipt']['entry_sha256']}]
                                uncertainty['resolution_observations'].append(observation)
                            previous=self.content_known['uncertainty']['content']['record_sha256']
                            self.commit(ledger,store,'uncertainty',uncertainty);self.bind_uncertainty(ledger,self.content_known['uncertainty'],previous)
                            self.case_cuts[state]=len(self.rows)
                    for family,status in (content_status or {}).items():
                        original=self.content_rows[family]['record']
                        gov=seal({'schema':'mfm-world-supplied-governance-v1','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
                            'decision_id':'content-'+family+'-'+status,'issuer_id':'fictional-issuer','authority_basis_ref':'fictional-basis','decision':status,
                            'subject_ref':original['entity_id'],'expected_current_claim_version_id':None,'request_digest':None,'source_refs':[],'origin':'research_explicit_records'})
                        self.append(ledger,'input_content_visibility',{'family':family,'entity_id':original['entity_id'],'expected_record_sha256':original['record_sha256'],
                            'status':status,'governance_ref':self.add('visibility-'+family,gov)})
                    assert ledger.verify_integrity().entry_count==len(self.rows)
            finally:store.close()
        self.world=seal({'schema':'mfm-longitudinal-world-v3','world_id':'world-a','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'provenance':self.provenance,'ledger_profile':self.ledger_profile,'authority_profile':self.authority_profile,
            'episode_projection_profile':self.episode_projection_profile,'input_content_profile':self.input_content_profile,
            'branches':[{'branch_id':'main','parent_cut':None,'records':self.rows}]},'world_sha256')

    def add(self,name,value):
        raw=value if type(value) is bytes else canonical_json_bytes(value)
        assert name not in self.artifacts or self.artifacts[name]==raw
        self.artifacts[name]=raw
        return {'artifact_id':name,'bytes':len(raw),'file_sha256':sha256(raw).hexdigest(),'media_type':'application/octet-stream' if type(value) is bytes else 'application/json'}

    def append(self,ledger,kind,payload,*,parents=None,occurred=TIME):
        from cognitive_kernel.experience import ExperienceEvent
        n=len(self.rows)+1
        event=ExperienceEvent.create(event_type=kind.replace('_','-'),scope=SCOPE,occurred_at=occurred,
            content_digest=payload['source_artifact']['file_sha256'] if kind=='observation' else canonical_sha256(payload),provenance=PROVENANCE,
            retention_class='ordinary_experience',storage_tier='hot',payload_reference=f'task-payload-{n}',
            parent_event_ids=tuple(parents) if parents is not None else (() if not self.events else (self.events[-1]['event_id'],)))
        transaction=ledger.append_event(event,committed_at=TIME);self.events.append(event.metadata_record())
        record=seal({'schema':'mfm-world-record-v3','world_id':'world-a','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,'sequence':n,
            'previous_record_sha256':self.genesis if n==1 else self.rows[-1]['record_sha256'],'event':event.metadata_record(),
            'entry_receipt':transaction.entries[0].record(),'payload_kind':kind,'payload':deepcopy(payload)})
        self.rows.append({'sequence':n,'record_sha256':record['record_sha256'],'artifact':self.add(f'task-world-record-{n}',record)})
        return record

    def commit(self,ledger,store,family,record):
        from scripts.mfm import formation_task_content_input as T
        record=seal({k:v for k,v in record.items() if k!='record_sha256'});raw=canonical_json_bytes(record)
        parts={r['part_name']:self.f.parts[r['part_name']] for r in T.owned_artifacts(record)}
        receipt=store.commit(family,raw,parts,operation_id=f'task-{family}-{record["entity_id"]}-{record["content_version"]}',
            expected_previous_content_sha256=record['previous_content_sha256'],committed_at=TIME)
        for part in receipt['original_artifacts']:
            name=part['part_name'];ref=part['artifact'];self.artifacts[ref['artifact_id']]=raw if name=='content-metadata' else parts[name]
        payload={'family':family,'entity_id':record['entity_id'],'content_version':record['content_version'],'record_sha256':record['record_sha256'],
            'content_artifact':receipt['original_artifacts'][0]['artifact'],'commit_artifact':self.add('content-commit-'+family+'-'+record['entity_id']+'-'+str(record['content_version']),receipt),'profile_artifact':self.input_content_profile}
        ancestors=[]
        def collect(value):
            if type(value) is dict:
                if set(value)=={'event_id','event_sha256','known_at_sequence','entry_sha256'}:ancestors.append(value['event_id'])
                else:
                    for child in value.values():collect(child)
            elif type(value) is list:
                for child in value:collect(child)
        collect(record)
        world=self.append(ledger,'input_content_commit',payload,parents=tuple(dict.fromkeys([self.events[-1]['event_id'],*ancestors])))
        known={'content':{k:payload[k] for k in ('family','entity_id','content_version','record_sha256')},
            'commit':{'event_id':world['event']['event_id'],'event_sha256':world['event']['event_sha256'],
                'known_at_sequence':world['sequence'],'entry_sha256':world['entry_receipt']['entry_sha256']}}
        self.content_known[family]=known;self.content_rows[family]={'record':record,'receipt':receipt,'payload':payload}
        return known

    def bind_uncertainty(self,ledger,content,expected):
        payload={'ref_id':'uncertainty-1','identity_registration':self.identity,'content_ref':deepcopy(content),
            'expected_registry_record_sha256':expected,'description':'Why the fictional lamp remained dark','rights_refs':['fictional-rights']}
        gov=seal({'schema':'mfm-world-supplied-governance-v1','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'decision_id':'uncertainty-binding','issuer_id':'fictional-issuer','authority_basis_ref':'fictional-basis','decision':'accepted',
            'subject_ref':'uncertainty-1','expected_current_claim_version_id':None,'request_digest':canonical_sha256(payload),
            'source_refs':[],'origin':'research_explicit_records'})
        payload['governance_ref']=self.add('uncertainty-bind-governance-'+str(content['content']['content_version']),gov)
        identity_event=self.events[self.identity['known_at_sequence']-1]['event_id']
        self.append(ledger,'uncertainty_material_binding',payload,parents=(identity_event,content['commit']['event_id']))

    def resolution_observation(self,ledger,n):
        from cognitive_kernel.formation_contracts import FormationEvidenceRef
        from cognitive_kernel.formation_sources import RegisteredFormationSource
        source=self.f.text('resolution-source-'+str(n),'Independent observer saw the disconnected wire.')
        self.artifacts[source['artifact_id']]=self.f.parts[source['artifact_id']]
        evidence=FormationEvidenceRef(ref_id='resolution-'+str(n),scope=SCOPE,authority_namespace_id=NAMESPACE,
            content_digest=source['file_sha256'],role='hypothetical_training',modality='text',subject_ref='lamp',
            speaker_ref='fictional-observer',observed_at=TIME,parent_refs=())
        registration=RegisteredFormationSource.create(evidence=evidence,object_ref=source['artifact_id'])
        previous=self.events[-1];descriptor=self.rows[-1]
        ancestry={'event_id':previous['event_id'],'event_sha256':previous['event_sha256'],'known_at_sequence':descriptor['sequence'],
            'entry_sha256':__import__('json').loads(self.artifacts[descriptor['artifact']['artifact_id']])['entry_receipt']['entry_sha256']}
        world=self.append(ledger,'observation',{'registration':{'evidence':evidence.metadata_record(),'object_ref':registration.object_ref,'registration_sha256':registration.registration_sha256},
            'source_artifact':source,'source_rights_ref':'fictional-rights','observed_at':TIME,'speaker_ref':'fictional-observer','subject_ref':'lamp','epistemic_role':'hypothetical_training','modality':'text','locator':None})
        return {'observation_ref':{'ref_id':evidence.ref_id,'registration_sha256':registration.registration_sha256,
            'known_at_entry_sha256':world['entry_receipt']['entry_sha256'],'artifact':source,'availability':'present'},'selected_alternative_ids':['wire'],
            'report':self.f.text('resolution-report-'+str(n),'Observer attributes darkness to the disconnected wire.'),'source_event_refs':[ancestry]}

    def cut(self,n=None):
        n=len(self.rows) if n is None else n
        return seal({'schema':'mfm-world-cut-v1','world_id':'world-a','branch_id':'main','sequence':n,'head_sha256':self.genesis if n==0 else self.rows[n-1]['record_sha256']})

    def read(self,ref):self.calls.append(ref['artifact_id']);return self.artifacts[ref['artifact_id']]

    def verification_boundary(self):
        return seal({'scope':SCOPE.metadata_record(),'namespace':NAMESPACE,'purpose':'formation_input_verification','read_service_id':'fictional-proof-reader',
            'as_of':TIME,'policy_sha256':'b'*64,'rights_receipt_refs':['fictional-rights'],'lifecycle_receipt_ref':'fictional-current-life'})


class VerificationAdapter:
    def __init__(self,world):self.world=world;self.boundary=world.verification_boundary();self.calls=[];self.after=None
    def verification_boundary_record(self):return deepcopy(self.boundary)
    def read_verification_artifact(self,ref):
        self.calls.append(ref['artifact_id']);raw=self.world.artifacts[ref['artifact_id']]
        if self.after:self.after(self,ref)
        return raw
