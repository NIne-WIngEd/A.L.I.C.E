"""Original three-part Episodes from actual reversible stores, public fictional."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import tempfile
from pathlib import Path

from cognitive_kernel.canonical import canonical_json_bytes,canonical_sha256
from cognitive_kernel.projection_contracts import EpisodeRecord
from cognitive_kernel.projection_prototype import ProjectionPrototypeProfile,open_projection_prototype
from scripts.mfm import formation_caller_input as F,formation_caller_meaning as M
from tests.governance.mfm_world_fixtures import World,ReadAdapter,SCOPE,NAMESPACE,TIME,env,seal


class Episodes:
    def __init__(self,revision='revised',*,predecessor='episode-1',revision_generation=2,evidence='source-0',claim_version='version-1',alternatives=False):
        self.world=World();self.artifacts={};self.calls=[];self.transitions=[]
        with tempfile.TemporaryDirectory(prefix='mfm-episode-fixture-') as temporary:
            root=Path(temporary);repo=root/'repository';repo.mkdir()
            profile=ProjectionPrototypeProfile.create(scope=SCOPE,authority_namespace_id=NAMESPACE,store_id='episode-fixture')
            self.profile=profile.metadata_record()
            with open_projection_prototype(root/'private/episodes.sqlite3',profile=profile,repository_root=repo) as store:
                for n in ((1,2,3) if alternatives else (1,2)):
                    eid=f'episode-{n}'
                    full={'events':[{'source_ref':'source-0','text':'Café "A" \\ 😀 ana ana'}],
                        'interpretation':'initial' if n==1 else revision, 'count':n,'number':0.5,'flag':True,'missing':None,
                        'nested':{'literal/~':['ana','ana']}}
                    summary={'summary':'Initial mode choice.' if n==1 else 'Later mode revision.','uncertain':False}
                    episode=EpisodeRecord.create(envelope=env(eid,'episode','registered_projection',sources=(evidence,claim_version)),
                        episode_id=eid,episode_kind='interaction',episode_state='accepted',member_evidence_ids=(evidence,),
                        member_claim_version_ids=(claim_version,),member_candidate_ids=('historical-candidate-unsupplied',),
                        mission_node_ids=('mission-not-supplied',),participant_ids=('owner',),valid_from=TIME,valid_to=None,
                        formed_at=TIME,formation_component_id='fictional-episode-builder',formation_version='1.0.0',
                        summary_content_digest=canonical_sha256(summary),full_content_digest=canonical_sha256(full),confidence=0.8,
                        supersedes_episode_id=None if n==1 else predecessor,generation=1 if n==1 else revision_generation)
                    digest=canonical_sha256({'fictional-invocation':n})
                    receipt=store.append_episode(episode,full_content=full,summary_content=summary,
                        idempotency_namespace='episode-fixture',idempotency_key=eid,request_digest=digest)
                    loaded=store.get_episode(eid)
                    assert loaded=={'episode':episode.metadata_record(),'full_content':full,'summary_content':summary}
                    refs={key:self.add(f'{eid}-{key}',value) for key,value in {**loaded,
                        'append_receipt':receipt.metadata_record(),'projection_profile':self.profile}.items()}
                    invocation=seal({'schema':'mfm-episode-append-invocation-v1','episode_sha256':episode.episode_sha256,
                        'full_content_ref':refs['full_content'],'summary_content_ref':refs['summary_content'],
                        'profile_sha256':profile.profile_sha256,'idempotency_namespace':'episode-fixture',
                        'idempotency_key':eid,'request_digest':digest})
                    refs['append_invocation']=self.add(f'{eid}-invocation',invocation)
                    self.transitions.append(refs)
                assert store.verify_integrity().valid

    def add(self,key,value):
        raw=value if type(value) is bytes else canonical_json_bytes(value)
        assert key not in self.artifacts or self.artifacts[key]==raw
        self.artifacts[key]=raw
        return {'artifact_id':key,'bytes':len(raw),'file_sha256':sha256(raw).hexdigest(),
                'media_type':'application/json' if type(value) is not bytes else 'application/octet-stream'}

    def raws(self,n=1):
        refs=self.transitions[n-1]
        return tuple(self.artifacts[refs[k]['artifact_id']] for k in ('episode','full_content','summary_content','append_receipt','append_invocation','projection_profile'))

    def arguments(self,n=1):
        return {'refs':deepcopy(self.transitions[n-1]),'scope':SCOPE,'namespace':NAMESPACE}

    def meaning_inputs(self,n=1,status='present'):
        import json
        from scripts.mfm import formation_world_contracts as C,formation_world_replay_v2 as R
        causal=EpisodeWorld(register_participant=False)
        cut=causal.cut(3 if n==1 else 6);binding=R.world_contract_binding_v2()
        view=R.replay_world_prefix_v2(causal.world,world_sha256=causal.world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],
            read_artifact=causal.read,limits=C.ConstructionLimits(40,100000,500000),contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
        refs=self.transitions[n-1];episode=json.loads(self.raws(n)[0]);eid=episode['episode_id']
        context=self.world.context(('source-0',))
        from cognitive_kernel.formation_semantics_v16 import context_v16_from_record
        registry={'schema':F.REGISTRY_SCHEMA,'host_scope':SCOPE.metadata_record(),'namespace':NAMESPACE,'generation':'episode-read-1',
            'records':[{'ref_id':eid,'kind':'episode','record_sha256':episode['episode_sha256'],'status':status,'description':''}]}
        registry['records_sha256']=canonical_sha256(registry['records']);registry['registry_sha256']=canonical_sha256(registry)
        caller=F.caller_input_record(context_v16_from_record(context),[('source-0',self.world.artifacts['original-source-0'])],registry)
        claim_snapshot=seal({'schema':M.SNAPSHOT_SCHEMA,'scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'generation':'episode-read-1','authority_generation':1,'as_of':TIME,'source_position':view['authority_source_position'],
            'read_service_id':'fictional-current-read','read_receipt_sha256':'c'*64,'policy_sha256':'d'*64})
        historical=next(r for r in view['episodes'] if r['ref_id']==eid)
        known={k:deepcopy(historical[k]) for k in ('ref_id','known_at_sequence','known_at_entry_sha256','member_resolution','member_registrations')}
        known.update(episode_sha256=episode['episode_sha256'],generation=n)
        episode_snapshot=seal({'schema':'mfm-episode-semantic-snapshot-v1','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'projection_profile_sha256':self.profile['profile_sha256'],'cut':cut,
            'episodes':[known],'historical_prefix_proof':view['historical_prefix_proof'],
            'read_service_id':'fictional-current-read','read_receipt_sha256':'c'*64,'policy_sha256':'d'*64})
        return caller,claim_snapshot,episode_snapshot


def entry(slot,family,availability='present',body=None,**extra):
    raw=None if body is None else canonical_json_bytes(body)
    from base64 import b64encode
    return {'slot':slot,'family':family,'availability':availability,'body_base64':None if raw is None else b64encode(raw).decode(),
            'body_bytes':None if raw is None else len(raw),'body_sha256':None if raw is None else sha256(raw).hexdigest(),**extra}


class EpisodeWorld(World):
    """One actual new Experience chain, never filtered/re-authored old receipts."""
    def __init__(self,revision='revised',status='withheld',confirmation=False,register_participant=True,
                 late_participant=False,registered_mission=False,**episode_options):
        import json
        from cognitive_kernel.ledger_store import open_experience_ledger
        from cognitive_kernel.formation_semantics_v16 import RegisteredFormationTarget
        base=World();episodes=Episodes(revision,**episode_options)
        self.artifacts={**base.artifacts,**episodes.artifacts};self.rows=[];self.calls=[];self.events=[];self.later_prefix=''
        self.registrations=base.registrations;self.provenance=base.provenance;self.authority_profile=base.authority_profile
        self.episode_projection_profile=self.add('world-episode-profile',episodes.profile)
        self.episodes=episodes;self.target=RegisteredFormationTarget('owner','entity',SCOPE,NAMESPACE,('source-0',))
        with tempfile.TemporaryDirectory(prefix='mfm-episode-world-') as temporary:
            root=Path(temporary);repo=root/'repository';repo.mkdir()
            with open_experience_ledger(root/'private/ledger.sqlite3',scope=SCOPE,repository_root=repo,created_at=TIME) as ledger:
                self.ledger_profile={'schema':'experience-ledger-records-v1','ledger_schema_version':'1.0.0',
                    'ledger_id':ledger.ledger_id,'scope_sha256':ledger.scope_digest,'created_at':TIME,'genesis_sha256':ledger.genesis_sha256}
                self.genesis=canonical_sha256({'schema':'mfm-world-genesis-v2','world_id':'world-a','scope':SCOPE.metadata_record(),
                    'namespace':NAMESPACE,'ledger_profile':self.ledger_profile,'authority_profile':self.authority_profile,
                    'episode_projection_profile':self.episode_projection_profile})
                for n in (1,2):
                    original=json.loads(base.artifacts[f'world-record-{n}']);self.append(ledger,original['payload_kind'],original['payload'])
                target_payload={'target':self.target.record(),'registration_ref':'fictional-registration','rights_refs':['fictional-rights']}
                if register_participant and not late_participant:self.append(ledger,'registry_target',target_payload)
                if registered_mission:
                    mission=RegisteredFormationTarget('mission-not-supplied','mission',SCOPE,NAMESPACE,('source-0',))
                    self.append(ledger,'registry_target',{'target':mission.record(),'registration_ref':'fictional-mission-registration',
                        'rights_refs':['fictional-rights']})
                self.episode_commit(ledger,1)
                if late_participant:self.append(ledger,'registry_target',target_payload)
                for n in (4,5):
                    original=json.loads(base.artifacts[f'world-record-{n}']);self.append(ledger,original['payload_kind'],original['payload'])
                self.episode_commit(ledger,2,occurred='2026-08-05T04:00:00Z')
                deny=self.episode_governance('episode-deny','deny','episode-3',None,'episode-2',None)
                self.append(ledger,'episode_no_commit',{'episode_id':'episode-3','expected_predecessor_id':'episode-2','governance_ref':deny})
                meta=json.loads(episodes.raws(2)[0])
                gov=self.episode_governance('episode-visibility',status,'episode-2',meta['episode_sha256'],'episode-1',meta)
                self.append(ledger,'episode_visibility',{'episode_id':'episode-2','expected_episode_sha256':meta['episode_sha256'],
                    'status':status,'governance_ref':gov})
                if confirmation:self.episode_commit(ledger,1)
                if episode_options.get('alternatives'):self.episode_commit(ledger,3)
                assert ledger.verify_integrity().entry_count==len(self.rows)
        self.world=seal({'schema':'mfm-longitudinal-world-v2','world_id':'world-a','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'provenance':self.provenance,'ledger_profile':self.ledger_profile,'authority_profile':self.authority_profile,
            'episode_projection_profile':self.episode_projection_profile,'branches':[{'branch_id':'main','parent_cut':None,'records':self.rows}]},'world_sha256')

    def append(self,ledger,kind,payload,*,occurred=TIME):
        from cognitive_kernel.contracts import ProvenanceReference
        from cognitive_kernel.experience import ExperienceEvent
        n=len(self.rows)+1;digest=payload['source_artifact']['file_sha256'] if kind=='observation' else canonical_sha256(payload)
        event=ExperienceEvent.create(event_type=kind.replace('_','-'),scope=SCOPE,occurred_at=occurred,content_digest=digest,
            provenance=ProvenanceReference.create(provenance_type='derived_inference',source_reference_ids=(f'episode-payload-{n}',),
                derivation_activity_id=f'episode-event-{n}',responsible_component='fictional-control',model_id=None,confidence=None),
            retention_class='ordinary_experience',storage_tier='hot',payload_reference=f'episode-payload-{n}',
            parent_event_ids=() if not self.events else (self.events[-1]['event_id'],))
        receipt=ledger.append_event(event,committed_at=TIME);self.events.append(event.metadata_record())
        record=seal({'schema':'mfm-world-record-v2','world_id':'world-a','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,'sequence':n,
            'previous_record_sha256':self.genesis if n==1 else self.rows[-1]['record_sha256'],'event':event.metadata_record(),
            'entry_receipt':receipt.entries[0].record(),'payload_kind':kind,'payload':deepcopy(payload)})
        # Base fixture originals remain available under their distinct IDs.
        artifact=self.add(f'episode-world-record-{n}',record)
        self.rows.append({'sequence':n,'record_sha256':record['record_sha256'],'artifact':artifact})

    def episode_governance(self,key,action,eid,digest,predecessor,metadata):
        members=None if metadata is None else {name:metadata[name] for name in
            ('member_evidence_ids','member_claim_version_ids','member_candidate_ids','mission_node_ids','participant_ids')}
        return self.add(key,seal({'schema':'mfm-world-episode-governance-v1','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'episode_id':eid,'episode_sha256':digest,'action':action,'expected_predecessor_id':predecessor,
            'member_ancestry_sha256':None if members is None else canonical_sha256(members),'rights_refs':['fictional-rights'],
            'origin':'research_explicit_records'}))

    def episode_commit(self,ledger,n,**kwargs):
        import json
        refs=deepcopy(self.episodes.transitions[n-1]);metadata=json.loads(self.episodes.raws(n)[0])
        refs['governance_ref']=self.episode_governance(f'episode-accept-{n}','accept',metadata['episode_id'],metadata['episode_sha256'],
            metadata['supersedes_episode_id'],metadata)
        self.append(ledger,'episode_commit',refs,**kwargs)

    def context(self,refs=('source-0','source-1')):
        from cognitive_kernel.formation_semantics_v16 import FormationContextV16,context_v16_from_record
        old=context_v16_from_record(super().context(refs))
        return FormationContextV16(old.base,old.source_sensitivities,(self.target,)).record()


def episode_request(world,view):
    from tests.governance.mfm_world_fixtures import input_request
    old=input_request(world,view,priors=('claim-1',*[r['ref_id'] for r in view['episodes']]))
    return seal({**{k:v for k,v in old.items() if k!='record_sha256'},'schema':'mfm-world-input-request-v2'})


class EpisodeReadAdapter(ReadAdapter):
    """Explicit fictional current policy checked around every original I/O."""
    def __init__(self,world,request):
        super().__init__(world,request);self.part_allowed={'episode':True,'full_content':True,'summary_content':True};self.part_after=None

    def current_permission(self):
        from cognitive_kernel.canonical import CognitiveKernelContractError
        if self.boundary_record()!=self.request['read_boundary']:
            raise CognitiveKernelContractError('fictional current policy changed')

    def read_claim_originals(self,ref_id,refs):
        self.calls.append(('claim',ref_id));result=[]
        for name in ('identity','version','projection'):
            self.current_permission();result.append(self.world.artifacts[refs[name]['artifact_id']])
            if self.after:self.after(self)
            self.current_permission()
        return tuple(result)

    def read_episode_originals(self,ref_id,refs):
        from cognitive_kernel.canonical import CognitiveKernelContractError
        result=[]
        for name in ('episode','full_content','summary_content'):
            self.current_permission()
            if not self.part_allowed[name]:raise CognitiveKernelContractError('fictional Episode part permission denied')
            self.calls.append(('episode_part',(ref_id,name)))
            result.append(self.world.artifacts[refs[name]['artifact_id']])
            if self.part_after:self.part_after(self,ref_id,name)
            self.current_permission()
            if not self.part_allowed[name]:raise CognitiveKernelContractError('fictional Episode part permission changed')
        return tuple(result)
