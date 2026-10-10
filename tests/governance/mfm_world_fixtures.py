"""Tiny actual reversible ledger/Claim records, never a product data source."""
from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
import tempfile
from pathlib import Path

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, normalize_timestamp
from cognitive_kernel.contracts import ProductHostScope, ProvenanceReference
from cognitive_kernel.experience import ExperienceEvent
from cognitive_kernel.ledger_store import open_experience_ledger
from cognitive_kernel.claim_authority_prototype import open_claim_authority_prototype, ClaimAuthorityAppendRequest
from cognitive_kernel.claim_contracts import CanonicalTaggedValue, ClaimIdentity
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope
from cognitive_kernel.formation_contracts import FormationEvidenceRef, FormationContextPacket
from cognitive_kernel.formation_semantics_v16 import FormationContextV16, RegisteredSourceSensitivity
from cognitive_kernel.formation_sources import RegisteredFormationSource
from cognitive_kernel.formation_sources import RegisteredFormationStore

TIME = normalize_timestamp('2026-08-06T04:00:00Z')
SCOPE = ProductHostScope.create(product_id='alice', host_instance_id='host-a', schema_version='1.0.0', encryption_domain='private-a')
NAMESPACE = 'authority-a'

def seal(core, key='record_sha256'):
    return {**deepcopy(core), key:canonical_sha256(core)}

def env(record_id, record_type, authority_role, *, sources=(), supersedes=(), deletion='active'):
    return MemoryUnitEnvelope.create(scope=SCOPE, record_id=record_id, record_type=record_type,
        authority_namespace_id=NAMESPACE, host_or_cluster_id=SCOPE.host_instance_id,
        authority_role=authority_role, deployment_profile='single_workstation', created_at=TIME,
        valid_from=TIME, valid_to=None, transaction_time=TIME, logical_clock=1, generation=0,
        state='committed', data_classification='ordinary', retention_class='authoritative_source',
        deletion_state=deletion, provenance_digest='a'*64, content_digest='b'*64,
        writer='fictional-authority', workflow_or_request_id=record_id,
        idempotency_namespace='fictional', idempotency_key=record_id,
        source_records=sources, supersedes=supersedes)

def value(text):
    return CanonicalTaggedValue.create(type_tag='map', value={'text':text, 'count':2, 'ratio':0.5,
        'flag':True, 'missing':None, 'list':['ana','ana'], 'literal/~':'Café "A" \\ 😀'})

def request(identity, n, text, *, deletion='active'):
    prior = None if n==1 else f'version-{n-1}'
    return ClaimAuthorityAppendRequest.create(identity=identity,
        version_envelope=env(f'version-{n}','claim_version','claim_authority',
            sources=('claim-1',f'source-{0 if n==1 else 1}'), supersedes=() if prior is None else (prior,)),
        projection_envelope=env(f'projection-{n}','current_claim_projection','registered_projection',
            sources=('claim-1',f'version-{n}'), deletion=deletion),
        value=value(text), authority_class='owner_attested', confidence=0.8,
        adjudication_state='accepted', evidence_relation_ids=(f'fictional-binding-{n}',),
        correction_of=() if prior is None else (prior,), request_digest=canonical_sha256({'request':n,'text':text}),
        expected_current_claim_version_id=prior, deletion_state=deletion)

def request_record(req):
    def material(v):
        if hasattr(v,'metadata_record'): return v.metadata_record()
        if isinstance(v,tuple): return [material(x) for x in v]
        return v
    return {f.name:material(getattr(req,f.name)) for f in fields(req)}

class World:
    def __init__(self, *, corrected='light mode', deleted=False, later_occurred='2026-08-05T04:00:00Z', later_prefix='', confirmation=False):
        self.artifacts={}; self.calls=[]; self.rows=[]; self.registrations={}; self.claim_raws=[]
        self.later_prefix=later_prefix
        self.provenance={'author_id':'fictional-worker','generator_family':'public-world-control',
            'renderer_family':'exact-json-control','scenario_families':['revision'],
            'host_lineage':['fictional-host-a'],'source_person_lineage':['fictional-source-person'],
            'parent_world_refs':[], 'rights_refs':['fictional-rights']}
        with tempfile.TemporaryDirectory(prefix='mfm-world-fixture-') as temporary:
            base=Path(temporary); repo=base/'repository'; repo.mkdir()
            with open_experience_ledger(base/'vault/ledger.sqlite3',scope=SCOPE,repository_root=repo,created_at=TIME) as ledger:
                store=open_claim_authority_prototype(base/'vault/claim.sqlite3',scope=SCOPE,
                    authority_namespace_id=NAMESPACE,created_at=TIME,repository_root=repo)
                try:
                    self.ledger_profile={'schema':'experience-ledger-records-v1','ledger_schema_version':'1.0.0',
                        'ledger_id':ledger.ledger_id,'scope_sha256':ledger.scope_digest,'created_at':TIME,
                        'genesis_sha256':ledger.genesis_sha256}
                    self.authority_profile={'schema':'claim-prototype-records-v1','authority_id':store.authority_id,
                        'authority_generation':1,'prototype_state':'reversible_nonproduction'}
                    self.genesis=canonical_sha256({'schema':'mfm-world-genesis-v1','world_id':'world-a',
                        'scope':SCOPE.metadata_record(),'namespace':NAMESPACE,'ledger_profile':self.ledger_profile,
                        'authority_profile':self.authority_profile})
                    self.observation(ledger,0,'Owner prefers dark mode. Café "A" \\ 😀')
                    identity=ClaimIdentity.create(envelope=env('claim-1','claim_identity','claim_authority'),
                        claim_id='claim-1',canonical_subject=CanonicalTaggedValue.create(type_tag='identifier',value='owner'),
                        canonical_predicate='stored_preference',canonical_value=value('dark mode'),semantic_scope=('fictional',))
                    self.commit(ledger,store,request(identity,1,'dark mode'),'append')
                    self.append(ledger,'registry_metadata',{'ref_id':'episode-1','kind':'episode','record_sha256':'d'*64,
                        'status':'present','description':'Episode full body unavailable.','authority_ref':'fictional-episode-read'})
                    self.observation(ledger,1,'Source-person reports a later owner correction.')
                    self.commit(ledger,store,request(identity,2,corrected, deletion='deleted' if deleted else 'active'),
                                'delete' if deleted else 'correct', occurred=normalize_timestamp(later_occurred))
                    denied=self.governance('denied-1','deny',None,'version-2',['source-1'])
                    self.append(ledger,'governance_no_commit',{'verdict':'deny','governance_ref':denied,
                        'expected_current_claim_version_id':'version-2','ref_id':'claim-1'})
                    withheld=self.governance('withheld-1','withheld',None,'version-2',['source-1'])
                    self.append(ledger,'visibility_transition',{'ref_id':'claim-1','kind':'claim',
                        'expected_record_sha256':self.claims[-1]['version']['version_sha256'],
                        'status':'withheld','governance_ref':withheld})
                    if confirmation:
                        original=self.artifacts['world-record-2']; payload=deepcopy(__import__('json').loads(original)['payload'])
                        confirmed=store.append(request(identity,1,'dark mode'),committed_at=TIME)
                        payload['append_receipt']=self.add('confirmed-receipt',confirmed.metadata_record())
                        self.append(ledger,'claim_commit',payload)
                    assert ledger.verify_integrity().entry_count==len(self.rows)
                    store.verify_integrity()
                finally: store.close()
        self.world=seal({'schema':'mfm-longitudinal-world-v1','world_id':'world-a','scope':SCOPE.metadata_record(),
            'namespace':NAMESPACE,'provenance':self.provenance,'ledger_profile':self.ledger_profile,
            'authority_profile':self.authority_profile,'branches':[{'branch_id':'main','parent_cut':None,'records':self.rows}]},'world_sha256')

    def add(self, key, content):
        if len(self.rows)>=4: key=self.later_prefix+key
        raw=content if type(content) is bytes else canonical_json_bytes(content)
        if key in self.artifacts and self.artifacts[key]!=raw: raise ValueError('fixture object collision')
        self.artifacts[key]=raw
        return {'artifact_id':key,'bytes':len(raw),'file_sha256':sha256(raw).hexdigest(),
                'media_type':'application/octet-stream' if type(content) is bytes else 'application/json'}

    def read(self, ref):
        self.calls.append(ref['artifact_id']); return self.artifacts[ref['artifact_id']]

    def append(self, ledger, kind, payload, *, occurred=TIME):
        n=len(self.rows)+1
        digest=payload['source_artifact']['file_sha256'] if kind=='observation' else canonical_sha256(payload)
        event=ExperienceEvent.create(event_type=kind.replace('_','-'),scope=SCOPE,occurred_at=occurred,
            content_digest=digest, provenance=ProvenanceReference.create(provenance_type='derived_inference',
                source_reference_ids=(f'payload-{n}',),derivation_activity_id=f'event-{n}',responsible_component='fictional-control',
                model_id=None,confidence=None), retention_class='ordinary_experience',storage_tier='hot',
            payload_reference=f'payload-{n}',parent_event_ids=() if n==1 else (self.events[-1]['event_id'],))
        receipt=ledger.append_event(event,committed_at=TIME)
        if not hasattr(self,'events'): self.events=[]
        self.events.append(event.metadata_record())
        core={'schema':'mfm-world-record-v1','world_id':'world-a','scope':SCOPE.metadata_record(),'namespace':NAMESPACE,
            'sequence':n,'previous_record_sha256':self.genesis if n==1 else self.rows[-1]['record_sha256'],
            'event':event.metadata_record(),'entry_receipt':receipt.entries[0].record(),
            'payload_kind':kind,'payload':deepcopy(payload)}
        record=seal(core)
        artifact=self.add(f'world-record-{n}',record)
        self.rows.append({'sequence':n,'record_sha256':record['record_sha256'],'artifact':artifact})

    def observation(self, ledger, n, text):
        source=self.add(f'original-source-{n}',text.encode())
        evidence=FormationEvidenceRef(ref_id=f'source-{n}',scope=SCOPE,authority_namespace_id=NAMESPACE,
            content_digest=source['file_sha256'],role='hypothetical_training',modality='text',subject_ref='owner',
            speaker_ref='fictional-owner',observed_at=TIME,recorded_at=TIME)
        reg=RegisteredFormationSource.create(evidence=evidence,object_ref=source['artifact_id'])
        self.registrations[evidence.ref_id]=reg
        self.append(ledger,'observation',{'registration':{'evidence':evidence.metadata_record(),
            'object_ref':reg.object_ref,'registration_sha256':reg.registration_sha256},'source_artifact':source,
            'source_rights_ref':'fictional-rights','observed_at':TIME,'speaker_ref':'fictional-owner',
            'subject_ref':'owner','epistemic_role':'hypothetical_training','modality':'text','locator':None})

    def governance(self,key,decision,req_digest,expected,sources):
        record=seal({'schema':'mfm-world-supplied-governance-v1','scope':SCOPE.metadata_record(),
            'namespace':NAMESPACE,'decision_id':key,'issuer_id':'fictional-owner',
            'authority_basis_ref':'fictional-research-authorization','decision':decision,'subject_ref':'claim-1',
            'expected_current_claim_version_id':expected,'request_digest':req_digest,'source_refs':sources,
            'origin':'research_explicit_records'})
        return self.add(key,record)

    def commit(self,ledger,store,req,action,*,occurred=TIME):
        receipt=store.append(req,committed_at=TIME)
        triple={'identity':req.identity.metadata_record(),'version':store.history('claim-1')[-1],
            'projection':store.load_current('claim-1')}
        if not hasattr(self,'claims'): self.claims=[]
        self.claims.append(triple)
        refs={key:self.add(f'{key}-{len(self.claims)}',record) for key,record in triple.items()}
        self.claim_raws.append({key:self.artifacts[ref['artifact_id']] for key,ref in refs.items()})
        governing=self.governance(f'accepted-{len(self.claims)}','accepted',req.request_digest,
                                 req.expected_current_claim_version_id,[f'source-{0 if len(self.claims)==1 else 1}'])
        self.append(ledger,'claim_commit',{'action':action,'governance_ref':governing,
            'append_request':self.add(f'request-{len(self.claims)}',request_record(req)),
            'append_receipt':self.add(f'append-receipt-{len(self.claims)}',receipt.metadata_record()),**refs},occurred=occurred)

    def cut(self,n=5,branch='main'):
        return seal({'schema':'mfm-world-cut-v1','world_id':'world-a','branch_id':branch,'sequence':n,
                     'head_sha256':self.genesis if n==0 else self.rows[n-1]['record_sha256']})

    def context(self, refs=('source-0','source-1')):
        evidence=tuple(self.registrations[ref].evidence for ref in refs)
        return FormationContextV16(FormationContextPacket(SCOPE,NAMESPACE,tuple(refs),evidence),
            tuple(RegisteredSourceSensitivity(ref,'private') for ref in refs),()).record()


def input_request(world, view, *, refs=None, priors=('claim-1','episode-1'), context=None):
    refs=tuple(s['registration']['evidence']['ref_id'] for s in view['sources']) if refs is None else tuple(refs)
    boundary=seal({'scope':SCOPE.metadata_record(),'namespace':NAMESPACE,'purpose':'formation_training',
        'read_service_id':'fictional-read','as_of':TIME,'policy_sha256':'c'*64,
        'rights_receipt_refs':['fictional-rights'],'lifecycle_receipt_ref':'fictional-lifecycle'})
    return seal({'schema':'mfm-world-input-request-v1','cut':view['cut'],
        'context':world.context(refs) if context is None else context,'opened_source_refs':list(refs),
        'prior_ref_ids':list(priors),'target_descriptions':[],'read_boundary':boundary})


class ReadAdapter:
    """Fictional permission, real RegisteredFormationStore race checks."""
    def __init__(self, world, request):
        self.world=world; self.request=request; self.allowed=True; self.calls=[]; self.after=None
        self.boundary=deepcopy(request['read_boundary']); self.registry=self
        self.store=RegisteredFormationStore(scope=SCOPE,authority_namespace_id=NAMESPACE,
            registry=self,custody=self,permits=lambda source,purpose:self.allowed and purpose=='formation_training',
            purpose='formation_training')
    def lookup(self,ref_id): return self.world.registrations.get(ref_id)
    def read(self,scope,object_ref):
        raw=self.world.artifacts[object_ref]
        if self.after: self.after(self)
        return raw
    def boundary_record(self):
        if not self.allowed: raise __import__('cognitive_kernel.canonical',fromlist=['CognitiveKernelContractError']).CognitiveKernelContractError('fictional policy denied')
        return deepcopy(self.boundary)
    def read_source(self,registration):
        self.calls.append(('source',registration.evidence.ref_id)); return self.store.read(registration.evidence)
    def read_claim_originals(self,ref_id,artifact_refs):
        self.boundary_record(); self.calls.append(('claim',ref_id))
        originals=tuple(self.world.artifacts[artifact_refs[k]['artifact_id']] for k in ('identity','version','projection'))
        if self.after: self.after(self)
        self.boundary_record(); return originals
