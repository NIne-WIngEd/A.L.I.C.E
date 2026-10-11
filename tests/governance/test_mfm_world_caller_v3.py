from copy import deepcopy
import unittest
from unittest.mock import patch
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_semantics_v16 import FormationContextV16,context_v16_from_record,RegisteredFormationTarget
from scripts.mfm import formation_world_caller_v3 as W,formation_world_replay_v3 as R,formation_world_contracts as C,formation_caller_meaning_v3 as M
from tests.governance.mfm_task_content_fixtures import TaskWorld,VerificationAdapter,seal,SCOPE,NAMESPACE,TIME

LIMITS=C.ConstructionLimits(128,2097152,2097152)

def request(world,view,audience='host'):
    from tests.governance.mfm_world_fixtures import World
    old=context_v16_from_record(World.context(world.base,('source-0','source-1')))
    targets=tuple(RegisteredFormationTarget(ref,kind,SCOPE,NAMESPACE,('source-0',)) for ref,kind in
        ((world.f.root.node_id,'mission'),(world.content_rows['workspace']['record']['entity_id'],'workspace')))
    context=FormationContextV16(old.base,old.source_sensitivities,targets).record()
    boundary=seal({'scope':SCOPE.metadata_record(),'namespace':NAMESPACE,'purpose':'formation_training','read_service_id':'fictional-delivery',
        'as_of':TIME,'policy_sha256':'c'*64,'rights_receipt_refs':['fictional-rights'],'lifecycle_receipt_ref':'fictional-current-life','audience':audience})
    refs=[]
    for ref_id,kind,family in ((world.f.root.node_id,'mission','mission'),(world.content_rows['workspace']['record']['entity_id'],'workspace','workspace')):
        rows=[r for r in view['input_contents'] if r['family']==family]
        if rows:refs.append({'ref_id':ref_id,'kind':kind,**rows[-1]['known_content_ref']['content']})
    return seal({'schema':'mfm-world-input-request-v3','cut':view['cut'],'context':context,'opened_source_refs':['source-0','source-1'],
        'prior_ref_ids':['claim-1',*[r['ref_id'] for r in view['episodes']],'uncertainty-1'],'target_descriptions':[],
        'target_content_refs':refs,'uncertainty_ref_ids':['uncertainty-1'],'audience':audience,'read_boundary':boundary})

class Delivery:
    def __init__(self,world,req):self.world=world;self.boundary=deepcopy(req['read_boundary']);self.calls=[];self.after=None
    def delivery_boundary_record(self):return deepcopy(self.boundary)
    def _read(self,kind,key,ref):
        self.calls.append((kind,deepcopy(key),deepcopy(ref)));raw=self.world.artifacts[ref['artifact_id']]
        if self.after:self.after(self,kind,key,ref)
        return raw
    def read_content_part(self,key,ref):return self._read('content',key,ref)
    def read_claim_part(self,key,ref):return self._read('claim',key,ref)
    def read_episode_part(self,key,ref):return self._read('episode',key,ref)
    def read_source_part(self,registration,ref):return self._read('source',registration,ref)


def build(world=None,req=None,delivery=None,verification=None,n=None):
    world=world or TaskWorld(longitudinal=True);binding=R.world_contract_binding_v3();vb=world.verification_boundary();verification=verification or VerificationAdapter(world)
    cut=world.cut(n)
    view,vr=R.replay_world_prefix_v3(world.world,world_sha256=world.world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],
        verification_mode='current_verification_reads',verification_adapter=verification,verification_boundary=vb,verification_boundary_sha256=vb['record_sha256'],
        limits=LIMITS,contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
    req=req or request(world,view,world.audience);delivery=delivery or Delivery(world,req);policy=M.normative_view_policy_v3()
    kwargs=dict(replay_sha256=view['replay_sha256'],verification_receipt=vr,verification_receipt_sha256=vr['record_sha256'],input_request=req,
        input_request_sha256=req['record_sha256'],verification_adapter=verification,verification_boundary=vb,verification_boundary_sha256=vb['record_sha256'],
        read_adapter=delivery,view_policy=policy,view_policy_sha256=policy['record_sha256'],contract_binding=binding,contract_binding_sha256=binding['record_sha256'],limits=LIMITS)
    bundle=W.world_caller_bundle_v3(view,**kwargs)
    return world,view,req,delivery,kwargs,bundle

class CallerTests(unittest.TestCase):
    def test_real_originals_full_roles_distinct_legacy_counters_and_no_teacher(self):
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',side_effect=AssertionError('teacher inaccessible')):
            world,view,req,adapter,kw,b=build()
        families=[x['family'] for x in b['meaning']['candidate_meanings']]
        self.assertIn('mission-content-v1',families);self.assertIn('workspace-content-v1',families);self.assertIn('uncertainty-content-v1',families)
        claim=[x for kind,x,ref in adapter.calls if kind=='claim'];self.assertEqual([x['part_name'] for x in claim],['identity','version','projection'])
        self.assertTrue(all('content_version' not in x for x in claim));self.assertEqual(claim[0]['authority_source_position'],2)
        self.assertFalse(b['production_authenticated_adapter_claimed'])
        self.assertEqual(b['meaning'],M.caller_meaning_v3_from_record(b['meaning'],b['caller'],meaning_sha256=b['meaning']['record_sha256'],caller_sha256=b['caller']['record_sha256']))

    def test_every_part_changed_delivery_boundary_stops_later_reads(self):
        world,view,req,adapter,kw,b=build()
        count=len(adapter.calls);self.assertGreater(count,20)
        for stop in range(count):
            bad=Delivery(world,req)
            def mutate(a,*unused):
                if len(a.calls)==stop+1:a.boundary=seal({**{k:v for k,v in a.boundary.items() if k!='record_sha256'},'policy_sha256':'f'*64})
            bad.after=mutate
            with self.subTest(stop=stop),self.assertRaises(CognitiveKernelContractError):W.world_caller_bundle_v3(view,**{**kw,'read_adapter':bad})
            self.assertEqual(len(bad.calls),stop+1)

    def test_proof_boundary_changed_refuses_before_any_delivery_and_pure_receipt_cannot_admit(self):
        world,view,req,adapter,kw,b=build();proof=VerificationAdapter(world)
        proof.after=lambda a,ref:a.boundary.update(policy_sha256='f'*64)
        delivery=Delivery(world,req)
        with self.assertRaises(CognitiveKernelContractError):W.world_caller_bundle_v3(view,**{**kw,'verification_adapter':proof,'read_adapter':delivery})
        self.assertEqual(delivery.calls,[]);self.assertEqual(len(proof.calls),1)
        pure=R.prefix_proof_replay_v3(view['historical_prefix_proof'],cut=view['cut'])[1]
        with self.assertRaises(CognitiveKernelContractError):W.world_caller_bundle_v3(view,**{**kw,'verification_receipt':pure,'verification_receipt_sha256':pure['record_sha256'],'read_adapter':delivery})
        self.assertEqual(delivery.calls,[])

    def test_non_host_delivers_only_committed_projection_never_attention_override(self):
        world=TaskWorld(longitudinal=True,audience='non_host');binding=R.world_contract_binding_v3();view,_=R.replay_world_prefix_v3(world.world,
            world_sha256=world.world['world_sha256'],cut=world.cut(),cut_sha256=world.cut()['record_sha256'],verification_mode='supplied_bytes_only',
            supplied_artifacts=world.artifacts,limits=LIMITS,contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
        req=request(world,view,'non_host');_,_,_,adapter,_,b=build(world,req)
        names=[key['part_name'] for kind,key,ref in adapter.calls if kind=='content'];self.assertNotIn('attention',names)
        ws=next(x for x in b['meaning']['candidate_meanings'] if x['family']=='workspace-content-v1')
        body=M.M._body(ws);self.assertNotIn('attention',body['contents'][0]['parts']);self.assertFalse(ws['observed_source_eligible'])
        bad=deepcopy(req);bad['audience']='host';bad['read_boundary']=seal({**{k:v for k,v in bad['read_boundary'].items() if k!='record_sha256'},'audience':'host'});bad=seal({k:v for k,v in bad.items() if k!='record_sha256'})
        _,_,_,a,_,b=build(world,bad);self.assertFalse(any(x['part_name']=='workspace-non_host' for kind,x,ref in a.calls if kind=='content'))

    def test_non_host_privileged_originals_verified_outside_delivery(self):
        world=TaskWorld(longitudinal=True,audience='non_host');verification=VerificationAdapter(world)
        _,_,_,delivery,_,bundle=build(world,verification=verification)
        record=world.content_rows['workspace']['record']
        self.assertTrue(record['override_artifacts']);self.assertTrue(record['override_reason_texts'])
        expected=[record['attention_artifact']['artifact_id'],*[r['artifact_id'] for r in record['override_artifacts']],
            *[r['text']['artifact_id'] for r in record['override_reason_texts']]]
        actual_delivery=[ref['artifact_id'] for kind,key,ref in delivery.calls]
        for part in expected:self.assertIn(part,verification.calls);self.assertNotIn(part,actual_delivery)
        entry=next(x for x in bundle['meaning']['candidate_meanings'] if x['family']=='workspace-content-v1')
        body=M.M._body(entry)['contents'][0]['parts'];self.assertNotIn('override-secret-reason',body)
        projection=body['workspace-non_host'];self.assertEqual(projection['items'][0]['privacy_class'],'sensitive')
        self.assertEqual(projection['items'][0]['redaction_state'],'metadata_only')
        self.assertEqual(projection['layout']['total_candidate_count'],2);self.assertEqual(len(projection['items']),1)

    def test_typed_role_promotion_and_changed_reload_refuse(self):
        world,view,req,adapter,kw,b=build();bad=deepcopy(req);bad['target_content_refs'][0]['kind']='entity';bad=seal({k:v for k,v in bad.items() if k!='record_sha256'})
        delivery=Delivery(world,bad)
        with self.assertRaises(CognitiveKernelContractError):W.world_caller_bundle_v3(view,**{**kw,'input_request':bad,'input_request_sha256':bad['record_sha256'],'read_adapter':delivery})
        self.assertEqual(delivery.calls,[])
        changed=Delivery(world,req);changed.boundary=seal({**{k:v for k,v in changed.boundary.items() if k!='record_sha256'},'audience':'non_host'})
        with self.assertRaises(CognitiveKernelContractError):W.world_caller_bundle_v3(view,**{**kw,'read_adapter':changed})
        self.assertEqual(changed.calls,[])

    def test_withheld_and_deleted_typed_families_have_no_delivery_callbacks(self):
        for status in ('withheld','revoked','deleted'):
            world=TaskWorld(longitudinal=True,content_status={'mission':status,'uncertainty':status})
            *_,adapter,kw,bundle=build(world)
            names=[r['family'] for kind,r,ref in adapter.calls if kind=='content']
            self.assertNotIn('mission',names);self.assertNotIn('uncertainty',names)
            candidates=bundle['meaning']['candidate_inventory'];rows=bundle['meaning']['candidate_meanings']
            for candidate,row in zip(candidates,rows):
                if candidate['ref_id'] in (world.f.root.node_id,'uncertainty-1'):
                    self.assertEqual(row['availability'],status);self.assertIsNone(row['body_base64'])

    def test_original_historical_priors_without_current_targets_and_full_current_reload(self):
        from tests.governance.mfm_world_fixtures import World
        world,view,req,adapter,kwargs,bundle=build()
        rebuilt=W.verify_world_caller_bundle_v3(bundle,bundle_sha256=bundle['bundle_sha256'],world=world.world,
            world_sha256=world.world['world_sha256'],cut=view['cut'],cut_sha256=view['cut']['record_sha256'],**kwargs)
        self.assertEqual(rebuilt,bundle)
        historical=deepcopy(req);historical.update(context=World.context(world.base,('source-1',)),opened_source_refs=['source-1'],
            target_content_refs=[],target_descriptions=[])
        historical=seal({k:v for k,v in historical.items() if k!='record_sha256'})
        _,_,_,delivery,_,original=build(world,historical)
        self.assertEqual([key['evidence']['ref_id'] for kind,key,ref in delivery.calls if kind=='source'],['source-1'])
        self.assertFalse(any(c['origin']=='context_target' for c in original['meaning']['candidate_inventory']))
        self.assertTrue(any(c['family']=='episode-snapshot-v1' for c in original['meaning']['candidate_meanings']))

    def test_early_actor_prior_has_no_future_observation_or_uncertainty_material(self):
        world=TaskWorld(longitudinal=True,complete_trajectory=True)
        _,view,req,delivery,_,bundle=build(world,n=world.case_cuts['actor'])
        row=next(x for x in bundle['meaning']['candidate_meanings'] if x['family']=='mission-content-v1')
        contents=M.M._body(row)['contents'];self.assertTrue(any(x['family']=='result' for x in contents))
        self.assertFalse(any(x['family']=='outcome' for x in contents))
        slot=next(i for i,c in enumerate(bundle['meaning']['candidate_inventory']) if c['ref_id']=='uncertainty-1')
        self.assertEqual(bundle['meaning']['candidate_meanings'][slot]['availability'],'metadata_only')
        self.assertFalse(any(key.get('family')=='uncertainty' for kind,key,ref in delivery.calls if kind=='content'))

if __name__=='__main__':unittest.main()
