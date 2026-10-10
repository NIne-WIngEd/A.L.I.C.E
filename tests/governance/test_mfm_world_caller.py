from copy import deepcopy
from dataclasses import replace
import importlib
import unittest
from unittest.mock import patch
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_semantics_v16 import RegisteredFormationTarget,context_v16_from_record
from scripts.mfm import formation_world_contracts as C, formation_caller_meaning as M
from tests.governance.mfm_world_fixtures import World,ReadAdapter,input_request,seal,SCOPE,NAMESPACE
from tests.governance import test_mfm_world_replay as replay_fixture


class WorldCallerTests(unittest.TestCase):
    def build(self, world=None,n=5, *, request=None,adapter=None):
        api=importlib.import_module('scripts.mfm.formation_world_caller'); world=world or World()
        view=replay_fixture.ReplayTests().replay(world,n); request=request or input_request(world,view)
        adapter=adapter or ReadAdapter(world,request); policy=M.normative_view_policy(); binding=C.world_contract_binding()
        result=api.world_caller_bundle(view,replay_sha256=view['replay_sha256'],input_request=request,
            input_request_sha256=request['record_sha256'],read_adapter=adapter,view_policy=policy,
            view_policy_sha256=policy['record_sha256'],contract_binding=binding,contract_binding_sha256=binding['record_sha256'],limits=C.ConstructionLimits(30,100000,100000))
        return api,world,view,request,adapter,result

    def test_actual_original_prior_delivered_not_new_observation(self):
        _,world,view,request,adapter,bundle=self.build()
        body=[r for r in bundle['meaning']['candidate_meanings'] if r['family']=='claim-snapshot-v1'][0]
        actual=M._body(body)
        self.assertEqual(actual['version'],world.claims[1]['version'])
        self.assertEqual(actual['identity'],world.claims[0]['identity'])
        self.assertEqual(len(bundle['caller']['opened_sources']),2)
        self.assertEqual(bundle['meaning']['snapshot']['authority_generation'],1)
        self.assertEqual(bundle['meaning']['snapshot']['source_position'],2)
        self.assertEqual(bundle['cut']['sequence'],5)
        self.assertNotIn('claim-1',[s['ref_id'] for s in bundle['caller']['opened_sources']])
        self.assertEqual(adapter.calls,[('source','source-0'),('source','source-1'),('claim','claim-1')])
        _,_,_,_,_,different=self.build(World(corrected='sepia mode'))
        self.assertNotEqual(bundle['meaning'],different['meaning'])

    def test_teacher_denied_future_tail_and_exact_reload(self):
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',side_effect=AssertionError('teacher forbidden')):
            api,world,view,request,adapter,bundle=self.build()
        policy=M.normative_view_policy(); binding=C.world_contract_binding(); cut=world.cut()
        kwargs=dict(bundle_sha256=bundle['bundle_sha256'],world=world.world,world_sha256=world.world['world_sha256'],
            cut=cut,cut_sha256=cut['record_sha256'],input_request=request,input_request_sha256=request['record_sha256'],
            read_artifact=world.read,read_adapter=adapter,limits=C.ConstructionLimits(30,100000,100000),
            contract_binding=binding,contract_binding_sha256=binding['record_sha256'],view_policy=policy,view_policy_sha256=policy['record_sha256'])
        self.assertEqual(api.verify_world_caller_bundle(bundle,**kwargs),bundle)
        changed=deepcopy(world.world); changed['branches'][0]['records'][-1]['artifact']['file_sha256']='0'*64
        kwargs['world']=seal({k:v for k,v in changed.items() if k!='world_sha256'},'world_sha256');kwargs['world_sha256']=kwargs['world']['world_sha256']
        self.assertEqual(api.verify_world_caller_bundle(bundle,**kwargs),bundle)
        adapter.boundary['policy_sha256']='f'*64
        with self.assertRaises(CognitiveKernelContractError): api.verify_world_caller_bundle(bundle,**kwargs)

    def test_revoked_and_source_and_claim_races_refuse(self):
        for kind in ('before','source','claim','registration'):
            world=World();view=replay_fixture.ReplayTests().replay(world);request=input_request(world,view);adapter=ReadAdapter(world,request)
            if kind=='before': adapter.allowed=False
            elif kind=='registration': adapter.after=lambda a:a.world.registrations.pop('source-0')
            elif kind=='source': adapter.after=lambda a:setattr(a,'allowed',False)
            else:
                old=adapter.read_claim_originals
                def read(ref,refs):
                    result=old(ref,refs);adapter.allowed=False;return result
                adapter.read_claim_originals=read
            with self.assertRaises(CognitiveKernelContractError): self.build(world,request=request,adapter=adapter)

    def test_context_order_future_source_extra_target_and_hash_refuse(self):
        world=World();view=replay_fixture.ReplayTests().replay(world,2); request=input_request(world,view,priors=('claim-1',))
        for key,value in [('opened_source_refs',['source-1']),('teacher_output',{}),('prior_ref_ids',['absent']),('target_descriptions',[{'ref_id':'invented'}])]:
            bad=deepcopy(request);bad[key]=value;bad=seal({k:v for k,v in bad.items() if k!='record_sha256'})
            with self.assertRaises(CognitiveKernelContractError): self.build(world,2,request=bad)

    def test_typed_same_id_descriptions_and_absent_families(self):
        world=World();view=replay_fixture.ReplayTests().replay(world); context=context_v16_from_record(world.context())
        targets=tuple(RegisteredFormationTarget(ref,kind,SCOPE,NAMESPACE,('source-0',)) for ref,kind in
                      [('owner','entity'),('scene-1','scene'),('mission-1','mission'),('workspace-1','workspace')])
        request=input_request(world,view,context=replace(context,targets=targets).record())
        request['target_descriptions']=[{'ref_id':t.ref_id,'kind':t.kind,'description':'supplied '+t.kind+' meaning'} for t in targets]
        request=seal({k:v for k,v in request.items() if k!='record_sha256'})
        _,_,_,_,_,bundle=self.build(world,request=request)
        candidates=bundle['meaning']['candidate_inventory']; roles=[(r['ref_id'],r['kind'],r['origin']) for r in candidates]
        self.assertIn(('owner','source_subject','source_subject'),roles);self.assertIn(('owner','entity','context_target'),roles)
        episode=[r for r in bundle['meaning']['candidate_meanings'] if r['family']=='absent-body-v1'][0]
        self.assertEqual(episode['availability'],'metadata_only');self.assertIsNone(episode['body_base64'])

    def test_withheld_and_deleted_claims_never_read(self):
        for world,n,status in [(World(),7,'withheld'),(World(deleted=True),5,'deleted')]:
            _,_,_,_,adapter,bundle=self.build(world,n)
            self.assertFalse(any(kind=='claim' for kind,_ in adapter.calls))
            entry=[r for c,r in zip(bundle['meaning']['candidate_inventory'],bundle['meaning']['candidate_meanings']) if c['ref_id']=='claim-1'][0]
            self.assertEqual(entry['availability'],status);self.assertIsNone(entry['body_base64'])

    def test_exact_delivery_byte_budget_and_corrupted_original_refuse(self):
        api,world,view,request,adapter,bundle=self.build();policy=M.normative_view_policy();binding=C.world_contract_binding()
        total=sum(world.artifacts[s['registration']['object_ref']].__len__() for s in view['sources'])
        total+=sum(len(raw) for raw in world.claim_raws[1].values())
        kwargs=dict(replay_sha256=view['replay_sha256'],input_request=request,input_request_sha256=request['record_sha256'],
            read_adapter=adapter,view_policy=policy,view_policy_sha256=policy['record_sha256'],contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
        self.assertEqual(api.world_caller_bundle(view,limits=C.ConstructionLimits(30,100000,total),**kwargs),bundle)
        with self.assertRaises(CognitiveKernelContractError):api.world_caller_bundle(view,limits=C.ConstructionLimits(30,100000,total-1),**kwargs)
        raw=world.artifacts['version-2'];world.artifacts['version-2']=raw[:-1]+b' '
        with self.assertRaises(CognitiveKernelContractError):api.world_caller_bundle(view,limits=C.ConstructionLimits(30,100000,total),**kwargs)
