from copy import deepcopy
from base64 import b64decode,b64encode
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from tests.governance.mfm_episode_fixtures import EpisodeWorld,EpisodeReadAdapter,episode_request,seal,World
from scripts.mfm import formation_world_contracts as C,formation_world_replay_v2 as R,formation_caller_meaning_v2 as M


def build(n=7,world=None,request=None,adapter=None,limits=None):
    world=world or EpisodeWorld();binding=R.world_contract_binding_v2();policy=M.normative_view_policy_v2()
    limits=limits or C.ConstructionLimits(40,100000,500000);cut=world.cut(n)
    view=R.replay_world_prefix_v2(world.world,world_sha256=world.world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],
        read_artifact=world.read,limits=limits,contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
    request=request or episode_request(world,view);adapter=adapter or EpisodeReadAdapter(world,request)
    kwargs=dict(replay_sha256=view['replay_sha256'],input_request=request,input_request_sha256=request['record_sha256'],
        read_adapter=adapter,limits=limits,view_policy=policy,view_policy_sha256=policy['record_sha256'],contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
    from scripts.mfm import formation_world_caller_v2 as W
    bundle=W.world_caller_bundle_v2(view,**kwargs)
    return W,world,view,request,adapter,kwargs,bundle


class EpisodeCallerTests(unittest.TestCase):
    def test_historical_original_proof_revalidated_before_any_current_io(self):
        W,world,view,request,_,kwargs,_=build()
        self.assertEqual(R.prefix_proof_replay(view['historical_prefix_proof'],cut=view['cut']),view)
        for change in ('future','foreign','bytes','receipt','version','registration'):
            bad=deepcopy(view);proof=bad['historical_prefix_proof']
            if change=='future':proof['cut']=world.cut(9)
            elif change=='foreign':proof['world']['namespace']='foreign'
            elif change=='bytes':proof['original_artifacts'][0]['base64']='e30='
            elif change=='receipt':
                # A self-consistent real receipt still cannot acquire a foreign previous ledger head.
                from cognitive_kernel.ledger import ExperienceLedgerEntryReceipt
                original=next(r for r in proof['original_artifacts'] if r['artifact']['artifact_id']=='episode-world-record-1')
                record=json.loads(b64decode(original['base64']));receipt=record['entry_receipt']
                record['entry_receipt']=ExperienceLedgerEntryReceipt.create(**{**{k:v for k,v in receipt.items() if k!='entry_sha256'},
                    'previous_entry_sha256':'f'*64}).record()
                record=seal({k:v for k,v in record.items() if k!='record_sha256'});raw=C.canonical_json_bytes(record)
                original['artifact'].update(bytes=len(raw),file_sha256=sha256(raw).hexdigest());original['base64']=b64encode(raw).decode()
                descriptor=proof['world']['branches'][0]['records'][0]
                descriptor.update(record_sha256=record['record_sha256'],artifact=deepcopy(original['artifact']))
                proof['world']=seal({k:v for k,v in proof['world'].items() if k!='world_sha256'},'world_sha256')
            elif change=='version':bad['episodes'][0]['member_registrations']['claim_versions'][0]['record_sha256']='f'*64
            else:bad['episodes'][0]['member_registrations']['participants'][0]['known_at_sequence']=7
            proof=seal({k:v for k,v in proof.items() if k!='record_sha256'});bad['historical_prefix_proof']=proof
            bad=seal({k:v for k,v in bad.items() if k!='replay_sha256'},'replay_sha256')
            adapter=EpisodeReadAdapter(world,request)
            with self.subTest(change=change),self.assertRaises(CognitiveKernelContractError):
                W.world_caller_bundle_v2(bad,**{**kwargs,'replay_sha256':bad['replay_sha256'],'read_adapter':adapter})
            self.assertEqual(adapter.calls,[])

    def test_historical_members_absent_from_current_context_remain_prior_only(self):
        _,world,view,request,_,_,_=build()
        request['context']=World.context(world,('source-1',));request['opened_source_refs']=['source-1']
        request=seal({k:v for k,v in request.items() if k!='record_sha256'});adapter=EpisodeReadAdapter(world,request)
        _,_,_,_,_,_,bundle=build(world=world,request=request,adapter=adapter)
        roles=[(c['ref_id'],c['kind']) for c in bundle['meaning']['candidate_inventory']]
        self.assertNotIn(('source-0','evidence'),roles);self.assertNotIn(('owner','entity'),roles)
        self.assertEqual([key for kind,key in adapter.calls if kind=='source'],['source-1'])
        self.assertEqual(len([e for e in bundle['meaning']['candidate_meanings'] if e['family']==M.EPISODE_FAMILY]),2)

    def test_late_registration_never_resolves_earlier_episode_and_mission_is_historical_only(self):
        world=EpisodeWorld(late_participant=True,registered_mission=True)
        # Both actual Episodes are committed by sequence8; owner was registered after Episode1.
        binding=R.world_contract_binding_v2();cut=world.cut(8)
        view=R.replay_world_prefix_v2(world.world,world_sha256=world.world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],
            read_artifact=world.read,limits=C.ConstructionLimits(40,100000,500000),contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
        request=episode_request(world,view);request['context']=World.context(world,('source-1',));request['opened_source_refs']=['source-1']
        request=seal({k:v for k,v in request.items() if k!='record_sha256'})
        adapter=EpisodeReadAdapter(world,request);_,_,_,_,_,_,bundle=build(8,world=world,request=request,adapter=adapter)
        known=bundle['meaning']['episode_snapshot']['episodes']
        self.assertEqual(known[0]['member_resolution']['participants'][0]['status'],'not_supplied')
        self.assertEqual(known[0]['member_registrations']['participants'],[])
        self.assertEqual(known[1]['member_resolution']['participants'][0]['status'],'resolved')
        self.assertEqual(known[0]['member_resolution']['missions'][0]['status'],'resolved')
        self.assertEqual(known[0]['member_registrations']['missions'][0]['kind'],'mission')
        self.assertNotIn(('owner','entity'),[(r['ref_id'],r['kind']) for r in bundle['meaning']['candidate_inventory']])
        self.assertFalse(any(r['kind']=='mission' for r in bundle['meaning']['candidate_inventory']))
        self.assertEqual([key for kind,key in adapter.calls if kind=='source'],['source-1'])

    def test_real_complete_originals_caller_roles_and_reload_teacher_inaccessible(self):
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',side_effect=AssertionError('teacher inaccessible')),\
             patch('scripts.mfm.v16_prior_memory_inputs.local_uncertainty_allocations',side_effect=AssertionError('local output inaccessible')):
            W,world,view,request,adapter,kwargs,bundle=build()
            reload={k:v for k,v in kwargs.items() if k!='replay_sha256'}
            actual=W.verify_world_caller_bundle_v2(bundle,bundle_sha256=bundle['bundle_sha256'],world=world.world,world_sha256=world.world['world_sha256'],
                cut=world.cut(7),cut_sha256=world.cut(7)['record_sha256'],read_artifact=world.read,**reload)
            self.assertEqual(actual,bundle)
        roles=[(r['ref_id'],r['kind'],r['origin']) for r in bundle['meaning']['candidate_inventory']]
        self.assertIn(('owner','source_subject','source_subject'),roles);self.assertIn(('owner','entity','context_target'),roles)
        bodies=[r for r in bundle['meaning']['candidate_meanings'] if r['family']==M.EPISODE_FAMILY]
        self.assertEqual(len(bodies),2);self.assertEqual(set(bodies[0]['original_parts']),set(M.PARTS))
        self.assertFalse(bundle['prior_read_receipt']['authenticated_service_claimed'])
        self.assertEqual(bundle['meaning']['episode_snapshot']['cut'],view['cut'])

    def test_current_withheld_or_deleted_episode_has_no_body_callbacks(self):
        for status in ('withheld','deleted'):
            _,_,_,_,adapter,_,bundle=build(9,world=EpisodeWorld(status=status))
            self.assertFalse(any(kind=='episode_part' and key[0]=='episode-2' for kind,key in adapter.calls))
            entry=next(row for candidate,row in zip(bundle['meaning']['candidate_inventory'],bundle['meaning']['candidate_meanings']) if candidate['ref_id']=='episode-2')
            self.assertEqual(entry['availability'],status);self.assertIsNone(entry['body_base64'])

    def test_policy_or_part_permission_change_at_each_episode_io_refuses(self):
        _,world,view,request,_,kwargs,_=build()
        for part in M.PARTS:
            adapter=EpisodeReadAdapter(world,request)
            adapter.part_after=lambda a,ref,name,p=part: a.boundary.update(policy_sha256='f'*64) if name==p else None
            with self.subTest(part=part),self.assertRaises(CognitiveKernelContractError):build(world=world,request=request,adapter=adapter)
            observed=[key[1] for kind,key in adapter.calls if kind=='episode_part']
            self.assertEqual(observed,list(M.PARTS[:M.PARTS.index(part)+1]))
        adapter=EpisodeReadAdapter(world,request);adapter.part_allowed['summary_content']=False
        with self.assertRaises(CognitiveKernelContractError):build(world=world,request=request,adapter=adapter)
        adapter=EpisodeReadAdapter(world,request);adapter.allowed=False
        with self.assertRaises(CognitiveKernelContractError):build(world=world,request=request,adapter=adapter)
        self.assertEqual(adapter.calls,[])
        # Current denial also defeats an earlier prefix with formerly present originals.
        early_request=episode_request(world,{**view,'cut':world.cut(4),'episodes':view['episodes'][:1]})
        early_request['context']=world.context(('source-0',));early_request['opened_source_refs']=['source-0']
        early_request=seal({k:v for k,v in early_request.items() if k!='record_sha256'})
        early=EpisodeReadAdapter(world,early_request);early.allowed=False
        with self.assertRaises(CognitiveKernelContractError):build(4,world=world,request=early_request,adapter=early)
        self.assertEqual(early.calls,[])

    def test_exact_delivery_budget_and_per_part_corruption_refuse(self):
        W,world,view,request,adapter,kwargs,bundle=build()
        refs=[row['source_artifact'] for row in view['sources']]
        refs += [original['artifact'] for row in view['current_claims'] for original in row['originals'].values()]
        refs += [row['artifacts'][part] for row in view['episodes'] for part in M.PARTS]
        total=sum(ref['bytes'] for ref in refs)
        exact=C.ConstructionLimits(40,100000,total)
        self.assertEqual(W.world_caller_bundle_v2(view,**{**kwargs,'limits':exact}),bundle)
        with self.assertRaises(CognitiveKernelContractError):W.world_caller_bundle_v2(view,**{**kwargs,'limits':C.ConstructionLimits(40,100000,total-1)})
        key='episode-2-summary_content';world.artifacts[key]=world.artifacts[key][:-1]+b' '
        with self.assertRaises(CognitiveKernelContractError):build(world=world)

    def test_context_cannot_promote_unsupplied_future_or_foreign_target(self):
        _,world,view,request,_,_,_=build()
        bad=deepcopy(request);bad['context']['targets'][0]['ref_id']='invented-participant';bad=seal({k:v for k,v in bad.items() if k!='record_sha256'})
        adapter=EpisodeReadAdapter(world,bad)
        with self.assertRaises(CognitiveKernelContractError):build(world=world,request=bad,adapter=adapter)
        self.assertEqual(adapter.calls,[])

if __name__=='__main__':unittest.main()
