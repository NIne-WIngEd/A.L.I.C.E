from copy import deepcopy
import json
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from tests.governance.mfm_episode_fixtures import EpisodeWorld,seal


class EpisodeReplayTests(unittest.TestCase):
    def setUp(self):
        self.world=EpisodeWorld()
        from scripts.mfm import formation_world_replay_v2 as R
        from scripts.mfm.formation_world_contracts import ConstructionLimits
        self.R=R;self.limits=ConstructionLimits(40,100000,500000);self.binding=R.world_contract_binding_v2()

    def replay(self,n=7,world=None,cut=None,limits=None):
        world=self.world.world if world is None else world;cut=self.world.cut(n) if cut is None else cut
        return self.R.replay_world_prefix_v2(world,world_sha256=world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],
            read_artifact=self.world.read,limits=self.limits if limits is None else limits,contract_binding=self.binding,contract_binding_sha256=self.binding['record_sha256'])

    def test_actual_originals_backdating_and_distinct_counters_no_body_reads(self):
        early=self.replay(4);late=self.replay(7)
        self.assertEqual(len(early['episodes']),1);self.assertEqual(len(late['episodes']),2)
        self.assertEqual(early['authority_source_position'],1)
        self.assertEqual(late['authority_source_position'],2)
        self.assertEqual(late['episodes'][1]['records']['episode']['generation'],2)
        self.assertEqual(late['episodes'][0]['records']['episode']['envelope']['generation'],0)
        self.assertEqual(late['episodes'][0],early['episodes'][0])
        self.assertEqual(late['episodes'][1]['known_at_sequence'],7)
        self.assertNotIn('episode-2-full_content',self.world.calls)
        self.assertNotIn('episode-1-summary_content',self.world.calls)
        self.assertEqual(late['episodes'][0]['member_resolution']['participants'][0]['status'],'resolved')
        self.assertEqual(late['episodes'][0]['member_resolution']['candidates'][0]['status'],'not_supplied')

    def test_deny_visibility_and_idempotent_confirmation_do_not_rewrite_original(self):
        accepted=self.replay(7);denied=self.replay(8);withheld=self.replay(9)
        self.assertEqual(denied['episodes'],accepted['episodes']);self.assertEqual(withheld['episodes'],accepted['episodes'])
        self.assertEqual(withheld['visibility']['episode-2'],'withheld')
        confirmed=EpisodeWorld(confirmation=True);self.world=confirmed
        result=self.replay(10)
        self.assertEqual(len(result['episodes']),2);self.assertEqual(result['authority_source_position'],2)
        self.assertEqual(result['episodes'][0]['records']['episode']['episode_state'],'accepted')

    def test_changed_future_tail_has_same_prefix_no_reads(self):
        left=self.replay(4);other=EpisodeWorld('alternate')
        self.world=other;right=self.replay(4)
        self.assertEqual(left,right)
        self.assertFalse(any('episode-2' in key for key in other.calls))

    def test_branch_prefix_suffix_gap_cycle_and_wrong_head_refuse(self):
        parent=self.world.cut(4);world=deepcopy(self.world.world)
        world['branches'].append({'branch_id':'fork','parent_cut':parent,'records':deepcopy(world['branches'][0]['records'][4:])})
        world=seal({k:v for k,v in world.items() if k!='world_sha256'},'world_sha256')
        fork=seal({**{k:v for k,v in self.world.cut(7).items() if k!='record_sha256'},'branch_id':'fork'})
        self.assertEqual(self.replay(world=world,cut=fork)['episodes'],self.replay()['episodes'])
        world['branches'][1]['records'][0]['sequence']=99;world=seal({k:v for k,v in world.items() if k!='world_sha256'},'world_sha256')
        with self.assertRaises(CognitiveKernelContractError):self.replay(world=world,cut=fork)
        wrong=seal({**{k:v for k,v in self.world.cut(4).items() if k!='record_sha256'},'head_sha256':'f'*64})
        with self.assertRaises(CognitiveKernelContractError):self.replay(4,cut=wrong)

    def test_limits_and_foreign_profile_refuse_before_body_reads(self):
        from scripts.mfm.formation_world_contracts import ConstructionLimits
        self.world.calls=[]
        with self.assertRaises(CognitiveKernelContractError):self.replay(7,limits=ConstructionLimits(6,100000,500000))
        self.assertEqual(self.world.calls,[])
        world=deepcopy(self.world.world);world['namespace']='foreign';world=seal({k:v for k,v in world.items() if k!='world_sha256'},'world_sha256')
        with self.assertRaises(CognitiveKernelContractError):self.replay(world=world)

    def test_actual_multiple_successors_survive_without_winner_or_predecessor_rewrite(self):
        self.world=EpisodeWorld(alternatives=True)
        result=self.replay(10)
        self.assertEqual([r['ref_id'] for r in result['episodes']],['episode-1','episode-2','episode-3'])
        self.assertEqual([r['records']['episode']['generation'] for r in result['episodes']],[1,2,2])
        self.assertEqual(result['visibility']['episode-1'],'present')
        self.assertEqual(result['visibility']['episode-3'],'present')
        self.assertEqual(result['episodes'][0]['records']['episode']['episode_state'],'accepted')

    def test_self_consistent_store_originals_cannot_acquire_missing_predecessor_or_members(self):
        for kwargs,n in [({'predecessor':'episode-unknown'},7),({'revision_generation':3},7),({'evidence':'source-unknown'},4),({'claim_version':'version-unknown'},4)]:
            with self.subTest(kwargs=kwargs):
                self.world=EpisodeWorld(**kwargs)
                with self.assertRaises(CognitiveKernelContractError):self.replay(n)
                self.assertFalse(any(k.endswith(('full_content','summary_content')) for k in self.world.calls))

    def test_branch_cycle_is_rejected_before_any_artifact_callback(self):
        world=deepcopy(self.world.world)
        parent=seal({**{k:v for k,v in self.world.cut(0).items() if k!='record_sha256'},'branch_id':'loop'})
        world['branches'].append({'branch_id':'loop','parent_cut':parent,'records':[]})
        world=seal({k:v for k,v in world.items() if k!='world_sha256'},'world_sha256');self.world.calls=[]
        with self.assertRaises(CognitiveKernelContractError):self.replay(world=world)
        self.assertEqual(self.world.calls,[])

if __name__=='__main__':unittest.main()
