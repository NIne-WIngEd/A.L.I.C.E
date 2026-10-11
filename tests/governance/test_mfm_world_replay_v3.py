import unittest
from copy import deepcopy
from cognitive_kernel.canonical import CognitiveKernelContractError
from scripts.mfm import formation_world_replay_v3 as R,formation_world_contracts as C
from tests.governance.mfm_task_content_fixtures import TaskWorld,VerificationAdapter,seal


class ReplayTests(unittest.TestCase):
    def replay(self,world,n=None,mode='supplied_bytes_only',adapter=None):
        cut=world.cut(n);binding=R.world_contract_binding_v3();args=dict(world_sha256=world.world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],
            limits=C.ConstructionLimits(128,2097152,2097152),contract_binding=binding,contract_binding_sha256=binding['record_sha256'],verification_mode=mode)
        if mode=='supplied_bytes_only':args['supplied_artifacts']=world.artifacts
        else:
            boundary=world.verification_boundary();args.update(verification_adapter=adapter,verification_boundary=boundary,verification_boundary_sha256=boundary['record_sha256'])
        return R.replay_world_prefix_v3(world.world,**args)

    def test_actual_custody_ledger_and_exact_earlier_original_dependencies(self):
        world=TaskWorld();view,receipt=self.replay(world)
        self.assertEqual(len(view['input_contents']),6);self.assertEqual(view['authority_source_position'],2)
        self.assertEqual(view['input_contents'][0]['commit']['content_store_sequence'],1)
        self.assertGreater(view['input_contents'][0]['known_at_sequence'],1)
        self.assertEqual(next(x for x in view['registry_metadata'] if x['ref_id']=='uncertainty-1')['record_sha256'],world.content_known['uncertainty']['content']['record_sha256'])
        self.assertEqual(receipt['reads'],[]);self.assertFalse(receipt['current_permission_established']);self.assertFalse(view['authenticated_service_claimed'])

    def test_identity_content_cut_does_not_attach_body_until_explicit_binding(self):
        world=TaskWorld();early,_=self.replay(world,world.before_binding);later,_=self.replay(world)
        self.assertEqual(early['uncertainty_bindings'],[]);self.assertEqual(next(x for x in early['registry_metadata'] if x['ref_id']=='uncertainty-1')['record_sha256'],'d'*64)
        self.assertEqual(len(later['uncertainty_bindings']),1)
        self.assertEqual(early,R.prefix_proof_replay_v3(early['historical_prefix_proof'],cut=early['cut'])[0])

    def test_future_known_content_ref_refuses_even_when_body_and_custody_exist(self):
        world=TaskWorld(future_dependency=True)
        with self.assertRaises(CognitiveKernelContractError):self.replay(world)

    def test_real_longitudinal_versions_keep_graph_action_and_observation_counters_distinct(self):
        world=TaskWorld(longitudinal=True);view,_=self.replay(world)
        mission=[r for r in view['input_contents'] if r['family']=='mission'];attempt=[r for r in view['input_contents'] if r['family']=='attempt']
        uncertainty=[r for r in view['input_contents'] if r['family']=='uncertainty']
        self.assertEqual([r['content_version'] for r in mission],[1,2]);self.assertEqual(mission[1]['validated']['typed']['graph']['revision'],2)
        self.assertEqual([r['record']['phase'] for r in attempt],['planned','attempted']);self.assertEqual([r['record']['state'] for r in uncertainty],['open','resolved','reopened'])
        self.assertEqual(attempt[1]['record']['mission_content_ref'],mission[0]['known_content_ref'])
        self.assertEqual(uncertainty[1]['record']['resolution_observations'],uncertainty[2]['record']['resolution_observations'])
        self.assertNotIn('winner',uncertainty[2]['record']);self.assertEqual(view,R.prefix_proof_replay_v3(view['historical_prefix_proof'],cut=view['cut'])[0])

    def test_authentic_unrelated_event_cannot_replace_resolution_original_ancestry(self):
        world=TaskWorld(longitudinal=True,wrong_resolution_ancestry=True)
        with self.assertRaises(CognitiveKernelContractError):self.replay(world)

    def test_actual_reopened_node_corrected_observation_and_early_cut_are_not_current_truth(self):
        world=TaskWorld(longitudinal=True,complete_trajectory=True);view,_=self.replay(world)
        missions=[x for x in view['input_contents'] if x['family']=='mission'];observations=[x for x in view['input_contents'] if x['family']=='outcome']
        self.assertEqual([x['validated']['typed']['graph']['nodes'][0]['status'] for x in missions],['active','active','completed','active'])
        self.assertEqual([x['record']['postcondition_reports'][0]['reported_state'] for x in observations],['unmet','met'])
        self.assertEqual(observations[1]['record']['supersedes_observation_refs'],[observations[0]['known_content_ref']])
        self.assertEqual(observations[1]['record']['postcondition_reports'][1]['reported_state'],'unknown')
        early,_=self.replay(world,world.case_cuts['actor'])
        self.assertFalse(any(x['family']=='outcome' for x in early['input_contents']))
        self.assertEqual(early,R.prefix_proof_replay_v3(early['historical_prefix_proof'],cut=early['cut'])[0])
        self.assertTrue(all(x['known_at_sequence']<=world.case_cuts['actor'] for x in early['input_contents']))

    def test_current_verification_checks_each_io_and_stops_after_first_changed_boundary(self):
        world=TaskWorld();adapter=VerificationAdapter(world)
        def change(adapter,ref):adapter.boundary['policy_sha256']='c'*64;adapter.boundary=seal({k:v for k,v in adapter.boundary.items() if k!='record_sha256'})
        adapter.after=change
        with self.assertRaises(CognitiveKernelContractError):self.replay(world,mode='current_verification_reads',adapter=adapter)
        self.assertEqual(len(adapter.calls),1)
        adapter=VerificationAdapter(world);view,receipt=self.replay(world,mode='current_verification_reads',adapter=adapter)
        self.assertTrue(receipt['current_service_read_claimed']);self.assertTrue(receipt['current_permission_established']);self.assertFalse(receipt['authenticated_service_claimed'])
        self.assertEqual(len(adapter.calls),len(receipt['reads']));self.assertEqual(view,self.replay(world)[0])

    def test_future_and_unselected_sibling_bytes_are_never_acquired(self):
        world=TaskWorld(longitudinal=True,complete_trajectory=True);n=world.case_cuts['actor'];expected=self.replay(world,n)[0]
        sibling=deepcopy(world.rows[n:])
        for row in sibling:
            row['artifact']={**row['artifact'],'artifact_id':'sibling-'+row['artifact']['artifact_id'],'file_sha256':'e'*64}
        changed=deepcopy(world.world);changed['branches'].append({'branch_id':'sibling','parent_cut':world.cut(n),'records':sibling})
        changed['branches'][0]['records'][-1]['artifact']['file_sha256']='f'*64
        world.world=seal({k:v for k,v in changed.items() if k!='world_sha256'},'world_sha256')
        adapter=VerificationAdapter(world);forbidden={r['artifact']['artifact_id'] for r in world.rows[n:]}|{r['artifact']['artifact_id'] for r in sibling}
        adapter.after=lambda a,ref:self.assertNotIn(ref['artifact_id'],forbidden)
        actual,_=self.replay(world,n,mode='current_verification_reads',adapter=adapter)
        self.assertEqual(actual,expected)

    def test_frozen_v2_world_semantics_and_originals_remain_original_under_new_profile(self):
        from tests.governance.mfm_episode_fixtures import EpisodeWorld
        from scripts.mfm import formation_world_replay_v2 as Old
        world=EpisodeWorld();cut=world.cut();limits=C.ConstructionLimits(128,2097152,2097152);old_binding=Old.world_contract_binding_v2()
        old=Old.replay_world_prefix_v2(world.world,world_sha256=world.world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],read_artifact=world.read,
            limits=limits,contract_binding=old_binding,contract_binding_sha256=old_binding['record_sha256'])
        binding=R.world_contract_binding_v3();view,_=R.replay_world_prefix_v3(world.world,world_sha256=world.world['world_sha256'],cut=cut,cut_sha256=cut['record_sha256'],
            limits=limits,contract_binding=binding,contract_binding_sha256=binding['record_sha256'],verification_mode='supplied_bytes_only',supplied_artifacts=world.artifacts)
        for key in ('current_claims','episodes','sources','registry_targets','registry_metadata','visibility','authority_source_position','experience_head'):
            self.assertEqual(view[key],old[key],key)

    def test_real_ledger_malformed_capture_refuses_before_later_body_parts(self):
        import json,tempfile
        from pathlib import Path
        from cognitive_kernel.experience import ExperienceEvent
        from cognitive_kernel.ledger_store import open_experience_ledger
        from tests.governance.mfm_task_content_fixtures import SCOPE,TIME
        for malformed in (False,True):
            with self.subTest(malformed=malformed):
                world=TaskWorld();originals=[json.loads(world.artifacts[r['artifact']['artifact_id']]) for r in world.rows]
                stop=next(i+1 for i,r in enumerate(originals) if r['payload_kind']=='input_content_commit')
                originals=originals[:stop];last=originals[-1];payload=deepcopy(last['payload'])
                commit=json.loads(world.artifacts[payload['commit_artifact']['artifact_id']])
                if malformed:
                    commit['raw_buffer_receipts'][-1]['reference']['byte_length']=str(commit['raw_buffer_receipts'][-1]['reference']['byte_length'])
                    commit=seal({k:v for k,v in commit.items() if k!='commit_sha256'},'commit_sha256')
                    payload['commit_artifact']=world.add('malformed-original-commit',commit)
                rows=[]
                with tempfile.TemporaryDirectory(prefix='mfm-original-replay-') as temporary:
                    root=Path(temporary);repo=root/'repo';repo.mkdir()
                    with open_experience_ledger(root/'ledger.sqlite3',scope=SCOPE,repository_root=repo,created_at=TIME) as ledger:
                        for i,record in enumerate(originals,1):
                            event=ExperienceEvent.from_metadata_record(record['event'])
                            if i==stop and malformed:
                                args={k:v for k,v in event.__dict__.items() if k not in ('event_id','event_sha256')}
                                event=ExperienceEvent.create(**{**args,'content_digest':C.canonical_sha256(payload)})
                            transaction=ledger.append_event(event,committed_at=TIME)
                            revised=seal({**{k:v for k,v in record.items() if k!='record_sha256'},'event':event.metadata_record(),
                                'entry_receipt':transaction.entries[0].record(),'payload':payload if i==stop else record['payload'],
                                'previous_record_sha256':world.genesis if i==1 else rows[-1]['record_sha256']})
                            rows.append({'sequence':i,'record_sha256':revised['record_sha256'],'artifact':world.add('replayed-original-'+str(i),revised)})
                        self.assertEqual(ledger.verify_integrity().entry_count,stop)
                world.rows=rows;world.world=seal({**{k:v for k,v in world.world.items() if k!='world_sha256'},
                    'branches':[{'branch_id':'main','parent_cut':None,'records':rows}]},'world_sha256')
                adapter=VerificationAdapter(world)
                if malformed:
                    with self.assertRaises(CognitiveKernelContractError):self.replay(world,mode='current_verification_reads',adapter=adapter)
                    self.assertIn('malformed-original-commit',adapter.calls)
                    self.assertNotIn('graph',adapter.calls)
                else:
                    view,_=self.replay(world,mode='current_verification_reads',adapter=adapter)
                    self.assertEqual(len(view['input_contents']),1);self.assertIn('graph',adapter.calls)

if __name__=='__main__':unittest.main()
