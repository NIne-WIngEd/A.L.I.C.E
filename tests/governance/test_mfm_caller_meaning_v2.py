from copy import deepcopy
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from tests.governance.mfm_episode_fixtures import Episodes,entry,seal


class MeaningV2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=Episodes();self.caller,self.claim,self.snapshot=self.fixture.meaning_inputs()
        from scripts.mfm import formation_episode_input as E,formation_caller_meaning_v2 as M
        self.E=E;self.M=M;self.policy=M.normative_view_policy_v2()
        binding=E.episode_input_binding()
        parsed=E.episode_originals(*self.fixture.raws(),**self.fixture.arguments(),contract_binding=binding,contract_binding_sha256=binding['record_sha256'])
        self.body={'schema':'mfm-episode-snapshot-meaning-v1',**parsed['records']}
        self.resolution=deepcopy(self.snapshot['episodes'][0]['member_resolution'])
        self.entries=[entry(0,'opened-source-v1'),entry(1,'source-subject-v1','metadata_only'),
            entry(2,'episode-snapshot-v1',body=self.body,original_parts=parsed['originals'],member_resolution=self.resolution)]

    def build(self,**changes):
        args={'caller_sha256':self.caller['record_sha256'],'claim_snapshot':self.claim,'claim_snapshot_sha256':self.claim['record_sha256'],
            'episode_snapshot':self.snapshot,'episode_snapshot_sha256':self.snapshot['record_sha256'],'candidate_meanings':self.entries,
            'view_policy':self.policy,'view_policy_sha256':self.policy['record_sha256']}
        return self.M.caller_meaning_v2_record(self.caller,**{**args,**changes})

    def test_complete_family_reload_original_parts_and_unknown_members(self):
        record=self.build()
        self.assertEqual(self.M.caller_meaning_v2_from_record(record,self.caller,meaning_sha256=record['record_sha256'],caller_sha256=self.caller['record_sha256']),record)
        semantic=self.M.semantic_view_v2(record,self.caller,meaning_sha256=record['record_sha256'],caller_sha256=self.caller['record_sha256'])
        self.assertEqual(semantic[2]['provenance_class'],'validated_episode_originals')
        self.assertEqual(semantic[2]['member_resolution']['participants'][0]['status'],'not_supplied')
        self.assertFalse(semantic[2]['independent_observation'])

    def test_source_subject_cannot_be_promoted_to_registered_participant(self):
        resolution=deepcopy(self.resolution);resolution['participants'][0].update(status='resolved',provenance_sha256='f'*64)
        snapshot=deepcopy(self.snapshot);snapshot['episodes'][0]['member_resolution']=resolution;snapshot=seal({k:v for k,v in snapshot.items() if k!='record_sha256'})
        entries=deepcopy(self.entries);entries[2]['member_resolution']=resolution
        with self.assertRaises(CognitiveKernelContractError):self.build(candidate_meanings=entries,episode_snapshot=snapshot,episode_snapshot_sha256=snapshot['record_sha256'])

    def test_changed_original_summary_future_metadata_and_arbitrary_policy_refuse(self):
        rows=deepcopy(self.entries);rows[2]['original_parts']['summary_content']['base64']='e30='
        with self.assertRaises(CognitiveKernelContractError):self.build(candidate_meanings=rows)
        snapshot=deepcopy(self.snapshot);snapshot['episodes'][0]['known_at_sequence']=snapshot['cut']['sequence']+1;snapshot=seal({k:v for k,v in snapshot.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.build(episode_snapshot=snapshot,episode_snapshot_sha256=snapshot['record_sha256'])
        policy=deepcopy(self.policy);policy['eligible_episode_states'].append('candidate');policy=seal({k:v for k,v in policy.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.build(view_policy=policy,view_policy_sha256=policy['record_sha256'])

    def test_withheld_body_absent_and_old_family_unknown_fields_refuse(self):
        caller,claim,snapshot=self.fixture.meaning_inputs(status='withheld')
        entries=[entry(0,'opened-source-v1'),entry(1,'source-subject-v1','metadata_only'),entry(2,'absent-body-v1','withheld')]
        value=self.M.caller_meaning_v2_record(caller,caller_sha256=caller['record_sha256'],claim_snapshot=claim,claim_snapshot_sha256=claim['record_sha256'],
            episode_snapshot=snapshot,episode_snapshot_sha256=snapshot['record_sha256'],candidate_meanings=entries,view_policy=self.policy,view_policy_sha256=self.policy['record_sha256'])
        self.assertIsNone(value['candidate_meanings'][2]['body_base64'])
        entries[0]['original_parts']={}
        with self.assertRaises(CognitiveKernelContractError):self.M.caller_meaning_v2_record(caller,caller_sha256=caller['record_sha256'],claim_snapshot=claim,claim_snapshot_sha256=claim['record_sha256'],episode_snapshot=snapshot,episode_snapshot_sha256=snapshot['record_sha256'],candidate_meanings=entries,view_policy=self.policy,view_policy_sha256=self.policy['record_sha256'])

if __name__=='__main__':unittest.main()
