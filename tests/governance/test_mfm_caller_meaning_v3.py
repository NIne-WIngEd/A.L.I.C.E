from copy import deepcopy
import unittest
from base64 import b64encode
from hashlib import sha256
from cognitive_kernel.canonical import canonical_json_bytes,CognitiveKernelContractError
from scripts.mfm import formation_caller_meaning_v3 as M
from tests.governance.test_mfm_world_caller_v3 import build
from tests.governance.mfm_task_content_fixtures import seal

class MeaningTests(unittest.TestCase):
    def rebuild(self,value,caller):return M.caller_meaning_v3_from_record(value,caller,meaning_sha256=value['record_sha256'],caller_sha256=caller['record_sha256'])
    def test_requested_present_typed_body_cannot_be_replaced_by_an_absence_claim(self):
        *_,bundle=build();value=deepcopy(bundle['meaning']);row=next(r for r in value['candidate_meanings'] if r['family']=='mission-content-v1')
        slot=row['slot'];value['candidate_meanings'][slot]={'slot':slot,'family':'absent-body-v1','availability':'unavailable','body_base64':None,'body_bytes':None,'body_sha256':None}
        value=seal({k:v for k,v in value.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.rebuild(value,bundle['caller'])
    def test_raw_original_role_and_future_known_at_mutations_refuse(self):
        *_,bundle=build()
        for change in ('raw','role','future'):
            value=deepcopy(bundle['meaning']);row=next(r for r in value['candidate_meanings'] if r['family']=='mission-content-v1')
            if change=='raw':row['original_parts'][0]['parts']['mission-title']['base64']=b64encode(b'wrong').decode()
            elif change=='role':row['family']='uncertainty-content-v1'
            else:value['task_snapshot']['content_refs'][0]['commit']['known_at_sequence']=999;value['task_snapshot']=seal({k:v for k,v in value['task_snapshot'].items() if k!='record_sha256'})
            value=seal({k:v for k,v in value.items() if k!='record_sha256'})
            with self.subTest(change=change),self.assertRaises(CognitiveKernelContractError):self.rebuild(value,bundle['caller'])
    def test_actor_success_and_independent_unmet_observation_remain_separate(self):
        *_,bundle=build();row=next(r for r in bundle['meaning']['candidate_meanings'] if r['family']=='mission-content-v1');body=M.M._body(row)
        result=next(r for r in body['contents'] if r['family']=='result');outcome=next(r for r in body['contents'] if r['family']=='outcome')
        self.assertEqual(result['parts']['capsule']['status'],'succeeded');self.assertEqual(outcome['record']['postcondition_reports'][0]['reported_state'],'unmet')
        self.assertNotIn('verified_success',body);self.assertFalse(row['observed_source_eligible'])

if __name__=='__main__':unittest.main()
