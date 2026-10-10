from copy import deepcopy
import json
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError,canonical_json_bytes
from tests.governance.mfm_episode_fixtures import Episodes,seal


class EpisodeInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture=Episodes()
        from scripts.mfm import formation_episode_input as E
        self.E=E;self.binding=E.episode_input_binding()

    def args(self,n=1):
        return {**self.fixture.arguments(n),'contract_binding':self.binding,'contract_binding_sha256':self.binding['record_sha256']}

    def test_real_originals_distinct_generation_and_supplied_request_digest(self):
        result=self.E.episode_originals(*self.fixture.raws(),**self.args())
        self.assertTrue(result['body_bytes_verified'])
        self.assertEqual(result['records']['episode']['generation'],1)
        self.assertEqual(result['records']['episode']['envelope']['generation'],0)
        self.assertEqual(set(result['originals']),{'episode','full_content','summary_content'})
        self.assertNotEqual(result['invocation']['record_sha256'],result['receipt']['request_digest'])
        self.assertNotEqual(result['records']['full_content'],result['records']['summary_content'])

    def test_metadata_path_does_not_read_or_claim_body_verification(self):
        episode,full,summary,receipt,invocation,profile=self.fixture.raws()
        result=self.E.episode_transition_metadata(episode,receipt,invocation,profile,**self.args())
        self.assertFalse(result['body_bytes_verified'])
        self.assertNotIn('full_content',result['records'])

    def test_summary_corruption_and_duplicate_nonfinite_noncanonical_refuse(self):
        raws=list(self.fixture.raws());raws[2]=canonical_json_bytes({'summary':'different'})
        with self.assertRaises(CognitiveKernelContractError):self.E.episode_originals(*raws,**self.args())
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{ "x": 1 }'):
            refs=deepcopy(self.fixture.transitions[0]);refs['full_content']=self.fixture.add('bad-'+str(len(raw)),raw)
            mutated=list(self.fixture.raws());mutated[1]=raw
            with self.assertRaises(CognitiveKernelContractError):self.E.episode_originals(*mutated,**{**self.args(),'refs':refs})

    def test_receipt_invocation_profile_and_external_binding_mutation_refuse(self):
        for part,index,key,value in (('append_receipt',3,'generation',True),('append_invocation',4,'request_digest','e'*64),
                                     ('projection_profile',5,'canonical_authority',True)):
            refs=deepcopy(self.fixture.transitions[0]);raws=list(self.fixture.raws());record=json.loads(raws[index]);record[key]=value
            if 'record_sha256' in record:record=seal({k:v for k,v in record.items() if k!='record_sha256'})
            refs[part]=self.fixture.add('mutated-'+part,record);raws[index]=self.fixture.artifacts[refs[part]['artifact_id']]
            with self.assertRaises(CognitiveKernelContractError):self.E.episode_originals(*raws,**{**self.args(),'refs':refs})
        changed=seal({**{k:v for k,v in self.binding.items() if k!='record_sha256'},'authority_granted':True})
        with self.assertRaises(CognitiveKernelContractError):self.E.episode_originals(*self.fixture.raws(),**{**self.args(),'contract_binding':changed,'contract_binding_sha256':changed['record_sha256']})

if __name__=='__main__':unittest.main()
