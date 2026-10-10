from copy import deepcopy
from hashlib import sha256
import importlib
import unittest
from cognitive_kernel.canonical import CognitiveKernelContractError
from tests.governance.mfm_world_fixtures import World, SCOPE, NAMESPACE

class WorldContractTests(unittest.TestCase):
    def api(self): return importlib.import_module('scripts.mfm.formation_world_contracts')

    def test_real_records_preserve_originals_and_independent_counters(self):
        world=World(); api=self.api(); limits=api.ConstructionLimits(30,100000,100000)
        record=api.read_json(world.rows[4]['artifact'],world.read,limits)
        result=api.claim_transition(record['payload'],world.read,limits,SCOPE,NAMESPACE,world.authority_profile)
        self.assertEqual(result['receipt'].store_sequence,2)
        self.assertEqual(result['projection'].authority_generation,1)
        self.assertEqual(result['projection'].projection_generation,2)
        self.assertEqual(result['version'].event_stream_position,None)
        self.assertEqual(record['sequence'],5)
        for key in ('identity','version','projection'):
            self.assertEqual(result['originals'][key],world.claim_raws[1][key])
            self.assertEqual(sha256(result['originals'][key]).hexdigest(),record['payload'][key]['file_sha256'])

    def test_original_json_and_artifact_fail_closed(self):
        world=World(); api=self.api(); limits=api.ConstructionLimits(30,100000,100000)
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{ "x": 1 }'):
            ref=world.add('malformed-'+str(len(raw)),raw)
            with self.assertRaises(CognitiveKernelContractError): api.read_json(ref,world.read,limits)
        ref=deepcopy(world.rows[1]['artifact']); ref['bytes']=True
        with self.assertRaises(CognitiveKernelContractError): api.read_json(ref,world.read,limits)
        ref=deepcopy(world.rows[1]['artifact']); ref['file_sha256']='0'*64
        with self.assertRaises(CognitiveKernelContractError): api.read_json(ref,world.read,limits)

    def test_wrong_state_receipt_fields_and_extra_authority_refuse(self):
        world=World(); api=self.api(); limits=api.ConstructionLimits(30,100000,100000)
        payload=api.read_json(world.rows[1]['artifact'],world.read,limits)['payload']
        for mutation in ('receipt','request','teacher'):
            changed=deepcopy(payload)
            if mutation=='teacher': changed['teacher_target']={}
            else:
                key='append_receipt' if mutation=='receipt' else 'append_request'
                value=api.read_json(payload[key],world.read,limits)
                value['claim_id' if mutation=='receipt' else 'expected_current_claim_version_id']='wrong'
                changed[key]=world.add('bad-'+mutation,value)
            with self.assertRaises(CognitiveKernelContractError):
                api.claim_transition(changed,world.read,limits,SCOPE,NAMESPACE,world.authority_profile)

    def test_declared_bounds_never_truncate(self):
        world=World(); api=self.api(); ref=world.rows[0]['artifact']
        self.assertEqual(api.read_bytes(ref,world.read,api.ConstructionLimits(7,ref['bytes'],9999)),world.artifacts[ref['artifact_id']])
        world.calls.clear()
        with self.assertRaises(CognitiveKernelContractError): api.read_bytes(ref,world.read,api.ConstructionLimits(7,ref['bytes']-1,9999))
        self.assertEqual(world.calls,[])
