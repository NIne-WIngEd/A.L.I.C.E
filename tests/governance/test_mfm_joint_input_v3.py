import unittest,json
from base64 import b64decode
from copy import deepcopy
from unittest.mock import patch
from cognitive_kernel.canonical import CognitiveKernelContractError
from tokenizers import processors
from scripts.mfm import formation_joint_input_v3 as J,formation_source_alignment as S
from tests.governance.test_mfm_world_caller_v3 import build as caller_build
from tests.governance.test_mfm_joint_input import source_identity
from tests.governance.test_mfm_source_alignment import Processor
from tests.governance.mfm_task_content_fixtures import seal,TaskWorld


def build(world=None,p=None):
    _,_,_,_,_,bundle=caller_build(world=world);p=p or Processor();identity=source_identity()
    kwargs=dict(caller_sha256=bundle['caller']['record_sha256'],meaning_sha256=bundle['meaning']['record_sha256'],processor_binding=S.processor_binding(p),
        source_identity=identity,source_identity_sha256=identity['record_sha256'],max_source_tokens=200000)
    batch,row=J.joint_source_batch_with_alignment_v3(p,bundle['caller'],bundle['meaning'],**kwargs)
    return p,bundle,kwargs,batch,row

class JointTests(unittest.TestCase):
    def test_actual_full_contextual_ids_masks_original_parts_and_reload(self):
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',side_effect=AssertionError('teacher inaccessible')):
            p,b,kw,batch,row=build()
            self.assertEqual(row['input_ids'],batch['input_ids'][0].tolist());self.assertEqual(row['attention_mask'],batch['attention_mask'][0].tolist())
            self.assertEqual(J.verify_joint_alignment_v3(p,b['caller'],b['meaning'],row,alignment_sha256=row['record_sha256'],**kw),row)
        self.assertFalse(row['teacher_accessed']);self.assertFalse(row['learning_qualified']);self.assertFalse(row['authority_granted'])
        parts=row['prior_original_parts'];self.assertTrue(any(r['media_type']=='application/octet-stream' for r in parts));self.assertTrue(any(r['family']=='result' for r in parts))
        self.assertTrue({'mission','attempt','result','outcome','workspace','uncertainty'}<={r['family'] for r in parts})
        fields=[f for f in row['fields'] if f['original_part_file_sha256'] is not None]
        self.assertTrue({'null','boolean','integer','number','array','object','string'}<={f['kind'] for f in fields})
        for token in row['tokens']:
            if any(row['fields'][i]['original_part_file_sha256'] is not None for i in token['field_indices']):self.assertFalse(token['observed_source_eligible'])
        self.assertTrue(all(min(c['index_token_indices'])>max(c['meaning_token_indices']) for c in row['candidates']))

    def test_text_original_utf8_atoms_roundtrip_escaping_and_repeated_content(self):
        _,b,_,_,row=build();count=0
        for part in row['prior_original_parts']:
            if part['media_type']!='application/octet-stream':continue
            raw=b64decode(part['original_base64'])
            self.assertEqual(part['bytes'],len(raw))
            for atom in part['characters']:
                text=row['rendered_text'][slice(*atom['rendered_chars'])]
                decoded=json.loads('"'+json.loads('"'+text+'"')+'"')
                self.assertEqual(decoded.encode(),raw[slice(*atom['original_utf8'])]);count+=1
        self.assertGreater(count,100)
        self.assertTrue(any('Café'.encode() in b64decode(p['original_base64']) for p in row['prior_original_parts']))

    def test_full_budget_binding_mutation_and_special_exclusion(self):
        p=Processor();p.add_special_tokens=True;p.tokenizer.backend_tokenizer.post_processor=processors.TemplateProcessing(single='<s> $A',special_tokens=[('<s>',257)])
        p,b,kw,_,row=build(p=p);n=len(row['input_ids'])
        _,exact=J.joint_source_batch_with_alignment_v3(p,b['caller'],b['meaning'],**{**kw,'max_source_tokens':n});self.assertEqual(exact['input_ids'],row['input_ids'])
        with self.assertRaises(CognitiveKernelContractError):J.joint_source_batch_with_alignment_v3(p,b['caller'],b['meaning'],**{**kw,'max_source_tokens':n-1})
        self.assertTrue(row['tokens'][0]['special_mask']);self.assertFalse(row['tokens'][0]['field_indices'])
        changed=deepcopy(row);next(f for f in changed['fields'] if f['original_part_file_sha256'])['original_part_utf8'][0]+=1;changed=seal({k:v for k,v in changed.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):J.verify_joint_alignment_v3(p,b['caller'],b['meaning'],changed,alignment_sha256=changed['record_sha256'],**kw)

    def test_non_host_proof_bytes_are_not_rendered_or_acquire_source_authority(self):
        _,b,_,_,row=build(TaskWorld(longitudinal=True,audience='non_host'))
        proof=b['meaning']['task_snapshot']['historical_prefix_proof']
        self.assertTrue(any(x['family']=='workspace-content-v1' for x in b['meaning']['candidate_meanings']))
        self.assertNotIn('NEVER-MODEL-PRIVATE',row['rendered_text'])
        self.assertNotIn('restricted-node',row['rendered_text'])
        for original in proof['original_artifacts']:self.assertNotIn(original['base64'],row['rendered_text'])
        self.assertFalse(any(p['part_name']=='attention' for p in row['prior_original_parts']))
        snapshot=json.loads(json.loads(row['blocks'][0]['json_string']))['task_snapshot']
        self.assertNotIn('world',snapshot['historical_prefix_proof']);self.assertNotIn('base64',json.dumps(snapshot))

if __name__=='__main__':unittest.main()
