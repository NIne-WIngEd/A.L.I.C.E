"""Complete original semantics with custody-only companion; no learning claim."""
from base64 import b64decode
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import unittest
from unittest.mock import patch
from tokenizers import processors
from cognitive_kernel.canonical import CognitiveKernelContractError,canonical_json_bytes,canonical_sha256
from scripts.mfm import formation_joint_input_v3 as OLD,formation_source_alignment as S,formation_caller_meaning_v3 as M
from tests.governance.test_mfm_world_caller_v3 import build as caller_build
from tests.governance.test_mfm_joint_input import source_identity
from tests.governance.test_mfm_source_alignment import Processor
from tests.governance.mfm_task_content_fixtures import TaskWorld,seal


class CountProcessor(Processor):
    def __init__(self):super().__init__();self.calls=0
    def apply_chat_template(self,*a,**kw):self.calls+=1;return super().apply_chat_template(*a,**kw)


class CompleteSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=caller_build()[5]

    def api(self):return importlib.import_module('scripts.mfm.formation_joint_input_v4')
    def arguments(self,p,b=None):
        b=b or self.bundle;J=self.api();policy=J.normative_joint_render_policy_v4();identity=source_identity()
        return dict(caller_sha256=b['caller']['record_sha256'],meaning_sha256=b['meaning']['record_sha256'],
            render_policy=policy,render_policy_sha256=policy['record_sha256'],processor_binding=S.processor_binding(p),
            source_identity=identity,source_identity_sha256=identity['record_sha256'],max_source_tokens=200000)
    def render(self,b=None):
        b=b or self.bundle;J=self.api();policy=J.normative_joint_render_policy_v4()
        return J._render(b['caller'],b['meaning'],b['caller']['record_sha256'],b['meaning']['record_sha256'],policy,policy['record_sha256'])
    def process(self,p=None,b=None):
        p=p or Processor();b=b or self.bundle;kw=self.arguments(p,b)
        batch,row=self.api().joint_source_batch_with_alignment_v4(p,b['caller'],b['meaning'],**kw)
        return p,b,kw,batch,row

    def test_complete_bodies_and_historical_relationships_keep_exact_original_values(self):
        J=self.api();b=self.bundle;messages,text,blocks,fields,sources,parts=self.render()
        values=[json.loads(json.loads(r['json_string'])) for r in blocks]
        semantic=M.semantic_view_v3(b['meaning'],b['caller'],meaning_sha256=b['meaning']['record_sha256'],caller_sha256=b['caller']['record_sha256'])
        for expected in semantic:
            actual=next(v for v in values if v['section']=='candidate_meaning' and v['slot']==expected['slot'])
            self.assertEqual({k:v for k,v in actual.items() if k!='section'}, {k:v for k,v in expected.items() if k!='original_parts'})
            if actual['body_file_sha256'] is not None:self.assertEqual(sha256(canonical_json_bytes(actual['content'])).hexdigest(),actual['body_file_sha256'])
        self.assertEqual(values[0]['context'],json.loads(json.loads(OLD._render(b['caller'],b['meaning'],b['caller']['record_sha256'],b['meaning']['record_sha256'])[2][0]['json_string']))['context'])
        self.assertEqual(values[0]['claim_snapshot'],b['meaning']['claim_snapshot'])
        for name in ('episode_snapshot','task_snapshot'):
            self.assertEqual({k:v for k,v in values[0][name].items() if k!='historical_prefix_proof'}, {k:v for k,v in b['meaning'][name].items() if k!='historical_prefix_proof'})
        self.assertEqual([r['ref_id'] for r in sources],['source-0','source-1'])
        self.assertTrue({'mission','attempt','result','outcome','workspace','uncertainty'}<={p['family'] for p in parts})

    def test_only_extra_custody_projections_leave_text_full_inventory_stays_in_companion(self):
        b=self.bundle;_,text,blocks,fields,sources,parts=self.render()
        old=OLD._render(b['caller'],b['meaning'],b['caller']['record_sha256'],b['meaning']['record_sha256'])
        def custody(rows):
            return [{**{k:v for k,v in r.items() if k not in ('rendered_chars','characters')},
                'characters':[a['original_utf8'] for a in r['characters']]} for r in rows]
        self.assertEqual(custody(parts),custody(old[-1]));self.assertLess(len(text.encode()),len(old[1].encode()))
        values=[json.loads(json.loads(r['json_string'])) for r in blocks]
        for name in ('episode_snapshot','task_snapshot'):
            source=b['meaning'][name]['historical_prefix_proof'];actual=values[0][name]['historical_prefix_proof']
            artifacts=[r['artifact'] for r in source['original_artifacts']]
            self.assertEqual(actual,dict(schema=source['schema'],record_sha256=source['record_sha256'],cut_sha256=source['cut']['record_sha256'],
                world_sha256=source['world']['world_sha256'],artifact_count=len(artifacts),artifact_roster_sha256=canonical_sha256(artifacts)))
        self.assertFalse(any('/original_parts' in f['pointer'] or '/historical_prefix_proof/original_artifacts' in f['pointer'] for f in fields))
        self.assertFalse(any('original_parts' in v for v in values if v['section']=='candidate_meaning'))
        self.assertTrue(any('/record/mission_title/file_sha256' in f['pointer'] for f in fields))
        self.assertNotIn('base64',json.dumps(values));self.assertFalse(any(p['part_name']=='raw-prefix-proof' for p in parts))

    def test_actual_contextual_ids_masks_all_scalar_original_joins_and_reload(self):
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',side_effect=AssertionError('teacher inaccessible')):
            p,b,kw,batch,row=self.process()
            self.assertEqual(row,self.api().verify_joint_alignment_v4(p,b['caller'],b['meaning'],row,alignment_sha256=row['record_sha256'],**kw))
        self.assertEqual(row['input_ids'],batch['input_ids'][0].tolist());self.assertEqual(row['attention_mask'],batch['attention_mask'][0].tolist())
        self.assertTrue({'null','boolean','integer','number','string','array','object'}<={f['kind'] for f in row['fields'] if f['original_part_file_sha256']})
        self.assertTrue(all(min(c['index_token_indices'])>max(c['meaning_token_indices']) for c in row['candidates']))
        for t in row['tokens']:
            if any(row['fields'][i]['original_part_file_sha256'] for i in t['field_indices']):self.assertFalse(t['observed_source_eligible'])
        self.assertFalse(row['learning_qualified']);self.assertFalse(row['teacher_accessed']);self.assertFalse(row['authority_granted'])
        self.assertEqual(row['render_policy'],kw['render_policy'])

    def test_exact_byte_geometry_survives_escape_repetition_and_removed_custody_text(self):
        _,text,blocks,fields,sources,parts=self.render();count=0
        for part in parts:
            raw=b64decode(part['original_base64']);self.assertEqual(sha256(raw).hexdigest(),part['file_sha256'])
            if part['media_type']=='application/json':
                displayed=text[slice(*part['rendered_chars'])]
                decoded=json.loads('"'+displayed+'"').encode()
                self.assertEqual(decoded,raw)
            else:
                for atom in part['characters']:
                    displayed=text[slice(*atom['rendered_chars'])]
                    decoded=json.loads('"'+json.loads('"'+displayed+'"')+'"').encode()
                    self.assertEqual(decoded,raw[slice(*atom['original_utf8'])]);count+=1
        self.assertGreater(count,100)
        self.assertTrue(any('Café'.encode() in b64decode(p['original_base64']) for p in parts))

    def test_closed_external_policy_and_source_binding_refuse_before_processor(self):
        p=CountProcessor();kw=self.arguments(p);J=self.api();b=self.bundle
        bad=deepcopy(kw['render_policy']);bad['proof_projection']='arbitrary-omit';bad=seal({k:v for k,v in bad.items() if k!='record_sha256'})
        for changes in ({'render_policy':bad,'render_policy_sha256':bad['record_sha256']}, {'render_policy_sha256':'0'*64},
                        {'source_identity_sha256':'0'*64},{'processor_binding':{}},{'max_source_tokens':True}):
            with self.subTest(changes=changes),self.assertRaises(CognitiveKernelContractError):
                J.joint_source_batch_with_alignment_v4(p,b['caller'],b['meaning'],**{**kw,**changes})
        self.assertEqual(p.calls,0)

    def test_forged_proof_only_bytes_are_rejected_without_rendering_or_processor_effect(self):
        p=CountProcessor();b=deepcopy(self.bundle);original_text=self.render()[1]
        for name in ('episode_snapshot','task_snapshot'):
            proof=b['meaning'][name]['historical_prefix_proof'];proof['original_artifacts'][0]['base64']='Zm9yZ2Vk'
        kw=self.arguments(p,b)
        with self.assertRaises(CognitiveKernelContractError):self.api().joint_source_batch_with_alignment_v4(p,b['caller'],b['meaning'],**kw)
        self.assertEqual(p.calls,0);self.assertNotIn('Zm9yZ2Vk',original_text)

    def test_full_budget_specials_and_transient_settings_restored(self):
        p=Processor();p.padding=True;p.add_special_tokens=True
        p.tokenizer.backend_tokenizer.post_processor=processors.TemplateProcessing(single='<s> $A',special_tokens=[('<s>',257)])
        p.tokenizer.backend_tokenizer.enable_padding(length=64,direction='left');p.tokenizer.backend_tokenizer.enable_truncation(max_length=128)
        binding=S.processor_binding(p);p,b,kw,_,row=self.process(p);J=self.api();n=len(row['input_ids'])
        _,exact=J.joint_source_batch_with_alignment_v4(p,b['caller'],b['meaning'],**{**kw,'max_source_tokens':n});self.assertEqual(exact['input_ids'],row['input_ids'])
        with self.assertRaises(CognitiveKernelContractError):J.joint_source_batch_with_alignment_v4(p,b['caller'],b['meaning'],**{**kw,'max_source_tokens':n-1})
        self.assertEqual(S.processor_binding(p),binding);self.assertTrue(row['tokens'][0]['special_mask']);self.assertFalse(row['tokens'][0]['field_indices'])

    def test_changed_geometry_cache_policy_or_companion_cannot_reload(self):
        p,b,kw,_,row=self.process();J=self.api()
        for path in ('geometry','cache','policy'):
            changed=deepcopy(row)
            if path=='geometry':next(f for f in changed['fields'] if f['original_part_file_sha256'])['original_part_utf8'][0]+=1
            elif path=='cache':changed['cache']['provenance_sha256']='f'*64
            else:changed['render_policy']['code_sha256']='f'*64
            changed=seal({k:v for k,v in changed.items() if k!='record_sha256'})
            with self.subTest(path=path),self.assertRaises(CognitiveKernelContractError):J.verify_joint_alignment_v4(p,b['caller'],b['meaning'],changed,alignment_sha256=changed['record_sha256'],**kw)

    def test_non_host_and_missing_originals_remain_absent_no_current_promotion(self):
        for world in (TaskWorld(longitudinal=True,audience='non_host'),TaskWorld(longitudinal=True,content_status={'mission':'withheld','uncertainty':'revoked'})):
            b=caller_build(world=world)[5];_,text,blocks,fields,sources,parts=self.render(b)
            self.assertNotIn('NEVER-MODEL-PRIVATE',text);self.assertNotIn('restricted-node',text)
            self.assertFalse(any(p['part_name'] in ('attention','override-secret-reason') for p in parts))
            self.assertEqual([r['ref_id'] for r in sources],['source-0','source-1'])
            for candidate,row in zip(b['meaning']['candidate_inventory'],b['meaning']['candidate_meanings']):
                value=next(json.loads(json.loads(v['json_string'])) for v in blocks if v['candidate_slot']==row['slot'])
                if row['availability']!='present':self.assertEqual(value['content'],M.M._body(row))

if __name__=='__main__':unittest.main()
