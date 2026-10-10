from base64 import b64decode
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from tokenizers import processors
from cognitive_kernel.canonical import CognitiveKernelContractError
from tests.governance import test_mfm_source_alignment as sources,test_mfm_joint_input as original_joint
from tests.governance.test_mfm_world_caller_v2 import build as build_bundle
from tests.governance.mfm_episode_fixtures import EpisodeWorld,EpisodeReadAdapter,World,seal
from scripts.mfm import formation_source_alignment as S


def build(n=7,world=None,processor=None):
    _,world,view,request,adapter,_,bundle=build_bundle(n,world=world)
    from scripts.mfm import formation_joint_input_v2 as J
    p=processor or sources.Processor();identity=original_joint.source_identity()
    caller=bundle['caller'];meaning=bundle['meaning']
    kwargs=dict(caller_sha256=caller['record_sha256'],meaning_sha256=meaning['record_sha256'],processor_binding=S.processor_binding(p),
        source_identity=identity,source_identity_sha256=identity['record_sha256'],max_source_tokens=80000)
    batch,aligned=J.joint_source_batch_with_alignment_v2(p,caller,meaning,**kwargs)
    return J,p,bundle,kwargs,batch,aligned


class EpisodeJointTests(unittest.TestCase):
    def test_historical_proof_is_not_rendered_or_promoted_in_current_only_input(self):
        _,world,view,request,_,_,_=build_bundle()
        request['context']=World.context(world,('source-1',));request['opened_source_refs']=['source-1']
        request['prior_ref_ids']=['episode-1','episode-2']
        request=seal({k:v for k,v in request.items() if k!='record_sha256'})
        _,_,_,_,adapter,_,bundle=build_bundle(world=world,request=request,adapter=EpisodeReadAdapter(world,request))
        from scripts.mfm import formation_joint_input_v2 as J
        p=sources.Processor();identity=original_joint.source_identity()
        _,row=J.joint_source_batch_with_alignment_v2(p,bundle['caller'],bundle['meaning'],
            caller_sha256=bundle['caller']['record_sha256'],meaning_sha256=bundle['meaning']['record_sha256'],
            processor_binding=S.processor_binding(p),source_identity=identity,source_identity_sha256=identity['record_sha256'],max_source_tokens=80000)
        snapshot=json.loads(json.loads(row['blocks'][0]['json_string']))['episode_snapshot']
        projection=snapshot['historical_prefix_proof'];proof=bundle['meaning']['episode_snapshot']['historical_prefix_proof']
        self.assertNotIn('world',projection);self.assertNotIn('base64',json.dumps(projection))
        self.assertEqual(projection['record_sha256'],proof['record_sha256'])
        self.assertEqual(projection['original_artifacts'],[r['artifact'] for r in proof['original_artifacts']])
        for original in proof['original_artifacts']:self.assertNotIn(original['base64'],row['rendered_text'])
        self.assertFalse(any(f['pointer']=='/episode_snapshot/historical_prefix_proof/world' or
            f['pointer'].startswith('/episode_snapshot/historical_prefix_proof/world/') for f in row['fields']))
        self.assertNotIn(('owner','entity'),[(c['ref_id'],c['kind']) for c in row['candidates']])
        self.assertEqual([r['ref_id'] for r in row['sources']],['source-1'])
        self.assertNotIn('dark mode',row['rendered_text'])
        for token in row['tokens']:
            if any(row['fields'][i]['original_part'] is not None for i in token['field_indices']):
                self.assertFalse(token['observed_source_eligible'])

    def test_whole_ids_masks_rebuild_and_teacher_label_independence(self):
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',side_effect=AssertionError('teacher inaccessible')),\
             patch('scripts.mfm.v16_prior_memory_inputs.local_uncertainty_allocations',side_effect=AssertionError('local output inaccessible')):
            J,p,bundle,kwargs,batch,row=build()
            self.assertEqual(J.verify_joint_alignment_v2(p,bundle['caller'],bundle['meaning'],row,alignment_sha256=row['record_sha256'],**kwargs),row)
        self.assertEqual(row['input_ids'],batch['input_ids'][0].tolist());self.assertEqual(row['attention_mask'],batch['attention_mask'][0].tolist())
        self.assertFalse(row['teacher_accessed']);self.assertFalse(row['authority_granted']);self.assertFalse(row['learning_qualified'])
        self.assertIn('episode_snapshot',json.loads(row['blocks'][0]['json_string']))

    def test_every_original_part_field_slice_matches_distinct_canonical_bytes(self):
        _,_,bundle,_,_,row=build();parts=set();matched=0
        for field in row['fields']:
            if field['original_part'] is None:continue
            entry=bundle['meaning']['candidate_meanings'][field['candidate_slot']]
            original=entry['original_parts'][field['original_part']];raw=b64decode(original['base64'])
            self.assertEqual(field['original_part_file_sha256'],original['artifact']['file_sha256'])
            rendered=row['rendered_text'][slice(*field['rendered_chars'])]
            self.assertEqual(json.loads('"'+rendered+'"').encode(),raw[slice(*field['original_part_utf8'])])
            self.assertTrue(field['supporting_token_indices']);parts.add(field['original_part']);matched+=1
        self.assertEqual(parts,{'episode','full_content','summary_content'});self.assertGreater(matched,100)
        self.assertTrue(any(f['original_part']=='full_content' and '/literal~1~0' in f['pointer'] for f in row['fields']))
        self.assertTrue({'null','boolean','integer','number','array','object','string'}<=
            {f['kind'] for f in row['fields'] if f['original_part']=='full_content'})

    def test_prior_episode_tokens_never_gain_observation_authority_and_footer_comes_later(self):
        _,_,bundle,_,_,row=build()
        for token in row['tokens']:
            episode_links=[row['fields'][i] for i in token['field_indices'] if row['fields'][i]['original_part'] is not None]
            if episode_links:self.assertFalse(token['observed_source_eligible'])
        roles=[(c['ref_id'],c['kind'],c['origin']) for c in row['candidates']]
        self.assertIn(('owner','source_subject','source_subject'),roles);self.assertIn(('owner','entity','context_target'),roles)
        for candidate in row['candidates']:
            self.assertGreater(min(candidate['index_token_indices']),max(candidate['meaning_token_indices']))
        self.assertNotIn('historical-candidate-unsupplied',[c['ref_id'] for c in row['candidates']])

    def test_backdated_future_and_withheld_absence_preserve_causal_view(self):
        _,_,_,_,_,early=build(4)
        _,_,_,_,_,alternate=build(4,world=EpisodeWorld('different future'))
        self.assertEqual(early,alternate)
        _,_,_,_,_,late=build(7);self.assertNotEqual(early['input_ids'],late['input_ids'])
        _,_,bundle,_,_,withheld=build(9)
        slot=next(c['slot'] for c in withheld['candidates'] if c['ref_id']=='episode-2')
        self.assertFalse(any(f['candidate_slot']==slot and f['original_part'] is not None for f in withheld['fields']))
        self.assertIsNone(bundle['meaning']['candidate_meanings'][slot]['body_base64'])

    def test_exact_budget_complete_processor_binding_and_resealed_original_map_refuse(self):
        J,p,bundle,kwargs,_,row=build();caller=bundle['caller'];meaning=bundle['meaning'];n=len(row['input_ids'])
        _,exact=J.joint_source_batch_with_alignment_v2(p,caller,meaning,**{**kwargs,'max_source_tokens':n})
        self.assertEqual(exact['input_ids'],row['input_ids'])
        with self.assertRaises(CognitiveKernelContractError):J.joint_source_batch_with_alignment_v2(p,caller,meaning,**{**kwargs,'max_source_tokens':n-1})
        altered=deepcopy(row);next(f for f in altered['fields'] if f['original_part'] is not None)['original_part_utf8'][0]+=1
        altered=seal({k:v for k,v in altered.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):J.verify_joint_alignment_v2(p,caller,meaning,altered,alignment_sha256=altered['record_sha256'],**kwargs)
        with patch.object(p,'apply_chat_template',side_effect=AssertionError('invalid ancestry before processor')):
            with self.assertRaises(CognitiveKernelContractError):J.joint_source_batch_with_alignment_v2(p,caller,meaning,**{**kwargs,'caller_sha256':'f'*64})

    def test_specials_and_supported_execution_settings_restored_on_success_and_refusal(self):
        for fail in (False,True):
            p=sources.Processor();p.padding=True;p.add_special_tokens=True
            p.tokenizer.backend_tokenizer.post_processor=processors.TemplateProcessing(single='<s> $A',special_tokens=[('<s>',257)])
            p.tokenizer.backend_tokenizer.enable_padding(length=80000,direction='left');p.tokenizer.backend_tokenizer.enable_truncation(max_length=80001)
            before=S.processor_binding(p)
            if fail:
                original=p.apply_chat_template
                def corrupt(*args,**kwargs):
                    value=original(*args,**kwargs)
                    if kwargs.get('processor_kwargs',{}).get('return_offsets_mapping'):value['offset_mapping'][0,0,0]=-1
                    return value
                p.apply_chat_template=corrupt
                with self.assertRaises(CognitiveKernelContractError):build(processor=p)
            else:
                _,_,_,_,_,row=build(processor=p)
                self.assertEqual(row['input_ids'][0],257);self.assertEqual(row['tokens'][0]['field_indices'],[])
                self.assertFalse(row['tokens'][0]['observed_source_eligible'])
            self.assertEqual(S.processor_binding(p),before)

if __name__=='__main__':unittest.main()
