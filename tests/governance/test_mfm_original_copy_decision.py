"""Actual opened-source copy consumer controls; synthetic mechanics, no learning."""
from copy import deepcopy
import importlib
import importlib.util
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, CognitiveKernelContractError
from cognitive_kernel import formation_decision_v16 as local
from scripts.mfm import formation_joint_target_alignment as original
from tests.governance import test_mfm_reference_decision as fixture


class OriginalCopyDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ReferenceDecisionTests.setUpClass()
        cls.h = fixture.ReferenceDecisionTests()
        cls.processor = cls.h.processor

    def api(self):
        names = ('cognitive_kernel.formation_original_copy_decision',
                 'scripts.mfm.formation_original_copy_target_alignment')
        for name in names:
            self.assertIsNotNone(importlib.util.find_spec(name), 'missing actual original copy consumer')
        return tuple(importlib.import_module(name) for name in names)

    def material(self, C, A, *, selected=True):
        example, pair, choices = self.h.complete_pair()
        ref = example.target.proposals[0].base.evidence_refs[0]
        raw = dict(example.opened_sources)[ref]
        value = example.target.proposals[0].base.value_text.encode()
        start = len(value) - len(raw)
        pair['literal_declarations'] = [{'pointer': '/proposals/0/proposal/value_text',
            'target_value_utf8': [start, len(value)], 'source_occurrences':
            [{'source_ref': ref, 'original_utf8': [0, len(raw)]}] if selected else None}]
        pair['literal_segments'] = original.literal_rows(pair['geometry'], pair['tokens'], self.h.frozen,
                                                         pair['literal_declarations'])
        pair['record_sha256'] = canonical_sha256({k:v for k,v in pair.items() if k != 'record_sha256'})
        inventory = C.bind_original_copy_inventory(self.h.bundle['caller'], self.h.joint,
            caller_sha256=self.h.binding['caller_sha256'], joint_alignment_sha256=self.h.joint['record_sha256'],
            expected_binding=self.h.binding)
        view = A.original_copy_decision_view(example, pair, self.h.frozen, pair_sha256=pair['record_sha256'],
            frozen_input_sha256=self.h.frozen['record_sha256'], inventory=inventory)
        return example, pair, choices, inventory, view

    def compile(self, C, example, choices, inventory, view, copy_rows=None):
        rows = [dict(pointer=r['pointer'], **r['alternatives'][0]) for r in view['authored_copy_tuples']['rows']]
        return C.compile_original_copy_decision(example.context, example.opened_sources, view['decision'],
            {'binding': C.copy_binding(inventory), 'rows': rows if copy_rows is None else copy_rows},
            {'binding': self.h.binding, 'rows': [dict(pointer=p,candidate_ordinal=v) for p,v in choices.items()]},
            caller=self.h.bundle['caller'], caller_sha256=self.h.binding['caller_sha256'],
            joint_alignment=self.h.joint, joint_alignment_sha256=self.h.joint['record_sha256'],
            expected_binding=C.copy_binding(inventory), artifact_sha256='a'*64, inference_run_id='copy-mechanics')

    def test_actual_complete_partition_reconstructs_and_grounds_original(self):
        C,A = self.api(); ex,pair,choices,inventory,view = self.material(C,A)
        self.assertEqual(set(inventory['source_geometry']), {'token_count','attention_mask'})
        self.assertEqual(set(C.copy_binding(inventory)['reference_binding']), set(self.h.binding))
        parts = view['decision']['proposals'][0]['proposal']['value_text']['parts']
        self.assertEqual(parts[-1], {'original_copy': True})
        out = self.compile(C,ex,choices,inventory,view)
        self.assertEqual(local.decision_from_bundle(out['bundle']), local.decision_record(ex))
        prov = out['copy_provenance'][0]
        self.assertEqual(prov['value_utf8'], pair['literal_declarations'][0]['target_value_utf8'])
        self.assertEqual(prov['original_utf8'], [0,inventory['sources'][0]['original_bytes']])

    def test_inventory_contains_every_actual_boundary_and_no_teacher(self):
        C,A = self.api(); _,_,_,inventory,_ = self.material(C,A)
        opened = dict(self.h.complete_pair()[0].opened_sources)
        for row in inventory['sources']:
            raw = opened[row['source_ref']]; offsets=[0]; n=0
            for char in raw.decode(): n += len(char.encode()); offsets.append(n)
            self.assertEqual([b['byte_offset'] for b in row['boundaries']], offsets)
            for b in row['boundaries']:
                off=b['byte_offset']
                self.assertEqual(b['original_byte_window'], [raw[j] if 0<=j<len(raw) else 256 for j in range(off-4,off+4)])
                self.assertEqual(len(b['coordinate_features']),8)
        self.assertNotIn('authored', canonical_json_bytes(inventory).decode())

    def test_whole_mixed_geometry_is_shared_and_actual_ids_guarded(self):
        C,A = self.api(); _,_,_,inventory,view = self.material(C,A)
        teacher=A.align_original_copy_decision_target(self.processor,view,view_sha256=view['record_sha256'],max_target_tokens=20000)
        generated=A.align_generated_original_copy_decision(self.processor,view['decision'],teacher['body_token_ids'],inventory,
            max_target_tokens=20000,eos_observed=True)
        self.assertEqual(teacher['requests'],generated['requests'])
        self.assertEqual(teacher['copy_requests'],generated['copy_requests'])
        self.assertEqual(generated['geometry'],teacher['geometry'])
        bad=list(teacher['body_token_ids']); bad[0]+=1
        with self.assertRaises(CognitiveKernelContractError):
            A.align_generated_original_copy_decision(self.processor,view['decision'],bad,inventory,max_target_tokens=20000,eos_observed=True)

    def test_unresolved_preserved_overlap_refused_and_undeclared_semantic(self):
        C,A=self.api(); ex,pair,_,inventory,view=self.material(C,A,selected=False)
        self.assertIsNone(view['authored_copy_tuples']['rows'][0]['alternatives'])
        bad=deepcopy(pair); bad['literal_declarations'].append(deepcopy(bad['literal_declarations'][0]))
        bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):
            A.original_copy_decision_view(ex,bad,self.h.frozen,pair_sha256=bad['record_sha256'],frozen_input_sha256=self.h.frozen['record_sha256'],inventory=inventory)
        pair['literal_declarations']=[];pair['literal_segments']=[]
        pair['record_sha256']=canonical_sha256({k:v for k,v in pair.items() if k!='record_sha256'})
        zero=A.original_copy_decision_view(ex,pair,self.h.frozen,pair_sha256=pair['record_sha256'],frozen_input_sha256=self.h.frozen['record_sha256'],inventory=inventory)
        self.assertEqual(zero['copy_leaves'],[])
        self.assertEqual(len(zero['decision']['proposals'][0]['proposal']['value_text']['parts']),1)

    def test_exact_actions_and_actual_citation_anchor_coverage_refuse(self):
        C,A=self.api(); ex,_,choices,inventory,view=self.material(C,A)
        row=dict(pointer=view['copy_leaves'][0]['pointer'],**view['authored_copy_tuples']['rows'][0]['alternatives'][0])
        for rows in ([],[row,row],[{**row,'source_slot':True}],[{**row,'end_byte':10**8}],[{**row,'source_slot':len(inventory['sources'])}]):
            with self.subTest(rows=rows),self.assertRaises(CognitiveKernelContractError): self.compile(C,ex,choices,inventory,view,rows)
        bad=deepcopy(view);bad['decision']['proposals'][0]['proposal']['anchors'][0]['end_byte']=1
        with self.assertRaises(CognitiveKernelContractError):self.compile(C,ex,choices,inventory,bad)
        raw=dict(ex.opened_sources)[inventory['sources'][row['source_slot']]['source_ref']]
        split=raw.index('é'.encode())+1
        with self.assertRaises(CognitiveKernelContractError):self.compile(C,ex,choices,inventory,view,[{**row,'start_byte':split}])
        bad=deepcopy(view);bad['decision']['proposals'][0]['proposal']['value_text']['parts'][-1]['source_slot']=0
        with self.assertRaises(CognitiveKernelContractError):self.compile(C,ex,choices,inventory,bad)

    def test_missing_contextual_atom_and_stale_or_changed_original_geometry_refuse(self):
        C,A=self.api(); _,_,_,inventory,_=self.material(C,A)
        original_joint=self.h.joint
        first=original_joint['sources'][0]['characters'][0]['rendered_chars']
        for mode in ('missing','special','geometry','boolean','stale'):
            bad=deepcopy(original_joint)
            if mode=='geometry':bad['sources'][0]['characters'][0]['original_utf8'][1]+=1
            elif mode=='boolean':bad['sources'][0]['characters'][0]['original_utf8'][0]=False
            elif mode=='stale':bad['rendered_utf8_sha256']='f'*64
            elif mode in ('missing','special'):
                for t in bad['tokens']:
                    if t['rendered_chars'][0]<first[1] and t['rendered_chars'][1]>first[0]:
                        if mode=='special':t['special_mask']=1
                        else:t['attention_mask']=0;bad['attention_mask'][t['token_index']]=0
            bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
            pin=original_joint['record_sha256'] if mode=='stale' else bad['record_sha256']
            with self.subTest(mode=mode),self.assertRaises(CognitiveKernelContractError):
                C.bind_original_copy_inventory(self.h.bundle['caller'],bad,caller_sha256=self.h.binding['caller_sha256'],
                    joint_alignment_sha256=pin,expected_binding=self.h.binding)

    def test_resealed_teacher_view_missing_actual_marker_set_refuses(self):
        C,A=self.api(); _,_,_,_,view=self.material(C,A)
        bad=deepcopy(view);bad['copy_leaves']=[];bad['authored_copy_tuples']['rows']=[]
        bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):
            A.align_original_copy_decision_target(self.processor,bad,view_sha256=bad['record_sha256'],max_target_tokens=20000)

    def test_sealed_inventory_nested_packet_and_boolean_coordinates_refuse(self):
        C,A=self.api(); _,_,_,inventory,view=self.material(C,A)
        for change in (
            lambda i:i['sources'][0]['boundaries'][0].update(byte_offset=False),
            lambda i:i['source_geometry'].update(target_token_count=1),
            lambda i:i['sources'][0].update(target_literal='forbidden'),
            lambda i:i['sources'][0]['boundaries'][0]['coordinate_features'].__setitem__(0,float('nan')),
        ):
            bad=deepcopy(inventory);change(bad)
            if any(x!=x for x in bad['sources'][0]['boundaries'][0]['coordinate_features']):
                # A nonfinite packet cannot even obtain a valid canonical JSON seal.
                with self.assertRaises((ValueError,CognitiveKernelContractError)):
                    bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
                continue
            bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
            with self.assertRaises(CognitiveKernelContractError): C.copy_requests(view['decision'],bad)

    def test_selected_complete_alternatives_and_adjacent_parts_keep_occurrences(self):
        C,A=self.api(); ex,pair,choices,inventory,_=self.material(C,A)
        ref=ex.target.proposals[0].base.evidence_refs[0];raw=dict(ex.opened_sources)[ref]
        value=ex.target.proposals[0].base.value_text.encode();start=len(value)-len(raw)
        occurrences=[i for i,b in enumerate(raw) if b==ord('e')]
        self.assertGreaterEqual(len(occurrences),2)
        first=occurrences[0]
        pair['literal_declarations']=[{'pointer':'/proposals/0/proposal/value_text',
            'target_value_utf8':[start+first,start+first+1],
            'source_occurrences':[{'source_ref':ref,'original_utf8':[i,i+1]} for i in occurrences[:2]]},
            {'pointer':'/proposals/0/proposal/value_text','target_value_utf8':[start+first+1,start+first+2],
             'source_occurrences':[{'source_ref':ref,'original_utf8':[first+1,first+2]}]}]
        pair['literal_segments']=original.literal_rows(pair['geometry'],pair['tokens'],self.h.frozen,pair['literal_declarations'])
        pair['record_sha256']=canonical_sha256({k:v for k,v in pair.items() if k!='record_sha256'})
        view=A.original_copy_decision_view(ex,pair,self.h.frozen,pair_sha256=pair['record_sha256'],
            frozen_input_sha256=self.h.frozen['record_sha256'],inventory=inventory)
        parts=view['decision']['proposals'][0]['proposal']['value_text']['parts']
        self.assertEqual(parts[1:3],[{'original_copy':True},{'original_copy':True}])
        rows=view['authored_copy_tuples']['rows']
        self.assertEqual([(t['start_byte'],t['end_byte']) for t in rows[0]['alternatives']],[(i,i+1) for i in occurrences[:2]])
        predictions=[dict(pointer=r['pointer'],**r['alternatives'][-1]) for r in rows]
        result=self.compile(C,ex,choices,inventory,view,predictions)
        self.assertEqual(result['bundle'].proposals[0].base.value_text,ex.target.proposals[0].base.value_text)
        self.assertEqual(result['copy_provenance'][0]['original_utf8'],[occurrences[1],occurrences[1]+1])

    def test_actual_j4_mixed_edge_context_unicode_newline_and_empty_source(self):
        C,A=self.api()
        from tokenizers import Tokenizer,decoders,models,pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from tests.governance.mfm_world_fixtures import World
        class EdgeProcessor(fixture.NativeProcessor):
            def __init__(self):
                super().__init__();vocab=self.tokenizer.get_vocab()
                vocab['\\"']=len(vocab);vocab['\\"O']=len(vocab)
                backend=Tokenizer(models.BPE(vocab=vocab,merges=[('\\','"'),('\\"','O')],unk_token='<unk>'))
                backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
                backend.decoder=decoders.ByteLevel()
                self.tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<s>',eos_token='</s>',pad_token='<pad>',unk_token='<unk>')
        class EdgeFixture(fixture.ReferenceDecisionTests): pass
        observation=World.observation
        def original_fixture_source(world,ledger,n,text):
            return observation(world,ledger,n,'Owner é "q" \\\n😀 Owner' if n==0 else '')
        with patch.object(World,'observation',original_fixture_source),patch.object(fixture,'NativeProcessor',EdgeProcessor):
            EdgeFixture.setUpClass()
        self.h=EdgeFixture();self.processor=self.h.processor
        ex,_,choices,inventory,view=self.material(C,A)
        empty=inventory['sources'][1]
        self.assertEqual(empty['original_bytes'],0);self.assertEqual(empty['source_token_indices'],[])
        self.assertEqual(len(empty['boundaries']),1)
        self.assertEqual(empty['boundaries'][0]['original_byte_window'],[256]*8)
        self.assertEqual(empty['boundaries'][0]['coordinate_features'][-2:],[1.,1.])
        first=self.h.joint['sources'][0]['characters'][0]['rendered_chars']
        supports=inventory['sources'][0]['boundaries'][0]['right_token_indices']
        edge=[self.h.joint['tokens'][i] for i in supports if self.h.joint['tokens'][i]['rendered_chars'][0]<first[0]]
        self.assertTrue(edge);self.assertTrue(all(t['observed_source_eligible'] is False for t in edge))
        out=self.compile(C,ex,choices,inventory,view)
        self.assertEqual(out['bundle'].proposals[0].base.value_text,ex.target.proposals[0].base.value_text)
        copied=dict(ex.opened_sources)[out['copy_provenance'][0]['source_ref']]
        self.assertIn(b'\n',copied);self.assertIn('😀'.encode(),copied)

    def test_teacher_reference_packet_matches_actual_whole_wire_and_authored_rows(self):
        C,A=self.api(); _,_,_,_,view=self.material(C,A)
        for change in (
            lambda v:(v.update(reference_leaves=[],authored_ordinals=[])),
            lambda v:v['reference_leaves'].pop(),
            lambda v:v['reference_leaves'].append(deepcopy(v['reference_leaves'][0])),
            lambda v:v['reference_leaves'][0].update(target_literal='forbidden'),
            lambda v:v['authored_ordinals'][0].update(candidate_ordinal=True),
            lambda v:v['authored_ordinals'].pop(),
            lambda v:v['authored_ordinals'][0].update(candidate_ordinal=10**9),
        ):
            bad=deepcopy(view);change(bad)
            bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
            with self.assertRaises(CognitiveKernelContractError):
                A.align_original_copy_decision_target(self.processor,bad,view_sha256=bad['record_sha256'],max_target_tokens=20000)


if __name__ == '__main__': unittest.main()
