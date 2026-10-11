"""Public responsibility/literal component controls over actual byte-level IDs."""
from copy import deepcopy
from hashlib import sha256
import unittest
from cognitive_kernel.canonical import canonical_json_bytes,CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import supervised_output_record_v16
from scripts.mfm import formation_source_alignment as S,formation_target_alignment as T
from scripts.mfm.formation_caller_input import caller_input_record
from tests.governance.test_mfm_target_alignment import example_for_text
from tests.governance.test_mfm_source_alignment import Processor
from tests.governance.test_mfm_paired_teaching import api

class LiteralRoles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P=api();from scripts.mfm import formation_joint_target_alignment;cls.G=formation_joint_target_alignment;cls.ex=example_for_text('banana banana Café "A" \\ 😀',value='banana banana Café "A" \\ 😀')
        cls.caller=caller_input_record(cls.ex.context,cls.ex.opened_sources);p=Processor();binding=S.processor_binding(p)
        _,src=S.text_source_batch_with_alignment(p,cls.caller,caller_sha256=cls.caller['record_sha256'],processor_binding=binding,max_source_tokens=20000)
        record=supervised_output_record_v16(cls.ex);cls.g=T.canonical_field_geometry(record,expected_target_sha256=sha256(canonical_json_bytes(record)).hexdigest())
        obs=p.tokenizer(canonical_json_bytes(record).decode(),add_special_tokens=False,padding=False,truncation=False,return_offsets_mapping=True,return_attention_mask=True,return_special_tokens_mask=True)
        cls.tokens=T._token_rows(cls.g,obs['input_ids'],obs['attention_mask'],obs['offset_mapping'],obs['special_tokens_mask'])
        _,_,registry=S._caller(cls.caller,cls.caller['record_sha256'])
        cls.frozen={'caller':cls.caller,'joint':{'sources':src['sources'],'tokens':src['tokens']},'meaning':{'candidate_inventory':S.selection_candidates(cls.ex.context,cls.ex.opened_sources,registry)}}
    def declaration(self,span,selected=None):return [{'pointer':'/proposals/0/proposal/value_text','target_value_utf8':span,'source_occurrences':selected}]
    def test_repeated_overlap_and_escaped_unicode_keep_all_original_and_target_support(self):
        row=self.G.literal_rows(self.g,self.tokens,self.frozen,self.declaration([1,4]))[0]
        self.assertEqual([o['original_utf8'] for o in row['occurrences']],[[1,4],[3,6],[8,11],[10,13]])
        self.assertTrue(row['target_body_indices']);self.assertTrue(all(o['supporting_token_indices'] for o in row['occurrences']));self.assertIsNone(row['selected_occurrences']);self.assertFalse(row['token_id_copy_claimed'])
        raw=self.ex.target.proposals[0].base.value_text.encode();start=raw.index('😀'.encode())
        row=self.G.literal_rows(self.g,self.tokens,self.frozen,self.declaration([start,start+4]))[0];self.assertEqual(row['literal_utf8_hex'],'😀'.encode().hex())
    def test_partial_utf8_atom_no_original_and_wrong_occurrence_refuse(self):
        raw=self.ex.target.proposals[0].base.value_text.encode();start=raw.index('é'.encode())
        for declarations in (self.declaration([start+1,start+2]),self.declaration([0,1],[{'source_ref':'foreign','original_utf8':[0,1]}])):
            with self.subTest(declarations=declarations),self.assertRaises(CognitiveKernelContractError):self.G.literal_rows(self.g,self.tokens,self.frozen,declarations)
    def test_source_occurrence_requires_exact_integer_coordinates(self):
        valid=self.declaration([0,1],[{'source_ref':'source-1','original_utf8':[0,1]}]);self.G.literal_rows(self.g,self.tokens,self.frozen,valid)
        for coords in ([False,True],[0.0,1.0]):
            invalid=self.declaration([0,1],[{'source_ref':'source-1','original_utf8':coords}])
            with self.subTest(coords=coords),self.assertRaises(CognitiveKernelContractError):self.G.literal_rows(self.g,self.tokens,self.frozen,invalid)
    def test_candidate_ordinals_explicit_and_role_reclassification_refuses(self):
        fields=self.G.field_rows(self.g,self.frozen,[]);row=next(r for r in fields if r['pointer']=='/proposals/0/proposal/evidence_refs/0');index=row['matching_candidate_indices'][0]
        declared=[{'pointer':row['pointer'],'mode':'external_selection','candidate_ordinal':index}]
        bound=self.G.field_rows(self.g,self.frozen,declared);self.assertEqual(next(r for r in bound if r['pointer']==row['pointer'])['selected_candidate_ordinal'],index)
        for entry in ({**declared[0],'candidate_ordinal':True},{**declared[0],'candidate_ordinal':len(self.frozen['meaning']['candidate_inventory'])},
                      {'pointer':'/proposals/0/proposal/disposition_scope_ref','mode':'external_selection','candidate_ordinal':index}):
            with self.subTest(entry=entry),self.assertRaises(CognitiveKernelContractError):self.G.field_rows(self.g,self.frozen,[entry])
    def test_same_string_source_subject_and_registered_entity_keep_ambiguity(self):
        fields=self.G.field_rows(self.g,self.frozen,[]);subject=next(r for r in fields if r['pointer']=='/proposals/0/proposal/subject_ref')
        self.assertEqual(subject['candidate_status'],'ambiguous');self.assertGreater(len(subject['matching_candidate_indices']),1)
        participant=next(r for r in fields if r['pointer']=='/proposals/0/episode/participant_refs/0')
        self.assertTrue(all(self.frozen['meaning']['candidate_inventory'][i]['kind']=='entity' for i in participant['eligible_candidate_indices']))
        with self.assertRaises(CognitiveKernelContractError):self.G.field_rows(self.g,self.frozen,[{'pointer':participant['pointer'],'mode':'external_selection',
            'candidate_ordinal':next(i for i in subject['matching_candidate_indices'] if self.frozen['meaning']['candidate_inventory'][i]['kind']!='entity')}])

if __name__=='__main__':unittest.main()
