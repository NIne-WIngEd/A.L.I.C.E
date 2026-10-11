"""Private connected interface controls; fictional authored labels, no corpus."""
from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
import importlib.util, json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, CognitiveKernelContractError
from cognitive_kernel import formation_learning_v16 as L
from cognitive_kernel.formation_contracts import FormationProposal,FormationEvidenceAnchor
from cognitive_kernel.formation_semantics_v16 import FormationProposalV16
from scripts.mfm import formation_joint_input_v4 as J, formation_source_alignment as S, formation_field_responsibility as F
from tests.governance.test_mfm_world_caller_v3 import build
from tests.governance.test_mfm_joint_input import source_identity
from tests.governance.test_mfm_source_alignment import Processor

HERE=Path(__file__).resolve().parent
def api():
    from scripts.mfm import formation_paired_teaching
    return formation_paired_teaching

def seal(value):return {**value,'record_sha256':canonical_sha256(value)}
def reseal(value):return seal({k:v for k,v in value.items() if k!='record_sha256'})
def split(case='case-1',name='train',group='world-1'):
    return {'case_id':case,'split':name,'lineage':dict(world_id=group,branch_id='main',case_family=group,
        world_author_family=group,generator_family=group,rendering_family=group,source_system_family=group,
        ancestry_group_ids=[group],parent_case_ids=[])}

class Connected(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P=api();from scripts.mfm import formation_joint_target_alignment;cls.G=formation_joint_target_alignment;cls.world,_,_,_,_,cls.bundle=build();cls.p=Processor();identity=source_identity()
        cls.kw=dict(caller_sha256=cls.bundle['caller']['record_sha256'],meaning_sha256=cls.bundle['meaning']['record_sha256'],
            processor_binding=S.processor_binding(cls.p),source_identity=identity,source_identity_sha256=identity['record_sha256'],
            render_policy=J.normative_joint_render_policy_v4(),render_policy_sha256=J.normative_joint_render_policy_v4()['record_sha256'],max_source_tokens=200000)
        batch,cls.joint=J.joint_source_batch_with_alignment_v4(cls.p,cls.bundle['caller'],cls.bundle['meaning'],**cls.kw)
        cls.batch={k:batch[k].tolist() for k in ('input_ids','attention_mask')}
        cls.expected=dict(caller_sha256=cls.kw['caller_sha256'],meaning_sha256=cls.kw['meaning_sha256'],joint_sha256=cls.joint['record_sha256'],
            batch_sha256=canonical_sha256(cls.batch),joint_code_binding=J.joint_code_binding_v4(),processor_binding=cls.kw['processor_binding'],
            source_identity=identity,max_source_tokens=200000)
        cls.frozen=cls.P.freeze_input(cls.bundle['caller'],cls.bundle['meaning'],cls.joint,cls.batch,expected=cls.expected,split_row=split())
    def target(self,unknown=False):
        caller=self.bundle['caller'];context,opened,_=J.caller_api.caller_input_from_record(caller)
        return {'schema':L.TARGET_SCHEMA_V16,'context':context.record(),'source_ids':[r for r,_ in opened],
            'proposals':[],'dispositions':[{'scope_ref':'authored-scope','action':'abstain','evidence_refs':['source-0'],'target_refs':[]}],
            'adjudications':{k:('present' if k=='abstention' else 'unknown' if unknown and k=='relationship' else 'negative') for k in L.FULL_ROLE_DIMENSIONS}}
    def admitted(self,target=None):
        row=target or self.target();raw=canonical_json_bytes(row)
        return self.P.admit_target(self.frozen,expected_freeze_sha256=self.frozen['record_sha256'],expected=self.expected,
            target_reader=lambda:raw,expected_target_file_sha256=sha256(raw).hexdigest())
    def test_real_world_freeze_is_target_free_and_exact_independent_batch(self):
        with patch.object(L,'supervised_output_record_v16',side_effect=AssertionError('teacher accessed')) as teacher:
            row=self.P.freeze_input(self.bundle['caller'],self.bundle['meaning'],self.joint,self.batch,expected=self.expected,split_row=split())
        teacher.assert_not_called();self.assertEqual(row,self.frozen);self.assertFalse(row['teacher_accessed'])
        self.assertTrue({'mission','attempt','result','outcome','workspace','uncertainty'}<={r['family'] for r in self.joint['prior_original_parts']})
    def test_changed_batch_or_expected_code_refuses_before_teacher(self):
        for key in ('batch_sha256','caller_sha256'):
            expected={**self.expected,key:'0'*64};reader=Mock(side_effect=AssertionError('target read'))
            with self.subTest(key=key),self.assertRaises(CognitiveKernelContractError):
                self.P.admit_target(self.frozen,expected_freeze_sha256=self.frozen['record_sha256'],expected=expected,
                    target_reader=reader,expected_target_file_sha256='0'*64)
            reader.assert_not_called()
        bad=deepcopy(self.frozen);bad['batch']['input_ids'][0][0]+=1;bad=reseal(bad);reader=Mock()
        with self.assertRaises(CognitiveKernelContractError):
            self.P.admit_target(bad,expected_freeze_sha256=bad['record_sha256'],expected=self.expected,target_reader=reader,expected_target_file_sha256='0'*64)
        reader.assert_not_called()
    def test_changed_field_token_and_source_original_resealed_refuse(self):
        for key in ('fields','tokens','sources','prior_original_parts'):
            bad=deepcopy(self.joint);bad[key].pop();bad=reseal(bad)
            with self.subTest(key=key),self.assertRaises(CognitiveKernelContractError):
                self.P.freeze_input(self.bundle['caller'],self.bundle['meaning'],bad,self.batch,
                    expected={**self.expected,'joint_sha256':bad['record_sha256']},split_row=split())
    def test_final_refuses_before_target_reader(self):
        bad=deepcopy(self.frozen);bad['split_row']['split']='final';bad=reseal(bad);reader=Mock()
        with self.assertRaises(CognitiveKernelContractError):self.P.admit_target(bad,expected_freeze_sha256=bad['record_sha256'],expected=self.expected,target_reader=reader,expected_target_file_sha256='0'*64)
        reader.assert_not_called()
    def test_unknown_is_not_negative_and_has_no_decoder_labels(self):
        accepted=self.admitted(self.target(True));self.assertEqual(accepted['state'],'non_supervised_unknown')
        self.assertNotIn('labels',accepted)
        with self.assertRaises(CognitiveKernelContractError):self.G.align_target(self.p,self.frozen,accepted,expected=self.expected,
            expected_freeze_sha256=self.frozen['record_sha256'],expected_target_record_sha256=accepted['record_sha256'],max_target_tokens=10000)
    def test_foreign_context_and_original_file_and_duplicate_json_refuse(self):
        target=self.target();target['source_ids'].reverse()
        with self.assertRaises(CognitiveKernelContractError):self.admitted(target)
        raw=canonical_json_bytes(self.target())
        for data,expected in ((raw,'0'*64),(b'{"schema":1,"schema":2}',sha256(b'{"schema":1,"schema":2}').hexdigest())):
            with self.subTest(data=len(data)),self.assertRaises(CognitiveKernelContractError):
                self.P.admit_target(self.frozen,expected_freeze_sha256=self.frozen['record_sha256'],expected=self.expected,target_reader=lambda:data,expected_target_file_sha256=expected)
    def test_complete_target_matches_unchanged_shift_and_normative_fields(self):
        accepted=self.admitted();row=self.G.align_target(self.p,self.frozen,accepted,expected=self.expected,
            expected_freeze_sha256=self.frozen['record_sha256'],expected_target_record_sha256=accepted['record_sha256'],max_target_tokens=10000)
        self.assertEqual(row['decoder_inputs'][1:],row['labels'][:-1]);self.assertEqual(len(row['tokens'])+1,len(row['labels']))
        self.assertTrue(row['fields']);self.assertTrue(all('matching_candidate_indices' in r for r in row['fields']))
        self.assertEqual(S.processor_binding(self.p),self.kw['processor_binding']);self.assertFalse(row['learning_qualified'])
        self.assertEqual(row,self.G.validate_pair(row,self.frozen,accepted,expected=self.expected,expected_pair_sha256=row['record_sha256']))
    def test_complete_proposal_unicode_semantics_literal_and_missing_uncertainty_full_packet(self):
        context,opened,_=J.caller_api.caller_input_from_record(self.bundle['caller']);ref,raw=opened[0]
        prefix='Café 😀 banana banana ';value=prefix+raw.decode('utf-8')
        base=FormationProposal('authored-proposal','claim','host',context.base.evidence[0].subject_ref or 'owner','authored-value',(ref,),
            value_text=value,anchors=(FormationEvidenceAnchor(ref,0,len(raw)),),epistemic_status='hypothetical',confidence=0.625,
            uncertainty_ref='not-supplied-uncertainty',disposition_scope_ref='extensible-authored-scope')
        mission=tuple(t.ref_id for t in context.targets if t.kind=='mission');workspace=tuple(t.ref_id for t in context.targets if t.kind=='workspace')
        proposal=FormationProposalV16(base,context.source_sensitivities[0].minimum,mission_target_refs=mission,workspace_target_refs=workspace)
        authored=self.target();authored['proposals']=[proposal.record()];authored['dispositions']=[{'scope_ref':'extensible-authored-scope','action':'propose','evidence_refs':[ref],'target_refs':[]}]
        authored['adjudications']={k:('present' if k in {'sensitivity','mission','workspace'} else 'negative') for k in L.FULL_ROLE_DIMENSIONS}
        accepted=self.admitted(authored);ordinal=next(i for i,c in enumerate(self.bundle['meaning']['candidate_inventory']) if c['ref_id']==ref and 'proposals[].proposal.evidence_refs' in c['eligible_paths'])
        annotations=[{'pointer':'/proposals/0/proposal/evidence_refs/0','mode':'external_selection','candidate_ordinal':ordinal},
            {'pointer':'/proposals/0/proposal/value_text','mode':'semantic_authoring'}]
        declarations=[{'pointer':'/proposals/0/proposal/value_text','target_value_utf8':[len(prefix.encode()),len(value.encode())],
            'source_occurrences':[{'source_ref':ref,'original_utf8':[0,len(raw)]}]}]
        pair=self.G.align_target(self.p,self.frozen,accepted,expected=self.expected,expected_freeze_sha256=self.frozen['record_sha256'],
            expected_target_record_sha256=accepted['record_sha256'],max_target_tokens=20000,annotations=annotations,literal_declarations=declarations)
        self.assertEqual(next(r for r in pair['fields'] if r['pointer']=='/proposals/0/proposal/uncertainty_ref')['candidate_status'],'missing')
        self.assertEqual(next(r for r in pair['fields'] if r['pointer']=='/proposals/0/proposal/evidence_refs/0')['selected_candidate_ordinal'],ordinal)
        self.assertEqual(pair['literal_segments'][0]['literal_utf8_hex'],raw.hex());self.assertTrue(pair['literal_segments'][0]['target_body_indices'])
        ex=self.P.example(self.frozen,authored);baseline=self.G.V._target_ids(self.p.tokenizer,ex,20000)
        self.assertEqual((pair['decoder_inputs'],pair['labels']),baseline)
        exported=self.P.export_pair(self.frozen,accepted,pair,expected=self.expected,expected_pair_sha256=pair['record_sha256'])
        self.assertEqual(exported['batch'],self.batch);self.assertEqual(exported['literal_segments'],pair['literal_segments']);self.assertFalse(exported['training_authorized'])
    def test_target_budget_complete_boundary_and_boolean_budget(self):
        accepted=self.admitted();row=self.G.align_target(self.p,self.frozen,accepted,expected=self.expected,expected_freeze_sha256=self.frozen['record_sha256'],expected_target_record_sha256=accepted['record_sha256'],max_target_tokens=10000)
        for budget in (len(row['labels'])-1,True):
            with self.subTest(budget=budget),self.assertRaises(CognitiveKernelContractError):
                self.G.align_target(self.p,self.frozen,accepted,expected=self.expected,expected_freeze_sha256=self.frozen['record_sha256'],expected_target_record_sha256=accepted['record_sha256'],max_target_tokens=budget)
    def test_target_change_keeps_input_and_processed_cache_unchanged(self):
        before=deepcopy(self.frozen);a=self.admitted();t=self.target();t['dispositions'][0]['scope_ref']='other-authored-scope';b=self.admitted(t)
        self.assertNotEqual(a['target_file_sha256'],b['target_file_sha256']);self.assertEqual(self.frozen,before)
        self.assertEqual(a['processed_view_sha256'],b['processed_view_sha256']);self.assertEqual(a['provenance_sha256'],b['provenance_sha256'])
    def test_ancestry_parent_cycles_and_cross_split_are_refused(self):
        for rows in ([split(),split('case-2','development')],[split(),{**split('case-2',group='world-2'),'lineage':{**split(group='world-2')['lineage'],'parent_case_ids':['case-1']},'split':'development'}],
                     [{**split(),'lineage':{**split()['lineage'],'parent_case_ids':['absent']}}]):
            with self.subTest(rows=len(rows)),self.assertRaises(CognitiveKernelContractError):self.P.validate_splits(rows)
        self.assertEqual(self.P.validate_splits([split(),split('case-2','development','world-2')])['component_count'],2)
    def test_saved_pair_export_deep_copy_and_no_final_or_implicit_training(self):
        accepted=self.admitted();pair=self.G.align_target(self.p,self.frozen,accepted,expected=self.expected,expected_freeze_sha256=self.frozen['record_sha256'],expected_target_record_sha256=accepted['record_sha256'],max_target_tokens=10000)
        result=self.P.export_pair(self.frozen,accepted,pair,expected=self.expected,expected_pair_sha256=pair['record_sha256'])
        self.assertEqual(result['batch'],self.batch);self.assertEqual(result['labels'],pair['labels']);self.assertFalse(result['training_authorized'])
        result['batch']['input_ids'][0][0]+=1;self.assertEqual(self.frozen['batch'],self.batch)

class ProfileBoundary(unittest.TestCase):
    def test_product_profile_is_distinct_from_private_supplied_records(self):
        from scripts.mfm import formation_paired_teaching as P,formation_joint_target_alignment as G
        private=seal({'schema':'private-mfm-paired-saved-input-v1'})
        with self.assertRaises(CognitiveKernelContractError):P.pin(private,private['record_sha256'],P.INPUT)
        self.assertEqual(P.INPUT,'mfm-paired-supplied-input-v1');self.assertEqual(G.PAIR,'mfm-paired-supplied-joint-target-geometry-v1')

if __name__=='__main__':unittest.main()
