"""Tests for diagnosis validity; fixtures are not learned MFM evidence."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from scripts.mfm import analyze_v16_budget_failure as analysis
from scripts.mfm import run_v16_decision_diagnostic as original


class BudgetFailureAnalysisTests(unittest.TestCase):
    def test_scope_names_are_separate_from_actual_source_reference_fields(self):
        body={'dispositions':[{'scope_ref':'owner_profile', 'evidence_refs':['source-a']}],
              'proposals':[{'proposal':{'disposition_scope_ref':'owner_profile','value_text':'é'}}]}
        text, spans=analysis.precise_scalar_spans(body)
        names={kind for _,_,kind in spans}
        self.assertIn('scope_label:dispositions.*.scope_ref',names)
        self.assertIn('source_reference:dispositions.*.evidence_refs.*',names)
        self.assertIn('scope_label:proposals.*.proposal.disposition_scope_ref',names)
        self.assertIn('value_text:proposals.*.proposal.value_text',names)

    def test_privacy_admission_precedes_component_and_raw_reads(self):
        with patch.object(analysis.source, 'load', side_effect=CognitiveKernelContractError('privacy rejected')):
            with self.assertRaisesRegex(CognitiveKernelContractError, 'privacy rejected'):
                analysis.run(SimpleNamespace())

    def test_validation_stages_reproduce_original_failure_before_semantics(self):
        row={'eos_observed':True,'raw_output_text':'{"x":}', 'validation_status':'invalid',
             'validation_error':'original parse failure'}
        result=analysis.validation_stage(row,None,'a'*64,'diagnosis')
        self.assertEqual(result['first_failed_stage'],'json')
        row['raw_output_text']='{"x":1}'
        result=analysis.validation_stage(row,SimpleNamespace(context=None),'a'*64,'diagnosis')
        self.assertEqual(result['first_failed_stage'],'decision_codec')
        row['validation_status']='grounded_teacher_proposal_only'
        with self.assertRaises(CognitiveKernelContractError):
            analysis.validation_stage(row,SimpleNamespace(context=None),'a'*64,'diagnosis')

    def test_family_metrics_reveal_masked_loss_and_keep_objective_weighting(self):
        def row(case, family, n, loss):
            return {'case_id':case, 'family':family, 'tokens':n, 'mean_loss':loss,
                'next_token_accuracy':.5, 'groups':{'reference_identity':{
                    'tokens':n, 'mean_loss':loss, 'accuracy':.5}},
                'groups_by_positive_train_corpus_frequency':{'reference_identity':{
                    '1-9':{'tokens':n, 'mean_loss':loss, 'accuracy':.5}}}}
        rows = [row('base', 'base-train', 20, 4.), row('a', 'longitudinal-train', 2, .1),
                row('b', 'longitudinal-train', 2, .1)]
        result = analysis.aggregate_measurements(rows)
        self.assertAlmostEqual(result['all-train']['equal_case_mean_loss'], 1.4)
        self.assertAlmostEqual(result['all-train']['token_weighted_mean_loss'], 80.4/24)
        self.assertEqual(result['base-train']['equal_case_mean_loss'], 4.)
        self.assertEqual(result['all-train']['per_case_loss_distribution']['median'], .1)
        self.assertEqual(result['all-train']['per_case_loss_distribution']['maximum'], 4.)
        self.assertAlmostEqual(result['longitudinal-train']['equal_case_mean_loss'], .1)
        self.assertAlmostEqual(result['all-train']['groups']['reference_identity']['mean_loss'], 80.4/24)
        with self.assertRaises(CognitiveKernelContractError):
            analysis.aggregate_measurements([*rows, rows[0]])
        with self.assertRaises(CognitiveKernelContractError):
            analysis.aggregate_measurements([row('bad', 'base-train', 0, float('nan'))])

    def test_fixed_roster_selects_all_diverse_cases_and_complete_scenario(self):
        make = lambda cid, split: SimpleNamespace(case_id=cid, split=split)
        base = [make(f'v16-seed-{i:02}', 'train') for i in (*range(1,13),19,20,21)] + [make('role-source-person-revocation','train')]
        long = [make(f'long-atlas-east-op-{i}', 'train') for i in range(13)]
        long += [make(f'long-other-north-op-{i}', 'train') for i in range(403)]
        dev = [make(f'det-dev-{i}', 'development') for i in range(9)]
        prepared = SimpleNamespace(train=tuple(base+long), development=tuple(dev))
        selected = analysis.verify_roster(prepared)
        self.assertEqual([e.case_id for e in selected], [e.case_id for e in base+long[:13]])
        prepared.train = tuple(base+long[:-1])
        with self.assertRaises(CognitiveKernelContractError): analysis.verify_roster(prepared)
        prepared.train = tuple(base+long[:-1]+[base[0]])
        with self.assertRaises(CognitiveKernelContractError): analysis.verify_roster(prepared)

    def test_teacher_divergence_is_separate_from_json_parse_fault_and_eos(self):
        class Tokenizer:
            def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)
        text = '{"x":"wrong"}'
        tokens = list(map(ord,text))
        row = {'raw_output_text':text, 'generated_token_ids':tokens, 'eos_observed':True,
            'validation_status':'invalid', 'validation_error':'grounding failure'}
        labels = list(map(ord,'{"x":"right"}')) + [1]
        result = analysis.raw_fault(row, labels, ['reference_identity']*len(labels), Counter(), Tokenizer(), 1)
        self.assertEqual(result['json_status'], 'parsed')
        self.assertIsNone(result['json_error_char'])
        self.assertEqual(result['first_exact_teacher_divergence_token'], 6)
        self.assertFalse(result['semantic_mismatch_established'])
        row.update(raw_output_text='{"x":}', generated_token_ids=list(map(ord,'{"x":}')))
        result = analysis.raw_fault(row, labels, ['reference_identity']*len(labels), Counter(), Tokenizer(), 1)
        self.assertEqual(result['json_status'], 'malformed')
        self.assertEqual(result['json_error_char'], 5)
        self.assertIn(5, result['later_reference_checkpoints'])

    def test_later_retained_cache_checks_reproduce_real_greedy_tokens(self):
        torch.manual_seed(8)
        model = FormationDecisionSpecialist(SpecialistConfig(12,31,16,2,4,90,0,1,2))
        states = torch.randn(1,7,12)
        mask = torch.ones(1,7,dtype=torch.long)
        tokens, ended, _ = original.greedy_fp32(model, states, mask, 70)
        payload = {'base_states':states, 'source_mask':mask}
        result = analysis.parity_retained(model,payload,tokens,ended,[32,64,len(tokens)-1])
        self.assertTrue(result['argmaxes_equal'])
        self.assertIn(64,result['checked_positions'])
        wrong = [*tokens]; wrong[64]=(wrong[64]+1)%31
        with self.assertRaises(CognitiveKernelContractError):
            analysis.parity_retained(model,payload,wrong,ended,[64])

    def test_retained_eos_is_replayed_and_reference_checked(self):
        torch.manual_seed(9)
        model=FormationDecisionSpecialist(SpecialistConfig(12,31,16,2,4,16,0,1,2))
        with torch.no_grad(): model.output_bias[model.config.end_token_id]=20.
        states=torch.randn(1,4,12);mask=torch.ones(1,4,dtype=torch.long)
        tokens,ended,_=original.greedy_fp32(model,states,mask,12)
        self.assertTrue(ended)
        result=analysis.parity_retained(model,{'base_states':states,'source_mask':mask},tokens,ended,[len(tokens)])
        self.assertEqual(result['checked_positions'],[len(tokens)])
        self.assertTrue(result['argmaxes_equal'])

    def test_full_train_remeasurement_must_reproduce_sealed_objective(self):
        progress={'train_cases':432,'target_tokens':175357,'equal_case_mean_loss':.378443,
            'token_weighted_mean_loss':.524076}
        metric={'cases':432,'tokens':175357,'equal_case_mean_loss':.37844301,'token_weighted_mean_loss':.52407599}
        self.assertTrue(analysis.verify_loss_remeasurement(metric,progress)['matched'])
        for bad in ({**metric,'cases':416},{**metric,'tokens':175358},
                    {**metric,'equal_case_mean_loss':.524076}):
            with self.assertRaises(CognitiveKernelContractError): analysis.verify_loss_remeasurement(bad,progress)


if __name__ == '__main__': unittest.main()
