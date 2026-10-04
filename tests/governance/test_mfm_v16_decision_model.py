"""Construction contracts and source-view rejection, not learned capability."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record
from scripts.mfm import formation_source_view as source
from scripts.mfm.build_v16_authoring_seed import CASES, build_one


class DecisionModelTests(unittest.TestCase):
    def test_calibration_changes_only_fresh_lexical_embeddings(self):
        import torch
        from cognitive_kernel.formation_v1_specialist import FormationSpecialist, SpecialistConfig
        from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
        config = SpecialistConfig(12, 31, 16, 2, 4, 16, 0, 1, 2)
        torch.manual_seed(42)
        old = FormationSpecialist(config).state_dict()
        torch.manual_seed(42)
        new = FormationDecisionSpecialist(config).state_dict()
        self.assertEqual(set(old), set(new))
        for name in old:
            torch.testing.assert_close(new[name], old[name] * (0.02 if name in {
                'token_embedding.weight', 'position_embedding.weight'} else 1.0), atol=0, rtol=0)

    def test_view_rejects_resealed_wrong_targets_missing_cases_and_unresolved_refs(self):
        example = learning_example_v16_from_record(build_one(next(s for s in CASES if s['case_id'] == 'v16-seed-02')), split='train')
        parent = SimpleNamespace(train=(example,), development=(), config=SimpleNamespace(max_target_tokens=8192),
            preflight={'corpus_sha256': 'c', 'owner_authorization_ref': 'owner', 'record_sha256': 'p'},
            bank={'record_sha256': 'b'}, export={'record_sha256': 'e', 'prepared_base_receipt_sha256': 'base'},
            component={'record_sha256': 'failed'})
        view = {'objective': source.preparation.OBJECTIVE, 'complete': True, 'corpus_sha256': 'c',
            'owner_authorization_ref': 'owner', 'parent_component_sha256': 'failed',
            'parent_feature_bank_sha256': 'b', 'parent_export_sha256': 'e', 'parent_preflight_sha256': 'p',
            'prepared_base_receipt_sha256': 'base', 'code_binding': source.preparation.code_binding(),
            'train_cases': 1, 'development_cases': 0, 'max_target_tokens': 8192,
            'independent_gold': False, 'final_payloads_opened': False, 'full_fit': False,
            'qualified_for_product': False, 'memory_gate_executed': False,
            'augmented_source_cases': 0, 'unchanged_source_cases_reused': 1, 'unresolved_reference_input_gap_cases': 0,
            'cases': [{'case_id': example.case_id, 'split': 'train', 'source_augmented': False,
                'parent_model_input_sha256': source.features.case_key(example), 'model_input_sha256': source.features.case_key(example),
                'decision_target_sha256': sha256(canonical_json_bytes(source.preparation.decision_for_example(example))).hexdigest(),
                'existing_reference_input_gaps': []}]}
        source.verify_rows(view, parent, (example,))
        for mutate in (lambda v: v['cases'][0].update(decision_target_sha256='wrong'),
                       lambda v: v.update(cases=[]), lambda v: v.update(independent_gold=True),
                       lambda v: v['cases'][0].update(existing_reference_input_gaps=['unknown'])):
            altered = deepcopy(view)
            mutate(altered)
            with self.assertRaises(CognitiveKernelContractError):
                source.verify_rows(altered, parent, (example,))

    def test_view_privacy_admission_precedes_successor_reads(self):
        with patch.object(source.preparation.diagnostic, 'prepare_diagnostic', side_effect=CognitiveKernelContractError('privacy rejected')):
            with self.assertRaisesRegex(CognitiveKernelContractError, 'privacy rejected'):
                source.load(SimpleNamespace())

    def test_actual_optimizer_restart_preserves_next_weights_and_moments(self):
        import torch
        from cognitive_kernel.formation_v1_specialist import SpecialistConfig
        from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
        from scripts.mfm import train_v16_decision_specialist as trainer
        torch.manual_seed(123)
        model = FormationDecisionSpecialist(SpecialistConfig(12, 31, 16, 2, 4, 16, 0, 1, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        example = SimpleNamespace(case_id='stress', split='train')
        prepared = SimpleNamespace(train=(example,), view={'stress_train_case_ids': ['stress']}, device=torch.device('cpu'))
        payload = {'base_states': torch.randn(1, 7, 12), 'source_mask': torch.ones(1, 7, dtype=torch.long),
                   'input_ids': torch.tensor([[1, 7, 9]]), 'labels': torch.tensor([[7, 9, 2]])}
        with TemporaryDirectory() as directory, patch.object(source, 'payload', return_value=payload):
            args = SimpleNamespace(output_dir=Path(directory), gradient_accumulation=1)
            result = trainer.probe(model, optimizer, prepared, args, 'a' * 64)
        self.assertEqual(result['model_max_abs_difference'], 0)
        self.assertEqual(result['optimizer_max_abs_difference'], 0)
        self.assertTrue(result['backward_and_restart_passed'])
        with self.assertRaisesRegex(CognitiveKernelContractError, 'only admitted training'):
            trainer.optimizer_step(model, optimizer, (SimpleNamespace(split='development'),), prepared=prepared)

    def test_generated_grounding_is_separate_from_exact_teacher_agreement(self):
        from scripts.mfm import run_v16_decision_diagnostic as runner
        row = build_one(next(s for s in CASES if s['case_id'] == 'v16-seed-02'))
        example = learning_example_v16_from_record(row, split=row['split'])
        decision = source.preparation.decision_for_example(example)
        decision['proposals'][0]['proposal']['value_text'] = 'Owner values quiet, calm planning with Mara, who is lively and sociable.'
        result = runner.validate(canonical_json_bytes(decision).decode(), True, example, 'a' * 64, 'diagnostic')
        self.assertEqual(result['validation_status'], 'grounded_teacher_proposal_only')
        self.assertFalse(result['exact_teacher_decision_agreement'])
        no_end = runner.validate(canonical_json_bytes(decision).decode(), False, example, 'a' * 64, 'diagnostic')
        self.assertEqual(no_end['validation_status'], 'invalid')
        self.assertIsNone(no_end['exact_teacher_decision_agreement'])
        self.assertEqual(no_end['proposal_count'], 0)
        self.assertEqual(result['proposal_count'], len(example.target.proposals))

    def test_generated_cache_checks_real_greedy_reference_in_fp32(self):
        import torch
        from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
        from cognitive_kernel.formation_v1_specialist import SpecialistConfig
        from scripts.mfm import run_v16_decision_diagnostic as runner
        from scripts.mfm.run_v16_cached_diagnostic import greedy_fp32
        torch.manual_seed(31)
        model = FormationDecisionSpecialist(SpecialistConfig(12, 31, 16, 2, 4, 40, 0, 1, 2)).eval()
        states, mask = torch.randn(1, 7, 12), torch.ones(1, 7, dtype=torch.long)
        tokens, ended, parity = runner.greedy_fp32(model, states, mask, 35)
        self.assertEqual((tokens, ended), greedy_fp32(model, states=states, mask=mask, max_new_tokens=35))
        self.assertEqual(parity['compared_greedy_prefixes'], min(32, len(tokens) + int(ended)))
        self.assertTrue(parity['argmaxes_equal'])


if __name__ == '__main__':
    unittest.main()
