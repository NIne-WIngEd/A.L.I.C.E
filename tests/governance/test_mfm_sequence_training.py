import unittest
from types import SimpleNamespace

from cognitive_kernel.canonical import CognitiveKernelContractError


class SequenceTrainingTests(unittest.TestCase):
    def test_batch_weights_are_global_importance_weights_not_batch_normalized(self):
        from scripts.mfm.train_v16_sequence_specialist import batch_coefficients
        cases = [SimpleNamespace(case_id='base', split='train'),
                 SimpleNamespace(case_id='long', split='train')]
        coefficients = batch_coefficients(cases, {'base': 1/29, 'long': 1/928}, 432)
        self.assertAlmostEqual(coefficients['base'], 432 / 29 / 2)
        self.assertAlmostEqual(coefficients['long'], 432 / 928 / 2)
        self.assertAlmostEqual(coefficients['base'] / coefficients['long'], 32)
        with self.assertRaises(CognitiveKernelContractError):
            batch_coefficients([SimpleNamespace(case_id='base', split='development')], {'base': 1.}, 432)
        with self.assertRaises(CognitiveKernelContractError):
            batch_coefficients([cases[0], cases[0]], {'base': 1.}, 432)

    def test_stress_includes_original_and_new_lexical_geometry_extremes(self):
        from scripts.mfm.train_v16_sequence_specialist import stress_ids
        prepared = SimpleNamespace(view={'stress_train_case_ids': ['original'], 'cases': [
            {'case_id':'original', 'split':'train', 'source_tokens':200},
            {'case_id':'long-source', 'split':'train', 'source_tokens':500},
            {'case_id':'long-lexical', 'split':'train', 'source_tokens':80},
            {'case_id':'dev', 'split':'development', 'source_tokens':800}]})
        receipt = {'cases':[{'case_id':'original', 'lexical_target_tokens':20},
            {'case_id':'long-source', 'lexical_target_tokens':90},
            {'case_id':'long-lexical', 'lexical_target_tokens':300}]}
        self.assertEqual(stress_ids(prepared, receipt), ['original','long-lexical','long-source'])

    def test_new_task_seed_and_parent_optimizer_reset_are_explicit(self):
        from scripts.mfm.train_v16_sequence_specialist import OPTIMIZER_INITIALIZATION
        self.assertIn('reset', OPTIMIZER_INITIALIZATION)
        self.assertIn('objective', OPTIMIZER_INITIALIZATION)

    def test_stress_retains_augmented_prior_memory_inputs_even_when_small(self):
        from scripts.mfm.train_v16_sequence_specialist import stress_ids
        prepared = SimpleNamespace(view={'stress_train_case_ids':['large'], 'cases':[
            {'case_id':'large','split':'train','source_tokens':100},
            {'case_id':'prior','split':'train','source_tokens':5,'source_augmented':True}]})
        receipt = {'cases':[{'case_id':'large','lexical_target_tokens':100},
                            {'case_id':'prior','lexical_target_tokens':5}]}
        self.assertEqual(stress_ids(prepared,receipt),['large','prior'])

    def test_stratum_measurement_survives_canonical_key_reordering(self):
        from scripts.mfm.train_v16_sequence_specialist import task_strata
        loss = {'long-theatre-east-norm-agreed':[1.,1.],
                'long-ceramics-east-norm-agreed':[1.,1.],
                'long-atlas-east-norm-agreed':[1e16,1e16]}
        reordered = {key:loss[key] for key in sorted(loss)}
        self.assertEqual(task_strata(loss),task_strata(reordered))


if __name__ == '__main__': unittest.main()
