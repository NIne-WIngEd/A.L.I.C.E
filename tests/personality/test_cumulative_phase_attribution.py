"""Public phase-attribution checks; no private execution or qualification."""
import copy
import importlib
import unittest


NAME = 'scripts.eipm.gemma_n0.magnolia_cumulative_phase_attribution_20261018'


class PhaseAttributionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec(NAME), 'phase attribution worker is missing')
        return importlib.import_module(NAME)

    def test_guard_refusal_precedes_private_access(self):
        class Refused(Exception):
            pass

        class Namespace:
            def _p2_guard(self):
                raise Refused()

            def _bounded_path(self, *args, **kwargs):
                raise AssertionError('private access before guard')

        with self.assertRaises(Refused):
            self.api().run(Namespace())

    def test_actual_native_hook_records_raw_n2_without_mutating_model(self):
        import torch
        from src.alice_personality.identity.codec import fingerprint
        from scripts.eipm.gemma_n0 import cumulative_successor
        from tests.personality.test_cumulative_successor import CumulativeSuccessorTests
        api = self.api()
        fixture = CumulativeSuccessorTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        construction = fixture.load()
        consumer = construction.consumer()
        document = fixture.fixture.fixture.doc
        produced = fixture.fixture.replay.produce(document)
        session = fixture.fixture.session
        state = fingerprint(construction.model.state_dict())
        hooks = len(construction.model.n2._forward_hooks)
        observations = []
        def observe(_module, _inputs, output):
            observations.append(output['scalar'].detach().clone())
        observer = construction.model.n2.register_forward_hook(observe)
        try:
            response, raw = api.capture_diagnostic_response(cumulative_successor.respond_with_governance,
                consumer, document, produced, session)
        finally:
            observer.remove()
        self.assertEqual(len(observations), 2)
        self.assertTrue(torch.equal(observations[0], observations[1]))
        self.assertEqual(raw, observations[-1].tolist())
        with torch.no_grad():
            representation, query = construction.model.n1(produced.frame)
            expected = construction.model.n2(produced.frame, representation, query)['scalar']
        self.assertEqual(raw, expected.tolist())
        packet = response['decision']
        expected_probability = torch.sigmoid(construction.model.n3.scale('scalar', expected[..., 1]))
        self.assertEqual(packet['co_valid_probabilities'], expected_probability.tolist())
        again, _ = api.capture_diagnostic_response(cumulative_successor.respond_with_governance,
            consumer, document, produced, session, expected_packet_sha256=response['packet_sha256'])
        self.assertEqual(again['packet_sha256'], response['packet_sha256'])
        with self.assertRaisesRegex(ValueError, 'reproduction'):
            api.capture_diagnostic_response(cumulative_successor.respond_with_governance,
                consumer, document, produced, session, expected_packet_sha256='0'*64)
        self.assertEqual(len(construction.model.n2._forward_hooks), hooks)
        self.assertEqual(fingerprint(construction.model.state_dict()), state)
        self.assertTrue(all(parameter.grad is None for parameter in construction.model.parameters()))

    def fixture(self):
        rows, targets = [], {}
        for family in ('one', 'two'):
            for kind in ('original', 'decisive_condition', 'paraphrase', 'irrelevant_social', 'neutral_extra_text'):
                name = family + '-' + kind
                rows.append({'document_id': name, 'parent_document_id': family,
                    'comparison_kind': kind, 'document_sha256': name,
                    'response': {'decision': {'candidate_ids': [['a','b','c']],
                        'co_valid_probabilities': [[.8,.2,.1]], 'contraindications': [[.2,.8,.9]]}},
                    'raw_n2_scalar_logits': [[[0.,2.,0.,0.,-2.,0.], [0.,-2.,0.,0.,2.,0.], [0.,-3.,0.,0.,3.,0.]]]})
                positives = {'b'} if kind == 'decisive_condition' else {'a'}
                targets[name] = {'document_sha256': name, 'review': {'candidate_regions': [
                    {'candidate_id': pointer, 'co_valid_region': 'POSITIVE' if pointer in positives else 'NEGATIVE',
                     'contraindication_region': 'NEGATIVE' if pointer in positives else 'POSITIVE'}
                    for pointer in ('a','b','c')]}}
        return rows, targets

    def test_bias_shift_is_not_mistaken_for_conditional_separation(self):
        rows, targets = self.fixture()
        shifted = copy.deepcopy(rows)
        for row in shifted:
            row['response']['decision']['co_valid_probabilities'] = [[.6,.4,.3]]
            row['response']['decision']['contraindications'] = [[.4,.6,.7]]
            for values in row['raw_n2_scalar_logits'][0]:
                values[1] += .25
                values[4] -= .25
        result = self.api().summarize_diagnostics({'parent':rows, 'after_n1':shifted, 'successor':shifted}, targets)
        metric = result['transitions']['parent_to_after_n1']['decisive_condition']['co_valid_probabilities']
        self.assertEqual(metric['wrong_to_right'], 0)
        self.assertEqual(metric['right_to_wrong'], 0)
        self.assertEqual(metric['comparisons'], 6)
        self.assertAlmostEqual(metric['mean_raw_logit_change'], .25)
        paired = result['context_deltas']['after_n1']['decisive_condition']['co_valid_probabilities']
        self.assertEqual(paired['max_absolute_probability_delta'], 0.)
        self.assertEqual(paired['max_absolute_raw_logit_delta'], 0.)
        self.assertEqual(paired['changed_target_comparisons'], 4)
        self.assertEqual(paired['mean_changed_target_signed_probability_delta'], 0.)

    def test_missing_rows_pointer_mismatch_and_nonfinite_raw_are_refused(self):
        api = self.api()
        rows, targets = self.fixture()
        for change in ('missing', 'pointer', 'nan'):
            bad = copy.deepcopy(rows)
            if change == 'missing': bad.pop()
            if change == 'pointer': bad[0]['response']['decision']['candidate_ids'][0][0] = 'other'
            if change == 'nan': bad[0]['raw_n2_scalar_logits'][0][0][1] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                api.summarize_diagnostics({'parent':rows,'after_n1':bad,'successor':rows}, targets)

    def test_equal_error_totals_still_report_one_repair_and_one_regression(self):
        rows, targets = self.fixture()
        changed = copy.deepcopy(rows)
        row = next(row for row in changed if row['document_id'] == 'one-decisive_condition')
        row['response']['decision']['co_valid_probabilities'] = [[.2,.2,.8]]
        result = self.api().summarize_diagnostics({'parent':rows,'after_n1':changed,'successor':changed}, targets)
        metric = result['transitions']['parent_to_after_n1']['decisive_condition']['co_valid_probabilities']
        self.assertEqual(metric['wrong_to_right'], 1)
        self.assertEqual(metric['right_to_wrong'], 1)
        self.assertEqual(metric['remaining_wrong'], 3)
        self.assertEqual(metric['remaining_right'], 1)
        self.assertEqual(result['regions']['parent']['decisive_condition']['co_valid_probabilities']['errors'], 4)
        self.assertEqual(result['regions']['after_n1']['decisive_condition']['co_valid_probabilities']['errors'], 4)

    def test_n1_phase_control_rejects_changed_auxiliary_or_any_n1_case(self):
        api = self.api()
        self.assertTrue(hasattr(api, 'verify_n1_phase_invariance'), 'N1 phase attribution control is missing')
        rows = [{'packet':'original','phase':'n1','case_id':'one','batch_sha256':'a'*64,
                 'total_loss':1.,'field_loss':{'concept_applicability':1.}},
                {'packet':'broader','phase':'n2','case_id':'two','batch_sha256':'b'*64,
                 'total_loss':2.,'field_loss':{'preferences':2.}}]
        after = copy.deepcopy(rows); after[1]['total_loss'] = 3.
        result = api.verify_n1_phase_invariance(rows,after,'c'*64,'c'*64)
        self.assertEqual(result['n1_case_count'],1)
        self.assertTrue(result['all_n1_objective_rows_exact'])
        with self.assertRaisesRegex(ValueError,'auxiliary'):
            api.verify_n1_phase_invariance(rows,after,'c'*64,'d'*64)
        after[0]['field_loss']['concept_applicability'] = .9
        with self.assertRaisesRegex(ValueError,'N1'):
            api.verify_n1_phase_invariance(rows,after,'c'*64,'c'*64)


if __name__ == '__main__':
    unittest.main()
