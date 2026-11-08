"""Saved-result comparisons must preserve scope and incomparable loss families."""
import copy
import importlib.util
import unittest


class CumulativeResultAnalysisTests(unittest.TestCase):
    def api(self):
        name = 'scripts.eipm.gemma_n0.cumulative_result_analysis'
        self.assertIsNotNone(importlib.util.find_spec(name), 'complete saved-result comparison is missing')
        return __import__(name, fromlist=['compare_objectives'])

    def rows(self):
        return [{'packet': 'broader', 'phase': 'n2', 'case_id': 'one', 'case_sha256': 'a'*64,
                 'batch_sha256': 'b'*64, 'total_loss': 3.,
                 'field_loss': {'preferences': 1., 'voice_control_values': 2.},
                 'available_counts': {'preferences': 1, 'voice_control_values': 3},
                 'descriptive_training_targets_not_independent_qualification': True},
                {'packet': 'broader', 'phase': 'n2', 'case_id': 'two', 'case_sha256': 'c'*64,
                 'batch_sha256': 'd'*64, 'total_loss': 1., 'field_loss': {'preferences': 1.},
                 'available_counts': {'preferences': 1},
                 'descriptive_training_targets_not_independent_qualification': True}]

    def test_all_matched_cases_and_distinct_fields_are_measured(self):
        before = self.rows(); after = copy.deepcopy(before)
        after[0]['total_loss'] = 2.5
        after[0]['field_loss'] = {'preferences': .5, 'voice_control_values': 2.}
        result = self.api().compare_objectives(before, list(reversed(after)))
        self.assertEqual(result['case_count'], 2)
        fields = result['groups']['broader/n2']['fields']
        self.assertEqual(fields['preferences']['before_mean_case_loss'], 1.)
        self.assertEqual(fields['preferences']['after_mean_case_loss'], .75)
        self.assertEqual(fields['preferences']['improved_cases'], 1)
        self.assertEqual(fields['voice_control_values']['available_facts'], 3)
        self.assertEqual(fields['voice_control_values']['unchanged_cases'], 1)
        self.assertFalse(result['independent_qualification'])

    def test_missing_duplicate_or_changed_case_cannot_look_like_improvement(self):
        api = self.api(); before = self.rows()
        for after in (before[:1], before + before[:1]):
            with self.assertRaises(ValueError): api.compare_objectives(before, after)
        for key, value in [('case_sha256', 'e'*64), ('batch_sha256', 'f'*64),
                           ('available_counts', {'preferences': 9, 'voice_control_values': 3}),
                           ('phase', 'n1')]:
            after=copy.deepcopy(before); after[0][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): api.compare_objectives(before, after)

    def test_nonfinite_or_incomplete_metrics_are_refused(self):
        api=self.api(); before=self.rows()
        for change in ({'total_loss': float('nan')}, {'field_loss': {'preferences': .2}},
                       {'descriptive_training_targets_not_independent_qualification': False}):
            after=copy.deepcopy(before); after[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): api.compare_objectives(before,after)


if __name__ == '__main__':
    unittest.main()
