import ast
import importlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class ConditionalStageWorkerTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('scripts.eipm.gemma_n0.magnolia_conditional_stage_trace_20261024')

    def test_full_current_patch_preserves_unknowns_and_includes_unchanged_known_facts(self):
        fields = ('concept_applicability', 'concept_uncertainty', 'concept_source_support', 'concept_source_disconfirmation')
        facts = [NS(pointers=('c',), value=.4, available=True), NS(pointers=('unknown',), value=None, available=False)]
        case = NS(targets=[NS(field=name, facts=facts) for name in fields])
        result = self.api.all_available_channel_values(case, fields)
        self.assertEqual(result, {name: {('c',): .4} for name in fields})

    def test_rank_transitions_detect_offsetting_repairs_and_harms(self):
        packet = lambda values: {'candidate_ids': [['a','b']], 'preferences': [values]}
        desired = {'a': .9, 'b': .1}
        repaired = self.api.rank_transition(desired, packet([.2,.8]), packet([.8,.2]))
        harmed = self.api.rank_transition(desired, packet([.8,.2]), packet([.2,.8]))
        self.assertEqual(repaired['context_repairs'], 1)
        self.assertEqual(harmed['context_harms'], 1)
        self.assertEqual(repaired['changed_top_sets'] + harmed['changed_top_sets'], 2)
        tied = self.api.rank_transition(desired, packet([.5,.5]), packet([.8,.2]))
        self.assertEqual(tied['context_repairs'], 1)

    def test_stage_aggregation_preserves_zero_and_nonzero_effects(self):
        result = self.api.aggregate_stage_effects([
            {'stage': {'elements': 2, 'changed_elements': 0, 'delta_rms': 0., 'relative_delta_rms': None}},
            {'stage': {'elements': 2, 'changed_elements': 1, 'delta_rms': .2, 'relative_delta_rms': .1}}])
        self.assertEqual(result['stage']['contexts_with_exact_nonzero_change'], 1)
        self.assertEqual(result['stage']['delta_rms'], {'count': 2, 'mean': .1, 'min': 0., 'max': .2})
        self.assertEqual(result['stage']['relative_delta_rms']['count'], 1)

    def test_startup_guard_runs_first_and_no_optimization_exists(self):
        class Refused(Exception): pass
        class Namespace:
            def _p2_guard(self): raise Refused()
        with patch.object(Path, 'stat', side_effect=AssertionError('before isolation')), \
             patch.object(Path, 'open', side_effect=AssertionError('before isolation')):
            with self.assertRaises(Refused): self.api.run(Namespace())
        tree = ast.parse(Path(self.api.__file__).read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        self.assertEqual(ast.unparse(run.body[0].value), 'namespace._p2_guard()')
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr in ('step','backward') for n in calls))
        self.assertEqual(sum(isinstance(n.func, ast.Attribute) and n.func.attr == '_p2_guard' for n in calls), 1)


if __name__ == '__main__': unittest.main()
