import copy
import unittest
from types import SimpleNamespace as NS

from scripts.eipm.gemma_n0 import magnolia_paired_readback_20261023 as api
from scripts.eipm.gemma_n0 import magnolia_saved_path_readback_20261020 as saved


def target(field, values, mode='independent'):
    return NS(field=field, mode=mode, facts=[NS(pointers=k, value=v, available=True) for k, v in values.items()])


def decision():
    return {'candidate_ids': [['a', 'b']], 'candidate_mask': [[True, True]],
        'concept_ids': [['c', 'd']], 'sources': [[{'record_id': 'fixed'}, {'record_id': 'host-left'}]],
        'conditional_concepts': {'concept_applicability': [[.2, .8]], 'concept_uncertainty': [[.1, .3]],
            'concept_source_support': [[[.4, .5], [.6, .7]]], 'concept_source_disconfirmation': [[[.9, .8], [.7, .6]]],
            'concept_source_mask': [[[True, False], [False, True]]]},
        'native_latent': {'concepts': [[[1., 2.], [3., 4.]]], 'candidates': [[[5., 6.], [7., 8.]]],
            'concept_mask': [[True, True]], 'views': {'identity': [[[9., 10.], [0., 0.], [1., 2.], [0., 0.]]]},
            'view_masks': {'identity': [[True, False, True, False]]}},
        'preference_logits': [[2., 1.]], 'preferences': [[.73, .27]],
        'co_valid_probabilities': [[.8, .2]], 'contraindications': [[.1, .9]]}


class PairedReadbackTests(unittest.TestCase):
    def test_actual_native_codec_source_plus_concept_views(self):
        from dataclasses import replace
        import torch
        from tests.personality.test_identity_codec import document, Tokenizer, Backbone
        from src.alice_personality.identity.codec import decision_to_mapping
        from src.alice_personality.identity.feature_producer import FrozenFrameProducer
        from src.alice_personality.identity.model import IdentityModel
        from src.alice_personality.identity.contracts import IdentityModelConfig
        torch.set_num_threads(1)
        doc = document()
        doc = replace(doc, candidates=tuple(replace(c, authority_allowed=True) for c in doc.candidates))
        config = IdentityModelConfig(8, 3, learned_width=16, attention_heads=4)
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), config, _fixture=True)
        with torch.no_grad():
            decoded = decision_to_mapping(IdentityModel(config)(producer.produce(doc).frame))
        fields, geometry = api.coordinate_fields(decoded, {doc.sources[0].record.record_id: 'fixed:public'})
        self.assertEqual(len(fields['n1:view:identity']), len(doc.sources)+len(doc.concepts))
        self.assertEqual(len(geometry['view_masks']['identity']), len(doc.sources)+len(doc.concepts))
        producer.close_and_verify()

    def test_semantic_permutation_and_context_role_do_not_invent_a_change(self):
        left = decision()
        right = copy.deepcopy(left)
        right['sources'][0][1]['record_id'] = 'host-right'
        for field in ('candidate_ids', 'candidate_mask', 'preference_logits', 'preferences', 'co_valid_probabilities', 'contraindications'):
            right[field][0].reverse()
        right['native_latent']['candidates'][0].reverse()
        for field in ('concept_ids',):
            right[field][0].reverse()
        for field in ('concepts', 'concept_mask'):
            right['native_latent'][field][0].reverse()
        for collection in ('views', 'view_masks'):
            for values in right['native_latent'][collection].values():
                values[0] = [values[0][i] for i in (1, 0, 3, 2)]
        for field, values in right['conditional_concepts'].items():
            values[0].reverse()
        right['sources'][0].reverse()
        for field in ('concept_source_support', 'concept_source_disconfirmation', 'concept_source_mask'):
            for row in right['conditional_concepts'][field][0]:
                row.reverse()
        a = api.coordinate_fields(left, {'fixed': 'fixed:id', 'host-left': 'current-role:HOST'})
        b = api.coordinate_fields(right, {'fixed': 'fixed:id', 'host-right': 'current-role:HOST'})
        self.assertEqual(a, b)
        for field in a[0]:
            self.assertEqual(saved.difference(a[0][field], b[0][field])['rms_delta'], 0.)

    def test_common_logit_shift_is_not_conditioned_ranking(self):
        left, right = decision(), decision()
        right['preference_logits'] = [[12., 11.]]
        aliases = {'fixed': 'fixed:id', 'host-left': 'current-role:HOST'}
        a, b = [api.coordinate_fields(d, aliases)[0] for d in (left, right)]
        self.assertEqual(saved.difference(a['output:preference_logits'], b['output:preference_logits'])['rms_delta'], 10.)
        self.assertEqual(a['output:centered_preference_logits'], b['output:centered_preference_logits'])

    def test_tied_targets_keep_exact_test_and_show_outside_margin_separately(self):
        case = NS(targets=[target('preferences', {('a',): .5, ('b',): .5}, 'closed_distribution')])
        result = api.preference_detail(case, decision())
        self.assertFalse(result['exact_top_set_correct'])
        self.assertEqual(result['target_top_count'], 2)
        self.assertIsNone(result['target_set_vs_outside_margin'])
        self.assertEqual(result['within_expected_top_spread'], 1.)

    def test_probability_rounding_ties_reproduce_original_metric(self):
        case = NS(targets=[target('preferences', {('a',): .5, ('b',): .5}, 'closed_distribution')])
        d = decision()
        d['preferences'] = [[.5, .5]]
        result = api.preference_detail(case, d)
        self.assertTrue(result['exact_top_set_correct'])
        self.assertEqual(result['within_expected_top_spread'], 1.)

    def test_native_source_authority_drift_refused_before_role_aliasing(self):
        from dataclasses import asdict, replace
        from tests.personality.test_identity_codec import document
        doc = document()
        case = NS(document=doc)
        d = {'sources': [[asdict(e.record) for e in doc.sources]],
             'concepts': [[asdict(e.record) for e in doc.concepts]],
             'candidate_ids': [[c.candidate_id for c in doc.candidates]]}
        api.validate_saved_authority(case, d)
        for field in ('source_family_id', 'identity_core_allowed'):
            altered = copy.deepcopy(d)
            altered['sources'][0][0][field] = 'changed' if field == 'source_family_id' else False
            with self.assertRaises(ValueError):
                api.validate_saved_authority(case, altered)

    def test_interpretation_switch_errors_are_not_just_latent_distance(self):
        left, right = decision(), decision()
        right['conditional_concepts']['concept_applicability'][0][0] = .3
        cases = []
        for truth in (0., 1.):
            cases.append(NS(targets=[target(field, {('c', 'host-left') if 'source_' in field else ('c',): truth}) for field in api.CONDITIONAL]))
        alias = {'fixed': 'fixed:id', 'host-left': 'current-role:HOST'}
        result = api.conditional_target_detail(cases, (left, right), (alias, alias))
        self.assertEqual(result['concept_applicability']['correct_changed_target_direction'], 1)
        self.assertEqual(result['concept_applicability']['binary_errors'], 1)
        self.assertAlmostEqual(result['concept_applicability']['target_signed_condition_deltas'][0], .1)
        self.assertEqual(result['concept_source_support']['correct_changed_target_direction'], 0)

    def test_ambiguous_or_truncated_coordinates_refused(self):
        for mutate in (lambda d: d['candidate_ids'][0].__setitem__(1, 'a'),
                       lambda d: d['native_latent']['concepts'][0].pop(),
                       lambda d: d['conditional_concepts']['concept_source_mask'][0][0].pop()):
            d = decision()
            mutate(d)
            with self.assertRaises(ValueError):
                api.coordinate_fields(d, {'fixed': 'fixed:id', 'host-left': 'current-role:HOST'})

    def test_startup_guard_runs_before_private_operations(self):
        class Refused(Exception):
            pass
        class Namespace:
            def _p2_guard(self):
                raise Refused()
        with self.assertRaises(Refused):
            api.run(Namespace())


if __name__ == '__main__':
    unittest.main()
