"""Real public optimization through the complete personality response handoff."""
import importlib
from dataclasses import replace
import unittest

import torch

from src.alice_personality.identity.codec import canonical, fingerprint
from src.alice_personality.identity.consumer import CheckpointPin, DecisionConsumer, RenderedResponse
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.teaching.contracts import TeachingBatch, TrainingRecipe
from src.alice_personality.teaching.trainer import IdentityTrainer
from tests.personality import test_identity_consumer as consumer_fixture
from tests.personality.test_identity_codec import Backbone, Tokenizer
from tests.personality.test_identity_teaching import _admission, _target

try:
    api = importlib.import_module('scripts.eipm.gemma_n0.governed_response')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.governed_response':
        raise
    api = None


class GovernedResponseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(api, 'Complete governing response interface is missing')
        self.fixture = consumer_fixture.ConsumerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def respond(self, consumer, **options):
        return api.respond_with_governance(consumer, self.fixture.doc,
            self.fixture.produced, self.fixture.session, **options)

    def test_live_governed_response_and_saved_json_have_identical_reviewed_regions(self):
        import json
        from scripts.eipm.gemma_n0.conditional_successor_report import reviewed_region_fit

        response = self.respond(self.fixture.build(co_valid=True))
        decision = response['decision']
        self.assertIsInstance(decision['candidate_ids'], tuple)
        self.assertIsInstance(decision['concept_ids'], tuple)
        targets = []
        for field, pointer_key, desired in (
            ('co_valid_probabilities', 'candidate_ids', 1),
            ('contraindications', 'candidate_ids', 0),
            ('concept_applicability', 'concept_ids', 1),
        ):
            targets.append({'field': field, 'facts': [
                {'pointers': [pointer], 'available': True, 'value': desired}
                for pointer in decision[pointer_key][0]]})
        before = canonical(response)
        saved_result = reviewed_region_fit(json.loads(before)['decision'], targets)
        live_result = reviewed_region_fit(decision, targets)
        self.assertEqual(live_result, saved_result)
        self.assertEqual(sum(live_result[row['field']]['comparisons'] for row in targets), 5)
        for target in targets:
            field = target['field']
            values = decision['conditional_concepts'][field][0] if field == 'concept_applicability' else decision[field][0]
            expected = sum((value >= .5) != bool(fact['value']) for value, fact in zip(values, target['facts']))
            self.assertEqual(live_result[field]['errors'], expected)
        self.assertFalse(live_result['qualification'])
        self.assertEqual(canonical(response), before)

    def test_preferred_but_invalid_behavior_remains_unresolved_without_silent_reranking(self):
        # Reuse the original diagnostic's actual converged control. The short
        # sparse fixture does not establish all three target preconditions.
        targets = {
            'preferences': _target([[.02, .98]]),
            'co_valid_probabilities': _target([[1., 0.]]),
            'contraindications': _target([[0., 1.]]),
            'uncertainty': _target([[.1, .9]]),
            'evidence_sufficiency': _target([[.9, .1]])}
        batch = TeachingBatch(self.fixture.produced.frame, targets, ('public-governance-conflict',),
            (('fictional-source-family',),), 'n2', 'TRAIN', 'public_mechanical_fixture')
        recipe = TrainingRecipe('n2', seed=411, learning_rate=.01, max_grad_norm=1, weight_decay=0)
        trainer = IdentityTrainer(self.fixture.model, recipe,
            _admission(self.fixture.path, self.fixture.model, recipe, [batch]), first_batch=batch)
        for _ in range(120):
            trainer.step(batch)
        path = self.fixture.path / 'governance-conflict'
        manifest = trainer.save_checkpoint(path)
        consumer = DecisionConsumer(self.fixture.model,
            (CheckpointPin(str(path), manifest['manifest_sha256']),))
        # Match the consumer's actual inference path, including attention mode.
        self.fixture.model.set_phase('inference')
        with torch.no_grad():
            packet = self.fixture.model(self.fixture.produced.frame)
        self.assertGreater(float(packet.preferences[0, 1]), .5)
        self.assertLess(float(packet.co_valid_probabilities[0, 1]), .5)
        self.assertGreater(float(packet.contraindications[0, 1]), .5)
        called = []
        result = self.respond(consumer, renderer=lambda envelope: called.append(envelope))
        self.assertEqual(result['status'], 'UNRESOLVED_GOVERNING_CONFLICT')
        self.assertEqual(result['preferred_candidate_ids'], ['opaque-candidate-b'])
        self.assertEqual(result['selected_candidate_ids'], [])
        self.assertEqual(result['preference_selection']['selected_candidate_ids'], ['opaque-candidate-b'])
        self.assertNotIn('opaque-candidate-b', result['acceptable_candidate_ids'])
        self.assertIsNone(result['response_text'])
        self.assertEqual(called, [])
        self.assertEqual(result['packet_sha256'], fingerprint(packet))

    def test_complete_native_verdict_and_unknown_scope_reach_replaceable_renderers(self):
        consumer = self.fixture.build(co_valid=True)
        seen = []
        def render(envelope):
            seen.append(envelope)
            return RenderedResponse(envelope['response_text'], {}, 'public-renderer-' + str(len(seen)), None)
        first = self.respond(consumer, renderer=render)
        second = self.respond(consumer, renderer=render)
        self.assertEqual(first['status'], 'SELECTED')
        self.assertEqual(first['rendering']['status'], 'TEXT_REALIZED_DELIVERY_UNVERIFIED')
        self.assertEqual(canonical(seen[0]['decision']), canonical(seen[1]['decision']))
        self.assertEqual(first['packet_sha256'], second['packet_sha256'])
        for name in ('evidence_pointers', 'historical_evidence_pointers', 'alice_experience_pointers',
                     'uncertainty', 'contraindications', 'failure_tail_risk', 'owner_fidelity_uncertainty',
                     'conditional_concepts', 'native_latent'):
            self.assertIn(name, seen[0]['decision'])
        self.assertFalse(seen[0]['output_scope']['owner_fidelity_uncertainty']['taught'])
        self.assertFalse(seen[0]['output_scope']['voice_control_values']['taught'])
        self.assertFalse(seen[0]['output_scope']['evidence_pointers']['taught'])
        self.assertFalse(first['owner_fidelity_approved'])
        self.assertFalse(first['spoken_fidelity_verified'])

    def test_realized_words_cannot_replace_the_personal_choice(self):
        consumer = self.fixture.build(co_valid=True)
        result = self.respond(consumer, renderer=lambda envelope:
            RenderedResponse('I agree without checking.', {}, 'public-changed-renderer', None))
        self.assertEqual(result['status'], 'UNRESOLVED_RENDERER')
        self.assertEqual(result['response_text'], self.fixture.doc.candidates[1].text)
        self.assertEqual(result['rendering']['text'], 'I agree without checking.')

    def test_untaught_validity_is_unknown_instead_of_automatic_acceptability(self):
        result = self.respond(self.fixture.build())
        self.assertEqual(result['preferred_candidate_ids'], ['opaque-candidate-b'])
        self.assertEqual(result['acceptable_candidate_ids'], [])
        self.assertEqual(result['validity_status'], 'UNTAUGHT')
        self.assertFalse(result['output_scope']['co_valid_probabilities']['taught'])

    def test_renderer_cannot_mutate_the_bound_original_governing_packet(self):
        consumer = self.fixture.build(co_valid=True)
        original = self.respond(consumer)
        def render(envelope):
            envelope['decision']['uncertainty'][0][0] = 999
            return RenderedResponse(envelope['response_text'], {}, 'public-mutating-renderer', None)
        changed = self.respond(consumer, renderer=render)
        self.assertEqual(canonical(original['decision']), canonical(changed['decision']))
        self.assertEqual(original['packet_sha256'], changed['packet_sha256'])

    def test_changed_voice_meaning_remains_untaught_for_current_output(self):
        consumer, _ = self.fixture.build_sparse({'preferences': _target([[.02, .98]]),
            'voice_control_values': _target([[[.4], [.4]]]),
            'voice_control_confidence': _target([[[.9], [.9]]])})
        before = self.respond(consumer)
        self.assertTrue(before['output_scope']['voice_control_values']['taught'])
        labels = dict(self.fixture.doc.labels)
        labels['voice'] = (replace(labels['voice'][0], name='pause duration', description='Pause before a phrase.',
            unit='milliseconds', minimum=100., maximum=200.),)
        changed = replace(self.fixture.doc, labels=labels)
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.fixture.config, _fixture=True)
        produced = producer.produce(changed)
        result = api.respond_with_governance(consumer, changed, produced, producer.close_and_verify())
        self.assertTrue(all(not item['voice_controls'] for item in result['preference_selection']['intents']))
        self.assertFalse(result['output_scope']['voice_control_values']['taught'])
        self.assertFalse(result['output_scope']['voice_control_confidence']['taught'])

    def test_n1_conditional_scope_comes_from_actual_optimized_n1_checkpoint(self):
        with torch.no_grad():
            representation = self.fixture.model(self.fixture.produced.frame).representation
        batch = TeachingBatch(self.fixture.produced.frame,
            {'concept_applicability': _target(torch.full_like(representation.concept_applicability, .8),
                                            representation.concept_mask)},
            ('public-n1-scope',), (('fictional-source-family',),), 'n1', 'TRAIN', 'public_mechanical_fixture')
        recipe = TrainingRecipe('n1', seed=411, learning_rate=.01, max_grad_norm=1, weight_decay=0)
        admission_path = self.fixture.path / 'n1-admission'
        admission_path.mkdir()
        trainer = IdentityTrainer(self.fixture.model, recipe,
            _admission(admission_path, self.fixture.model, recipe, [batch]), first_batch=batch)
        trainer.step(batch)
        checkpoint = self.fixture.path / 'n1-checkpoint'
        manifest = trainer.save_checkpoint(checkpoint)
        n2 = self.fixture.build()
        consumer = DecisionConsumer(self.fixture.model,
            (CheckpointPin(str(checkpoint), manifest['manifest_sha256']), *n2.checkpoints))
        result = self.respond(consumer)
        self.assertTrue(result['output_scope']['concept_applicability']['n1_taught'])
        self.assertTrue(result['output_scope']['concept_applicability']['taught'])
        for field in ('concept_uncertainty', 'concept_source_support', 'concept_source_disconfirmation'):
            self.assertFalse(result['output_scope'][field]['taught'])
