"""Actual public consumer routing; no Gemma, identity or acoustic qualification."""
from copy import deepcopy
from dataclasses import replace
import importlib
import unittest

import torch

from src.alice_personality.identity.codec import canonical, fingerprint
from src.alice_personality.identity.consumer import CheckpointPin, DecisionConsumer, RenderedResponse
from src.alice_personality.identity.contracts import IdentityError
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from tests.personality.test_identity_codec import Backbone, Tokenizer, document
from tests.personality import test_identity_consumer as consumer_fixture
from src.alice_personality.teaching.contracts import TeachingBatch, TrainingRecipe
from src.alice_personality.teaching.trainer import IdentityTrainer
from tests.personality.test_identity_teaching import _admission, _target

try:
    api = importlib.import_module('scripts.eipm.gemma_n0.typed_candidates')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.typed_candidates':
        raise
    api = None


class TypedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(api, 'Typed candidate producer and governed consumer connection is missing')

    def candidates(self, doc, kind='utterance'):
        if kind == 'utterance':
            payloads = ({'text': 'All right.', 'delivery': {'intent': 'Dismissive and impatient.'}},
                        {'text': 'All right.', 'delivery': {'intent': 'Warm, measured reassurance.'}})
        elif kind == 'interpretation':
            payloads = ({'proposition': 'The speaker has already consented.'},
                        {'proposition': 'The speaker is uncertain; present permission is unknown.'})
        else:
            payloads = ({'operation': 'share_note', 'description': 'Share the note now.', 'arguments': {'note': 'example'}},
                        {'operation': 'ask_permission', 'description': 'Ask before sharing the note.', 'arguments': {'note': 'example'}})
        return tuple(api.TypedCandidate(entry.candidate_id, kind, payload)
                     for entry, payload in zip(doc.candidates, payloads))

    def build(self, kind='utterance', *, tie=False, n1_parent=False, locations=False):
        fixture = consumer_fixture.ConsumerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        candidates = self.candidates(fixture.doc, kind)
        if locations:
            for candidate in candidates:
                candidate.payload['delivery']['pause_locations'] = [{'offset': 1, 'intent': 'brief pause'}]
        prepared = api.compile_typed_candidates(fixture.doc, candidates)
        fixture.doc = prepared.document
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), fixture.config, _fixture=True)
        fixture.produced = producer.produce(fixture.doc)
        fixture.session = producer.close_and_verify()
        n1_pin = None
        if n1_parent:
            batch = TeachingBatch(fixture.produced.frame, {'concept_applicability': _target([[.8]])},
                ('typed-n1',), (('fictional-source-family',),), 'n1', 'TRAIN', 'public_mechanical_fixture')
            recipe = TrainingRecipe('n1', seed=411, learning_rate=.01, max_grad_norm=1, weight_decay=0)
            directory = fixture.path / 'typed-n1-admission'
            directory.mkdir()
            trainer = IdentityTrainer(fixture.model, recipe,
                _admission(directory, fixture.model, recipe, [batch]), first_batch=batch)
            trainer.step(batch)
            path = fixture.path / 'typed-n1-checkpoint'
            manifest = trainer.save_checkpoint(path)
            n1_pin = CheckpointPin(str(path), manifest['manifest_sha256'])
        consumer = fixture.build(tie=tie, co_valid=True)
        if n1_pin is not None:
            consumer = DecisionConsumer(fixture.model, (n1_pin, *consumer.checkpoints))
        return fixture, prepared, consumer

    def respond(self, fixture, prepared, consumer, **kwargs):
        return api.respond_typed(consumer, prepared, fixture.produced, fixture.session,
            expected_binding_sha256=prepared.binding_sha256, **kwargs)

    def test_complete_delivery_changes_source_text_while_ids_and_flags_do_not(self):
        doc = document()
        candidates = self.candidates(doc)
        prepared = api.compile_typed_candidates(doc, candidates)
        self.assertNotEqual(prepared.document.candidates[0].text, prepared.document.candidates[1].text)
        self.assertEqual(prepared.document.fields, doc.fields)
        self.assertEqual(prepared.document.sources, doc.sources)
        self.assertEqual(prepared.document.concepts, doc.concepts)
        self.assertEqual([e.authority_allowed for e in prepared.document.candidates], [True, False])
        self.assertEqual(prepared.document.candidates[0].text, candidates[0].semantic_text())
        self.assertNotIn(candidates[0].candidate_id, candidates[0].semantic_text())
        self.assertNotEqual(prepared.document.sha256, doc.sha256)
        renamed = replace(candidates[0], candidate_id='different-opaque-id')
        self.assertEqual(renamed.semantic_text(), candidates[0].semantic_text())

    def test_mapping_must_cover_exact_order_and_payload_is_copied(self):
        doc = document()
        candidates = self.candidates(doc)
        for changed in (candidates[:1], tuple(reversed(candidates)), (candidates[0], candidates[0])):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(IdentityError, 'ordered'):
                    api.compile_typed_candidates(doc, changed)
        prepared = api.compile_typed_candidates(doc, candidates)
        old_pin = prepared.binding_sha256
        candidates[0].payload['delivery']['intent'] = 'changed outside the compiled packet'
        prepared.validate()
        self.assertEqual(prepared.binding_sha256, old_pin)

    def test_types_do_not_accept_unknown_or_hidden_payload_fields(self):
        doc = document()
        first, second = self.candidates(doc)
        for candidate in (replace(first, kind='execute'),
                          replace(first, payload={**first.payload, 'preferred': True}),
                          replace(first, payload={'text': '', 'delivery': {}}),
                          replace(first, payload={'text': 'All right.', 'delivery': {'bad': float('nan')}})):
            with self.subTest(candidate=candidate):
                with self.assertRaises(IdentityError):
                    api.compile_typed_candidates(doc, (candidate, second))

    def test_delivery_locations_bind_original_unicode_speech_text(self):
        doc = document()
        first, second = self.candidates(doc)
        speech = 'Yes 🙂, really.'
        delivery = {'intent': 'Measured reassurance.',
            'emphasis_spans': [{'start': 7, 'end': 13, 'text': 'really', 'intent': 'gentle emphasis'}],
            'pause_locations': [{'offset': 6, 'intent': 'brief hesitation'}]}
        first = replace(first, payload={'text': speech, 'delivery': delivery})
        prepared = api.compile_typed_candidates(doc, (first, second))
        self.assertIn('unicode_code_point', prepared.document.candidates[0].text)
        invalid = [dict(delivery, emphasis_spans=None), dict(delivery, pause_locations='after Yes'),
            dict(delivery, emphasis_spans=[{'start': 0, 'end': 3, 'text': 'really', 'intent': 'emphasis'}]),
            dict(delivery, pause_locations=[{'offset': 99, 'intent': 'pause'}])]
        for value in invalid:
            with self.subTest(delivery=value):
                with self.assertRaises(IdentityError):
                    api.compile_typed_candidates(doc, (replace(first, payload={'text': speech, 'delivery': value}), second))

    def test_actual_optimized_same_words_delivery_reaches_replaceable_renderer(self):
        fixture, prepared, consumer = self.build()
        seen = []
        def renderer(request):
            seen.append(deepcopy(request))
            return RenderedResponse(request['speech_text'], request['delivery'], 'public-renderer', None)
        result = self.respond(fixture, prepared, consumer, renderer=renderer)
        self.assertEqual(result['status'], 'SELECTED')
        self.assertEqual(result['selected_candidate']['candidate_id'], 'opaque-candidate-b')
        self.assertEqual(result['speech_text'], 'All right.')
        self.assertEqual(result['delivery']['candidate_intent'], {'intent': 'Warm, measured reassurance.'})
        self.assertEqual(result['delivery']['native_voice_controls'], {})
        self.assertEqual(result['rendering']['status'], 'TEXT_REALIZED_DELIVERY_UNVERIFIED')
        self.assertIn('conditional_concepts', seen[0]['governance']['decision'])
        self.assertIn('native_latent', seen[0]['governance']['decision'])
        self.assertEqual(result['governance']['preference_selection']['document_sha256'], prepared.document.sha256)
        self.assertNotEqual(result['governance']['response_text'], result['speech_text'])
        self.assertFalse(result['execution_performed'])
        self.assertFalse(result['spoken_fidelity_verified'])
        self.assertFalse(result['acceptance_authority'])

    def test_interpretation_and_action_remain_typed_nonspoken_proposals(self):
        for kind in ('interpretation', 'action'):
            with self.subTest(kind=kind):
                fixture, prepared, consumer = self.build(kind)
                called = []
                result = self.respond(fixture, prepared, consumer, renderer=lambda request: called.append(request))
                self.assertEqual(result['status'], 'SELECTED')
                self.assertEqual(result['selected_candidate']['kind'], kind)
                self.assertEqual(result['disposition'], 'PROPOSAL_ONLY')
                self.assertEqual(called, [])
                self.assertIsNone(result['speech_text'])
                self.assertIsNone(result['delivery'])
                self.assertFalse(result['execution_performed'])
                self.assertFalse(result['experience_recorded'])

    def test_tie_does_not_choose_by_order_or_call_renderer(self):
        fixture, prepared, consumer = self.build(tie=True)
        called = []
        result = self.respond(fixture, prepared, consumer, renderer=lambda request: called.append(request))
        self.assertEqual(result['status'], 'UNRESOLVED_PLURALITY')
        self.assertIsNone(result['selected_candidate'])
        self.assertIsNone(result['speech_text'])
        self.assertEqual(called, [])
        self.assertEqual(len(result['governance']['selected_candidate_ids']), 2)

    def test_changed_typed_payload_or_old_features_are_refused(self):
        fixture, prepared, consumer = self.build()
        old_pin = prepared.binding_sha256
        prepared.candidates[1].payload['delivery']['intent'] = 'Impatient dismissal.'
        with self.assertRaisesRegex(IdentityError, 'typed'):
            api.respond_typed(consumer, prepared, fixture.produced, fixture.session, expected_binding_sha256=old_pin)
        plain = consumer_fixture.ConsumerTests()
        plain.setUp()
        self.addCleanup(plain.doCleanups)
        other = api.compile_typed_candidates(plain.doc, self.candidates(plain.doc))
        with self.assertRaisesRegex(IdentityError, 'document'):
            api.respond_typed(consumer, other, plain.produced, plain.session,
                expected_binding_sha256=other.binding_sha256)

    def test_renderer_changes_are_preserved_as_unresolved_and_mutation_is_isolated(self):
        fixture, prepared, consumer = self.build()
        original = self.respond(fixture, prepared, consumer)
        def changed(request):
            request['governance']['decision']['uncertainty'][0][0] = 999
            return RenderedResponse('Changed words.', {'candidate_intent': {'intent': 'Different'}}, 'changed', None)
        result = self.respond(fixture, prepared, consumer, renderer=changed)
        self.assertEqual(result['status'], 'UNRESOLVED_RENDERER')
        self.assertEqual(result['rendering']['text'], 'Changed words.')
        self.assertEqual(result['speech_text'], 'All right.')
        self.assertEqual(canonical(result['governance']), canonical(original['governance']))
        self.assertEqual(fingerprint(consumer.model.state_dict()), original['governance']['preference_selection']['model_state_sha256'])

    def test_callback_cannot_change_bound_typed_state(self):
        fixture, prepared, consumer = self.build()
        def mutate(request):
            prepared.candidates[1].payload['text'] = 'Changed words.'
            return RenderedResponse(request['speech_text'], request['delivery'], 'mutator', None)
        with self.assertRaisesRegex(IdentityError, 'typed'):
            self.respond(fixture, prepared, consumer, renderer=mutate)

    def test_renderer_boolean_cannot_substitute_for_an_integer_text_position(self):
        fixture, prepared, consumer = self.build(locations=True)
        def changed(request):
            request['delivery']['candidate_intent']['pause_locations'][0]['offset'] = True
            return RenderedResponse(request['speech_text'], request['delivery'], 'changed-position-type', None)
        result = self.respond(fixture, prepared, consumer, renderer=changed)
        self.assertEqual(result['status'], 'UNRESOLVED_RENDERER')
        self.assertIs(result['rendering']['delivery']['candidate_intent']['pause_locations'][0]['offset'], True)
        self.assertIs(type(result['delivery']['candidate_intent']['pause_locations'][0]['offset']), int)

    def test_actual_successor_keeps_current_scope_and_calibration_inapplicability(self):
        import json
        from pathlib import Path
        from scripts.eipm.gemma_n0.cumulative_successor import CumulativeSuccessor
        fixture, prepared, parent = self.build(n1_parent=True)
        successor = CumulativeSuccessor.load(fixture.config, parent.checkpoints)
        batch = TeachingBatch(fixture.produced.frame,
            {'preferences': _target([[.02, .98]]), 'co_valid_probabilities': _target([[1., 1.]])},
            ('fictional-judgment',), (('fictional-source-family',),), 'n2', 'TRAIN', 'public_mechanical_fixture')
        retained = tuple(sorted({event['teaching_batch_sha256'] for pin in parent.checkpoints
            for event in json.loads((Path(pin.path) / 'manifest.json').read_bytes())['events']}))
        recipe = TrainingRecipe('n2', seed=412, learning_rate=.01, max_grad_norm=1, weight_decay=0)
        directory = fixture.path / 'typed-successor-admission'
        directory.mkdir()
        admission = _admission(directory, successor.model, recipe, [batch])
        successor.run_phase(recipe, admission, batch_loader={batch.fingerprint(): batch}.__getitem__, epochs=1,
            destination=fixture.path / 'typed-successor-checkpoint', lineage_destination=fixture.path / 'typed-successor.json',
            retained_parent_batches=retained, superseded_parent_batches={}, objective_transfer=parent.checkpoints[0])
        consumer = successor.consumer()
        seen = []
        def renderer(request):
            seen.append(deepcopy(request))
            return RenderedResponse(request['speech_text'], request['delivery'], 'successor-renderer', None)
        result = self.respond(fixture, prepared, consumer, renderer=renderer)
        self.assertEqual(result['status'], 'SELECTED')
        self.assertFalse(result['governance']['calibration_applicability']['applicable'])
        self.assertTrue(result['governance']['calibration_applicability']['fresh_outcome_calibration_required'])
        self.assertTrue(result['governance']['output_scope']['concept_applicability']['n1_taught'])
        self.assertEqual(seen[0]['governance']['successor_lineage'], result['governance']['successor_lineage'])
        self.assertFalse(any(value['n3_reviewed'] for value in seen[0]['governance']['output_scope'].values()))
