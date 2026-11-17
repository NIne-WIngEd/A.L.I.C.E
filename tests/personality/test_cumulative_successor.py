"""Actual public-fictional continuation; no private identity or learned-fidelity claim."""
from dataclasses import replace
import importlib
from pathlib import Path
import unittest

import torch

from src.alice_personality.identity.codec import canonical, fingerprint
from src.alice_personality.identity.consumer import CheckpointPin, DecisionConsumer, RenderedResponse
from src.alice_personality.teaching.cases import NamedTarget
from src.alice_personality.teaching.contracts import TeachingError, TrainingRecipe
from tests.personality import test_connected_construction as fixture_api
from tests.personality import test_reviewed_teaching_cases as case_api
from tests.personality.test_identity_teaching import _admission

try:
    successor_api = importlib.import_module('scripts.eipm.gemma_n0.cumulative_successor')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.cumulative_successor':
        raise
    successor_api = None


class CumulativeSuccessorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(successor_api, 'The bounded cumulative successor interface is missing')
        self.fixture = fixture_api.ConstructionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.parent = fixture_api.ConnectedConstruction(self.fixture.model)
        for phase in ('n1', 'n2'):
            if phase == 'n1':
                from src.alice_personality.identity.contracts import SOURCE_KINDS
                self.parent_n1_case = replace(self.fixture.fixture.case(phase='n1', targets=(
                    NamedTarget('concept_applicability', 'independent',
                        (case_api.fact(self.fixture.fixture.concept),)),
                    NamedTarget('provenance', 'closed_distribution', tuple(
                        case_api.fact(self.fixture.fixture.source, kind, value=1.0 if kind == 'E0' else 0.0)
                        for kind in SOURCE_KINDS)))), case_id='public-n1')
                batch = self.fixture.fixture.build(self.parent_n1_case,
                    produced=self.fixture.produced, session=self.fixture.session).batch
                directory = self.fixture.path / 'n1'
                directory.mkdir()
                recipe = TrainingRecipe('n1', learning_rate=.01, max_grad_norm=1, seed=51)
                admission = _admission(directory, self.fixture.model, recipe, [batch],
                    reviewed_changes={'reviewed_case_artifact_sha256':
                        {self.parent_n1_case.sha256: dict(self.parent_n1_case.artifact_sha256)}})
                loader = {batch.fingerprint(): batch}.__getitem__
            else:
                original = replace(self.fixture.fixture.case(), case_id='public-n2')
                label = original.document.labels['emotion'][0].label_id
                self.parent_n2_case = replace(original, targets=(*original.targets,
                    NamedTarget('head:emotion', 'independent',
                        (case_api.fact(self.fixture.fixture.a, label, value=.7),))))
                batch = self.fixture.fixture.build(self.parent_n2_case,
                    produced=self.fixture.produced, session=self.fixture.session).batch
                directory = self.fixture.path / 'n2'
                directory.mkdir()
                recipe = TrainingRecipe('n2', learning_rate=.01, max_grad_norm=1, seed=51)
                admission = _admission(directory, self.fixture.model, recipe, [batch],
                    reviewed_changes={'reviewed_case_artifact_sha256':
                        {self.parent_n2_case.sha256: dict(self.parent_n2_case.artifact_sha256)}})
                loader = {batch.fingerprint(): batch}.__getitem__
            self.parent.run_phase(recipe, admission, batch_loader=loader, epochs=2,
                destination=self.fixture.path / ('parent-' + phase))
        self.parent_pins = self.parent.checkpoints
        self.parent_bytes = {str(path): path.read_bytes() for pin in self.parent_pins
            for path in Path(pin.path).iterdir()}
        self.parent_batches = tuple(sorted({event['teaching_batch_sha256']
            for pin in self.parent_pins for event in self.manifest(pin)['events']}))

    def manifest(self, pin):
        import json
        return json.loads((Path(pin.path) / 'manifest.json').read_bytes())

    def load(self, *, successors=()):
        return successor_api.CumulativeSuccessor.load(self.fixture.config, self.parent_pins,
            successors=successors)

    def admission(self, successor, *, phase='n2', supersede=False, new_label=False, preferences_only=False):
        old = self.parent_n2_case
        if phase == 'n1':
            old = self.parent_n1_case
        targets = (NamedTarget('co_valid_probabilities', 'independent',
            (case_api.fact(self.fixture.fixture.a, value=.15),
             case_api.fact(self.fixture.fixture.b, value=.85))),) if phase == 'n2' else (
                NamedTarget('concept_uncertainty', 'independent',
                    (case_api.fact(self.fixture.fixture.concept, value=.25),)),)
        if preferences_only:
            targets = (NamedTarget('preferences', 'closed_distribution',
                (case_api.fact(self.fixture.fixture.a, value=.1), case_api.fact(self.fixture.fixture.b, value=.9))),)
        doc = old.document
        produced = self.fixture.replay.produce(doc)
        session = self.fixture.session
        if new_label:
            from src.alice_personality.identity.contracts import LabelSpec
            from src.alice_personality.identity.feature_producer import FrozenFrameProducer
            from src.alice_personality.identity.feature_store import FrozenFeatureStore
            from tests.personality.test_identity_codec import Backbone, Tokenizer
            labels = dict(doc.labels)
            spec = doc.labels['emotion'][0]
            labels['emotion'] = (LabelSpec(spec.label_id, 'fictional delight',
                'Positive pleasure on receiving an explicitly welcome fictional surprise.'),)
            doc = replace(doc, labels=labels)
            store = FrozenFeatureStore.create(self.fixture.path / 'new-label-store',
                source_package_sha256=case_api.PINS['source_package'])
            producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.fixture.config,
                _fixture=True, feature_store=store)
            produced = producer.produce(doc)
            session = producer.close_and_verify()
            self.new_produced, self.new_session = produced, session
            targets += (NamedTarget('head:emotion', 'independent',
                (case_api.fact(self.fixture.fixture.a, spec.label_id, value=.9),)),)
        new = replace(old, case_id='public-added-' + phase, document=doc, targets=targets)
        batches = [] if supersede else [self.fixture.fixture.build(old,
            produced=self.fixture.produced, session=self.fixture.session).batch]
        batches.append(self.fixture.fixture.build(new, produced=produced, session=session).batch)
        folder = self.fixture.path / ('admission-' + phase)
        folder.mkdir()
        recipe = TrainingRecipe(phase, learning_rate=.02, max_grad_norm=1, seed=113)
        cases = (new,) if supersede else (old, new)
        admission = _admission(folder, successor.model, recipe, batches,
            reviewed_changes={'reviewed_case_artifact_sha256':
                {case.sha256: dict(case.artifact_sha256) for case in cases}})
        indexed = {batch.fingerprint(): batch for batch in batches}
        return recipe, admission, indexed.__getitem__, batches

    def run_n2(self, successor, **changes):
        recipe, admission, loader, batches = self.admission(successor, **changes)
        args = dict(recipe=recipe, admission=admission, batch_loader=loader, epochs=4,
            destination=self.fixture.path / 'successor-n2',
            lineage_destination=self.fixture.path / 'successor-n2-lineage.json',
            retained_parent_batches=self.parent_batches, superseded_parent_batches={},
            objective_transfer=self.parent_pins[0])
        return successor.run_phase(**args), batches

    def test_real_n2_only_updates_warm_parent_and_replays_valid_prior_teaching(self):
        successor = self.load()
        before = {name: value.clone() for name, value in successor.model.state_dict().items()}
        self.assertEqual(fingerprint(before), self.manifest(self.parent_pins[-1])['model_state_sha256'])
        result, batches = self.run_n2(successor)
        self.assertEqual(result['checkpoint']['phase'], 'n2')
        self.assertEqual(result['checkpoint']['step'], 8)
        self.assertEqual(result['lineage']['optimizer_continuation'], 'NEW_ADMISSION_FRESH_ADAMW')
        self.assertEqual(result['lineage']['updated_stages'], ['n2'])
        self.assertEqual(result['lineage']['parent_model_state_sha256'], fingerprint(before))
        self.assertEqual(result['lineage']['objective_initial_state_sha256'],
            fingerprint(torch.load(Path(self.parent_pins[0].path) / 'resume.pt', weights_only=True)['objectives']))
        self.assertEqual(result['lineage']['objective_initial_state_sha256'],
            result['lineage']['objective_final_state_sha256'])
        self.assertTrue(all(torch.equal(value, successor.model.state_dict()[name])
            for name, value in before.items() if name.startswith(('n1.', 'n3.'))))
        self.assertTrue(any(not torch.equal(value, successor.model.state_dict()[name])
            for name, value in before.items() if name.startswith('n2.')))
        self.assertEqual([event['teaching_batch_sha256'] for event in result['checkpoint']['events']],
            [batch.fingerprint() for batch in batches] * 4)
        self.assertTrue(all(Path(path).read_bytes() == data for path, data in self.parent_bytes.items()))

    def test_reload_full_governed_consumer_uses_original_methods_and_exact_lineage(self):
        successor = self.load()
        result, _ = self.run_n2(successor)
        restored = self.load(successors=successor.successors)
        consumer = restored.consumer()
        self.assertIsInstance(consumer, DecisionConsumer)
        self.assertIs(type(consumer).respond, DecisionConsumer.respond)
        response = successor_api.respond_with_governance(consumer, self.fixture.fixture.doc,
            self.fixture.replay.produce(self.fixture.fixture.doc), self.fixture.session)
        self.assertEqual(response['preference_selection']['model_state_sha256'],
            result['checkpoint']['model_state_sha256'])
        self.assertEqual(len(response['preference_selection']['checkpoint_manifest_sha256']), 3)
        self.assertEqual(response['successor_lineage']['parent_checkpoint_manifest_sha256'],
            [pin.manifest_sha256 for pin in self.parent_pins])
        self.assertTrue(response['output_scope']['concept_applicability']['n1_taught'])
        self.assertTrue(response['output_scope']['co_valid_probabilities']['n2_taught'])
        self.assertIn('native_latent', response['decision'])
        self.assertIn('conditional_concepts', response['decision'])
        self.assertFalse(response['calibration_applicability']['applicable'])
        self.assertFalse(response['owner_fidelity_approved'])

    def test_retained_same_phase_replay_cannot_be_omitted_before_updates(self):
        successor = self.load()
        recipe, admission, loader, _ = self.admission(successor, supersede=True)
        before = fingerprint(successor.model.state_dict())
        with self.assertRaisesRegex(TeachingError, 'replay'):
            successor.run_phase(recipe, admission, batch_loader=loader, epochs=1,
                destination=self.fixture.path / 'bad', lineage_destination=self.fixture.path / 'bad.json',
                retained_parent_batches=self.parent_batches, superseded_parent_batches={})
        self.assertEqual(fingerprint(successor.model.state_dict()), before)

    def test_native_successor_checkpoint_exact_resume_preserves_original_admission_checks(self):
        from src.alice_personality.identity import IdentityModel
        from src.alice_personality.teaching.trainer import IdentityTrainer
        successor = self.load()
        recipe, admission, loader, batches = self.admission(successor)
        initial = {name: value.clone() for name, value in successor.model.state_dict().items()}
        result = successor.run_phase(recipe, admission, batch_loader=loader, epochs=1,
            destination=self.fixture.path / 'resumeable-child',
            lineage_destination=self.fixture.path / 'resumeable-child.json',
            retained_parent_batches=self.parent_batches, superseded_parent_batches={},
            objective_transfer=self.parent_pins[0])
        model = IdentityModel(self.fixture.config)
        model.load_state_dict(initial, strict=True)
        trainer = IdentityTrainer(model, recipe, admission, first_batch=batches[0])
        trainer.resume_checkpoint(self.fixture.path / 'resumeable-child',
            expected_manifest_sha256=result['checkpoint']['manifest_sha256'])
        self.assertEqual(trainer.steps, 2)
        self.assertEqual(trainer.epoch, 1)
        self.assertEqual(trainer.batch_index, 0)
        self.assertEqual(fingerprint(model.state_dict()), result['checkpoint']['model_state_sha256'])
        self.assertEqual(fingerprint(trainer.objectives.state_dict()), result['lineage']['objective_final_state_sha256'])
        self.assertEqual(trainer.step(batches[0])['step'], 3)
        directory = self.fixture.path / 'changed-resume-recipe'
        directory.mkdir()
        changed_recipe = replace(recipe, learning_rate=.03)
        model = IdentityModel(self.fixture.config)
        model.load_state_dict(initial, strict=True)
        changed = _admission(directory, model, changed_recipe, batches,
            reviewed_changes={'reviewed_case_artifact_sha256':
                successor_api._json(admission.artifacts['reviewed_targets'].verify())['reviewed_case_artifact_sha256']})
        different = IdentityTrainer(model, changed_recipe, changed, first_batch=batches[0])
        with self.assertRaisesRegex(TeachingError, 'binding differs'):
            different.resume_checkpoint(self.fixture.path / 'resumeable-child',
                expected_manifest_sha256=result['checkpoint']['manifest_sha256'])

    def test_second_actual_n2_successor_keeps_full_parent_links_and_cumulative_replay(self):
        successor = self.load()
        first, batches = self.run_n2(successor)
        second_case = replace(self.parent_n2_case, case_id='public-third-n2', targets=(
            NamedTarget('evidence_sufficiency', 'independent',
                (case_api.fact(self.fixture.fixture.a, value=.8), case_api.fact(self.fixture.fixture.b, value=.8))),))
        batch = self.fixture.fixture.build(second_case, produced=self.fixture.produced,
            session=self.fixture.session).batch
        batches.append(batch)
        directory = self.fixture.path / 'second-admission'
        directory.mkdir()
        recipe = TrainingRecipe('n2', learning_rate=.01, max_grad_norm=1, seed=127)
        reviewed = successor_api._json(self.fixture.path / 'admission-n2' / 'reviewed_targets.json')
        case_pins = dict(reviewed['reviewed_case_artifact_sha256'])
        case_pins[second_case.sha256] = dict(second_case.artifact_sha256)
        admission = _admission(directory, successor.model, recipe, batches,
            reviewed_changes={'reviewed_case_artifact_sha256': case_pins})
        indexed = {item.fingerprint(): item for item in batches}
        result = successor.run_phase(recipe, admission, batch_loader=indexed.__getitem__, epochs=1,
            destination=self.fixture.path / 'second-n2', lineage_destination=self.fixture.path / 'second-n2.json',
            retained_parent_batches=tuple(first['lineage']['valid_governing_batches']), superseded_parent_batches={},
            objective_transfer=CheckpointPin(**first['lineage']['checkpoint']))
        self.assertEqual(result['checkpoint']['step'], 3)
        self.assertEqual(result['lineage']['parent_model_state_sha256'], first['checkpoint']['model_state_sha256'])
        reloaded = self.load(successors=successor.successors)
        self.assertEqual(len(reloaded.successors), 2)
        self.assertEqual(len(reloaded.consumer().checkpoints), 4)

    def test_explicit_superseded_target_revision_is_excluded_from_current_scope(self):
        successor = self.load()
        recipe, admission, loader, batches = self.admission(successor, supersede=True)
        old_n2 = self.manifest(self.parent_pins[-1])['events'][0]['teaching_batch_sha256']
        retained = tuple(digest for digest in self.parent_batches if digest != old_n2)
        result = successor.run_phase(recipe, admission, batch_loader=loader, epochs=2,
            destination=self.fixture.path / 'corrected', lineage_destination=self.fixture.path / 'corrected.json',
            retained_parent_batches=retained,
            superseded_parent_batches={old_n2: 'fictional target revision: preference gold withdrawn'})
        response = successor_api.respond_with_governance(successor.consumer(), self.fixture.fixture.doc,
            self.fixture.replay.produce(self.fixture.fixture.doc), self.fixture.session)
        self.assertFalse(response['output_scope']['preferences']['n2_taught'])
        self.assertNotIn(old_n2, result['lineage']['valid_governing_batches'])
        self.assertEqual(response['preference_selection']['status'], 'UNRESOLVED_UNTAUGHT')
        batches.append(self.fixture.fixture.build(self.parent_n2_case,
            produced=self.fixture.produced, session=self.fixture.session).batch)
        directory = self.fixture.path / 'resurrected-admission'
        directory.mkdir()
        reviewed = successor_api._json(admission.artifacts['reviewed_targets'].verify())
        pins = dict(reviewed['reviewed_case_artifact_sha256'])
        pins[self.parent_n2_case.sha256] = dict(self.parent_n2_case.artifact_sha256)
        readmission = _admission(directory, successor.model, recipe, batches,
            reviewed_changes={'reviewed_case_artifact_sha256': pins})
        indexed = {batch.fingerprint(): batch for batch in batches}
        before = fingerprint(successor.model.state_dict())
        with self.assertRaisesRegex(TeachingError, 'superseded'):
            successor.run_phase(recipe, readmission, batch_loader=indexed.__getitem__, epochs=1,
                destination=self.fixture.path / 'resurrected', lineage_destination=self.fixture.path / 'resurrected.json',
                retained_parent_batches=tuple(result['lineage']['valid_governing_batches']),
                superseded_parent_batches={})
        self.assertEqual(fingerprint(successor.model.state_dict()), before)

    def test_parent_payload_admission_and_lineage_tampering_are_rejected(self):
        successor = self.load()
        self.run_n2(successor)
        path = Path(self.manifest(self.parent_pins[-1])['binding']['artifacts']['reviewed_targets']['path'])
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(TeachingError, 'artifact file'):
            self.load(successors=successor.successors)
        path.write_bytes(original)
        pin = successor.successors[0]
        with Path(pin.path).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(TeachingError):
            self.load(successors=successor.successors)
        with (Path(self.parent_pins[-1].path) / 'resume.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(TeachingError, 'payload'):
            self.load()

    def test_explicit_n2_adaptation_reports_both_actual_changed_governors(self):
        successor = self.load()
        recipe, original_admission, loader, batches = self.admission(successor)
        recipe = replace(recipe, adapt_n1=True)
        folder = self.fixture.path / 'adapted-admission'
        folder.mkdir()
        reviewed = successor_api._json(original_admission.artifacts['reviewed_targets'].verify())
        admission = _admission(folder, successor.model, recipe, batches,
            reviewed_changes={'reviewed_case_artifact_sha256': reviewed['reviewed_case_artifact_sha256']})
        before = {name: value.clone() for name, value in successor.model.state_dict().items()}
        result = successor.run_phase(recipe, admission, batch_loader=loader, epochs=1,
            destination=self.fixture.path / 'adapted-n2', lineage_destination=self.fixture.path / 'adapted-n2.json',
            retained_parent_batches=self.parent_batches, superseded_parent_batches={})
        self.assertEqual(result['lineage']['updated_stages'], ['n1', 'n2'])
        self.assertEqual(result['lineage']['preserved_stages'], ['n3'])
        self.assertTrue(any(not torch.equal(value, successor.model.state_dict()[name])
            for name, value in before.items() if name.startswith('n1.')))
        self.assertTrue(all(torch.equal(value, successor.model.state_dict()[name])
            for name, value in before.items() if name.startswith('n3.')))

    def test_n1_successor_requires_explicit_auxiliary_state_transfer(self):
        successor = self.load()
        recipe, admission, loader, _ = self.admission(successor, phase='n1')
        args = dict(batch_loader=loader, epochs=1, destination=self.fixture.path / 'n1-added',
            lineage_destination=self.fixture.path / 'n1-added.json',
            retained_parent_batches=self.parent_batches, superseded_parent_batches={})
        with self.assertRaisesRegex(TeachingError, 'objective'):
            successor.run_phase(recipe, admission, **args)
        result = successor.run_phase(recipe, admission, objective_transfer=self.parent_pins[0], **args)
        self.assertEqual(result['lineage']['updated_stages'], ['n1'])
        self.assertEqual(result['lineage']['objective_transfer']['manifest_sha256'], self.parent_pins[0].manifest_sha256)
        from src.alice_personality.teaching.objectives import IdentityObjectives
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51)
            initial = IdentityObjectives(self.fixture.config.learned_width)
        transferred = torch.load(Path(self.parent_pins[0].path) / 'resume.pt', weights_only=True)['objectives']
        self.assertNotEqual(fingerprint(initial.state_dict()), fingerprint(transferred))
        self.assertNotEqual(result['lineage']['objective_initial_state_sha256'],
            result['lineage']['objective_final_state_sha256'])

    def test_new_semantic_meaning_is_taught_without_classifier_resize_or_old_meaning_alias(self):
        successor = self.load()
        geometry = {name: value.shape for name, value in successor.model.state_dict().items()}
        result, _ = self.run_n2(successor, new_label=True)
        consumer = successor.consumer()
        doc = replace(self.fixture.fixture.doc, labels=dict(self.fixture.fixture.doc.labels,
            emotion=(self.new_produced.frame.label_specs['emotion'][0][0],)))
        response = successor_api.respond_with_governance(consumer, doc, self.new_produced, self.new_session)
        self.assertTrue(response['output_scope']['head:emotion']['n2_taught'])
        self.assertEqual(response['output_scope']['head:emotion']['reviewed_current_label_ids'], ['emotion-id'])
        self.assertEqual(geometry, {name: value.shape for name, value in successor.model.state_dict().items()})
        from src.alice_personality.identity.feature_producer import FrozenFrameProducer
        from tests.personality.test_identity_codec import Backbone, Tokenizer
        labels = dict(doc.labels)
        labels['emotion'] = (replace(labels['emotion'][0], name='fictional grief',
            description='Sadness following the irreversible fictional loss of a close friend.'),)
        changed = replace(doc, labels=labels)
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.fixture.config, _fixture=True)
        produced = producer.produce(changed)
        session = producer.close_and_verify()
        changed_response = successor_api.respond_with_governance(consumer, changed, produced, session)
        self.assertFalse(changed_response['output_scope']['head:emotion']['n2_taught'])
        self.assertEqual(changed_response['output_scope']['head:emotion']['reviewed_current_label_ids'], [])
        self.assertEqual(changed_response['preference_selection']['intents'][0]['heads']['emotion'], [])

    def test_parent_only_calibration_cannot_teach_semantic_meanings_or_voice_units(self):
        from src.alice_personality.teaching.lineage import ConstructionLineage
        from src.alice_personality.teaching.outcomes import calibration_case_from_review
        from scripts.eipm.gemma_n0.governed_response import respond_with_governance as native_governed
        from tests.personality.test_construction_lineage import allocation
        doc = self.fixture.fixture.doc
        produced = self.fixture.replay.produce(doc)
        response = self.parent.consumer().respond(doc, produced, self.fixture.session)
        candidate = response['selected_candidate_ids'][0]
        case = calibration_case_from_review(response_receipt=response,
            expected_response_sha256=response['receipt_sha256'], document=doc,
            case_id='fictional-parent-semantic-outcome', split='CALIBRATION_TRAIN',
            source_class='public_mechanical_fixture',
            lineage=ConstructionLineage((self.fixture.fixture.family,), (), ('fictional-semantic-outcome',)),
            targets=(NamedTarget('head:communication', 'independent',
                         (case_api.fact(candidate, doc.labels['communication'][0].label_id, value=.8),)),
                     NamedTarget('voice_control_confidence', 'independent',
                         (case_api.fact(candidate, doc.labels['voice'][0].label_id, value=.6),)),
                     NamedTarget('failure_tail_risk', 'independent', (case_api.fact(candidate, value=.8),))),
            artifact_sha256=case_api.PINS, review_id='fictional-parent-semantic-review',
            annotation_note='Explicit fictional outcome review; no semantic teaching or spoken approval.')
        batch = self.fixture.fixture.build(case, produced=self.fixture.produced, session=self.fixture.session,
            families=allocation(self.fixture.fixture.family, 'fictional-semantic-outcome')).batch
        directory = self.fixture.path / 'parent-semantic-n3-admission'
        directory.mkdir()
        recipe = TrainingRecipe('calibration', learning_rate=.02, max_grad_norm=1, seed=119)
        admission = _admission(directory, self.fixture.model, recipe, [batch],
            reviewed_changes={'reviewed_case_artifact_sha256': {case.sha256: dict(case.artifact_sha256)}})
        before = {name: value.clone() for name, value in self.fixture.model.state_dict().items()}
        self.parent.run_phase(recipe, admission, batch_loader={batch.fingerprint(): batch}.__getitem__,
            epochs=2, destination=self.fixture.path / 'parent-semantic-n3')
        changed = {name for name, value in self.fixture.model.state_dict().items()
            if not torch.equal(before[name], value)}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith('n3.') for name in changed))
        self.parent_pins = self.parent.checkpoints
        calls = {}
        def renderer(route):
            def realize(packet):
                calls[route] = packet
                return RenderedResponse(packet['response_text'], {}, 'fictional-semantic-renderer', None)
            return realize
        native = native_governed(self.parent.consumer(), doc, produced, self.fixture.session,
            renderer=renderer('native'))
        cumulative = successor_api.respond_with_governance(self.load().consumer(), doc, produced,
            self.fixture.session, renderer=renderer('cumulative'))
        self.assertTrue(cumulative['calibration_applicability']['applicable'])
        self.assertEqual(cumulative['calibration_applicability']['numerical_scaling'], 'PARENT_N3')
        self.assertEqual(set(calls), {'native', 'cumulative'})
        for field in ('head:communication', 'voice_control_confidence'):
            for route, packet in (('native-final', native), ('native-callback', calls['native'])):
                with self.subTest(control=route, field=field):
                    scope = packet['output_scope'][field]
                    self.assertFalse(scope['n2_taught'])
                    self.assertTrue(scope['n3_reviewed'])
                    self.assertFalse(scope['taught'])
            for route, packet in (('cumulative-final', cumulative), ('cumulative-callback', calls['cumulative'])):
                with self.subTest(route=route, field=field):
                    self.assertEqual(packet['output_scope'][field], native['output_scope'][field])
        for field in ('concept_applicability', 'preferences', 'head:emotion', 'failure_tail_risk'):
            with self.subTest(retained_control=field):
                self.assertTrue(cumulative['output_scope'][field]['taught'])
                self.assertEqual(cumulative['output_scope'][field], native['output_scope'][field])

    def test_changed_governor_preserves_real_historical_n3_but_never_advertises_it_to_renderer(self):
        from src.alice_personality.teaching.lineage import ConstructionLineage
        from src.alice_personality.teaching.outcomes import calibration_case_from_review
        from tests.personality.test_construction_lineage import allocation
        response = self.parent.consumer().respond(self.fixture.fixture.doc,
            self.fixture.replay.produce(self.fixture.fixture.doc), self.fixture.session)
        candidate = response['selected_candidate_ids'][0]
        case = calibration_case_from_review(response_receipt=response,
            expected_response_sha256=response['receipt_sha256'], document=self.fixture.fixture.doc,
            case_id='fictional-observed-response', split='CALIBRATION_TRAIN',
            source_class='public_mechanical_fixture',
            lineage=ConstructionLineage((self.fixture.fixture.family,), (), ('fictional-outcome',)),
            targets=(NamedTarget('failure_tail_risk', 'independent', (case_api.fact(candidate, value=.8),)),),
            artifact_sha256=case_api.PINS, review_id='fictional-reviewed-outcome',
            annotation_note='Explicit fictional development outcome; not owner fidelity.')
        batch = self.fixture.fixture.build(case, produced=self.fixture.produced, session=self.fixture.session,
            families=allocation(self.fixture.fixture.family, 'fictional-outcome')).batch
        directory = self.fixture.path / 'n3'
        directory.mkdir()
        recipe = TrainingRecipe('calibration', learning_rate=.02, max_grad_norm=1, seed=119)
        admission = _admission(directory, self.fixture.model, recipe, [batch],
            reviewed_changes={'reviewed_case_artifact_sha256': {case.sha256: dict(case.artifact_sha256)}})
        self.parent.run_phase(recipe, admission, batch_loader={batch.fingerprint(): batch}.__getitem__,
            epochs=4, destination=self.fixture.path / 'parent-n3')
        self.parent_pins = self.parent.checkpoints
        successor = self.load()
        result, _ = self.run_n2(successor, preferences_only=True)
        calls = []
        def renderer(packet):
            calls.append(packet)
            return RenderedResponse(packet['response_text'], {}, 'fictional-renderer', None)
        consumer = successor.consumer()
        original = successor_api.respond_with_governance(consumer, self.fixture.fixture.doc,
            self.fixture.replay.produce(self.fixture.fixture.doc), self.fixture.session, renderer=renderer)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]['calibration_applicability']['applicable'])
        self.assertFalse(calls[0]['output_scope']['failure_tail_risk']['n3_reviewed'])
        self.assertFalse(original['calibration_applicability']['applicable'])
        self.assertEqual(original['calibration_applicability']['historical_checkpoint_manifest_sha256'],
            [self.parent_pins[-1].manifest_sha256])
        self.assertEqual(original['calibration_applicability']['numerical_scaling'], 'RETAINED_HISTORICAL_N3_INAPPLICABLE')
        self.assertFalse(original['output_scope']['failure_tail_risk']['n3_reviewed'])
        self.assertFalse(original['output_scope']['failure_tail_risk']['taught'])
        self.assertEqual(result['lineage']['preserved_stages'], ['n1', 'n3'])



if __name__ == '__main__':
    unittest.main()
