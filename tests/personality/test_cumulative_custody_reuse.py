"""Native public mechanics: custody reuse is not publisher latency or learning proof."""
from contextlib import contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.alice_personality.identity.codec import canonical
from src.alice_personality.teaching.contracts import TeachingError
from src.alice_personality.teaching.trainer import TeachingAdmission, model_fingerprint
from tests.personality import test_cumulative_successor as fixture_api

successor_api = fixture_api.successor_api


@contextmanager
def native_admission_trace():
    """Record requested depths while executing the actual admission validator."""
    records = []
    native_verify = TeachingAdmission.verify

    def verify(admission, model, recipe, **kwargs):
        records.append({'phase': recipe.phase, 'initial': kwargs.get('initial', False),
            'deep_custody': kwargs.get('deep_custody', True),
            'reviewed_targets': admission.artifacts['reviewed_targets'].path})
        return native_verify(admission, model, recipe, **kwargs)

    with patch.object(TeachingAdmission, 'verify', new=verify):
        yield records


def emit_trace(stage, records):
    # These small native traces are retained in the RED/GREEN test receipts.
    print(canonical({'stage': stage, 'native_admission_calls': records}).decode())


class CumulativeCustodyReuseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_api.CumulativeSuccessorTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def respond(self, consumer):
        fixture = self.fixture.fixture
        return successor_api.respond_with_governance(consumer, fixture.fixture.doc,
            fixture.replay.produce(fixture.fixture.doc), fixture.session)

    def test_parent_load_is_deep_and_repeated_response_reuses_native_admission(self):
        with native_admission_trace() as records:
            successor = self.fixture.load()
            load = list(records)
            records.clear()
            consumer = successor.consumer()
            construction = list(records)
            records.clear()
            response = self.respond(consumer)
            emit_trace('parent-load', load)
            emit_trace('parent-consumer', construction)
            emit_trace('parent-response', records)
            self.assertEqual([item['deep_custody'] for item in load], [True, True])
            self.assertEqual(response['preference_selection']['model_state_sha256'],
                self.fixture.manifest(self.fixture.parent_pins[-1])['model_state_sha256'])
            self.assertEqual(response['preference_selection']['checkpoint_manifest_sha256'],
                [pin.manifest_sha256 for pin in self.fixture.parent_pins])
            self.assertIn('native_latent', response['decision'])
            self.assertFalse(response['owner_fidelity_approved'])
            self.assertEqual([item['deep_custody'] for item in construction], [False] * 4)
            self.assertEqual([item['deep_custody'] for item in records], [False] * 8)

    def test_child_and_auxiliary_are_deep_on_reload_then_reused_in_response(self):
        successor = self.fixture.load()
        result, _ = self.fixture.run_n2(successor)
        with native_admission_trace() as records:
            restored = self.fixture.load(successors=successor.successors)
            load = list(records)
            records.clear()
            consumer = restored.consumer()
            records.clear()
            response = self.respond(consumer)
            emit_trace('child-load-with-auxiliary', load)
            emit_trace('child-response-with-auxiliary', records)
            self.assertEqual([item['phase'] for item in load], ['n1', 'n2', 'n2', 'n1'])
            self.assertEqual([item['deep_custody'] for item in load], [True] * 4)
            self.assertEqual(response['preference_selection']['model_state_sha256'],
                result['checkpoint']['model_state_sha256'])
            self.assertFalse(response['calibration_applicability']['applicable'])
            self.assertEqual([item['deep_custody'] for item in records], [False] * 16)
            records.clear()
            restored._verify(deep_custody=True)
            emit_trace('child-explicit-full-recheck', records)
            self.assertEqual([item['deep_custody'] for item in records], [True] * 4)

    def test_new_admission_and_training_initialization_keep_full_checks(self):
        successor = self.fixture.load()
        recipe, admission, loader, _ = self.fixture.admission(successor)
        with native_admission_trace() as records:
            successor.run_phase(recipe, admission, batch_loader=loader, epochs=1,
                destination=self.fixture.fixture.path / 'new-admission-checkpoint',
                lineage_destination=self.fixture.fixture.path / 'new-admission-lineage.json',
                retained_parent_batches=self.fixture.parent_batches, superseded_parent_batches={},
                objective_transfer=self.fixture.parent_pins[0])
            emit_trace('fresh-admission-initialization', records)
            initial = [item for item in records if item['initial']]
            self.assertEqual(len(initial), 2)
            self.assertTrue(all(item['deep_custody'] is True for item in initial))
            self.assertTrue(all(item['reviewed_targets'] == admission.artifacts['reviewed_targets'].path
                for item in initial))
            self.assertEqual([item['deep_custody'] for item in records[:2]], [True, True])
            self.assertEqual(successor._verify()['manifests'][-1]['step'], 2)

    def test_bound_artifact_changes_are_refused_on_repeated_consumer_path(self):
        consumer = self.fixture.load().consumer()
        binding = self.fixture.manifest(self.fixture.parent_pins[0])['binding']['artifacts']
        for role, pin in binding.items():
            with self.subTest(role=role):
                path = Path(pin['path'])
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b'\n')
                    with native_admission_trace() as records:
                        with self.assertRaisesRegex(TeachingError, 'artifact file'):
                            self.respond(consumer)
                        emit_trace('refused-changed-' + role, records)
                        self.assertEqual([item['deep_custody'] for item in records], [False])
                finally:
                    path.write_bytes(original)
        self.assertEqual(self.respond(consumer)['qualification'], 'UNQUALIFIED')

    def test_resume_payload_changes_are_refused_on_repeated_consumer_path(self):
        consumer = self.fixture.load().consumer()
        path = Path(self.fixture.parent_pins[0].path) / 'resume.pt'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'changed-payload')
            with self.assertRaisesRegex(TeachingError, 'payload'):
                self.respond(consumer)
        finally:
            path.write_bytes(original)
        self.assertEqual(self.respond(consumer)['qualification'], 'UNQUALIFIED')

    def test_actual_model_mutation_is_refused_after_custody_reuse(self):
        successor = self.fixture.load()
        consumer = successor.consumer()
        before = {name: value.detach().clone() for name, value in successor.model.state_dict().items()}
        with torch.no_grad():
            next(successor.model.parameters()).add_(1)
        with self.assertRaisesRegex(TeachingError, 'actual successor model'):
            self.respond(consumer)
        successor.model.load_state_dict(before, strict=True)
        self.assertEqual(self.respond(consumer)['preference_selection']['model_state_sha256'],
            model_fingerprint(successor.model))

    def test_reuse_cannot_skip_a_cold_constructions_initial_full_check(self):
        cold = successor_api.CumulativeSuccessor()
        cold.config = self.fixture.fixture.config
        cold.parent_checkpoints = self.fixture.parent_pins
        cold.successors = ()
        cold.model = None
        with native_admission_trace() as records:
            cold._verify(deep_custody=False)
            emit_trace('cold-explicit-reuse-still-full', records)
            self.assertEqual([item['deep_custody'] for item in records], [True, True])

    def test_custody_depth_requires_an_actual_boolean(self):
        successor = self.fixture.load()
        for invalid in (None, 0, 1, 'false'):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(TeachingError, 'Boolean'):
                    successor._verify(deep_custody=invalid)
                with self.assertRaisesRegex(TeachingError, 'Boolean'):
                    successor_api._checkpoint(self.fixture.fixture.config,
                        self.fixture.parent_pins[0], deep_custody=invalid)


if __name__ == '__main__':
    unittest.main()
