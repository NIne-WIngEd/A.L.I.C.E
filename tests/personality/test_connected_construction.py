"""Full source features streamed through actual N1/N2 construction phases."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.alice_personality.identity import IdentityModel
from src.alice_personality.identity.feature_store import FrozenFeatureStore
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.teaching.cases import NamedTarget
from src.alice_personality.teaching.construction import ConnectedConstruction
from src.alice_personality.teaching.contracts import TeachingError, TrainingRecipe
from tests.personality import test_reviewed_teaching_cases as case_fixture
from tests.personality.test_identity_codec import Backbone, Tokenizer
from tests.personality.test_identity_teaching import _admission


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = case_fixture.ReviewedCaseTests()
        self.fixture.setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve()
        self.config = self.fixture.config
        self.model = IdentityModel(self.config)
        store = FrozenFeatureStore.create(self.path / 'store', source_package_sha256=case_fixture.PINS['source_package'])
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.config, _fixture=True, feature_store=store)
        self.produced = producer.produce(self.fixture.doc)
        self.session = producer.close_and_verify()
        self.replay = FrozenFrameProducer.replay(store, self.config, self.session,
            expected_session_sha256=self.session['receipt_sha256'])

    def phase(self, phase):
        targets = (NamedTarget('concept_applicability', 'independent',
                   (case_fixture.fact(self.fixture.concept),)),) if phase == 'n1' else None
        case = replace(self.fixture.case(phase=phase, targets=targets), case_id='public-' + phase)
        built = self.fixture.build(case, produced=self.produced, session=self.session)
        folder = self.path / phase
        folder.mkdir()
        recipe = TrainingRecipe(phase, learning_rate=.01, max_grad_norm=1, seed=51)
        admission = _admission(folder, self.model, recipe, [built.batch],
            reviewed_changes={'reviewed_case_artifact_sha256': {case.sha256: dict(case.artifact_sha256)}})
        seen = []
        def loader(digest):
            seen.append(digest)
            batch = self.fixture.build(case, produced=self.replay.produce(case.document), session=self.session).batch
            self.assertEqual(batch.fingerprint(), digest)
            return batch
        return recipe, admission, loader, seen

    def test_complete_phases_stream_batches_and_bind_actual_checkpoint_chain(self):
        construction = ConnectedConstruction(self.model)
        for phase in ('n1', 'n2'):
            recipe, admission, loader, seen = self.phase(phase)
            receipt = construction.run_phase(recipe, admission, batch_loader=loader, epochs=2,
                destination=self.path / ('checkpoint-' + phase))
            self.assertEqual(receipt['step'], 2)
            self.assertEqual(len(seen), 2)
            self.assertFalse(receipt['acceptance_authority'])
        self.assertEqual(len(construction.checkpoints), 2)
        response = construction.consumer().respond(self.fixture.doc, self.replay.produce(self.fixture.doc), self.session)
        self.assertEqual(len(response['checkpoint_manifest_sha256']), 2)
        self.assertEqual(response['training_state'], 'OPTIMIZED_UNQUALIFIED')

    def test_no_skipped_governor_or_changed_initial_model_between_phases(self):
        construction = ConnectedConstruction(self.model)
        recipe, admission, loader, seen = self.phase('n2')
        with self.assertRaisesRegex(TeachingError, 'stage order'):
            construction.run_phase(recipe, admission, batch_loader=loader, epochs=1, destination=self.path / 'illegal')
        self.assertEqual(seen, [])

    def test_stale_or_missing_streamed_batch_cannot_enter_optimizer(self):
        construction = ConnectedConstruction(self.model)
        recipe, admission, loader, _ = self.phase('n1')
        def stale(digest):
            return replace(loader(digest), case_ids=('changed-case',))
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with self.assertRaises(TeachingError):
            construction.run_phase(recipe, admission, batch_loader=stale, epochs=1, destination=self.path / 'stale')
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[k]) for k, v in before.items()))

    def test_epoch_checkpoints_preserve_actual_optimizer_progress_without_advancing_stage(self):
        construction = ConnectedConstruction(self.model)
        recipe, admission, loader, seen = self.phase('n1')
        epochs = self.path / 'epochs'
        epochs.mkdir()
        final = construction.run_phase(recipe, admission, batch_loader=loader, epochs=2,
            destination=self.path / 'n1-final', epoch_checkpoint_directory=epochs)
        manifests = [json.loads((p / 'manifest.json').read_bytes()) for p in sorted(epochs.iterdir())]
        self.assertEqual([m['step'] for m in manifests], [1, 2])
        self.assertTrue(all(m['phase'] == 'n1' and m['acceptance_authority'] is False for m in manifests))
        self.assertEqual(manifests[-1]['model_state_sha256'], final['model_state_sha256'])
        self.assertEqual(len(construction.checkpoints), 1)
