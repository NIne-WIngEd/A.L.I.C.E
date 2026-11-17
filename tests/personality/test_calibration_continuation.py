"""Public fictional N3 continuation; no identity teaching or private inputs."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import json
from pathlib import Path
import unittest

import torch

from src.alice_personality.identity.codec import canonical, fingerprint
from src.alice_personality.identity.consumer import CheckpointPin
from src.alice_personality.teaching.cases import NamedTarget
from src.alice_personality.teaching.contracts import TeachingError, TrainingRecipe
from src.alice_personality.teaching.lineage import ConstructionLineage
from src.alice_personality.teaching.outcomes import calibration_case_from_review
from tests.personality import test_connected_construction as construction_fixture
from tests.personality import test_reviewed_teaching_cases as case_fixture
from tests.personality.test_construction_lineage import allocation
from tests.personality.test_identity_teaching import _admission

try:
    continuation_api = importlib.import_module('scripts.eipm.gemma_n0.calibrate_personality')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.calibrate_personality':
        raise
    continuation_api = None


class CalibrationContinuationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(continuation_api, 'Public calibration continuation interface is missing')
        self.fixture = construction_fixture.ConstructionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.original = construction_fixture.ConnectedConstruction(self.fixture.model)
        for phase in ('n1', 'n2'):
            recipe, admission, loader, _ = self.fixture.phase(phase)
            self.original.run_phase(recipe, admission, batch_loader=loader, epochs=2,
                                    destination=self.fixture.path / ('final-' + phase))
        self.pins = self.original.checkpoints
        self.response = self.original.consumer().respond(self.fixture.fixture.doc,
            self.fixture.replay.produce(self.fixture.fixture.doc), self.fixture.session)
        actual = self.response['selected_candidate_ids'][0]
        self.case = calibration_case_from_review(response_receipt=self.response,
            expected_response_sha256=self.response['receipt_sha256'], document=self.fixture.fixture.doc,
            case_id='fictional-actual-outcome', split='CALIBRATION_TRAIN',
            source_class='public_mechanical_fixture',
            lineage=ConstructionLineage((self.fixture.fixture.family,), (), ('fresh-outcome',)),
            targets=(NamedTarget('failure_tail_risk', 'independent',
                      (case_fixture.fact(actual, value=.8),)),),
            artifact_sha256=case_fixture.PINS, review_id='fictional-primary-review',
            annotation_note='Fictional explicit outcome, not source confidence or owner acceptance.')
        self.allocation = allocation(self.fixture.fixture.family, 'fresh-outcome')

    def load(self, pins=None, config=None):
        return continuation_api.load_n1_n2_checkpoint_chain(config or self.fixture.config,
            self.pins if pins is None else pins)

    def prepare(self, construction, **changes):
        args = dict(cases=(self.case,), expected_case_sha256={self.case.case_id: self.case.sha256},
            response_receipts={self.response['receipt_sha256']: self.response},
            artifact_sha256=case_fixture.PINS, family_allocation=self.allocation,
            feature_store=self.fixture.replay.feature_store, closed_session=self.fixture.session,
            expected_session_sha256=self.fixture.session['receipt_sha256'])
        args.update(changes)
        return continuation_api.prepare_calibration(construction, **args)

    def rebind_final_payload(self, mutate):
        pin = self.pins[-1]
        path = Path(pin.path)
        payload = torch.load(path / 'resume.pt', weights_only=True)
        mutate(payload['model'])
        torch.save(payload, path / 'resume.pt')
        manifest = json.loads((path / 'manifest.json').read_bytes())
        manifest['resume_sha256'] = sha256((path / 'resume.pt').read_bytes()).hexdigest()
        manifest['model_state_sha256'] = fingerprint(payload['model'])
        manifest.pop('manifest_sha256')
        manifest['manifest_sha256'] = sha256(canonical(manifest)).hexdigest()
        (path / 'manifest.json').write_bytes(canonical(manifest))
        return (self.pins[0], CheckpointPin(pin.path, manifest['manifest_sha256']))

    def test_reload_reconstructs_the_exact_actual_n1_n2_model_and_chain(self):
        loaded = self.load()
        self.assertEqual(fingerprint(loaded.model.state_dict()), self.response['model_state_sha256'])
        self.assertEqual(loaded.checkpoints, self.pins)
        reproduced = loaded.consumer().respond(self.case.document,
            self.fixture.replay.produce(self.case.document), self.fixture.session)
        self.assertEqual(reproduced['receipt_sha256'], self.response['receipt_sha256'])

    def test_reordered_missing_or_changed_checkpoint_payload_is_refused(self):
        for pins in (self.pins[::-1], self.pins[:1]):
            with self.subTest(pins=pins), self.assertRaises(TeachingError):
                self.load(pins)
        with (Path(self.pins[-1].path) / 'resume.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(TeachingError, 'payload'):
            self.load()

    def test_rehashed_nonfinite_model_cannot_become_a_calibration_initial_state(self):
        pins = self.rebind_final_payload(lambda state: next(iter(state.values())).fill_(float('nan')))
        with self.assertRaisesRegex(TeachingError, 'finite'):
            self.load(pins)

    def test_rehashed_incomplete_or_wrong_dtype_state_is_refused(self):
        for mutation in ('missing', 'dtype'):
            with self.subTest(mutation=mutation):
                path = Path(self.pins[-1].path)
                original_payload, original_manifest = (path / 'resume.pt').read_bytes(), (path / 'manifest.json').read_bytes()
                def mutate(state):
                    key = next(iter(state))
                    if mutation == 'missing':
                        del state[key]
                    else:
                        state[key] = state[key].double()
                pins = self.rebind_final_payload(mutate)
                with self.assertRaisesRegex(TeachingError, 'complete|geometry|dtype'):
                    self.load(pins)
                (path / 'resume.pt').write_bytes(original_payload)
                (path / 'manifest.json').write_bytes(original_manifest)

    def test_new_calibration_recipe_fits_only_n3_and_emits_full_connected_decisions(self):
        loaded = self.load()
        prepared = self.prepare(loaded)
        reviewed = prepared.reviewed_targets
        digest = reviewed['batch_order']['CALIBRATION_TRAIN'][0]
        first = prepared.batch_loader(digest)
        recipe = TrainingRecipe('calibration', learning_rate=.02, max_grad_norm=1, seed=119, weight_decay=0)
        directory = self.fixture.path / 'calibration-artifacts'
        directory.mkdir()
        admission = _admission(directory, loaded.model, recipe, [first], reviewed_changes=reviewed)
        before = {name: tensor.clone() for name, tensor in loaded.model.state_dict().items()}
        result = continuation_api.run_calibration(loaded, recipe, admission.artifacts,
            prepared=prepared, epochs=12, destination=self.fixture.path / 'n3-final')
        self.assertEqual(result['phase'], 'calibration')
        self.assertEqual(result['step'], 12)
        self.assertTrue(all(torch.equal(tensor, loaded.model.state_dict()[name])
                            for name, tensor in before.items() if name.startswith(('n1.', 'n2.'))))
        self.assertTrue(any(not torch.equal(tensor, loaded.model.state_dict()[name])
                            for name, tensor in before.items() if name.startswith('n3.')))
        rows = list(continuation_api.iter_connected_responses(loaded, prepared,
            documents={'fictional-new-context': ('UNLABELED', self.case.document)}))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(len(row['response']['checkpoint_manifest_sha256']), 3)
        self.assertFalse(row['response']['owner_fidelity_approved'])
        self.assertFalse(row['response']['spoken_fidelity_verified'])
        self.assertIn('failure_tail_risk', row['decision'])
        self.assertIn('owner_fidelity_uncertainty', row['decision'])
        self.assertEqual(canonical(row), canonical(json.loads(canonical(row))))
        selected = row['response']['selected_candidate_ids'][0]
        index = [entry.candidate_id for entry in self.case.document.candidates].index(selected)
        self.assertGreater(row['decision']['failure_tail_risk'][0][index], .6)

    def test_case_response_model_and_session_pins_are_required_before_materialization(self):
        loaded = self.load()
        changed_response = deepcopy(self.response)
        changed_response['response_text'] = 'unobserved words'
        changed_case = replace(self.case, annotation_note='Changed review')
        for changes in (
            {'expected_case_sha256': {self.case.case_id: '0' * 64}},
            {'cases': (changed_case,)},
            {'response_receipts': {self.response['receipt_sha256']: changed_response}},
            {'expected_session_sha256': '0' * 64},
        ):
            with self.subTest(changes=changes), self.assertRaises(TeachingError):
                self.prepare(loaded, **changes)
        wrong = deepcopy(self.response)
        wrong['model_state_sha256'] = '0' * 64
        wrong.pop('receipt_sha256')
        wrong['receipt_sha256'] = sha256(canonical(wrong)).hexdigest()
        case = calibration_case_from_review(response_receipt=wrong,
            expected_response_sha256=wrong['receipt_sha256'], document=self.case.document,
            case_id=self.case.case_id, split=self.case.split, source_class=self.case.source_class,
            lineage=self.case.lineage, targets=self.case.targets, artifact_sha256=self.case.artifact_sha256,
            review_id=self.case.review_id, annotation_note=self.case.annotation_note)
        with self.assertRaisesRegex(TeachingError, 'model|checkpoint'):
            self.prepare(loaded, cases=(case,), expected_case_sha256={case.case_id: case.sha256},
                         response_receipts={wrong['receipt_sha256']: wrong})

    def test_outcome_derivative_cannot_cross_the_disclosed_supervision_split(self):
        allocation_value = deepcopy(self.allocation)
        allocation_value['supervision_family_splits']['parent-training-episode'] = 'TRAIN'
        allocation_value['supervision_parents']['parent-training-episode'] = []
        allocation_value['supervision_parents']['fresh-outcome'] = ['parent-training-episode']
        with self.assertRaisesRegex(TeachingError, 'ancestor|derivative'):
            self.prepare(self.load(), family_allocation=allocation_value)

    def test_changed_initial_model_or_noncalibration_recipe_cannot_reuse_prepared_cases(self):
        loaded = self.load()
        prepared = self.prepare(loaded)
        digest = prepared.reviewed_targets['batch_order']['CALIBRATION_TRAIN'][0]
        recipe = TrainingRecipe('calibration', learning_rate=.01, max_grad_norm=1, seed=119)
        directory = self.fixture.path / 'calibration-artifacts'
        directory.mkdir()
        admission = _admission(directory, loaded.model, recipe, [prepared.batch_loader(digest)],
                               reviewed_changes=prepared.reviewed_targets)
        with self.assertRaisesRegex(TeachingError, 'calibration'):
            continuation_api.run_calibration(loaded, replace(recipe, phase='n2'), admission.artifacts,
                prepared=prepared, epochs=1, destination=self.fixture.path / 'wrong-phase')
        with torch.no_grad():
            next(loaded.model.n2.parameters()).add_(1)
        with self.assertRaisesRegex(TeachingError, 'initial|model|checkpoint'):
            continuation_api.run_calibration(loaded, recipe, admission.artifacts,
                prepared=prepared, epochs=1, destination=self.fixture.path / 'changed-initial')

