"""Conditional-case independence preserves the fixed identity reference bank."""
from dataclasses import replace
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from src.alice_personality.identity import IdentityModel, IdentityModelConfig
from src.alice_personality.identity.contracts import SourceRecord
from src.alice_personality.teaching import IdentityTrainer, TeachingError, TrainingRecipe
from src.alice_personality.teaching.cases import CASE_LINEAGE_SCHEMA, NamedTarget, ReviewedTeachingCase
from src.alice_personality.teaching.contracts import canonical
from src.alice_personality.teaching.lineage import ConstructionLineage, ALLOCATION_SCHEMA, verify_allocation
from tests.personality.test_identity_teaching import _admission
from tests.personality import test_reviewed_teaching_cases as case_fixtures


def allocation(reference='anchor', episode='new-outcome', split='CALIBRATION_TRAIN'):
    return {'schema': ALLOCATION_SCHEMA, 'family_splits': {reference: 'TRAIN'},
            'reference_family_ids': [reference], 'supervision_family_splits': {episode: split},
            'supervision_parents': {episode: []}, 'qualification_scope': 'DEVELOPMENT_CONDITIONAL_CASES'}


class ConstructionLineageTests(unittest.TestCase):
    def setUp(self):
        self.records = (SourceRecord('e0', 'anchor', 'E0', 'E0', True, True, True, False, False, False),)
        self.lineage = ConstructionLineage(('anchor',), (), ('new-outcome',))

    def test_new_calibration_context_can_reference_train_identity_without_relabeling_it(self):
        value = allocation()
        verify_allocation(self.lineage, self.records, 'CALIBRATION_TRAIN', value)
        self.assertEqual(value['family_splits']['anchor'], 'TRAIN')

    def test_reusing_training_episode_or_its_derivative_cannot_create_a_holdout(self):
        for split in ('TRAIN', 'DEV'):
            value = allocation()
            value['supervision_family_splits']['new-outcome'] = split
            with self.assertRaisesRegex(TeachingError, 'supervision'):
                verify_allocation(self.lineage, self.records, 'CALIBRATION_TRAIN', value)
        value = allocation()
        value['supervision_family_splits']['original-teaching'] = 'TRAIN'
        value['supervision_parents']['original-teaching'] = []
        value['supervision_parents']['new-outcome'] = ['original-teaching']
        with self.assertRaisesRegex(TeachingError, 'ancestor|derivative'):
            verify_allocation(self.lineage, self.records, 'CALIBRATION_TRAIN', value)

    def test_dynamic_host_or_alice_experience_cannot_be_laundered_as_fixed_identity(self):
        host = replace(self.records[0], source_kind='HOST', provenance_class='HOST',
                       identity_core_allowed=False, historical_truth_allowed=False)
        with self.assertRaisesRegex(TeachingError, 'reference'):
            verify_allocation(self.lineage, (host,), 'CALIBRATION_TRAIN', allocation())

    def test_missing_parent_cycle_and_changed_reference_allocation_are_refused(self):
        mutations = []
        value = allocation(); value['supervision_parents']['new-outcome'] = ['missing']; mutations.append(value)
        value = allocation(); value['supervision_parents']['new-outcome'] = ['new-outcome']; mutations.append(value)
        value = allocation(); value['family_splits']['anchor'] = 'CALIBRATION_TRAIN'; mutations.append(value)
        value = allocation(); value['reference_family_ids'] = []; mutations.append(value)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(TeachingError):
                verify_allocation(self.lineage, self.records, 'CALIBRATION_TRAIN', value)

    def test_lineage_covers_all_original_source_families_and_roundtrips(self):
        self.assertEqual(ConstructionLineage.from_mapping(self.lineage.to_mapping()), self.lineage)
        for bad in (ConstructionLineage((), (), ('new-outcome',)),
                    ConstructionLineage(('anchor',), ('anchor',), ('new-outcome',)),
                    ConstructionLineage(('anchor',), (), ())):
            with self.assertRaises(TeachingError):
                verify_allocation(bad, self.records, 'CALIBRATION_TRAIN', allocation())

    def test_malformed_unhashable_names_are_refused_as_lineage_errors(self):
        bad = ConstructionLineage((['not-a-name'],), (), ('new-outcome',))
        with self.assertRaises(TeachingError):
            verify_allocation(bad, self.records, 'CALIBRATION_TRAIN', allocation())

    def test_new_dynamic_context_keeps_its_actual_episode_split(self):
        host = SourceRecord('host-event', 'runtime-event', 'HOST', 'HOST', False, True, False, False, False, False)
        lineage = ConstructionLineage(('anchor',), ('runtime-event',), ('new-outcome',))
        value = allocation(); value['family_splits']['runtime-event'] = 'CALIBRATION_TRAIN'
        verify_allocation(lineage, (*self.records, host), 'CALIBRATION_TRAIN', value)
        value['family_splits']['runtime-event'] = 'TRAIN'
        with self.assertRaisesRegex(TeachingError, 'context'):
            verify_allocation(lineage, (*self.records, host), 'CALIBRATION_TRAIN', value)


class ConstructionLineageIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = case_fixtures.ReviewedCaseTests()
        self.fixture.setUp()
    def test_case_materializer_and_actual_calibration_trainer_preserve_both_lineages(self):
        import torch
        fixture = self.fixture
        lineage = ConstructionLineage((fixture.family,), (), ('fresh-calibration-episode',))
        alloc = allocation(fixture.family, 'fresh-calibration-episode')
        case = replace(fixture.case(phase='calibration'), schema=CASE_LINEAGE_SCHEMA, lineage=lineage)
        restored = ReviewedTeachingCase.from_json(canonical(case.to_mapping()))
        self.assertEqual(case.sha256, restored.sha256)
        built = fixture.build(case, families=alloc)
        self.assertEqual(built.batch.lineage, (lineage,))
        with tempfile.TemporaryDirectory() as tmp:
            model = IdentityModel(fixture.config)
            recipe = TrainingRecipe('calibration', learning_rate=0.01, max_grad_norm=1, seed=10)
            admission = _admission(Path(tmp), model, recipe, [built.batch],
                reviewed_changes={'reviewed_case_artifact_sha256': {case.sha256: dict(case.artifact_sha256)}})
            before = {name: param.detach().clone() for name, param in model.named_parameters()}
            trainer = IdentityTrainer(model, recipe, admission, first_batch=built.batch)
            trainer.step(built.batch)
            changed = {name for name, param in model.named_parameters() if not torch.equal(before[name], param)}
            self.assertTrue(changed)
            self.assertTrue(all(name.startswith('n3.') for name in changed))

    def test_legacy_case_retains_original_bytes_and_strict_source_split(self):
        fixture = self.fixture
        case = fixture.case(phase='calibration')
        self.assertNotIn('lineage', case.to_mapping())
        self.assertEqual(case.sha256, ReviewedTeachingCase.from_json(canonical(case.to_mapping())).sha256)
        with self.assertRaisesRegex(TeachingError, 'split'):
            fixture.build(case, families={fixture.family: 'TRAIN'})

    def test_changed_lineage_cannot_reuse_old_materialization_receipt(self):
        fixture = self.fixture
        lineage = ConstructionLineage((fixture.family,), (), ('fresh-calibration-episode',))
        case = replace(fixture.case(phase='calibration'), schema=CASE_LINEAGE_SCHEMA, lineage=lineage)
        built = fixture.build(case, families=allocation(fixture.family, 'fresh-calibration-episode'))
        changed = replace(built.batch, lineage=(replace(lineage, supervision_family_ids=('different-outcome',)),))
        with tempfile.TemporaryDirectory() as tmp:
            model = IdentityModel(fixture.config)
            recipe = TrainingRecipe('calibration', learning_rate=0.01, max_grad_norm=1, seed=10)
            admission = _admission(Path(tmp), model, recipe, [changed],
                reviewed_changes={'reviewed_case_artifact_sha256': {case.sha256: dict(case.artifact_sha256)}})
            with self.assertRaisesRegex(TeachingError, 'materialization'):
                IdentityTrainer(model, recipe, admission, first_batch=changed)
