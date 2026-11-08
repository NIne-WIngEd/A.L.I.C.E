"""Public fictional scope mechanics; no teacher/model qualification evidence."""
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import replace
import importlib
import json
from pathlib import Path
import unittest

from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.identity.feature_store import FrozenFeatureStore
from src.alice_personality.teaching.cases import NamedTarget
from src.alice_personality.teaching.construction import ConnectedConstruction
from src.alice_personality.teaching.contracts import TeachingError, TrainingRecipe
from src.alice_personality.teaching.lineage import ALLOCATION_SCHEMA, validate_allocation
from tests.personality import test_connected_construction as construction_fixture
from tests.personality import test_reviewed_teaching_cases as case_fixture
from tests.personality.test_identity_codec import Backbone, Tokenizer
from tests.personality.test_identity_teaching import _admission

try:
    bindings = importlib.import_module('scripts.eipm.gemma_n0.cumulative_bindings')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.cumulative_bindings':
        raise
    bindings = None


def allocation(*, reference='reference-a', context='episode-a', supervision='review-a'):
    return {'schema': ALLOCATION_SCHEMA,
        'qualification_scope': 'DEVELOPMENT_CONDITIONAL_CASES',
        'family_splits': {reference: 'TRAIN', context: 'TRAIN'},
        'reference_family_ids': [reference],
        'supervision_family_splits': {supervision: 'TRAIN'},
        'supervision_parents': {supervision: []}}


class AllocationMergeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(bindings, 'The bounded cumulative binding helper is missing')

    def test_merge_preserves_all_references_and_detaches_nested_ancestry(self):
        first = allocation()
        second = allocation(reference='reference-b', context='episode-b', supervision='review-b')
        second['supervision_family_splits']['review-a'] = 'TRAIN'
        second['supervision_parents']['review-a'] = []
        second['supervision_parents']['review-b'] = ['review-a']
        originals = deepcopy([first, second])
        merged = bindings.merge_allocations([first, second])
        self.assertEqual(merged, {'schema': ALLOCATION_SCHEMA,
            'qualification_scope': 'DEVELOPMENT_CONDITIONAL_CASES',
            'family_splits': {'reference-a': 'TRAIN', 'episode-a': 'TRAIN',
                'reference-b': 'TRAIN', 'episode-b': 'TRAIN'},
            'reference_family_ids': ['reference-a', 'reference-b'],
            'supervision_family_splits': {'review-a': 'TRAIN', 'review-b': 'TRAIN'},
            'supervision_parents': {'review-a': [], 'review-b': ['review-a']}})
        validate_allocation(merged)
        merged['supervision_parents']['review-b'].append('changed-return-only')
        self.assertEqual([first, second], originals)

    def test_identical_repeated_allocations_are_idempotent(self):
        original = allocation()
        self.assertEqual(bindings.merge_allocations([original, deepcopy(original)]), original)

    def test_conflicting_duplicate_original_family_split_is_rejected(self):
        first, second = allocation(), allocation()
        second['family_splits']['episode-a'] = 'DEV'
        with self.assertRaises(TeachingError):
            bindings.merge_allocations([first, second])

    def test_conflicting_duplicate_supervision_split_is_rejected(self):
        first, second = allocation(), allocation()
        second['supervision_family_splits']['review-a'] = 'DEV'
        with self.assertRaises(TeachingError):
            bindings.merge_allocations([first, second])

    def test_conflicting_duplicate_ancestry_is_rejected(self):
        first, second = allocation(), allocation()
        for value in (first, second):
            value['supervision_family_splits']['review-parent'] = 'TRAIN'
            value['supervision_parents']['review-parent'] = []
        second['supervision_parents']['review-a'] = ['review-parent']
        with self.assertRaises(TeachingError):
            bindings.merge_allocations([first, second])

    def test_missing_ancestor_cannot_be_repaired_by_another_input(self):
        child = allocation(supervision='review-child')
        child['supervision_parents']['review-child'] = ['review-a']
        with self.assertRaises(TeachingError):
            bindings.merge_allocations([allocation(), child])

    def test_cycles_and_cross_split_derivatives_are_rejected(self):
        for split, parents in [('TRAIN', ['review-a']), ('DEV', [])]:
            with self.subTest(split=split):
                value = allocation()
                value['supervision_family_splits']['review-b'] = split
                value['supervision_parents']['review-b'] = parents
                value['supervision_parents']['review-a'] = ['review-b']
                with self.assertRaises(TeachingError):
                    bindings.merge_allocations([value])

    def test_malformed_or_empty_allocations_are_rejected(self):
        invalid = allocation()
        invalid['qualification_scope'] = 'QUALIFIED'
        for values in ([], (), [invalid], [dict(allocation(), extra=True)]):
            with self.subTest(values=values):
                with self.assertRaises(TeachingError):
                    bindings.merge_allocations(values)


class ParentScopeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if bindings is None:
            return
        cls.fixture = construction_fixture.ConstructionTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.original_cases, cls.original_batches, cls.parent_manifests = {}, {}, []
        construction = ConnectedConstruction(cls.fixture.model)
        for phase in ('n1', 'n2'):
            targets = (NamedTarget('concept_applicability', 'independent',
                (case_fixture.fact(cls.fixture.fixture.concept),)),) if phase == 'n1' else None
            case = replace(cls.fixture.fixture.case(phase=phase, targets=targets),
                case_id='public-original-' + phase)
            batch = cls.fixture.fixture.build(case, produced=cls.fixture.produced,
                session=cls.fixture.session).batch
            folder = cls.fixture.path / ('admission-' + phase)
            folder.mkdir()
            recipe = TrainingRecipe(phase, learning_rate=.01, max_grad_norm=1, seed=51)
            admission = _admission(folder, cls.fixture.model, recipe, [batch],
                reviewed_changes={'reviewed_case_artifact_sha256':
                    {case.sha256: dict(case.artifact_sha256)}})
            construction.run_phase(recipe, admission,
                batch_loader={batch.fingerprint(): batch}.__getitem__, epochs=2,
                destination=cls.fixture.path / ('parent-' + phase))
            cls.original_cases[case.case_id] = case
            cls.original_batches[case.case_id] = batch
            cls.parent_manifests.append(json.loads(
                (Path(construction.checkpoints[-1].path) / 'manifest.json').read_bytes()))

    def setUp(self):
        self.assertIsNotNone(bindings, 'The bounded cumulative binding helper is missing')
        self.cases = dict(self.original_cases)
        self.batches = dict(self.original_batches)
        self.manifests = deepcopy(self.parent_manifests)
        self.selection = {'retained_original_cases': [self.row('n1')],
            'superseded_original_annotations': [dict(self.row('n2'),
                replacement_packet='conditional', replacement_case_id='public-replacement-n2',
                replacement_case_sha256='f' * 64)],
            'unrelated_controller_metadata': {'qualification': 'UNQUALIFIED'}}

    def row(self, phase):
        case = self.original_cases['public-original-' + phase]
        return {'packet': 'original', 'case_id': case.case_id, 'case_sha256': case.sha256,
            'phase': phase, 'reason': 'Explicit public mechanical scope choice.'}

    def bind(self):
        return bindings.bind_parent_scope(self.selection, self.cases, self.batches, self.manifests)

    def test_selection_binds_actual_optimized_batches_and_keeps_case_hashes_separate(self):
        selection_before = deepcopy(self.selection)
        manifests_before = deepcopy(self.manifests)
        result = self.bind()
        n1, n2 = (self.batches['public-original-' + phase].fingerprint() for phase in ('n1', 'n2'))
        self.assertEqual(result, {'retained_parent_batches': (n1,),
            'superseded_parent_batches': {n2: 'Explicit public mechanical scope choice.'},
            'case_batch_sha256': {'public-original-n1': n1, 'public-original-n2': n2}})
        self.assertTrue(all(result['case_batch_sha256'][key] != case.sha256
            for key, case in self.cases.items()))
        self.assertEqual(self.selection, selection_before)
        self.assertEqual(self.manifests, manifests_before)

    def test_calibration_event_digest_never_enters_governing_scope(self):
        # Actual native parent evidence remains the positive control; the excluded
        # calibration envelope carries a foreign digest to catch scope leakage.
        self.manifests.append({'phase': 'calibration',
            'events': [{'phase': 'calibration', 'teaching_batch_sha256': 'f' * 64}]})
        self.assertEqual(self.bind()['retained_parent_batches'],
            (self.batches['public-original-n1'].fingerprint(),))

    def test_duplicate_or_overlapping_selection_rows_are_rejected(self):
        for group in ('retained_original_cases', 'superseded_original_annotations'):
            with self.subTest(group=group):
                original = deepcopy(self.selection)
                self.selection[group].append(deepcopy(self.selection['retained_original_cases'][0]))
                with self.assertRaises(TeachingError):
                    self.bind()
                self.selection = original

    def test_missing_original_case_selection_is_rejected(self):
        self.selection['superseded_original_annotations'] = []
        with self.assertRaises(TeachingError):
            self.bind()

    def test_wrong_packet_case_hash_or_phase_is_rejected(self):
        for field, value in [('packet', 'conditional'), ('case_sha256', 'a' * 64),
                ('phase', 'n2'), ('case_id', 'unknown-case')]:
            with self.subTest(field=field):
                original = deepcopy(self.selection)
                self.selection['retained_original_cases'][0][field] = value
                with self.assertRaises(TeachingError):
                    self.bind()
                self.selection = original

    def test_retained_and_superseded_reasons_must_be_nonempty(self):
        for group in ('retained_original_cases', 'superseded_original_annotations'):
            with self.subTest(group=group):
                original = deepcopy(self.selection)
                self.selection[group][0]['reason'] = '  '
                with self.assertRaises(TeachingError):
                    self.bind()
                self.selection = original

    def test_case_hash_cannot_substitute_for_actual_optimized_batch_digest(self):
        self.manifests[0]['events'][0]['teaching_batch_sha256'] = self.cases['public-original-n1'].sha256
        with self.assertRaises(TeachingError):
            self.bind()

    def test_wrong_event_phase_or_split_is_rejected(self):
        for field, value in [('phase', 'n2'), ('split', 'DEV')]:
            with self.subTest(field=field):
                original = deepcopy(self.manifests)
                self.manifests[0]['events'][0][field] = value
                with self.assertRaises(TeachingError):
                    self.bind()
                self.manifests = original

    def test_unoptimized_original_batch_or_foreign_digest_is_rejected(self):
        for events in ([], [{'phase': 'n1', 'split': 'TRAIN', 'teaching_batch_sha256': 'a' * 64}]):
            with self.subTest(events=events):
                original = deepcopy(self.manifests)
                self.manifests[0]['events'] = events
                with self.assertRaises(TeachingError):
                    self.bind()
                self.manifests = original

    def test_original_feature_session_is_required_for_the_optimized_digest(self):
        store = FrozenFeatureStore.create(self.fixture.path / 'different-session-store',
            source_package_sha256=case_fixture.PINS['source_package'])
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.fixture.config,
            _fixture=True, feature_store=store)
        case = self.cases['public-original-n1']
        produced = producer.produce(case.document)
        additional = replace(case.document, fields=tuple(
            replace(field, text=field.text + ' Additional public fictional session evidence.')
            if field.available else field for field in case.document.fields))
        producer.produce(additional)
        session = producer.close_and_verify()
        changed = self.fixture.fixture.build(case, produced=produced, session=session).batch
        self.assertNotEqual(session['receipt_sha256'], self.fixture.session['receipt_sha256'])
        self.assertEqual(changed.materialization_binding['case_sha256'], case.sha256)
        self.assertNotEqual(changed.fingerprint(), self.batches[case.case_id].fingerprint())
        self.batches[case.case_id] = changed
        with self.assertRaises(TeachingError):
            self.bind()

    def test_native_batch_case_phase_split_and_materialization_must_match(self):
        key = 'public-original-n1'
        old = self.batches[key]
        for changes in ({'case_ids': ('public-original-n2',)}, {'phase': 'n2'},
                {'split': 'DEV'}, {'materialization_binding': None},
                {'materialization_binding': dict(old.materialization_binding, case_sha256='a' * 64)}):
            with self.subTest(changes=tuple(changes)):
                self.batches[key] = replace(old, **changes)
                with self.assertRaises(TeachingError):
                    self.bind()
        self.batches[key] = old

    def test_batch_and_case_maps_must_account_for_the_same_unique_case_ids(self):
        for changed in ({}, dict(self.original_batches, unknown=self.original_batches['public-original-n1'])):
            with self.subTest(keys=tuple(changed)):
                self.batches = changed
                with self.assertRaises(TeachingError):
                    self.bind()

    def test_lazy_mapping_materializes_each_original_batch_once(self):
        fixture, cases, reads = self.fixture, self.cases, []
        class NativeReplayBatches(Mapping):
            def __iter__(self):
                return iter(cases)

            def __len__(self):
                return len(cases)

            def __getitem__(self, key):
                reads.append(key)
                case = cases[key]
                return fixture.fixture.build(case,
                    produced=fixture.replay.produce(case.document), session=fixture.session).batch

        self.batches = NativeReplayBatches()
        result = self.bind()
        self.assertEqual(result['retained_parent_batches'],
            (self.original_batches['public-original-n1'].fingerprint(),))
        self.assertEqual(reads, ['public-original-n1', 'public-original-n2'])


if __name__ == '__main__':
    unittest.main()
