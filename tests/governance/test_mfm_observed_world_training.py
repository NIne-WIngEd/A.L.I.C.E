"""Observed-world trainer boundaries, separate from production learning.

Small reader fixtures prove ordering/refusal only. The opt-in integration
class below uses retained actual source/pair/corpus material without weights.
"""
from copy import deepcopy
from hashlib import sha256
import json
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from scripts.mfm import train_native_joint_formation as trainer


def case_ref(case_id):
    return {'case_id': case_id, 'profile': 'observed-world-firstparty',
        'variant_key': 'original', 'max_target_tokens': 8192,
        **{role: {'path': case_id + '/' + role + '.json', 'sha256': '1' * 64}
           for role in ('frozen', 'accepted', 'pair')}}


class ObservedTrainerBoundaryTests(unittest.TestCase):
    def preflight(self):
        function = getattr(trainer, 'preflight_observed_author_metadata', None)
        self.assertTrue(callable(function), 'observed source-first trainer metadata preflight is missing')
        return function

    def arguments(self, cases):
        return {'material_root': Path.cwd(), 'case_rows': cases, 'j5_source': object(),
            'admission_manifest_path': 'unopened-corpus.json',
            'admission_manifest_sha256': '2' * 64, 'owner_authorization_ref': 'fixture-authority',
            'pair_construction_ref': {'path':'unopened-pair-result.json','sha256':'3'*64}}

    def test_last_source_failure_precedes_every_author_and_corpus_reader(self):
        preflight = self.preflight()
        seen = []
        def source_reader(root, ref, *, j5_source):
            seen.append(ref['path'])
            if ref['path'].startswith('last/'):
                raise ValueError('actual last source refused')
            return {'split_row': {'case_id': 'first', 'split': 'train',
                'lineage': {'world_id': 'fixture-world'}}, 'variant_key': 'original'}, {}
        with patch.object(trainer, 'read_observed_source', side_effect=source_reader), \
             patch.object(trainer, '_material_json', side_effect=AssertionError('author read before all sources')), \
             patch.object(trainer, 'admit_formation_corpus', side_effect=AssertionError('corpus labels read before all sources')) as corpus:
            with self.assertRaisesRegex(ValueError, 'actual last source refused'):
                preflight(**self.arguments([case_ref('first'), case_ref('last')]))
            self.assertEqual(seen, ['first/frozen.json', 'last/frozen.json'])
            corpus.assert_not_called()

    def test_final_source_refuses_before_author_or_corpus_read(self):
        preflight = self.preflight()
        frozen = {'split_row': {'case_id': 'final', 'split': 'final',
            'lineage': {'world_id': 'fixture-world'}}, 'variant_key': 'original'}
        with patch.object(trainer, 'read_observed_source', return_value=(frozen, {})), \
             patch.object(trainer, '_material_json', side_effect=AssertionError('FINAL author read')), \
             patch.object(trainer, 'admit_formation_corpus', side_effect=AssertionError('FINAL corpus read')) as corpus:
            with self.assertRaisesRegex(ValueError, 'train'):
                preflight(**self.arguments([case_ref('final')]))
            corpus.assert_not_called()

    def test_duplicate_case_identity_refuses_before_any_source_or_label_read(self):
        preflight = self.preflight()
        with patch.object(trainer, 'read_observed_source', side_effect=AssertionError('duplicate reached source')) as source, \
             patch.object(trainer, 'admit_formation_corpus', side_effect=AssertionError('duplicate reached labels')) as corpus:
            with self.assertRaisesRegex(ValueError, 'unique'):
                preflight(**self.arguments([case_ref('same'), case_ref('same')]))
            source.assert_not_called(); corpus.assert_not_called()

    def test_historical_fit_program_refuses_observed_data_before_heavy_or_corpus_readers(self):
        from transformers import AutoProcessor
        from scripts.mfm import export_joint_source_features as features
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('material', 'features', 'prepared'):
                (root / name).mkdir()
            receipt = root / 'prepared/receipt.json'; receipt.write_bytes(b'{}')
            program = {'schema': trainer.PROGRAM_SCHEMA, 'cases': [case_ref('observed-case')]}
            path = root / 'program.json'; raw = canonical_json_bytes(program); path.write_bytes(raw)
            args = {'program_path': path, 'program_file_sha256': sha256(raw).hexdigest(),
                'material_root': root / 'material', 'admission_manifest_path': root / 'absent-corpus.json',
                'admission_manifest_sha256': '1' * 64, 'owner_authorization_ref': 'fixture-authority',
                'feature_root': root / 'features', 'feature_roster_sha256': '2' * 64,
                'prepared_base_dir': root / 'prepared', 'prepared_receipt_path': receipt,
                'prepared_receipt_file_sha256': '3' * 64, 'prepared_receipt_sha256': '4' * 64,
                'parent_source_receipt_sha256': '5' * 64, 'seed_weight_path': root / 'absent-weights',
                'seed_config_path': root / 'absent-config', 'seed_manifest_path': root / 'absent-manifest',
                'seed_weight_sha256': '6' * 64, 'seed_config_sha256': '7' * 64,
                'seed_manifest_sha256': '8' * 64, 'seed_provenance_sha256': '9' * 64,
                'seed_source_pins': {}, 'recipe_path': root / 'absent-recipe',
                'recipe_file_sha256': 'a' * 64, 'output_dir': root / 'output'}
            with patch.object(AutoProcessor, 'from_pretrained') as processor, \
                 patch.object(features, 'read_case') as feature, \
                 patch.object(FormationJointLearning, 'load_component') as model, \
                 patch.object(trainer, 'admit_formation_corpus') as corpus:
                with self.assertRaisesRegex(ValueError, 'observed.*data.phase'):
                    trainer.fit_joint_from_program(**args)
                processor.assert_not_called(); feature.assert_not_called()
                model.assert_not_called(); corpus.assert_not_called()


ROOT = Path(os.environ.get('ALICE_MFM_OBSERVED_ORIGINAL_ROOT', str(Path(__file__).resolve().parents[4])))
CORPUS = ROOT / 'work/mfm-rill-corpus-assembly-20261022/actual-001'
NATIVE = ROOT / 'work/mfm-rill-target-authoring-20261022/native-geometry-001/actual-001'
INTEGRATION = os.environ.get('ALICE_MFM_OBSERVED_ORIGINAL_INTEGRATION') == '1'
ACTUAL_SUMMARY = {}


def actual_j5_module():
    path = ROOT / ('work/mfm-joint-original-inventory-roundtrip-design-20261011/'
        'material-read-export-construction-001/dependency-tree-001/scripts/mfm/formation_joint_input_v5.py')
    name = 'scripts.mfm.formation_joint_input_v5'
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise ValueError('retained actual J5 module origin differs')
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(INTEGRATION, 'set ALICE_MFM_OBSERVED_ORIGINAL_INTEGRATION=1 for actual corpus/metadata')
class ObservedTrainerActualMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt_raw = (CORPUS / 'receipt.json').read_bytes()
        if sha256(receipt_raw).hexdigest() != '799b6b5ff1106c04a648ecf4b5f311d849e08059f0827e17e03ed37b998c75a2':
            raise ValueError('completed actual corpus receipt FILE differs')
        receipt = json.loads(receipt_raw)
        cls.manifest_raw = (CORPUS / 'corpus.json').read_bytes()
        cls.manifest = json.loads(cls.manifest_raw)
        cls.manifest_pin = sha256(cls.manifest_raw).hexdigest()
        if cls.manifest_pin != receipt['corpus']['sha256'] or receipt['train_cases'] != 9:
            raise ValueError('actual complete corpus differs')
        cls.authority = cls.manifest['authorization_id']
        handoff = json.loads((CORPUS / 'material-handoff.json').read_bytes())
        cls.handoff = handoff['cases']
        cls.result_ref = {'path': handoff['original_pair_root'] + '/result.json',
            'sha256': receipt['pair_receipt_file_sha256']}
        cls.case_rows = [{key: row[key] for key in ('case_id','profile','variant_key','accepted','pair','max_target_tokens')}
            | {'frozen': row['source_input']} for row in cls.handoff]
        cls.original_files = [path for path in CORPUS.rglob('*') if path.is_file()]
        cls.original_pins = {str(path): sha256(path.read_bytes()).hexdigest() for path in cls.original_files}
        cls.source_order = []
        cls.label_reads = []
        real_source, real_json, real_admit = trainer.read_observed_source, trainer._material_json, trainer.admit_formation_corpus
        author_paths = {row[key]['path'] for row in cls.case_rows for key in ('accepted','pair')}
        def read_source(*args, **kwargs):
            result = real_source(*args, **kwargs)
            cls.source_order.append(result[0]['split_row']['case_id'])
            return result
        def read_json(root, ref):
            if root == ROOT.resolve() and ref['path'] in author_paths:
                if len(cls.source_order) != 9:
                    raise AssertionError('actual author read before all nine physical sources')
                cls.label_reads.append(ref['path'])
            return real_json(root, ref)
        def admit(*args, **kwargs):
            if len(cls.source_order) != 9:
                raise AssertionError('actual corpus labels read before all nine physical sources')
            return real_admit(*args, **kwargs)
        cls.bridge_only = os.environ.get('ALICE_MFM_OBSERVED_BRIDGE_ONLY') == '1'
        if cls.bridge_only:
            # Reuse the completed original source/geometry evidence; exercise the
            # corrected actual bridge directly, without mocking any production reader.
            evidence_path = CORPUS.parent/'trainer-controls-001/actual-metadata-red-summary-001.json'
            evidence_raw = evidence_path.read_bytes(); evidence = json.loads(evidence_raw)
            if (evidence['corpus_file_sha256'] != cls.manifest_pin or
                evidence['complete_actual_metadata_cases'] != 9 or
                evidence['saved_native_choice_view_cases'] != 9 or
                evidence['all_sources_preceded_every_author_and_corpus'] is not True):
                raise ValueError('prior complete actual source/choice evidence required')
            cls.cases=[]
            native=json.loads((NATIVE/'receipt.json').read_bytes())
            native_files={row['path']:row for row in native['files']}
            for row, handoff_row in zip(cls.case_rows,cls.handoff,strict=True):
                frozen=real_json(ROOT,row['frozen'])[1];accepted=real_json(ROOT,row['accepted'])[1]
                pair=real_json(ROOT,row['pair'])[1]
                ref=native_files[handoff_row['cut_id']+'/choice-view.json']
                raw=(NATIVE/ref['path']).read_bytes()
                if sha256(raw).hexdigest()!=ref['file_sha256']:raise ValueError('saved choice FILE differs')
                view=json.loads(raw);joint=frozen['source']['alignment']
                lowered=trainer._observed_lowered_metadata(frozen,accepted,pair,
                    trainer._observed_source_descriptor(ROOT,frozen,actual_j5_module()),view,joint,
                    frozen['split_row']['lineage']['world_id'],trainer._observed_source_family(ROOT,frozen))
                cls.cases.append({'frozen':frozen,'accepted':accepted,'pair':pair,'joined':{'lowered':lowered}})
            ACTUAL_SUMMARY.update(prior_actual_source_choice_evidence_file_sha256=sha256(evidence_raw).hexdigest(),
                source_and_native_geometry_reexecuted_this_run=False)
        else:
            with patch.object(trainer, 'read_observed_source', side_effect=read_source), \
                 patch.object(trainer, '_material_json', side_effect=read_json), \
                 patch.object(trainer, 'admit_formation_corpus', side_effect=admit):
                cls.cases = trainer.preflight_observed_author_metadata(material_root=ROOT,
                    case_rows=cls.case_rows, j5_source=actual_j5_module(),
                    admission_manifest_path=CORPUS/'corpus.json', admission_manifest_sha256=cls.manifest_pin,
                    owner_authorization_ref=cls.authority,pair_construction_ref=cls.result_ref)
        ACTUAL_SUMMARY.update(corpus_file_sha256=cls.manifest_pin,
            corpus_receipt_file_sha256=sha256(receipt_raw).hexdigest(),
            actual_source_order=cls.source_order, author_file_reads=len(cls.label_reads),
            complete_actual_metadata_cases=len(cls.cases),
            source_first_full_preflight_performed_this_run=not cls.bridge_only,
            processor_executed=False, features_joined=False, model_or_optimizer_executed=False)

    @classmethod
    def tearDownClass(cls):
        if {str(path): sha256(path.read_bytes()).hexdigest() for path in cls.original_files} != cls.original_pins:
            raise AssertionError('actual corpus originals changed during controls')
        ACTUAL_SUMMARY['actual_corpus_originals_preserved'] = True

    def bridge(self, root, cases=None, manifest=None):
        path = root / 'corpus.json'
        if manifest is not None:
            path.write_bytes(canonical_json_bytes(manifest) + b'\n')
        return trainer.preflight_admitted_author_cases(admission_manifest_path=path,
            admission_manifest_sha256=sha256(path.read_bytes()).hexdigest(),
            owner_authorization_ref=self.authority, cases=self.cases if cases is None else cases,
            observed_material_root=ROOT,observed_pair_construction_ref=self.result_ref)

    def test_all_nine_final_corpus_author_bridge(self):
        admission,audit=self.bridge(CORPUS)
        self.assertEqual(len(admission.train),9)
        ACTUAL_SUMMARY.update(final_actual_corpus_author_bridge_cases=9,
            source_identity_family_and_converter_provenance_joined=True)

    def with_corpus_copy(self, mutate, message):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'corpus'
            shutil.copytree(CORPUS, root, copy_function=shutil.copyfile)
            manifest = deepcopy(self.manifest)
            mutate(root, manifest)
            with self.assertRaisesRegex(ValueError, message):
                self.bridge(root, manifest=manifest)

    def test_all_nine_metadata_and_saved_choice_views_match_complete_originals(self):
        if self.bridge_only:self.skipTest('full original source/choice evidence already retained; final run tests corrected bridge')
        from scripts.mfm import formation_observed_world_paired_teaching as observed
        native = json.loads((NATIVE / 'receipt.json').read_bytes())
        files = {row['path']: row for row in native['files']}
        counts = [0,0,0]
        for row, handoff in zip(self.cases, self.handoff, strict=True):
            frozen, accepted, pair = row['frozen'], row['accepted'], row['pair']
            example, _ = observed.require_complete_labels(frozen, accepted)
            _, view, joint, world = trainer._full_source_view(frozen, example, pair)
            ref = files[handoff['cut_id'] + '/choice-view.json']
            raw = (NATIVE / ref['path']).read_bytes()
            self.assertEqual((len(raw),sha256(raw).hexdigest()),(ref['bytes'],ref['file_sha256']))
            golden = json.loads(raw)
            for key in ('decision','binding','copy_binding','processor_binding','reference_leaves',
                        'authored_ordinals','copy_leaves','authored_copy_tuples'):
                self.assertEqual(view[key],golden[key],key)
            lowered = row['joined']['lowered']
            self.assertNotIn('material15_file_sha256', lowered)
            self.assertEqual(lowered['mass_world_id'],world)
            self.assertEqual(lowered['source_descriptor'], trainer._observed_source_descriptor(ROOT,frozen,actual_j5_module()))
            self.assertEqual(lowered['source_descriptor']['profile'],'j6-source-only')
            standalone = json.loads((CORPUS / handoff['derived_standalone_target']['path']).read_bytes())
            original = json.loads(bytes.fromhex(accepted['original_material_hex']['target']))
            self.assertEqual(len(original),4); self.assertEqual(len(standalone),6)
            self.assertNotEqual(handoff['derived_standalone_target']['sha256'],accepted['original_target_file_sha256'])
            counts[0] += len(view['decision']['proposals'])
            counts[1] += len(view['reference_leaves']); counts[2] += len(view['copy_leaves'])
        self.assertEqual(counts,[41,329,9])
        self.assertEqual(self.source_order,[row['case_id'] for row in self.case_rows])
        self.assertEqual(len(self.label_reads),18)
        ACTUAL_SUMMARY.update(saved_native_choice_view_cases=9,proposals=counts[0],reference_choices=counts[1],literal_segments=counts[2],
            original_four_field_and_derived_six_field_files_distinct=True)

    def test_original_author_file_cannot_be_replaced_by_standalone_file_identity(self):
        cases = list(self.cases)
        first = self.cases[0]
        cases[0] = {**first,'joined':{'lowered':{**first['joined']['lowered'],
            'original_target_file_sha256': self.manifest['cases'][0]['target']['sha256']}}}
        with self.assertRaisesRegex(ValueError,'original lowerer'):
            self.bridge(CORPUS,cases=cases)

    def test_lowered_reference_choices_must_match_complete_pair(self):
        cases = list(self.cases); first = self.cases[0]
        lowered = deepcopy(first['joined']['lowered']); lowered['authored_ordinals'] = []
        cases[0] = {**first,'joined':{'lowered':lowered}}
        with self.assertRaisesRegex(ValueError,'complete pair/reference/literal'):
            self.bridge(CORPUS,cases=cases)

    def test_standalone_body_must_be_exact_six_field_conversion(self):
        original_raw = bytes.fromhex(self.cases[0]['accepted']['original_material_hex']['target'])
        def mutate(root, manifest):
            ref = manifest['cases'][0]['target']
            (root/ref['path']).write_bytes(original_raw)
            ref['sha256'] = sha256(original_raw).hexdigest()
        self.with_corpus_copy(mutate,'derived standalone target')

    def test_conversion_must_bind_the_original_author_manifest_file(self):
        def mutate(root, manifest):
            ref = manifest['cases'][0]['target_provenance']['conversion']
            path = root/ref['path']; record = json.loads(path.read_bytes())
            record['author_manifest']['file_sha256'] = '0'*64
            record['record_sha256'] = canonical_sha256({k:v for k,v in record.items() if k!='record_sha256'})
            raw = canonical_json_bytes(record)+b'\n'; path.write_bytes(raw);ref['sha256']=sha256(raw).hexdigest()
        self.with_corpus_copy(mutate,'conversion|author and derived standalone FILE')

    def test_revoked_source_use_refuses_even_when_all_rights_file_pins_are_updated(self):
        def mutate(root, manifest):
            source = manifest['cases'][0]['sources'][0]
            path = root/source['rights_path']; rights=json.loads(path.read_bytes());rights['revoked']=True
            raw=canonical_json_bytes(rights)+b'\n';path.write_bytes(raw)
            for row in manifest['cases']:
                for item in row['sources']:
                    if item['rights_path']==source['rights_path']:item['rights_sha256']=sha256(raw).hexdigest()
        self.with_corpus_copy(mutate,'revoked')

    def test_coherent_manifest_and_rights_source_id_relabel_cannot_cross_original_caller(self):
        def mutate(root, manifest):
            source = manifest['cases'][0]['sources'][0]
            old_id, new_id = source['source_id'], source['source_id']+'-relabelled-control'
            path=root/source['rights_path']; rights=json.loads(path.read_bytes());rights['source_id']=new_id
            raw=canonical_json_bytes(rights)+b'\n';path.write_bytes(raw)
            for row in manifest['cases']:
                for item in row['sources']:
                    if item['source_id']==old_id:
                        item['source_id']=new_id;item['rights_sha256']=sha256(raw).hexdigest()
            # Source payload and canonical target bytes are intentionally unchanged.
            # Corpus-local source/rights IDs agree; the real caller identity must still win.
        self.with_corpus_copy(mutate,'source identit')

    def test_source_family_authorization_cannot_relabel_pinned_composition(self):
        def mutate(root, manifest):
            old_family = manifest['cases'][0]['source_family']
            for row in manifest['cases']:
                if row['split']=='train':row['source_family']='coherently-relabelled-source-family-control'
            for row in manifest['authorizations']:
                if row['source_family']==old_family:
                    row['source_family']='coherently-relabelled-source-family-control'
        self.with_corpus_copy(mutate,'source family')

    def test_resealed_conversion_cannot_claim_an_unbound_producer_file(self):
        def mutate(root, manifest):
            ref=manifest['cases'][0]['target_provenance']['conversion']
            path=root/ref['path']; record=json.loads(path.read_bytes())
            record['conversion_producer_file_sha256']='0'*64
            record['record_sha256']=canonical_sha256({k:v for k,v in record.items() if k!='record_sha256'})
            raw=canonical_json_bytes(record)+b'\n';path.write_bytes(raw);ref['sha256']=sha256(raw).hexdigest()
        self.with_corpus_copy(mutate,'conversion|producer')


if __name__ == '__main__':
    unittest.main()
