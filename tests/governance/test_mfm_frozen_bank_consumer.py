"""Metadata custody mechanics; assembled bank bindings are not produced features."""
from copy import deepcopy
import ast
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from scripts.mfm import train_native_joint_formation as trainer
from scripts.mfm import train_public_conditional_foundation as public
from scripts.mfm import train_v1_formation_specialist as shared


PROFILE = 'frozen-bank-tokenizer-metadata-v1'
VERIFIER_FILE = '7848c5f2b12db7d08fc1ff41df5680d9b7d25b4101e6dad54e6eb4c558e0c7d5'


def binding_fixture(receipt, receipt_file_sha256):
    """Assemble metadata join operands only; no producer bank is fabricated."""
    files = {row['path']: row for row in receipt['files']}
    return {'base': {
        'weight_file_sha256': files['model.safetensors']['sha256'],
        'config_file_sha256': files['config.json']['sha256'],
        'hidden_width': 3840, 'fixture_small_width': False,
        'representation': 'base.model.last_hidden_state',
        'backbone_device': 'cpu',
        'runtime': {'torch_version': '2.7.1+cu118', 'transformers_version': '5.17.0'},
        'prepared_base': {
            'receipt_sha256': receipt['receipt_sha256'],
            'receipt_file_sha256': receipt_file_sha256,
            'parent_source_receipt_sha256': receipt['parent_source_receipt_sha256'],
            'source_weight_sha256': shared.SOURCE_WEIGHT_SHA256,
            'foundation_commit': shared.FOUNDATION_COMMIT,
            'foundation_verifier_sha256': shared.FOUNDATION_VERIFIER_SHA256,
            'foundation_inventory_sha256': shared.FOUNDATION_INVENTORY_SHA256,
            'prepared_files': receipt['files'], 'prepared_schema': receipt['schema']}}}


def recipe_fixture(receipt, *, profile=PROFILE):
    assets = {row['path']: row for row in receipt['files']}
    record = {'schema': 'mfm-native-joint-fit-recipe-v2' if profile else trainer.RECIPE_SCHEMA,
        'program_sha256': '1'*64, 'admission_sha256': '2'*64,
        'feature_roster_sha256': '3'*64, 'seed_manifest_sha256': '4'*64,
        'case_order': ['fixture-case'], 'update_budget': 1, 'accumulation_cases': 1,
        'learning_rate': 0.001, 'weight_decay': 0.0, 'adam_betas': [0.9, 0.999],
        'adam_eps': 1e-8, 'device': 'cpu', 'seed': 7, 'checkpoint_every_updates': 1,
        'precision': 'owned-fp32-source-bf16',
        'term_weights': {'lexical': 1, 'reference': 1, 'copy': 1},
        'mass_policy': 'equal-world-equal-cut-equal-variant',
        'update_policy': 'frozen-order-case-window-admitted-mass',
        'runtime': {'torch_version': torch.__version__, 'transformers_version': version('transformers')},
        'trainer_file_sha256': sha256(Path(trainer.__file__).read_bytes()).hexdigest(),
        'generation_max_new_tokens': 8}
    if profile:
        record['prepared_consumer'] = {'profile': profile,
            'metadata_verifier_file_sha256': VERIFIER_FILE,
            'asset_roster_sha256': canonical_sha256([assets[name] for name in public.METADATA_NAMES])}
    return record


class FrozenBankMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = self.root/'processor'
        self.metadata.mkdir()
        files = []
        for name in public.METADATA_NAMES:
            data = canonical_json_bytes({'binding-mechanics-fixture': name})
            (self.metadata/name).write_bytes(data)
            files.append({'path': name, 'size': len(data), 'sha256': sha256(data).hexdigest()})
        files.append({'path': 'model.safetensors', 'size': 23919549408,
                      'sha256': shared.SOURCE_WEIGHT_SHA256})
        self.receipt = {'schema': shared.CLONE_SCHEMA, 'role': 'mfm',
            'snapshot_path': '/historical-producer/fixture-clone',
            'parent_source_receipt_sha256': 'a'*64, 'qualification': 'unqualified',
            'files': files}
        self.receipt['receipt_sha256'] = canonical_sha256(self.receipt)
        self.receipt_path = self.root/'clone.json'
        self.receipt_path.write_bytes(canonical_json_bytes(self.receipt))
        self.receipt_file = sha256(self.receipt_path.read_bytes()).hexdigest()
        self.roster = binding_fixture(self.receipt, self.receipt_file)
        self.recipe = recipe_fixture(self.receipt)

    def verify(self, *, previous=None, previous_metadata=None, roster=None,
               recipe=None, profile=PROFILE, receipt_file=None):
        return trainer._verified_fit_prepared(previous, self.receipt_path, self.metadata,
            receipt_file or self.receipt_file, self.receipt['receipt_sha256'],
            self.receipt['parent_source_receipt_sha256'],
            roster=self.roster if roster is None else roster,
            recipe=self.recipe if recipe is None else recipe,
            prepared_consumer_profile=profile, prior_metadata=previous_metadata)

    def test_explicit_metadata_role_authenticates_subset_without_weight_or_live_producer_runtime(self):
        # Catch an accidental full-clone call, local weight open, or live CPU/Torch requirement.
        from scripts.mfm import export_joint_source_features as features
        original_read = Path.open
        def reject_weight(path, *args, **kwargs):
            if path.name == 'model.safetensors':
                self.fail('metadata consumer attempted to open base weights')
            return original_read(path, *args, **kwargs)
        with patch.object(Path, 'open', reject_weight), \
             patch.object(features, '_verified_prepared', side_effect=AssertionError('complete clone forbidden')), \
             patch.object(torch, '__version__', '2.11.0+cu128'):
            consumer_recipe = deepcopy(self.recipe)
            consumer_recipe['runtime']['torch_version'] = '2.11.0+cu128'
            consumer_recipe['device'] = 'cuda:0'
            trainer._recipe(consumer_recipe, program_sha256='1'*64, admission_sha256='2'*64,
                feature_roster_sha256='3'*64, seed_manifest_sha256='4'*64,
                prepared_consumer_profile=PROFILE)
            row, metadata = self.verify(recipe=consumer_recipe)
        self.assertEqual(row, self.receipt)
        self.assertEqual(metadata['profile'], PROFILE)
        self.assertEqual(metadata['asset_roster_sha256'], self.recipe['prepared_consumer']['asset_roster_sha256'])
        self.assertFalse(metadata['base_weights_opened'])
        self.assertFalse(metadata['complete_clone_weight_verification_performed_here'])
        self.assertFalse((self.metadata/'model.safetensors').exists())

    def test_recipe_profile_is_explicit_strict_and_pinned_for_resume(self):
        kwargs = {'program_sha256': '1'*64, 'admission_sha256': '2'*64,
                  'feature_roster_sha256': '3'*64, 'seed_manifest_sha256': '4'*64}
        trainer._recipe(self.recipe, **kwargs, prepared_consumer_profile=PROFILE)
        legacy = recipe_fixture(self.receipt, profile=None)
        trainer._recipe(legacy, **kwargs)
        for altered, profile in ((self.recipe, None), (legacy, PROFILE),
                                 (self.recipe, 'unknown-profile')):
            with self.subTest(profile=profile, schema=altered['schema']), self.assertRaises(ValueError):
                trainer._recipe(altered, **kwargs, prepared_consumer_profile=profile)
        for field in ('profile', 'metadata_verifier_file_sha256', 'asset_roster_sha256'):
            changed = deepcopy(self.recipe)
            changed['prepared_consumer'][field] = 'bad'
            with self.subTest(field=field), self.assertRaises(ValueError):
                trainer._recipe(changed, **kwargs, prepared_consumer_profile=PROFILE)

    def test_wrong_bank_clone_and_producer_ancestry_refuse(self):
        # Every variant must be stopped while all inputs are still metadata.
        for location, field, value in (
            ('base', 'weight_file_sha256', 'b'*64),
            ('base', 'config_file_sha256', 'b'*64),
            ('base', 'hidden_width', 4),
            ('base', 'fixture_small_width', True),
            ('prepared', 'receipt_file_sha256', 'b'*64),
            ('prepared', 'receipt_sha256', 'b'*64),
            ('prepared', 'parent_source_receipt_sha256', 'b'*64),
            ('prepared', 'source_weight_sha256', 'b'*64),
            ('prepared', 'foundation_commit', 'b'*40),
            ('prepared', 'foundation_verifier_sha256', 'b'*64),
            ('prepared', 'foundation_inventory_sha256', 'b'*64),
            ('prepared', 'prepared_schema', 'wrong')):
            changed = deepcopy(self.roster)
            target = changed['base'] if location == 'base' else changed['base']['prepared_base']
            target[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(roster=changed)
        changed = deepcopy(self.roster)
        changed['base']['prepared_base']['prepared_files'][0]['size'] += 1
        with self.assertRaises(ValueError):
            self.verify(roster=changed)
        with self.assertRaises(ValueError):
            self.verify(receipt_file='b'*64)

    def test_changed_or_extra_asset_and_resealed_foreign_receipt_refuse(self):
        # A good FILE hash alone cannot substitute another receipt/asset role.
        for extra in ('added_tokens.json', 'custom.py'):
            (self.metadata/extra).write_bytes(b'{}')
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.verify()
            (self.metadata/extra).unlink()
        asset = self.metadata/'tokenizer.json'
        original = asset.read_bytes()
        asset.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.verify()
        asset.write_bytes(original)
        asset.rename(self.metadata/'missing-tokenizer.json')
        with self.assertRaises(ValueError):
            self.verify()
        (self.metadata/'missing-tokenizer.json').rename(asset)
        changed = deepcopy(self.receipt)
        changed['role'] = 'host'
        changed['receipt_sha256'] = canonical_sha256({k:v for k,v in changed.items() if k != 'receipt_sha256'})
        self.receipt = changed
        self.receipt_path.write_bytes(canonical_json_bytes(changed))
        self.roster = binding_fixture(changed, sha256(self.receipt_path.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            self.verify(receipt_file=sha256(self.receipt_path.read_bytes()).hexdigest())

    def test_brackets_reject_midrun_asset_or_custody_mutation(self):
        first, metadata = self.verify()
        second, repeated = self.verify(previous=first, previous_metadata=metadata)
        self.assertEqual(repeated, metadata)
        asset = self.metadata/'config.json'
        original = asset.read_bytes()
        asset.write_bytes(b'changed during fit')
        with self.assertRaises(ValueError):
            self.verify(previous=second, previous_metadata=repeated)
        asset.write_bytes(original)
        changed = deepcopy(metadata)
        changed['asset_roster_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            self.verify(previous=first, previous_metadata=changed)

    def test_asset_link_refuses(self):
        target = self.root/'original-config.json'
        asset = self.metadata/'config.json'
        asset.rename(target)
        try:
            asset.symlink_to(target)
        except OSError as error:
            self.skipTest(f'host does not permit physical symlink control: {error}')
        with self.assertRaises(ValueError):
            self.verify()

    @unittest.skipUnless(os.name == 'nt', 'physical Windows junction control')
    def test_processor_directory_junction_refuses(self):
        import _winapi
        target = self.root/'actual-processor'
        self.assertTrue(self.metadata.resolve().is_relative_to(self.root.resolve()))
        self.assertTrue(target.resolve().is_relative_to(self.root.resolve()))
        self.metadata.rename(target)
        _winapi.CreateJunction(str(target), str(self.metadata))
        try:
            with self.assertRaises(ValueError):
                self.verify()
        finally:
            self.metadata.rmdir()

    def test_default_does_not_fall_back_when_complete_clone_custody_refuses(self):
        from scripts.mfm import export_joint_source_features as features
        # Complete base custody is expensive/external; retain its refusal at this boundary.
        with patch.object(features, '_verified_prepared', side_effect=ValueError('full clone refused')):
            with self.assertRaisesRegex(ValueError, 'full clone refused'):
                self.verify(profile=None, recipe=recipe_fixture(self.receipt, profile=None))

    def test_actual_fit_entry_refuses_bank_mismatch_before_heavy_calls(self):
        from scripts.mfm import export_joint_source_features as features
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        from transformers import AutoProcessor
        material = self.root/'material'
        bank = self.root/'binding-fixture'
        material.mkdir()
        bank.mkdir()
        program = {'schema': trainer.PROGRAM_SCHEMA, 'cases': [{
            'case_id': 'fixture-case', 'profile': 'j4-p1', 'variant_key': 'fixture-variant',
            'frozen': {'path': 'missing-frozen.json', 'sha256': '5'*64},
            'accepted': {'path': 'missing-accepted.json', 'sha256': '6'*64},
            'pair': {'path': 'missing-pair.json', 'sha256': '7'*64},
            'max_target_tokens': 8}]}
        program_path = self.root/'program.json'
        program_path.write_bytes(canonical_json_bytes(program))
        roster = deepcopy(self.roster)
        roster.update(schema=features.SCHEMA, cases=[])
        roster['base']['weight_file_sha256'] = 'b'*64
        roster['record_sha256'] = canonical_sha256(roster)
        (bank/'roster.json').write_bytes(canonical_json_bytes(roster))
        recipe = deepcopy(self.recipe)
        recipe['program_sha256'] = sha256(program_path.read_bytes()).hexdigest()
        recipe['feature_roster_sha256'] = roster['record_sha256']
        recipe_path = self.root/'recipe.json'
        recipe_path.write_bytes(canonical_json_bytes(recipe))
        output = self.root/'forbidden-output'
        # Only corpus authority is replaced: this test isolates the real custody entry boundary.
        with patch.object(trainer, 'admit_formation_corpus', return_value=None), \
             patch.object(AutoProcessor, 'from_pretrained') as processor_load, \
             patch.object(features, 'read_case') as tensor_load, \
             patch.object(FormationJointLearning, 'load_component') as seed_load:
            with self.assertRaisesRegex(ValueError, 'producer and metadata clone ancestry'):
                trainer.fit_joint_from_program(
                    program_path=program_path, program_file_sha256=recipe['program_sha256'],
                    material_root=material, admission_manifest_path=self.root/'no-admission.json',
                    admission_manifest_sha256='2'*64, owner_authorization_ref='fixture',
                    feature_root=bank, feature_roster_sha256=roster['record_sha256'],
                    prepared_base_dir=self.metadata, prepared_receipt_path=self.receipt_path,
                    prepared_receipt_file_sha256=self.receipt_file,
                    prepared_receipt_sha256=self.receipt['receipt_sha256'],
                    parent_source_receipt_sha256=self.receipt['parent_source_receipt_sha256'],
                    seed_weight_path=self.root/'no-seed.safetensors', seed_config_path=self.root/'no-config.json',
                    seed_manifest_path=self.root/'no-seed.json', seed_weight_sha256='8'*64,
                    seed_config_sha256='9'*64, seed_manifest_sha256='4'*64,
                    seed_provenance_sha256='a'*64, seed_source_pins={'fixture': 'b'*64},
                    recipe_path=recipe_path, recipe_file_sha256=sha256(recipe_path.read_bytes()).hexdigest(),
                    output_dir=output, prepared_consumer_profile=PROFILE)
        processor_load.assert_not_called()
        tensor_load.assert_not_called()
        seed_load.assert_not_called()
        self.assertFalse(output.exists())

    def emit_actual_final_custody(self, metadata, processor):
        # Execute the real final provenance block without loading/fitting a large owned model.
        tree = ast.parse(Path(trainer.__file__).read_text(encoding='utf-8'))
        fit = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'fit_joint_from_program')
        block = next(node for node in fit.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Name)
                     and node.test.left.id == 'consumer_metadata')
        namespace = dict(vars(trainer), consumer_metadata=metadata, processor=processor,
                         provenance={}, trained_source_pins={})
        exec(compile(ast.Module(body=[block], type_ignores=[]), '<actual-fit-provenance-block>', 'exec'), namespace)
        return namespace['provenance']['processor_custody'], namespace['trained_source_pins']

    def test_actual_emitted_custody_seals_metadata_and_real_processor_and_detects_mutation(self):
        from tests.governance.test_mfm_source_alignment import Processor
        processor = Processor()
        _, metadata = self.verify()
        emitted, source_pins = self.emit_actual_final_custody(metadata, processor)
        self.assertEqual(emitted['record_sha256'],
                         canonical_sha256({k:v for k,v in emitted.items() if k != 'record_sha256'}))
        self.assertEqual(emitted['metadata'], metadata)
        self.assertEqual(emitted['processor_binding'], trainer.source.processor_binding(processor))
        self.assertEqual(source_pins['native_joint_processor_custody'], emitted)
        altered = deepcopy(emitted)
        altered['processor_binding']['tokenizer_wrapper']['padding_side'] = 'changed'
        self.assertNotEqual(altered['record_sha256'],
                            canonical_sha256({k:v for k,v in altered.items() if k != 'record_sha256'}))

    def test_final_custody_refuses_modified_inner_evidence(self):
        from tests.governance.test_mfm_source_alignment import Processor
        _, metadata = self.verify()
        metadata['asset_roster_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            self.emit_actual_final_custody(metadata, Processor())


@unittest.skipUnless(os.environ.get('ALICE_MFM_FROZEN_BANK_ORIGINAL_INTEGRATION') == '1',
                     'opt in to genuine original processor assets')
class GenuineFrozenBankMetadataTests(unittest.TestCase):
    def test_genuine_seven_assets_under_distinct_consumer_role(self):
        # This is real metadata with an explicitly assembled binding fixture, not a production bank.
        workspace = Path(os.environ.get('ALICE_MFM_ORIGINAL_WORKSPACE_ROOT', Path(__file__).resolve().parents[4]))
        inputs = json.loads((workspace/'work/mfm-frozen-bank-consumer-20261015/GENUINE_METADATA_INPUTS_001.json').read_bytes())
        path = Path(inputs['receipt_path'])
        receipt = json.loads(path.read_bytes())
        recipe = recipe_fixture(receipt)
        row, metadata = trainer._verified_fit_prepared(None, path, Path(inputs['metadata_directory']),
            inputs['receipt_FILE_sha256'], inputs['receipt_sha256'],
            receipt['parent_source_receipt_sha256'],
            roster=binding_fixture(receipt, inputs['receipt_FILE_sha256']), recipe=recipe,
            prepared_consumer_profile=PROFILE, prior_metadata=None)
        self.assertEqual(row['receipt_sha256'], '54484dd523c396648fdf7069abe219c3f8f7c1fdcfec026cda1038810d4d7ac7')
        self.assertEqual(metadata['asset_roster_sha256'], 'cbc727fb52e8d5495bf96e48e13a8a9c045fe64386ff68dc9873873810a288bc')
        self.assertFalse(metadata['base_weights_opened'])
