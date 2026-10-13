"""Original-packet lowering controls. Fictional records prove mechanics only."""
import unittest
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import torch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from tests.governance import test_mfm_original_copy_decision as copy_fixture
from tests.governance import test_mfm_joint_learning as joint_fixture


class JointLoweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        copy_fixture.OriginalCopyDecisionTests.setUpClass()
        cls.fixture = copy_fixture.OriginalCopyDecisionTests()
        from cognitive_kernel import formation_original_copy_decision as copies
        from scripts.mfm import formation_original_copy_target_alignment as target
        from scripts.mfm import formation_joint_target_alignment as original
        from scripts.mfm import formation_paired_teaching as paired
        from cognitive_kernel import formation_learning_v16 as learning
        from scripts.mfm.train_native_joint_formation import lower_original_case
        ex, pair, _, cls.inventory, _ = cls.fixture.material(copies, target)
        output = ex.target.output_record()
        target_record = {'schema': learning.TARGET_SCHEMA_V16,
            'context': ex.context.record(), 'source_ids': [ref for ref, _ in ex.opened_sources],
            'proposals': output['proposals'], 'dispositions': output['dispositions'],
            'adjudications': dict(ex.adjudications)}
        raw = canonical_json_bytes(target_record)
        accepted = paired.admit_target(cls.fixture.h.frozen,
            expected_freeze_sha256=cls.fixture.h.frozen['record_sha256'],
            expected=cls.fixture.h.expected, target_reader=lambda: raw,
            expected_target_file_sha256=sha256(raw).hexdigest())
        annotations = list(pair['annotations']) + [
            {'pointer': row['pointer'], 'mode': 'semantic_authoring'}
            for row in pair['fields'] if row['responsibility'] == 'semantic_authoring']
        pair = original.align_target(cls.fixture.processor, cls.fixture.h.frozen,
            accepted, expected=cls.fixture.h.expected,
            expected_freeze_sha256=cls.fixture.h.frozen['record_sha256'],
            expected_target_record_sha256=accepted['record_sha256'],
            max_target_tokens=20000, annotations=annotations,
            literal_declarations=pair['literal_declarations'])
        result = lower_original_case(profile='j4-p1', frozen=cls.fixture.h.frozen,
            accepted=accepted,
            pair=pair, expected=cls.fixture.h.expected, processor=cls.fixture.processor,
            max_target_tokens=20000)
        cls.original_raw = raw
        cls.accepted = accepted
        cls.pair = pair
        cls.lowered = result
        cls.example = ex

    def test_complete_original_packet_lowers_whole_authored_choices(self):
        result = self.lowered
        self.assertEqual(result['inventory']['record_sha256'], self.inventory['record_sha256'])
        proof = self.fixture.h.frozen['meaning']['episode_snapshot']['historical_prefix_proof']
        self.assertEqual(result['cut_sha256'], proof['cut']['record_sha256'])
        self.assertTrue(result['target_view']['authored_ordinals'])
        self.assertTrue(result['target_view']['authored_copy_tuples']['rows'][0]['alternatives'])

    def test_admitted_author_conversion_binds_pair_and_choice_bytes(self):
        from scripts.mfm.train_native_joint_formation import (
            preflight_admitted_cases, PAIR_AUTHORING_SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, raw):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                return {'path': name, 'sha256': sha256(raw).hexdigest()}
            def source_row(split, source_id, raw):
                payload = write(f'{split}/{source_id}.bin', raw)
                rights = write(f'{split}/{source_id}-rights.json', canonical_json_bytes({
                    'schema': 'mfm-source-rights-v1', 'issuer_id': 'fixture-owner',
                    'source_id': source_id, 'source_sha256': payload['sha256'],
                    'host_family': train_host if split == 'train' else 'host-development',
                    'authority_ref': 'fixture-rights',
                    'revoked': False, 'formation_training': True,
                    'model_distribution': True, 'formation_evaluation': True}))
                return {'source_id': source_id, **payload,
                    'rights_path': rights['path'], 'rights_sha256': rights['sha256'],
                    'parent_source_ids': []}
            case_id = self.lowered['case_id']
            train_host = self.example.context.base.scope.host_instance_id
            train_sources = [source_row('train', ref, raw)
                for ref, raw in self.example.opened_sources]
            target = write('train/target.json', self.original_raw)
            prompt = write('train/prompt.txt', b'fictional teacher prompt')
            output = write('train/output.txt', b'fictional teacher output')
            lowered = deepcopy(self.lowered)
            lowered['variant_key'] = 'fixture-variant-one'
            conversion = {'schema': PAIR_AUTHORING_SCHEMA,
                'case_id': case_id, 'author_id': 'fixture-train-author',
                'original_frozen_sha256': self.fixture.h.frozen['record_sha256'],
                'original_target_file_sha256': target['sha256'],
                'original_pair_sha256': self.pair['record_sha256'],
                'source_alignment_sha256': lowered['source_alignment_sha256'],
                'source_file_sha256s': sorted(row['sha256'] for row in train_sources),
                'target_provenance_output_sha256': output['sha256'],
                'annotations_sha256': canonical_sha256(self.pair['annotations']),
                'literal_declarations_sha256': canonical_sha256(
                    self.pair['literal_declarations']),
                'authored_ordinals_sha256': canonical_sha256(
                    lowered['authored_ordinals']),
                'authored_copy_tuples_sha256': canonical_sha256(
                    lowered['authored_copy_tuples']),
                'world_sha256': lowered['world_sha256'],
                'cut_sha256': lowered['cut_sha256'],
                'variant_key': lowered['variant_key']}
            conversion_ref = write('train/conversion.json',
                                   canonical_json_bytes(conversion))
            dev_source = source_row('development', 'dev-original', b'fictional dev source')
            dev_target = write('development/target.json', b'fictional dev target')
            dev_input = write('development/prompt.txt', b'fictional dev prompt')
            dev_output = write('development/output.txt', b'fictional dev output')
            dev_conversion = write('development/conversion.json', b'{}')
            def corpus_row(name, split, sources, target_ref, input_ref,
                           output_ref, conversion_ref):
                return {'case_id': name, 'split': split,
                    'author_id': f'fixture-{split}-author',
                    'host_family': train_host if split == 'train' else 'host-development',
                    'source_family': f'fixture-{split}-source',
                    'generator_family': f'fixture-{split}-generator',
                    'scenario_family': f'fixture-{split}-scenario',
                    'duplicate_group': f'fixture-{split}-duplicate',
                    'parent_case_ids': [],
                    'authorization_id': 'fixture-owner-authorized',
                    'target_origin': 'owner-authorized-service-teacher',
                    'target_provenance': {
                        'producer_id': f'fixture-{split}-author',
                        'producer_version': 'fixture-v1',
                        'input': input_ref, 'output': output_ref,
                        'conversion': conversion_ref},
                    'sources': sources, 'target': target_ref, 'reviews': []}
            manifest = {'schema': 'mfm-v16-owner-teacher-corpus-v1',
                'corpus_id': 'fictional-joint-corpus',
                'status': 'owner-authorized-teacher-training-only-unqualified',
                'authorization_id': 'fixture-owner-authorized',
                'case_count': 2, 'cases': [
                    corpus_row(case_id, 'train', train_sources, target,
                               prompt, output, conversion_ref),
                    corpus_row('dev-case', 'development', [dev_source], dev_target,
                               dev_input, dev_output, dev_conversion)]}
            manifest_path = root / 'manifest.json'
            joined = {'lowered': lowered, 'feature_receipt': {
                'case_id': case_id, 'base': {'fixture_small_width': False},
                'target_visible_to_backbone': False}}
            cases = [{'frozen': self.fixture.h.frozen, 'accepted': self.accepted,
                      'pair': self.pair, 'joined': joined}]
            def preflight():
                manifest_path.write_bytes(canonical_json_bytes(manifest))
                return preflight_admitted_cases(
                    admission_manifest_path=manifest_path,
                    admission_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                    owner_authorization_ref='fixture-owner-authorized',
                    cases=cases)
            admission, audit = preflight()
            self.assertEqual(admission.train[0].case_id, case_id)
            self.assertEqual(len(audit['gradient_inputs']), len(train_sources) + 1)
            conversion['original_pair_sha256'] = 'f'*64
            manifest['cases'][0]['target_provenance']['conversion'] = write(
                'train/conversion.json', canonical_json_bytes(conversion))
            with self.assertRaisesRegex(ValueError, 'author conversion'):
                preflight()


class JointMassWindowTests(unittest.TestCase):
    def test_partial_windows_preserve_global_world_cut_variant_mass(self):
        from scripts.mfm.train_native_joint_formation import _fit_update

        class ConstantJoint(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def joint_loss(self, **_):
                loss = self.weight * 2
                return {'total_loss': loss, 'lexical_loss': loss,
                        'reference_loss': None, 'copy_loss': None,
                        'token_count': 1, 'reference_request_count': 0,
                        'copy_request_count': 0}

        rows = []
        for case_id, world, cut, variant in (
                ('first', 'a', 'c', 'v1'), ('second', 'a', 'c', 'v2'),
                ('third', 'b', 'd', 'v1')):
            lower = {'case_id': case_id, 'world_sha256': world*64,
                     'cut_sha256': cut*64, 'variant_key': variant,
                     'alignment': {'decoder_inputs': [1], 'labels': [1],
                                   'record_sha256': 'e'*64},
                     'inventory': {}, 'decision': {},
                     'authored_ordinals': [], 'authored_copy_tuples': {},
                     'target_view': {'record_sha256': 'f'*64}}
            rows.append({'joined': {'lowered': lower,
                'source_states': torch.ones(1, 1, 1, dtype=torch.bfloat16),
                'source_mask': torch.ones(1, 1, dtype=torch.long)}})
        for index, coefficient in ((0, 0.75), (2, 1.5)):
            model = ConstantJoint()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0)
            observed = _fit_update(model, object(), rows, [rows[index]],
                optimizer, device=torch.device('cpu'))
            self.assertEqual(observed[0]['global_mass_coefficient'], coefficient)
            self.assertEqual(float(model.weight.grad), 2 * coefficient)


class JointOwnedUpdateTests(unittest.TestCase):
    _fixtures = joint_fixture.JointLearningTests._fixtures
    setUpClass = classmethod(joint_fixture.JointLearningTests.setUpClass.__func__)
    setUp = joint_fixture.JointLearningTests.setUp
    packet = joint_fixture.JointLearningTests.packet
    record = joint_fixture.JointLearningTests.record

    def test_production_epoch_reopens_one_frozen_bank_per_case(self):
        from scripts.mfm import export_joint_source_features as features
        from scripts.mfm.train_native_joint_formation import _fit_epoch
        packet = self.packet()
        source = packet['base_states'].detach().clone().requires_grad_()
        original_batch = {'input_ids': [self.joint['input_ids']],
                          'attention_mask': [self.joint['attention_mask']]}
        receipt = {'case_id': 'fixture-one', 'profile': 'j4-p1',
                   'frozen_batch': original_batch,
                   'base': {'fixture_small_width': False}}
        tensors = {'states': source,
            'input_ids': torch.tensor(original_batch['input_ids']),
            'attention_mask': packet['source_mask']}
        lower = {'case_id': 'fixture-one', 'profile': 'j4-p1',
            'world_sha256': 'a'*64, 'cut_sha256': 'b'*64,
            'variant_key': 'authored-variant-one',
            'source_alignment_sha256': self.joint['record_sha256'],
            'source_cache': self.joint['cache'],
            'source_projection': original_batch,
            'alignment': packet['alignment'], 'inventory': packet['inventory'],
            'decision': packet['decision'], 'target_view': packet['target_view'],
            'authored_ordinals': packet['authored_ordinals'],
            'authored_copy_tuples': packet['authored_copy_tuples']}
        rows = [{'joined': {'lowered': lower, 'feature_receipt': receipt}}]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with patch.object(features, 'read_case', return_value=(receipt, tensors)) as reader:
            observations = _fit_epoch(self.model, self.processor, rows,
                optimizer, device=torch.device('cpu'), feature_root='physical-fixture',
                feature_roster_sha256='3'*64)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(observations[0]['case_id'], 'fixture-one')
        self.assertIsNone(source.grad)
        self.assertNotIn('source_states', rows[0]['joined'])

    def test_actual_joint_owned_step_and_complete_restart(self):
        from scripts.mfm.train_native_joint_formation import _fit_update, _save_checkpoint, _restore_checkpoint
        packet = self.packet()
        source = packet['base_states'].detach().clone().requires_grad_()
        lower = {'case_id': 'fixture-one',
            'split_row': {'lineage': {'world_id': 'world-one', 'branch_id': 'cut-one'}},
            'world_sha256': 'a'*64, 'cut_sha256': 'b'*64,
            'variant_key': 'authored-variant-one',
            'alignment': packet['alignment'], 'inventory': packet['inventory'],
            'decision': packet['decision'], 'target_view': packet['target_view'],
            'authored_ordinals': packet['authored_ordinals'],
            'authored_copy_tuples': packet['authored_copy_tuples']}
        second = deepcopy(lower)
        second['case_id'] = 'fixture-two'
        second['variant_key'] = 'authored-variant-two'
        rows = [{'joined': {'lowered': item, 'source_states': source,
                            'source_mask': packet['source_mask']}}
                for item in (lower, second)]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3,
                                      weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        observations = _fit_update(self.model, self.processor, rows, [rows[0]],
            optimizer, device=torch.device('cpu'))
        self.assertEqual([entry['case_id'] for entry in observations], ['fixture-one'])
        self.assertEqual(observations[0]['reference_request_count'],
                         len(packet['alignment']['requests']))
        self.assertEqual(observations[0]['copy_request_count'],
                         len(packet['alignment']['copy_requests']))
        self.assertIsNone(source.grad)
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in self.model.state_dict().items()))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'checkpoint-000001'
            pins = dict(recipe_sha256='1'*64, program_sha256='2'*64,
                        admission_sha256='3'*64, feature_roster_sha256='4'*64)
            _save_checkpoint(target, model=self.model, optimizer=optimizer,
                             update_index=1, consumed_cases=1,
                             case_order=['fixture-one', 'fixture-two'],
                             history=[{'update_index': 1, 'next_case_index': 1,
                                       'completed_passes': 0,
                                       'cases': observations}], **pins)
            uninterrupted = _fit_update(self.model, self.processor, rows,
                [rows[1]], optimizer, device=torch.device('cpu'))
            from cognitive_kernel.formation_joint_learning import FormationJointLearning
            fresh = FormationJointLearning(self.model.config)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3,
                weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
            checkpoint = _restore_checkpoint(target, model=fresh,
                optimizer=fresh_optimizer,
                expected_checkpoint_file_sha256=sha256(
                    (target / 'checkpoint.json').read_bytes()).hexdigest(),
                case_order=['fixture-one', 'fixture-two'],
                accumulation_cases=1, **pins)
            self.assertEqual(checkpoint['update_index'], 1)
            self.assertEqual(checkpoint['next_case_index'], 1)
            self.assertEqual(checkpoint['history'][0]['cases'], observations)
            restarted = _fit_update(fresh, self.processor, rows, [rows[1]],
                fresh_optimizer, device=torch.device('cpu'))
            self.assertEqual(uninterrupted, restarted)
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(fresh.state_dict()[name], value,
                                           atol=0, rtol=0)
            original = {key: value.clone() for key, value in fresh.state_dict().items()}
            with self.assertRaises(Exception):
                _restore_checkpoint(target, model=fresh, optimizer=fresh_optimizer,
                    expected_checkpoint_file_sha256=sha256(
                        (target / 'checkpoint.json').read_bytes()).hexdigest(),
                    case_order=['fixture-one', 'fixture-two'], accumulation_cases=1,
                    **dict(pins, admission_sha256='f'*64))
            malformed_groups = Path(directory) / 'malformed-groups'
            shutil.copytree(target, malformed_groups)
            record = json.loads((malformed_groups / 'checkpoint.json').read_bytes().decode('utf-8'))
            record['owned_parameter_groups'][0].reverse()
            (malformed_groups / 'checkpoint.json').write_bytes(canonical_json_bytes(record))
            with patch.object(fresh, 'load_state_dict', wraps=fresh.load_state_dict) as model_load, \
                 patch.object(fresh_optimizer, 'load_state_dict', wraps=fresh_optimizer.load_state_dict) as optimizer_load:
                with self.assertRaises(Exception):
                    _restore_checkpoint(malformed_groups, model=fresh,
                        optimizer=fresh_optimizer,
                        expected_checkpoint_file_sha256=sha256(
                            (malformed_groups / 'checkpoint.json').read_bytes()).hexdigest(),
                        case_order=['fixture-one', 'fixture-two'],
                        accumulation_cases=1, **pins)
                model_load.assert_not_called()
                optimizer_load.assert_not_called()
            malformed_rng = Path(directory) / 'malformed-rng'
            shutil.copytree(target, malformed_rng)
            trainer_path = malformed_rng / 'trainer.pt'
            trainer = torch.load(trainer_path, map_location='cpu', weights_only=True)
            trainer['cpu_rng'] = torch.full_like(trainer['cpu_rng'], 255)
            torch.save(trainer, trainer_path)
            record = json.loads((malformed_rng / 'checkpoint.json').read_bytes().decode('utf-8'))
            record['trainer_file_sha256'] = sha256(trainer_path.read_bytes()).hexdigest()
            (malformed_rng / 'checkpoint.json').write_bytes(canonical_json_bytes(record))
            with patch.object(fresh, 'load_state_dict', wraps=fresh.load_state_dict) as model_load, \
                 patch.object(fresh_optimizer, 'load_state_dict', wraps=fresh_optimizer.load_state_dict) as optimizer_load, \
                 patch.object(torch.random, 'set_rng_state', wraps=torch.random.set_rng_state) as rng_load:
                with self.assertRaises(Exception):
                    _restore_checkpoint(malformed_rng, model=fresh,
                        optimizer=fresh_optimizer,
                        expected_checkpoint_file_sha256=sha256(
                            (malformed_rng / 'checkpoint.json').read_bytes()).hexdigest(),
                        case_order=['fixture-one', 'fixture-two'],
                        accumulation_cases=1, **pins)
                model_load.assert_not_called()
                optimizer_load.assert_not_called()
                rng_load.assert_not_called()
            for name, value in fresh.state_dict().items():
                torch.testing.assert_close(value, original[name], atol=0, rtol=0)


class JointBoundaryTests(unittest.TestCase):
    def test_private_original_successor_modules_are_exact_physical_files(self):
        from scripts.mfm.train_native_joint_formation import _private_original_modules

        folder = (Path(__file__).resolve().parents[3] /
            'mfm-joint-original-inventory-roundtrip-design-20261011' /
            'material-read-export-construction-001' / 'dependency-tree-001' /
            'scripts' / 'mfm')
        names = {'source': 'formation_joint_input_v5.py',
                 'paired': 'formation_paired_teaching_v2.py',
                 'target': 'formation_joint_target_alignment_v2.py'}
        refs = {role: {'path': str(folder / name),
                       'sha256': sha256((folder / name).read_bytes()).hexdigest()}
                for role, name in names.items()}
        modules = _private_original_modules(refs)
        self.assertEqual(set(modules), set(names))
        self.assertEqual(modules['source'].__name__,
                         'scripts.mfm.formation_joint_input_v5')
        self.assertTrue(callable(modules['paired'].validate_input))
        self.assertTrue(callable(modules['target'].validate_pair))
        changed = deepcopy(refs)
        changed['paired']['sha256'] = 'f'*64
        with self.assertRaises(ValueError):
            _private_original_modules(changed)

    def test_full_native_source_fields_join_original_two_field_projection(self):
        from scripts.mfm import export_joint_source_features as features
        from scripts.mfm.train_native_joint_formation import join_frozen_source
        lowered = {'case_id': 'fixture-one', 'profile': 'j4-p1',
            'source_alignment_sha256': '1'*64, 'source_cache': {'pin': '2'*64},
            'source_projection': {'input_ids': [[2, 8]],
                                  'attention_mask': [[1, 1]]},
            'inventory': {'source_geometry': {'attention_mask': [1, 1]}}}
        receipt = {'case_id': 'fixture-one', 'profile': 'j4-p1',
            'frozen_batch': {'input_ids': [[2, 8]], 'attention_mask': [[1, 1]],
                             'mm_token_type_ids': [[0, 0]]}}
        tensors = {'states': torch.ones(1, 2, 4, dtype=torch.bfloat16),
            'input_ids': torch.tensor([[2, 8]]),
            'attention_mask': torch.tensor([[1, 1]]),
            'mm_token_type_ids': torch.tensor([[0, 0]])}
        with patch.object(features, 'read_case', return_value=(receipt, tensors)) as reader:
            joined = join_frozen_source(lowered, feature_root='unused-fixture-path',
                feature_roster_sha256='3'*64, allow_fixture=True)
            self.assertEqual(set(joined['native_batch']),
                             {'input_ids', 'attention_mask', 'mm_token_type_ids'})
            self.assertFalse(joined['source_states'].requires_grad)
            reader.assert_called_once()
        altered = deepcopy(receipt)
        altered['frozen_batch']['input_ids'][0][1] = 7
        with patch.object(features, 'read_case', return_value=(altered, tensors)):
            with self.assertRaises(ValueError):
                join_frozen_source(lowered, feature_root='unused-fixture-path',
                    feature_roster_sha256='3'*64, allow_fixture=True)

    def test_supplied_p2_program_refuses_before_processor_or_seed(self):
        from scripts.mfm.train_native_joint_formation import fit_joint_from_program, PROGRAM_SCHEMA
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        from transformers import AutoProcessor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('material', 'features', 'prepared'):
                (root / name).mkdir()
            receipt = root / 'prepared' / 'receipt.json'
            receipt.write_bytes(b'{}')
            program = {'schema': PROGRAM_SCHEMA, 'cases': [{
                'case_id': 'private-supplied-one', 'profile': 'j5-p2',
                'variant_key': 'private-variant',
                'frozen': {'path': 'frozen.json', 'sha256': '1'*64},
                'accepted': {'path': 'accepted.json', 'sha256': '2'*64},
                'pair': {'path': 'pair.json', 'sha256': '3'*64},
                'max_target_tokens': 100}]}
            program_path = root / 'program.json'
            program_path.write_bytes(canonical_json_bytes(program))
            with patch.object(AutoProcessor, 'from_pretrained') as processor_load, \
                 patch.object(FormationJointLearning, 'load_component') as seed_load:
                with self.assertRaisesRegex(ValueError, 'not independently admitted successor'):
                    fit_joint_from_program(
                        program_path=program_path,
                        program_file_sha256=sha256(program_path.read_bytes()).hexdigest(),
                        material_root=root / 'material',
                        admission_manifest_path=root / 'missing-admission.json',
                        admission_manifest_sha256='4'*64,
                        owner_authorization_ref='fixture',
                        feature_root=root / 'features',
                        feature_roster_sha256='5'*64,
                        prepared_base_dir=root / 'prepared',
                        prepared_receipt_path=receipt,
                        prepared_receipt_file_sha256='6'*64,
                        prepared_receipt_sha256='7'*64,
                        parent_source_receipt_sha256='8'*64,
                        seed_weight_path=root / 'missing-seed.safetensors',
                        seed_config_path=root / 'missing-config.json',
                        seed_manifest_path=root / 'missing-seed-manifest.json',
                        seed_weight_sha256='9'*64,
                        seed_config_sha256='a'*64,
                        seed_manifest_sha256='b'*64,
                        seed_provenance_sha256='c'*64,
                        seed_source_pins={'fixture': 'd'*64},
                        recipe_path=root / 'missing-recipe.json',
                        recipe_file_sha256='e'*64,
                        output_dir=root / 'output')
            processor_load.assert_not_called()
            seed_load.assert_not_called()
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
