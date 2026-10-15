"""Configuration custody controls. Meta models and binding fixtures are not features."""
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from scripts.mfm import export_joint_source_features as features


def seal(record):
    return {**record, 'record_sha256': canonical_sha256(record)}


def identity_fixture():
    # Explicitly assembled metadata mechanics; no model/bank construction claim.
    prefix = 'transformers.models.gemma4_unified.'
    return seal({'schema': 'mfm-loaded-source-config-identity-v1',
        'physical_config_sha256': 'a'*64,
        'expected_full_config_sha256': 'b'*64, 'loaded_full_config_sha256': 'b'*64,
        'config_class': prefix+'configuration_gemma4_unified.Gemma4UnifiedConfig',
        'model_class': prefix+'modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration',
        'backbone_class': prefix+'modeling_gemma4_unified.Gemma4UnifiedModel',
        'text_model_class': prefix+'modeling_gemma4_unified.Gemma4UnifiedTextModel',
        'derivation': {'dtype': 'bfloat16', 'reference': 'fixed-Gemma4UnifiedConfig.from_dict',
            'dtype_resolution': 'transformers.modeling_utils._get_dtype-explicit-bfloat16',
            'path_context': 'supplied-producer-prepared-directory',
            'transformers_version': '5.17.0', 'source_files': {
                'configuration_utils': '02826ac9e987487b645135d8f1d747f60a7e567d1b8f72327de8514d02b6f15d',
                'modeling_utils': '38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d',
                'gemma4_config': 'f484527a992041a950b86d433430739fdc0f6000c03af9d87ed8b122b4715119',
                'gemma4_model': 'ae181490a16d2f36cd8f74d4d268757a5388dafc421be0ccc5dbefeff3ba9e57'}},
        'requested_attention': 'sdpa', 'observed_attention': {'root': 'sdpa', 'text': 'sdpa'}})


class StoredConfigIdentityTests(unittest.TestCase):
    def records(self):
        base = {'fixture_small_width': False, 'config_file_sha256': 'a'*64,
                'runtime': {'transformers_version': '5.17.0', 'torch_version': '2.7.1+cu118'},
                'config_identity': identity_fixture()}
        return {'base': base, 'execution_policy': {'attention_implementation': 'sdpa'}}, {'base': deepcopy(base)}

    def test_stored_producer_proof_does_not_use_consumer_runtime(self):
        roster, receipt = self.records()
        with patch.object(torch, '__version__', '2.11.0+cu128'), \
             patch.object(features, 'version', side_effect=AssertionError('live consumer version used')):
            features._validate_config_identity(roster, receipt)

    def test_seal_normalized_raw_class_and_execution_cross_bindings_refuse(self):
        for field, value in (
            ('physical_config_sha256', 'c'*64),
            ('loaded_full_config_sha256', 'c'*64), ('model_class', 'foreign.Model'),
            ('config_class', 'foreign.Config'), ('backbone_class', 'foreign.Backbone'),
            ('text_model_class', 'foreign.Text'), ('requested_attention', None),
            ('observed_attention', {'root': 'sdpa', 'text': 'eager'})):
            roster, receipt = self.records()
            core = {k:v for k,v in identity_fixture().items() if k != 'record_sha256'}
            core[field] = value
            roster['base']['config_identity'] = seal(core)
            receipt['base']['config_identity'] = seal(core)
            with self.subTest(field=field), self.assertRaises(ValueError):
                features._validate_config_identity(roster, receipt)
        roster, receipt = self.records()
        receipt['base']['config_identity']['expected_full_config_sha256'] = 'c'*64
        with self.assertRaises(ValueError):
            features._validate_config_identity(roster, receipt)

    def test_actual_reader_refuses_identity_before_source_or_tensor_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root/'cases/config-fixture'
            case.mkdir(parents=True)
            roster, receipt = self.records()
            roster['base']['prepared_base'] = {}
            roster['base']['backbone_device'] = 'cpu'
            roster['execution_policy'] = seal({'schema': features.CPU_POLICY_SCHEMA,
                'profile': features.CPU_FLASH_PROFILE, 'device': 'cpu',
                'attention_implementation': 'sdpa', 'backend': 'FLASH_ATTENTION',
                'torch_version': '2.7.1+cu118', 'transformers_version': '5.17.0',
                'python_version': '3.11.0', 'platform': 'Linux', 'cpu_threads': 12,
                'text_attention_layers': 48,
                'peak_rss_measurement': 'linux-getrusage-self-process-lifetime-high-water-bytes',
                'whole_source_single_forward': True})
            receipt['base'] = deepcopy(roster['base'])
            receipt['base']['config_identity']['physical_config_sha256'] = 'c'*64
            receipt.update(schema=features.CASE_SCHEMA, profile='j4-p1', case_id='config-fixture',
                alignment_sha256='d'*64, source_length=8, execution={
                    'schema': features.CPU_FORWARD_SCHEMA,
                    'policy_sha256': roster['execution_policy']['record_sha256'],
                    'case_id': 'config-fixture', 'source_alignment_sha256': 'd'*64,
                    'source_length': 8, 'cpu_flash_operator_count': 48, 'math_operator_count': 0,
                    'cpu_flash_operator_time_us': 1.0, 'wall_seconds': 0.001, 'cpu_seconds': 0.002,
                    'process_lifetime_peak_rss_before_bytes': 1000,
                    'process_lifetime_peak_rss_after_bytes': 2000})
            features._validate_execution_records(roster, receipt)
            receipt = seal(receipt)
            (case/'receipt.json').write_bytes(canonical_json_bytes(receipt))
            (case/'features.safetensors').write_bytes(b'mechanics-not-a-tensor')
            roster.update(schema=features.SCHEMA, cases=[{'case_id': 'config-fixture',
                                                        'receipt_sha256': receipt['record_sha256']}])
            roster = seal(roster)
            (root/'roster.json').write_bytes(canonical_json_bytes(roster))
            with patch.object(features, '_source_api', side_effect=AssertionError('source read')), \
                 patch.object(features, 'load_file', side_effect=AssertionError('tensor read')):
                with self.assertRaisesRegex(ValueError, 'config identity'):
                    features.read_case(root, roster_sha256=roster['record_sha256'],
                        case_id='config-fixture', expected_alignment_sha256='d'*64, expected_cache={})


@unittest.skipUnless(os.environ.get('ALICE_MFM_SOURCE_CONFIG_ORIGINAL_INTEGRATION') == '1',
                     'opt in to pinned actual Gemma4 config/meta controls')
class ActualGemmaConfigIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoModelForMultimodalLM
        workspace = Path(os.environ.get('ALICE_MFM_ORIGINAL_WORKSPACE_ROOT', Path(__file__).resolve().parents[4]))
        cls.config_path = workspace/'work/gemma4-mfm-tokenizer-metadata-20261007/config.json'
        cls.config_sha = sha256(cls.config_path.read_bytes()).hexdigest()
        if cls.config_sha != '14f38c5492ffc9cbcdf808647ca0c025bb5b9b4eb737526347134d500ace6098':
            raise AssertionError('genuine physical config FILE differs')
        config = AutoConfig.from_pretrained(str(cls.config_path.parent), local_files_only=True,
                                            trust_remote_code=False)
        with torch.device('meta'):
            cls.base = AutoModelForMultimodalLM.from_config(config, dtype=torch.bfloat16,
                                                           attn_implementation='sdpa')
        cls.base.requires_grad_(False).eval()
        assert all(parameter.is_meta for parameter in cls.base.parameters())

    def identity(self, base=None, attention='sdpa'):
        return features._loaded_config_identity(self.base if base is None else base, self.config_path,
            config_sha256=self.config_sha, requested_attention=attention)

    def test_live_text_config_aliases_refuse_independent_drift(self):
        text = self.base.model.language_model
        holders = [module for module in text.modules() if hasattr(module, 'config')]
        self.assertGreater(len(holders), len(text.layers))
        self.assertTrue(all(module.config is self.base.config.text_config for module in holders))
        # Root serialization stays genuine while a live consumer has its own config.
        for module in (text, text.layers[0], text.layers[0].self_attn,
                       text.layers[0].mlp, text.rotary_emb):
            original = module.config
            try:
                module.config = deepcopy(original)
                module.config.sliding_window = 2048
                with self.subTest(module=type(module).__name__), self.assertRaisesRegex(
                        ValueError, 'text config alias'):
                    self.identity()
            finally:
                module.config = original

    def test_actual_typed_meta_positive_preserves_dtype_rng_and_opens_no_checkpoint(self):
        from transformers import modeling_utils
        default_dtype = torch.get_default_dtype()
        rng = torch.random.get_rng_state().clone()
        with patch.object(modeling_utils, 'load_state_dict', side_effect=AssertionError('checkpoint read')):
            record = self.identity()
        self.assertEqual(record['physical_config_sha256'], self.config_sha)
        self.assertEqual(record['expected_full_config_sha256'], record['loaded_full_config_sha256'])
        self.assertEqual(record['record_sha256'], canonical_sha256({k:v for k,v in record.items() if k != 'record_sha256'}))
        self.assertEqual(torch.get_default_dtype(), default_dtype)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        physical = json.loads(self.config_path.read_bytes())
        # Original red predicate is false even though the actual typed/meta config is correct.
        self.assertNotEqual(physical, json.loads(self.base.config.to_json_string(use_diff=True)))

    def test_complete_config_semantic_dtype_and_unknown_field_mutations_refuse(self):
        original = self.base.config
        text_holders = [module for module in self.base.model.language_model.modules()
                        if hasattr(module, 'config')]
        mutations = (
            lambda cfg: setattr(cfg.text_config, 'num_attention_heads', 8),
            lambda cfg: setattr(cfg.text_config, 'sliding_window', 2048),
            lambda cfg: cfg.text_config.rope_parameters['full_attention'].update(rope_theta=12345.0),
            lambda cfg: setattr(cfg.text_config, 'per_layer_config', {5: {'head_dim': 256}}),
            lambda cfg: setattr(cfg.text_config, 'dtype', torch.float32),
            lambda cfg: setattr(cfg, 'unexpected_semantic_field', True),
            lambda cfg: setattr(cfg, 'name_or_path', '/foreign-producer'))
        try:
            for index, mutate in enumerate(mutations):
                changed = deepcopy(original)
                mutate(changed)
                self.base.config = self.base.model.config = changed
                for module in text_holders:
                    module.config = changed.text_config
                with self.subTest(index=index), self.assertRaisesRegex(
                        ValueError, 'full typed source config|config origin'):
                    self.identity()
        finally:
            self.base.config = self.base.model.config = original
            for module in text_holders:
                module.config = original.text_config

    def test_foreign_model_config_and_backbone_classes_refuse(self):
        with self.assertRaises(ValueError):
            self.identity(SimpleNamespace(config=self.base.config, model=self.base.model))
        original = self.base.config
        try:
            self.base.config = SimpleNamespace(model_type='gemma4_unified',
                to_json_string=original.to_json_string, text_config=original.text_config)
            with self.assertRaises(ValueError):
                self.identity()
        finally:
            self.base.config = original
        backbone = self.base.model
        try:
            self.base.model = torch.nn.Identity()
            with self.assertRaises(ValueError):
                self.identity()
        finally:
            self.base.model = backbone

    def test_default_attention_preserved_and_explicit_eager_refused(self):
        original = self.base.config._attn_implementation
        try:
            self.base.config._attn_implementation = 'eager'
            record = self.identity(attention=None)
            self.assertIsNone(record['requested_attention'])
            self.assertEqual(record['observed_attention'], {'root': 'eager', 'text': 'eager'})
            self.assertEqual(self.base.config._attn_implementation, 'eager')
            with self.assertRaises(ValueError):
                self.identity(attention='sdpa')
        finally:
            self.base.config._attn_implementation = original

    def test_default_bank_shape_and_explicit_identity_and_raw_hash_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            operand = Path(directory)/'mechanics-file.bin'
            operand.write_bytes(b'not pretrained model weights; binding mechanics only')
            args = (self.base, operand, sha256(operand.read_bytes()).hexdigest(),
                    self.config_path, self.config_sha,
                    {'torch_version': torch.__version__, 'transformers_version': version('transformers')}, False)
            default = features._base_binding(*args)
            self.assertNotIn('config_identity', default)
            explicit = features._base_binding(*args, execution_policy={
                'profile': features.CPU_FLASH_PROFILE, 'attention_implementation': 'sdpa'})
            self.assertEqual(explicit['config_identity']['physical_config_sha256'], self.config_sha)
            with self.assertRaisesRegex(ValueError, 'actual frozen base FILE'):
                features._base_binding(*args[:4], 'e'*64, *args[5:])
