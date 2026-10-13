"""Target-free joint source representation mechanics; no actual model capacity claim."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from scripts.mfm import formation_joint_input_v4 as J, formation_source_alignment as S
from scripts.mfm.export_joint_source_features import export_bank, read_case
from scripts.mfm import export_joint_source_features as features
from scripts.mfm import train_v1_formation_specialist as shared
from tests.governance import test_mfm_joint_input_v4 as fixture
from tests.governance.test_mfm_source_alignment import Processor


class TinyBackbone(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(300, width, dtype=torch.bfloat16)
        self.calls = 0
        self.native_inputs = None

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict, **native):
        assert use_cache is False and return_dict is True
        assert torch.is_grad_enabled() is False
        assert input_ids.shape == attention_mask.shape
        assert all(value.shape == input_ids.shape for value in native.values())
        self.calls += 1
        self.native_inputs = {key: value.detach().cpu().clone() for key, value in native.items()}
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyBase(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.model = TinyBackbone(width)
        self.config = SimpleNamespace(model_type='gemma4_unified',
                                      text_config=SimpleNamespace(hidden_size=width))

    def forward(self, **kwargs):
        raise AssertionError('LM head may not be invoked')


class JointSourceFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixture.CompleteSemanticTests.setUpClass()
        original = fixture.CompleteSemanticTests()
        cls.source_packet = original.process()

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.processor, self.bundle, self.kwargs, batch, self.alignment = self.source_packet
        self.batch = {key: tensor.tolist() for key, tensor in batch.items()}
        self.base = TinyBase().eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.weight = self.root / 'model.safetensors'
        self.config = self.root / 'config.json'
        self.weight.write_bytes(b'fixture-untrained-weight-file')
        self.config.write_text('{"text_config":{"hidden_size":4}}', encoding='utf-8')
        self.weight_sha = sha256(self.weight.read_bytes()).hexdigest()
        self.config_sha = sha256(self.config.read_bytes()).hexdigest()
        self.record = dict(case_id='fixture-j4', profile='j4-p1', caller=self.bundle['caller'],
                           meaning=self.bundle['meaning'], source_kwargs=self.kwargs,
                           frozen_batch=self.batch, frozen_alignment=self.alignment)

    def export(self, records=None, j5_source=None):
        return export_bank(self.root / 'bank', cases=[self.record] if records is None else records,
                           processor=self.processor, base=self.base, j5_source=j5_source,
                           base_weight_path=self.weight, base_weight_sha256=self.weight_sha,
                           base_config_path=self.config, base_config_sha256=self.config_sha,
                           runtime={'torch_version': torch.__version__, 'transformers_version': 'fixture'},
                           fixture_small_width=True)

    def test_original_source_full_mask_and_exact_readback(self):
        roster = self.export()
        self.assertEqual(self.base.model.calls, 1)
        receipt, tensors = read_case(self.root / 'bank', roster_sha256=roster['record_sha256'],
                                     case_id='fixture-j4', expected_alignment_sha256=self.alignment['record_sha256'],
                                     expected_cache=self.alignment['cache'], allow_fixture=True)
        self.assertEqual(receipt['source_length'], len(self.alignment['input_ids']))
        self.assertTrue(torch.equal(tensors['input_ids'], torch.tensor(self.batch['input_ids'])))
        self.assertTrue(torch.equal(tensors['attention_mask'], torch.tensor(self.batch['attention_mask'])))
        self.assertEqual(set(tensors), {'states', *self.batch})
        self.assertEqual(tensors['states'].shape, (1, len(self.alignment['input_ids']), 4))
        self.assertEqual(tensors['states'].dtype, torch.bfloat16)
        self.assertEqual(receipt['cache'], self.alignment['cache'])

    def test_source_tamper_and_budget_refuse_before_forward(self):
        changed = deepcopy(self.record)
        changed['frozen_batch']['attention_mask'][0][-1] = 0
        with self.assertRaises(ValueError):
            self.export([changed])
        self.assertEqual(self.base.model.calls, 0)
        changed = deepcopy(self.record)
        changed['source_kwargs']['max_source_tokens'] = len(self.alignment['input_ids']) - 1
        with self.assertRaises(Exception):
            self.export([changed])
        self.assertEqual(self.base.model.calls, 0)

    def test_provenance_tamper_and_target_operand_refused(self):
        changed = deepcopy(self.record)
        changed['frozen_alignment']['cache']['provenance_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.export([changed])
        self.assertEqual(self.base.model.calls, 0)
        changed = deepcopy(self.record)
        changed['target'] = {'answer': 'leak'}
        with self.assertRaises(ValueError):
            self.export([changed])
        self.assertEqual(self.base.model.calls, 0)

    def test_file_tamper_refused_during_consumption(self):
        roster = self.export()
        with self.assertRaises(ValueError):
            read_case(self.root / 'bank', roster_sha256=roster['record_sha256'], case_id='fixture-j4',
                      expected_alignment_sha256=self.alignment['record_sha256'],
                      expected_cache=self.alignment['cache'])
        feature = self.root / 'bank' / 'cases' / 'fixture-j4' / 'features.safetensors'
        raw = bytearray(feature.read_bytes())
        raw[-1] ^= 1
        feature.write_bytes(raw)
        with self.assertRaises(ValueError):
            read_case(self.root / 'bank', roster_sha256=roster['record_sha256'], case_id='fixture-j4',
                      expected_alignment_sha256=self.alignment['record_sha256'],
                      expected_cache=self.alignment['cache'], allow_fixture=True)

    def test_physical_weight_pin_refuses_before_forward(self):
        self.weight.write_bytes(b'altered fixture weight file')
        with self.assertRaises(ValueError):
            self.export()
        self.assertEqual(self.base.model.calls, 0)

    def test_explicit_private_j5_source_profile(self):
        name = 'scripts.mfm.formation_joint_input_v5'
        path = Path(__file__).resolve().parents[3] / 'mfm-joint-original-inventory-roundtrip-design-20261011' / 'construction-001' / 'formation_joint_input_v5.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        policy = module.normative_joint_render_policy_v5()
        kw = {**self.kwargs, 'render_policy': policy,
              'render_policy_sha256': policy['record_sha256'],
              'processor_binding': S.processor_binding(self.processor)}
        batch, alignment = module.joint_source_batch_with_alignment_v5(
            self.processor, self.bundle['caller'], self.bundle['meaning'], **kw)
        case = {**self.record, 'case_id': 'fixture-j5', 'profile': 'j5-p2',
                'source_kwargs': kw, 'frozen_batch': {k: v.tolist() for k, v in batch.items()},
                'frozen_alignment': alignment}
        roster = self.export([case], j5_source=module)
        receipt, tensors = read_case(self.root / 'bank', roster_sha256=roster['record_sha256'],
                                     case_id='fixture-j5', expected_alignment_sha256=alignment['record_sha256'],
                                     expected_cache=alignment['cache'], j5_source=module, allow_fixture=True)
        self.assertEqual(receipt['profile'], 'j5-p2')
        self.assertEqual(tensors['states'].shape[1], len(alignment['input_ids']))
        self.assertEqual(self.base.model.calls, 1)


class NativeTextFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixture.CompleteSemanticTests.setUpClass()

        class UnifiedTextProcessor(Processor):
            def __init__(self):
                super().__init__()
                self.mm_dtype = torch.int64
                self.mm_short = False
                self.add_target = False

            def apply_chat_template(self, messages, *, chat_template, tokenize, **kwargs):
                batch = super().apply_chat_template(messages, chat_template=chat_template,
                                                    tokenize=tokenize, **kwargs)
                if tokenize:
                    batch.pop('token_type_ids', None)
                    mm = torch.zeros_like(batch['input_ids'], dtype=self.mm_dtype)
                    batch['mm_token_type_ids'] = mm[:, :-1] if self.mm_short else mm
                    if self.add_target:
                        batch['labels'] = batch['input_ids'].clone()
                return batch

        cls.processor = UnifiedTextProcessor()
        _, bundle, kwargs, batch, alignment = fixture.CompleteSemanticTests().process(cls.processor)
        cls.case = {'case_id': 'native-text-field', 'profile': 'j4-p1',
                    'caller': bundle['caller'], 'meaning': bundle['meaning'],
                    'source_kwargs': kwargs,
                    'frozen_batch': {key: value.tolist() for key, value in batch.items()},
                    'frozen_alignment': alignment}

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = TinyBase().eval().requires_grad_(False)
        self.weight = self.root / 'model.safetensors'
        self.config = self.root / 'config.json'
        self.weight.write_bytes(b'fixture-untrained-native-text')
        self.config.write_text('{"text_config":{"hidden_size":4}}', encoding='utf-8')

    def export(self, case=None):
        return export_bank(self.root / 'bank', cases=[self.case if case is None else case],
                           processor=self.processor, base=self.base, j5_source=None,
                           base_weight_path=self.weight,
                           base_weight_sha256=sha256(self.weight.read_bytes()).hexdigest(),
                           base_config_path=self.config,
                           base_config_sha256=sha256(self.config.read_bytes()).hexdigest(),
                           runtime={'torch_version': torch.__version__,
                                    'transformers_version': 'fixture'},
                           fixture_small_width=True)

    def test_complete_native_text_field_forward_save_and_readback(self):
        self.assertEqual(set(self.case['frozen_batch']),
                         {'input_ids', 'attention_mask', 'mm_token_type_ids'})
        roster = self.export()
        receipt, tensors = read_case(
            self.root / 'bank', roster_sha256=roster['record_sha256'],
            case_id=self.case['case_id'],
            expected_alignment_sha256=self.case['frozen_alignment']['record_sha256'],
            expected_cache=self.case['frozen_alignment']['cache'], allow_fixture=True)
        expected = torch.tensor(self.case['frozen_batch']['mm_token_type_ids'], dtype=torch.int64)
        self.assertEqual(set(tensors), {'states', *self.case['frozen_batch']})
        self.assertEqual(receipt['frozen_batch']['mm_token_type_ids'], expected.tolist())
        self.assertTrue(torch.equal(self.base.model.native_inputs['mm_token_type_ids'], expected))
        self.assertTrue(torch.equal(tensors['mm_token_type_ids'], expected))
        self.assertEqual(tensors['mm_token_type_ids'].shape, tensors['input_ids'].shape)

    def test_native_field_input_and_saved_tensor_tamper_refusals(self):
        for change in ('missing', 'value', 'shape'):
            altered = deepcopy(self.case)
            mm = altered['frozen_batch']['mm_token_type_ids']
            if change == 'missing':
                del altered['frozen_batch']['mm_token_type_ids']
            elif change == 'value':
                mm[0][0] = 1
            else:
                mm[0].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.export(altered)
        self.assertEqual(self.base.model.calls, 0)
        self.processor.mm_dtype = torch.float32
        try:
            with self.assertRaises(ValueError):
                self.export()
        finally:
            self.processor.mm_dtype = torch.int64
        self.processor.mm_short = True
        try:
            with self.assertRaises(ValueError):
                self.export()
        finally:
            self.processor.mm_short = False
        self.processor.add_target = True
        try:
            with self.assertRaises(ValueError):
                self.export()
        finally:
            self.processor.add_target = False
        self.assertEqual(self.base.model.calls, 0)

        roster = self.export()
        receipt, tensors = read_case(
            self.root / 'bank', roster_sha256=roster['record_sha256'],
            case_id=self.case['case_id'],
            expected_alignment_sha256=self.case['frozen_alignment']['record_sha256'],
            expected_cache=self.case['frozen_alignment']['cache'], allow_fixture=True)
        mm = tensors['mm_token_type_ids']
        for changed in (mm.to(torch.float32), mm[:, :-1],
                        torch.ones_like(mm, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                features._tensor_check({**tensors, 'mm_token_type_ids': changed},
                                       receipt, 4)

    def test_nontext_modality_value_refuses_in_text_profile(self):
        with self.assertRaises(ValueError):
            features._batch_plain({
                'input_ids': torch.tensor([[1, 2]], dtype=torch.int64),
                'attention_mask': torch.tensor([[1, 1]], dtype=torch.int64),
                'mm_token_type_ids': torch.tensor([[0, 1]], dtype=torch.int64)})


class PreparedLoadPathTests(unittest.TestCase):
    @classmethod
    def j4_packet(cls):
        if not hasattr(cls, '_j4_packet'):
            torch.set_num_threads(1)
            fixture.CompleteSemanticTests.setUpClass()
            processor, bundle, kwargs, batch, alignment = fixture.CompleteSemanticTests().process()
            case = {'case_id': 'mock-loaded-j4', 'profile': 'j4-p1',
                    'caller': bundle['caller'], 'meaning': bundle['meaning'],
                    'source_kwargs': kwargs,
                    'frozen_batch': {k: v.tolist() for k, v in batch.items()},
                    'frozen_alignment': alignment}
            cls._j4_packet = (processor, case, alignment)
        return cls._j4_packet

    @staticmethod
    def prepared_fixture(root, processor):
        prepared = root / 'prepared'
        prepared.mkdir()
        config = {'model_type': 'gemma4_unified', 'text_config': {'hidden_size': 3840}}
        for name, data in [('model.safetensors', b'fixture-prepared-source'),
                           ('config.json', json.dumps(config).encode('utf-8')),
                           ('tokenizer.json', b'fixture-tokenizer')]:
            (prepared / name).write_bytes(data)
        files = [{'path': p.name, 'size': p.stat().st_size,
                  'sha256': sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(prepared.iterdir())]
        receipt = root / 'clone.json'
        receipt.write_bytes(b'{"fixture":"prepared"}')
        row = {'schema': shared.CLONE_SCHEMA, 'role': 'mfm',
               'receipt_sha256': 'c' * 64, 'parent_source_receipt_sha256': 'd' * 64,
               'snapshot_path': str(prepared.resolve()), 'files': files}
        loaded = TinyBase(3840).requires_grad_(False).eval()
        loaded.config.to_json_string = lambda use_diff: json.dumps(config)
        runtime = ModuleType('transformers')
        runtime.AutoProcessor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        runtime.AutoModelForMultimodalLM = SimpleNamespace(from_pretrained=Mock(return_value=loaded))
        return prepared, receipt, row, runtime, loaded

    @staticmethod
    def prepared_call(root, prepared, receipt, row, cases, j5_source=None):
        return features.export_prepared_bank(
            root, prepared_base_dir=prepared, cases=cases, j5_source=j5_source,
            prepared_receipt_path=receipt,
            expected_prepared_receipt_file_sha256=sha256(receipt.read_bytes()).hexdigest(),
            expected_prepared_receipt_sha256=row['receipt_sha256'],
            expected_parent_source_receipt_sha256=row['parent_source_receipt_sha256'],
            expected_cases_sha256=canonical_sha256(cases), device='cpu')

    def test_caller_supplied_base_cannot_claim_production(self):
        with self.assertRaises(ValueError):
            export_bank(Path('unused'), cases=[], processor=None, base=None,
                        j5_source=None, base_weight_path=Path('unused'),
                        base_weight_sha256='0' * 64, base_config_path=Path('unused'),
                        base_config_sha256='0' * 64, runtime={},
                        fixture_small_width=False)

    def test_private_j5_budget_change_refuses_before_model_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / 'prepared'
            prepared.mkdir()
            receipt = root / 'clone.json'
            receipt.write_bytes(b'{}')
            cases = [{'profile': 'j5-p2',
                      'source_kwargs': {'max_source_tokens': 38564}}]
            runtime = ModuleType('transformers')
            runtime.AutoProcessor = SimpleNamespace(from_pretrained=Mock())
            runtime.AutoModelForMultimodalLM = SimpleNamespace(from_pretrained=Mock())
            with patch.dict(sys.modules, {'transformers': runtime}):
                with self.assertRaises(ValueError):
                    features.export_prepared_bank(
                        root / 'bank', prepared_base_dir=prepared, cases=cases,
                        j5_source=None, prepared_receipt_path=receipt,
                        expected_prepared_receipt_file_sha256=sha256(receipt.read_bytes()).hexdigest(),
                        expected_prepared_receipt_sha256='c' * 64,
                        expected_parent_source_receipt_sha256='d' * 64,
                        expected_cases_sha256=canonical_sha256(cases), device='cpu')
            self.assertFalse(runtime.AutoProcessor.from_pretrained.called)
            self.assertFalse(runtime.AutoModelForMultimodalLM.from_pretrained.called)

    def test_actual_original_j5_overflow_refuses_before_model_load(self):
        torch.set_num_threads(1)
        fixture.CompleteSemanticTests.setUpClass()
        name = 'scripts.mfm.formation_joint_input_v5'
        path = Path(__file__).resolve().parents[3] / 'mfm-joint-original-inventory-roundtrip-design-20261011' / 'construction-001' / 'formation_joint_input_v5.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        class LongProcessor(Processor):
            def __init__(self):
                super().__init__()
                self.source_calls = 0
                self.offset_calls = 0

            def apply_chat_template(self, messages, *, chat_template, tokenize, **kwargs):
                if not tokenize:
                    return super().apply_chat_template(messages, chat_template=chat_template,
                                                       tokenize=False, **kwargs)
                if kwargs.get('processor_kwargs', {}).get('return_offsets_mapping'):
                    self.offset_calls += 1
                    raise AssertionError('overflow must precede offsets')
                self.source_calls += 1
                return {'input_ids': torch.zeros((1, 38564), dtype=torch.int64),
                        'attention_mask': torch.ones((1, 38564), dtype=torch.int64)}

        processor = LongProcessor()
        kwargs = fixture.CompleteSemanticTests().arguments(processor)
        policy = module.normative_joint_render_policy_v5()
        kwargs.update(render_policy=policy, render_policy_sha256=policy['record_sha256'],
                      max_source_tokens=32768)
        case = {'case_id': 'retained-overflow', 'profile': 'j5-p2',
                'caller': fixture.CompleteSemanticTests.bundle['caller'],
                'meaning': fixture.CompleteSemanticTests.bundle['meaning'],
                'source_kwargs': kwargs,
                'frozen_batch': {'input_ids': [[0]], 'attention_mask': [[1]]},
                'frozen_alignment': {'schema': module.SCHEMA}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, receipt, row, runtime, loaded = self.prepared_fixture(root, processor)
            weight_sha = next(x['sha256'] for x in row['files'] if x['path'] == 'model.safetensors')
            with patch.object(shared, '_prepared_base', return_value=row), \
                 patch.object(shared, 'SOURCE_WEIGHT_SHA256', weight_sha), \
                 patch.dict(sys.modules, {'transformers': runtime}):
                with self.assertRaisesRegex(Exception, 'complete source budget overflow'):
                    self.prepared_call(root / 'bank', prepared, receipt, row, [case], module)
            self.assertEqual(processor.source_calls, 1)
            self.assertEqual(processor.offset_calls, 0)
            self.assertEqual(runtime.AutoProcessor.from_pretrained.call_count, 1)
            self.assertEqual(runtime.AutoModelForMultimodalLM.from_pretrained.call_count, 0)
            self.assertEqual(loaded.model.calls, 0)
            self.assertFalse((root / 'bank').exists())

    def test_output_dotdot_alias_into_prepared_refuses_before_loading_or_writing(self):
        processor, case, _ = self.j4_packet()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, receipt, row, runtime, _ = self.prepared_fixture(root, processor)
            (root / 'other').mkdir()
            aliases = [root / 'other' / '..' / 'prepared' / 'invalid-bank']
            link = root / 'prepared-link'
            try:
                link.symlink_to(prepared, target_is_directory=True)
            except (OSError, NotImplementedError):
                pass
            else:
                aliases.append(link / 'invalid-bank')
            before = {p.relative_to(prepared).as_posix() for p in prepared.rglob('*')}
            weight_sha = next(x['sha256'] for x in row['files'] if x['path'] == 'model.safetensors')
            def allocate(path, **kwargs):
                Path(path).mkdir()
                (Path(path) / 'marker.txt').write_bytes(b'path-only witness')
                return {'record_sha256': 'e' * 64}
            with patch.dict(sys.modules, {'transformers': runtime}), \
                 patch.object(shared, '_prepared_base', return_value=row), \
                 patch.object(shared, 'SOURCE_WEIGHT_SHA256', weight_sha), \
                 patch.object(features, '_export_bank', side_effect=allocate) as allocate_call:
                for alias in aliases:
                    with self.subTest(alias=str(alias)), self.assertRaises(ValueError):
                        self.prepared_call(alias, prepared, receipt, row, [case])
            self.assertEqual({p.relative_to(prepared).as_posix() for p in prepared.rglob('*')},
                             before)
            self.assertFalse(runtime.AutoProcessor.from_pretrained.called)
            self.assertFalse(runtime.AutoModelForMultimodalLM.from_pretrained.called)
            self.assertFalse(allocate_call.called)

    def test_complete_j4_packet_tamper_refuses_before_model_load(self):
        processor, case, _ = self.j4_packet()
        changed = deepcopy(case)
        changed['frozen_batch']['attention_mask'][0][-1] = 0
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, receipt, row, runtime, loaded = self.prepared_fixture(root, processor)
            weight_sha = next(x['sha256'] for x in row['files'] if x['path'] == 'model.safetensors')
            with patch.dict(sys.modules, {'transformers': runtime}), \
                 patch.object(shared, '_prepared_base', return_value=row), \
                 patch.object(shared, 'SOURCE_WEIGHT_SHA256', weight_sha):
                with self.assertRaises(ValueError):
                    self.prepared_call(root / 'bank', prepared, receipt, row, [changed])
            self.assertEqual(runtime.AutoProcessor.from_pretrained.call_count, 1)
            self.assertEqual(runtime.AutoModelForMultimodalLM.from_pretrained.call_count, 0)
            self.assertEqual(loaded.model.calls, 0)
            self.assertFalse((root / 'bank').exists())

    def test_local_clone_verifier_brackets_actual_loader(self):
        processor, case, _ = self.j4_packet()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / 'prepared'
            prepared.mkdir()
            files = []
            for name, data in [('model.safetensors', b'untrained fixture'),
                               ('config.json', b'{"model_type":"gemma4_unified"}'),
                               ('tokenizer.json', b'{"fixture":true}')]:
                path = prepared / name
                path.write_bytes(data)
                files.append({'path': name, 'size': len(data),
                              'sha256': sha256(data).hexdigest()})
            receipt_path = root / 'clone-receipt.json'
            receipt_path.write_bytes(b'{"fixture":"prepared"}')
            receipt_file_sha = sha256(receipt_path.read_bytes()).hexdigest()
            parent_sha = 'b' * 64
            receipt_sha = 'c' * 64
            row = {'schema': shared.CLONE_SCHEMA, 'role': 'mfm',
                   'receipt_sha256': receipt_sha,
                   'parent_source_receipt_sha256': parent_sha,
                   'snapshot_path': str(prepared.resolve()), 'files': files}
            loaded = TinyBase().eval()
            runtime = ModuleType('transformers')
            runtime.AutoProcessor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
            runtime.AutoModelForMultimodalLM = SimpleNamespace(from_pretrained=Mock(return_value=loaded))
            def fake_export(path, **kwargs):
                Path(path).mkdir()
                return {'record_sha256': 'e' * 64}
            with patch.object(shared, '_prepared_base', return_value=row) as verify, \
                 patch.object(shared, 'SOURCE_WEIGHT_SHA256', files[0]['sha256']), \
                 patch.dict(sys.modules, {'transformers': runtime}), \
                 patch.object(features, '_export_bank', side_effect=fake_export) as export:
                result = features.export_prepared_bank(
                    root / 'bank', prepared_base_dir=prepared, cases=[case],
                    j5_source=None, prepared_receipt_path=receipt_path,
                    expected_prepared_receipt_file_sha256=receipt_file_sha,
                    expected_prepared_receipt_sha256=receipt_sha,
                    expected_parent_source_receipt_sha256=parent_sha,
                    expected_cases_sha256=canonical_sha256([case]),
                    device='cpu')
                self.assertEqual(result['record_sha256'], 'e' * 64)
                self.assertEqual(verify.call_count, 3)
                self.assertTrue(runtime.AutoProcessor.from_pretrained.call_args.kwargs['local_files_only'])
                self.assertTrue(runtime.AutoModelForMultimodalLM.from_pretrained.call_args.kwargs['local_files_only'])
                self.assertEqual(runtime.AutoModelForMultimodalLM.from_pretrained.call_args.args,
                                 (str(prepared.resolve()),))
                self.assertEqual(export.call_args.kwargs['prepared_binding']['prepared_files'], files)
                self.assertFalse(export.call_args.kwargs['fixture_small_width'])
                self.assertTrue((root / 'bank').is_dir())
            changed = {**row, 'parent_source_receipt_sha256': 'f' * 64}
            with patch.object(shared, '_prepared_base', side_effect=[row, changed]) as verify, \
                 patch.object(shared, 'SOURCE_WEIGHT_SHA256', files[0]['sha256']), \
                 patch.dict(sys.modules, {'transformers': runtime}), \
                 patch.object(features, '_export_bank', side_effect=fake_export):
                with self.assertRaises(ValueError):
                    features.export_prepared_bank(
                        root / 'other-bank', prepared_base_dir=prepared,
                        cases=[case], j5_source=None,
                        prepared_receipt_path=receipt_path,
                        expected_prepared_receipt_file_sha256=receipt_file_sha,
                        expected_prepared_receipt_sha256=receipt_sha,
                        expected_parent_source_receipt_sha256=parent_sha,
                        expected_cases_sha256=canonical_sha256([case]),
                        device='cpu')
                self.assertEqual(verify.call_count, 2)
                self.assertFalse((root / 'other-bank').exists())

    def test_mock_loaded_clone_writes_readable_original_j4_bank(self):
        processor, case, alignment = self.j4_packet()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / 'prepared'
            prepared.mkdir()
            config_record = {'model_type': 'gemma4_unified',
                             'text_config': {'hidden_size': 3840}}
            (prepared / 'model.safetensors').write_bytes(b'mock local prepared source')
            (prepared / 'config.json').write_text(json.dumps(config_record), encoding='utf-8')
            (prepared / 'tokenizer.json').write_bytes(b'fixture-tokenizer')
            files = [{'path': p.name, 'size': p.stat().st_size,
                      'sha256': sha256(p.read_bytes()).hexdigest()}
                     for p in sorted(prepared.iterdir())]
            receipt = root / 'clone.json'
            receipt.write_bytes(b'{"mock":"verified-by-fixture"}')
            receipt_file_sha = sha256(receipt.read_bytes()).hexdigest()
            row = {'schema': shared.CLONE_SCHEMA, 'role': 'mfm',
                   'receipt_sha256': 'c' * 64,
                   'parent_source_receipt_sha256': 'd' * 64,
                   'snapshot_path': str(prepared.resolve()), 'files': files}
            loaded = TinyBase(3840).eval().requires_grad_(False)
            loaded.config.to_json_string = lambda use_diff: json.dumps(config_record)
            runtime = ModuleType('transformers')
            runtime.AutoProcessor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
            runtime.AutoModelForMultimodalLM = SimpleNamespace(from_pretrained=Mock(return_value=loaded))
            with patch.object(shared, '_prepared_base', return_value=row) as verify, \
                 patch.object(shared, 'SOURCE_WEIGHT_SHA256',
                              next(x['sha256'] for x in files if x['path'] == 'model.safetensors')), \
                 patch.dict(sys.modules, {'transformers': runtime}):
                roster = features.export_prepared_bank(
                    root / 'bank', prepared_base_dir=prepared, cases=[case],
                    j5_source=None, prepared_receipt_path=receipt,
                    expected_prepared_receipt_file_sha256=receipt_file_sha,
                    expected_prepared_receipt_sha256=row['receipt_sha256'],
                    expected_parent_source_receipt_sha256=row['parent_source_receipt_sha256'],
                    expected_cases_sha256=canonical_sha256([case]),
                    device='cpu')
            self.assertEqual(verify.call_count, 3)
            self.assertEqual(loaded.model.calls, 1)
            feature, tensors = read_case(
                root / 'bank', roster_sha256=roster['record_sha256'],
                case_id='mock-loaded-j4', expected_alignment_sha256=alignment['record_sha256'],
                expected_cache=alignment['cache'])
            self.assertFalse(roster['base']['fixture_small_width'])
            self.assertEqual(feature['base']['prepared_base']['receipt_sha256'], row['receipt_sha256'])
            self.assertEqual(tensors['states'].shape,
                             (1, len(alignment['input_ids']), 3840))


class J6SourceFeatureTests(NativeTextFieldTests):
    """Actual saved geometry through a tiny frozen backbone, no capacity claim."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from scripts.mfm import formation_joint_input_v6 as j6
        cls.j6 = j6
        name = 'scripts.mfm.formation_joint_input_v5'
        path = Path(__file__).resolve().parents[3] / 'mfm-joint-original-inventory-roundtrip-design-20261011/construction-001/formation_joint_input_v5.py'
        spec = importlib.util.spec_from_file_location(name, path)
        cls.j5 = importlib.util.module_from_spec(spec)
        sys.modules[name] = cls.j5
        spec.loader.exec_module(cls.j5)
        cls.j5_sha = sha256(path.read_bytes()).hexdigest()
        original = fixture.CompleteSemanticTests()
        bundle = original.bundle
        kw = original.arguments(cls.processor)
        batch, alignment = j6.joint_source_batch_with_alignment_v6(
            cls.j5, cls.processor, bundle['caller'], bundle['meaning'],
            caller_sha256=kw['caller_sha256'], meaning_sha256=kw['meaning_sha256'],
            render_policy=cls.j5.normative_joint_render_policy_v5(),
            processor_binding=kw['processor_binding'], source_identity=kw['source_identity'],
            proposed_max_source_tokens=200000, j5_file_sha256=cls.j5_sha)
        cls.frozen = j6.freeze_source(cls.j5, bundle['caller'], bundle['meaning'],
            alignment, {k: v.tolist() for k, v in batch.items()},
            j5_file_sha256=cls.j5_sha, proposed_max_source_tokens=200000,
            cut_id='fixture-cut', expected_cut_sha256=bundle['meaning']['episode_snapshot']['historical_prefix_proof']['cut']['record_sha256'])

    def setUp(self):
        super().setUp()
        self.source_path = self.root / 'source.json'
        self.source_path.write_bytes(canonical_json_bytes(self.frozen) + b'\n')
        self.j6_case = dict(case_id='j6-source', profile='j6-source-only',
            source_file=str(self.source_path), source_file_sha256=sha256(self.source_path.read_bytes()).hexdigest(),
            source_record_sha256=self.frozen['record_sha256'], j5_file_sha256=self.j5_sha,
            proposed_max_source_tokens=200000)

    def export_j6(self):
        return export_bank(self.root / 'j6-bank', cases=[self.j6_case],
            processor=self.processor, base=self.base, j5_source=self.j5,
            base_weight_path=self.weight, base_weight_sha256=sha256(self.weight.read_bytes()).hexdigest(),
            base_config_path=self.config, base_config_sha256=sha256(self.config.read_bytes()).hexdigest(),
            runtime={'torch_version': torch.__version__, 'transformers_version': 'fixture'},
            fixture_small_width=True)

    def read_j6(self, roster, **extra):
        return read_case(self.root / 'j6-bank', roster_sha256=roster['record_sha256'],
            case_id='j6-source', expected_alignment_sha256=self.frozen['alignment']['record_sha256'],
            expected_cache=self.frozen['alignment']['cache'], j5_source=self.j5,
            expected_j6_source=self.j6_case, **extra)

    def test_j6_complete_native_backbone_and_physical_readback(self):
        roster = self.export_j6()
        row, tensors = self.read_j6(roster, allow_fixture=True)
        self.assertEqual(set(tensors), {'states', *self.j6.NATIVE_FIELDS})
        self.assertEqual(set(self.base.model.native_inputs), {'mm_token_type_ids'})
        for key in self.j6.NATIVE_FIELDS:
            self.assertTrue(torch.equal(tensors[key], torch.tensor(self.frozen['native_batch'][key])))
        self.assertTrue(torch.equal(tensors['states'], self.base.model.embedding(tensors['input_ids'])))
        self.assertFalse(row['j6_source']['capacity_admitted'])
        self.assertEqual(row['j6_source']['source_file_sha256'], self.j6_case['source_file_sha256'])
        with self.assertRaisesRegex(ValueError, 'fixture representation'):
            self.read_j6(roster)
        self.source_path.write_bytes(self.source_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'physical J6 source'):
            self.read_j6(roster, allow_fixture=True)

    def test_j6_changed_original_refuses_before_backbone(self):
        self.source_path.write_bytes(self.source_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'physical J6 source'):
            self.export_j6()
        self.assertEqual(self.base.model.calls, 0)
        self.assertFalse((self.root / 'j6-bank').exists())

    def test_j6_target_operand_and_wrong_processor_refuse_before_backbone(self):
        self.j6_case['target'] = {'answer': 'not a source operand'}
        with self.assertRaisesRegex(ValueError, 'target-free J6'):
            self.export_j6()
        del self.j6_case['target']
        with patch.object(features.source, 'processor_binding', return_value={}):
            with self.assertRaisesRegex(ValueError, 'actual J6 processor binding'):
                self.export_j6()
        self.assertEqual(self.base.model.calls, 0)
        self.assertFalse((self.root / 'j6-bank').exists())

    def test_j6_observation_change_during_forward_prevents_bank_publication(self):
        original = self.base.model.forward
        def mutate(**kwargs):
            result = original(**kwargs)
            self.source_path.write_bytes(self.source_path.read_bytes() + b' ')
            return result
        with patch.object(self.base.model, 'forward', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'physical J6 source changed'):
                self.export_j6()
        self.assertEqual(self.base.model.calls, 1)
        self.assertFalse((self.root / 'j6-bank' / 'roster.json').exists())


if __name__ == '__main__':
    unittest.main()
