"""Real custody/CPU tensor tests; a tiny backbone does not prove MFM learning."""
from hashlib import sha256
from contextlib import redirect_stdout
import gc
import importlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from scripts.mfm import preflight_public_conditional_material as census_code
from scripts.mfm import train_v1_formation_specialist as shared


class TinyRepresentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 3, dtype=torch.bfloat16)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(48).reshape(16, 3))
        self.grad_seen = None
        self.forward_calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        if use_cache is not False or return_dict is not True:
            raise AssertionError('source representation call changed')
        if not torch.equal(attention_mask, torch.ones_like(input_ids)):
            raise AssertionError('complete source mask changed')
        self.grad_seen = torch.is_grad_enabled()
        self.forward_calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyRepresentation()
        self.config = SimpleNamespace(model_type='gemma4_unified',
                                      text_config=SimpleNamespace(hidden_size=3, vocab_size=16))

    def forward(self, **kwargs):
        raise AssertionError('language head must never be called')


class PublicConditionalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.console = StringIO()
        self.redirect = redirect_stdout(self.console)
        self.redirect.__enter__()
        self.root = Path(self.tmp.name)
        self.census_dir = self.root / 'census'
        self.census_dir.mkdir()
        self.output = self.root / 'features'
        self.sources = [self.source([2, 4], 'case-a'), self.source([2, 5], 'case-b')]
        self.prepared = {'receipt_sha256': 'c' * 64, 'schema': shared.CLONE_SCHEMA,
                         'role': 'mfm', 'snapshot_path': str(self.root / 'base'),
                         'files': [{'path': 'model.safetensors', 'sha256': 'd' * 64}]}
        self.census = {'schema': census_code.SCHEMA, 'task': census_code.TASK,
                      'source_inputs_sha256': '', 'unique_source_inputs': 2,
                      'foundation_vocabulary': 16, 'base_hidden_size': 3,
                      'prepared_clone_receipt_sha256': 'c' * 64,
                      'prepared_base_weight_sha256': 'd' * 64,
                      'material_receipt_sha256': 'a' * 64,
                      'material_receipt_file_sha256': 'b' * 64,
                      'source_manifest_sha256': 'e' * 64,
                      'processor_template_sha256': sha256(shared.SOURCE_TEMPLATE.encode()).hexdigest(),
                      'source_processor_helper_sha256': shared._digest(Path(shared.__file__)),
                      'code_sha256': shared._digest(Path(census_code.__file__)),
                      'torch_version': torch.__version__, 'transformers_version': 'fixture',
                      'max_source_tokens': 8, 'statistics': {'rows': 2, 'split_counts': {'train': 2}},
                      'private_evidence_opened': False, 'model_weights_loaded': False,
                      'optimizer_created': False, 'authority_written': False,
                      'qualified_for_product': False, 'final_payloads_opened': False}
        self.write_census()

    def tearDown(self):
        self.redirect.__exit__(None, None, None)
        self.tmp.cleanup()

    def interface(self):
        name = 'scripts.mfm.export_public_conditional_features'
        self.assertIsNotNone(importlib.util.find_spec(name), 'public source feature exporter is missing')
        return importlib.import_module(name)

    def source(self, ids, case_id):
        return {'schema': 'mfm-public-conditional-source-input-v1', 'task': census_code.TASK,
                'source_key': canonical_sha256({'task': census_code.TASK, 'source_ids': ids}),
                'source_ids': ids, 'attention_mask': [1] * len(ids), 'split': 'train',
                'case_ids': [case_id], 'semantic_input_sha256s': ['f' * 64],
                'source_payload_sha256s': ['1' * 64]}

    def write_census(self):
        path = self.census_dir / 'source-inputs.jsonl'
        path.write_bytes(b''.join(canonical_json_bytes(row) + b'\n' for row in self.sources))
        self.census['source_inputs_sha256'] = shared._digest(path)
        sealed = shared._seal(self.census)
        (self.census_dir / 'receipt.json').write_bytes(canonical_json_bytes(sealed) + b'\n')
        self.external_pin = sealed['record_sha256']

    def binding(self, module):
        census, sources = module.read_census(self.census_dir, self.external_pin)
        export = module.prepare_export(self.output, census, self.prepared,
                                       torch_version=torch.__version__, transformers_version='fixture',
                                       resume=self.output.exists())
        return census, sources, export

    def child(self, mode):
        # Kill a real worker after durable tensor/record write but before atomic
        # publication. os._exit bypasses finally blocks and releases OS locks.
        script = '''
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests' / 'governance'))
from test_mfm_public_conditional_features import TinyBase
from scripts.mfm import export_public_conditional_features as module
from safetensors import torch as serialization
root, census_dir, pin, mode = sys.argv[1:]
census, sources = module.read_census(Path(census_dir), pin)
export = module._read_sealed(Path(root) / 'export.json', module.EXPORT_SCHEMA)
if mode == 'tensor':
    save = serialization.save_file
    def interrupted_save(tensors, filename):
        save(tensors, filename)
        os._exit(73)
    serialization.save_file = interrupted_save
elif mode in ('record', 'quarantine'):
    replace = module.os.replace
    def interrupted_replace(source, target):
        if Path(target).name == ('progress.json' if mode == 'record' else 'quarantine.json'):
            os._exit(74 if mode == 'record' else 75)
        return replace(source, target)
    module.os.replace = interrupted_replace
module.run_export(Path(root), census, sources, export, TinyBase(), stop_after_seconds=60)
'''
        return subprocess.run([sys.executable, '-c', script, str(self.output),
                               str(self.census_dir), self.external_pin, mode],
                              cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, timeout=60)

    def test_source_census_keeps_exact_encodings_in_one_semantic_group(self):
        module = self.interface()
        census, sources = module.read_census(self.census_dir, self.external_pin)
        self.assertEqual(census['unique_source_inputs'], 2)
        self.assertEqual([row['source_ids'] for row in sources], [[2, 4], [2, 5]])
        self.assertNotEqual(sources[0]['source_key'], sources[1]['source_key'])

    def test_external_census_pin_and_source_file_bytes_are_required(self):
        module = self.interface()
        with self.assertRaisesRegex(ValueError, 'external.*census'):
            module.read_census(self.census_dir, '0' * 64)
        with (self.census_dir / 'source-inputs.jsonl').open('ab') as stream:
            stream.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'source.*digest'):
            module.read_census(self.census_dir, self.external_pin)

    def test_unsealed_census_mutation_is_rejected(self):
        module = self.interface()
        receipt = json.loads((self.census_dir / 'receipt.json').read_text())
        receipt['foundation_vocabulary'] = 17
        (self.census_dir / 'receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'seal'):
            module.read_census(self.census_dir, self.external_pin)

    def test_target_fields_are_rejected_even_under_valid_external_seal(self):
        module = self.interface()
        for field in ('labels', 'decoder_inputs', 'target_text', 'gold', 'targets'):
            with self.subTest(field=field):
                self.sources[0][field] = [3]
                self.write_census()
                with self.assertRaisesRegex(ValueError, 'source-only.*fields'):
                    module.read_census(self.census_dir, self.external_pin)
                del self.sources[0][field]

    def test_invalid_mask_vocabulary_and_key_rejected(self):
        module = self.interface()
        for field, value in (('attention_mask', [1, 0]), ('source_ids', [2, 16]),
                             ('source_key', '9' * 64), ('source_ids', [True, 4])):
            with self.subTest(field=field, value=value):
                original = self.sources[0][field]
                self.sources[0][field] = value
                self.write_census()
                with self.assertRaises(ValueError):
                    module.read_census(self.census_dir, self.external_pin)
                self.sources[0][field] = original

    def test_duplicate_source_case_and_incomplete_roster_rejected(self):
        module = self.interface()
        original = self.sources[1]
        self.sources[1] = self.sources[0]
        self.write_census()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.read_census(self.census_dir, self.external_pin)
        self.sources[1] = original
        self.sources[1]['case_ids'] = ['case-a']
        self.write_census()
        with self.assertRaisesRegex(ValueError, 'case'):
            module.read_census(self.census_dir, self.external_pin)
        self.sources[1]['case_ids'] = ['case-b']
        self.census['statistics']['rows'] = 3
        self.write_census()
        with self.assertRaisesRegex(ValueError, 'roster'):
            module.read_census(self.census_dir, self.external_pin)

    def test_final_split_or_private_census_is_rejected(self):
        module = self.interface()
        self.sources[0]['split'] = 'FINAL'
        self.write_census()
        with self.assertRaisesRegex(ValueError, 'split'):
            module.read_census(self.census_dir, self.external_pin)
        self.sources[0]['split'] = 'train'
        self.census['private_evidence_opened'] = True
        self.write_census()
        with self.assertRaisesRegex(ValueError, 'public'):
            module.read_census(self.census_dir, self.external_pin)

    def test_clone_receipt_and_weight_must_match_census(self):
        module = self.interface()
        census, _ = module.read_census(self.census_dir, self.external_pin)
        self.prepared['receipt_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'clone'):
            module.prepare_export(self.output, census, self.prepared,
                                  torch_version=torch.__version__, transformers_version='fixture')
        self.assertFalse(self.output.exists())
        self.prepared['receipt_sha256'] = 'c' * 64
        self.prepared['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'weight'):
            module.prepare_export(self.output, census, self.prepared,
                                  torch_version=torch.__version__, transformers_version='fixture')

    def test_real_bf16_cpu_representations_are_lossless_and_bank_complete(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        base = TinyBase()
        result = module.run_export(self.output, census, sources, export, base,
                                   stop_after_seconds=60)
        self.assertTrue(result['complete'])
        self.assertFalse(base.model.grad_seen)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        tensors = load_file(str(self.output / 'sources' / sources[0]['source_key'] / 'features.safetensors'))
        self.assertEqual(set(tensors), {'states', 'input_ids', 'attention_mask'})
        self.assertTrue(torch.equal(tensors['states'], torch.tensor([[[6, 7, 8], [12, 13, 14]]], dtype=torch.bfloat16)))
        bank = module.verify_bank(self.output, census, sources, export)
        self.assertEqual(bank['record_sha256'], result['public_conditional_feature_bank_sha256'])
        self.assertEqual(len(bank['sources']), 2)

    def test_time_budget_stops_only_between_complete_inputs_and_resume_finishes(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        ticks = iter([0., 0., 2., 2.])
        result = module.run_export(self.output, census, sources, export, TinyBase(),
                                   stop_after_seconds=1, clock=lambda: next(ticks))
        self.assertFalse(result['complete'])
        self.assertEqual(result['completed'], 1)
        self.assertFalse((self.output / 'bank.json').exists())
        self.assertEqual(len(list((self.output / 'sources').iterdir())), 1)
        census, sources, resumed_export = self.binding(module)
        self.assertEqual(resumed_export, export)
        result = module.run_export(self.output, census, sources, resumed_export, TinyBase(),
                                   stop_after_seconds=60)
        self.assertTrue(result['complete'])

    def test_resume_rejects_tampered_tensor_before_skipping(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        ticks = iter([0., 0., 2., 2.])
        module.run_export(self.output, census, sources, export, TinyBase(),
                          stop_after_seconds=1, clock=lambda: next(ticks))
        path = self.output / 'sources' / sources[0]['source_key'] / 'features.safetensors'
        with path.open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'feature.*digest'):
            module.run_export(self.output, census, sources, export, TinyBase(), stop_after_seconds=60)
        self.assertFalse((self.output / 'sources' / sources[1]['source_key']).exists())

    def rewrite_owned_source_tensors(self, module, replacements):
        """Keep the corruption fixture independent of the original mmap inode."""
        census, sources, export = self.binding(module)
        module.run_export(self.output, census, sources, export, TinyBase(), stop_after_seconds=60)
        directory = self.output / 'sources' / sources[0]['source_key']
        tensor_path = directory / 'features.safetensors'
        mapped = load_file(str(tensor_path))
        owned = {name: tensor.clone() for name, tensor in mapped.items()}
        del mapped
        gc.collect()
        # v0.8 save_file replaces the destination inode. Retaining mapped
        # tensors during replacement creates .nfs files on the actual runtime,
        # so receipt-directory rejection would hide the tensor claim under test.
        self.assertEqual({p.name for p in directory.iterdir()}, {'source.json', 'features.safetensors'})
        owned.update(replacements)
        save_file(owned, str(tensor_path))
        row = json.loads((directory / 'source.json').read_text())
        row.pop('record_sha256')
        row['features_sha256'] = shared._digest(tensor_path)
        (directory / 'source.json').write_bytes(canonical_json_bytes(shared._seal(row)))
        receipt = module._read_sealed(directory / 'source.json', module.FEATURE_SCHEMA)
        # Establish every preceding custody predicate before claiming a tensor
        # guard was exercised. These checks intentionally retain the strict
        # production directory and digest contracts.
        self.assertEqual({p.name for p in directory.iterdir()}, {'source.json', 'features.safetensors'})
        self.assertEqual(receipt['features_sha256'], shared._digest(tensor_path))
        self.assertEqual(receipt['source_input'], sources[0])
        self.assertEqual(receipt['export_sha256'], export['record_sha256'])
        self.assertIs(receipt['serialized_tensors_exact'], True)
        self.assertIs(receipt['target_visible_to_backbone'], False)
        return census, sources, export

    def assert_source_and_bank_tensor_guard(self, module, census, sources, export, pattern):
        readers = [('source', lambda: module.read_source(self.output, sources[0],
                                                        export['record_sha256'], census['base_hidden_size'])),
                   ('bank', lambda: module.verify_bank(self.output, census, sources, export))]
        for name, read in readers:
            with self.subTest(reader=name):
                try:
                    read()
                except ValueError as error:
                    self.assertRegex(str(error), pattern)
                    # Inspect the real exception path rather than replacing or
                    # mocking the validator. An earlier custody rejection must
                    # never count as evidence for this semantic guard.
                    frames = traceback.extract_tb(error.__traceback__)
                    self.assertIn('_validate_tensors', [frame.name for frame in frames])
                else:
                    self.fail('tampered tensor was accepted by ' + name)

    def test_resealed_wrong_input_ids_reach_tensor_guard_and_reject_bank(self):
        module = self.interface()
        census, sources, export = self.rewrite_owned_source_tensors(
            module, {'input_ids': torch.tensor([[2, 5]], dtype=torch.int64)})
        self.assert_source_and_bank_tensor_guard(module, census, sources, export, 'input.*differs')

    def test_resealed_wrong_mask_reaches_tensor_guard_and_rejects_bank(self):
        module = self.interface()
        census, sources, export = self.rewrite_owned_source_tensors(
            module, {'attention_mask': torch.tensor([[1, 0]], dtype=torch.int64)})
        self.assert_source_and_bank_tensor_guard(module, census, sources, export, 'mask.*differs')

    def test_resealed_wrong_state_geometry_reaches_tensor_guard_and_rejects_bank(self):
        module = self.interface()
        census, sources, export = self.rewrite_owned_source_tensors(
            module, {'states': torch.zeros((1, 2, 4), dtype=torch.bfloat16)})
        self.assert_source_and_bank_tensor_guard(module, census, sources, export, 'geometry differs')

    def test_resealed_wrong_state_dtype_reaches_tensor_guard_and_rejects_bank(self):
        module = self.interface()
        census, sources, export = self.rewrite_owned_source_tensors(
            module, {'states': torch.tensor([[[6, 7, 8], [12, 13, 14]]], dtype=torch.float32)})
        self.assert_source_and_bank_tensor_guard(module, census, sources, export, 'dtype.*differs')

    def test_resealed_nonfinite_states_reach_tensor_guard_and_reject_bank(self):
        module = self.interface()
        census, sources, export = self.rewrite_owned_source_tensors(
            module, {'states': torch.tensor([[[float('nan'), 7, 8], [12, 13, 14]]], dtype=torch.bfloat16)})
        self.assert_source_and_bank_tensor_guard(module, census, sources, export, 'finite.*differs')

    def test_valid_owned_clone_rewrite_preserves_source_and_complete_bank(self):
        module = self.interface()
        census, sources, export = self.rewrite_owned_source_tensors(module, {})
        receipt, tensors = module.read_source(self.output, sources[0], export['record_sha256'], 3)
        self.assertTrue(torch.equal(tensors['states'], torch.tensor([[[6, 7, 8], [12, 13, 14]]], dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(tensors['input_ids'], torch.tensor([[2, 4]], dtype=torch.int64)))
        self.assertTrue(torch.equal(tensors['attention_mask'], torch.tensor([[1, 1]], dtype=torch.int64)))
        del tensors
        bank = module.verify_bank(self.output, census, sources, export)
        self.assertTrue(bank['complete'])
        self.assertEqual(bank['sources'][0]['source_receipt_sha256'], receipt['record_sha256'])

    def test_complete_bank_rejects_unlisted_artifact_and_receipt_tamper(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        module.run_export(self.output, census, sources, export, TinyBase(), stop_after_seconds=60)
        (self.output / 'sources' / ('0' * 64)).mkdir()
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            module.verify_bank(self.output, census, sources, export)
        (self.output / 'sources' / ('0' * 64)).rmdir()
        bank_path = self.output / 'bank.json'
        bank = json.loads(bank_path.read_text())
        bank['complete'] = False
        bank_path.write_text(json.dumps(bank))
        with self.assertRaisesRegex(ValueError, 'seal'):
            module.verify_bank(self.output, census, sources, export)

    def test_resume_rejects_changed_export_code_binding(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        path = self.output / 'export.json'
        row = dict(export); row.pop('record_sha256')
        row['exporter_sha256'] = '0' * 64
        path.write_bytes(canonical_json_bytes(shared._seal(row)))
        with self.assertRaisesRegex(ValueError, 'export.*binding'):
            self.binding(module)

    def test_backbone_geometry_mismatch_does_not_publish_source(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        base = TinyBase(); base.config.text_config.hidden_size = 4
        with self.assertRaisesRegex(ValueError, 'geometry'):
            module.run_export(self.output, census, sources, export, base, stop_after_seconds=60)
        self.assertFalse((self.output / 'sources').exists())

    def test_consumer_cannot_finalize_a_subset_of_the_census_roster(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        with self.assertRaisesRegex(ValueError, 'roster'):
            module.run_export(self.output, census, sources[:1], export, TinyBase(), stop_after_seconds=60)
        self.assertFalse((self.output / 'bank.json').exists())

    def test_progress_pin_has_bounded_size_and_complete_bank_binds_it(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        result = module.run_export(self.output, census, sources, export, TinyBase(), stop_after_seconds=60)
        progress = json.loads((self.output / 'progress.json').read_text())
        self.assertNotIn('sources', progress)
        self.assertEqual(progress['completed'], 2)
        self.assertTrue(progress['complete'])
        self.assertEqual(result['public_conditional_feature_progress_sha256'], progress['record_sha256'])
        bank = json.loads((self.output / 'bank.json').read_text())
        self.assertEqual(bank['progress_sha256'], progress['record_sha256'])
        lines = [json.loads(line) for line in self.console.getvalue().splitlines()]
        self.assertEqual(lines[-1]['public_conditional_feature_progress_sha256'], progress['record_sha256'])

    def test_resealed_progress_claim_cannot_skip_incomplete_source_custody(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        ticks = iter([0., 0., 2., 2.])
        module.run_export(self.output, census, sources, export, TinyBase(),
                          stop_after_seconds=1, clock=lambda: next(ticks))
        path = self.output / 'progress.json'
        progress = json.loads(path.read_text())
        progress.pop('record_sha256')
        progress.update(completed=2, complete=True)
        path.write_bytes(canonical_json_bytes(shared._seal(progress)))
        with self.assertRaisesRegex(ValueError, 'progress'):
            module.run_export(self.output, census, sources, export, TinyBase(), stop_after_seconds=60)
        self.assertFalse((self.output / 'sources' / sources[1]['source_key']).exists())

    def test_real_census_source_mapper_discards_targets_and_preserves_ancestry(self):
        module = self.interface()
        encoded = [dict(source_ids=[2, 4], split='train', case_id='case-a',
                        semantic_input_sha256='f' * 64, source_payload_sha256='1' * 64,
                        decoder_inputs=[2, 10], labels=[10, 3]),
                   dict(source_ids=[2, 4], split='train', case_id='case-b',
                        semantic_input_sha256='f' * 64, source_payload_sha256='2' * 64,
                        decoder_inputs=[2, 11], labels=[11, 3])]
        self.sources = census_code.source_inputs(encoded)
        self.census['unique_source_inputs'] = 1
        self.write_census()
        census, sources = module.read_census(self.census_dir, self.external_pin)
        self.assertEqual(sources[0]['case_ids'], ['case-a', 'case-b'])
        self.assertEqual(sources[0]['source_payload_sha256s'], ['1' * 64, '2' * 64])
        self.assertNotIn('labels', sources[0])
        self.assertEqual(census['statistics']['rows'], 2)

    def test_abrupt_tensor_worker_exit_preserves_scratch_and_regenerates_input(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        child = self.child('tensor')
        self.assertEqual(child.returncode, 73, child.stderr)
        staging = next((self.output / 'sources').glob('.staging-*'))
        scratch_hash = shared._digest(staging / 'features.safetensors')
        original_path = staging.relative_to(self.output).as_posix()
        base = TinyBase()
        result = module.run_export(self.output, census, sources, export, base, stop_after_seconds=60)
        self.assertTrue(result['complete'])
        self.assertEqual(base.model.forward_calls, 2)
        custody = module._read_sealed(self.output / 'quarantine.json', module.QUARANTINE_SCHEMA)
        self.assertEqual(custody['entries'][0]['original_path'], original_path)
        self.assertEqual(custody['entries'][0]['files'][0]['sha256'], scratch_hash)
        retained = self.output / custody['entries'][0]['quarantine_path'] / 'features.safetensors'
        self.assertEqual(shared._digest(retained), scratch_hash)
        module.verify_bank(self.output, census, sources, export)

    def test_abrupt_record_exit_preserves_committed_tensor_and_unpublished_record(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        child = self.child('record')
        self.assertEqual(child.returncode, 74, child.stderr)
        source_path = self.output / 'sources' / sources[0]['source_key'] / 'features.safetensors'
        committed_hash = shared._digest(source_path)
        scratch = next(self.output.glob('.record-*'))
        scratch_hash = shared._digest(scratch)
        base = TinyBase()
        result = module.run_export(self.output, census, sources, export, base, stop_after_seconds=60)
        self.assertTrue(result['complete'])
        self.assertEqual(base.model.forward_calls, 1)
        self.assertEqual(shared._digest(source_path), committed_hash)
        custody = module._read_sealed(self.output / 'quarantine.json', module.QUARANTINE_SCHEMA)
        entry = custody['entries'][0]
        self.assertEqual(entry['original_path'], scratch.name)
        self.assertEqual(shared._digest(self.output / entry['quarantine_path']), scratch_hash)

    def test_runtime_worker_lock_rejects_concurrent_export(self):
        module = self.interface()
        self.binding(module)
        with (self.output / 'worker.lock').open('w+b') as stream:
            stream.write(b'0'); stream.flush(); stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            child = self.child('normal')
        self.assertNotEqual(child.returncode, 0)
        self.assertRegex(child.stderr, 'worker.*already running')
        self.assertFalse((self.output / 'sources').exists())

    def run_cli(self, module, *, resume):
        base = TinyBase()

        def assert_locked():
            with self.assertRaisesRegex(ValueError, 'worker.*already running'):
                with module.worker_lock(self.output):
                    pass

        def prepared(args):
            assert_locked()
            return self.prepared

        def load(snapshot_path, **kwargs):
            assert_locked()
            self.assertEqual(snapshot_path, self.prepared['snapshot_path'])
            self.assertEqual(kwargs, {'trust_remote_code': False, 'local_files_only': True,
                                     'use_safetensors': True, 'dtype': torch.bfloat16,
                                     'device_map': {'': 'cpu'}, 'low_cpu_mem_usage': True})
            return base

        actual_read = module.read_census

        def read(*args):
            assert_locked()
            return actual_read(*args)

        fixture = SimpleNamespace(__version__='fixture',
                                  AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=load))
        argv = ['--census-dir', str(self.census_dir), '--census-sha256', self.external_pin,
                '--prepared-base-dir', self.prepared['snapshot_path'],
                '--prepared-base-receipt', str(self.root / 'base-receipt.json'),
                '--output-dir', str(self.output), '--stop-after-seconds', '60']
        if resume:
            argv.append('--resume')
        # Substitute only unavailable heavyweight loading/clone custody. Actual
        # CLI receipt checks, lock, source forward, serialization and recovery run.
        with patch.object(shared, '_prepared_base', prepared), \
                patch.object(shared, '_require_private_network_isolation'), \
                patch.object(module, 'read_census', read), \
                patch.dict(sys.modules, {'transformers': fixture}):
            module.main(argv)
        return base

    def test_cli_resume_recovers_partial_bytes_before_inspection_and_model_load(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        staging = self.output / 'sources' / ('.staging-' + 'a' * 32)
        staging.mkdir(parents=True)
        partial = b'interrupted and unusable tensor bytes'
        (staging / 'features.safetensors').write_bytes(partial)
        base = self.run_cli(module, resume=True)
        self.assertEqual(base.model.forward_calls, 2)
        retained = self.output / 'quarantine' / ('source-' + 'a' * 32) / 'features.safetensors'
        self.assertEqual(retained.read_bytes(), partial)
        module.verify_bank(self.output, census, sources, export)

    def test_fresh_cli_locks_before_census_and_clone_verification(self):
        module = self.interface()
        base = self.run_cli(module, resume=False)
        self.assertEqual(base.model.forward_calls, 2)
        census, sources = module.read_census(self.census_dir, self.external_pin)
        export = module._read_sealed(self.output / 'export.json', module.EXPORT_SCHEMA)
        module.verify_bank(self.output, census, sources, export)

    def test_abrupt_quarantine_publication_exit_finishes_custody_without_promoting_bytes(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        staging = self.output / 'sources' / ('.staging-' + 'b' * 32)
        staging.mkdir(parents=True)
        partial = b'partial tensor remains diagnostic evidence'
        (staging / 'features.safetensors').write_bytes(partial)
        child = self.child('quarantine')
        self.assertEqual(child.returncode, 75, child.stderr)
        self.assertFalse((self.output / 'quarantine.json').exists())
        self.assertFalse((self.output / 'sources' / sources[0]['source_key']).exists())
        record = next(self.output.glob('.record-*'))
        record_hash = shared._digest(record)
        record_name = record.name
        base = TinyBase()
        result = module.run_export(self.output, census, sources, export, base, stop_after_seconds=60)
        self.assertTrue(result['complete'])
        self.assertEqual(base.model.forward_calls, 2)
        custody = module._read_sealed(self.output / 'quarantine.json', module.QUARANTINE_SCHEMA)
        self.assertEqual(len(custody['entries']), 2)
        entries = {entry['original_path']: entry for entry in custody['entries']}
        self.assertEqual(shared._digest(self.output / entries[record_name]['quarantine_path']), record_hash)
        retained = self.output / entries['sources/.staging-' + 'b' * 32]['quarantine_path'] / 'features.safetensors'
        self.assertEqual(retained.read_bytes(), partial)
        module.verify_bank(self.output, census, sources, export)

    def test_quarantined_byte_tamper_blocks_resume_before_model_forward(self):
        module = self.interface()
        census, sources, export = self.binding(module)
        staging = self.output / 'sources' / ('.staging-' + 'c' * 32)
        staging.mkdir(parents=True)
        (staging / 'features.safetensors').write_bytes(b'original partial bytes')
        ticks = iter([0., 1., 1.])
        result = module.run_export(self.output, census, sources, export, TinyBase(),
                                   stop_after_seconds=1, clock=lambda: next(ticks))
        self.assertFalse(result['complete'])
        retained = self.output / 'quarantine' / ('source-' + 'c' * 32) / 'features.safetensors'
        retained.write_bytes(b'changed retained bytes')
        base = TinyBase()
        with self.assertRaisesRegex(ValueError, 'quarantine.*digest'):
            module.run_export(self.output, census, sources, export, base, stop_after_seconds=60)
        self.assertEqual(base.model.forward_calls, 0)
        self.assertFalse((self.output / 'sources' / sources[0]['source_key']).exists())


if __name__ == '__main__':
    unittest.main()
