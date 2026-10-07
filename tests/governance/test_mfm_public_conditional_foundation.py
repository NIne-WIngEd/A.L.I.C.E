"""Tiny actual-model contracts; these fixtures prove no production resource fit."""
import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel.formation_conditional_task import bind_conditional_task
from scripts.mfm import train_v16_cached_specialist as cached
import test_mfm_public_conditional_training as fixtures


class PublicFoundationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = fixtures.PublicConditionalTrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.prepared = self.fixture.load()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(71)
        self.model = FormationDecisionSpecialist(SpecialistConfig(3, 16, 8, 1, 2, 16, 0, 2, 1, 2))
        self.view = bind_conditional_task(self.model, task='conditional_meaning_to_text')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.03)

    def module(self):
        name = 'scripts.mfm.train_public_conditional_foundation'
        self.assertIsNotNone(importlib.util.find_spec(name), 'public foundation trainer is missing')
        return importlib.import_module(name)

    def test_real_weighted_sequence_means_include_eos_and_reach_original_weights(self):
        m = self.module()
        from scripts.mfm.public_conditional_training import payload
        before = cached._cpu_tree(self.model.state_dict())
        with torch.no_grad():
            losses = [self.view.forward(**payload(self.prepared, p, device='cpu'))[1].item()
                      for p in self.prepared.train]
        metrics = m.optimizer_step(self.view, self.optimizer, self.prepared.train, prepared=self.prepared)
        self.assertAlmostEqual(metrics['objective_loss'], .25 * losses[0] + .25 * losses[1] + .5 * losses[2], places=6)
        self.assertEqual(metrics['label_tokens'], 9)
        self.assertGreater(metrics['gradient_norm'], 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in self.model.state_dict().items()))

    def test_development_forged_case_and_unbound_task_rejected_before_update(self):
        m = self.module()
        before = cached._cpu_tree(self.model.state_dict())
        for batch in ([self.prepared.development[0]], [dict(self.prepared.train[0], labels=[9, 1])]):
            with self.assertRaises(ValueError):
                m.optimizer_step(self.view, self.optimizer, batch, prepared=self.prepared)
        with self.assertRaisesRegex(ValueError, 'task'):
            m.optimizer_step(self.model, self.optimizer, self.prepared.train, prepared=self.prepared)
        cached.compare_tree(self.model.state_dict(), before, atol=0, rtol=0)
        self.assertFalse(self.optimizer.state)

    def test_roster_exhaustion_partial_batch_and_completed_epoch_cursor(self):
        m = self.module()
        cursor = dict(epoch=0, next_case=0, step=0)
        batches = []
        for _ in range(4):
            batch, cursor = m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=cursor)
            batches.append([p['case_id'] for p in batch])
        self.assertEqual(batches, [['c', 'b'], ['a'], ['c', 'a'], ['b']])
        self.assertEqual(cursor, dict(epoch=2, next_case=0, step=4))
        self.assertEqual(m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=cursor)[0], ())
        for bad in (dict(epoch=0, next_case=3, step=1), dict(epoch=1, next_case=0, step=1), dict(epoch=0, next_case=1, step=1)):
            with self.assertRaises(ValueError):
                m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=bad)

    def test_real_next_step_checkpoint_replay_different_partial_batch(self):
        m = self.module()
        binding = {'run': 'a' * 64, 'seed': 'b' * 64, 'epochs': 2, 'batch_size': 2, 'train_cases': 3}
        cursor = dict(epoch=0, next_case=0, step=0)
        batch, cursor = m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=cursor)
        m.optimizer_step(self.view, self.optimizer, batch, prepared=self.prepared)
        checkpoint = m.save_checkpoint(self.root, self.model, self.optimizer, binding=binding, cursor=cursor)
        next_pairs, next_cursor = m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=cursor)
        expected = m.optimizer_step(self.view, self.optimizer, next_pairs, prepared=self.prepared)
        expected_model = cached._cpu_tree(self.model.state_dict())
        expected_adam = cached._cpu_tree(self.optimizer.state_dict())
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(999)
        other = FormationDecisionSpecialist(self.model.config)
        adam = torch.optim.AdamW(other.parameters(), lr=.001, weight_decay=.03)
        restored = m.resume_checkpoint(checkpoint, self.root, other, adam, binding=binding)
        actual_pairs, actual_cursor = m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=restored)
        self.assertEqual([p['case_id'] for p in actual_pairs], ['a'])
        self.assertEqual(actual_cursor, next_cursor)
        actual = m.optimizer_step(bind_conditional_task(other, task='conditional_meaning_to_text'), adam, actual_pairs, prepared=self.prepared)
        self.assertEqual(actual, expected)
        cached.compare_tree(other.state_dict(), expected_model, atol=0, rtol=0)
        cached.compare_tree(adam.state_dict(), expected_adam, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        complete = m.save_checkpoint(self.root, other, adam, binding=binding, cursor=actual_cursor)
        self.assertEqual(m.resume_checkpoint(complete, self.root, other, adam, binding=binding), dict(epoch=1, next_case=0, step=2))

    def test_checkpoint_rejects_changed_binding_foreign_partial_stale_and_overwrite(self):
        m = self.module()
        binding = {'run': 'a' * 64, 'seed': 'b' * 64, 'epochs': 2, 'batch_size': 2, 'train_cases': 3}
        cursor = dict(epoch=0, next_case=2, step=1)
        m.optimizer_step(self.view, self.optimizer, self.prepared.train[:2], prepared=self.prepared)
        path = m.save_checkpoint(self.root, self.model, self.optimizer, binding=binding, cursor=cursor)
        with self.assertRaises(ValueError):
            m.resume_checkpoint(path, self.root, self.model, self.optimizer, binding=dict(binding, seed='c' * 64))
        with self.assertRaises((ValueError, FileExistsError)):
            m.save_checkpoint(self.root, self.model, self.optimizer, binding=binding, cursor=cursor)
        scratch = self.root / '.checkpoint-abrupt'
        scratch.mkdir()
        (scratch / 'evidence').write_text('retain')
        with self.assertRaises(ValueError):
            m.resume_checkpoint(scratch, self.root, self.model, self.optimizer, binding=binding)
        m.optimizer_step(self.view, self.optimizer, self.prepared.train[2:], prepared=self.prepared)
        latest = m.save_checkpoint(self.root, self.model, self.optimizer, binding=binding, cursor=dict(epoch=1, next_case=0, step=2))
        with self.assertRaises(ValueError):
            m.resume_checkpoint(path, self.root, self.model, self.optimizer, binding=binding)
        self.assertEqual((scratch / 'evidence').read_text(), 'retain')
        with self.assertRaises(ValueError):
            m.resume_checkpoint(latest, self.root.parent, self.model, self.optimizer, binding=binding)

    def test_resealed_cursor_cannot_hide_missing_adam_updates(self):
        m = self.module()
        binding = {'run': 'a' * 64, 'seed': 'b' * 64, 'epochs': 2, 'batch_size': 2, 'train_cases': 3}
        m.optimizer_step(self.view, self.optimizer, self.prepared.train[:2], prepared=self.prepared)
        path = m.save_checkpoint(self.root, self.model, self.optimizer, binding=binding, cursor=dict(epoch=0, next_case=2, step=1))
        from scripts.mfm import train_v1_formation_specialist as shared
        receipt = shared._read_sealed(path / 'checkpoint.json', m.CHECKPOINT_SCHEMA)
        state = torch.load(path / 'optimizer.pt', weights_only=True)
        state['cursor'] = dict(epoch=1, next_case=0, step=2)
        torch.save(state, path / 'optimizer.pt')
        receipt.update(cursor=state['cursor'], optimizer_sha256=shared._digest(path / 'optimizer.pt'))
        from scripts.mfm import export_public_conditional_features as features
        features._write_record(path / 'checkpoint.json', {k: v for k, v in receipt.items() if k != 'record_sha256'}, replace=True)
        forged = self.root / 'checkpoint-00000002'
        path.rename(forged)
        with self.assertRaisesRegex(ValueError, 'Adam.*cursor'):
            m.resume_checkpoint(forged, self.root, self.model, self.optimizer, binding=binding)

    def test_owned_run_path_inventory_rejects_unknown_and_retains_abrupt_scratch(self):
        m = self.module()
        scratch = self.root / '.checkpoint-abrupt'
        scratch.mkdir()
        (scratch / 'optimizer.pt').write_bytes(b'partial')
        m._validate_run_paths(self.root)
        extra = self.root / 'unowned.txt'
        extra.write_text('preserve')
        with self.assertRaises(ValueError):
            m._validate_run_paths(self.root)
        self.assertEqual(extra.read_text(), 'preserve')
        self.assertEqual((scratch / 'optimizer.pt').read_bytes(), b'partial')

    def test_production_roster_gate_cannot_accept_tiny_material(self):
        with self.assertRaisesRegex(ValueError, 'complete.*roster'):
            self.module().require_full_material(self.prepared)

    def test_measurement_restores_seed_empty_adam_and_rng_with_real_replay(self):
        m = self.module()
        before = cached._cpu_tree(self.model.state_dict())
        rng = torch.random.get_rng_state().clone()
        result = m.measure_construction(self.view, self.optimizer, self.prepared,
            output_dir=self.root, binding={'run': 'a' * 64, 'seed': 'b' * 64, 'epochs': 2, 'batch_size': 2, 'train_cases': 3}, seed=123)
        self.assertEqual(result['replay_next_case_ids'], ['a'])
        self.assertEqual(result['replay_max_difference'], 0)
        self.assertTrue(result['control_updates_restored'])
        self.assertGreater(result['checkpoint_bytes'], 0)
        cached.compare_tree(self.model.state_dict(), before, atol=0, rtol=0)
        self.assertFalse(self.optimizer.state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_metadata_custody_requires_independent_clone_pin_and_used_assets(self):
        from cognitive_kernel.canonical import canonical_sha256, canonical_json_bytes
        from scripts.mfm import train_v1_formation_specialist as shared
        m = self.module()
        metadata = self.root / 'metadata'
        metadata.mkdir()
        names = ('config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'README.md', '.gitattributes')
        for name in names:
            (metadata / name).write_bytes(b'{}')
        receipt = {'schema': 'alice-gemma4-v1-clone-v1', 'role': 'mfm', 'files':
            [{'path': name, 'sha256': shared._digest(metadata / name), 'size': 2} for name in names] +
            [{'path': 'model.safetensors', 'sha256': self.prepared.census['prepared_base_weight_sha256'], 'size': 23919549408}]}
        receipt['receipt_sha256'] = canonical_sha256(receipt)
        path = self.root / 'clone.json'
        path.write_bytes(canonical_json_bytes(receipt))
        census = dict(self.prepared.census, prepared_clone_receipt_sha256=receipt['receipt_sha256'])
        result = m.verify_tokenizer_metadata(metadata, path, receipt['receipt_sha256'], census)
        self.assertFalse(result['base_weights_opened'])
        self.assertEqual(len(result['assets']), 7)
        with self.assertRaises(ValueError):
            m.verify_tokenizer_metadata(metadata, path, '0' * 64, census)
        (metadata / 'tokenizer.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            m.verify_tokenizer_metadata(metadata, path, receipt['receipt_sha256'], census)

    def test_run_refuses_reduced_roster_before_model_or_output_creation(self):
        from types import SimpleNamespace
        m = self.module()
        output = self.root / 'forbidden-fit'
        args = SimpleNamespace(census_dir=self.fixture.directory, census_sha256=self.fixture.pin,
            feature_bank=self.fixture.bank_dir, feature_bank_sha256=self.fixture.bank_pin, output_dir=output)
        with self.assertRaisesRegex(ValueError, 'complete.*roster'):
            m.run(args)
        self.assertFalse(output.exists())

    def test_cli_requires_explicit_recipe_and_has_no_geometry_reduction(self):
        m = self.module()
        argv = []
        fields = {'census-dir': 'census', 'feature-bank': 'bank', 'output-dir': 'out',
            'prepared-base-dir': 'metadata', 'prepared-base-receipt': 'clone.json',
            'census-sha256': 'a' * 64, 'feature-bank-sha256': 'b' * 64, 'prepared-clone-sha256': 'c' * 64,
            'allocation-ref': 'reviewed-allocation', 'provider': 'local', 'config-sha256': 'd' * 64,
            'device': 'cpu', 'epochs': '2', 'batch-size': '2', 'save-every-steps': '10', 'seed': '71',
            'logit-chunk-tokens': '2', 'generation-tokens': '8', 'learning-rate': '.001',
            'weight-decay': '.03', 'beta1': '.9', 'beta2': '.999', 'adam-eps': '1e-8'}
        for name, value in fields.items():
            argv.extend(['--' + name, value])
        args = m.arguments(argv + ['--measure-only'])
        self.assertTrue(args.measure_only)
        with self.assertRaises(SystemExit):
            m.arguments(argv + ['--specialist-width', '8'])


if __name__ == '__main__':
    unittest.main()
