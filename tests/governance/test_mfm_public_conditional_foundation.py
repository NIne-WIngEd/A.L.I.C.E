"""Tiny actual-model contracts; these fixtures prove no production resource fit."""
import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from time import perf_counter

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

    def test_actual_early_process_exit_resumes_restored_run_root_zero_boundary(self):
        # Interrupt the actual model forward after real computation, both during
        # step1 and after step1 completed but before save interval99. The real
        # production fitting loop, optimizer, serializer and construction run.
        import json
        m = self.module()
        baseline = cached._cpu_tree(self.model.state_dict())
        initial_rng = torch.random.get_rng_state().clone()
        binding = {'run': 'a' * 64, 'seed': 'b' * 64, 'epochs': 2, 'batch_size': 2, 'train_cases': 3}
        code = r'''
import json, os, sys, torch
from pathlib import Path
from time import perf_counter
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel.formation_conditional_task import bind_conditional_task
from scripts.mfm import public_conditional_training as reader
from scripts.mfm import train_public_conditional_foundation as m
paths, binding, interrupt_at = json.loads(sys.argv[1]), json.loads(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(1)
torch.manual_seed(71)
class InterruptedProducer(FormationDecisionSpecialist):
    calls = 0
    interrupt_at = None
    def forward(self, **data):
        result = super().forward(**data)
        self.calls += 1
        if self.interrupt_at == self.calls:
            os._exit(73)
        return result
model = InterruptedProducer(SpecialistConfig(3, 16, 8, 1, 2, 16, 0, 2, 1, 2))
view = bind_conditional_task(model, task='conditional_meaning_to_text')
adam = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
prepared = reader.load_public_training(Path(paths['census']), paths['census_pin'], Path(paths['bank']), paths['bank_pin'])
root = Path(paths['output'])
seed_receipt = m._seed_control(root, model, binding['run'], model.config)
binding['seed'] = seed_receipt['record_sha256']
m.measure_construction(view, adam, prepared, output_dir=root, binding=binding, seed=123)
model.calls, model.interrupt_at = 0, interrupt_at
m.fit_epochs(view, adam, prepared, output_dir=root, binding=binding, seed=123,
    cursor=dict(epoch=0, next_case=0, step=0), save_every_steps=99,
    started=perf_counter(), session_seconds=None, purpose='full-roster-public-fit')
'''
        for interrupt_at in (1, 3):
            with self.subTest(interrupt_at=interrupt_at):
                root = self.root / f'early-{interrupt_at}'
                root.mkdir()
                paths = {'output': str(root), 'census': str(self.fixture.directory),
                    'census_pin': self.fixture.pin, 'bank': str(self.fixture.bank_dir), 'bank_pin': self.fixture.bank_pin}
                process = subprocess.run([sys.executable, '-c', code, json.dumps(paths), json.dumps(binding), str(interrupt_at)],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 73, process.stdout + process.stderr)
                seed_receipt = m._verify_seed(root, binding['run'], self.model.config)
                case_binding = dict(binding, seed=seed_receipt['record_sha256'])
                checkpoint = root / 'checkpoint-00000000'
                self.assertTrue(checkpoint.is_dir(), 'early fitting exit has no admissible run-root boundary')
                self.assertFalse((root / 'checkpoint-00000001').exists())
                torch.manual_seed(999)
                restored = FormationDecisionSpecialist(self.model.config)
                adam = torch.optim.AdamW(restored.parameters(), lr=.001, weight_decay=.03)
                cursor = m.resume_checkpoint(checkpoint, root, restored, adam, binding=case_binding)
                self.assertEqual(cursor, dict(epoch=0, next_case=0, step=0))
                self.assertFalse(adam.state)
                cached.compare_tree(restored.state_dict(), baseline, atol=0, rtol=0)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), initial_rng))
                batch, following = m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=cursor)
                self.assertEqual([p['case_id'] for p in batch], ['c', 'b'])
                self.assertEqual(following, dict(epoch=0, next_case=2, step=1))
                reference = FormationDecisionSpecialist(self.model.config)
                reference.load_state_dict(baseline)
                reference_adam = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=.03)
                torch.random.set_rng_state(initial_rng)
                expected = m.optimizer_step(bind_conditional_task(reference, task='conditional_meaning_to_text'), reference_adam, batch, prepared=self.prepared)
                expected_rng = torch.random.get_rng_state().clone()
                torch.random.set_rng_state(initial_rng)
                actual = m.optimizer_step(bind_conditional_task(restored, task='conditional_meaning_to_text'), adam, batch, prepared=self.prepared)
                self.assertEqual(actual, expected)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
                cached.compare_tree(restored.state_dict(), cached._cpu_tree(reference.state_dict()), atol=0, rtol=0)
                cached.compare_tree(adam.state_dict(), cached._cpu_tree(reference_adam.state_dict()), atol=0, rtol=0)
                with self.assertRaises(ValueError):
                    m.resume_checkpoint(root / 'construction-controls' / 'checkpoint-00000001', root, restored, adam, binding=case_binding)

    def test_measurement_purpose_cannot_enter_fitting_or_create_fit_boundary(self):
        m = self.module()
        binding = {'run': 'a' * 64, 'seed': 'b' * 64, 'epochs': 2, 'batch_size': 2, 'train_cases': 3}
        with self.assertRaisesRegex(ValueError, 'measurement'):
            m.fit_epochs(self.view, self.optimizer, self.prepared, output_dir=self.root, binding=binding, seed=123,
                cursor=dict(epoch=0, next_case=0, step=0), save_every_steps=99,
                started=perf_counter(), session_seconds=None, purpose='construction-measurement-only')
        self.assertFalse(self.optimizer.state)
        self.assertEqual(list(self.root.glob('checkpoint-*')), [])

    def test_scheduled_zero_boundary_is_immutable_on_zero_step_continuation(self):
        m = self.module()
        binding = {'run': 'a' * 64, 'seed': 'b' * 64, 'epochs': 2, 'batch_size': 2, 'train_cases': 3}
        initial = dict(epoch=0, next_case=0, step=0)
        kwargs = dict(output_dir=self.root, binding=binding, seed=123, cursor=initial,
            save_every_steps=99, started=perf_counter() - 10, session_seconds=1, purpose='full-roster-public-fit')
        cursor, checkpoint = m.fit_epochs(self.view, self.optimizer, self.prepared, **kwargs)
        original = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
        rng = torch.random.get_rng_state().clone()
        self.assertEqual(cursor, initial)
        restored = m.resume_checkpoint(checkpoint, self.root, self.model, self.optimizer, binding=binding)
        self.assertEqual(restored, initial)
        cursor, repeated = m.fit_epochs(self.view, self.optimizer, self.prepared, **kwargs)
        self.assertEqual(repeated, checkpoint)
        self.assertEqual({p.name: p.read_bytes() for p in repeated.iterdir()}, original)
        self.assertFalse(self.optimizer.state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))


if __name__ == '__main__':
    unittest.main()
