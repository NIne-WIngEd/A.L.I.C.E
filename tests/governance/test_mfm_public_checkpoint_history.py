"""Real small-producer continuation; no production learning/resource claim."""
from pathlib import Path
import os
import stat
from time import perf_counter
from unittest.mock import patch
import unittest

import torch
from cognitive_kernel.formation_conditional_task import bind_conditional_task
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from scripts.mfm import train_v1_formation_specialist as shared
from scripts.mfm import train_v16_cached_specialist as cached
import test_mfm_public_conditional_foundation as fixtures


class PublicCheckpointHistoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PublicFoundationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.m = self.fixture.module()
        self.root = self.fixture.root
        self.prepared = self.fixture.prepared
        self.model = self.fixture.model
        self.view = self.fixture.view
        self.optimizer = self.fixture.optimizer

    def test_successful_exact_stop_preserves_next_partial_batch_and_optimizer_boundary(self):
        # Removing the complete-update stop would train all four updates.
        binding = dict(run='a' * 64, seed='b' * 64, epochs=2, batch_size=2, train_cases=3)
        cursor, checkpoint = self.m.fit_epochs(self.view, self.optimizer, self.prepared,
            output_dir=self.root, binding=binding, seed=123,
            cursor=dict(epoch=0, next_case=0, step=0), save_every_steps=99,
            started=perf_counter(), session_seconds=None, purpose='full-roster-public-fit', stop_step=1)
        self.assertEqual(cursor, dict(epoch=0, next_case=2, step=1))
        self.assertEqual(checkpoint.name, 'checkpoint-00000001')
        self.assertEqual(sorted(p.name for p in self.root.glob('checkpoint-*')),
                         ['checkpoint-00000000', 'checkpoint-00000001'])
        batch, following = self.m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=cursor)
        self.assertEqual([p['case_id'] for p in batch], ['a'])
        self.assertEqual(following, dict(epoch=1, next_case=0, step=2))
        for invalid in (0, 5, True):
            with self.subTest(stop_step=invalid), self.assertRaises(ValueError):
                self.m.fit_epochs(self.view, self.optimizer, self.prepared, output_dir=self.root,
                    binding=binding, seed=123, cursor=cursor, save_every_steps=99,
                    started=perf_counter(), session_seconds=None, purpose='full-roster-public-fit', stop_step=invalid)

    def initial_history(self):
        self.assertTrue(hasattr(self.m, 'checkpoint_history'), 'authenticated public history interface missing')
        manifest = shared._seal(dict(schema=self.m.RUN_SCHEMA, purpose='full-roster-public-fit',
            specialist_config=self.model.config.record(), source='fixture-only', runtime='actual-local-cpu'))
        curriculum = shared._seal(dict(schema=self.m.CURRICULUM_SCHEMA, epochs=2, batch_size=2, seed=123))
        seed = self.m._seed_control(self.root, self.model, manifest['record_sha256'], self.model.config)
        binding = dict(run=manifest['record_sha256'], seed=seed['record_sha256'], epochs=2, batch_size=2, train_cases=3)
        shared._write_new(self.root / 'run.json', {k: v for k, v in manifest.items() if k != 'record_sha256'})
        shared._write_new(self.root / 'curriculum.json', {k: v for k, v in curriculum.items() if k != 'record_sha256'})
        shared._write_new(self.root / 'seed-generation.json', dict(schema='fixture-seed-generation', run_sha256=binding['run']))
        construction = self.m.measure_construction(self.view, self.optimizer, self.prepared,
            output_dir=self.root, binding=binding, seed=123)
        shared._write_new(self.root / 'construction.json', dict(construction, run_sha256=binding['run']))
        history = self.m.checkpoint_history.start(self.root, manifest=manifest, curriculum=curriculum,
            binding=binding, session_index=0, allocation_ref='fixture-session0/v1')
        return manifest, curriculum, binding, history

    def test_authenticated_mounted_history_replays_exactly_and_publishes_global_final_lineage(self):
        # Losing earlier entries, copying history, or loading a stale state breaks
        # the next partial-batch replay or the final artifact's four checkpoints.
        manifest, curriculum, binding, history = self.initial_history()
        cursor, first = self.m.fit_epochs(self.view, self.optimizer, self.prepared, output_dir=self.root,
            binding=binding, seed=123, cursor=dict(epoch=0, next_case=0, step=0), save_every_steps=99,
            started=perf_counter(), session_seconds=None, purpose='full-roster-public-fit', stop_step=1, history=history)
        parent_bytes = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        if os.name == 'nt':
            # Actual Windows readonly source bytes, distinct from a Linux mount.
            readonly_files = [p for p in self.root.rglob('*') if p.is_file()]
            for path in readonly_files:
                path.chmod(stat.S_IREAD)
            self.addCleanup(lambda: [p.chmod(stat.S_IWRITE) for p in readonly_files if p.exists()])
            with self.assertRaises(PermissionError):
                (first / 'specialist.safetensors').write_bytes(b'forbidden')
        batch, expected_cursor = self.m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=cursor)
        expected = self.m.optimizer_step(self.view, self.optimizer, batch, prepared=self.prepared)
        expected_model = cached._cpu_tree(self.model.state_dict())
        expected_adam = cached._cpu_tree(self.optimizer.state_dict())
        expected_rng = torch.random.get_rng_state().clone()
        expected_cuda = cached._cpu_tree(torch.cuda.get_rng_state_all())
        output = self.root.parent / (self.root.name + '-session1')
        output.mkdir()
        self.addCleanup(lambda: __import__('shutil').rmtree(output))
        # Only the environmental readonly-filesystem probe is substituted on
        # Windows. All source bytes, model, Adam, serializers and fitting are real.
        with patch.object(self.m.checkpoint_history, 'require_readonly_root', side_effect=self.m._safe_root):
            accepted = self.m.checkpoint_history.load(history.latest_path, history.record['record_sha256'],
                sources={0: self.root}, manifest=manifest, curriculum=curriculum, binding=binding)
        torch.manual_seed(999)
        other = FormationDecisionSpecialist(self.model.config)
        adam = torch.optim.AdamW(other.parameters(), lr=.001, weight_decay=.03)
        view = bind_conditional_task(other, task=self.m.TASK)
        restored = self.m.resume_checkpoint(first, output, other, adam, binding=binding, history=accepted)
        next_pairs, next_cursor = self.m.next_batch(self.prepared, seed=123, epochs=2, batch_size=2, cursor=restored)
        self.assertEqual([p['case_id'] for p in next_pairs], ['a'])
        actual = self.m.optimizer_step(view, adam, next_pairs, prepared=self.prepared)
        self.assertEqual(actual, expected)
        self.assertEqual(next_cursor, expected_cursor)
        cached.compare_tree(other.state_dict(), expected_model, atol=0, rtol=0)
        cached.compare_tree(adam.state_dict(), expected_adam, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        cached.compare_tree(torch.cuda.get_rng_state_all(), expected_cuda, atol=0, rtol=0)
        # Start from the accepted boundary again; this proves the production
        # fitting loop also uses the authenticated external parent directly.
        restored = self.m.resume_checkpoint(first, output, other, adam, binding=binding, history=accepted)
        current = self.m.checkpoint_history.start(output, manifest=manifest, curriculum=curriculum,
            binding=binding, session_index=1, allocation_ref='fixture-session1/v1', parent=accepted)
        cursor, final = self.m.fit_epochs(view, adam, self.prepared, output_dir=output, binding=binding,
            seed=123, cursor=restored, save_every_steps=2, started=perf_counter(), session_seconds=None,
            purpose='full-roster-public-fit', stop_step=4, history=current)
        self.assertEqual(cursor, dict(epoch=2, next_case=0, step=4))
        self.assertEqual(sorted(p.name for p in output.glob('checkpoint-*')),
                         ['checkpoint-00000002', 'checkpoint-00000004'])
        component = self.m.publish_component(view, self.prepared, output_dir=output, binding=binding,
            curriculum=curriculum, cursor=cursor, checkpoint=final, generation={'fixture': True}, history=current)
        self.assertEqual([e['receipt']['cursor']['step'] for e in current.record['entries']], [0, 1, 2, 4])
        self.assertEqual(component['checkpoint_lineage'], [e['receipt']['record_sha256'] for e in current.record['entries']])
        self.assertEqual(component['checkpoint_history_sha256'], current.record['record_sha256'])
        self.assertEqual({str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}, parent_bytes)
        self.assertEqual(component['training_objective_report']['cases'], 3)
        self.assertEqual(component['development_objective_report']['cases'], len(self.prepared.development))
        with patch.object(self.m.checkpoint_history, 'require_readonly_root', side_effect=self.m._safe_root):
            final_history = self.m.checkpoint_history.load(current.latest_path, current.record['record_sha256'],
                sources={0: self.root, 1: output}, manifest=manifest, curriculum=curriculum, binding=binding)
        self.assertEqual(final_history.latest_checkpoint, final)
        completed_output = self.root.parent / (self.root.name + '-completed-continuation')
        completed_output.mkdir()
        self.addCleanup(lambda: __import__('shutil').rmtree(completed_output))
        with self.assertRaises(ValueError):
            self.m.checkpoint_history.start(completed_output, manifest=manifest, curriculum=curriculum,
                binding=binding, session_index=2, allocation_ref='fixture-session2/v1', parent=final_history)

    def test_history_rejects_changed_pin_recipe_mutated_bytes_foreign_stale_and_measurement(self):
        manifest, curriculum, binding, history = self.initial_history()
        cursor, latest = self.m.fit_epochs(self.view, self.optimizer, self.prepared, output_dir=self.root,
            binding=binding, seed=123, cursor=dict(epoch=0, next_case=0, step=0), save_every_steps=99,
            started=perf_counter(), session_seconds=None, purpose='full-roster-public-fit', stop_step=1, history=history)
        def load(**changes):
            kwargs = dict(sources={0: self.root}, manifest=manifest, curriculum=curriculum, binding=binding)
            kwargs.update(changes)
            with patch.object(self.m.checkpoint_history, 'require_readonly_root', side_effect=self.m._safe_root):
                return self.m.checkpoint_history.load(history.latest_path, history.record['record_sha256'], **kwargs)
        with self.assertRaises(ValueError):
            with patch.object(self.m.checkpoint_history, 'require_readonly_root', side_effect=self.m._safe_root):
                self.m.checkpoint_history.load(history.latest_path, 'f' * 64, sources={0: self.root},
                    manifest=manifest, curriculum=curriculum, binding=binding)
        for changes in (dict(binding=dict(binding, epochs=3)), dict(manifest=dict(manifest, source='changed')),
                        dict(curriculum=dict(curriculum, seed=124)), dict(sources={0: self.root.parent})):
            with self.subTest(changes=changes), self.assertRaises((ValueError, FileNotFoundError)):
                load(**changes)
        accepted = load()
        with self.assertRaises(ValueError):
            self.m.resume_checkpoint(self.root / 'checkpoint-00000000', self.root, self.model, self.optimizer,
                binding=binding, history=accepted)
        with self.assertRaises(ValueError):
            self.m.resume_checkpoint(self.root / 'construction-controls' / 'checkpoint-00000001', self.root,
                self.model, self.optimizer, binding=binding, history=accepted)
        file = latest / 'specialist.safetensors'
        original = file.read_bytes()
        file.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaises(ValueError):
            load()
        file.write_bytes(original)
        with self.assertRaises(ValueError):
            self.m.checkpoint_history.start(self.root, manifest=dict(manifest, purpose='construction-measurement-only'),
                curriculum=curriculum, binding=binding, session_index=0, allocation_ref='fixture/v1')
        with self.assertRaises(ValueError):
            self.m.checkpoint_history.require_readonly_root(self.root)

    def test_authenticated_history_cannot_omit_original_seed_tensor_inventory(self):
        # A valid external digest cannot excuse a structurally incomplete index.
        manifest, curriculum, binding, history = self.initial_history()
        self.m.fit_epochs(self.view, self.optimizer, self.prepared, output_dir=self.root,
            binding=binding, seed=123, cursor=dict(epoch=0, next_case=0, step=0), save_every_steps=99,
            started=perf_counter(), session_seconds=None, purpose='full-roster-public-fit', stop_step=1, history=history)
        record = {k: v for k, v in history.record.items() if k != 'record_sha256'}
        record['origin'] = dict(record['origin'], files=[r for r in record['origin']['files']
                                                       if r['path'] != 'seed-control.safetensors'])
        history.latest_path.unlink()
        changed = shared._write_new(history.latest_path, record)
        with patch.object(self.m.checkpoint_history, 'require_readonly_root', side_effect=self.m._safe_root):
            with self.assertRaises(ValueError):
                self.m.checkpoint_history.load(history.latest_path, changed['record_sha256'], sources={0: self.root},
                    manifest=manifest, curriculum=curriculum, binding=binding)

    def test_checkpoint_cannot_override_exact_current_optimizer_recipe(self):
        # load_state_dict otherwise silently adopts a forged learning rate.
        binding = dict(run='a' * 64, seed='b' * 64, epochs=2, batch_size=2, train_cases=3)
        self.m.optimizer_step(self.view, self.optimizer, self.prepared.train[:2], prepared=self.prepared)
        path = self.m.save_checkpoint(self.root, self.model, self.optimizer, binding=binding,
            cursor=dict(epoch=0, next_case=2, step=1))
        state = torch.load(path / 'optimizer.pt', weights_only=True)
        state['optimizer']['param_groups'][0]['lr'] = .25
        torch.save(state, path / 'optimizer.pt')
        receipt = shared._read_sealed(path / 'checkpoint.json', self.m.CHECKPOINT_SCHEMA)
        path.joinpath('checkpoint.json').unlink()
        shared._write_new(path / 'checkpoint.json', dict(
            {k: v for k, v in receipt.items() if k != 'record_sha256'},
            optimizer_sha256=shared._digest(path / 'optimizer.pt')))
        with self.assertRaisesRegex(ValueError, 'optimizer.*recipe'):
            self.m.resume_checkpoint(path, self.root, self.model, self.optimizer, binding=binding)

    def test_checkpoint_cannot_permute_adam_parameter_identity_order(self):
        # Native loading zips saved IDs to current parameters by list order.
        binding = dict(run='a' * 64, seed='b' * 64, epochs=2, batch_size=2, train_cases=3)
        self.m.optimizer_step(self.view, self.optimizer, self.prepared.train[:2], prepared=self.prepared)
        path = self.m.save_checkpoint(self.root, self.model, self.optimizer, binding=binding,
            cursor=dict(epoch=0, next_case=2, step=1))
        state = torch.load(path / 'optimizer.pt', weights_only=True)
        state['optimizer']['param_groups'][0]['params'].reverse()
        shadow = FormationDecisionSpecialist(self.model.config)
        shadow_adam = torch.optim.AdamW(shadow.parameters(), lr=.001, weight_decay=.03)
        shadow_adam.load_state_dict(state['optimizer'])
        parameter = shadow_adam.param_groups[0]['params'][0]
        self.assertNotEqual(tuple(shadow_adam.state[parameter]['exp_avg'].shape), tuple(parameter.shape))
        torch.save(state, path / 'optimizer.pt')
        receipt = shared._read_sealed(path / 'checkpoint.json', self.m.CHECKPOINT_SCHEMA)
        (path / 'checkpoint.json').unlink()
        shared._write_new(path / 'checkpoint.json', dict(
            {k: v for k, v in receipt.items() if k != 'record_sha256'},
            optimizer_sha256=shared._digest(path / 'optimizer.pt')))
        with self.assertRaisesRegex(ValueError, 'optimizer.*recipe'):
            self.m.resume_checkpoint(path, self.root, self.model, self.optimizer, binding=binding)


if __name__ == '__main__':
    unittest.main()
