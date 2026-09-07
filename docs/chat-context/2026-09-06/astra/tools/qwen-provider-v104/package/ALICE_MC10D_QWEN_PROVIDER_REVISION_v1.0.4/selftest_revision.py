"""Actual producer/consumer and a2 transition tests. No live compute or models."""
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import contract as c
import controller
import evidence_origins
import fixture_origins as fixtures
import infra_recovery
import package_revision as revision
import remote_agent as remote

NEW_SHA = '2' * 64


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = mock.patch.object(c, 'ROOT', self.root)
        self.root_patch.start()
        self.source, self.packages, self.original = fixtures.install_sources(self.root)
        self.run = self.root / 'runs' / c.RUN_ID
        fixtures.prior_a2(self.run)
        self.commands = []; self.jobs = []

    def tearDown(self):
        self.root_patch.stop(); self.temp.cleanup()

    def scheduler(self, argv, check=True):
        self.commands.append(argv)
        if argv[0] == 'squeue':
            text = '' if '-j' in argv else ''.join(job + '|' + c.RUN_ID + '|RUNNING\n' for job in self.jobs)
        elif argv[0] == 'sacct':
            text = ('575089|' + c.CALIBRATION_ID + '|FAILED|76:0|\n' if '-j' in argv
                    else ''.join(job + '|' + c.RUN_ID + '|RUNNING|0:0|\n' for job in self.jobs))
        elif argv[0] == 'sbatch':
            self.jobs.append('61234'); text = '61234\n'
        else:
            raise AssertionError('Unexpected command: ' + str(argv))
        return subprocess.CompletedProcess(argv, 0, text, '')

    def revise(self, run=None):
        with mock.patch.object(remote, 'cmd', side_effect=self.scheduler):
            return revision.apply_remote(run or self.run, NEW_SHA, self.scheduler, remote.find_submitted)

    def test_original_collector_live_layout_now_verifies_without_materializing_export(self):
        before = snapshot(self.source)
        infra_recovery.verify_source_files(self.source)
        self.assertFalse((self.source / 'package-manifest.json').exists())
        terminal = {'terminal': True, 'job_id': '575089', 'scheduler_state': 'FAILED'}
        with mock.patch.object(self.original, 'status', return_value=terminal), \
             mock.patch.object(self.original, 'repair_terminal_telemetry'):
            receipt = self.original.collect(self.source)
        with zipfile.ZipFile(receipt['remote_path']) as z:
            exported = z.read('package-manifest.json')
        self.assertEqual(exported, (self.packages['a1'] / 'PACKAGE_MANIFEST.json').read_bytes())
        self.assertFalse((self.source / 'package-manifest.json').exists())
        self.assertTrue(all((self.source / name).read_bytes() == raw for name, raw in before.items()))

    def test_correct_export_copy_cannot_mask_changed_original_package(self):
        origin = self.packages['a1'] / 'PACKAGE_MANIFEST.json'
        (self.source / 'package-manifest.json').write_bytes(origin.read_bytes())
        origin.write_bytes(origin.read_bytes() + b' ')
        with self.assertRaises(c.Stop):
            infra_recovery.verify_source_files(self.source)

    def test_changed_original_worker_refuses_before_revision_or_submission(self):
        before = (self.run / 'run.json').read_bytes()
        worker = self.packages['a1'] / 'worker.py'; worker.write_bytes(worker.read_bytes() + b'\n')
        with self.assertRaises(c.Stop): self.revise()
        self.assertEqual((self.run / 'run.json').read_bytes(), before)
        self.assertFalse(revision.history(self.run).exists())
        self.assertFalse(self.jobs)

    def test_failed_source_queue_query_cannot_be_interpreted_as_absence(self):
        def failed(argv, check=True):
            return subprocess.CompletedProcess(argv, 1, '', 'unavailable')
        with self.assertRaisesRegex(c.Stop, 'absence is unknown'):
            infra_recovery.reconcile_source(failed)
        self.assertFalse((self.run / 'package-revision.json').exists())

    def test_existing_a2_job_or_any_execution_marker_refuses_revision(self):
        before = (self.run / 'run.json').read_bytes()
        self.jobs.append('61234')
        with self.assertRaises(c.Stop): self.revise()
        self.jobs.clear()
        for name in (*revision.FORBIDDEN, 'tasks/Q01_COMPATIBLE_NOVELTY/attempt.json'):
            with self.subTest(name=name):
                path = self.run / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'{}')
                with self.assertRaises(c.Stop): self.revise()
                path.unlink()
                self.assertEqual((self.run / 'run.json').read_bytes(), before)
        self.assertFalse(revision.history(self.run).exists())

    def test_successful_revision_preserves_exact_prior_bytes_and_source(self):
        source_before = snapshot(self.source)
        old_run = (self.run / 'run.json').read_bytes()
        old_failure = (self.run / 'infrastructure-preflight-failure.json').read_bytes()
        result = self.revise()
        self.assertEqual(result, revision.receipt(NEW_SHA))
        self.assertEqual((revision.history(self.run) / 'run-v102.json').read_bytes(), old_run)
        self.assertEqual((revision.history(self.run) / 'preflight-failure-v102.json').read_bytes(), old_failure)
        self.assertEqual(snapshot(self.source), source_before)
        self.assertEqual(c.read(self.run / 'run.json'), revision.descriptor(NEW_SHA))
        self.assertFalse(self.jobs)

    def test_each_interrupted_revision_write_resumes_with_same_archives_and_identity(self):
        real_write = c.atomic_bytes
        for index, name in enumerate(('run-v102.json', 'preflight-failure-v102.json', 'intent.json', 'run.json', 'package-revision.json')):
            with self.subTest(crash_after=name):
                run = self.root / ('crash-' + str(index)); fixtures.prior_a2(run)
                raised = []
                def write(path, data):
                    real_write(path, data)
                    if Path(path).name == name and not raised:
                        raised.append(True); raise RuntimeError('synthetic process interruption after durable write')
                with mock.patch.object(c, 'atomic_bytes', side_effect=write), self.assertRaises(RuntimeError):
                    self.revise(run)
                self.assertEqual(self.revise(run), revision.receipt(NEW_SHA))
                self.assertEqual(self.revise(run), revision.receipt(NEW_SHA))
                revision.verify_current(run, NEW_SHA)
                self.assertFalse((run / 'submission-intent.json').exists())
        self.assertFalse(self.jobs)

    def test_new_descriptor_without_intent_is_not_inferred_as_completed_revision(self):
        c.write(self.run / 'run.json', revision.descriptor(NEW_SHA))
        with self.assertRaises((c.Stop, FileNotFoundError)):
            self.revise()
        self.assertFalse((self.run / 'package-revision.json').exists())

    def test_actual_old_v102_submit_refuses_revised_descriptor(self):
        self.revise()
        old = fixtures.load_frozen_remote(self.packages['a2'])
        with mock.patch.object(old, 'submission_lock', controller.controller_lock), \
             mock.patch.object(old, 'cmd') as command, self.assertRaisesRegex(old.c.Stop, 'immutable evidence differs: run.json'):
            old.submit(self.run, revision.policy()['old_package_sha256'])
        command.assert_not_called()

    def test_actual_new_collector_exports_revision_lineage_and_verifier_detects_tampering(self):
        self.revise()
        state = {'terminal': True, 'job_id': '61234', 'scheduler_state': 'FAILED'}
        with mock.patch.object(remote, 'status', return_value=state), mock.patch.object(remote, 'repair_terminal_telemetry'):
            result = remote.collect(self.run)
        folder = self.root / 'exported'
        with zipfile.ZipFile(result['remote_path']) as z: z.extractall(folder)
        revision.verify_evidence(folder, NEW_SHA)
        self.assertEqual((folder / 'package-manifest.json').read_bytes(), (c.BASE / 'PACKAGE_MANIFEST.json').read_bytes())
        self.assertFalse((self.run / 'package-manifest.json').exists())
        for target in (folder / 'run.json', folder / 'package-revision.json', revision.history(folder) / 'run-v102.json'):
            before = target.read_bytes(); target.write_bytes(b'{}')
            with self.assertRaises(c.Stop): revision.verify_evidence(folder, NEW_SHA)
            target.write_bytes(before)

    def test_changed_local_state_stops_before_archive_or_network(self):
        state_path, rules = fixtures.local_prepared(self.root / 'vault')
        state_path.write_bytes(state_path.read_bytes() + b' ')
        with mock.patch.object(revision, 'policy', return_value=rules), self.assertRaises(c.Stop):
            revision.begin_local(state_path, NEW_SHA)
        self.assertFalse(revision.history(state_path.parent).exists())

    def test_lost_revision_ack_then_controller_resume_submits_only_one_job(self):
        vault = self.root / 'vault'; state_path, rules = fixtures.local_prepared(vault)
        old_state = state_path.read_bytes()
        package = self.root / 'synthetic-transfer.zip'; package.write_bytes(b'synthetic transfer fixture')
        new_sha = c.file_sha(package)
        helper = self.root / 'bin/rayan-telemetry-push.sh'; helper.parent.mkdir(); helper.write_text('fixture'); helper.chmod(0o700)
        (self.root / 'telemetry/ledger/.git').mkdir(parents=True)
        args = SimpleNamespace(package_zip=package, package_sha=new_sha, vault_root=vault,
            repo_root=self.root / 'repo', output_root=self.root / 'downloads', ssh_key=self.root / 'key')
        lost = [True]; events = []; test = self
        class Client:
            def __init__(self, *args): pass
            def stage(self, *args): events.append('stage')
            def action(self, action):
                events.append(action)
                if action == 'revise':
                    value = remote.revise(test.run, new_sha)
                    if lost[0]:
                        lost[0] = False; raise controller.TransportPending('lost acknowledgement after completed remote revision')
                    return value
                if action == 'submit': return remote.submit(test.run, new_sha)
                if action == 'status': raise controller.TransportPending('synthetic monitoring disconnection')
                raise AssertionError(action)
        with mock.patch.object(revision, 'policy', return_value=rules), \
             mock.patch.object(controller, 'Magnolia', Client), mock.patch.object(controller, 'validate_local'), \
             mock.patch.object(remote, 'submission_lock', controller.controller_lock), \
             mock.patch.object(remote, 'cmd', side_effect=self.scheduler), \
             mock.patch.object(remote.shutil, 'which', return_value='/fixture'), \
             mock.patch.object(infra_recovery, 'preflight'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(controller.TransportPending): controller.execute(args)
            self.assertEqual(state_path.read_bytes(), old_state)
            self.assertFalse(self.jobs)
            with self.assertRaises(controller.TransportPending): controller.execute(args)
            with self.assertRaises(controller.TransportPending): controller.execute(args)
            self.assertEqual(c.read(state_path)['phase'], 'SUBMITTED')
            self.assertEqual((revision.history(state_path.parent) / 'controller-state-v102.json').read_bytes(), old_state)
            revision.verify_current(self.run, new_sha)
        self.assertEqual(self.jobs, ['61234'])
        self.assertEqual(events.count('revise'), 2)
        self.assertEqual(sum(argv[0] == 'sbatch' for argv in self.commands), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
