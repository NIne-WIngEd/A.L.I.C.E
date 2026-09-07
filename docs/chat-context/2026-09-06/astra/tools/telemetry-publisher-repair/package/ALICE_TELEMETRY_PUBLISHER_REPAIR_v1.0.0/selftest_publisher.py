import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import publisher as p

REAL_GIT = shutil.which('git')

@unittest.skipIf(os.name == 'nt', 'Publisher runs on Magnolia POSIX; actual Git fixtures run in the Linux release build')
class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='alice-publisher-test-')
        self.root = Path(self.tmp.name)
        self.ledger = self.root / 'telemetry/ledger'; self.ledger.parent.mkdir(parents=True)
        self.remote = self.root / 'remote.git'
        self.git('init', '--bare', str(self.remote), cwd=self.root)
        self.git('init', str(self.ledger), cwd=self.root)
        for k, v in [('user.name', 'Local fixture'), ('user.email', 'fixture@example.invalid'), ('commit.gpgsign', 'false')]:
            self.git('config', k, v)
        (self.ledger / 'README').write_text('initial\n')
        self.git('add', 'README'); self.git('commit', '-m', 'initial'); self.git('branch', '-M', 'main')
        self.git('remote', 'add', 'origin', str(self.remote)); self.git('push', 'origin', 'main')
        self.initial = self.git('rev-parse', 'HEAD').strip()
        payload = self.root / 'payload'; payload.mkdir()
        for n in ('application.log', 'manifest.json', 'receipt.json'): (payload / n).write_text('{}\n')
        self.args = ['fixture-1', 'FIXTURE', 'FAILED', '1', *[str(payload / n) for n in ('application.log', 'manifest.json', 'receipt.json')]]
        self.bin = self.root / 'fixture-bin'; self.bin.mkdir()
        # Modern Git normally masks the observed old-provider defect. Disabling
        # its refmap only for the historical source-only command models
        # the documented pre-1.8.4 behavior. Destination refspecs stay explicit.
        wrapper = self.bin / 'git'
        wrapper.write_text('#!/bin/bash\nif [ "$1" = fetch ] && [ "${3:-}" = main ] && [ "$(' + REAL_GIT + ' --version)" != "git version 1.8.3.1" ]; then exec ' + REAL_GIT + ' fetch --refmap= origin main; fi\nexec ' + REAL_GIT + ' "$@"\n')
        wrapper.chmod(0o700)
        self.env = patch.dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'])
        self.env.start()

    def tearDown(self): self.env.stop(); self.tmp.cleanup()

    def git(self, *args, cwd=None):
        r = subprocess.run([REAL_GIT, *args], cwd=cwd or self.ledger, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.decode()

    def advance_remote(self, name='external.txt'):
        other = self.root / ('writer-' + name)
        self.git('clone', str(self.remote), str(other), cwd=self.root)
        self.git('checkout', 'main', cwd=other)
        self.git('config', 'user.name', 'Other fixture', cwd=other)
        self.git('config', 'user.email', 'other@example.invalid', cwd=other)
        self.git('config', 'commit.gpgsign', 'false', cwd=other)
        (other / name).write_text('concurrent append\n')
        self.git('add', name, cwd=other); self.git('commit', '-m', name, cwd=other)
        self.git('push', 'origin', 'main', cwd=other)
        return self.git('rev-parse', 'HEAD', cwd=other).strip()

    def publish(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return p.publish(self.args, self.root)

    def test_stale_ref_reproduction_and_destination_refspec_repair_preserve_checkout(self):
        newer = self.advance_remote()
        r = subprocess.run(['git', 'fetch', 'origin', 'main'], cwd=self.ledger, capture_output=True)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(self.git('rev-parse', 'FETCH_HEAD').strip(), newer)
        self.assertEqual(self.git('rev-parse', 'refs/remotes/origin/main').strip(), self.initial)
        (self.ledger / 'README').write_text('owner uncommitted change\n')
        (self.ledger / 'untracked').write_text('preserve me\n')
        index_before = (self.ledger / '.git/index').read_bytes()
        self.assertEqual(self.publish(), 0)
        self.assertEqual(self.git('rev-parse', 'HEAD').strip(), self.initial)
        self.assertEqual((self.ledger / '.git/index').read_bytes(), index_before)
        self.assertEqual((self.ledger / 'README').read_text(), 'owner uncommitted change\n')
        self.assertEqual((self.ledger / 'untracked').read_text(), 'preserve me\n')
        self.assertEqual(self.git('show', 'refs/remotes/origin/main:external.txt'), 'concurrent append\n')

    def test_retry_freezes_metadata_and_changed_input_cannot_overwrite(self):
        self.assertEqual(self.publish(), 0)
        spool = self.root / 'telemetry/spool/fixture-1'
        first = p.file_map(spool)
        remote_head = self.git('rev-parse', 'refs/remotes/origin/main')
        with patch.object(p, 'now', return_value='2099-01-01T00:00:00+00:00'):
            self.assertEqual(self.publish(), 0)
        self.assertEqual(first, p.file_map(spool))
        self.assertEqual(remote_head, self.git('rev-parse', 'refs/remotes/origin/main'))
        Path(self.args[4]).write_text('changed\n')
        self.assertEqual(self.publish(), 73)
        self.assertEqual(first, p.file_map(spool))

    def test_competing_append_rebuilds_on_fresh_parent_without_force(self):
        original = p.Git.call
        raced = [False]
        def call(git, *args, **kwargs):
            if args[0] == 'push' and not raced[0]:
                raced[0] = True; self.advance_remote('racing.txt')
            return original(git, *args, **kwargs)
        with patch.object(p.Git, 'call', call): self.assertEqual(self.publish(), 0)
        self.assertTrue(raced[0])
        self.assertEqual(self.git('show', 'refs/remotes/origin/main:racing.txt'), 'concurrent append\n')
        receipts = [json.loads(f.read_text()) for f in (self.root / 'telemetry/publisher-receipts').rglob('git-*.json')]
        self.assertTrue(any(x['argv'][1] == 'push' and x['exit_code'] != 0 for x in receipts))
        self.assertFalse(any('--force' in x['argv'] for x in receipts))

    def test_identical_partial_connector_snapshot_can_be_completed_without_rewriting_payload(self):
        folder = self.ledger / 'runs/fixture-1'; folder.mkdir(parents=True)
        for n in ('application.log', 'manifest.json', 'receipt.json'): shutil.copyfile(self.root / 'payload' / n, folder / n)
        self.git('add', 'runs'); self.git('commit', '-m', 'connector three-file recovery'); self.git('push', 'origin', 'main')
        self.assertEqual(self.publish(), 0)
        self.assertEqual(self.git('show', 'refs/remotes/origin/main:runs/fixture-1/application.log'), '{}\n')
        self.assertIn('telemetry.json', self.git('show', 'refs/remotes/origin/main:runs/fixture-1/bundle.sha256'))

    def test_transport_failure_keeps_raw_error_and_releases_owned_lock(self):
        self.git('remote', 'set-url', 'origin', str(self.root / 'missing.git'))
        self.assertEqual(self.publish(), 74)
        errors = [f.read_bytes() for f in (self.root / 'telemetry/publisher-receipts').rglob('git-*.stderr.bin')]
        self.assertTrue(any(b'does not appear to be a git repository' in b for b in errors))
        self.assertFalse((self.root / 'telemetry/push.lock').exists())

    def test_split_log_retains_original_part_schema_and_is_idempotent(self):
        Path(self.args[4]).write_bytes(b'a' * 120)
        with patch.object(p, 'CHUNK_BYTES', 45):
            self.assertEqual(self.publish(), 0)
            self.assertEqual(self.publish(), 0)
        parts = self.root / 'telemetry/spool/fixture-1/log-parts'
        self.assertEqual(b''.join(f.read_bytes() for f in sorted(parts.glob('application.log.part.*'))), b'a' * 120)

class PortableTests(unittest.TestCase):
    def test_source_compiles_and_invalid_identity_stops_before_spool_creation(self):
        compile(Path(p.__file__).read_bytes(), str(p.__file__), 'exec')
        with self.assertRaises(p.Stop) as error: p.publish(['..', 'FIXTURE', 'FAILED', '1', 'log', 'manifest', 'receipt'])
        self.assertEqual(error.exception.code, 65)

if __name__ == '__main__': unittest.main()
