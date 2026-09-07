"""Tests for the diagnostic's actual observation and transport boundaries."""
import contextlib
import io
import json
from pathlib import Path, PureWindowsPath
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.request
import zipfile

import collect_trace as client
import remote_trace as remote


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.policy = json.loads((client.BASE / 'trace_policy.json').read_bytes())
        self.policy['remote_root'] = str(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_origin_report_distinguishes_export_name_from_package_manifest(self):
        run = self.root / 'runs' / self.policy['source_id']; run.mkdir(parents=True)
        pkg = self.policy['packages']['source']
        base = self.root / 'packages' / pkg['run_id'] / pkg['zip_sha256'] / pkg['name']
        base.mkdir(parents=True)
        data = b'fixture package manifest\n'
        (base / 'PACKAGE_MANIFEST.json').write_bytes(data)
        pkg['manifest_sha256'] = remote.sha(data)
        self.policy['source_critical_files']['package-manifest.json'] = remote.sha(data)
        before = {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        commands = []
        def command(argv, **kwargs):
            commands.append(argv)
            return subprocess.CompletedProcess(argv, 0, '', '')
        result = remote.inspect(self.policy, command=command, network=lambda _: {'status': 'MOCKED'})
        self.assertEqual(result['source_files']['package-manifest.json']['status'], 'MISSING')
        self.assertEqual(result['packages']['source']['files']['PACKAGE_MANIFEST.json']['status'], 'MATCH')
        self.assertEqual([argv[0] for argv in commands], ['squeue', 'sacct', 'sacct'])
        self.assertTrue(all('--starttime' in argv for argv in commands[1:]))
        self.assertEqual(before, {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse(result['execution_authority_granted'])

    def test_file_mismatch_unreadable_and_non_regular_are_distinct(self):
        path = self.root / 'file'; path.write_bytes(b'changed')
        self.assertEqual(remote.file_fact(path, '1' * 64)['status'], 'HASH_MISMATCH')
        self.assertEqual(remote.file_fact(self.root)['status'], 'NOT_REGULAR_FILE')
        self.assertEqual(remote.file_fact(self.root / 'missing')['status'], 'MISSING')
        with mock.patch.object(Path, 'read_bytes', side_effect=PermissionError):
            self.assertEqual(remote.file_fact(path)['status'], 'READ_ERROR')

    def test_scheduler_failure_is_unknown_and_rows_include_wrong_source_identity(self):
        def bad(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, '', 'unavailable')
        self.assertFalse(remote.scheduler(['squeue'], (self.policy['source_id'],), bad)['query_succeeded'])
        def rows(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, '575089|unexpected_name|FAILED|76:0|\n12345|unrelated|RUNNING|0:0|\n', '')
        found = remote.scheduler(['sacct'], (self.policy['source_id'],), rows)['matching_rows']
        self.assertEqual(found, [['575089', 'unexpected_name', 'FAILED', '76:0']])

    def test_no_ca_makes_no_network_request_and_http_downgrade_is_refused(self):
        self.policy['ca_paths'] = [str(self.root / 'absent.pem')]
        with mock.patch.object(urllib.request, 'build_opener') as network:
            self.assertEqual(remote.https_heads(self.policy)['status'], 'NO_READABLE_CA')
        network.assert_not_called()
        with self.assertRaises(ValueError):
            remote.HTTPSOnly().redirect_request(urllib.request.Request('https://example.invalid'), None,
                302, '', {}, 'http://example.invalid')

    def test_ssh_uses_script_stdin_and_preserves_native_windows_key(self):
        key = PureWindowsPath('C:/Users/rayns/.ssh/rayan_magnolia_ed25519')
        argv = client.ssh_argv(key)
        self.assertEqual(argv[argv.index('-i') + 1], str(key))
        self.assertEqual(argv[-1], 'bash -s')
        script = client.remote_script(self.policy).decode()
        body = script.split("<<'ALICE_READ_ONLY_PY'\n", 1)[1].rsplit('\nALICE_READ_ONLY_PY\n', 1)[0]
        compile(body, '<literal ssh stdin>', 'exec')
        self.assertIn('exec "$PY" -B -', script)
        # The real inspect() test above checks every invoked command; a text
        # search here would confuse reading job.sbatch with executing sbatch.

    def test_local_collection_and_ssh_failure_preserve_existing_controller_state(self):
        vault = self.root / 'vault'
        policy = json.loads((client.BASE / 'trace_policy.json').read_bytes())
        files = []
        for suffix in ('a1', 'a2'):
            path = vault / 'tools/alice-astra' / ('qwen-fallback-' + suffix) / 'controller-state.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'phase': 'PREPARED', 'staged': True, 'package_sha256': '1' * 64}))
            files.append(path)
        before = [p.read_bytes() for p in files]
        key = self.root / 'key'; key.write_text('fixture')
        response = {'schema': 'alice.qwen.read-only-remote-trace.v1', 'read_only': True,
                    'source_id': policy['source_id'], 'current_id': policy['current_id']}
        for index, exit_code in enumerate((0, 255)):
            output = self.root / ('output' + str(index)); output.mkdir()
            args = type('Args', (), {'vault_root': vault, 'ssh_key': key, 'output': output})()
            def execute(argv, **kwargs):
                self.assertEqual(argv[0], 'ssh')
                self.assertIsInstance(kwargs['input'], bytes)
                self.assertNotIn('shell', kwargs)
                return subprocess.CompletedProcess(argv, exit_code, json.dumps(response).encode(), b'')
            with mock.patch.object(client.shutil, 'which', return_value='ssh'), \
                 mock.patch.object(client.subprocess, 'run', side_effect=execute), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(client.execute(args), 0 if exit_code == 0 else 74)
            with zipfile.ZipFile(next(output.glob('ALICE_QWEN_TRACE_*.zip'))) as z:
                report = json.loads(z.read('report.json'))
                manifest = json.loads(z.read('DIAGNOSTIC_MANIFEST.json'))
                self.assertEqual(manifest['files'][0]['sha256'], remote.sha(z.read('report.json')))
                self.assertTrue(report['local_controller_states_unchanged'])
                self.assertFalse(report['compute_job_submitted_by_diagnostic'])
        self.assertEqual(before, [p.read_bytes() for p in files])


if __name__ == '__main__':
    unittest.main(verbosity=2)
