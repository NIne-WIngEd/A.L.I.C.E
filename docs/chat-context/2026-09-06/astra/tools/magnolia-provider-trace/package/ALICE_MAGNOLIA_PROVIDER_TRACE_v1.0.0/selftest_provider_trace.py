"""Raw receipt and state-preservation tests. All scheduler results are synthetic."""
import base64
import contextlib
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import collect_provider_trace as collect
import trace_provider as trace


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class ProviderTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.policy = json.loads((collect.BASE / 'trace_policy.json').read_bytes())
        self.remote = self.root / 'remote'; self.remote.mkdir()
        self.vault = self.root / 'vault'; self.vault.mkdir()
        self.key = self.root / 'native-windows-identity'; self.key.write_bytes(b'SYNTHETIC_TEST_NOT_A_KEY')
        self.invoked = []
        self.expected = {tuple(s['argv']): s['name'] for s in self.policy['commands']}
        state = self.vault / self.policy['local_inventory']['files'][0]['path']
        state.parent.mkdir(parents=True); state.write_bytes(b'{"fixture":"SYNTHETIC_LOCAL_STATE"}\n')
        rev = self.vault / self.policy['local_inventory']['groups'][0]['path'] / 'a2-origin-v103-02ba19cdef70'
        rev.mkdir(parents=True)
        (rev / 'intent.json').write_bytes(b'{"fixture":"SYNTHETIC_PRIOR_INTENT"}\n')

    def tearDown(self): self.temp.cleanup()

    def scheduler(self, argv, **kwargs):
        name = self.expected[tuple(argv)]  # Unknown query shapes fail this fixture.
        self.invoked.append(name)
        if name == 'source_job_queue_comparison_only':
            return subprocess.CompletedProcess(argv, 87, b'', b'\xffSYNTHETIC rejection; actual Magnolia cause unknown\n')
        if name == 'active_user_queue':
            return subprocess.CompletedProcess(argv, 0, b'999|SYNTHETIC_OTHER_RUN|RUNNING\n', b'')
        if name == 'source_historical_accounting':
            return subprocess.CompletedProcess(argv, 0, b'575089|' + self.policy['source_id'].encode() + b'|FAILED|76:0|\n', b'')
        return subprocess.CompletedProcess(argv, 0, b'', b'')

    def args(self, name='output'):
        return SimpleNamespace(output=self.root / name, vault_root=self.vault, ssh_key=self.key)

    def bundle(self, args):
        archive, = list(args.output.glob('ALICE_MAGNOLIA_PROVIDER_TRACE_*.zip'))
        with zipfile.ZipFile(archive) as z:
            raw = z.read('report.json'); manifest = json.loads(z.read('EVIDENCE_MANIFEST.json'))
        self.assertEqual(manifest['files'][0]['sha256'], trace.sha(raw))
        self.assertEqual(manifest['files'][0]['bytes'], len(raw))
        return json.loads(raw)

    def test_exact_distinct_query_shapes_preserve_error_bytes_and_all_other_receipts(self):
        before = files(self.remote)
        report = trace.observe_remote(self.policy, self.remote, self.scheduler)
        self.assertEqual(len(self.invoked), 11)
        indexed = {r['name']: r for r in report['commands']}
        bad = indexed['source_job_queue_comparison_only']
        self.assertEqual(bad['argv'], ['squeue', '-h', '-j', '575089', '-o', '%A|%j|%T'])
        self.assertEqual(bad['exit_code'], 87)
        self.assertEqual(base64.b64decode(bad['stderr']['base64']), b'\xffSYNTHETIC rejection; actual Magnolia cause unknown\n')
        self.assertTrue(indexed['active_user_queue']['query_succeeded'])
        self.assertIn('SYNTHETIC_OTHER_RUN', indexed['active_user_queue']['stdout']['text'])
        self.assertTrue(indexed['source_historical_accounting']['query_succeeded'])
        self.assertFalse(report['execution_authority_granted'])
        self.assertEqual(files(self.remote), before)

    def test_scheduler_timeout_retains_partial_raw_streams(self):
        def timeout(argv, **kwargs): raise subprocess.TimeoutExpired(argv, 20, output=b'partial out\n', stderr=b'partial err\xff')
        report = trace.command_receipt(self.policy['commands'][7], timeout)
        self.assertEqual(report['status'], 'TIMEOUT'); self.assertIsNone(report['exit_code'])
        self.assertEqual(base64.b64decode(report['stdout']['base64']), b'partial out\n')
        self.assertEqual(base64.b64decode(report['stderr']['base64']), b'partial err\xff')
        self.assertFalse(report['query_succeeded'])

    def test_capture_limit_is_explicit_and_full_stream_hash_is_not_prefix_hash(self):
        raw = b'abcdef'; report = trace.byte_stream(raw, 3)
        self.assertTrue(report['truncated']); self.assertEqual(report['bytes'], 6)
        self.assertEqual(base64.b64decode(report['base64']), b'abc')
        self.assertEqual(report['sha256'], trace.sha(raw))
        self.assertNotEqual(report['sha256'], trace.sha(b'abc'))

    def test_missing_nonregular_unreadable_and_linked_parent_are_distinct(self):
        (self.remote / 'directory').mkdir(); (self.remote / 'file').write_bytes(b'raw')
        self.assertEqual(trace.file_fact(self.remote, 'missing')['status'], 'MISSING')
        self.assertEqual(trace.file_fact(self.remote, 'directory')['status'], 'NON_REGULAR_FILE')
        with mock.patch.object(Path, 'open', side_effect=PermissionError('synthetic denial')):
            self.assertEqual(trace.file_fact(self.remote, 'file')['status'], 'READ_ERROR')
        with mock.patch.object(trace, 'guard', return_value='LINKED_PARENT'):
            self.assertEqual(trace.file_fact(self.remote, 'file')['status'], 'LINKED_PARENT')
        with self.assertRaises(ValueError): trace.file_fact(self.remote, '../outside')

    def test_partial_revision_is_observed_byte_exact_without_repair(self):
        before = files(self.vault)
        inventory = trace.inventory(self.vault, 'local', self.policy)
        key = next(k for k in inventory['files'] if k.endswith('/intent.json'))
        fact = inventory['files'][key]
        self.assertEqual(base64.b64decode(fact['base64']), before[key])
        self.assertFalse(fact['expected_hash_matches'])
        self.assertEqual(inventory['files'][key.rsplit('/', 1)[0] + '/remote-receipt.json']['status'], 'MISSING')
        self.assertEqual(files(self.vault), before)

    def test_concurrent_state_change_is_reported_without_changing_its_bytes(self):
        target = self.remote / 'runs' / self.policy['current_id'] / 'submission-intent.json'
        def changed(argv, **kwargs):
            if not target.exists():
                target.parent.mkdir(parents=True); target.write_bytes(b'{"fixture":"SYNTHETIC_OTHER_ACTOR"}')
            return self.scheduler(argv, **kwargs)
        report = trace.observe_remote(self.policy, self.remote, changed)
        self.assertFalse(report['inventories_equal'])
        key = target.relative_to(self.remote).as_posix()
        self.assertEqual(report['inventory_before']['files'][key]['status'], 'MISSING')
        self.assertEqual(report['inventory_after']['files'][key]['sha256'], trace.sha(target.read_bytes()))

    def test_client_uses_native_key_and_script_stdin_and_retains_provider_failure_bundle(self):
        remote = trace.observe_remote(self.policy, self.remote, self.scheduler)
        before = files(self.vault); args = self.args()
        def ssh(argv, **kwargs):
            self.assertEqual(argv, collect.ssh_argv(self.key))
            self.assertEqual(argv[-1], 'bash -s')
            self.assertEqual(argv[argv.index('-i') + 1], str(self.key))
            self.assertEqual(kwargs['input'], collect.remote_script(self.policy))
            self.assertNotIn(b'\r', kwargs['input'])
            return subprocess.CompletedProcess(argv, 0, trace.canonical(remote), b'SYNTHETIC_SSH_NOTICE\n')
        with mock.patch.object(collect.shutil, 'which', return_value='ssh'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(collect.execute(args, ssh), 0)
        result = self.bundle(args)
        self.assertEqual(result['status'], 'COLLECTED_PROVIDER_ERRORS')
        self.assertEqual(result['remote']['provider_command_failures'], 1)
        self.assertTrue(result['local_inventories_equal']); self.assertEqual(files(self.vault), before)
        self.assertEqual(base64.b64decode(result['transport']['stderr']['base64']), b'SYNTHETIC_SSH_NOTICE\n')

    def test_transport_error_timeout_and_invalid_json_all_preserve_existing_state_and_raw_bytes(self):
        before = files(self.vault)
        for mode in ('nonzero', 'timeout', 'invalid_json', 'wrong_json_type'):
            with self.subTest(mode=mode):
                args = self.args(mode)
                def ssh(argv, **kwargs):
                    if mode == 'timeout': raise subprocess.TimeoutExpired(argv, 300, output=b'partial', stderr=b'detail')
                    payload = b'[]' if mode == 'wrong_json_type' else b'not-json\xff'
                    return subprocess.CompletedProcess(argv, 255 if mode == 'nonzero' else 0, payload, b'exact ssh detail')
                with mock.patch.object(collect.shutil, 'which', return_value='ssh'), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(collect.execute(args, ssh), 74)
                result = self.bundle(args)
                expected = b'partial' if mode == 'timeout' else b'[]' if mode == 'wrong_json_type' else b'not-json\xff'
                self.assertEqual(base64.b64decode(result['transport']['stdout']['base64']), expected)
                self.assertEqual(files(self.vault), before)
                self.assertEqual(result['model_jobs_submitted'], 0)

    def test_missing_tools_still_preserve_local_evidence_without_attempting_ssh(self):
        args = self.args(); command = mock.Mock()
        with mock.patch.object(collect.shutil, 'which', return_value=None), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(collect.execute(args, command), 74)
        command.assert_not_called()
        result = self.bundle(args)
        self.assertEqual(result['transport']['status'], 'EXEC_ERROR')
        self.assertTrue(result['local_inventories_equal'])


if __name__ == '__main__': unittest.main(verbosity=2)
