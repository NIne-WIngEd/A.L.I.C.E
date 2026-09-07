"""Cross-platform regressions for actual prior infrastructure failures. No live services."""
from pathlib import Path, PureWindowsPath
import os
import ssl
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.request
import zipfile

import contract as c
import controller
import infra_recovery as infra
import remote_agent as remote
import secure_https as tls
import telemetry_snapshots as telemetry
import worker
import fixture_origins


class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = mock.patch.object(c, 'ROOT', self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def source(self):
        folder, _, _ = fixture_origins.install_sources(self.root)
        return folder

    def test_source_is_exact_zero_inference_failure_and_remains_unchanged(self):
        source = self.source()
        before = {p.relative_to(source).as_posix(): c.file_sha(p) for p in source.rglob('*') if p.is_file()}
        infra.verify_source_files(source)
        after = {p.relative_to(source).as_posix(): c.file_sha(p) for p in source.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_any_prior_probe_or_task_intent_blocks_successor(self):
        source = self.source()
        for name in ('probe-attempt.json', 'tasks/Q01_COMPATIBLE_NOVELTY/attempt.json'):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
            with self.assertRaises(c.Stop):
                infra.verify_source_files(source)
            path.unlink()

    def test_changed_parent_result_is_refused(self):
        source = self.source()
        (source / 'result.json').write_bytes((source / 'result.json').read_bytes() + b' ')
        with self.assertRaises(c.Stop):
            infra.verify_source_files(source)

    def test_live_source_scheduler_identity_must_match_closed_failure(self):
        self.source()
        def command(argv, check=True):
            return subprocess.CompletedProcess(argv, 0, '' if argv[0] == 'squeue' else '575089|' + c.CALIBRATION_ID + '|FAILED|76:0|\n', '')
        with mock.patch.object(c, 'ROOT', self.root):
            self.assertTrue(infra.reconcile_source(command)['zero_inference_confirmed'])
            for row in ('575089|' + c.CALIBRATION_ID + '|RUNNING|0:0|\n', '575089|wrong|FAILED|76:0|\n', ''):
                def bad(argv, check=True):
                    return subprocess.CompletedProcess(argv, 0, row if argv[0] == 'sacct' else '', '')
                with self.assertRaises(c.Stop):
                    infra.reconcile_source(bad)

    def test_failed_preflight_cannot_reach_sbatch_or_write_submission_intent(self):
        run = self.root / 'run'
        fixture_origins.completed_revision(run, '1' * 64)
        helper = self.root / 'bin/rayan-telemetry-push.sh'
        helper.parent.mkdir(); helper.write_text('fixture'); helper.chmod(0o700)
        (self.root / 'telemetry/ledger/.git').mkdir(parents=True)
        with mock.patch.object(c, 'ROOT', self.root), mock.patch.object(remote, 'submission_lock', controller.controller_lock), mock.patch.object(remote, 'find_submitted', return_value=None), mock.patch.object(remote.shutil, 'which', return_value='/fixture'), mock.patch.object(infra, 'preflight', side_effect=c.Stop('fixture TLS failure')), mock.patch.object(remote, 'cmd') as command:
            with self.assertRaises(c.Stop):
                remote.submit(run, '1' * 64)
        command.assert_not_called()
        self.assertFalse((run / 'submission-intent.json').exists())
        self.assertFalse(c.read(run / 'infrastructure-preflight-failure.json')['job_submitted'])

    def test_successful_preflight_captures_source_tls_and_frozen_manifest_together(self):
        self.source()
        run = self.root / 'next'; run.mkdir()
        raw = c.canonical({'schemaVersion': 2, 'config': {'digest': 'sha256:' + 'a' * 64, 'size': 1}, 'layers': []}).encode()
        def command(argv, check=True):
            return subprocess.CompletedProcess(argv, 0, '' if argv[0] == 'squeue' else '575089|' + c.CALIBRATION_ID + '|FAILED|76:0|\n', '')
        calls = []
        def https(request, timeout=60):
            calls.append((request.get_method(), request.full_url))
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.status = 200
            response.geturl.return_value = 'https://release-assets.githubusercontent.com/fixture'
            response.read.return_value = raw
            response.headers = {'Docker-Content-Digest': 'sha256:' + c.sha(raw)}
            return response
        original_configure = tls.configure
        with mock.patch.object(c, 'ROOT', self.root), mock.patch.dict(os.environ), mock.patch.object(tls, 'configure', side_effect=lambda path: original_configure(path, (c.BASE / 'fixtures/test-ca.pem',))), mock.patch.object(tls, 'open_url', side_effect=https):
            receipt = infra.preflight(run, command)
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual([method for method, _ in calls], ['HEAD', 'GET'])
        self.assertEqual(receipt['model_manifest_sha256'], c.sha(raw))
        (run / 'authority').mkdir()
        (run / 'authority/infra_recovery.json').write_bytes((c.BASE / 'authority/infra_recovery.json').read_bytes())
        infra.verify_evidence(run)
        self.assertFalse((run / 'submission-intent.json').exists())

    def test_explicit_ca_context_keeps_chain_and_hostname_checks(self):
        with mock.patch.dict(os.environ):
            receipt = tls.configure(self.root, (c.BASE / 'fixtures/test-ca.pem',))
            context = tls.context()
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            self.assertGreater(context.cert_store_stats()['x509_ca'], 0)
            self.assertEqual(receipt['ca_sha256'], c.file_sha(c.BASE / 'fixtures/test-ca.pem'))
            self.assertEqual(os.environ['PIP_CERT'], os.environ['SSL_CERT_FILE'])

    def test_missing_or_invalid_ca_stops_without_network(self):
        with mock.patch.object(urllib.request, 'build_opener') as network:
            with self.assertRaises(c.Stop):
                tls.configure(self.root, (self.root / 'absent',))
            bad = self.root / 'invalid.pem'; bad.write_text('not a certificate')
            with self.assertRaises(ssl.SSLError):
                tls.configure(self.root, (bad,))
        network.assert_not_called()

    def test_external_https_downgrade_is_refused(self):
        with self.assertRaises(c.Stop):
            tls.open_url('http://fixture.invalid')
        with self.assertRaises(c.Stop):
            tls.HTTPSOnlyRedirect().redirect_request(urllib.request.Request('https://fixture.invalid'), None, 302, '', {}, 'http://fixture.invalid')

    def test_windows_paths_stay_native_only_for_local_arguments(self):
        root = PureWindowsPath('/homes/01/mxrayan/rayan-compute')
        key = PureWindowsPath('C:/Users/rayns/.ssh/key')
        with mock.patch.object(c, 'ROOT', root), mock.patch.object(c, 'REMOTE_RUN', root / 'runs' / c.RUN_ID):
            remote_client = controller.Magnolia(key, '1' * 64)
            self.assertTrue(remote_client.directory.startswith('/homes/'))
            self.assertNotIn('\\', remote_client.remote_zip)
            self.assertEqual(remote_client.options[-1], str(key))
            result = self.root / 'result.zip'; result.write_bytes(b'fixture')
            receipt = {'remote_path': root.as_posix() + '/runs/' + c.RUN_ID + '/result-bundle.zip', 'zip_sha256': c.file_sha(result), 'bytes': result.stat().st_size}
            with mock.patch.object(remote_client, 'execute', return_value=subprocess.CompletedProcess([], 0)):
                remote_client.download(receipt, result)

    def test_cached_manifest_reuses_frozen_bytes_without_refetch(self):
        run = self.root
        raw = c.canonical({'schemaVersion': 2, 'config': {'digest': 'sha256:' + 'a' * 64, 'size': 1}, 'layers': []}).encode()
        c.atomic_bytes(run / 'model-manifest.json', raw)
        lock = {'source_url': 'https://registry.ollama.ai/v2/library/qwen3.8/manifests/27b-q4_K_M', 'model_tag': c.TAG, 'approved_amendment_sha256': c.APPROVED_SHA, 'inference_started': False, 'full_manifest_digest': c.sha(raw), 'manifest_sha256': c.sha(raw), 'registry_header_digest': 'sha256:' + c.sha(raw)}
        c.write(run / 'model-lock.json', lock)
        with mock.patch.object(tls, 'open_url') as network:
            observed, _, bytes_again = worker.resolve_manifest(run)
            self.assertEqual(observed, lock); self.assertEqual(bytes_again, raw)
            (run / 'model-manifest.json').write_bytes(raw + b' ')
            with self.assertRaises(c.Stop):
                worker.resolve_manifest(run)
        network.assert_not_called()

    def test_probe_and_all_sixteen_requests_match_original_v100_hashes(self):
        hashes = c.read(c.BASE / 'authority/request_hashes_v100.json')
        approved, tasks, _ = c.authority()
        self.assertEqual(c.sha(c.canonical(c.probe_request(approved, tasks)).encode()), hashes['probe_sha256'])
        self.assertEqual({t['task_id']: c.sha(c.canonical(c.request(t, i, approved)).encode()) for i, t in enumerate(tasks, 1)}, hashes['tasks'])

    def test_changed_snapshot_uses_new_id_and_retry_preserves_exact_bytes(self):
        run = self.root / 'run'; run.mkdir()
        c.write(run / 'run.json', {'package_sha256': '1' * 64})
        original = {'schema': 'rayan.compute.run-receipt.v1', 'run_id': c.RUN_ID, 'stage': telemetry.STAGE, 'status': 'RUNNING', 'slurm_job_id': '12345', 'phase': 'PREFLIGHT', 'tasks_attempted': 0}
        ledger = {}; calls = []
        def helper(argv, **kwargs):
            identity = argv[1]; current = tuple(Path(p).read_bytes() for p in argv[5:]); calls.append((identity, current))
            if identity in ledger and ledger[identity] != current:
                return subprocess.CompletedProcess(argv, 73, b'RUN_ID_COLLISION', b'')
            ledger[identity] = current
            return subprocess.CompletedProcess(argv, 0, b'', b'')
        with mock.patch.object(telemetry.subprocess, 'run', side_effect=helper):
            self.assertEqual(telemetry.publish(run, original), 0)
            first = c.read(run / 'telemetry-publication.json')
            self.assertEqual(telemetry.publish(run, {**original, 'status': 'FAILED', 'phase': 'TERMINAL_COLLECTION'}), 0)
            second = c.read(run / 'telemetry-publication.json')
            self.assertNotEqual(first['snapshot_id'], second['snapshot_id'])
            self.assertEqual(telemetry.publish_stored(run, run / second['snapshot_path']), 0)
            self.assertEqual(calls[-1], calls[-2])
            self.assertEqual(len(ledger), 2)
            (run / second['snapshot_path'] / 'receipt.json').write_text('changed')
            with self.assertRaises(c.Stop):
                telemetry.publish_stored(run, run / second['snapshot_path'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
