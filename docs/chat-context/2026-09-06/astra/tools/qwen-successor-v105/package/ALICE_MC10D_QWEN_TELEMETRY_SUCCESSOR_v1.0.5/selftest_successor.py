"""Offline boundary tests; synthetic responses never count as Qwen evidence."""
import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import zipfile

import contract as c
import controller
import evidence
import infra_recovery as infra
import magnolia_scheduler as scheduler
import publisher_capture as capture
import remote_agent as remote
import secure_https
import telemetry_snapshots as telemetry
import worker

FAKE_SHA = '1' * 64
REAL_RUN = subprocess.run


def hashes(root):
    return {p.relative_to(root).as_posix(): c.file_sha(p) for p in root.rglob('*') if p.is_file()}


def make_parents(root):
    rules = infra.policy()
    for parent in rules['parents']:
        run = root / 'runs' / parent['run_id']; run.mkdir(parents=True)
        with zipfile.ZipFile(c.BASE / 'source' / parent['result_archive']) as z:
            for name in parent['critical_files']:
                if name == 'package-manifest.json': continue
                p = run / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(name))
        base = root / 'packages' / parent['run_id']; base.mkdir(parents=True)
        archive = base / (parent['package_sha256'] + '.zip')
        shutil.copyfile(c.BASE / 'source' / parent['package_archive'], archive)
        with zipfile.ZipFile(archive) as z: z.extractall(base / parent['package_sha256'])
    helper = root / 'bin/rayan-telemetry-push.sh'; helper.parent.mkdir()
    shutil.copyfile(c.BASE / 'fixtures/rayan-telemetry-push.sh', helper); helper.chmod(0o700)
    wrapper = helper.parent / 'rayan-github-ssh'; wrapper.write_text('fixture wrapper')
    with zipfile.ZipFile(c.BASE / 'source/ALICE_PublisherRepair_f3dc451ff517.zip') as z:
        report = json.loads(z.read('result.json'))['report']
    for name, item in report['publisher_journals'].items():
        if name.endswith('/result.json'):
            p = root / 'telemetry/publisher-receipts/alice-qwen38-a2-c926d9e355dd-t-161d1ba71ed308ed' / name
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(item['base64']))
    (root / 'telemetry/ledger/.git').mkdir(parents=True)
    # The SSH wrapper fixture is explicitly substituted; no network connection is made.
    return {**rules, 'installed_wrapper_sha256': c.file_sha(wrapper)}


def fake_scheduler(argv, check=True):
    rules = infra.policy()
    if argv == scheduler.USER_QUEUE: return subprocess.CompletedProcess(argv, 0, '', '')
    if argv[0] == 'sacct' and '-j' in argv:
        job = argv[argv.index('-j') + 1]
        p = next(p for p in rules['parents'] if p['job_id'] == job)
        return subprocess.CompletedProcess(argv, 0, '|'.join([job, p['run_id'], 'FAILED', '76:0']) + '\n', '')
    raise AssertionError(argv)


def trust(run):
    doc = {'ca_sha256': '4' * 64, 'verify_mode': 'CERT_REQUIRED', 'check_hostname': True}
    c.write(run / 'tls-trust.json', doc); return doc


class SuccessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='alice-a3-test-')
        self.root = Path(self.temp.name)
        self.rules = make_parents(self.root)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(c, 'ROOT', self.root))
        self.run = self.root / 'runs' / c.RUN_ID; self.run.mkdir()
        self.stack.enter_context(mock.patch.object(c, 'REMOTE_RUN', self.run))
        self.stack.enter_context(mock.patch.object(infra, 'policy', return_value=self.rules))
        self.parent_before = {p['run_id']: hashes(self.root / 'runs' / p['run_id']) for p in self.rules['parents']}

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def preflight(self):
        response = mock.MagicMock(); response.__enter__.return_value = response
        response.status = 200; response.geturl.return_value = 'https://fixture.invalid/runtime'
        with mock.patch.object(secure_https, 'configure', side_effect=trust), mock.patch.object(secure_https, 'open_url', return_value=response):
            return infra.preflight(self.run, fake_scheduler)

    def test_exact_both_parent_origins_and_frozen_manifest_without_source_mutation(self):
        self.preflight()
        self.assertEqual(c.file_sha(self.run / 'model-manifest.json'), self.rules['frozen_model_manifest_sha256'])
        for p in self.rules['parents']:
            self.assertEqual(hashes(self.root / 'runs' / p['run_id']), self.parent_before[p['run_id']])
        self.assertEqual(len(c.read(self.run / 'parent-reconciliation.json')['parents']), 2)
        self.assertFalse((self.run / 'probe-attempt.json').exists())

    def test_prior_probe_attempt_changed_parent_and_unknown_scheduler_each_block(self):
        parent = self.root / 'runs' / self.rules['parents'][1]['run_id']
        (parent / 'probe-attempt.json').write_text('{}')
        with self.assertRaises(c.Stop): infra.reconcile_source(fake_scheduler)
        (parent / 'probe-attempt.json').unlink()
        old = (parent / 'result.json').read_bytes(); (parent / 'result.json').write_bytes(old + b' ')
        with self.assertRaises(c.Stop): infra.reconcile_source(fake_scheduler)
        (parent / 'result.json').write_bytes(old)
        with self.assertRaises(c.Stop): infra.reconcile_source(lambda a, check=True: subprocess.CompletedProcess(a, 1, '', 'actual query failed'))

    def test_publisher_drift_prevents_submission_and_preserves_old_states(self):
        (self.root / 'bin/rayan-telemetry-push.sh').write_text('unexpected helper')
        with mock.patch.object(remote, 'submission_lock', controller.controller_lock), mock.patch.object(remote, 'find_submitted', return_value=None), mock.patch.object(remote, 'cmd', side_effect=fake_scheduler), mock.patch.object(remote.shutil, 'which', return_value='/fixture/tool'):
            with self.assertRaises(c.Stop): remote.submit(self.run, FAKE_SHA)
        self.assertFalse((self.run / 'submission-intent.json').exists())
        self.assertEqual(c.read(self.run / 'infrastructure-preflight-failure.json')['status'], 'STOP')

    def test_submission_acknowledgement_loss_never_creates_second_job(self):
        jobs = []; calls = []
        def command(argv, check=True):
            calls.append(argv)
            if argv[0] == 'sbatch':
                jobs.append('12345'); raise controller.TransportPending('lost acknowledgement')
            raise AssertionError(argv)
        with mock.patch.object(remote, 'submission_lock', controller.controller_lock), mock.patch.object(remote, 'find_submitted', side_effect=lambda: jobs[0] if jobs else None), mock.patch.object(remote, 'cmd', side_effect=command), mock.patch.object(remote.shutil, 'which', return_value='/fixture/tool'), mock.patch.object(infra, 'preflight'):
            with self.assertRaises(controller.TransportPending): remote.submit(self.run, FAKE_SHA)
            self.assertEqual(remote.submit(self.run, FAKE_SHA)['job_id'], '12345')
            self.assertEqual(remote.submit(self.run, FAKE_SHA)['job_id'], '12345')
        self.assertEqual(len(calls), 1)

    def test_unresolved_intent_never_resubmits(self):
        c.write(self.run / 'submission-intent.json', {'unknown': True})
        with mock.patch.object(remote, 'submission_lock', controller.controller_lock), mock.patch.object(remote, 'find_submitted', return_value=None), mock.patch.object(remote, 'cmd') as submit:
            with self.assertRaises(c.Stop): remote.submit(self.run, FAKE_SHA)
            submit.assert_not_called()

    def test_original_probe_and_all_task_requests_are_byte_exact(self):
        approved, tasks, _ = c.authority(); expected = c.read(c.BASE / 'authority/request_hashes_v100.json')
        self.assertEqual(c.sha(c.canonical(c.probe_request(approved, tasks)).encode()), expected['probe_sha256'])
        self.assertEqual({t['task_id']: c.sha(c.canonical(c.request(t, i, approved)).encode()) for i, t in enumerate(tasks, 1)}, expected['tasks'])

    def receipt(self):
        c.write(self.run / 'run.json', infra.descriptor(FAKE_SHA))
        return {'schema': 'rayan.compute.run-receipt.v1', 'run_id': c.RUN_ID, 'stage': telemetry.STAGE,
            'status': 'FAILED', 'provider': 'magnolia', 'slurm_job_id': '12345', 'phase': 'FIXTURE',
            'tasks_completed': 0, 'tasks_attempted': 0, 'tasks_required': 16}

    def test_raw_nonzero_and_timeout_bytes_survive_collection(self):
        for response in (subprocess.CompletedProcess([], 74, b'outer raw stdout\x00', b'raw error\xff'), subprocess.TimeoutExpired('helper', 90, output=b'partial stdout', stderr=b'partial stderr')):
            with self.subTest(response=type(response).__name__):
                patch = mock.patch.object(capture.subprocess, 'run', side_effect=response) if isinstance(response, Exception) else mock.patch.object(capture.subprocess, 'run', return_value=response)
                with patch: self.assertNotEqual(telemetry.publish(self.run, self.receipt()), 0)
                capture.evidence_paths(self.run); capture.verify_evidence(self.run)
        records = [c.strict(x) for x in (self.run / 'publisher-invocations.jsonl').read_bytes().splitlines()]
        self.assertEqual(base64.b64decode(records[0]['stderr']['base64']), b'raw error\xff')
        self.assertEqual(base64.b64decode(records[1]['stdout']['base64']), b'partial stdout')

    @unittest.skipIf(os.name == 'nt', 'Actual Git publisher integration runs in the Linux release build; installed Magnolia publisher already verified')
    def test_real_git_publisher_to_capture_to_archive_and_identical_retry(self):
        spec = importlib.util.spec_from_file_location('checked_fixture_publisher', c.BASE / 'fixtures/publisher.py')
        publisher = importlib.util.module_from_spec(spec); spec.loader.exec_module(publisher)
        ledger = self.root / 'telemetry/ledger'; (ledger / '.git').rmdir()
        def git(*args, cwd=ledger):
            return REAL_RUN(['git', *args], cwd=cwd, capture_output=True, check=True).stdout.decode().strip()
        git('init'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid'); git('config', 'commit.gpgsign', 'false')
        (ledger / 'README').write_text('preserve')
        git('add', 'README'); git('commit', '-m', 'fixture'); git('branch', '-M', 'main')
        bare = self.root / 'remote.git'; git('init', '--bare', str(bare)); git('remote', 'add', 'origin', str(bare)); git('push', 'origin', 'main')
        original = git('rev-parse', 'HEAD'); index = (ledger / '.git/index').read_bytes()
        def engine(argv, **kwargs):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()): code = publisher.publish(argv[1:], self.root)
            return subprocess.CompletedProcess(argv, code, out.getvalue().encode(), b'')
        with mock.patch.object(capture.subprocess, 'run', side_effect=engine):
            self.assertEqual(telemetry.publish(self.run, self.receipt()), 0)
            snapshot = c.read(self.run / 'telemetry-publication.json')['snapshot_path']
            self.assertEqual(telemetry.publish_stored(self.run, self.run / snapshot), 0)
        capture.evidence_paths(self.run); capture.verify_evidence(self.run)
        self.assertEqual(git('rev-parse', 'HEAD'), original); self.assertEqual((ledger / '.git/index').read_bytes(), index)
        self.assertNotEqual(git('rev-parse', 'refs/remotes/origin/main'), original)
        with zipfile.ZipFile(self.run / 'publisher-journals.zip') as z:
            results = [c.strict(z.read(n)) for n in z.namelist() if n.endswith('/result.json')]
            self.assertEqual(sorted(x['idempotent'] for x in results), [False, True])

    def test_native_key_and_remote_posix_address_and_failure_stop_zip(self):
        key = Path(r'C:\Users\rayns\.ssh\rayan_magnolia_ed25519')
        remote_client = controller.Magnolia(key, FAKE_SHA, self.run / 'controller-diagnostics.jsonl')
        raw = b'{"run_id":"' + c.RUN_ID.encode() + b'","message":"exact stop","last_scheduler_command":{"stderr":"raw reason"}}\n'
        with mock.patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 76, raw, b'raw ssh stderr')) as process:
            with self.assertRaises(c.Stop): remote_client.action('submit')
        args = process.call_args.args[0]
        self.assertEqual(args[args.index('-i') + 1], str(key)); self.assertIn('bash -s', args[-1])
        self.assertNotIn('\\', remote_client.directory)
        path = self.root / 'tools/alice-astra/qwen-fallback-a3/runs' / c.RUN_ID
        path.mkdir(parents=True); shutil.copyfile(self.run / 'controller-diagnostics.jsonl', path / 'controller-diagnostics.jsonl')
        with contextlib.redirect_stdout(io.StringIO()): controller.preserve_stop(SimpleNamespace(vault_root=self.root, output_root=self.root), c.Stop('fixture stop'))
        with zipfile.ZipFile(next(self.root.glob('ALICE_QWEN_A3_STOP_*.zip'))) as z:
            record = c.strict(z.read('controller-diagnostics.jsonl'))
            self.assertEqual(base64.b64decode(record['stdout']['base64']), raw)

    def simulate(self, slow=False, truncated=False):
        self.preflight(); approved, tasks, policy = c.authority()
        c.write(self.run / 'run.json', infra.descriptor(FAKE_SHA))
        lock = c.read(self.run / 'model-lock.json'); manifest = c.read(self.run / 'model-manifest.json')
        model = {'name': c.TAG, 'digest': lock['full_manifest_digest'], 'details': {'family': 'qwen35', 'quantization_level': 'Q4_K_M'}, 'size_vram': 0, 'context_length': 8192}
        def runtime(run, policy, deadline):
            c.write(run / 'runtime-files.json', {'files': [{'path': 'bin/ollama', 'sha256': policy['binary_sha256']}]})
            c.write(run / 'runtime-decoder.json', {'backend': 'system_tar_zstd', 'pinned_runtime_archive_verified': True})
            return run / 'fixture-binary', {'archive_sha256': policy['archive_sha256'], 'binary_sha256': policy['binary_sha256'], 'runtime_files_sha256': c.file_sha(run / 'runtime-files.json'), 'runtime_decoder_sha256': c.file_sha(run / 'runtime-decoder.json')}
        def prepare_model(run, policy, deadline):
            c.write(run / 'model-blobs.json', {'manifest_sha256': lock['full_manifest_digest'], 'blobs': [{'digest': b['digest'], 'bytes': b['size']} for b in [manifest['config']] + manifest['layers']]})
            return self.root / 'fixture-models', lock
        def api(base, endpoint, body=None, timeout=60):
            return {'details': model['details'], 'capabilities': ['completion', 'thinking']} if endpoint == '/api/show' else {'models': [model]}
        calls = []
        def stream(base, body, path, deadline):
            calls.append(body)
            probe = path.name == 'probe-response.ndjson'
            answer = {} if probe else {**next(t for t in tasks if t['task_id'] == path.parent.name)['gold'], 'rationale': 'Synthetic test only.'}
            chunk = {'model': c.TAG, 'message': {'role': 'assistant', 'content': c.canonical(answer)}, 'done': True,
                'done_reason': 'length' if probe or truncated else 'stop', 'total_duration': 2_000_000_000,
                'load_duration': 1, 'prompt_eval_count': 256, 'prompt_eval_duration': 100_000_000,
                'eval_count': 128, 'eval_duration': 200_000_000_000 if slow else 1_000_000_000}
            c.atomic_bytes(path, c.canonical(chunk).encode())
        def fake_publisher(argv, **kwargs):
            journal = self.root / 'telemetry/publisher-receipts' / argv[1] / uuid.uuid4().hex; journal.mkdir(parents=True)
            c.write(journal / 'result.json', {'run_id': argv[1], 'exit_code': 0, 'published': True})
            return subprocess.CompletedProcess(argv, 0, ('TELEMETRY_RECEIPT_DIR=' + str(journal) + '\n').encode(), b'')
        proc = mock.Mock(); proc.poll.return_value = 0
        with mock.patch.dict(os.environ, SLURM_JOB_ID='12345', SLURM_CPUS_PER_TASK='20', SLURM_JOB_PARTITION='node', SLURM_MEM_PER_NODE='49152', SLURM_JOB_GPUS=''), mock.patch.object(worker.os, 'sched_getaffinity', return_value=set(range(20)), create=True), mock.patch.object(secure_https, 'configure', side_effect=trust), mock.patch.object(worker, 'prepare_runtime', side_effect=runtime), mock.patch.object(worker, 'prepare_model', side_effect=prepare_model), mock.patch.object(worker, 'start_service', return_value=(proc, 'http://fixture.invalid', policy['version'])), mock.patch.object(worker, 'api', side_effect=api), mock.patch.object(worker, 'stream_request', side_effect=stream), mock.patch.object(capture.subprocess, 'run', side_effect=fake_publisher), contextlib.redirect_stdout(io.StringIO()):
            code = worker.work(self.run, FAKE_SHA)
            with self.assertRaises(c.Stop): worker.work(self.run, FAKE_SHA)
        c.write(self.run / 'submission.json', {'job_id': '12345', 'package_sha256': FAKE_SHA})
        c.write(self.run / 'scheduler-status.json', {'run_id': c.RUN_ID, 'job_id': '12345', 'terminal': True, 'scheduler_state': 'COMPLETED' if code == 0 else 'FAILED'})
        paths = remote.evidence_paths(self.run)
        export = self.root / 'export'; export.mkdir()
        for name, source in paths.items():
            p = export / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, p)
        self.seal(export)
        return export, calls

    def seal(self, folder):
        c.write(folder / 'EVIDENCE_MANIFEST.json', {'run_id': c.RUN_ID, 'files': [{'path': p.relative_to(folder).as_posix(), 'bytes': p.stat().st_size, 'sha256': c.file_sha(p)} for p in folder.rglob('*') if p.is_file() and p.name != 'EVIDENCE_MANIFEST.json']})

    def test_full_worker_collector_verifier_and_tampered_request_refused(self):
        folder, calls = self.simulate()
        summary = evidence.analyze(folder, FAKE_SHA)
        self.assertTrue(summary['qualification_passed']); self.assertTrue(summary['final_telemetry_published'])
        self.assertEqual(len(calls), 17); self.assertFalse(any(summary['no_authority'].values()))
        path = next(folder.glob('tasks/*/request.json')); path.write_bytes(path.read_bytes() + b' ')
        self.seal(folder)
        with self.assertRaises(c.Stop): evidence.analyze(folder, FAKE_SHA)

    def test_slow_cpu_stops_after_probe_with_zero_tasks(self):
        folder, calls = self.simulate(slow=True)
        summary = evidence.analyze(folder, FAKE_SHA)
        self.assertFalse(summary['qualification_passed']); self.assertEqual(len(calls), 1)
        self.assertEqual(summary['scores']['tasks_complete'], 0)

    def test_token_exhaustion_preserved_without_request_retry(self):
        folder, calls = self.simulate(truncated=True)
        summary = evidence.analyze(folder, FAKE_SHA)
        self.assertFalse(summary['qualification_passed']); self.assertEqual(len(calls), 17)


if __name__ == '__main__': unittest.main()
