"""Replay captured Magnolia responses; synthetic mutations are labelled below."""
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import contract as c
import fixture_origins as fixtures
import infra_recovery
import magnolia_scheduler as provider
import package_revision as revision
import remote_agent


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.patch = mock.patch.object(c, 'ROOT', self.root); self.patch.start()
        self.run = self.root / 'runs' / c.RUN_ID; fixtures.prior_a2(self.run)
        with zipfile.ZipFile(c.BASE / 'source/ALICE_MAGNOLIA_PROVIDER_TRACE_15f4edccafa3.zip') as z:
            report = json.loads(z.read('report.json'))
        self.observed = {tuple(x['argv']): x for x in report['remote']['commands']}
        self.calls = []; provider.LAST = None

    def tearDown(self):
        self.patch.stop(); self.temp.cleanup()

    def recorded_process(self, argv, **kwargs):
        self.calls.append(list(argv))
        row = self.observed[tuple(argv)]  # Unexpected shapes cannot receive invented success.
        return subprocess.CompletedProcess(argv, row['exit_code'],
            base64.b64decode(row['stdout']['base64']), base64.b64decode(row['stderr']['base64']))

    def test_actual_failed_job_query_and_actual_successful_source_and_current_queries(self):
        fixtures.install_sources(self.root)
        before = (self.run / 'run.json').read_bytes()
        with mock.patch.object(provider.subprocess, 'run', side_effect=self.recorded_process):
            bad = provider.command(['squeue', '-h', '-j', '575089', '-o', '%A|%j|%T'], check=False)
            self.assertEqual(bad.returncode, 1)
            self.assertEqual(bad.stderr, 'slurm_load_jobs error: Invalid job id specified\n')
            result = infra_recovery.reconcile_source(provider.command)
            self.assertTrue(result['zero_inference_confirmed'])
            self.assertIsNone(remote_agent.find_submitted())
            self.assertEqual(provider.job_state(provider.command, '575089', c.CALIBRATION_ID), ('FAILED', False))
        self.assertTrue(all('-j' not in x for x in self.calls[1:] if x[0] == 'squeue'))
        self.assertEqual((self.run / 'run.json').read_bytes(), before)
        self.assertFalse((self.run / 'submission-intent.json').exists())

    def test_synthetic_failed_or_malformed_user_queue_never_means_absence(self):
        for exit_code, stdout in ((1, b''), (0, b'bad-row\n')):
            with self.subTest(exit_code=exit_code, stdout=stdout), \
                 mock.patch.object(provider.subprocess, 'run', return_value=subprocess.CompletedProcess([], exit_code, stdout, b'SYNTHETIC_FAILURE')):
                with self.assertRaisesRegex(c.Stop, 'absence is unknown'):
                    provider.find_submitted(provider.command)
                self.assertEqual(provider.LAST['exit_code'], exit_code)
        self.assertFalse((self.run / 'submission-intent.json').exists())

    def test_synthetic_active_duplicate_and_wrong_historical_identity_are_refused(self):
        fixtures.install_sources(self.root)
        queue = tuple(provider.USER_QUEUE); account = tuple(provider.accounting_argv('575089'))
        for kind in ('active_source', 'wrong_source_name', 'duplicate_history', 'failed_accounting'):
            with self.subTest(kind=kind):
                def altered(argv, **kwargs):
                    if kind == 'active_source' and tuple(argv) == queue:
                        return subprocess.CompletedProcess(argv, 0, ('575089|' + c.CALIBRATION_ID + '|RUNNING\n').encode(), b'')
                    if tuple(argv) == account:
                        if kind == 'failed_accounting': return subprocess.CompletedProcess(argv, 1, b'', b'SYNTHETIC_ACCOUNT_FAILURE')
                        text = '575089|' + ('wrong' if kind == 'wrong_source_name' else c.CALIBRATION_ID) + '|FAILED|76:0\n'
                        return subprocess.CompletedProcess(argv, 0, (text * (2 if kind == 'duplicate_history' else 1)).encode(), b'')
                    return self.recorded_process(argv, **kwargs)
                with mock.patch.object(provider.subprocess, 'run', side_effect=altered), self.assertRaises(c.Stop):
                    infra_recovery.reconcile_source(provider.command)

    def test_synthetic_timeout_and_invalid_encoding_preserve_raw_journal_before_stop(self):
        cases = [subprocess.TimeoutExpired(['sbatch'], 90, output=b'partial\xff', stderr=b'detail'),
                 subprocess.CompletedProcess([], 0, b'bad\xff', b'')]
        for case in cases:
            with self.subTest(case=type(case).__name__):
                kwargs = {'side_effect': case} if isinstance(case, Exception) else {'return_value': case}
                with mock.patch.object(provider.subprocess, 'run', **kwargs), self.assertRaises(c.Stop):
                    provider.command(['sbatch', '--parsable', 'SYNTHETIC_TEST_ONLY'], check=False)
                journal = [json.loads(x) for x in (self.run / 'scheduler-commands.jsonl').read_text().splitlines()]
                raw = case.output if isinstance(case, Exception) else case.stdout
                self.assertEqual(base64.b64decode(journal[-1]['stdout']['base64']), raw)
        self.assertFalse((self.run / 'submission.json').exists())

    def test_actual_local_intent_is_preserved_through_new_revision_and_acknowledgement(self):
        fixtures.install_sources(self.root)
        state_path, rules = fixtures.local_prepared(self.root / 'vault')
        previous = state_path.parent / 'revisions' / rules['prior_uncommitted_revision_id']
        old = {p.name: p.read_bytes() for p in previous.iterdir()}
        new_sha = '2' * 64
        revision.begin_local(state_path, new_sha)
        with mock.patch.object(provider.subprocess, 'run', side_effect=self.recorded_process):
            acknowledgement = revision.apply_remote(self.run, new_sha, provider.command, remote_agent.find_submitted)
        revision.finish_local(state_path, new_sha, acknowledgement)
        self.assertEqual(revision.verify_local(state_path, new_sha)['run_id'], c.RUN_ID)
        self.assertEqual({p.name: p.read_bytes() for p in previous.iterdir()}, old)
        self.assertEqual(acknowledgement['prior_uncommitted_intent_sha256'], rules['prior_uncommitted_intent_sha256'])
        self.assertEqual(acknowledgement['new_execution_created'], False)
        self.assertFalse((self.run / 'submission-intent.json').exists())

    def test_changed_or_acknowledged_prior_intent_and_any_other_remote_revision_stop(self):
        state_path, rules = fixtures.local_prepared(self.root / 'vault')
        prior = state_path.parent / 'revisions' / rules['prior_uncommitted_revision_id']
        intent = prior / 'intent.json'; saved = intent.read_bytes()
        for name in ('changed_intent', 'acknowledged'):
            with self.subTest(name=name):
                if name == 'changed_intent': intent.write_bytes(saved + b' ')
                else: (prior / 'remote-receipt.json').write_bytes(b'{}')
                with self.assertRaises(c.Stop): revision.begin_local(state_path, '2' * 64)
                self.assertFalse(revision.history(state_path.parent).exists())
                intent.write_bytes(saved)
                (prior / 'remote-receipt.json').unlink(missing_ok=True)
        (self.run / 'revisions' / rules['prior_uncommitted_revision_id']).mkdir(parents=True)
        command = mock.Mock()
        with self.assertRaisesRegex(c.Stop, 'Another remote revision'):
            revision.apply_remote(self.run, '2' * 64, command, mock.Mock())
        command.assert_not_called()


if __name__ == '__main__': unittest.main(verbosity=2)
