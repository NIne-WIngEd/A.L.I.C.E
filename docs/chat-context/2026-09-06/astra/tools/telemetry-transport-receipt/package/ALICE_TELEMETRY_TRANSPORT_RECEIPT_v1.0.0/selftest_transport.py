import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import collect_transport as c

class ReceiptTests(unittest.TestCase):
    def test_actual_timeout_retains_partial_output_and_closes_posix_descendant_pipes(self):
        child = "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']); " if os.name != 'nt' else ''
        code = "import subprocess,sys,time; " + child + "print('raw-before-timeout', flush=True); time.sleep(20)"
        receipt = c.capture([sys.executable, '-u', '-c', code], 1)
        self.assertEqual(receipt['status'], 'TIMEOUT')
        self.assertIn(b'raw-before-timeout', base64.b64decode(receipt['stdout']['base64']))

    def test_nonzero_timeout_and_os_error_retain_available_raw_bytes(self):
        def failed(*a, **kw): return subprocess.CompletedProcess(a[0], 128, b'out\x00\xff', b'err\r\n')
        r = c.capture(['git', 'fetch', '--dry-run', 'origin', 'main'], 20, command=failed)
        self.assertEqual(r['exit_code'], 128)
        self.assertEqual(base64.b64decode(r['stdout']['base64']), b'out\x00\xff')
        def timeout(*a, **kw): raise subprocess.TimeoutExpired(a[0], 20, output=b'partial', stderr=b'error\xff')
        r = c.capture(['git'], 20, command=timeout)
        self.assertEqual(r['status'], 'TIMEOUT')
        self.assertEqual(base64.b64decode(r['stderr']['base64']), b'error\xff')
        def missing(*a, **kw): raise FileNotFoundError('git missing')
        self.assertEqual(c.capture(['git'], 20, command=missing)['status'], 'OS_ERROR')

    def test_actual_shell_stdin_bootstrap_and_native_windows_key(self):
        script = c.remote_script()
        if os.name != 'nt':
            self.assertEqual(subprocess.run(['bash', '-n'], input=script, capture_output=True).returncode, 0)
        python = script.split(b"<<'ALICE_TELEMETRY_TRANSPORT_PY'\n", 1)[1].rsplit(b'\nALICE_TELEMETRY_TRANSPORT_PY\n', 1)[0]
        compile(python, '<actual-remote-stdin>', 'exec')
        key = r'C:\Users\rayns\.ssh\rayan_magnolia_ed25519'
        argv = c.ssh_argv(key)
        self.assertEqual(argv[argv.index('-i') + 1], key)
        self.assertEqual(argv[-1], 'bash -s')

    def test_real_local_git_dry_runs_leave_repository_and_remote_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); remote = root / 'remote.git'; clone = root / 'clone'
            def git(*args):
                p = subprocess.run(['git', *args], capture_output=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                return p.stdout
            git('init', '--bare', str(remote)); git('init', str(clone))
            git('-C', str(clone), 'config', 'user.name', 'fixture')
            git('-C', str(clone), 'config', 'user.email', 'fixture@example.invalid')
            (clone / 'README').write_text('fixture\n')
            git('-C', str(clone), 'add', 'README'); git('-C', str(clone), 'commit', '-m', 'fixture')
            git('-C', str(clone), 'branch', '-M', 'main')
            git('-C', str(clone), 'remote', 'add', 'origin', str(remote))
            git('-C', str(clone), 'push', 'origin', 'main')
            def inventory(): return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            before = inventory()
            git('-C', str(clone), 'fetch', '--dry-run', 'origin', 'main')
            git('-C', str(clone), 'push', '--dry-run', 'origin', 'HEAD:refs/heads/main')
            self.assertEqual(before, inventory())

if __name__ == '__main__': unittest.main()
