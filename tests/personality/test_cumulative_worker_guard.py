"""Actual worker entry ordering; public checks grant no private qualification."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKER = Path(__file__).resolve().parents[2] / 'scripts/eipm/gemma_n0/magnolia_cumulative_learning_20261013.py'


class CumulativeWorkerGuardTests(unittest.TestCase):
    def test_guard_is_executed_before_any_private_operation(self):
        self.assertTrue(WORKER.is_file(), 'The owner cumulative learning worker is absent')
        tree = ast.parse(WORKER.read_bytes())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        calls = []

        class GuardRefusal(Exception):
            pass

        class Namespace:
            def _p2_guard(self):
                calls.append('actual_guard')
                raise GuardRefusal()

            def _bounded_path(self, *args, **kwargs):
                self.fail('Private path access preceded isolation')

        program = compile(ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[])), str(WORKER), 'exec')
        scope = {}
        exec(program, scope)
        with self.assertRaises(GuardRefusal):
            scope['run'](Namespace())
        self.assertEqual(calls, ['actual_guard'])

    def test_actual_dispatch_refuses_changed_worker_and_each_public_helper(self):
        wrapper = WORKER.with_suffix('.sbatch')
        checks = [line for line in wrapper.read_text().splitlines() if line.startswith('[[') and 'sha256sum' in line]
        self.assertEqual(len(checks), 7)
        program = 'set -euo pipefail\n' + '\n'.join(checks) + '\nprintf PUBLIC_PINS_ACCEPTED'
        script_directory = WORKER.parent
        helpers = ('governed_response.py', 'cumulative_successor.py', 'cumulative_bindings.py', 'connected_voice_references.py', 'n0_semantic_contrasts.py', 'conditional_successor_report.py')
        with tempfile.TemporaryDirectory(prefix='cumulative-public-dispatch-') as temporary:
            root = Path(temporary)
            jobs = root/'jobs'
            jobs.mkdir()
            worker = jobs/'rayan-cumulative-learning-r2-20261013.py'
            worker.write_bytes(WORKER.read_bytes())
            paths = [worker]
            for name in helpers:
                path = jobs/('rayan-20261013-r2-'+name)
                path.write_bytes((script_directory/name).read_bytes())
                paths.append(path)
            environment = dict(os.environ, AUDIT_HELPER=worker.as_posix(), COMPUTE_ROOT=root.as_posix())
            bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
            self.assertTrue(bash, 'The public Magnolia-dispatch fixture requires Bash')

            def dispatch():
                return subprocess.run([bash, '-c', program], env=environment, capture_output=True)

            result = dispatch()
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            self.assertEqual(result.stdout, b'PUBLIC_PINS_ACCEPTED')
            for path in paths:
                with self.subTest(source=path.name):
                    original = path.read_bytes()
                    path.write_bytes(original+b'\n# changed public source\n')
                    result = dispatch()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, b'')
                    path.write_bytes(original)
