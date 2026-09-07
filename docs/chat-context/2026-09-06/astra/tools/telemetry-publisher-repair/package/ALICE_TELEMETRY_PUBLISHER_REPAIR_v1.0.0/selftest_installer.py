import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import repair_publisher as r

class InstallerTests(unittest.TestCase):
    def test_guarded_install_preserves_original_retry_and_unexpected_live_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); helper = root / 'helper.sh'; revision = root / 'revision'
            old, new = b'old helper\n', b'new helper\n'
            helper.write_bytes(old)
            plan = {'old_helper_sha256': r.sha(old), 'new_helper_sha256': r.sha(new)}
            self.assertTrue(r.guarded_install(helper, revision, plan, new))
            self.assertFalse(r.guarded_install(helper, revision, plan, new))
            self.assertEqual((revision / 'original-rayan-telemetry-push.sh').read_bytes(), old)
            helper.write_bytes(b'another revision\n')
            with self.assertRaises(RuntimeError): r.guarded_install(helper, revision, plan, new)
            self.assertEqual(helper.read_bytes(), b'another revision\n')

    def test_real_stdin_program_compiles_and_unknown_helper_stops_before_staging(self):
        plan = json.loads((r.BASE / 'repair-plan.json').read_bytes())
        sources = {n: base64.b64encode((r.BASE / n).read_bytes()).decode() for n in ('publisher.py', 'selftest_publisher.py', 'rayan-telemetry-push.sh')}
        script = r.remote_script(plan, sources)
        if os.name != 'nt': self.assertEqual(subprocess.run(['bash', '-n'], input=script, capture_output=True).returncode, 0)
        code = script.split(b"<<'ALICE_PUBLISHER_REPAIR_PY'\n", 1)[1].rsplit(b'\nALICE_PUBLISHER_REPAIR_PY\n', 1)[0]
        compile(code, '<actual-installer-stdin>', 'exec')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'bin').mkdir(); (root / 'bin/rayan-telemetry-push.sh').write_bytes(b'changed')
            output = io.StringIO()
            with patch.object(r, 'ROOT', root), contextlib.redirect_stdout(output): r.run_remote(plan, sources)
            report = json.loads(output.getvalue())
            self.assertFalse(report['helper_installed'])
            self.assertFalse((root / 'telemetry').exists())
            self.assertEqual((root / 'bin/rayan-telemetry-push.sh').read_bytes(), b'changed')

if __name__ == '__main__': unittest.main()
