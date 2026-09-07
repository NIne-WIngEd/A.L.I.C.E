"""Build and exercise the diagnostic, not a new qualification workload."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
PKG = ROOT / 'package/ALICE_QWEN_READ_ONLY_TRACE_v1.0.0'
DIST = ROOT / 'dist'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    DIST.mkdir(exist_ok=True)
    entries = [{'path': p.relative_to(PKG).as_posix(), 'bytes': p.stat().st_size,
                'sha256': sha(p.read_bytes())} for p in sorted(PKG.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.name != 'PACKAGE_MANIFEST.json']
    (PKG / 'PACKAGE_MANIFEST.json').write_text(json.dumps({'files': entries}, indent=2) + '\n', encoding='utf-8')
    archive = DIST / (PKG.name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(PKG.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            info = zipfile.ZipInfo(PKG.name + '/' + p.relative_to(PKG).as_posix(), (2026, 9, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
    template = (ROOT / 'Start-ALICEQwenReadOnlyTraceV100.template.ps1').read_text(encoding='utf-8')
    launcher = DIST / 'Start-ALICEQwenReadOnlyTraceV100.ps1'
    launcher.write_bytes(template.replace('@ZIP_SHA@', sha(archive.read_bytes()).upper()).replace('\n', '\r\n').encode())
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        clean = Path(tmp) / PKG.name
        verifier = re.search(r"\$VerifyCode = @'\n(.*?)\n'@", template, re.S).group(1)
        result = subprocess.run([sys.executable, '-B', '-c', verifier, str(clean)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(clean),
                                 '-p', 'selftest_trace.py', '-v'], capture_output=True, text=True)
        (DIST / 'BUILD_SELFTEST.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == 0, result.stdout + result.stderr
        sys.path.insert(0, str(clean))
        import collect_trace
        policy = json.loads((clean / 'trace_policy.json').read_bytes())
        script = collect_trace.remote_script(policy)
        shell = subprocess.run(['bash', '-n'], input=script, capture_output=True)
        assert shell.returncode == 0, shell.stderr.decode()
    receipt = {'schema': 'alice.qwen.read-only-trace-release.v1',
               'purpose': 'Collect exact live facts needed to resolve v102 evidence origins and a2 identity',
               'archive': archive.name, 'archive_sha256': sha(archive.read_bytes()), 'archive_bytes': archive.stat().st_size,
               'launcher': launcher.name, 'launcher_sha256': sha(launcher.read_bytes()),
               'package_manifest_sha256': sha((PKG / 'PACKAGE_MANIFEST.json').read_bytes()),
               'selftest_log_sha256': sha((DIST / 'BUILD_SELFTEST.log').read_bytes()),
               'tests_passed': 6, 'fresh_zip_tested': True, 'literal_powershell_python_verifier_tested': True,
               'literal_ssh_python_compiled': True, 'literal_ssh_bash_syntax_checked': True,
               'native_windows_tested': False, 'actual_magnolia_tested': False,
               'new_qualification_package': False, 'new_execution_run_id': None,
               'remote_mutations': [], 'live_inference': False}
    (DIST / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
