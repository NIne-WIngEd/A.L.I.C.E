from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent
NAME = 'ALICE_TELEMETRY_TRANSPORT_RECEIPT_v1.0.0'
PKG = BASE / 'package' / NAME
DIST = BASE / 'dist'; DIST.mkdir(exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()

(PKG / 'README.md').write_text('''# ALICE telemetry transport receipt v1.0.0

Read-only follow-up to actual a2 job 575155 and uploaded helper SHA256
ba4482728c09313aede5031e682f68e8007c79c9cfa084d3da4b11aa984d0830.

The helper suppresses fetch/push errors and exits 74 after exhausting retries.
This receipt uses its installed GIT_SSH wrapper and existing ledger clone:

- git --version
- git fetch --dry-run origin main
- git push --dry-run origin HEAD:refs/heads/main

Git commands have 10/20/20-second limits; the outer SSH call has an 85-second
limit. Git stdout/stderr and exit status are retained as bytes. The launcher
returns a trace ZIP even when Git, SSH, parsing or helper verification fails.
It reads wrapper source and selected Git metadata before/after. It reads no
private key contents. It does not execute the telemetry helper, stage packages
on Magnolia, submit jobs, change a2 state, or publish repository content.
Returned raw diagnostics require review before any context publication.

These are current login-node transport observations. Successful dry runs cannot
recover the lost historical stderr or prove a real fetch/commit/push transaction
will succeed. A non-fast-forward dry-run is a distinct observation from an SSH
authentication failure. No scientific execution authority follows any result.

Git 1.8.3.1 documentation supports both dry-run flags:
https://github.com/git/git/blob/v1.8.3.1/Documentation/fetch-options.txt
https://github.com/git/git/blob/v1.8.3.1/Documentation/git-push.txt

Keep the exact ZIP beside Start-ALICETelemetryTransportReceiptV100.ps1.
Run the launcher in the established Windows/Anaconda workflow. Return TRACE_ZIP
and the transcript. Qwen remains unevaluated; do not rerun v104.
''')
entries = []
for p in sorted(PKG.iterdir()):
    if p.is_file() and p.name != 'PACKAGE_MANIFEST.json':
        b = p.read_bytes(); entries.append({'path': p.name, 'bytes': len(b), 'sha256': sha(b)})
(PKG / 'PACKAGE_MANIFEST.json').write_text(json.dumps({'files': entries}, indent=2) + '\n')
archive = DIST / (NAME + '.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.iterdir()):
        if p.is_file():
            info = zipfile.ZipInfo(NAME + '/' + p.name, date_time=(2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())

template = (BASE.parent / 'qwen-provider-v104/dist/Start-ALICEAstraQwenQualificationV104.ps1').read_text()
prefix = template[:template.index('    Write-Host "Verified observed scheduler repair.')]
prefix = prefix.replace('    [string]$VaultRoot = "C:\\ALICE_Vault",\n', '').replace('    [string]$RepoRoot = "C:\\A.L.I.C.E-main",\n', '')
prefix = prefix.replace('EDC325F70513CFE6E0E27289BA05132E2DEFBFDF306DCAFD0550740955293647', sha(archive.read_bytes()).upper())
prefix = prefix.replace('ALICE_MC10D_QWEN_PROVIDER_REVISION_v1.0.4', NAME).replace('v1.0.4 ZIP', 'telemetry receipt v1.0.0 ZIP')
prefix = prefix.replace('ALICE_QwenA2_RUN_', 'ALICE_TelemetryReceipt_').replace('$RunId', '$ReceiptToken')
suffix = '''    Write-Host "Collecting three read-only Git transport receipts with the installed SSH wrapper. Return the ZIP even if Git fails."
    Invoke-Python -Arguments @((Join-Path $PackageRoot "collect_transport.py"),"--output-root",$RunRoot)
    $Code = $script:LastPythonExitCode
    Write-Host "ALICE_TELEMETRY_TRANSPORT_RECEIPT_EXIT=$Code"
    Write-Host "Return TRACE_ZIP and this transcript. Keep existing a2 state; do not rerun v104."
    exit $Code
}
catch {
    Write-Host "ALICE_TELEMETRY_TRANSPORT_STOP: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "Return this output and any TRACE_ZIP already printed. No execution recovery is inferred."
    exit 74
}
finally {
    if ($script:TranscriptStarted) { Stop-Transcript | Out-Null }
}
'''
launcher = DIST / 'Start-ALICETelemetryTransportReceiptV100.ps1'
launcher.write_bytes((prefix + suffix).replace('\n', '\r\n').encode('utf-8'))
with tempfile.TemporaryDirectory(prefix='alice-transport-build-') as td:
    with zipfile.ZipFile(archive) as z: z.extractall(td)
    fresh = Path(td) / NAME
    verifier = re.search(r"\$VerifyCode = @'\n(.*?)\n'@", prefix, re.S).group(1)
    verify = subprocess.run([sys.executable, '-c', verifier, str(fresh)], capture_output=True)
    assert verify.returncode == 0, verify.stderr
    for p in fresh.glob('*.py'): compile(p.read_bytes(), str(p), 'exec')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(fresh), '-p', 'selftest*.py', '-v'], capture_output=True)
    (DIST / 'BUILD_SELFTEST.log').write_bytes(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stderr
receipt = {'schema': 'alice.telemetry.transport-receipt.build.v1', 'package': archive.name,
    'package_sha256': sha(archive.read_bytes()), 'package_bytes': archive.stat().st_size,
    'launcher': launcher.name, 'launcher_sha256': sha(launcher.read_bytes()),
    'package_manifest_sha256': sha((PKG / 'PACKAGE_MANIFEST.json').read_bytes()),
    'tests_passed': 4, 'fresh_zip_extraction_tested': True,
    'embedded_powershell_manifest_verifier_executed': True,
    'powershell_native_execution_observed': False, 'magnolia_execution_observed': False,
    'source_helper_sha256': 'ba4482728c09313aede5031e682f68e8007c79c9cfa084d3da4b11aa984d0830',
    'commands': [['git', '--version'], ['git', 'fetch', '--dry-run', 'origin', 'main'], ['git', 'push', '--dry-run', 'origin', 'HEAD:refs/heads/main']],
    'command_timeouts_seconds': [10, 20, 20], 'ssh_timeout_seconds': 85,
    'execution_authority_granted': False, 'new_scientific_execution_package': False}
(DIST / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))

