from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent
NAME = 'ALICE_TELEMETRY_PUBLISHER_REPAIR_v1.0.0'
PKG = BASE / 'package' / NAME
DIST = BASE / 'dist'; DIST.mkdir(exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
shell = '''#!/bin/bash
# ALICE Git 1.8.3 publisher repair v1.0.0; seven-argument interface preserved.
set -euo pipefail
umask 077
PYROOT="/modules/pkgs/common/python/3.11.5"
PY="$PYROOT/bin/python3.11"
PYLIB=""
for d in "$PYROOT/lib" "$PYROOT/lib64"; do
  if [ -e "$d/libpython3.11.so.1.0" ]; then PYLIB="$d"; break; fi
done
test -n "$PYLIB"
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PY" -B - "$@" <<'ALICE_TELEMETRY_PUBLISHER_PY'
'''
helper = shell.encode() + (PKG / 'publisher.py').read_bytes() + b'\nALICE_TELEMETRY_PUBLISHER_PY\n'
(PKG / 'rayan-telemetry-push.sh').write_bytes(helper)
snapshot = 'alice-qwen38-a2-c926d9e355dd-t-161d1ba71ed308ed'
run = 'runs/alice-qwen38-a2-c926d9e355dd'
payload = run + '/telemetry-events/' + snapshot
plan = {'schema': 'alice.telemetry.publisher-repair.plan.v1',
    'revision_id': 'git183-publisher-v100-' + sha(helper)[:12],
    'old_helper_sha256': 'ba4482728c09313aede5031e682f68e8007c79c9cfa084d3da4b11aa984d0830',
    'new_helper_sha256': sha(helper),
    'ssh_wrapper_sha256': '7ddc10dff3fa2ae387c1b2e023d770c6047e5d076ab4ef837247b22dc8870120',
    'origin': 'git@ssh.github.com:NIne-WIngEd/Rayan-Compute-Ledger.git',
    'snapshot_id': snapshot, 'payload_dir': payload,
    'stage': 'MC10D_QWEN_PUBLIC_CALIBRATION_V1', 'job_id': '575155',
    'a2_guard_sha256': {
        run + '/worker-finished.json': 'cef2de900be2c3ad129dbe7e11df5a79e33579014450e73f22f38a806110216a',
        run + '/result.json': '829e1e0bd81ac49bbe581c6597fd78b18252963507b7d9563ac99accb9b85097',
        run + '/run.json': '8822368909a17fc243f1e8c8cafb721c8df930401035cd83457bd779f1f2afc6',
        run + '/submission.json': 'a774fb22223b89f5078cc2bd156460f92113650304eed576b48713d028e9a5bf',
        payload + '/application.log': 'dc2358cdb72f872a3036b184e9143432d0d9216ca1bf91937bf903a18d85a3b7',
        payload + '/manifest.json': 'b0d545d80912d829203f532114ab7e813c8ba1c8cad0588a343554b2cc2376dc',
        payload + '/receipt.json': '96d16fbc806dec33add04d70b22ce9b4fe8e076f20648a8a1e1fb4fd57121fe1'},
    'remote_operation': 'Install exact shared publisher revision after actual-host tests; verify closed a2 terminal publication twice with unchanged payload',
    'scientific_execution_authorized': False, 'a2_resubmission': False}
(PKG / 'repair-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
(PKG / 'README.md').write_text('''# ALICE telemetry publisher repair v1.0.0

The actual Git 1.8.3.1 trace shows cached main/origin-main at ec4d378 and
FETCH_HEAD at 3f42de3. Git fetch origin main did not refresh origin/main before
Git 1.8.4. The old helper then based its commit on stale origin/main. It also
regenerated immutable metadata on retries and suppressed the Git error streams.

The replacement keeps the seven-argument helper interface. It uses an explicit
destination refspec and verifies FETCH_HEAD against the tracking ref. It builds
each append from that fetched parent with a private Git index. No checkout,
local branch reset, force push, or inference is performed. A competing append
is preserved by rebuilding on the next fetched parent. Every Git command gets
raw stdout/stderr, argv, exit status and timestamps in a private remote journal.
The complete spool, including the original metadata timestamp, stays frozen.
An existing three-file connector snapshot can receive its missing wrapper
metadata only if all already-published bytes are identical.

The owner launcher verifies this ZIP and runs local gates. On Magnolia it checks
the old/new helper hash, exact SSH wrapper, expected ledger URL and closed a2
evidence. Seven publisher tests run against the actual host Git using temporary
local repositories BEFORE installation. Changed fixtures stop installation.
The exact original helper is preserved under telemetry/helper-revisions. The
new helper is atomically installed under the existing publisher lock. One closed
a2 terminal snapshot is then published and retried after one second. Both calls
must verify the remote payload and the retry must retain identical spool bytes.

Local Windows and remote a2 controller/result bytes are preserved. The existing
shared checkout HEAD/index/local-main bytes are also checked before/after live
publication. Its local HEAD can remain behind the remote intentionally: future
verification must use the publisher receipt, not a raw push of that cached HEAD.
The helper has a 70-second internal budget, within the existing caller's 90 seconds.

The seven actual-host tests use no model, GPU, scheduler or real GitHub writes.
The subsequent live publisher calls write only the already-reviewed terminal
telemetry to the existing ledger. No new Qwen execution identity, calibration,
private evaluation, training, Stage G acceptance or Phase 2 replacement is granted.
Return RESULT_ZIP even on failure; preserve all helper and a2 state for review.

Primary compatibility source:
https://github.com/git/git/blob/v1.8.4/Documentation/RelNotes/1.8.4.txt
Offline modern-Git fixtures emulate the older fetch mapping only for the legacy
source-only fetch. On actual Git 1.8.3.1 that command runs unmodified.
''')
entries = []
for f in sorted(PKG.iterdir()):
    if f.is_file() and f.name != 'PACKAGE_MANIFEST.json':
        data = f.read_bytes(); entries.append({'path': f.name, 'bytes': len(data), 'sha256': sha(data)})
(PKG / 'PACKAGE_MANIFEST.json').write_text(json.dumps({'files': entries}, indent=2) + '\n')
archive = DIST / (NAME + '.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(PKG.iterdir()):
        if f.is_file():
            info = zipfile.ZipInfo(NAME + '/' + f.name, date_time=(2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            z.writestr(info, f.read_bytes())
template_path = BASE / 'launcher-template.ps1'
assert sha(template_path.read_bytes()) == '9ff8b0ba823b66190ee30bdb316aa3e500c8591dd7ad5faf28d01deeb3f4c5bb'
template = template_path.read_text()
prefix = template[:template.index('    Write-Host "Collecting three read-only Git transport receipts')]
prefix = re.sub(r'\$Expected = "[A-F0-9]{64}"', '$Expected = "' + sha(archive.read_bytes()).upper() + '"', prefix)
prefix = prefix.replace('ALICE_TELEMETRY_TRANSPORT_RECEIPT_v1.0.0', NAME).replace('telemetry receipt v1.0.0 ZIP', 'publisher repair v1.0.0 ZIP').replace('ALICE_TelemetryReceipt_', 'ALICE_PublisherRepair_')
suffix = '''    Write-Host "The publisher repair is checked on Magnolia before installation. Then it verifies the existing failed a2 terminal snapshot and an identical retry. No inference or new job."
    Invoke-Python -Arguments @((Join-Path $PackageRoot "repair_publisher.py"),"--output-root",$RunRoot)
    $Code = $script:LastPythonExitCode
    Write-Host "ALICE_TELEMETRY_PUBLISHER_REPAIR_EXIT=$Code"
    Write-Host "Return RESULT_ZIP and this transcript. Keep a2 closed; no reset or resubmission."
    exit $Code
}
catch {
    Write-Host "ALICE_TELEMETRY_PUBLISHER_REPAIR_STOP: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "Return this output and any RESULT_ZIP already printed. Preserve existing helper and execution state."
    exit 74
}
finally {
    if ($script:TranscriptStarted) { Stop-Transcript | Out-Null }
}
'''
launcher = DIST / 'Start-ALICETelemetryPublisherRepairV100.ps1'
launcher.write_bytes((prefix + suffix).replace('\n', '\r\n').encode())
with tempfile.TemporaryDirectory(prefix='alice-publisher-build-') as td:
    with zipfile.ZipFile(archive) as z: z.extractall(td)
    fresh = Path(td) / NAME
    verify = re.search(r"\$VerifyCode = @'\n(.*?)\n'@", prefix, re.S).group(1)
    check = subprocess.run([sys.executable, '-c', verify, str(fresh)], capture_output=True)
    assert check.returncode == 0, check.stderr
    for f in fresh.glob('*.py'): compile(f.read_bytes(), str(f), 'exec')
    syntax = subprocess.run(['bash', '-n', str(fresh / 'rayan-telemetry-push.sh')], capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(fresh), '-p', 'selftest*.py', '-v'], capture_output=True)
    (DIST / 'BUILD_SELFTEST.log').write_bytes(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stderr
receipt = {'schema': 'alice.telemetry.publisher-repair.build.v1', 'package': archive.name,
    'package_sha256': sha(archive.read_bytes()), 'package_bytes': archive.stat().st_size,
    'launcher': launcher.name, 'launcher_sha256': sha(launcher.read_bytes()),
    'new_helper_sha256': sha(helper), 'revision_id': plan['revision_id'],
    'package_manifest_sha256': sha((PKG / 'PACKAGE_MANIFEST.json').read_bytes()),
    'local_tests_passed': 9, 'actual_host_publisher_tests_required_before_install': 7,
    'fresh_zip_tested': True, 'native_windows_execution_observed': False,
    'actual_magnolia_installation_or_publication_observed': False,
    'exact_original_helper_backup_required': True, 'new_scientific_execution': False,
    'modern_git_fixture_scope': 'Real local repositories; --refmap= models old source-only fetch behavior. Actual Magnolia gate uses unmodified Git 1.8.3.1.'}
(DIST / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
