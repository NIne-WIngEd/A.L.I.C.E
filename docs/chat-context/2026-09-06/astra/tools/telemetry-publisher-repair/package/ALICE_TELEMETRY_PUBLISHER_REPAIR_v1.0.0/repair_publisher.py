"""Guarded publisher installation and verification of the closed a2 snapshot."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import uuid
import zipfile
import collect_transport as transport

BASE = Path(__file__).resolve().parent
ROOT = Path('/homes/01/mxrayan/rayan-compute')

def sha(data): return hashlib.sha256(data).hexdigest()
def check(ok, message):
    if not ok: raise RuntimeError(message)
def json_bytes(obj): return (json.dumps(obj, indent=2) + '\n').encode()
def immutable(path, data, mode=0o600):
    if path.exists(): check(path.is_file() and not path.is_symlink() and path.read_bytes() == data, 'Existing revision file differs: ' + str(path)); return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    path.chmod(mode)

def guarded_install(helper, revision, plan, new_bytes):
    check(not helper.is_symlink() and helper.is_file(), 'Helper must be a regular file')
    current = helper.read_bytes()
    check(sha(current) in (plan['old_helper_sha256'], plan['new_helper_sha256']), 'Installed helper changed; no replacement')
    check(sha(new_bytes) == plan['new_helper_sha256'], 'New helper bytes differ')
    backup = revision / 'original-rayan-telemetry-push.sh'
    if sha(current) == plan['old_helper_sha256']: immutable(backup, current, 0o700)
    check(backup.is_file() and not backup.is_symlink() and sha(backup.read_bytes()) == plan['old_helper_sha256'], 'Verified original backup missing')
    if sha(current) == plan['new_helper_sha256']: return False
    temporary = helper.with_name(helper.name + '.repair-' + uuid.uuid4().hex)
    with temporary.open('xb') as f: f.write(new_bytes); f.flush(); os.fsync(f.fileno())
    temporary.chmod(0o700)
    os.replace(temporary, helper)
    check(sha(helper.read_bytes()) == plan['new_helper_sha256'], 'Installed helper verification failed')
    return True

def run_remote(plan, sources):
    os.umask(0o077)
    report = {'schema': 'alice.telemetry.publisher-repair.remote.v1', 'started_at': transport.now(),
        'status': 'STOPPED', 'helper_installed': False, 'model_jobs_submitted': 0,
        'a2_controller_state_modified': False, 'commands': [], 'publisher_journals': {}}
    lock = ROOT / 'telemetry/push.lock'; owned = False
    helper = ROOT / 'bin/rayan-telemetry-push.sh'
    revision = ROOT / 'telemetry/helper-revisions' / plan['revision_id']
    try:
        before = helper.read_bytes()
        check(not helper.is_symlink() and sha(before) in (plan['old_helper_sha256'], plan['new_helper_sha256']), 'Helper revision drift')
        wrapper = ROOT / 'bin/rayan-github-ssh'
        check(not wrapper.is_symlink() and sha(wrapper.read_bytes()) == plan['ssh_wrapper_sha256'], 'Installed SSH wrapper differs')
        report['a2_before'] = {}
        for rel, expected in plan['a2_guard_sha256'].items():
            path = ROOT / rel
            check(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == expected, 'a2 evidence differs: ' + rel)
            report['a2_before'][rel] = expected
        version = transport.capture(['git', '--version'], 10, cwd=str(ROOT / 'telemetry/ledger'))
        report['commands'].append(version)
        check(version['exit_code'] == 0, 'Could not observe the installed Git version')
        report['actual_git_version'] = base64.b64decode(version['stdout']['base64']).decode().strip()
        origin = transport.capture(['git', 'config', '--get', 'remote.origin.url'], 10, cwd=str(ROOT / 'telemetry/ledger'))
        report['commands'].append(origin)
        check(origin['exit_code'] == 0 and base64.b64decode(origin['stdout']['base64']).decode().strip() == plan['origin'], 'Ledger origin differs')
        pushurl = transport.capture(['git', 'config', '--get-all', 'remote.origin.pushurl'], 10, cwd=str(ROOT / 'telemetry/ledger'))
        report['commands'].append(pushurl)
        check(pushurl['exit_code'] in (0, 1), 'Could not read push URL configuration')
        check(not base64.b64decode(pushurl['stdout']['base64']).strip() or base64.b64decode(pushurl['stdout']['base64']).decode().splitlines() == [plan['origin']], 'Ledger push URL differs')
        immutable(revision / 'repair-plan.json', json_bytes(plan))
        for name, content in sources.items(): immutable(revision / name, base64.b64decode(content))
        # These tests use disposable local bare Git repositories. They run on
        # Magnolia's actual Git before any installed helper is replaced.
        tests = transport.capture([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(revision), '-p', 'selftest_publisher.py', '-v'], 60)
        report['commands'].append(tests)
        check(tests['exit_code'] == 0, 'Actual Magnolia publisher selftests failed; helper unchanged')
        candidate = revision / 'rayan-telemetry-push.sh'
        syntax = transport.capture(['bash', '-n', str(candidate)], 10)
        report['commands'].append(syntax)
        check(syntax['exit_code'] == 0, 'Candidate shell syntax failed')
        cli = transport.capture(['bash', str(candidate)], 15)
        report['commands'].append(cli)
        check(cli['exit_code'] == 64, 'Actual shell/Python CLI boundary failed')
        for _ in range(20):
            try: lock.mkdir(mode=0o700); owned = True; break
            except FileExistsError: time.sleep(0.25)
        check(owned, 'Publisher lock busy; no lock removal or installation attempted')
        report['installed_in_this_invocation'] = guarded_install(helper, revision, plan, base64.b64decode(sources['rayan-telemetry-push.sh']))
        report.update(helper_installed=True, installed_helper_sha256=sha(helper.read_bytes()), original_backup=str(revision / 'original-rayan-telemetry-push.sh'))
        lock.rmdir(); owned = False
        control_paths = ['HEAD', 'index', 'refs/heads/main']
        def ledger_controls():
            return {name: sha((ROOT / 'telemetry/ledger/.git' / name).read_bytes()) if (ROOT / 'telemetry/ledger/.git' / name).is_file() else None for name in control_paths}
        report['ledger_controls_before'] = ledger_controls()
        argv = [str(helper), plan['snapshot_id'], plan['stage'], 'FAILED', plan['job_id']] + [str(ROOT / plan['payload_dir'] / n) for n in ('application.log', 'manifest.json', 'receipt.json')]
        metadata_hash = None
        for number in (1, 2):
            if number == 2: time.sleep(1.1)
            invocation = transport.capture(argv, 85)
            report['commands'].append(invocation)
            stdout = base64.b64decode(invocation['stdout']['base64']).decode('utf-8', 'replace')
            journals = [line.split('=', 1)[1] for line in stdout.splitlines() if line.startswith('TELEMETRY_RECEIPT_DIR=')]
            for locator in journals:
                folder = Path(locator)
                expected_root = ROOT / 'telemetry/publisher-receipts' / plan['snapshot_id']
                check(folder.resolve().parent == expected_root.resolve() and not folder.is_symlink(), 'Unexpected helper journal locator')
                for path in sorted(folder.iterdir()):
                    if path.is_file() and path.name != 'private-index':
                        report['publisher_journals'][folder.name + '/' + path.name] = transport.stream(path.read_bytes())
            check(invocation['exit_code'] == 0 and len(journals) == 1, 'Terminal publication failed; raw helper journal retained')
            result = json.loads((Path(journals[0]) / 'result.json').read_bytes())
            check(result['published'] and result['exit_code'] == 0, 'Publisher verification missing')
            metadata = result['spool_sha256']
            if number == 1: metadata_hash = metadata
            else: check(metadata == metadata_hash and result['idempotent'], 'Second call changed the snapshot or was not idempotent')
            report['terminal_verified_commit'] = result['verified_commit']
        report['a2_after'] = {rel: sha((ROOT / rel).read_bytes()) for rel in plan['a2_guard_sha256']}
        check(report['a2_before'] == report['a2_after'], 'a2 source evidence changed concurrently; review required')
        report['ledger_controls_after'] = ledger_controls()
        check(report['ledger_controls_before'] == report['ledger_controls_after'], 'Shared checkout control files changed concurrently; review required')
        report.update(status='INSTALLED_AND_TERMINAL_RETRY_VERIFIED', terminal_telemetry_published=True, immutable_retry_verified=True)
    except Exception as exc: report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if owned:
            try: lock.rmdir()
            except OSError as exc: report['lock_release_error'] = str(exc)
        report['finished_at'] = transport.now()
        if revision.exists():
            observation = revision / 'observations' / (uuid.uuid4().hex + '.json')
            immutable(observation, json_bytes(report))
        print(json.dumps(report), flush=True)

def remote_script(plan, sources):
    transport_source = (BASE / 'collect_transport.py').read_text().rsplit("\nif __name__ == '__main__':", 1)[0]
    source = Path(__file__).read_text().rsplit("\nif __name__ == '__main__':", 1)[0]
    code = 'import sys,types\nm=types.ModuleType("collect_transport")\nm.__file__="<embedded-transport>"\n'
    code += 'exec(' + repr(transport_source) + ',m.__dict__)\nsys.modules["collect_transport"]=m\n'
    code += source + '\nrun_remote(' + repr(plan) + ',' + repr(sources) + ')\n'
    shell = '''#!/bin/bash
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
exec "$PY" -B - <<'ALICE_PUBLISHER_REPAIR_PY'
'''
    return (shell + code + '\nALICE_PUBLISHER_REPAIR_PY\n').encode()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path.home() / 'Downloads')
    args = parser.parse_args()
    out = args.output_root / ('ALICE_PublisherRepair_' + uuid.uuid4().hex[:12]); out.mkdir(parents=True)
    plan = json.loads((BASE / 'repair-plan.json').read_bytes())
    names = ('publisher.py', 'selftest_publisher.py', 'rayan-telemetry-push.sh')
    sources = {n: base64.b64encode((BASE / n).read_bytes()).decode() for n in names}
    script = remote_script(plan, sources)
    (out / 'remote-input.sh').write_bytes(script)
    state = Path('C:/ALICE_Vault/tools/alice-astra/qwen-fallback-a2/controller-state.json')
    state_before = sha(state.read_bytes()) if state.is_file() else None
    print('action=GUARDED_PUBLISHER_REPAIR_AND_CLOSED_A2_TERMINAL_VERIFY; no model job or inference', flush=True)
    print('checking=actual Magnolia Git fixtures before install; then terminal publication and identical retry', flush=True)
    receipt = transport.capture(transport.ssh_argv(Path.home() / '.ssh/rayan_magnolia_ed25519'), 240, input=script)
    for key in ('stdout', 'stderr'): (out / ('ssh.' + key + '.bin')).write_bytes(base64.b64decode(receipt[key]['base64']))
    (out / 'transport.json').write_bytes(json_bytes(receipt))
    report = None; parse_error = None
    try: report = json.loads(base64.b64decode(receipt['stdout']['base64']))
    except ValueError as exc: parse_error = str(exc)
    state_after = sha(state.read_bytes()) if state.is_file() else None
    result = {'schema': 'alice.telemetry.publisher-repair.result.v1', 'report': report,
        'transport_exit': receipt['exit_code'], 'parse_error': parse_error,
        'local_a2_state_before_sha256': state_before, 'local_a2_state_after_sha256': state_after,
        'local_a2_state_unchanged': state_before == state_after, 'scientific_outcome': 'NOT_EVALUATED'}
    (out / 'result.json').write_bytes(json_bytes(result))
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in sorted(out.iterdir())]
    (out / 'EVIDENCE_MANIFEST.json').write_bytes(json_bytes({'files': files}))
    archive = out.parent / (out.name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()): z.write(path, path.name)
    print('RESULT_ZIP=' + str(archive)); print('RESULT_SHA256=' + sha(archive.read_bytes()))
    status = report.get('status') if report else 'INCOMPLETE_TRANSPORT'
    print('publisher_repair_status=' + str(status))
    if report and report.get('error'): print('stop=' + report['error'])
    print('Qwen remains NOT_EVALUATED. Return the result ZIP and transcript. Do not reset or resubmit a2.')
    return 0 if receipt['exit_code'] == 0 and status == 'INSTALLED_AND_TERMINAL_RETRY_VERIFIED' and state_before == state_after else 74

if __name__ == '__main__': sys.exit(main())
