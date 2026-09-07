"""Read-only Git transport receipt; original helper and execution state untouched."""
import argparse
import base64
import datetime
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

ROOT = Path('/homes/01/mxrayan/rayan-compute')
HELPER_SHA = 'ba4482728c09313aede5031e682f68e8007c79c9cfa084d3da4b11aa984d0830'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def stream(data):
    data = data or b''
    return {'bytes': len(data), 'sha256': sha(data), 'base64': base64.b64encode(data).decode('ascii')}

def run_bounded(argv, timeout, capture_output=True, input=None, **kwargs):
    # Kill the Git+SSH process group on POSIX so an inherited pipe cannot make
    # timeout collection wait for an orphaned transport process.
    p = subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name != 'nt', **kwargs)
    try:
        stdout, stderr = p.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == 'nt': p.kill()
        else:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        stdout, stderr = p.communicate()
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(argv, p.returncode, stdout, stderr)

def capture(argv, timeout, command=run_bounded, **kwargs):
    record = {'argv': argv, 'started_at': now(), 'timeout_seconds': timeout}
    try:
        p = command(argv, capture_output=True, timeout=timeout, **kwargs)
        record.update(exit_code=p.returncode, status='RETURNED', stdout=stream(p.stdout), stderr=stream(p.stderr))
    except subprocess.TimeoutExpired as exc:
        record.update(exit_code=None, status='TIMEOUT', stdout=stream(exc.stdout), stderr=stream(exc.stderr))
    except OSError as exc:
        record.update(exit_code=None, status='OS_ERROR', error=str(exc), stdout=stream(b''), stderr=stream(b''))
    record['finished_at'] = now()
    return record

def file_record(path, include_bytes=False):
    item = {'path': str(path)}
    try:
        if path.is_symlink(): return dict(item, status='SYMLINK_REFUSED')
        st = path.stat()
        item.update(mode=oct(st.st_mode & 0o777), uid=st.st_uid, bytes=st.st_size)
        if not path.is_file(): return dict(item, status='NON_REGULAR')
        if st.st_size > 1024 * 1024: return dict(item, status='OVER_LIMIT')
        data = path.read_bytes()
        item.update(status='READ', sha256=sha(data))
        if include_bytes: item['base64'] = base64.b64encode(data).decode('ascii')
    except FileNotFoundError: item['status'] = 'MISSING'
    except OSError as exc: item.update(status='UNREADABLE', error=str(exc))
    return item

def inventory():
    paths = ['bin/rayan-telemetry-push.sh', 'bin/rayan-github-ssh',
        'telemetry/ledger/.git/HEAD', 'telemetry/ledger/.git/FETCH_HEAD',
        'telemetry/ledger/.git/index', 'telemetry/ledger/.git/packed-refs',
        'telemetry/ledger/.git/refs/heads/main', 'telemetry/ledger/.git/refs/remotes/origin/main']
    return [file_record(ROOT / p, include_bytes=p == 'bin/rayan-github-ssh') for p in paths]

def remote():
    report = {'schema': 'alice.telemetry.transport.remote.v1', 'started_at': now(),
        'scope': 'Login-node observation; no execution authority or live worker success inferred',
        'before': inventory(), 'commands': [], 'helper_expected_sha256': HELPER_SHA}
    helper = report['before'][0]
    if helper.get('sha256') != HELPER_SHA:
        report['status'] = 'HELPER_CHANGED_OR_UNAVAILABLE'
    else:
        env = dict(os.environ, GIT_SSH=str(ROOT / 'bin/rayan-github-ssh'), GIT_SSH_VARIANT='ssh', GIT_TERMINAL_PROMPT='0')
        # Use the installed helper's transport wrapper and existing clone configuration.
        # Dry-run flags prevent fetching or publishing refs/objects. Raw errors are retained.
        commands = [(['git', '--version'], 10),
            (['git', 'fetch', '--dry-run', 'origin', 'main'], 20),
            (['git', 'push', '--dry-run', 'origin', 'HEAD:refs/heads/main'], 20)]
        for argv, limit in commands:
            receipt = capture(argv, limit, cwd=str(ROOT / 'telemetry/ledger'), env=env)
            report['commands'].append(receipt)
            print(json.dumps({'event': 'command', 'receipt': receipt}), flush=True)
        report['status'] = 'COLLECTED'
    report['after'] = inventory()
    report['selected_files_stable'] = report['before'] == report['after']
    report['finished_at'] = now()
    print(json.dumps({'event': 'report', 'report': report}), flush=True)

def remote_script():
    code = Path(__file__).read_text(encoding='utf-8')
    code = code.rsplit("\nif __name__ == '__main__':", 1)[0]
    # This bootstrap and native Windows SSH key handling are reused from the
    # successfully executed Magnolia provider trace, without module installation.
    shell = '''#!/bin/bash
set -euo pipefail
PYROOT="/modules/pkgs/common/python/3.11.5"
PY="$PYROOT/bin/python3.11"
PYLIB=""
for d in "$PYROOT/lib" "$PYROOT/lib64"; do
  if [ -e "$d/libpython3.11.so.1.0" ]; then PYLIB="$d"; break; fi
done
test -n "$PYLIB"
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PY" -B - <<'ALICE_TELEMETRY_TRANSPORT_PY'
'''
    return (shell + code + '\nremote()\nALICE_TELEMETRY_TRANSPORT_PY\n').encode('utf-8')

def ssh_argv(key):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
        '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
        '-i', str(key), 'mxrayan@magnolia.usm.edu', 'bash -s']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path.home() / 'Downloads')
    args = parser.parse_args()
    out = args.output_root / ('ALICE_TelemetryTransport_' + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    script = remote_script()
    (out / 'remote-input.sh').write_bytes(script)
    print('action=READ_ONLY_GIT_TRANSPORT; commands=3; no telemetry helper execution', flush=True)
    receipt = capture(ssh_argv(Path.home() / '.ssh/rayan_magnolia_ed25519'), 85, input=script)
    # Save the actual transport bytes before parsing. A failed command is evidence.
    for name in ['stdout', 'stderr']:
        (out / ('ssh.' + name + '.bin')).write_bytes(base64.b64decode(receipt[name]['base64']))
    (out / 'transport.json').write_text(json.dumps(receipt, indent=2) + '\n')
    report = None
    parse_errors = []
    for index, line in enumerate(base64.b64decode(receipt['stdout']['base64']).splitlines()):
        try:
            event = json.loads(line)
            if event.get('event') == 'report': report = event['report']
        except (ValueError, AttributeError) as exc: parse_errors.append({'line': index, 'error': str(exc)})
    result = {'schema': 'alice.telemetry.transport.result.v1', 'remote_script_sha256': sha(script),
        'status': report['status'] if receipt['exit_code'] == 0 and report is not None else 'INCOMPLETE_TRANSPORT',
        'report': report, 'parse_errors': parse_errors, 'telemetry_published': False,
        'execution_authority_granted': False, 'model_jobs_submitted': 0}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in sorted(out.iterdir())]
    (out / 'EVIDENCE_MANIFEST.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    archive = out.parent / (out.name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()): z.write(p, p.name)
    print('TRACE_ZIP=' + str(archive))
    print('TRACE_SHA256=' + sha(archive.read_bytes()))
    print('diagnostic_status=' + result['status'])
    if report:
        for cmd in report['commands']: print('command=' + ' '.join(cmd['argv']) + ' status=' + cmd['status'] + ' exit=' + str(cmd['exit_code']))
    print('Return TRACE_ZIP even if Git or SSH failed. Keep a2 state and existing launchers unchanged.')
    return 0 if result['status'] == 'COLLECTED' else 74

if __name__ == '__main__': sys.exit(main())
