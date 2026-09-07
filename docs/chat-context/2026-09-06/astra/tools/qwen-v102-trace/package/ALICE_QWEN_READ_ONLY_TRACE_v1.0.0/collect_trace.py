"""Windows/local controller for one read-only SSH observation. Standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import remote_trace

BASE = Path(__file__).resolve().parent
HOST = 'mxrayan@magnolia.usm.edu'


def canonical(doc):
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def verify_package(base=BASE):
    manifest = json.loads((base / 'PACKAGE_MANIFEST.json').read_bytes())
    listed = set()
    for entry in manifest['files']:
        path = base / entry['path']
        if (not path.resolve().is_relative_to(base.resolve()) or path.is_symlink()
                or entry['path'] in listed):
            raise ValueError('Invalid package manifest path')
        data = path.read_bytes()
        if len(data) != entry['bytes'] or remote_trace.sha(data) != entry['sha256']:
            raise ValueError('Package file hash differs: ' + entry['path'])
        listed.add(entry['path'])
    actual = {p.relative_to(base).as_posix() for p in base.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    if actual != listed | {'PACKAGE_MANIFEST.json'}:
        raise ValueError('Unexpected package files')
    return len(listed)


def remote_script(policy):
    code = (BASE / 'remote_trace.py').read_text(encoding='utf-8')
    code += '\nprint(json.dumps(inspect(' + repr(policy) + '), sort_keys=True))\n'
    compile(code, '<read-only-remote-trace>', 'exec')
    return ('''#!/bin/bash
set -euo pipefail
PYROOT="/modules/pkgs/common/python/3.11.5"
PY="$PYROOT/bin/python3.11"
PYLIB=""
for d in "$PYROOT/lib" "$PYROOT/lib64"; do
  if [ -e "$d/libpython3.11.so.1.0" ]; then PYLIB="$d"; break; fi
done
test -n "$PYLIB"
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PY" -B - <<'ALICE_READ_ONLY_PY'
''' + code + '\nALICE_READ_ONLY_PY\n').encode('utf-8')


def ssh_argv(key):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            '-i', str(key), HOST, 'bash -s']


def local_state(vault, policy):
    result = {}
    for label, suffix in (('source', 'a1'), ('current', 'a2')):
        path = vault / 'tools/alice-astra' / ('qwen-fallback-' + suffix) / 'controller-state.json'
        fact = remote_trace.file_fact(path, fields=True)
        # Retain only explicitly selected public lifecycle metadata, never keys or configuration.
        result[label] = fact
    return result


def execute(args):
    verify_package()
    if not args.ssh_key.is_file() or shutil.which('ssh') is None:
        raise ValueError('Recorded Magnolia SSH identity or ssh application is unavailable')
    policy = json.loads((BASE / 'trace_policy.json').read_bytes())
    before = local_state(args.vault_root, policy)
    script = remote_script(policy)
    print('action=READ_ONLY_HOST_TRACE; no package staging, job submission, inference or state repair', flush=True)
    print('checking=source origins; a2 state; scheduler; system CA; two HTTPS HEAD endpoints', flush=True)
    report = {'schema': 'alice.qwen.read-only-trace-bundle.v1', 'read_only': True,
              'observed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'policy_sha256': remote_trace.sha((BASE / 'trace_policy.json').read_bytes()),
              'remote_script_sha256': remote_trace.sha(script),
              'local_state_before': before, 'execution_authority_granted': False}
    code = 0
    try:
        process = subprocess.run(ssh_argv(args.ssh_key), input=script,
                                 capture_output=True, timeout=180)
        report['ssh_exit_code'] = process.returncode
        report['ssh_stderr_bytes'] = len(process.stderr)
        report['transport_error_markers'] = [s for s in (
            'Permission denied', 'Host key verification failed', 'Connection timed out',
            'Could not resolve hostname', 'error while loading shared libraries',
            'No such file or directory', 'Connection refused')
            if s.encode() in process.stderr]
        if process.returncode:
            report['transport_status'] = 'REMOTE_OR_TRANSPORT_ERROR'; code = 74
        else:
            remote = json.loads(process.stdout)
            if (remote.get('schema') != 'alice.qwen.read-only-remote-trace.v1'
                    or remote.get('source_id') != policy['source_id']
                    or remote.get('current_id') != policy['current_id']
                    or remote.get('read_only') is not True):
                raise ValueError('Unexpected diagnostic identity')
            report['remote'] = remote
            report['transport_status'] = 'COLLECTED'
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        report['transport_status'] = 'INCOMPLETE'
        report['error_class'] = type(exc).__name__; code = 74
    report['local_state_after'] = local_state(args.vault_root, policy)
    report['local_controller_states_unchanged'] = before == report['local_state_after']
    report['compute_job_submitted_by_diagnostic'] = False
    report['inference_requested_by_diagnostic'] = False
    # Only this launcher's fresh output directory is written.
    target = args.output / 'diagnostic-result'
    target.mkdir(parents=True, exist_ok=False)
    data = canonical(report)
    (target / 'report.json').write_bytes(data)
    manifest = canonical({'schema': 'alice.qwen.diagnostic-manifest.v1', 'files': [
        {'path': 'report.json', 'bytes': len(data), 'sha256': remote_trace.sha(data)}]})
    archive = args.output / ('ALICE_QWEN_TRACE_' + remote_trace.sha(data)[:12] + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('report.json', data)
        z.writestr('DIAGNOSTIC_MANIFEST.json', manifest)
    print('TRACE_ZIP=' + str(archive), flush=True)
    print('TRACE_SHA256=' + remote_trace.sha(archive.read_bytes()), flush=True)
    print('diagnostic_status=' + report['transport_status'], flush=True)
    print('No calibration outcome or new execution authority is inferred. Return the TRACE_ZIP.', flush=True)
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vault-root', type=Path, default=Path('C:/ALICE_Vault'))
    p.add_argument('--ssh-key', type=Path, default=Path.home() / '.ssh/rayan_magnolia_ed25519')
    p.add_argument('--output', type=Path, required=True)
    try:
        raise SystemExit(execute(p.parse_args()))
    except Exception as exc:
        print('READ_ONLY_TRACE_STOP=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        raise SystemExit(76)
