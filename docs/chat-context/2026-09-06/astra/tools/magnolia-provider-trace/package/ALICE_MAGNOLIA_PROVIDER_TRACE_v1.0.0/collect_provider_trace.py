"""One owner-machine observation; writes only its fresh diagnostic output folder."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import trace_provider as trace

BASE = Path(__file__).resolve().parent
HOST = 'mxrayan@magnolia.usm.edu'


def verify_package(base=BASE):
    manifest = json.loads((base / 'PACKAGE_MANIFEST.json').read_bytes())
    listed = set()
    for entry in manifest['files']:
        path = trace.rooted(base, entry['path'])
        if entry['path'] in listed or trace.guard(base, path) or path.is_symlink():
            raise ValueError('Invalid package manifest path')
        data = path.read_bytes()
        if len(data) != entry['bytes'] or trace.sha(data) != entry['sha256']:
            raise ValueError('Package hash differs: ' + entry['path'])
        listed.add(entry['path'])
    actual = {p.relative_to(base).as_posix() for p in base.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    if actual != listed | {'PACKAGE_MANIFEST.json'}: raise ValueError('Unexpected package files')
    return len(listed)


def remote_script(policy):
    code = (BASE / 'trace_provider.py').read_text(encoding='utf-8')
    code += '\nprint(canonical(observe_remote(' + repr(policy) + ')).decode(), end="")\n'
    compile(code, '<provider-observation>', 'exec')
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
exec "$PY" -B - <<'ALICE_PROVIDER_OBSERVATION_PY'
'''
    return (shell + code + '\nALICE_PROVIDER_OBSERVATION_PY\n').encode('utf-8')


def ssh_argv(key):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            '-i', str(key), HOST, 'bash -s']


def execute(args, command=subprocess.run):
    verify_package()
    policy = json.loads((BASE / 'trace_policy.json').read_bytes())
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(args.vault_root).resolve()):
        raise ValueError('Diagnostic output must be outside the existing Vault')
    target = output / 'provider-result'
    target.mkdir(parents=True, exist_ok=False)
    script = remote_script(policy)
    before = trace.inventory(args.vault_root, 'local', policy)
    report = {'schema': 'alice.magnolia.provider-trace-bundle.v1', 'read_only': True,
              'observed_at': trace.now(), 'policy_sha256': trace.sha((BASE / 'trace_policy.json').read_bytes()),
              'remote_script_sha256': trace.sha(script), 'local_inventory_before': before,
              'execution_authority_granted': False, 'model_jobs_submitted': 0,
              'remote_package_staging': False, 'state_repair': False, 'telemetry_publication': False}
    transport = {'argv': ssh_argv(args.ssh_key), 'observed_at': trace.now()}
    print('action=READ_ONLY_PROVIDER_AND_REVISION_TRACE', flush=True)
    print('checking=11 bounded host/scheduler commands; local and remote a2 revision inventories', flush=True)
    print('provider_errors_are_evidence=true; no submission, package staging, inference or state repair', flush=True)
    code = 0
    try:
        if not args.ssh_key.is_file() or shutil.which('ssh') is None:
            raise OSError('Recorded SSH identity file or ssh application is unavailable')
        process = command(ssh_argv(args.ssh_key), input=script, capture_output=True, timeout=300)
        transport.update(exit_code=process.returncode, status='RETURNED',
                         stdout=trace.byte_stream(process.stdout, 8 * 1024 * 1024),
                         stderr=trace.byte_stream(process.stderr, 2 * 1024 * 1024))
        # Retain raw transport first. A parse/assertion failure cannot erase it.
        if process.returncode:
            report['status'] = 'REMOTE_OR_TRANSPORT_ERROR'; code = 74
        elif transport['stdout']['truncated'] or transport['stderr']['truncated']:
            report['status'] = 'TRANSPORT_OUTPUT_OVER_LIMIT'; code = 74
        else:
            remote = json.loads(process.stdout)
            if (not isinstance(remote, dict)
                    or remote.get('schema') != 'alice.magnolia.read-only-provider-observation.v1'
                    or remote.get('source_id') != policy['source_id']
                    or remote.get('current_id') != policy['current_id'] or remote.get('read_only') is not True):
                raise ValueError('Unexpected remote observation identity')
            report['remote'] = remote
            report['status'] = 'COLLECTED_PROVIDER_ERRORS' if remote['provider_command_failures'] else 'COLLECTED'
            if not remote['all_raw_streams_complete'] or not remote['inventory_before']['inventory_complete'] or not remote['inventory_after']['inventory_complete']:
                report['status'] = 'COLLECTED_INCOMPLETE_OBSERVATION'; code = 74
            if not remote['inventories_equal']:
                report['status'] = 'STATE_CHANGED_DURING_OBSERVATION'; code = 74
    except subprocess.TimeoutExpired as exc:
        transport.update(exit_code=None, status='TIMEOUT', stdout=trace.byte_stream(exc.output, 8 * 1024 * 1024), stderr=trace.byte_stream(exc.stderr, 2 * 1024 * 1024))
        report['status'] = 'TRANSPORT_TIMEOUT'; code = 74
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if 'status' not in transport:
            transport.update(exit_code=None, status='EXEC_ERROR', stdout=trace.byte_stream(b''), stderr=trace.byte_stream(b''))
        report.update(status='INCOMPLETE_OBSERVATION', error_class=type(exc).__name__, error_message=str(exc)); code = 74
    transport['finished_at'] = trace.now()
    report['transport'] = transport
    report['local_inventory_after'] = trace.inventory(args.vault_root, 'local', policy)
    report['local_inventories_equal'] = before == report['local_inventory_after']
    if not report['local_inventories_equal']:
        report['status'] = 'LOCAL_STATE_CHANGED_DURING_OBSERVATION'; code = 74
    if not before['inventory_complete'] or not report['local_inventory_after']['inventory_complete']:
        report['status'] = 'LOCAL_INVENTORY_INCOMPLETE'; code = 74
    report['finished_at'] = trace.now()
    raw = trace.canonical(report)
    (target / 'report.json').write_bytes(raw)
    manifest = trace.canonical({'schema': 'alice.provider-trace.evidence-manifest.v1',
                                'files': [{'path': 'report.json', 'bytes': len(raw), 'sha256': trace.sha(raw)}]})
    archive = output / ('ALICE_MAGNOLIA_PROVIDER_TRACE_' + trace.sha(raw)[:12] + '.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('report.json', raw); z.writestr('EVIDENCE_MANIFEST.json', manifest)
    print('TRACE_ZIP=' + str(archive), flush=True)
    print('TRACE_SHA256=' + trace.sha(archive.read_bytes()), flush=True)
    print('diagnostic_status=' + report['status'], flush=True)
    print('provider_command_failures=' + str(report.get('remote', {}).get('provider_command_failures', 'UNKNOWN')), flush=True)
    print('Return this trace even when scheduler commands failed. No execution eligibility is inferred.', flush=True)
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check', action='store_true')
    p.add_argument('--vault-root', type=Path, default=Path('C:/ALICE_Vault'))
    p.add_argument('--ssh-key', type=Path, default=Path.home() / '.ssh/rayan_magnolia_ed25519')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        if args.self_check:
            count = verify_package()
            files = sorted(BASE.glob('*.py'))
            for path in files: compile(path.read_bytes(), str(path), 'exec')
            print('diagnostic_manifest_verified=true files=' + str(count))
            print('compile_gate_passed=true files=' + str(len(files)))
            return 0
        if args.output is None: p.error('--output is required for observation')
        return execute(args)
    except Exception as exc:
        print('PROVIDER_TRACE_STOP=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        return 76


if __name__ == '__main__': raise SystemExit(main())
