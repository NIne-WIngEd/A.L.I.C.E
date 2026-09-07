"""Read selected public-run metadata; never import or execute the ALICE workload.

Runs from SSH stdin with Python -B. No filesystem writes, model service, model
weights, telemetry publication, scheduler mutations or package installation.
"""
import hashlib
import json
import os
from pathlib import Path
import ssl
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request

MAX_FILE = 4 * 1024 * 1024
FIELDS = (
    'run_id', 'package_sha256', 'amendment_sha256', 'phase', 'state', 'status',
    'job_id', 'job_submitted', 'class', 'source_run_id', 'source_job_id',
    'execution_run_id', 'zero_inference_confirmed', 'source_run_modified',
    'tasks_attempted', 'qualification_passed', 'runtime_preflight_passed',
    'effective_contract_sha256', 'exit_code', 'recorded_at', 'observed_at',
    'started_at', 'finished_at', 'ca_path', 'ca_sha256', 'verify_mode',
    'check_hostname', 'full_manifest_digest', 'model_tag', 'inference_started',
    'staged', 'telemetry_pending',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_fact(path, expected=None, fields=False):
    """Distinguish absent, unreadable, linked, oversized and changed bytes."""
    p = Path(path)
    result = {'path': p.as_posix(), 'expected_sha256': expected}
    try:
        info = p.lstat()
        if stat.S_ISLNK(info.st_mode):
            return {**result, 'status': 'SYMLINK'}
        if not stat.S_ISREG(info.st_mode):
            return {**result, 'status': 'NOT_REGULAR_FILE'}
        if info.st_size > MAX_FILE:
            return {**result, 'status': 'OVER_READ_LIMIT', 'bytes': info.st_size}
        data = p.read_bytes()
        result.update(bytes=len(data), sha256=sha(data), status='PRESENT')
        if expected is not None:
            result['status'] = 'MATCH' if result['sha256'] == expected else 'HASH_MISMATCH'
        if fields:
            try:
                doc = json.loads(data)
                if isinstance(doc, dict):
                    result['fields'] = {k: v for k, v in doc.items()
                                        if k in FIELDS and (v is None or type(v) in (str, int, bool, float))}
                    # Report the known failure without arbitrary exception text.
                    message = doc.get('message', '')
                    result['known_failure_markers'] = [s for s in (
                        'Source failure evidence differs: package-manifest.json',
                        'CERTIFICATE_VERIFY_FAILED', 'immutable evidence differs: run.json')
                        if s in message]
            except (ValueError, TypeError):
                result['json_status'] = 'INVALID_JSON'
        return result
    except FileNotFoundError:
        return {**result, 'status': 'MISSING'}
    except OSError as exc:
        return {**result, 'status': 'READ_ERROR', 'error_class': type(exc).__name__}


def scheduler(argv, identities, command=subprocess.run):
    try:
        result = command(argv, capture_output=True, text=True, timeout=20)
        rows = []
        for line in result.stdout.splitlines():
            cols = line.strip().split('|')
            if len(cols) >= 3 and cols[0].isdigit() and (cols[1] in identities or cols[0] == '575089'):
                rows.append(cols[:4])
        return {'argv': argv, 'exit_code': result.returncode,
                'query_succeeded': result.returncode == 0,
                'matching_rows': rows,
                'stderr_bytes': len(result.stderr.encode('utf-8', 'replace'))}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'query_succeeded': False, 'error_class': type(exc).__name__}


class HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('HTTPS_DOWNGRADE_REFUSED')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def https_heads(policy):
    candidates = [Path(s) for s in policy['ca_paths']]
    selected = next((p for p in candidates if p.is_file() and os.access(p, os.R_OK)), None)
    result = {'scope': 'login node only; HEAD requests; no body download or model lock',
              'candidates': [{'path': p.as_posix(), 'readable': p.is_file() and os.access(p, os.R_OK)}
                             for p in candidates], 'endpoints': []}
    if selected is None:
        return {**result, 'status': 'NO_READABLE_CA'}
    # CA files are commonly symlinks; the system trust selection follows them,
    # exactly as v102 does. Only public CA bytes are hashed; none are exported.
    try:
        context = ssl.create_default_context(cafile=str(selected))
        result.update(ca_path=selected.as_posix(), ca_sha256=sha(selected.read_bytes()),
                      ca_bytes=selected.stat().st_size, openssl=ssl.OPENSSL_VERSION,
                      verify_mode=context.verify_mode.name, check_hostname=context.check_hostname,
                      ca_certificates=context.cert_store_stats()['x509_ca'])
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname or not result['ca_certificates']:
            return {**result, 'status': 'UNSAFE_OR_EMPTY_TRUST_REFUSED'}
    except (OSError, ssl.SSLError) as exc:
        return {**result, 'status': 'CA_CONFIGURATION_ERROR', 'error_class': type(exc).__name__}
    for endpoint in policy['https_endpoints']:
        row = {'name': endpoint['name'], 'url': endpoint['url'], 'method': 'HEAD'}
        try:
            if urllib.parse.urlsplit(endpoint['url']).scheme != 'https':
                raise ValueError('HTTPS_REQUIRED')
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), HTTPSOnly())
            with opener.open(urllib.request.Request(endpoint['url'], method='HEAD'), timeout=12) as response:
                row.update(status='RESPONSE', http_status=response.status,
                           final_host=urllib.parse.urlsplit(response.geturl()).hostname)
                # Deliberately do not read response bodies or retain signed redirect URLs.
        except Exception as exc:
            row.update(status='REQUEST_ERROR', error_class=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                row['http_status'] = exc.code
            reason = getattr(exc, 'reason', None)
            if reason is not None:
                row['reason_class'] = type(reason).__name__
            if isinstance(reason, ssl.SSLCertVerificationError):
                row['ssl_verify_code'] = reason.verify_code
        result['endpoints'].append(row)
    result['status'] = 'OBSERVED'
    return result


def inspect(policy, command=subprocess.run, network=https_heads):
    root = Path(policy['remote_root'])
    source_id, current_id = policy['source_id'], policy['current_id']
    source, current = root / 'runs' / source_id, root / 'runs' / current_id
    report = {'schema': 'alice.qwen.read-only-remote-trace.v1',
              'observed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'read_only': True, 'execution_authority_granted': False,
              'source_id': source_id, 'current_id': current_id,
              'python_version': sys.version.split()[0], 'python_executable': sys.executable,
              'source_files': {}, 'packages': {}, 'current_files': {}}
    for name, expected in policy['source_critical_files'].items():
        report['source_files'][name] = file_fact(source / name, expected, fields=True)
    for label, pkg in policy['packages'].items():
        base = root / 'packages' / pkg['run_id'] / pkg['zip_sha256'] / pkg['name']
        entries = {'PACKAGE_MANIFEST.json': pkg['manifest_sha256'], **pkg['files']}
        report['packages'][label] = {
            'root': base.as_posix(),
            'zip': file_fact(base.parent.with_suffix('.zip'), pkg['zip_sha256']),
            'files': {name: file_fact(base / name, expected) for name, expected in entries.items()}}
    for name in policy['current_files']:
        report['current_files'][name] = file_fact(current / name, fields=name.endswith('.json'))
    for label, run in (('source', source), ('current', current)):
        paths = sorted(run.glob('tasks/*/attempt.json'))
        report[label + '_task_intents'] = {
            'count': len(paths), 'paths': [p.relative_to(run).as_posix() for p in paths[:128]],
            'truncated': len(paths) > 128}
        report[label + '_progress_markers'] = {name: file_fact(run / name)
                                               for name in policy['progress_markers']}
    report['scheduler'] = {
        'queue': scheduler(['squeue', '-h', '-u', 'mxrayan', '-o', '%A|%j|%T'], (source_id, current_id), command),
        'source_accounting': scheduler(['sacct', '-X', '-n', '-P', '--starttime', '2026-09-06', '-j', '575089',
            '--format', 'JobIDRaw,JobName%100,State,ExitCode'], (source_id,), command),
        'current_accounting': scheduler(['sacct', '-X', '-n', '-P', '-u', 'mxrayan', '--starttime', '2026-09-06',
            '--name', current_id, '--format', 'JobIDRaw,JobName%100,State,ExitCode'], (current_id,), command)}
    report['telemetry_helper'] = file_fact(root / 'bin/rayan-telemetry-push.sh')
    report['https'] = network(policy)
    report['next_action'] = 'RETURN_DIAGNOSTIC_FOR_REVIEW; no submission or repair is performed'
    return report
