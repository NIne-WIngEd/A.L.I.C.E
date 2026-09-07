"""A single explicit infrastructure successor after the exact zero-inference failure."""
from pathlib import Path
import urllib.request
import urllib.parse

import contract as c
import secure_https
import evidence_origins


def policy():
    doc = c.read(c.BASE / 'authority/infra_recovery.json')
    c.require(doc['execution_run_id'] == c.RUN_ID and doc['calibration_id'] == c.CALIBRATION_ID, 'Infrastructure successor identity')
    return doc


def verify_source_files(source):
    source = Path(source)
    rules = policy()
    evidence_origins.verify_package_origin(evidence_origins.source_package(rules),
        rules['source_critical_files']['package-manifest.json'], rules['source_workload_sha256'])
    origins = evidence_origins.source_paths(source, rules)
    for name, expected in rules['source_critical_files'].items():
        file = origins[name]
        c.require(file.is_file() and not file.is_symlink(), 'Source evidence origin missing or linked: ' + name)
        c.require(c.file_sha(file) == expected, 'Source evidence origin hash differs: ' + name)
    result = c.read(source / 'result.json')
    failure = c.read(source / 'failure.json')
    c.require(result['run_id'] == c.CALIBRATION_ID and result['package_sha256'] == rules['source_workload_sha256'], 'Source workload identity')
    c.require(result['tasks_attempted'] == 0 and result['scores']['tasks_complete'] == 0 and all(x['status'] == 'NOT_ATTEMPTED' for x in result['scores']['matrix']), 'A prior judgment attempt forbids this infrastructure successor')
    c.require(result['effective_contract_sha256'] is None and result['runtime_preflight_passed'] is False, 'Source reached runtime or inference')
    c.require(failure['phase'] == rules['source_phase_required'] and rules['source_error_required'] in failure['message'], 'Unexpected source failure')
    c.require(not list(source.glob('tasks/*/attempt.json')), 'A source task intent exists')
    for name in ('probe-attempt.json', 'probe-response.ndjson', 'runtime.json', 'model-lock.json', 'effective-contract.json'):
        c.require(not (source / name).exists(), 'Source progressed beyond the approved repair boundary: ' + name)
    return rules


def reconcile_source(command):
    source = c.ROOT / 'runs' / c.CALIBRATION_ID
    rules = verify_source_files(source)
    job = rules['source_job_id']
    queue = command(['squeue', '-h', '-j', job, '-o', '%A|%j|%T'], check=False)
    c.require(queue.returncode == 0, 'Source queue query failed; absence is unknown')
    c.require(not any(line.split('|')[0] == job for line in queue.stdout.splitlines()), 'Source job is still queued or running')
    account = command(['sacct', '-X', '-n', '-P', '--starttime', '2026-09-06', '-j', job, '--format', 'JobIDRaw,JobName%100,State,ExitCode'])
    c.require(account.returncode == 0, 'Source accounting query failed')
    rows = [line.strip().split('|') for line in account.stdout.splitlines() if line.strip().split('|')[0] == job]
    c.require(len(rows) == 1 and len(rows[0]) >= 4, 'Source accounting receipt missing or ambiguous')
    row = rows[0]
    c.require(row[1] == c.CALIBRATION_ID and row[2].split()[0].rstrip('+') == 'FAILED' and row[3] == '76:0', 'Source scheduler terminal identity differs')
    return {'schema': 'alice.mc10d.infrastructure-parent-reconciliation.v1', 'source_run_id': c.CALIBRATION_ID, 'source_job_id': job, 'source_scheduler_state': 'FAILED', 'source_exit_code': row[3], 'source_result_zip_sha256': rules['source_result_zip_sha256'], 'source_result_sha256': rules['source_critical_files']['result.json'], 'zero_inference_confirmed': True, 'source_run_modified': False, 'execution_run_id': c.RUN_ID,
            'source_package_manifest_sha256': rules['source_critical_files']['package-manifest.json'],
            'source_package_sha256': rules['source_workload_sha256'],
            'source_package_origin': evidence_origins.source_package(rules).as_posix(),
            'source_manifest_origin': 'PACKAGE', 'source_queue_query_succeeded': True}


def preflight(run, command):
    """Bounded login-node checks before sbatch; no model service or inference."""
    run = Path(run)
    parent = reconcile_source(command)
    c.immutable(run / 'parent-reconciliation.json', parent)
    trust = secure_https.configure(run)
    _, _, runtime = c.authority()
    request = urllib.request.Request(runtime['archive_url'], method='HEAD')
    with secure_https.open_url(request, timeout=25) as response:
        c.require(response.status == 200, 'Runtime HTTPS HEAD did not succeed')
        final = urllib.parse.urlsplit(response.geturl())
        c.require(final.scheme == 'https', 'Runtime redirect downgraded HTTPS')
        destination = final.hostname
    # Freeze the public model manifest once. The worker reuses these exact bytes.
    import worker
    lock, _, _ = worker.resolve_manifest(run)
    receipt = {'schema': 'alice.mc10d.infrastructure-preflight.v1', 'status': 'PASS', 'execution_run_id': c.RUN_ID, 'source_job_id': parent['source_job_id'], 'parent_reconciliation_sha256': c.file_sha(run / 'parent-reconciliation.json'), 'tls_trust_sha256': c.file_sha(run / 'tls-trust.json'), 'ca_sha256': trust['ca_sha256'], 'runtime_https_head_status': 200, 'runtime_redirect_host': destination, 'model_manifest_sha256': lock['full_manifest_digest'], 'model_service_started': False, 'probe_or_task_attempted': False, 'checked_at': c.now()}
    c.write(run / 'infrastructure-preflight.json', receipt)
    return receipt


def verify_evidence(folder):
    """Evidence-only reconstruction; never makes an SSH or HTTPS request."""
    folder = Path(folder)
    for name in ('infra_recovery.json',):
        c.require((folder / 'authority' / name).read_bytes() == (c.BASE / 'authority' / name).read_bytes(), 'Infrastructure authority bytes differ')
    parent = c.read(folder / 'parent-reconciliation.json')
    rules = policy()
    c.require(parent['source_run_id'] == c.CALIBRATION_ID and parent['source_job_id'] == rules['source_job_id'] and parent['source_exit_code'] == '76:0' and parent['zero_inference_confirmed'] is True and parent['source_run_modified'] is False, 'Source reconciliation evidence')
    c.require(parent['source_result_zip_sha256'] == rules['source_result_zip_sha256'] and parent['source_result_sha256'] == rules['source_critical_files']['result.json'], 'Source evidence hash lineage')
    c.require(parent['source_package_manifest_sha256'] == rules['source_critical_files']['package-manifest.json'] and parent['source_package_sha256'] == rules['source_workload_sha256'] and parent['source_manifest_origin'] == 'PACKAGE' and parent['source_queue_query_succeeded'] is True, 'Source package-origin evidence')
    c.require(parent['source_package_origin'] == evidence_origins.source_package(rules).as_posix(), 'Source package physical locator differs')
    preflight = c.read(folder / 'infrastructure-preflight.json')
    trust = c.read(folder / 'tls-trust.json')
    c.require(preflight['status'] == 'PASS' and preflight['execution_run_id'] == c.RUN_ID and preflight['parent_reconciliation_sha256'] == c.file_sha(folder / 'parent-reconciliation.json'), 'Pre-submission infrastructure gate')
    c.require(preflight['tls_trust_sha256'] == c.file_sha(folder / 'tls-trust.json') and preflight['ca_sha256'] == trust['ca_sha256'], 'CA evidence lineage')
    c.require(trust['verify_mode'] == 'CERT_REQUIRED' and trust['check_hostname'] is True and preflight['runtime_https_head_status'] == 200, 'Verified TLS gate')
    c.require(preflight['model_manifest_sha256'] == c.file_sha(folder / 'model-manifest.json'), 'Pre-submission model manifest changed')
