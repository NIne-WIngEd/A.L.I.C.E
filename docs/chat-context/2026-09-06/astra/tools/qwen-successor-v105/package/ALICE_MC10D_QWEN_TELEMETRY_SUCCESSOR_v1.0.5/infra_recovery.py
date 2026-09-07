"""One explicit a3 execution; exact closed parents remain read-only."""
from pathlib import Path
import os
import urllib.request
import urllib.parse

import contract as c
import evidence_origins
import magnolia_scheduler as scheduler
import secure_https


def policy():
    rules = c.read(c.BASE / 'authority/infra_recovery.json')
    c.require(rules['execution_run_id'] == c.RUN_ID and rules['calibration_id'] == c.CALIBRATION_ID, 'Successor identity differs')
    c.require(rules['owner_approval_sha256'] == c.APPROVAL_SHA and rules['automatic_successor_after_this_attempt'] is False, 'Successor scope differs')
    return rules


def descriptor(package_sha):
    return {'run_id': c.RUN_ID, 'package_sha256': package_sha,
            'amendment_sha256': c.APPROVED_SHA, 'approved_profile_id': c.authority()[0]['profile']['id']}


def checked_file(path, expected):
    path = Path(path)
    c.require(path.is_file() and not path.is_symlink(), 'Required evidence absent or linked: ' + path.as_posix())
    c.require(c.file_sha(path) == expected, 'Evidence bytes changed: ' + path.as_posix())


def verify_local_parent(vault):
    rules = policy()
    checked_file(Path(vault) / rules['local_a2_state_relative'], rules['local_a2_state_sha256'])
    return rules['local_a2_state_sha256']


def verify_parent(parent):
    run = c.ROOT / 'runs' / parent['run_id']
    package = c.ROOT / 'packages' / parent['run_id'] / parent['package_sha256'] / parent['package_name']
    evidence_origins.verify_package_origin(package, parent['critical_files']['package-manifest.json'], parent['package_sha256'])
    for name, expected in parent['critical_files'].items():
        checked_file(package / 'PACKAGE_MANIFEST.json' if name == 'package-manifest.json' else run / name, expected)
    result = c.read(run / 'result.json')
    c.require(result['run_id'] == parent['run_id'] and result['package_sha256'] == parent['package_sha256'], 'Parent execution identity differs')
    c.require(result['tasks_attempted'] == 0 and result['scores']['tasks_complete'] == 0 and len(result['scores']['matrix']) == 16 and all(x['status'] == 'NOT_ATTEMPTED' for x in result['scores']['matrix']), 'A prior judgment forbids this infrastructure successor')
    c.require(result['effective_contract_sha256'] is None and result['runtime_preflight_passed'] is False, 'Parent reached inference boundary')
    failure = c.read(run / 'failure.json')
    c.require(failure['phase'] == parent['phase'] and failure['message'] == parent['error'], 'Unexpected parent failure')
    c.require(not list(run.glob('tasks/*/attempt.json')), 'Parent task intent exists')
    for name in ('probe-request.json', 'probe-attempt.json', 'probe-response.ndjson', 'runtime.json', 'effective-contract.json'):
        c.require(not (run / name).exists(), 'Parent has inference preparation: ' + name)
    return {'run_id': parent['run_id'], 'job_id': parent['job_id'], 'critical_files': parent['critical_files'],
            'package_origin': package.as_posix(), 'package_manifest_origin': 'PACKAGE',
            'package_sha256': parent['package_sha256'], 'result_archive_sha256': parent['result_archive_sha256'],
            'zero_inference_confirmed': True, 'source_run_modified': False}


def reconcile_source(command):
    rules = policy()
    parents = [verify_parent(p) for p in rules['parents']]
    active = scheduler.active_rows(command)
    for parent, receipt in zip(rules['parents'], parents):
        c.require(not any(row[0] == parent['job_id'] or row[1] == parent['run_id'] for row in active), 'Parent is queued or running')
        rows = [row for row in scheduler.historical_rows(command, parent['job_id']) if row[0] == parent['job_id']]
        c.require(len(rows) == 1, 'Parent accounting missing or ambiguous')
        row = rows[0]
        c.require(row[1] == parent['run_id'] and row[2].split()[0].rstrip('+') == parent['scheduler_state'] and row[3] == parent['scheduler_exit'], 'Parent scheduler terminal identity differs')
        receipt.update(scheduler_state=parent['scheduler_state'], scheduler_exit=parent['scheduler_exit'], queue_query_succeeded=True)
    return {'schema': 'alice.mc10d.qwen.closed-parents.v1', 'run_id': c.RUN_ID,
            'policy_sha256': c.file_sha(c.BASE / 'authority/infra_recovery.json'), 'parents': parents}


def verify_publisher():
    rules = policy()
    helper = c.ROOT / 'bin/rayan-telemetry-push.sh'
    checked_file(helper, rules['installed_helper_sha256'])
    c.require(os.access(helper, os.X_OK), 'Installed publisher is not executable')
    checked_file(c.ROOT / 'bin/rayan-github-ssh', rules['installed_wrapper_sha256'])
    for path, expected in rules['publisher_proofs'].items(): checked_file(c.ROOT / path, expected)
    return {'schema': 'alice.mc10d.qwen.publisher-binding.v1', 'run_id': c.RUN_ID,
            'helper_sha256': rules['installed_helper_sha256'], 'wrapper_sha256': rules['installed_wrapper_sha256'],
            'repair_zip_sha256': rules['publisher_repair_zip_sha256'], 'proofs': rules['publisher_proofs'],
            'verified_prior_terminal_commit': rules['verified_terminal_commit']}


def preflight(run, command):
    run = Path(run)
    c.immutable(run / 'parent-reconciliation.json', reconcile_source(command))
    c.immutable(run / 'publisher-installed.json', verify_publisher())
    trust = secure_https.configure(run)
    _, _, runtime = c.authority()
    with secure_https.open_url(urllib.request.Request(runtime['archive_url'], method='HEAD'), timeout=25) as response:
        c.require(response.status == 200, 'Runtime HTTPS HEAD did not succeed')
        final = urllib.parse.urlsplit(response.geturl())
        c.require(final.scheme == 'https', 'Runtime redirect downgraded HTTPS')
    rules = policy()
    # The exact a2 registry resolution is already frozen; preserve it without retagging.
    parent = next(p for p in rules['parents'] if p['label'] == 'a2')
    for name in ('model-lock.json', 'model-manifest.json'):
        source = c.ROOT / 'runs' / parent['run_id'] / name
        checked_file(source, parent['critical_files'][name])
        if (run / name).exists(): c.require((run / name).read_bytes() == source.read_bytes(), 'Frozen model copy differs')
        else: c.atomic_bytes(run / name, source.read_bytes())
    import worker
    lock, _, _ = worker.resolve_manifest(run)
    c.require(lock['full_manifest_digest'] == rules['frozen_model_manifest_sha256'], 'Resolved manifest differs from a2')
    receipt = {'schema': 'alice.mc10d.infrastructure-preflight.v1', 'status': 'PASS', 'execution_run_id': c.RUN_ID,
        'parent_reconciliation_sha256': c.file_sha(run / 'parent-reconciliation.json'),
        'publisher_binding_sha256': c.file_sha(run / 'publisher-installed.json'),
        'tls_trust_sha256': c.file_sha(run / 'tls-trust.json'), 'ca_sha256': trust['ca_sha256'],
        'runtime_https_head_status': 200, 'runtime_redirect_host': final.hostname,
        'model_manifest_sha256': lock['full_manifest_digest'], 'probe_or_task_attempted': False, 'checked_at': c.now()}
    c.write(run / 'infrastructure-preflight.json', receipt)
    return receipt


def evidence_paths(run):
    return {'publisher-installed.json': run / 'publisher-installed.json'} if (run / 'publisher-installed.json').is_file() else {}


def verify_evidence(folder):
    folder = Path(folder); rules = policy()
    c.require((folder / 'authority/infra_recovery.json').read_bytes() == (c.BASE / 'authority/infra_recovery.json').read_bytes(), 'Infrastructure authority differs')
    doc = c.read(folder / 'parent-reconciliation.json')
    c.require(doc['run_id'] == c.RUN_ID and doc['policy_sha256'] == c.file_sha(c.BASE / 'authority/infra_recovery.json') and len(doc['parents']) == len(rules['parents']), 'Parent receipt lineage')
    for receipt, parent in zip(doc['parents'], rules['parents']):
        for field in ('run_id', 'job_id', 'package_sha256', 'critical_files', 'result_archive_sha256', 'scheduler_state', 'scheduler_exit'):
            c.require(receipt[field] == parent[field], 'Closed parent receipt differs: ' + field)
        origin = c.ROOT / 'packages' / parent['run_id'] / parent['package_sha256'] / parent['package_name']
        c.require(receipt['package_origin'] == origin.as_posix() and receipt['package_manifest_origin'] == 'PACKAGE', 'Parent physical origin differs')
        c.require(receipt['zero_inference_confirmed'] is True and receipt['source_run_modified'] is False and receipt['queue_query_succeeded'] is True, 'Parent closure proof differs')
    pub = c.read(folder / 'publisher-installed.json')
    c.require(pub['run_id'] == c.RUN_ID and pub['helper_sha256'] == rules['installed_helper_sha256'] and pub['wrapper_sha256'] == rules['installed_wrapper_sha256'] and pub['proofs'] == rules['publisher_proofs'] and pub['repair_zip_sha256'] == rules['publisher_repair_zip_sha256'] and pub['verified_prior_terminal_commit'] == rules['verified_terminal_commit'], 'Publisher binding differs')
    pre = c.read(folder / 'infrastructure-preflight.json'); trust = c.read(folder / 'tls-trust.json')
    c.require(pre['status'] == 'PASS' and pre['execution_run_id'] == c.RUN_ID and pre['runtime_https_head_status'] == 200 and pre['probe_or_task_attempted'] is False, 'Pre-submission infrastructure gate')
    for field, name in [('parent_reconciliation_sha256', 'parent-reconciliation.json'), ('publisher_binding_sha256', 'publisher-installed.json'), ('tls_trust_sha256', 'tls-trust.json')]:
        c.require(pre[field] == c.file_sha(folder / name), 'Preflight lineage differs: ' + field)
    c.require(pre['ca_sha256'] == trust['ca_sha256'] and trust['verify_mode'] == 'CERT_REQUIRED' and trust['check_hostname'] is True, 'TLS verification gate')
    c.require(pre['model_manifest_sha256'] == c.file_sha(folder / 'model-manifest.json') == rules['frozen_model_manifest_sha256'], 'Frozen model manifest differs')
