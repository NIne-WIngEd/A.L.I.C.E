"""Immutable ledger snapshots. The execution ID never doubles as a mutable ledger ID."""
from pathlib import Path
import os
import subprocess
import threading

import contract as c
import publisher_capture

STAGE = 'MC10D_QWEN_PUBLIC_CALIBRATION_V1'
PAYLOAD_NAMES = ('manifest.json', 'receipt.json', 'application.log')


def payload(run, receipt):
    run = Path(run)
    log = (run / 'application.log').read_bytes()
    descriptor = c.read(run / 'run.json')
    identity = {'receipt': receipt, 'log_sha256': c.sha(log), 'package_sha256': descriptor['package_sha256']}
    snapshot_id = c.RUN_ID + '-t-' + c.sha(c.canonical(identity).encode())[:16]
    folder = run / 'telemetry-events' / snapshot_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'rayan.compute.run-manifest.v1', 'run_id': snapshot_id, 'execution_run_id': c.RUN_ID, 'calibration_id': c.CALIBRATION_ID, 'stage': STAGE, 'provider': 'magnolia', 'slurm_job_id': receipt['slurm_job_id'], 'opaque_workload_sha256': descriptor['package_sha256'], 'approved_amendment_sha256': c.APPROVED_SHA, 'accelerator': 'none', 'snapshot_is_immutable': True}
    snapshot_receipt = {**receipt, 'run_id': snapshot_id, 'execution_run_id': c.RUN_ID, 'calibration_id': c.CALIBRATION_ID}
    c.immutable(folder / 'manifest.json', manifest)
    c.immutable(folder / 'receipt.json', snapshot_receipt)
    if (folder / 'application.log').exists():
        c.require((folder / 'application.log').read_bytes() == log, 'Immutable snapshot log differs')
    else:
        c.atomic_bytes(folder / 'application.log', log)
    item = {'schema': 'alice.mc10d.immutable-telemetry-snapshot.v1', 'snapshot_id': snapshot_id, 'execution_run_id': c.RUN_ID, 'status': receipt['status'], 'phase': receipt['phase'], 'slurm_job_id': receipt['slurm_job_id'], 'payload_sha256': {name: c.file_sha(folder / name) for name in PAYLOAD_NAMES}}
    c.immutable(folder / 'snapshot.json', item)
    return folder


def publish_stored(run, folder):
    run, folder = Path(run), Path(folder)
    item = c.read(folder / 'snapshot.json')
    c.require(folder.resolve().is_relative_to((run / 'telemetry-events').resolve()) and folder.name == item['snapshot_id'] and item['execution_run_id'] == c.RUN_ID, 'Snapshot location or identity differs')
    for name in PAYLOAD_NAMES:
        c.require(c.file_sha(folder / name) == item['payload_sha256'][name], 'Snapshot bytes changed before retry')
    code, markers = publisher_capture.invoke(run, item, folder)
    publication = {**item, 'exit_code': code, 'observed_at': c.now(), 'diagnostic_markers': markers, 'raw_model_responses_published': False, 'snapshot_path': 'telemetry-events/' + item['snapshot_id'], 'ledger_url': 'https://github.com/NIne-WIngEd/Rayan-Compute-Ledger/tree/main/runs/' + item['snapshot_id']}
    c.write(run / 'telemetry-publication.json', publication)
    with (run / 'telemetry-events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(c.canonical(publication))
    # Export only the latest snapshot's three payloads; older snapshots remain on
    # Magnolia and in the immutable ledger. This keeps the evidence bundle bounded.
    for name in PAYLOAD_NAMES:
        c.atomic_bytes(run / 'telemetry-latest' / name, (folder / name).read_bytes())
    return code


def publish(run, receipt):
    run = Path(run)
    c.write(run / 'telemetry-receipt.json', receipt)
    with (run / 'application.log').open('a', encoding='utf-8') as stream:
        stream.write(c.canonical(receipt))
    folder = payload(run, receipt)
    code = publish_stored(run, folder)
    publication = c.read(run / 'telemetry-publication.json')
    c.write(run / 'progress.json', {**receipt, 'telemetry_snapshot_id': publication['snapshot_id'], 'telemetry_url': publication['ledger_url'], 'telemetry_exit_code': code})
    return code


def terminal_recovery(run, state):
    run = Path(run)
    publication = c.read(run / 'telemetry-publication.json') if (run / 'telemetry-publication.json').exists() else {}
    if publication.get('status') in ('COMPLETED', 'FAILED'):
        code = publication['exit_code']
        if code != 0:
            code = publish_stored(run, run / publication['snapshot_path'])
    else:
        result = c.read(run / 'result.json') if (run / 'result.json').exists() else {}
        finished = c.read(run / 'worker-finished.json') if (run / 'worker-finished.json').exists() else {}
        passed = bool(result) and result.get('qualification_passed') is True and finished.get('result_sha256') == c.file_sha(run / 'result.json')
        records = [c.read(p) for p in run.glob('tasks/*/record.json')]
        receipt = {'schema': 'rayan.compute.run-receipt.v1', 'run_id': c.RUN_ID, 'stage': STAGE, 'status': 'COMPLETED' if passed else 'FAILED', 'provider': 'magnolia', 'slurm_job_id': state['job_id'], 'phase': 'TERMINAL_COLLECTION', 'tasks_completed': sum(r.get('status') == 'COMPLETE' for r in records), 'tasks_attempted': len(list(run.glob('tasks/*/attempt.json'))), 'tasks_required': 16, 'heartbeat_is_task_completion': False, 'scheduler_state': state['scheduler_state'], 'observed_at': c.now()}
        code = publish(run, receipt)
    publication = c.read(run / 'telemetry-publication.json')
    c.write(run / 'telemetry-recovery.json', {'exit_code': code, 'scheduler_terminal_verified': True, 'job_id': state['job_id'], 'result_sha256': c.file_sha(run / 'result.json') if (run / 'result.json').exists() else None, 'snapshot_id': publication['snapshot_id'], 'raw_model_responses_published': False, 'observed_at': c.now()})


def verify_evidence(folder):
    folder = Path(folder)
    publisher_capture.verify_evidence(folder)
    if not (folder / 'telemetry-publication.json').exists():
        return
    publication = c.read(folder / 'telemetry-publication.json')
    c.require(publication['execution_run_id'] == c.RUN_ID and publication['snapshot_id'].startswith(c.RUN_ID + '-t-'), 'Telemetry snapshot execution differs')
    for name in PAYLOAD_NAMES:
        c.require(c.file_sha(folder / 'telemetry-latest' / name) == publication['payload_sha256'][name], 'Telemetry snapshot payload hash differs')
    receipt = c.read(folder / 'telemetry-latest/receipt.json')
    c.require(receipt['run_id'] == publication['snapshot_id'] and receipt['execution_run_id'] == c.RUN_ID and receipt['status'] == publication['status'] and receipt['slurm_job_id'] == publication['slurm_job_id'], 'Telemetry snapshot receipt differs')
    c.require({k: v for k, v in receipt.items() if k not in ('run_id', 'execution_run_id', 'calibration_id')} == {k: v for k, v in c.read(folder / 'telemetry-receipt.json').items() if k != 'run_id'}, 'Telemetry snapshot differs from logical receipt')


class Telemetry:
    def __init__(self, run, package_sha):
        self.run = run
        self.helper = c.ROOT / 'bin/rayan-telemetry-push.sh'
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.phase = 'PREFLIGHT'
        self.completed = self.attempted = 0
        self.thread = None
        self.last_rc = None
        self.job = os.environ.get('SLURM_JOB_ID', '')
        c.require(self.job.isdigit(), 'Slurm job identity missing')

    def push(self, status='RUNNING'):
        with self.lock:
            receipt = {'schema': 'rayan.compute.run-receipt.v1', 'run_id': c.RUN_ID, 'stage': STAGE, 'status': status, 'provider': 'magnolia', 'slurm_job_id': self.job, 'phase': self.phase, 'tasks_completed': self.completed, 'tasks_attempted': self.attempted, 'tasks_required': 16, 'heartbeat_is_task_completion': False, 'observed_at': c.now()}
            self.last_rc = publish(self.run, receipt)
            return self.last_rc

    def start(self):
        c.require(self.helper.is_file() and os.access(self.helper, os.X_OK), 'Magnolia telemetry helper missing')
        c.require(self.push() == 0, 'Initial immutable GitHub telemetry failed; inference not started')
        def heartbeat():
            while not self.done.wait(300):
                self.push()
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()

    def stop(self, status):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=95)
        return self.push(status)


def journal_evidence(run):
    return publisher_capture.evidence_paths(run)
