"""An explicit pre-submission revision of a2, with immutable prior bytes.

The remote transition runs under the same submission lock as both launchers.
Its deterministic intent is written before the active descriptor changes. A
retry can finish that transition; it cannot create another execution identity.
"""
from pathlib import Path

import contract as c
import evidence_origins

FORBIDDEN = ('submission-intent.json', 'submission.json', 'sbatch-response.json',
             'job.sbatch', 'worker-started.json', 'worker-finished.json',
             'probe-attempt.json', 'probe-response.ndjson', 'runtime.json',
             'model-lock.json', 'model-manifest.json', 'effective-contract.json',
             'result.json')


def policy():
    value = c.read(c.BASE / 'authority/revision_policy.json')
    c.require(value['execution_run_id'] == c.RUN_ID, 'Revision run identity')
    return value


def descriptor(package_sha):
    return {'run_id': c.RUN_ID, 'package_sha256': c.full_digest(package_sha),
            'amendment_sha256': c.APPROVED_SHA,
            'approved_profile_id': 'qwen38_thinking_on_public_calibration_v1'}


def history(root):
    return Path(root) / 'revisions' / policy()['revision_id']


def exact(path, expected):
    p = Path(path)
    c.require(p.is_file() and not p.is_symlink(), 'Revision evidence missing or linked: ' + p.name)
    c.require(c.file_sha(p) == expected, 'Revision evidence hash differs: ' + p.name)
    return p.read_bytes()


def preserve(path, data):
    path = Path(path)
    if path.exists() or path.is_symlink():
        c.require(path.is_file() and not path.is_symlink() and path.read_bytes() == data,
                  'Preserved revision bytes differ: ' + path.name)
    else:
        c.atomic_bytes(path, data)


def prior_local_intent(state_path):
    rules = policy()
    folder = Path(state_path).parent / 'revisions' / rules['prior_uncommitted_revision_id']
    c.require(folder.is_dir() and not folder.is_symlink(), 'Prior uncommitted revision directory differs')
    c.require({p.name for p in folder.iterdir()} == {'controller-state-v102.json', 'intent.json'},
              'Prior local revision has an acknowledgement or unexpected evidence; review required')
    exact(folder / 'controller-state-v102.json', rules['old_local_state_sha256'])
    return exact(folder / 'intent.json', rules['prior_uncommitted_intent_sha256'])


def remote_revision_scope(run):
    root = Path(run) / 'revisions'
    c.require(not root.is_symlink(), 'Linked revision root')
    if root.exists():
        c.require(root.is_dir() and {p.name for p in root.iterdir()} <= {policy()['revision_id']},
                  'Another remote revision exists; do not supersede or erase it')


def intent(new_sha):
    rules = policy()
    c.full_digest(new_sha)
    c.require(new_sha != rules['old_package_sha256'], 'Revision requires distinct, explicit package bytes')
    return {'schema': 'alice.qwen.a2.package-revision-intent.v1',
            'run_id': c.RUN_ID, 'revision_id': rules['revision_id'],
            'old_package_sha256': rules['old_package_sha256'], 'new_package_sha256': new_sha,
            'old_descriptor_sha256': rules['old_descriptor_sha256'],
            'new_descriptor_sha256': c.sha(c.canonical(descriptor(new_sha)).encode()),
            'old_preflight_failure_sha256': rules['old_preflight_failure_sha256'],
            'old_local_state_sha256': rules['old_local_state_sha256'],
            'trace_zip_sha256': rules['trace_zip_sha256'],
            'trace_report_sha256': rules['trace_report_sha256'],
            'approved_amendment_sha256': c.APPROVED_SHA,
            'transition': 'PRE_SUBMISSION_PACKAGE_REVISION',
            'provider_trace_zip_sha256': rules['provider_trace_zip_sha256'],
            'prior_uncommitted_revision_id': rules['prior_uncommitted_revision_id'],
            'prior_uncommitted_intent_sha256': rules['prior_uncommitted_intent_sha256'],
            'prior_revision_disposition': rules['prior_revision_disposition'],
            'source_a1_modified': False, 'new_execution_created': False}


def receipt(new_sha):
    entry = intent(new_sha)
    return {'schema': 'alice.qwen.a2.package-revision-receipt.v1',
            **{k: entry[k] for k in ('run_id', 'revision_id', 'old_package_sha256', 'new_package_sha256',
                                    'old_descriptor_sha256', 'new_descriptor_sha256', 'trace_zip_sha256',
                                    'provider_trace_zip_sha256', 'prior_uncommitted_revision_id',
                                    'prior_uncommitted_intent_sha256', 'prior_revision_disposition')},
            'intent_sha256': c.sha(c.canonical(entry).encode()),
            'status': 'APPLIED_BEFORE_SUBMISSION', 'source_a1_modified': False,
            'new_execution_created': False}


def verify_current(run, new_sha):
    """Pure evidence validation, usable on the live run or its exported ZIP."""
    run = Path(run); folder = history(run); rules = policy()
    remote_revision_scope(run)
    exact(folder / 'run-v102.json', rules['old_descriptor_sha256'])
    exact(folder / 'preflight-failure-v102.json', rules['old_preflight_failure_sha256'])
    c.require(c.read(folder / 'intent.json') == intent(new_sha), 'Revision intent differs')
    c.require(c.read(run / 'package-revision.json') == receipt(new_sha), 'Revision completion receipt differs')
    c.require(c.read(run / 'run.json') == descriptor(new_sha), 'Active revision descriptor differs')
    return receipt(new_sha)


def require_unsubmitted(run, find_job):
    run = Path(run)
    for name in FORBIDDEN:
        c.require(not (run / name).exists() and not (run / name).is_symlink(),
                  'Cannot revise after execution boundary: ' + name)
    c.require(not list(run.glob('tasks/*/attempt.json')), 'Cannot revise after a task intent')
    c.require(find_job() is None, 'An a2 scheduler job exists; do not revise or resubmit')


def apply_remote(run, new_sha, command, find_job):
    """Caller must hold remote_agent.submission_lock throughout this function."""
    import infra_recovery
    run = Path(run); rules = policy(); folder = history(run)
    remote_revision_scope(run)
    completed = run / 'package-revision.json'
    if completed.exists():
        # Attach/collect after a completed revision is valid even after submission.
        return verify_current(run, new_sha)
    current = c.read(run / 'run.json')
    old = descriptor(rules['old_package_sha256'])
    c.require(current in (old, descriptor(new_sha)), 'Unexpected a2 descriptor; no migration inferred')
    # A new descriptor with no committed receipt is only a recoverable interrupted
    # revision if its exact intent and original archives already exist.
    if current != old:
        c.require(c.read(folder / 'intent.json') == intent(new_sha), 'Changed descriptor lacks exact revision intent')
        exact(folder / 'run-v102.json', rules['old_descriptor_sha256'])
        exact(folder / 'preflight-failure-v102.json', rules['old_preflight_failure_sha256'])
    require_unsubmitted(run, find_job)
    # Recheck the real source/package origins and source scheduler closure now;
    # the uploaded observation is evidence, not a substitute for this live gate.
    infra_recovery.reconcile_source(command)
    old_package = c.ROOT / 'packages' / c.RUN_ID / rules['old_package_sha256'] / rules['old_package_name']
    evidence_origins.verify_package_origin(old_package, rules['old_package_manifest_sha256'], rules['old_package_sha256'])
    if current == old:
        before = exact(run / 'run.json', rules['old_descriptor_sha256'])
        failure = exact(run / 'infrastructure-preflight-failure.json', rules['old_preflight_failure_sha256'])
        preserve(folder / 'run-v102.json', before)
        preserve(folder / 'preflight-failure-v102.json', failure)
        c.immutable(folder / 'intent.json', intent(new_sha))
        # A lock plus exact old bytes make this a declared pre-submission pointer
        # transition. Prior evidence stays immutable in the revision archive.
        exact(run / 'run.json', rules['old_descriptor_sha256'])
        c.atomic_bytes(run / 'run.json', c.canonical(descriptor(new_sha)).encode())
    c.immutable(completed, receipt(new_sha))
    return verify_current(run, new_sha)


def begin_local(state_path, new_sha):
    """Archive the exact observed local state before any remote mutation."""
    state_path = Path(state_path); rules = policy()
    prior_local_intent(state_path)
    data = exact(state_path, rules['old_local_state_sha256'])
    state = c.strict(data)
    c.require(state['run_id'] == c.RUN_ID and state['package_sha256'] == rules['old_package_sha256']
              and state['amendment_sha256'] == c.APPROVED_SHA and state['phase'] == 'PREPARED'
              and state['staged'] is True and 'job_id' not in state, 'Local a2 state is not the observed preparation')
    folder = history(state_path.parent)
    preserve(folder / 'controller-state-v102.json', data)
    c.immutable(folder / 'intent.json', intent(new_sha))
    return state


def finish_local(state_path, new_sha, remote_receipt):
    state_path = Path(state_path)
    c.require(remote_receipt == receipt(new_sha), 'Remote revision acknowledgement differs')
    state = begin_local(state_path, new_sha)
    folder = history(state_path.parent)
    c.immutable(folder / 'remote-receipt.json', remote_receipt)
    state.update(package_sha256=new_sha, phase='PREPARED', staged=True,
                 superseded_package_sha256=policy()['old_package_sha256'],
                 package_revision_id=policy()['revision_id'],
                 package_revision_receipt_sha256=c.sha(c.canonical(remote_receipt).encode()),
                 updated_at=c.now())
    c.write(state_path, state)
    return verify_local(state_path, new_sha)


def verify_local(state_path, new_sha):
    state_path = Path(state_path); folder = history(state_path.parent); rules = policy()
    prior_local_intent(state_path)
    exact(folder / 'controller-state-v102.json', rules['old_local_state_sha256'])
    c.require(c.read(folder / 'intent.json') == intent(new_sha), 'Local revision intent differs')
    remote = c.read(folder / 'remote-receipt.json')
    c.require(remote == receipt(new_sha), 'Local saved remote revision differs')
    state = c.read(state_path)
    c.require(state['run_id'] == c.RUN_ID and state['package_sha256'] == new_sha
              and state['amendment_sha256'] == c.APPROVED_SHA
              and state['package_revision_id'] == rules['revision_id']
              and state['package_revision_receipt_sha256'] == c.sha(c.canonical(remote).encode()),
              'Local active revision identity differs')
    return state


def evidence_paths(run):
    run = Path(run)
    prefix = 'revisions/' + policy()['revision_id'] + '/'
    names = ['package-revision.json', *(prefix + name for name in
             ('run-v102.json', 'preflight-failure-v102.json', 'intent.json'))]
    return {name: run / name for name in names}


def verify_evidence(folder, package_sha):
    folder = Path(folder)
    c.require((folder / 'authority/revision_policy.json').read_bytes() ==
              (c.BASE / 'authority/revision_policy.json').read_bytes(), 'Returned revision authority differs')
    return verify_current(folder, package_sha)
