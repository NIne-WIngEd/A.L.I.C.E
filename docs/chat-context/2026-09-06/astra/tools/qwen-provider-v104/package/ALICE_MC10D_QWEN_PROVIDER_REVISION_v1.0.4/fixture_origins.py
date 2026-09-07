"""Offline-only fixtures. Real package/run origins stay separate from exports."""
import importlib.util
from pathlib import Path
import sys
from unittest import mock
import zipfile

import contract as c
import infra_recovery
import package_revision as revision


def load_frozen_remote(package):
    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    contract = load('frozen_fixture_contract', package / 'contract.py')
    with mock.patch.dict(sys.modules, {'contract': contract}):
        remote = load('frozen_fixture_remote', package / 'remote_agent.py')
    return remote


def install_sources(root):
    root = Path(root); rules = infra_recovery.policy(); rev = revision.policy()
    packages = {}
    for key, name, run_id, sha in (
        ('a1', rules['source_package_name'], c.CALIBRATION_ID, rules['source_workload_sha256']),
        ('a2', rev['old_package_name'], c.RUN_ID, rev['old_package_sha256'])):
        archive = c.BASE / 'source' / (name + '.zip')
        c.require(c.file_sha(archive) == sha, 'Frozen fixture archive changed')
        target = root / 'packages' / run_id / sha
        target.mkdir(parents=True, exist_ok=True)
        (target.with_suffix('.zip')).write_bytes(archive.read_bytes())
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        packages[key] = target / name
    source = root / 'runs' / c.CALIBRATION_ID
    source.mkdir(parents=True, exist_ok=True)
    original = load_frozen_remote(packages['a1'])
    # The actual original producer, not a duplicated filename allowlist, chooses
    # which members originate in its package rather than its live run.
    projections = original.evidence_paths(source)
    with zipfile.ZipFile(c.BASE / 'source/ALICE_QWEN_PUBLIC_RESULT_c926d9e355dd.zip') as z:
        for name in z.namelist():
            raw = z.read(name)
            if name in projections:
                c.require(projections[name].read_bytes() == raw, 'Frozen package projection differs')
            else:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    c.require(not (source / 'package-manifest.json').exists(), 'Fixture recreated the v102 layout defect')
    return source, packages, original


def prior_a2(run):
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    rules = revision.policy()
    c.write(run / 'run.json', revision.descriptor(rules['old_package_sha256']))
    c.write(run / 'infrastructure-preflight-failure.json', {
        'status': 'STOP', 'class': 'Stop',
        'message': 'Source failure evidence differs: package-manifest.json',
        'job_submitted': False, 'observed_at': '2026-09-06T23:41:21Z'})
    c.require(c.file_sha(run / 'run.json') == rules['old_descriptor_sha256'], 'Captured a2 descriptor fixture hash')
    c.require(c.file_sha(run / 'infrastructure-preflight-failure.json') == rules['old_preflight_failure_sha256'],
              'Captured a2 failure fixture hash')


def completed_revision(run, new_sha):
    """Real revision writer; upstream source/scheduler gates mocked for worker-unit scope.

Full physical source and scheduler boundaries are tested in selftest_revision.
"""
    prior_a2(run)
    with mock.patch.object(infra_recovery, 'reconcile_source'), \
         mock.patch.object(revision.evidence_origins, 'verify_package_origin'):
        return revision.apply_remote(run, new_sha, mock.Mock(), lambda: None)


def local_prepared(vault):
    """Exact owner-returned local state and unfinished v103 intent, on temporary paths."""
    import base64, json
    path = Path(vault) / 'tools/alice-astra/qwen-fallback-a2/controller-state.json'
    rules = dict(revision.policy())
    with zipfile.ZipFile(c.BASE / 'source/ALICE_MAGNOLIA_PROVIDER_TRACE_15f4edccafa3.zip') as z:
        files = json.loads(z.read('report.json'))['local_inventory_after']['files']
    prefix = 'tools/alice-astra/qwen-fallback-a2/'
    for rel in ('controller-state.json', 'revisions/' + rules['prior_uncommitted_revision_id'] + '/controller-state-v102.json',
                'revisions/' + rules['prior_uncommitted_revision_id'] + '/intent.json'):
        target = path.parent / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(files[prefix + rel]['base64']))
    return path, rules
