"""Reproduce v102's live/archive confusion with unmodified released modules.

All activity is in a temporary local tree. No network, scheduler or inference.
The original collector decides which archive members come from the package.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
import zipfile


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(original, successor, archive):
    sys.path.insert(0, str(original))
    old_contract = load('contract', original / 'contract.py')
    old = load('original_remote_agent', original / 'remote_agent.py')
    sys.path.insert(0, str(successor))
    new_contract = load('contract', successor / 'contract.py')
    recovery = load('infra_recovery', successor / 'infra_recovery.py')
    assert digest(archive) == recovery.policy()['source_result_zip_sha256']
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live = root / 'live-run'; live.mkdir()
        # Ask the real collector for its package projections on an empty run.
        projected = old.evidence_paths(live)
        assert set(projected) == {
            'package-manifest.json', *('authority/' + n for n in (
                'tasks.json', 'draft.json', 'approval.json', 'approved.json',
                'runtime_policy.json', 'source_binding_policy.json'))}
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if entry.filename in projected:
                    assert z.read(entry) == projected[entry.filename].read_bytes()
                else:
                    target = live / entry.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(entry))
        before = {p.relative_to(live).as_posix(): digest(p)
                  for p in live.rglob('*') if p.is_file()}
        terminal = {'terminal': True, 'job_id': '575089', 'scheduler_state': 'FAILED'}
        # Scheduler and already-completed telemetry are outside this boundary.
        # The production evidence_paths, manifest writer and ZIP collector run.
        with mock.patch.object(old, 'status', return_value=terminal), \
             mock.patch.object(old, 'repair_terminal_telemetry'):
            receipt = old.collect(live)
        assert not (live / 'package-manifest.json').exists()
        exported = root / 'exported'; exported.mkdir()
        with zipfile.ZipFile(receipt['remote_path']) as z:
            assert z.read('package-manifest.json') == projected['package-manifest.json'].read_bytes()
            z.extractall(exported)
        recovery.verify_source_files(exported)
        try:
            recovery.verify_source_files(live)
        except new_contract.Stop as exc:
            error = str(exc)
        else:
            raise AssertionError('The unmodified v102 verifier unexpectedly accepted the live layout')
        assert error == 'Source failure evidence differs: package-manifest.json'
        # Each actual run-origin critical byte survives collection unchanged.
        checks = {name: digest(live / name) == expected for name, expected in
                  recovery.policy()['source_critical_files'].items()
                  if name not in projected}
        assert all(checks.values())
        assert all(digest(live / name) == value for name, value in before.items())
        # A changed package is a separate incompatibility, even before sbatch.
        run = root / 'prepared-a2'; run.mkdir()
        descriptor = {'run_id': new_contract.RUN_ID, 'package_sha256':
                      'f196f3b91604f5c35ca594df001c10a6c1cec87aa2962e544fadc3ba99fa9999'}
        new_contract.immutable(run / 'run.json', descriptor)
        try:
            new_contract.immutable(run / 'run.json', {**descriptor, 'package_sha256': 'f' * 64})
        except new_contract.Stop as exc:
            identity_error = str(exc)
        else:
            raise AssertionError('Existing run descriptor accepted a changed workload')
        return {
            'schema': 'alice.qwen.v102.collector-boundary-reproduction.v1',
            'environment': 'local simulation using unmodified released production collector and verifier',
            'actual_magnolia_filesystem_observed': False,
            'source_zip_sha256': digest(archive),
            'original_collector_sha256': digest(original / 'remote_agent.py'),
            'v102_verifier_sha256': digest(successor / 'infra_recovery.py'),
            'package_projection_members': sorted(projected),
            'eight_run_origin_critical_files_match': checks,
            'actual_collector_export_accepted_by_v102': True,
            'same_collector_live_layout_rejected_by_v102': error,
            'collector_creates_manifest_in_live_run': False,
            'preexisting_source_files_unchanged_by_collection': True,
            'changed_package_same_run_descriptor_error': identity_error,
            'scheduler_and_telemetry_mocked': True,
            'model_or_network_requests': 0,
        }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--successor', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = reproduce(a.original.resolve(), a.successor.resolve(), a.archive.resolve())
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
