"""Guarded inventory of a node-interrupted attempt; no model loading or writes to it."""
import importlib.util
import json
import os
import stat
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
NAME = 'rayan-paired-failure-inventory-20261021'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
WORKER_SHA = '1b803a1b1565c2602f33c9206863a95616903027a881a0bca53c092dafe6b47f'


def run(namespace):
    isolation = namespace._p2_guard()
    os.umask(0o077)
    code = namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'], ROOT, file=False)
    namespace._clean_code(code, '333fcd0c8239fc81a6459b13d53bbe592b33fda6')
    assert sha256(Path(namespace.__file__).read_bytes()).hexdigest() == GUARD
    worker = namespace._bounded_path(str(ROOT/'jobs/rayan-paired-update-comparison-20261020.py'), ROOT, file=True)
    assert sha256(worker.read_bytes()).hexdigest() == WORKER_SHA
    output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-paired-update-comparison-578661'), ROOT, file=False)
    permitted = {'construction-allocation.protected.json', 'token-cost-profiles.protected.json',
                 'parent-case-batch-binding.protected.json'}
    found = set(p.name for p in output.iterdir())
    # Exact early-stage inventory proves no persisted experiment/authorization
    # exists here; it never upgrades an unpublished optimizer state to a checkpoint.
    assert found == permitted
    artifacts = {}
    for name in sorted(permitted):
        path = namespace._bounded_path(str(output/name), ROOT, file=True)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        data = path.read_bytes()
        json.loads(data)
        artifacts[name] = {'bytes': len(data), 'sha256': sha256(data).hexdigest()}
    result = {'schema': 'alice-personality-node-failure-inventory-v1',
        'job_id': int(os.environ['SLURM_JOB_ID']), 'failed_job_id': 578661, 'owner': 'mxrayan',
        'isolation': isolation, 'namespace_sha256': GUARD, 'original_worker_sha256': WORKER_SHA,
        'worker_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'artifacts': artifacts,
        'only_preoptimization_artifacts_present': True, 'published_experimental_checkpoint_exists': False,
        'new_forward_passes': 0, 'new_gradient_steps': 0, 'private_payload_exported': False,
        'n0_approved': False, 'personality_approved': False}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result['receipt_sha256'] = sha256(canonical(result)).hexdigest()
    with (ROOT/'jobs'/(NAME+'.json')).open('xb') as stream:
        stream.write(canonical(result)+b'\n')
    print(json.dumps({'state': 'PREOPTIMIZATION_INVENTORY_VERIFIED', 'receipt_sha256': result['receipt_sha256']}), flush=True)


if __name__ == '__main__':
    path = Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
    assert sha256(path.read_bytes()).hexdigest() == GUARD
    spec = importlib.util.spec_from_file_location('paired_failure_guard', path)
    namespace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(namespace)
    try:
        run(namespace)
    except BaseException as error:
        print(json.dumps({'state': 'INVENTORY_REFUSED', 'exception_type': type(error).__name__}), flush=True)
        raise SystemExit(2)
