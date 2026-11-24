"""Public tensor storage probe on the actual owner filesystem and Linux runtime."""
import importlib.util
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
NAME = 'rayan-progress-storage-probe-20261021'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
HELPER = 'bb7788a8c4243b14890c707eba4234e9d47c6c08495e3f7dded96fd0ec411fcb'
ORIGINAL = 'd3d9b52b7abbd702fd55fd39ab2c0e72c19b122d500f39953c8f5cfcba042bd2'


def load(name, path, expected):
    assert sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(namespace):
    isolation = namespace._p2_guard()
    os.umask(0o077)
    code = Path(os.environ['PERSONALITY_REPO_ROOT'])
    namespace._clean_code(code, '333fcd0c8239fc81a6459b13d53bbe592b33fda6')
    sys.path.insert(0, str(code))
    load('scripts.eipm.gemma_n0.paired_learning_experiment',
         ROOT/'jobs/rayan-paired-update-comparison-20261020-helper.py', ORIGINAL)
    helper = load('owner_progress_storage_probe_helper', ROOT/'jobs'/(NAME+'-helper.py'), HELPER)
    import torch
    assert sys.platform == 'linux' and torch.__version__ == '2.7.1+cu118'
    directory = ROOT/'jobs'/(NAME+'-'+os.environ['SLURM_JOB_ID'])
    directory.mkdir(mode=0o700, exist_ok=False)
    fixture = {'tensor': torch.tensor([1., 2.]), 'optimizer_keys': {0: {'step': torch.tensor(1.)}}}
    path = directory/'public-fixture.pt'
    helper._write(path, fixture, tensor=True)
    digest = helper._hash(path)
    helper._write(directory/'manifest.json', {'state_file_sha256': digest, 'public_fixture': True})
    helper._sync_directory(directory)
    helper._sync_directory(directory.parent)
    assert helper._hash(path) == digest
    restored = torch.load(path, weights_only=True)
    assert torch.equal(restored['tensor'], fixture['tensor']) and type(next(iter(restored['optimizer_keys']))) is int
    assert torch.equal(restored['optimizer_keys'][0]['step'], fixture['optimizer_keys'][0]['step'])
    result = {'schema': 'alice-personality-public-progress-storage-probe-v1',
              'job_id': int(os.environ['SLURM_JOB_ID']), 'owner': 'mxrayan', 'isolation': isolation,
              'namespace_sha256': GUARD, 'helper_sha256': HELPER,
              'worker_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'python': list(sys.version_info[:3]), 'torch': str(torch.__version__),
              'file_fsync_passed': True, 'directory_fsync_passed': True,
              'tensor_and_integer_key_readback_passed': True, 'private_inputs_accessed': False,
              'new_private_gradient_steps': 0, 'n0_approved': False, 'personality_approved': False}
    result['receipt_sha256'] = sha256(helper.canonical(result)).hexdigest()
    helper._write(ROOT/'jobs'/(NAME+'.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    namespace = load('owner_probe_namespace', Path(os.environ['PERSONALITY_REPO_ROOT'])/
                     'scripts/eipm/n1/enter_private_network_namespace.py', GUARD)
    run(namespace)
