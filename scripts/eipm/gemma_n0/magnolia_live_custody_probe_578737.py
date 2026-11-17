"""Actual Linux/P2 failure reproduction and native public grouped-update control."""
import errno
import importlib.util
import json
import os
from pathlib import Path
from hashlib import sha256
import subprocess
import sys

ROOT=Path('/homes/01/mxrayan/rayan-compute')
NAME='rayan-live-custody-probe-578737'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
LIVE='7bde912edf49ffd1227f47d2bbf77068d6c0d79c7f65159b9e49a36522505533'
RECOVERY='bb7788a8c4243b14890c707eba4234e9d47c6c08495e3f7dded96fd0ec411fcb'
ORIGINAL='d3d9b52b7abbd702fd55fd39ab2c0e72c19b122d500f39953c8f5cfcba042bd2'
FIXTURE='b1345db81422b5869126a96f0edde7b0379d984035e4f7b70ce263a0c432aaaf'


def load(name,path,digest):
    assert sha256(path.read_bytes()).hexdigest()==digest
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def negative(namespace):
    namespace._p2_guard()
    # A public file, opened AFTER startup cleanup, models a legitimately owned pin.
    fd=os.open(__file__,os.O_RDONLY)
    os.fstat(fd)
    namespace._p2_guard()
    try:os.fstat(fd)
    except OSError as error:
        assert error.errno==errno.EBADF
        print(json.dumps({'legacy_recheck_invalidates_owned_descriptor':True,'errno':error.errno}))
        return
    raise AssertionError('expected original descriptor-lifecycle failure was not reproduced')


def run(namespace):
    isolation=namespace._p2_guard()
    os.umask(0o077)
    code=Path(os.environ['PERSONALITY_REPO_ROOT'])
    namespace._clean_code(code,'333fcd0c8239fc81a6459b13d53bbe592b33fda6')
    result=subprocess.run([sys.executable,'-I',__file__,'--negative'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert result.returncode==0 and not result.stderr
    original_failure=json.loads(result.stdout)
    assert original_failure=={'legacy_recheck_invalidates_owned_descriptor':True,'errno':9}
    sys.path.insert(0,str(code))
    live=load('public_live_custody',ROOT/'jobs'/(NAME+'-live.py'),LIVE)
    load('scripts.eipm.gemma_n0.paired_learning_experiment',ROOT/'jobs/rayan-paired-update-comparison-20261020-helper.py',ORIGINAL)
    recovery=load('public_custody_grouped_recovery',ROOT/'jobs/rayan-paired-update-recovery-20261021-helper.py',RECOVERY)
    fixture=load('public_custody_actual_native_fixture',ROOT/'jobs'/(NAME+'-fixture.py'),FIXTURE)
    import torch
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    assert torch.__version__=='2.7.1+cu118'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    value=fixture.PairedLearningExperimentTests();value.setUp()
    directory=ROOT/'jobs'/(NAME+'-'+os.environ['SLURM_JOB_ID'])
    directory.mkdir(mode=0o700,exist_ok=False)
    checks=0
    with _PinnedFile(Path(__file__),sha256(Path(__file__).read_bytes()).hexdigest(),128*1024) as pinned:
        def guard():
            nonlocal checks
            live.verify_active_namespace(namespace,isolation)
            os.fstat(pinned.stream.fileno())
            pinned.verify()
            checks+=1
        value.guard=guard
        reference=value.run_trial()
        candidate=recovery.run_experiment(value.model,value.objectives,value.recipe,
            plan=value.plan,expected_plan_sha256=recovery.plan_sha256(value.plan),
            batch_loader=value.batches.__getitem__,validate_batch=value.validate,custody_guard=guard,
            progress_directory=directory/'groups',destination=directory/'finished')
        def exact(a,b):
            assert type(a) is type(b)
            if isinstance(a,torch.Tensor):assert torch.equal(a,b)
            elif isinstance(a,dict):
                assert list(a)==list(b)
                for key in a:exact(a[key],b[key])
            elif isinstance(a,(tuple,list)):
                assert len(a)==len(b)
                for x,y in zip(a,b):exact(x,y)
            else:assert a==b
        exact(reference['arms'],candidate['arms'])
        assert candidate['total_committed_updates']==candidate['new_updates_this_invocation']==4
        guard()
        original_pin=os.environ['PERSONALITY_ISOLATED_NETNS']
        os.environ['PERSONALITY_ISOLATED_NETNS']='net:[0]'
        try:
            try:live.verify_active_namespace(namespace,isolation)
            except ValueError:pass
            else:raise AssertionError('changed namespace binding was accepted')
        finally:os.environ['PERSONALITY_ISOLATED_NETNS']=original_pin
        guard()
    result={'schema':'alice-personality-live-custody-actual-runtime-probe-v1','job_id':int(os.environ['SLURM_JOB_ID']),
        'owner':'mxrayan','isolation':isolation,'namespace_sha256':GUARD,'live_helper_sha256':LIVE,
        'recovery_helper_sha256':RECOVERY,'fixture_sha256':FIXTURE,'worker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_failure':original_failure,'live_namespace_checks':checks,'owned_descriptor_survived':True,
        'pin_exit_reverification_passed':True,'native_two_arm_state_optimizer_rng_events_exact':True,
        'namespace_drift_refused':True,'public_native_updates_per_trial':4,'private_inputs_accessed':False,
        'new_private_gradient_steps':0,'n0_approved':False,'personality_approved':False}
    result['receipt_sha256']=sha256(recovery.canonical(result)).hexdigest()
    recovery._write(ROOT/'jobs'/(NAME+'.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    namespace=load('custody_probe_namespace',Path(os.environ['PERSONALITY_REPO_ROOT'])/
                   'scripts/eipm/n1/enter_private_network_namespace.py',GUARD)
    if sys.argv[1:]==['--negative']:negative(namespace)
    else:run(namespace)
