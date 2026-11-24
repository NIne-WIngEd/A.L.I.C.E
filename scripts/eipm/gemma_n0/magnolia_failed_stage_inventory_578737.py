"""Read-only guarded stage inventory; no inference, training or private payload export."""
import importlib.util
import json
import os
import stat
from hashlib import sha256
from pathlib import Path

ROOT=Path('/homes/01/mxrayan/rayan-compute')
NAME='rayan-failed-stage-inventory-578737'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
WORKER='8494f6f9717051596eb04f47ef68d18ae55ce5beaf4fd2916d1f69a7e654e69f'
FILES=('construction-allocation.protected.json','token-cost-profiles.protected.json',
       'parent-case-batch-binding.protected.json','source-pair-bindings.protected.json',
       'experiment-authorization.protected.json','paired-group-plan.protected.json')


def run(namespace):
    isolation=namespace._p2_guard()
    os.umask(0o077)
    code=namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'],ROOT,file=False)
    namespace._clean_code(code,'333fcd0c8239fc81a6459b13d53bbe592b33fda6')
    assert sha256(Path(namespace.__file__).read_bytes()).hexdigest()==GUARD
    assert sha256((ROOT/'jobs/rayan-paired-update-recovery-20261021.py').read_bytes()).hexdigest()==WORKER
    output=namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-paired-update-recovery-578737'),ROOT,file=False)
    names={p.name for p in output.iterdir()}
    assert names <= set(FILES)|{'experimental-progress','experimental-arms'}
    files={}
    for name in FILES:
        if name not in names:
            files[name]={'exists':False}
            continue
        path=namespace._bounded_path(str(output/name),ROOT,file=True)
        info=path.stat()
        assert stat.S_IMODE(info.st_mode)==0o600 and info.st_size <= 64*1024**2
        data=path.read_bytes()
        json.loads(data)
        files[name]={'exists':True,'bytes':len(data),'sha256':sha256(data).hexdigest()}
    result={'schema':'alice-personality-failed-stage-inventory-v1','job_id':int(os.environ['SLURM_JOB_ID']),
            'failed_job_id':578737,'owner':'mxrayan','isolation':isolation,'namespace_sha256':GUARD,
            'original_worker_sha256':WORKER,'worker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'artifacts':files,'progress_directory_exists':'experimental-progress' in names,
            'final_experimental_directory_exists':'experimental-arms' in names,
            'new_forward_passes':0,'new_gradient_steps':0,'private_payload_exported':False,
            'n0_approved':False,'personality_approved':False}
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result['receipt_sha256']=sha256(canonical(result)).hexdigest()
    with (ROOT/'jobs'/(NAME+'.json')).open('xb') as stream:stream.write(canonical(result)+b'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    path=Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
    assert sha256(path.read_bytes()).hexdigest()==GUARD
    spec=importlib.util.spec_from_file_location('failed_stage_namespace',path)
    namespace=importlib.util.module_from_spec(spec);spec.loader.exec_module(namespace)
    try:run(namespace)
    except BaseException as error:
        import traceback
        print(json.dumps({'state':'INVENTORY_REFUSED','exception_type':type(error).__name__,
              'errno':error.errno if isinstance(error,OSError) else None,
              'frames':[{'module':Path(f.filename).name,'line':f.lineno,'function':f.name}
                        for f in traceback.extract_tb(error.__traceback__)]}),flush=True)
        raise SystemExit(2)
