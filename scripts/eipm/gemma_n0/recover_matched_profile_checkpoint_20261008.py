"""Read-only fixed metadata inventory, only after the actual owner P2 guard."""
import importlib.util
import json
import os
from pathlib import Path
from hashlib import sha256
import stat

ROOT=Path('/homes/01/mxrayan/rayan-compute')
REV='333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'

def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def small_json(path):
    before=path.stat()
    if path.is_symlink() or not path.is_file() or before.st_size>32*1024**2:
        raise ValueError('bounded protected metadata required')
    payload=path.read_bytes();after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('metadata changed during recovery')
    return json.loads(payload),sha256(payload).hexdigest()

def file_hash(path):
    assert not path.is_symlink() and path.is_file()
    before=path.stat()
    assert stat.S_IMODE(before.st_mode)==0o600 and before.st_size < 8*1024**3
    digest=sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
    after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return digest.hexdigest(),after.st_size

def inventory(namespace):
    isolation=namespace._p2_guard()  # FIRST; private paths only below.
    os.umask(0o077)
    namespace._clean_code(Path(os.environ['PERSONALITY_REPO_ROOT']),REV)
    assert sha256(Path(namespace.__file__).read_bytes()).hexdigest()==GUARD
    result={'schema':'alice-personality-matched-profile-checkpoint-inventory-v1',
        'original_job':577088,'inventory_job':int(os.environ['SLURM_JOB_ID']),
        'owner':'mxrayan','scientific_revision':REV,'isolation':isolation,
        'inventory_only':True,'private_payload_exported':False,'n0_approved':False,
        'full_construction_complete':False,'qualification':'UNQUALIFIED','arms':{}}
    for threads in (1,16):
        old=namespace._bounded_path(str(ROOT/'rayan-personality/private-compile'/f'rayan-timeout-profile-577088-threads-{threads:02d}'),ROOT,file=False)
        assert stat.S_IMODE(old.stat().st_mode)==0o700
        row={'intraop_threads':threads,'root_present':True,
            'root_entry_count':len(list(old.iterdir())),'checkpoint_present':False,
            'unpublished_optimizer_progress':'UNKNOWN'}
        path=old/'one-step-checkpoint'
        if path.exists():
            namespace._bounded_path(str(path),ROOT,file=False)
            assert stat.S_IMODE(path.stat().st_mode)==0o700
            assert {p.name for p in path.iterdir()}=={'manifest.json','resume.pt'}
            obj,digest=small_json(path/'manifest.json')
            assert stat.S_IMODE((path/'manifest.json').stat().st_mode)==0o600
            body={k:v for k,v in obj.items() if k!='manifest_sha256'}
            assert sha256(canonical(body)).hexdigest()==obj['manifest_sha256']
            payload,payload_bytes=file_hash(path/'resume.pt')
            assert payload==obj['resume_sha256']
            assert obj['phase']=='n1' and obj['qualification']=='UNQUALIFIED' and obj['acceptance_authority'] is False
            if threads==1:
                assert obj['manifest_sha256']=='d1f30bed999746fd6c04c1e317e54274403cd7f22224dcd171670e1c48bf1cff'
                assert (obj['step'],obj['epoch'],obj['batch_index'])==(1,0,1)
            row.update(checkpoint_present=True,manifest_file_sha256=digest,
                manifest_sha256=obj['manifest_sha256'],resume_sha256=payload,
                payload_bytes=payload_bytes,step=obj['step'],epoch=obj['epoch'],batch_index=obj['batch_index'],
                schema=obj['schema'],model_state_sha256=obj['model_state_sha256'],payload_hash_verified=True)
        result['arms'][str(threads)]=row
    assert result['arms']['1']['checkpoint_present']
    result['receipt_sha256']=sha256(canonical(result)).hexdigest()
    dest=namespace._bounded_path(str(ROOT/'jobs'),ROOT,file=False)/'rayan-profile-checkpoint-inventory-20261008.json'
    with dest.open('xb') as stream:stream.write(canonical(result)+b'\n')
    print(json.dumps(result,sort_keys=True),flush=True)

code=Path(os.environ['PERSONALITY_REPO_ROOT'])
guard=code/'scripts/eipm/n1/enter_private_network_namespace.py'
assert sha256(guard.read_bytes()).hexdigest()==GUARD
spec=importlib.util.spec_from_file_location('owner_namespace_inventory',guard)
namespace=importlib.util.module_from_spec(spec);spec.loader.exec_module(namespace)
try:
    inventory(namespace)
except Exception as exc:
    print(json.dumps({'state':'INVENTORY_REFUSED','exception_type':type(exc).__name__}),flush=True)
    raise SystemExit(2)
