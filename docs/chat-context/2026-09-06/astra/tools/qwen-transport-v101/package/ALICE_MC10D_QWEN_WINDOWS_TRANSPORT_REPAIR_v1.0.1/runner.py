"""Client-only remote-path repair. The approved v100 workload bytes remain frozen."""
from __future__ import annotations

import argparse
import compileall
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

BASE=Path(__file__).resolve().parent
WORKLOAD_NAME='ALICE_MC10D_QWEN_PUBLIC_QUALIFICATION_v1.0.0'
WORKLOAD_SHA='bc7c76c29a4f0fcfbd4567a610737ec5bb2052e7165677562959ac4dca7c959e'
WORKLOAD_MANIFEST_SHA='b1d94af265eaac40d29314481358aa0f2a37b072bfa8c5542409fd641387175c'
RUN_ID='alice-qwen38-a1-8e7a384a745496f4'
REMOTE_ROOT='/homes/01/mxrayan/rayan-compute'
REMOTE_RUN=REMOTE_ROOT+'/runs/'+RUN_ID
INPUT_LOG_SHA='78b19e9e4f53a977547b639b839ba9988628eade5d970984f7961cfa72fa24ff'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_workload(destination,archive=None):
    archive=archive or BASE/'workload'/(WORKLOAD_NAME+'.zip')
    require(digest(archive)==WORKLOAD_SHA,'frozen v100 workload ZIP hash changed')
    destination=Path(destination)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        require(len(names)==len(set(names))==17,'frozen workload ZIP membership')
        for info in z.infolist():
            path=PurePosixPath(info.filename)
            require(not path.is_absolute() and path.parts[0]==WORKLOAD_NAME and '..' not in path.parts and '\\' not in info.filename,'workload ZIP path')
            require((info.external_attr>>16)&0o170000 != 0o120000,'workload ZIP symlink')
        if not destination.exists():
            destination.mkdir(parents=True)
            z.extractall(destination)
    root=destination/WORKLOAD_NAME
    require(digest(root/'PACKAGE_MANIFEST.json')==WORKLOAD_MANIFEST_SHA,'frozen workload manifest changed')
    manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_bytes())
    names=set()
    for item in manifest['files']:
        p=root/item['path']
        require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink(),'workload member location')
        require(item['path'] not in names,'duplicate workload manifest entry')
        names.add(item['path'])
        require(p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'frozen workload member changed: '+item['path'])
    observed={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(observed==names|{'PACKAGE_MANIFEST.json'},'unmanifested workload files')
    return root,Path(archive)


def load_workload(root):
    require(str(root) not in sys.path,'workload already loaded in this process')
    sys.path.insert(0,str(root))
    for name in ('contract','frozen_prompt','evidence','publish_context','controller'):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().is_relative_to(root.resolve()),'unrelated module already loaded: '+name)
    contract=importlib.import_module('contract')
    controller=importlib.import_module('controller')
    contract.verify_package();contract.authority()
    return contract,controller


def adapt_remote_paths(contract):
    """Only the client's two remote address values change representation, never disk files."""
    require(contract.RUN_ID==RUN_ID,'unexpected approved run identity')
    require(contract.ROOT.as_posix()==REMOTE_ROOT,'unexpected remote root; refusing normalization')
    require(contract.REMOTE_RUN.as_posix()==REMOTE_RUN,'unexpected remote run; refusing normalization')
    before={'root':str(contract.ROOT),'run':str(contract.REMOTE_RUN)}
    contract.ROOT=PurePosixPath(REMOTE_ROOT)
    contract.REMOTE_RUN=PurePosixPath(REMOTE_RUN)
    require(str(contract.ROOT).startswith('/') and '\\' not in str(contract.REMOTE_RUN),'remote POSIX address invariant')
    return {'schema':'alice.mc10d.qwen.windows-transport-repair.v1.0.1','run_id':RUN_ID,'workload_sha256':WORKLOAD_SHA,'workload_manifest_sha256':WORKLOAD_MANIFEST_SHA,'remote_paths_before':before,'remote_paths_after':{'root':str(contract.ROOT),'run':str(contract.REMOTE_RUN)},'disk_source_files_modified':False,'controller_state_identity_changed':False,'scientific_profile_changed':False,'reported_windows_v100_tests_passed':35,'source_terminal_sha256':INPUT_LOG_SHA}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vault-root',type=Path,default=Path(r'C:\ALICE_Vault'))
    p.add_argument('--repo-root',type=Path,default=Path(r'C:\A.L.I.C.E-main'))
    p.add_argument('--output-root',type=Path,default=Path.home()/'Downloads')
    p.add_argument('--ssh-key',type=Path,default=Path.home()/'.ssh/rayan_magnolia_ed25519')
    args=p.parse_args()
    try:
        root,archive=prepare_workload(BASE.parent/'verified-v100-workload')
        require(compileall.compile_dir(str(root),quiet=1),'frozen workload compile gate failed')
        contract,controller=load_workload(root)
        receipt=adapt_remote_paths(contract)
        contract.write(BASE.parent/'transport-repair-receipt.json',receipt)
        receipt_sha=contract.sha(contract.canonical(receipt).encode())
        receipt_path=args.vault_root/'tools/alice-astra/qwen-fallback-a1/runs'/RUN_ID/'client-transport'/('v101-'+receipt_sha[:16]+'.json')
        contract.immutable(receipt_path,receipt)
        print('transport_repair=WINDOWS_CLIENT_REMOTE_POSIX_PATHS',flush=True)
        print('frozen_workload_sha256='+WORKLOAD_SHA,flush=True)
        print('remote_root='+str(contract.ROOT),flush=True)
        print('existing_run_id_preserved='+RUN_ID,flush=True)
        print('client_transport_receipt='+str(receipt_path),flush=True)
        delegated=argparse.Namespace(package_zip=archive,package_sha=WORKLOAD_SHA,vault_root=args.vault_root,repo_root=args.repo_root,output_root=args.output_root,ssh_key=args.ssh_key)
        try:
            return controller.execute(delegated)
        except (controller.TransportPending,KeyboardInterrupt) as exc:
            print('QWEN_RUN_PENDING: '+str(exc)+'. Use this v101 launcher to resume the same run.',flush=True)
            return 74
    except Exception as exc:
        print('QWEN_TRANSPORT_REPAIR_STOP: '+type(exc).__name__+': '+str(exc)[:1600],flush=True)
        print('Keep the terminal transcript and existing controller state for review.',flush=True)
        return 76


if __name__=='__main__':raise SystemExit(main())
