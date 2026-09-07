"""Windows entrypoint: prepare once, attach on rerun, collect and verify before publishing."""
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid

import contract as c
import evidence
import publish_context
import package_revision

HOST='mxrayan@magnolia.usm.edu'
R13_SHA='490dc2983a5d2d1f605b0fa8799f75fcd2e2e93a6dd8a0c9a6a7824332fe56e1'
R18_SHA='400113e93b855c4ba65e2d2b04b5798b55652cec1d15817c45e98b5214760abd'


class TransportPending(RuntimeError):
    pass


@contextlib.contextmanager
def controller_lock(path):
    """OS lock releases on process death; no stale PID-file deletion heuristic."""
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    try:
        if stream.tell()==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


class Magnolia:
    def __init__(self,key,package_sha,diagnostics=None):
        self.key=key
        self.package_sha=package_sha
        self.directory=(c.ROOT/'packages'/c.RUN_ID).as_posix()
        self.remote_zip=self.directory+'/'+package_sha+'.zip'
        self.diagnostics=diagnostics
        self.options=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-i',str(key)]

    def execute(self,argv,*,data=None,timeout=180):
        try:
            result=subprocess.run(argv,input=data,capture_output=True,timeout=timeout)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise TransportPending('Magnolia connection unavailable; existing job state is preserved') from exc
        if self.diagnostics is not None and result.returncode:
            with self.diagnostics.open('a',encoding='utf-8') as out:
                out.write(c.canonical({'observed_at':c.now(),'program':Path(argv[0]).name,'exit_code':result.returncode,'stdout':result.stdout.decode('utf-8','replace')[-4000:],'stderr':result.stderr.decode('utf-8','replace')[-4000:]}))
            print('transport_diagnostics='+str(self.diagnostics),flush=True)
            print(result.stderr.decode('utf-8','replace')[-2000:],flush=True)
        return result

    def stage(self,archive):
        # Arguments are quoted by shlex for remote bash. No user-controlled command text.
        remote=shlex.join(['mkdir','-p','--',self.directory])
        result=self.execute(['ssh',*self.options,HOST,remote])
        if result.returncode:
            raise TransportPending('Magnolia staging connection failed; retain this run state')
        # Upload to a unique partial path, then rename atomically. Never overwrite an active extraction.
        temporary=self.remote_zip+'.upload-'+uuid.uuid4().hex[:12]
        result=self.execute(['scp',*self.options,str(archive),HOST+':'+temporary],timeout=300)
        if result.returncode:
            raise TransportPending('Package transfer interrupted before validation; rerun the same launcher')
        remote=shlex.join(['mv','--',temporary,self.remote_zip])
        result=self.execute(['ssh',*self.options,HOST,remote])
        if result.returncode:
            raise TransportPending('Package transfer acknowledgement lost; rerun to reconcile')

    def action(self,action):
        remote=shlex.join(['bash','-s','--',action,self.package_sha])
        result=self.execute(['ssh',*self.options,HOST,remote],data=(c.BASE/'bootstrap.sh').read_bytes(),timeout=240)
        if result.returncode==255:
            raise TransportPending('SSH connection interrupted; rerun to attach to the same run')
        try:
            doc=c.strict(result.stdout)
        except (ValueError,c.Stop):
            # SSH banners or Python/shared-library failures are transport diagnostics, not judge output.
            raise TransportPending('Remote response was not a valid receipt (exit '+str(result.returncode)+'); return the local controller log')
        if result.returncode:
            raise c.Stop(str(doc.get('message','remote deterministic stop')))
        c.require(doc.get('run_id')==c.RUN_ID,'remote receipt run mismatch')
        return doc

    def download(self,receipt,path):
        c.require(receipt['remote_path']==(c.REMOTE_RUN/'result-bundle.zip').as_posix(),'unexpected remote result path')
        c.full_digest(receipt['zip_sha256'])
        result=self.execute(['scp',*self.options,HOST+':'+receipt['remote_path'],str(path)],timeout=600)
        if result.returncode:
            raise TransportPending('Result download interrupted; remote evidence is preserved')
        c.require(path.stat().st_size==receipt['bytes'] and c.file_sha(path)==receipt['zip_sha256'],'result transfer hash/size mismatch')


def validate_local(vault,repo,key):
    for tool in ('ssh','scp','git'):
        c.require(shutil.which(tool) is not None,'Windows dependency missing: '+tool)
    c.require(key.is_file(),'recorded Magnolia SSH identity file missing')
    root=vault/'tools/rayan-compute'
    r13=root/'provider-neutral/r13-r17-receipt.json'
    r18=root/'infrastructure/r18-r25-receipt-v1.0.4.json'
    c.require(r13.is_file() and c.file_sha(r13)==R13_SHA,'R13 provider receipt missing or changed')
    c.require(r18.is_file() and c.file_sha(r18)==R18_SHA,'R18–R25 infrastructure receipt missing or changed')
    a,b=c.read(r13),c.read(r18)
    c.require(all(a.get('r'+str(i))=='PASS' for i in range(13,18)) and a.get('main_mutated') is False,'R13 provider authority incomplete')
    c.require(all(b.get('r'+str(i))=='PASS' for i in range(18,26)) and b.get('mc10d_science_mutated') is False and b.get('canonical_main_mutated') is False,'R18–R25 authority incomplete')
    publish_context.validate_repo(repo)


def execute(args):
    package_sha=c.full_digest(args.package_sha)
    c.require(c.file_sha(args.package_zip)==package_sha,'outer package ZIP hash mismatch')
    c.verify_package();c.authority()
    state_root=args.vault_root/'tools/alice-astra/qwen-fallback-a2'
    state_root.mkdir(parents=True,exist_ok=True)
    with controller_lock(state_root/'controller.lock'):
        state_path=state_root/'controller-state.json'
        c.require(state_path.is_file(), 'The observed a2 state is required; no fresh execution inferred')
        state=c.read(state_path)
        c.require(state['run_id']==c.RUN_ID,'existing approved run identity differs')
        local=state_root/'runs'/c.RUN_ID
        local.mkdir(parents=True,exist_ok=True)
        remote=Magnolia(args.ssh_key,package_sha,local/'controller-diagnostics.jsonl')
        if state['package_sha256']==package_revision.policy()['old_package_sha256']:
            validate_local(args.vault_root,args.repo_root,args.ssh_key)
            package_revision.begin_local(state_path,package_sha)
            print('package_revision=V102_TO_V104_WITH_V103_INTENT_PRESERVED; original state archived',flush=True)
            remote.stage(args.package_zip)
            acknowledged=remote.action('revise')
            state=package_revision.finish_local(state_path,package_sha,acknowledged)
        else:
            state=package_revision.verify_local(state_path,package_sha)
        def save(**fields):
            state.update(fields,updated_at=c.now());c.write(state_path,state)
        print('run_id='+c.RUN_ID,flush=True)
        print('persistent_state='+str(state_path),flush=True)
        print('package_revision_receipt='+str(package_revision.history(state_root)/'remote-receipt.json'),flush=True)
        print('calibration_id='+c.CALIBRATION_ID,flush=True)
        print('source_job_preserved=575089',flush=True)
        print('live_telemetry_snapshot_prefix='+c.RUN_ID+'-t-',flush=True)
        if state.get('evidence_directory') and not state.get('telemetry_pending'):
            folder=Path(state['evidence_directory'])
            c.require(folder.resolve().is_relative_to(local.resolve()),'local evidence path escaped run')
            summary=evidence.analyze(folder,package_sha)
            print('verified_result_reused=true',flush=True)
        else:
            validate_local(args.vault_root,args.repo_root,args.ssh_key)
            if not state.get('staged'):
                remote.stage(args.package_zip)
                save(staged=True)
            # Even if an earlier submission response vanished, this action reconciles first.
            receipt=remote.action('submit')
            c.write(local/'submission.json',receipt)
            save(phase='SUBMITTED',job_id=receipt['job_id'])
            print('magnolia_job_id='+receipt['job_id'],flush=True)
            started=time.monotonic();last_print=0
            while True:
                info=remote.action('status')
                c.write(local/'scheduler-status.json',info)
                if time.monotonic()-last_print>=55 or info['terminal']:
                    progress=info.get('progress') or {}
                    print('scheduler='+info['scheduler_state']+' phase='+str(progress.get('phase','AWAITING_WORKER'))+' tasks_completed='+str(progress.get('tasks_completed',0))+'/16 tasks_attempted='+str(progress.get('tasks_attempted',0))+'/16',flush=True)
                    last_print=time.monotonic()
                    if progress.get('telemetry_url'):print('live_telemetry='+progress['telemetry_url'],flush=True)
                if info['terminal']:break
                if time.monotonic()-started>=8*3600:
                    save(phase='MONITOR_DETACHED')
                    raise TransportPending('Eight-hour monitor window ended. The six-hour job limit still applies. Rerun to attach; no job was cancelled')
                time.sleep(30)
            save(phase='COLLECTING')
            receipt=remote.action('collect')
            c.write(local/'collection-receipt.json',receipt)
            temp=local/('download-'+uuid.uuid4().hex[:12]+'.zip')
            remote.download(receipt,temp)
            archive=local/('ALICE_QWEN_PUBLIC_RESULT_'+receipt['zip_sha256'][:12]+'.zip')
            os.replace(temp,archive)
            folder=local/('evidence-'+receipt['zip_sha256'][:12])
            if not folder.exists():
                evidence.extract_verified(archive,receipt['zip_sha256'],folder)
            summary=evidence.analyze(folder,package_sha)
            save(phase='EVIDENCE_VERIFIED',evidence_directory=str(folder),result_zip=str(archive),result_zip_sha256=receipt['zip_sha256'])
        c.write(local/'verified-summary.json',summary)
        # A copy beside the launcher downloads is the file to return in the next chat.
        source=Path(state['result_zip'])
        c.require(c.file_sha(source)==state['result_zip_sha256'],'retained result ZIP drift')
        args.output_root.mkdir(parents=True,exist_ok=True)
        public_file=args.output_root/source.name
        if public_file.exists():
            c.require(c.file_sha(public_file)==state['result_zip_sha256'],'output filename collision')
        else:
            shutil.copyfile(source,public_file)
        print('RESULT_ZIP='+str(public_file),flush=True)
        print('RESULT_SHA256='+state['result_zip_sha256'],flush=True)
        print('qualification_passed='+str(summary['qualification_passed']).lower()+' calibration_only=true',flush=True)
        save(telemetry_pending=not summary['final_telemetry_published'])
        try:
            publication,_=publish_context.publish(folder,package_sha,args.repo_root)
            c.write(local/'context-publication.json',publication)
            save(phase='COMPLETE' if summary['final_telemetry_published'] else 'TELEMETRY_PUBLICATION_PENDING',context_commit=publication['commit'])
            print('context_commit='+publication['commit'],flush=True)
        except Exception as exc:
            save(phase='CONTEXT_PUBLICATION_PENDING',publication_error_class=type(exc).__name__)
            print('CONTEXT_PUBLICATION_PENDING: verified result is preserved. Rerun only this launcher to retry publication.',flush=True)
            return 75
        if not summary['final_telemetry_published']:
            print('TERMINAL_TELEMETRY_PENDING: context and verified results are preserved. The same launcher retries terminal telemetry without inference.',flush=True)
            return 75
        return 0 if summary['qualification_passed'] else 76


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-zip',required=True,type=Path)
    p.add_argument('--package-sha',required=True)
    p.add_argument('--vault-root',type=Path,default=Path(r'C:\ALICE_Vault'))
    p.add_argument('--repo-root',type=Path,default=Path(r'C:\A.L.I.C.E-main'))
    p.add_argument('--output-root',type=Path,default=Path.home()/'Downloads')
    p.add_argument('--ssh-key',type=Path,default=Path.home()/'.ssh/rayan_magnolia_ed25519')
    args=p.parse_args()
    try:
        return execute(args)
    except (TransportPending,KeyboardInterrupt) as exc:
        print('QWEN_RUN_PENDING: '+str(exc)+'. Run the same launcher to resume monitoring/collection. No automatic new job.',flush=True)
        return 74
    except Exception as exc:
        print('QWEN_DETERMINISTIC_STOP: '+type(exc).__name__+': '+str(exc)[:1200],flush=True)
        print('Return this terminal output. Existing attempts and prior judge evidence remain preserved.',flush=True)
        return 76


if __name__=='__main__':
    raise SystemExit(main())
