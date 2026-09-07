"""Idempotent Magnolia submission, read-only monitoring, and evidence collection."""
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

import contract as c
import infra_recovery
import telemetry_snapshots
import evidence_origins
import package_revision

TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def cmd(argv, check=True):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    if check:
        c.require(result.returncode==0, 'scheduler command failed: '+argv[0]+'; '+result.stderr.strip()[:600])
    return result


def normalize_state(value):
    return value.split()[0].rstrip('+') if value.strip() else 'UNKNOWN'


def find_submitted():
    queue=cmd(['squeue','-h','-u','mxrayan','-o','%A|%j|%T'])
    accounting=cmd(['sacct','-X','-n','-P','-u','mxrayan','--starttime','2026-09-06','--name',c.RUN_ID,'--format','JobIDRaw,JobName%100,State'])
    found={}
    for line in (queue.stdout+'\n'+accounting.stdout).splitlines():
        parts=line.strip().split('|')
        if len(parts)>=3 and parts[0].isdigit() and parts[1]==c.RUN_ID:
            found[parts[0]]=normalize_state(parts[2])
    c.require(len(found)<=1, 'multiple scheduler jobs for the same approved run; review required')
    return list(found)[0] if found else None


def slurm_script(package_sha):
    run=c.REMOTE_RUN
    # All substituted paths are fixed package-owned paths; no user text enters bash code.
    return f'''#!/bin/bash
#SBATCH --job-name={c.RUN_ID}
#SBATCH --partition=node
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output={run}/slurm.out
#SBATCH --error={run}/slurm.err
set -euo pipefail
umask 077
PYROOT="/modules/pkgs/common/python/3.11.5"
PY="$PYROOT/bin/python3.11"
PYLIB=""
for d in "$PYROOT/lib" "$PYROOT/lib64"; do
  if [ -e "$d/libpython3.11.so.1.0" ]; then PYLIB="$d"; break; fi
done
test -n "$PYLIB"
export LD_LIBRARY_PATH="$PYLIB${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1
cd {run}
exec "$PY" {c.BASE}/worker.py --run {run} --package-sha {package_sha}
'''


@contextlib.contextmanager
def submission_lock(path):
    import fcntl
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def submit(run, package_sha):
    run.mkdir(parents=True,exist_ok=True)
    with submission_lock(run/'submission.lock'):
        package_revision.verify_current(run,package_sha)
        descriptor=package_revision.descriptor(package_sha)
        c.immutable(run/'run.json',descriptor)
        if (run/'submission.json').exists():
            return c.read(run/'submission.json')
        found=find_submitted()
        if found:
            receipt={**descriptor,'job_id':found,'state':'SUBMITTED','recovered_after_transport_loss':True,'recorded_at':c.now()}
            c.immutable(run/'submission.json',receipt)
            return receipt
        c.require(not (run/'submission-intent.json').exists(), 'SUBMISSION_UNRESOLVED: intent exists without a unique scheduler receipt; no duplicate submission')
        for tool in ('sbatch','squeue','sacct','tar'):
            c.require(shutil.which(tool) is not None, 'Magnolia dependency missing: '+tool)
        helper=c.ROOT/'bin/rayan-telemetry-push.sh'
        c.require(helper.is_file() and os.access(helper,os.X_OK), 'Magnolia telemetry helper missing')
        c.require((c.ROOT/'telemetry/ledger/.git').exists(), 'Magnolia telemetry checkout missing')
        try:
            infra_recovery.preflight(run,cmd)
        except Exception as exc:
            c.write(run/'infrastructure-preflight-failure.json',{'status':'STOP','class':type(exc).__name__,'message':str(exc)[:1200],'job_submitted':False,'observed_at':c.now()})
            raise
        script=slurm_script(package_sha)
        c.atomic_bytes(run/'job.sbatch',script.encode())
        c.immutable(run/'submission-intent.json',{**descriptor,'state':'SUBMIT_INTENT','script_sha256':c.file_sha(run/'job.sbatch'),'recorded_at':c.now()})
        result=cmd(['sbatch','--parsable',str(run/'job.sbatch')],check=False)
        c.write(run/'sbatch-response.json',{'exit_code':result.returncode,'stdout':result.stdout[:1000],'stderr':result.stderr[:2000]})
        match=re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?',result.stdout.strip())
        c.require(result.returncode==0 and match is not None, 'SUBMISSION_UNRESOLVED: sbatch exit '+str(result.returncode)+'; '+(result.stderr or result.stdout).strip()[:800]+'; rerun to reconcile, never resubmit blindly')
        receipt={**descriptor,'job_id':match.group(1),'state':'SUBMITTED','recovered_after_transport_loss':False,'recorded_at':c.now()}
        c.immutable(run/'submission.json',receipt)
        return receipt


def status(run):
    receipt=c.read(run/'submission.json')
    job=receipt['job_id']
    c.require(isinstance(job,str) and job.isdigit(), 'invalid stored job ID')
    queue=cmd(['squeue','-h','-j',job,'-o','%A|%j|%T'],check=False)
    lines=[x.split('|') for x in queue.stdout.splitlines() if x.strip()]
    state='UNKNOWN'
    in_queue=False
    for row in lines:
        if len(row)>=3 and row[0]==job:
            c.require(row[1]==c.RUN_ID,'scheduler job identity mismatch')
            state=normalize_state(row[2]);in_queue=True
    if not in_queue:
        result=cmd(['sacct','-X','-n','-P','-j',job,'--format','JobIDRaw,JobName%100,State,ExitCode'])
        for line in result.stdout.splitlines():
            row=line.strip().split('|')
            if len(row)>=4 and row[0]==job:
                c.require(row[1]==c.RUN_ID,'accounting job identity mismatch')
                state=normalize_state(row[2])
    obj={'run_id':c.RUN_ID,'job_id':job,'scheduler_state':state,'terminal':not in_queue and state in TERMINAL,'worker_finished':(run/'worker-finished.json').exists(),'progress':c.read(run/'progress.json') if (run/'progress.json').exists() else None,'observed_at':c.now()}
    c.write(run/'scheduler-status.json',obj)
    return obj


def revise(run, package_sha):
    c.require(run.is_dir(),'Observed a2 run directory is missing; no fresh run inferred')
    with submission_lock(run/'submission.lock'):
        return package_revision.apply_remote(run,package_sha,cmd,find_submitted)


def evidence_paths(run):
    # Explicit allowlist: excludes runtime/model cache, environment, key/config files.
    names=['run.json','submission.json','submission-intent.json','sbatch-response.json','job.sbatch','scheduler-status.json','worker-started.json','worker-finished.json','result.json','failure.json','model-lock.json','model-manifest.json','model-blobs.json','model-show.json','model-tags.json','model-loaded.json','runtime.json','runtime-files.json','effective-contract.json','preflight.json','probe-request.json','probe-attempt.json','probe-response.ndjson','progress.json','telemetry-manifest.json','telemetry-receipt.json','telemetry-publication.json','application.log','slurm.out','slurm.err','runtime-extraction.log','ollama-serve.log']
    paths={name:run/name for name in names if (run/name).is_file()}
    for name in ('runtime-decoder.json','zstandard-install.json','tls-trust.json','parent-reconciliation.json','infrastructure-preflight.json','infrastructure-preflight-failure.json','telemetry-events.jsonl','telemetry-latest/manifest.json','telemetry-latest/receipt.json','telemetry-latest/application.log'):
        if (run/name).is_file():paths[name]=run/name
    if (run/'telemetry-recovery.json').is_file():paths['telemetry-recovery.json']=run/'telemetry-recovery.json'
    _,tasks,_=c.authority()
    for task in tasks:
        for name in ('request.json','attempt.json','response.ndjson','record.json'):
            rel='tasks/'+task['task_id']+'/'+name
            if (run/rel).is_file():paths[rel]=run/rel
    paths.update(evidence_origins.package_projections(c.BASE,
        evidence_origins.SOURCE_AUTHORITIES + ('infra_recovery.json','request_hashes_v100.json','revision_policy.json')))
    paths.update(package_revision.evidence_paths(run))
    return paths


def repair_terminal_telemetry(run,state):
    telemetry_snapshots.terminal_recovery(run,state)


def collect(run):
    state=status(run)
    c.require(state['terminal'], 'scheduler is not terminal; collect only after logs and outputs are closed')
    repair_terminal_telemetry(run,state)
    paths=evidence_paths(run)
    manifest={'schema':'alice.mc10d.qwen.result-bundle-manifest.v1','run_id':c.RUN_ID,'files':[{'path':name,'bytes':p.stat().st_size,'sha256':c.file_sha(p)} for name,p in sorted(paths.items())]}
    c.write(run/'EVIDENCE_MANIFEST.json',manifest)
    temp=run/'result-bundle.zip.tmp'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(paths.items()):z.write(p,name)
        z.write(run/'EVIDENCE_MANIFEST.json','EVIDENCE_MANIFEST.json')
    os.replace(temp,run/'result-bundle.zip')
    return {'run_id':c.RUN_ID,'remote_path':str(run/'result-bundle.zip'),'zip_sha256':c.file_sha(run/'result-bundle.zip'),'bytes':(run/'result-bundle.zip').stat().st_size,'terminal_state':state['scheduler_state'],'raw_outputs_retained':True,'model_cache_deleted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('revise','submit','status','collect'))
    p.add_argument('--package-sha',required=True)
    args=p.parse_args()
    c.full_digest(args.package_sha)
    c.verify_package();c.authority()
    run=c.REMOTE_RUN
    try:
        if args.action=='revise':obj=revise(run,args.package_sha)
        elif args.action=='submit':obj=submit(run,args.package_sha)
        else:
            c.require(c.read(run/'run.json')['package_sha256']==args.package_sha,'remote package identity differs')
            obj=status(run) if args.action=='status' else collect(run)
        print(c.canonical(obj),end='')
        return 0
    except Exception as exc:
        print(c.canonical({'state':'STOP','run_id':c.RUN_ID,'action':args.action,'error_class':type(exc).__name__,'message':str(exc)[:1200]}),end='')
        return 76


if __name__=='__main__':
    raise SystemExit(main())
