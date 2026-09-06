"""One Slurm allocation, one public Qwen calibration, no inference retries."""
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

import contract as c


def command(argv, *, timeout=60, env=None, log=None):
    if log:
        with Path(log).open('ab') as out:
            result = subprocess.run([str(x) for x in argv], stdout=out, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    else:
        result = subprocess.run([str(x) for x in argv], capture_output=True, env=env, timeout=timeout)
    c.require(result.returncode == 0, 'command failed: ' + Path(str(argv[0])).name)
    return result.stdout.decode('utf-8', 'replace').strip() if not log else ''


def api(base, endpoint, body=None, timeout=60):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(base + endpoint, data=c.canonical(body).encode() if body is not None else None, headers={'Content-Type':'application/json'})
    with opener.open(request, timeout=timeout) as response:
        return c.strict(response.read(8 * 1024 * 1024))


def stream_request(base, body, path, deadline):
    """No retry wrapper: a recorded intent consumes this task's only request."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(base + '/api/chat', data=c.canonical(body).encode(), headers={'Content-Type':'application/json'})
    with Path(path).open('xb') as out:
        with opener.open(request, timeout=min(300, max(1,deadline-time.monotonic()))) as response:
            total = 0
            while True:
                c.require(time.monotonic() < deadline, 'request allocation deadline')
                line = response.readline(2 * 1024 * 1024)
                if not line:
                    break
                total += len(line)
                c.require(total <= 32 * 1024 * 1024, 'response byte limit')
                out.write(line)
                out.flush()
            os.fsync(out.fileno())


def download(url, target, expected_sha, expected_size, deadline):
    target = Path(target)
    if target.exists():
        c.require(target.stat().st_size == expected_size and c.file_sha(target) == expected_sha, 'existing cached artifact hash drift')
        return 'VERIFIED_CACHE'
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.partial-' + os.environ.get('SLURM_JOB_ID','local'))
    # Partial bytes are preserved for diagnosis; never mistaken for a verified cache hit.
    c.require(not temp.exists(), 'unverified partial download exists; no automatic restart')
    with urllib.request.urlopen(url, timeout=60) as source, temp.open('xb') as out:
        size = 0
        while True:
            c.require(time.monotonic() < deadline, 'artifact preparation deadline')
            data = source.read(8 * 1024 * 1024)
            if not data:
                break
            size += len(data)
            c.require(size <= expected_size, 'download size exceeds frozen size')
            out.write(data)
        out.flush()
        os.fsync(out.fileno())
    c.require(size == expected_size and c.file_sha(temp) == expected_sha, 'download identity mismatch')
    os.replace(temp, target)
    return 'DOWNLOADED_AND_VERIFIED'


class Telemetry:
    """Fixed operational metadata only; inference text never enters the ledger log."""
    def __init__(self, run, package_sha):
        self.run = run
        self.helper = c.ROOT / 'bin/rayan-telemetry-push.sh'
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.phase = 'PREFLIGHT'
        self.completed = 0
        self.attempted = 0
        self.last_rc = None
        self.thread = None
        self.job = os.environ.get('SLURM_JOB_ID','')
        c.require(self.job.isdigit(), 'Slurm job identity missing')
        c.write(run/'telemetry-manifest.json', {'schema':'rayan.compute.run-manifest.v1','run_id':c.RUN_ID,'stage':'MC10D_QWEN_PUBLIC_CALIBRATION_V1','provider':'magnolia','slurm_job_id':self.job,'opaque_workload_sha256':package_sha,'approved_amendment_sha256':c.APPROVED_SHA,'accelerator':'none'})

    def push(self, status='RUNNING'):
        with self.lock:
            receipt = {'schema':'rayan.compute.run-receipt.v1','run_id':c.RUN_ID,'stage':'MC10D_QWEN_PUBLIC_CALIBRATION_V1','status':status,'provider':'magnolia','slurm_job_id':self.job,'phase':self.phase,'tasks_completed':self.completed,'tasks_attempted':self.attempted,'tasks_required':16,'heartbeat_is_task_completion':False,'observed_at':c.now()}
            c.write(self.run/'telemetry-receipt.json', receipt)
            c.write(self.run/'progress.json', receipt)
            with (self.run/'application.log').open('a', encoding='utf-8') as out:
                out.write(c.canonical(receipt))
            try:
                result = subprocess.run([str(self.helper), c.RUN_ID, 'MC10D_QWEN_PUBLIC_CALIBRATION_V1', status, self.job, str(self.run/'application.log'), str(self.run/'telemetry-manifest.json'), str(self.run/'telemetry-receipt.json')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
                self.last_rc = result.returncode
            except (OSError, subprocess.TimeoutExpired):
                self.last_rc = 75
            c.write(self.run/'telemetry-publication.json', {'exit_code':self.last_rc,'phase':self.phase,'observed_at':c.now(),'raw_model_responses_published':False})
            return self.last_rc

    def start(self):
        c.require(self.helper.is_file() and os.access(self.helper, os.X_OK), 'Magnolia telemetry helper missing')
        c.require(self.push() == 0, 'initial GitHub telemetry publication failed; inference not started')
        def heartbeat():
            while not self.done.wait(300):
                self.push()
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()

    def stop(self, status):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=95)
        return self.push(status)


def prepare_runtime(run, policy, deadline):
    archive = c.ROOT / 'cache/runtime' / policy['archive_sha256'].upper() / policy['archive_name']
    archive.parent.mkdir(parents=True, exist_ok=True)
    c.require(shutil.disk_usage(archive.parent).free >= (0 if archive.exists() else policy['archive_size']) + 6*1024**3, 'runtime disk capacity insufficient')
    state = download(policy['archive_url'], archive, policy['archive_sha256'], policy['archive_size'], deadline)
    dest = run / 'runtime'
    c.require(not dest.exists(), 'runtime work directory already exists; do not restart this allocation')
    dest.mkdir()
    decoder={'backend':'system_tar_zstd','pinned_runtime_archive_verified':True}
    try:
        command(['tar', '--zstd', '-xf', archive, '-C', dest], timeout=min(600,max(1,deadline-time.monotonic())), log=run/'runtime-extraction.log')
    except (c.Stop,subprocess.TimeoutExpired):
        # The historical worker used the same exact zstandard version when tar lacked --zstd.
        # Install into this run only. The pinned runtime archive/binary never change.
        deps=run/'runtime-deps'
        command([sys.executable,'-m','pip','--isolated','install','--index-url','https://pypi.org/simple','--no-deps','--only-binary=:all:','--disable-pip-version-check','--no-input','--target',deps,'--report',run/'zstandard-install.json','zstandard==0.25.0'], timeout=min(300,max(1,deadline-time.monotonic())), log=run/'runtime-extraction.log')
        sys.path.insert(0,str(deps))
        import importlib
        importlib.invalidate_caches()
        import zstandard
        c.require(zstandard.__version__=='0.25.0','pinned decompressor version mismatch')
        c.require(getattr(tarfile,'data_filter',None) is not None,'safe tar extraction filter unavailable')
        with archive.open('rb') as src, zstandard.ZstdDecompressor().stream_reader(src) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as tar:
                for member in tar:
                    c.require(time.monotonic()<deadline,'runtime extraction deadline')
                    tar.extract(member,path=dest,filter='data')
        decoder={'backend':'zstandard','version':'0.25.0','install_report_sha256':c.file_sha(run/'zstandard-install.json'),'pinned_runtime_archive_verified':True,'files':[{'path':p.relative_to(deps).as_posix(),'sha256':c.file_sha(p)} for p in sorted(deps.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]}
    c.write(run/'runtime-decoder.json',decoder)
    binary = dest/'bin/ollama'
    c.require(binary.is_file() and c.file_sha(binary) == policy['binary_sha256'], 'pinned runtime binary mismatch')
    binary.chmod(binary.stat().st_mode | 0o111)
    files = []
    for p in sorted(dest.rglob('*')):
        if p.is_file():
            c.require(p.resolve().is_relative_to(dest.resolve()), 'runtime link escapes archive')
            files.append({'path':p.relative_to(dest).as_posix(), 'sha256':c.file_sha(p), 'bytes':p.stat().st_size})
    c.write(run/'runtime-files.json', {'files':files})
    return binary, {'archive_sha256':policy['archive_sha256'], 'binary_sha256':policy['binary_sha256'], 'archive_cache_status':state, 'runtime_files_sha256':c.file_sha(run/'runtime-files.json'),'runtime_decoder_sha256':c.file_sha(run/'runtime-decoder.json')}


def resolve_manifest(run):
    url = 'https://registry.ollama.ai/v2/library/qwen3.8/manifests/27b-q4_K_M'
    req = urllib.request.Request(url, headers={'Accept':'application/vnd.docker.distribution.manifest.v2+json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read(1024*1024)
        header = response.headers.get('Docker-Content-Digest')
    digest = c.sha(raw)
    if header:
        c.require(c.full_digest(header) == digest, 'registry manifest header mismatch')
    manifest = c.strict(raw)
    c.require(manifest.get('schemaVersion') == 2 and isinstance(manifest.get('layers'),list), 'unsupported model manifest')
    blobs = [manifest['config']] + manifest['layers']
    for blob in blobs:
        c.full_digest(blob['digest'])
        c.require(type(blob['size']) is int and blob['size'] > 0, 'invalid model blob size')
    c.require(sum(x['size'] for x in blobs) < 30*1024**3, 'unexpected model distribution size')
    lock = {'schema':'alice.mc10d.qwen.model-lock.v1','model_tag':c.TAG,'full_manifest_digest':digest,'source_url':url,'registry_header_digest':header,'resolved_at':c.now(),'manifest_sha256':digest,'approved_amendment_sha256':c.APPROVED_SHA,'inference_started':False}
    c.immutable(run/'model-lock.json', lock)
    c.atomic_bytes(run/'model-manifest.json', raw)
    return lock, manifest, raw


def prepare_model(run, policy, deadline):
    lock, manifest, raw = resolve_manifest(run)
    models = c.ROOT / policy['new_dedicated_model_cache']
    (models/'blobs').mkdir(parents=True, exist_ok=True)
    blobs = {b['digest']:b for b in [manifest['config']]+manifest['layers']}
    # Bounded known roots; no private corpus scans and no unverified reuse.
    prior_roots = [Path.home()/'.ollama/models', c.ROOT/'cache/ollama/models']
    missing = sum(b['size'] for d,b in blobs.items() if not (models/'blobs'/d.replace(':','-')).exists())
    c.require(shutil.disk_usage(models).free >= missing + 2*1024**3, 'model disk capacity insufficient')
    receipts = []
    for digest, blob in blobs.items():
        full = c.full_digest(digest)
        target = models/'blobs'/digest.replace(':','-')
        for previous in prior_roots:
            src = previous/'blobs'/target.name
            if not target.exists() and src.is_file() and src.stat().st_size==blob['size'] and c.file_sha(src)==full:
                try:
                    os.link(src,target)
                except OSError:
                    shutil.copyfile(src,target)
                break
        state = download('https://registry.ollama.ai/v2/library/qwen3.8/blobs/'+digest, target, full, blob['size'], deadline)
        receipts.append({'digest':digest,'bytes':blob['size'],'status':state})
    target = models/'manifests/registry.ollama.ai/library/qwen3.8/27b-q4_K_M'
    if target.exists():
        c.require(target.read_bytes()==raw, 'cached Qwen tag changed; no automatic retagging')
    else:
        c.atomic_bytes(target, raw)
    c.write(run/'model-blobs.json', {'manifest_sha256':lock['full_manifest_digest'],'blobs':receipts})
    return models, lock


def start_service(binary, models, run):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    env = {k:v for k,v in os.environ.items() if not k.startswith('OLLAMA_')}
    env.update(OLLAMA_HOST='127.0.0.1:'+str(port), OLLAMA_MODELS=str(models), OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_KEEP_ALIVE='30m', OLLAMA_CONTEXT_LENGTH='8192', OLLAMA_KV_CACHE_TYPE='f16', OLLAMA_FLASH_ATTENTION='false', OLLAMA_NO_CLOUD='1', CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1', GGML_VK_VISIBLE_DEVICES='-1')
    log = (run/'ollama-serve.log').open('ab')
    proc = subprocess.Popen([str(binary),'serve'], stdout=log, stderr=subprocess.STDOUT, env=env)
    log.close()
    base = 'http://127.0.0.1:'+str(port)
    for _ in range(45):
        if proc.poll() is not None:
            raise c.Stop('pinned Ollama exited before readiness; inspect retained server log')
        try:
            version = api(base,'/api/version')['version']
            return proc, base, version
        except (OSError, ValueError, KeyError):
            time.sleep(2)
    proc.terminate()
    proc.wait(timeout=10)
    raise c.Stop('Ollama readiness timed out')


def runtime_check(base, lock, *, loaded=False):
    models = api(base, '/api/ps' if loaded else '/api/tags')['models']
    found = [m for m in models if m.get('name')==c.TAG or m.get('model')==c.TAG]
    c.require(len(found)==1, 'exact Qwen runtime model absent/duplicated')
    model = found[0]
    c.require(c.full_digest(model['digest'])==lock['full_manifest_digest'], 'runtime full model digest drift')
    c.require(model.get('details',{}).get('quantization_level')=='Q4_K_M', 'runtime quantization mismatch')
    if loaded:
        c.require(type(model.get('size_vram')) is int and model['size_vram']==0, 'CPU-only runtime proof failed')
        c.require(model.get('context_length')==8192, 'runtime context length mismatch')
    return model


def run_task(run, base, task, index, approved, lock, contract_sha, deadline):
    folder = run/'tasks'/task['task_id']
    folder.mkdir(parents=True, exist_ok=True)
    intent_path = folder/'attempt.json'
    c.require(not intent_path.exists(), 'task already attempted; refusing repeat inference')
    body = c.request(task,index,approved)
    c.immutable(folder/'request.json',body)
    intent = {'task_id':task['task_id'],'attempt_count':1,'request_sha256':c.file_sha(folder/'request.json'),'contract_sha256':contract_sha,'model_digest':lock['full_manifest_digest'],'state':'IN_FLIGHT','started_at':c.now()}
    c.immutable(intent_path,intent)
    error = None
    try:
        stream_request(base,body,folder/'response.ndjson',deadline)
    except (Exception, KeyboardInterrupt) as exc:
        error = type(exc).__name__
    raw = (folder/'response.ndjson').read_bytes() if (folder/'response.ndjson').exists() else b''
    classification = c.classify(raw)
    record = {**intent, **classification, 'state':'RECORDED', 'finished_at':c.now(), 'stream_sha256':c.sha(raw), 'transport_error_class':error}
    if classification['status']=='COMPLETE':
        record['semantic_verdict_match'] = record['response']['verdict']==task['gold']['verdict']
        record['semantic_critical_decision_match'] = all(record['response'][f]==task['gold'][f] for f in ('verdict','critical_veto'))
    c.immutable(folder/'record.json',record)
    # A complete final chunk survives a connection-close error. It still requires identity revalidation.
    runtime_check(base,lock,loaded=True)
    return record


def existing_records(run, tasks):
    records = []
    for task in tasks:
        p = run/'tasks'/task['task_id']/'record.json'
        if p.exists():
            records.append(c.read(p))
        elif (p.parent/'attempt.json').exists():
            attempt = c.read(p.parent/'attempt.json')
            records.append({**attempt,'status':'INCOMPLETE','response':None,'error_code':'IN_FLIGHT_OUTCOME_UNRESOLVED'})
    return records


def work(run, package_sha):
    run.mkdir(parents=True, exist_ok=True)
    c.verify_package()
    approved, tasks, policy = c.authority()
    c.require(run==c.REMOTE_RUN, 'remote run path mismatch')
    c.require(not (run/'result.json').exists(), 'terminal calibration already recorded; collect it')
    c.require(not (run/'worker-started.json').exists(), 'allocation already started; no automatic inference restart')
    c.immutable(run/'worker-started.json', {'run_id':c.RUN_ID,'job_id':os.environ.get('SLURM_JOB_ID'),'started_at':c.now(),'package_sha256':package_sha})
    start = time.monotonic()
    deadline = start + 6*3600 - 150
    prepare_deadline = start + policy['maximum_prepare_seconds']
    telemetry = Telemetry(run,package_sha)
    proc = None
    failure = None
    runtime_ok = False
    contract_sha = None
    try:
        c.require(os.environ.get('SLURM_CPUS_PER_TASK')=='20', 'Slurm CPU request mismatch')
        c.require(os.environ.get('SLURM_JOB_PARTITION')=='node', 'unauthorized partition')
        c.require(os.environ.get('SLURM_MEM_PER_NODE')=='49152', 'Slurm 48 GiB allocation mismatch')
        c.require(len(os.sched_getaffinity(0))>=20, 'insufficient CPU affinity')
        c.require(not os.environ.get('SLURM_JOB_GPUS'), 'GPU allocation forbidden')
        telemetry.start()
        telemetry.phase = 'RUNTIME_PREPARATION'
        telemetry.push()
        binary, identity = prepare_runtime(run,policy,prepare_deadline)
        telemetry.phase = 'MODEL_PREPARATION'
        telemetry.push()
        models, lock = prepare_model(run,policy,prepare_deadline)
        proc, base, version = start_service(binary,models,run)
        c.require(version==policy['version'], 'pinned Ollama API version mismatch')
        tags = runtime_check(base,lock)
        details = api(base,'/api/show',{'model':c.TAG})
        c.require({'completion','thinking'} <= set(details.get('capabilities',[])), 'runtime lacks required completion/thinking capability')
        c.require(details.get('details',{}).get('family')=='qwen35', 'Qwen architecture mismatch')
        c.write(run/'model-show.json',details)
        c.write(run/'model-tags.json',tags)
        runtime = {**identity,'version':version,'model_tag':c.TAG,'model_digest':lock['full_manifest_digest'],'show_sha256':c.file_sha(run/'model-show.json'),'tags_sha256':c.file_sha(run/'model-tags.json'),'model_blobs_sha256':c.file_sha(run/'model-blobs.json'),'accelerator':'none','num_thread':20,'num_gpu':0,'context':8192,'kv_cache_type':'f16','flash_attention':False,'python_version':sys.version,'job_id':os.environ['SLURM_JOB_ID']}
        c.immutable(run/'runtime.json',runtime)
        effective = {'schema':'alice.mc10d.qwen.effective-execution-contract.v1','run_id':c.RUN_ID,'package_sha256':package_sha,'package_manifest_sha256':c.file_sha(c.BASE/'PACKAGE_MANIFEST.json'),'amendment_sha256':c.APPROVED_SHA,'owner_approval_sha256':c.APPROVAL_SHA,'tasks_sha256':c.TASK_SHA,'runtime_sha256':c.file_sha(run/'runtime.json'),'model_lock_sha256':c.file_sha(run/'model-lock.json'),'profile':approved['profile'],'pass_rule':approved['pass_rule'],'no_authority':c.NO_AUTHORITY}
        c.immutable(run/'effective-contract.json',effective)
        contract_sha = c.file_sha(run/'effective-contract.json')
        telemetry.phase = 'CPU_THROUGHPUT_PREFLIGHT'
        telemetry.push()
        probe = c.probe_request(approved,tasks)
        c.immutable(run/'probe-request.json',probe)
        c.immutable(run/'probe-attempt.json', {'attempt_count':1,'contract_sha256':contract_sha,'request_sha256':c.file_sha(run/'probe-request.json'),'qualification_task':False})
        stream_request(base,probe,run/'probe-response.ndjson',min(deadline-300,time.monotonic()+1800))
        probe_result = c.decode_stream((run/'probe-response.ndjson').read_bytes())
        c.require(probe_result.get('done_reason') in ('stop','length'), 'throughput probe finish reason unsupported')
        c.require(type(probe_result.get('eval_count')) is int and probe_result['eval_count']<=128, 'probe token ceiling exceeded')
        loaded = runtime_check(base,lock,loaded=True)
        c.write(run/'model-loaded.json',loaded)
        estimate = c.projection(probe_result,16,deadline-time.monotonic())
        c.immutable(run/'preflight.json', {'status':'PASS' if estimate['fits'] else 'INSUFFICIENT_THROUGHPUT','projection':estimate,'probe_is_qualification':False,'cpu_only_verified':True,'model_loaded_sha256':c.file_sha(run/'model-loaded.json'),'contract_sha256':contract_sha})
        c.require(estimate['fits'], 'CPU throughput cannot fit approved worst-case budget; no calibration tasks started')
        runtime_ok = True
        for index,task in enumerate(tasks,1):
            c.require(time.monotonic()<deadline-300, 'allocation cleanup reserve reached')
            runtime_check(base,lock)
            telemetry.phase = 'TASK_'+str(index).zfill(2)
            telemetry.attempted = index
            record = run_task(run,base,task,index,approved,lock,contract_sha,deadline-120)
            telemetry.completed += int(record['status']=='COMPLETE')
            telemetry.push()
            print('task_recorded='+str(index)+'/16 status='+record['status'], flush=True)
    except BaseException as exc:
        failure = {'class':type(exc).__name__,'message':str(exc)[:1200], 'phase':telemetry.phase, 'recorded_at':c.now()}
        c.write(run/'failure.json',failure)
        print('QWEN_STOP phase='+telemetry.phase+' class='+type(exc).__name__, flush=True)
    finally:
        # Never delete model blobs, attempts, responses, logs, or prior judge evidence.
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        records = existing_records(run,tasks)
        scores = c.score(records,tasks)
        qualified = runtime_ok and failure is None and scores['qualification_passed']
        status = 'PUBLIC_CALIBRATION_PASSED' if qualified else ('PUBLIC_CALIBRATION_FAILED' if failure is None else 'INCOMPLETE_OR_PREFLIGHT_STOP')
        result = {'schema':'alice.mc10d.qwen.public-calibration-result.v1','run_id':c.RUN_ID,'status':status,'family':'qwen','model_tag':c.TAG,'effective_contract_sha256':contract_sha,'package_sha256':package_sha,'tasks_attempted':len(records),'scores':scores,'qualification_passed':qualified,'runtime_preflight_passed':runtime_ok,'failure_present':failure is not None,'no_authority':c.NO_AUTHORITY,'preserved_counts':{'replacements':63,'deferred':1,'candidate_pool':287},'completed_at':c.now()}
        c.write(run/'result.json',result)
        telemetry.completed=scores['tasks_complete']
        telemetry.attempted=len(records)
        telemetry.phase=status
        telemetry_rc=telemetry.stop('COMPLETED' if qualified else 'FAILED')
        c.write(run/'worker-finished.json',{'run_id':c.RUN_ID,'result_sha256':c.file_sha(run/'result.json'),'application_exit_code':0 if qualified else 76,'telemetry_exit_code':telemetry_rc,'finished_at':c.now()})
    return (0 if telemetry_rc==0 else 86) if qualified else 76


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--package-sha',required=True)
    args=p.parse_args()
    c.full_digest(args.package_sha)
    def terminate(signum, frame):
        raise c.Stop('scheduler termination signal; preserve partial evidence')
    signal.signal(signal.SIGTERM,terminate)
    with (args.run/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return work(args.run,args.package_sha)


if __name__=='__main__':
    raise SystemExit(main())
