"""Rebuild public results from frozen requests and captured HTTP streams."""
from __future__ import annotations

from pathlib import Path
import zipfile

import contract as c
import infra_recovery
import telemetry_snapshots


def extract_verified(archive, expected_sha, destination):
    c.require(c.file_sha(archive)==expected_sha,'downloaded result ZIP hash mismatch')
    c.require(not destination.exists(),'result extraction destination already exists')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        c.require(len(names)==len(set(names)) and len(names)<150,'result archive duplicate/count')
        c.require(sum(x.file_size for x in z.infolist())<=600*1024*1024,'result archive uncompressed size')
        for info in z.infolist():
            c.safe_relative(info.filename)
            c.require((info.external_attr>>16)&0o170000 != 0o120000,'result archive symlink')
        c.require('EVIDENCE_MANIFEST.json' in names,'result manifest absent')
        manifest=c.strict(z.read('EVIDENCE_MANIFEST.json'))
        expected={x['path']:x for x in manifest['files']}
        c.require(len(expected)==len(manifest['files']) and set(expected)|{'EVIDENCE_MANIFEST.json'}==set(names),'result manifest membership')
        for name,item in expected.items():
            raw=z.read(name)
            c.require(len(raw)==item['bytes'] and c.sha(raw)==item['sha256'],'result member hash mismatch')
        destination.mkdir(parents=True)
        z.extractall(destination)
    return destination


def verify_files(folder):
    manifest=c.read(folder/'EVIDENCE_MANIFEST.json')
    c.require(manifest['run_id']==c.RUN_ID,'result run identity')
    expected=set()
    for item in manifest['files']:
        name=item['path'];c.safe_relative(name)
        c.require(name not in expected,'duplicate evidence path')
        expected.add(name)
        p=folder/name
        c.require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(folder.resolve()),'evidence path')
        c.require(p.stat().st_size==item['bytes'] and c.file_sha(p)==item['sha256'],'evidence drift: '+name)
    observed={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    c.require(observed==expected|{'EVIDENCE_MANIFEST.json'},'unmanifested evidence')


def verify_runtime(folder, approved, policy, package_sha):
    lock=c.read(folder/'model-lock.json')
    raw=(folder/'model-manifest.json').read_bytes()
    digest=c.sha(raw)
    c.require(lock['full_manifest_digest']==digest and lock['manifest_sha256']==digest and lock['model_tag']==c.TAG,'model lock does not bind manifest')
    c.full_digest(digest)
    c.require(lock['approved_amendment_sha256']==c.APPROVED_SHA and lock['inference_started'] is False,'model lock approval or freeze order')
    if lock.get('registry_header_digest'):
        c.require(c.full_digest(lock['registry_header_digest'])==digest,'registry header does not bind manifest')
    manifest=c.strict(raw)
    blobs=c.read(folder/'model-blobs.json')
    observed={x['digest']:x['bytes'] for x in blobs['blobs']}
    expected={x['digest']:x['size'] for x in [manifest['config']]+manifest['layers']}
    c.require(observed==expected and blobs['manifest_sha256']==digest,'model blob lineage mismatch')
    runtime=c.read(folder/'runtime.json')
    for key in ('archive_sha256','binary_sha256','version'):
        c.require(runtime[key]==policy[key],'pinned runtime identity drift')
    for name,key in [('model-show.json','show_sha256'),('model-tags.json','tags_sha256'),('model-blobs.json','model_blobs_sha256'),('runtime-files.json','runtime_files_sha256'),('runtime-decoder.json','runtime_decoder_sha256')]:
        c.require(c.file_sha(folder/name)==runtime[key],'runtime capture hash drift')
    decoder=c.read(folder/'runtime-decoder.json')
    c.require(decoder['pinned_runtime_archive_verified'] is True and decoder['backend'] in ('system_tar_zstd','zstandard'),'runtime decoder capture')
    if decoder['backend']=='zstandard':
        c.require(decoder['version']=='0.25.0' and decoder['install_report_sha256']==c.file_sha(folder/'zstandard-install.json'),'pinned decompressor identity')
    binaries=[x for x in c.read(folder/'runtime-files.json')['files'] if x['path']=='bin/ollama']
    c.require(len(binaries)==1 and binaries[0]['sha256']==policy['binary_sha256'],'runtime inventory binary identity')
    c.require(runtime['model_digest']==digest and runtime['model_tag']==c.TAG and runtime['num_gpu']==0 and runtime['accelerator']=='none' and runtime['num_thread']==20 and runtime['context']==8192,'runtime execution profile')
    c.require(type(runtime['num_gpu']) is int and type(runtime['num_thread']) is int and type(runtime['context']) is int,'runtime numeric types')
    c.require(runtime['kv_cache_type']=='f16' and runtime['flash_attention'] is False,'runtime cache profile')
    tags=c.read(folder/'model-tags.json')
    c.require(c.full_digest(tags['digest'])==digest and tags['name']==c.TAG and tags['details']['quantization_level']=='Q4_K_M','tag capture identity')
    show=c.read(folder/'model-show.json')
    c.require({'completion','thinking'}<=set(show['capabilities']) and show['details']['family']=='qwen35' and show['details']['quantization_level']=='Q4_K_M','model capabilities/architecture')
    effective=c.read(folder/'effective-contract.json')
    expected_effective={'schema':'alice.mc10d.qwen.effective-execution-contract.v1','run_id':c.RUN_ID,'package_sha256':package_sha,'package_manifest_sha256':c.file_sha(c.BASE/'PACKAGE_MANIFEST.json'),'amendment_sha256':c.APPROVED_SHA,'owner_approval_sha256':c.APPROVAL_SHA,'tasks_sha256':c.TASK_SHA,'runtime_sha256':c.file_sha(folder/'runtime.json'),'model_lock_sha256':c.file_sha(folder/'model-lock.json'),'profile':approved['profile'],'pass_rule':approved['pass_rule'],'no_authority':c.NO_AUTHORITY}
    c.require(c.canonical(effective)==c.canonical(expected_effective),'effective contract drift')
    return digest,c.file_sha(folder/'effective-contract.json')


def analyze(folder, package_sha):
    verify_files(folder)
    infra_recovery.verify_evidence(folder)
    telemetry_snapshots.verify_evidence(folder)
    approved,tasks,policy=c.authority()
    for name in ('tasks.json','draft.json','approval.json','approved.json','runtime_policy.json','source_binding_policy.json'):
        c.require((folder/'authority'/name).read_bytes()==(c.BASE/'authority'/name).read_bytes(),'returned authority differs')
    c.require((folder/'package-manifest.json').read_bytes()==(c.BASE/'PACKAGE_MANIFEST.json').read_bytes(),'returned package manifest differs')
    run=c.read(folder/'run.json')
    c.require(run['run_id']==c.RUN_ID and run['package_sha256']==package_sha and run['amendment_sha256']==c.APPROVED_SHA,'returned run descriptor')
    scheduler=c.read(folder/'scheduler-status.json')
    c.require(scheduler['terminal'] is True and scheduler['run_id']==c.RUN_ID,'scheduler terminal receipt missing')
    submission=c.read(folder/'submission.json')
    c.require(submission['job_id']==scheduler['job_id'] and submission['job_id'].isdigit(),'job identity mismatch')
    c.require(submission['package_sha256']==package_sha,'submission package drift')
    contract_sha=None;model_digest=None
    if (folder/'effective-contract.json').exists():
        model_digest,contract_sha=verify_runtime(folder,approved,policy,package_sha)
        c.require(c.read(folder/'runtime.json')['job_id']==submission['job_id'],'runtime scheduler identity')
    records=[]
    for index,task in enumerate(tasks,1):
        td=folder/'tasks'/task['task_id']
        if not td.exists():continue
        c.require(contract_sha is not None,'task attempted before runtime identity freeze')
        intent=c.read(td/'attempt.json')
        c.require(intent['task_id']==task['task_id'] and type(intent['attempt_count']) is int and intent['attempt_count']==1 and intent['state']=='IN_FLIGHT','task attempt identity/count')
        expected_request=c.canonical(c.request(task,index,approved)).encode()
        c.require((td/'request.json').read_bytes()==expected_request,'request differs from frozen prompt/profile/seed')
        c.require(intent['request_sha256']==c.sha(expected_request) and intent['contract_sha256']==contract_sha and intent['model_digest']==model_digest,'task request lineage')
        raw=(td/'response.ndjson').read_bytes() if (td/'response.ndjson').exists() else b''
        if (td/'record.json').exists():
            recorded=c.read(td/'record.json')
            computed=c.classify(raw)
            for key in computed:
                c.require(c.canonical(recorded[key])==c.canonical(computed[key]),'task classification disagrees with raw stream')
            for key,value in intent.items():
                if key!='state':c.require(c.canonical(recorded[key])==c.canonical(value),'record intent differs')
            c.require(recorded['state']=='RECORDED' and recorded['stream_sha256']==c.sha(raw),'raw stream hash differs')
            records.append(recorded)
        else:
            records.append({**intent,'status':'INCOMPLETE','response':None,'error_code':'IN_FLIGHT_OUTCOME_UNRESOLVED'})
    scores=c.score(records,tasks)
    result=c.read(folder/'result.json') if (folder/'result.json').exists() else None
    qualified=False
    status='INCOMPLETE_OR_PREFLIGHT_STOP'
    if result is not None:
        c.require(result['run_id']==c.RUN_ID and result['model_tag']==c.TAG and result['package_sha256']==package_sha,'result lineage')
        c.require(result['effective_contract_sha256']==contract_sha,'result effective contract hash')
        c.require(c.canonical(result['scores'])==c.canonical(scores),'worker scores disagree with independently rebuilt scores')
        c.require(result['tasks_attempted']==len(records),'attempt counter mismatch')
        c.require(c.canonical(result['no_authority'])==c.canonical(c.NO_AUTHORITY),'unauthorized successor authority')
        c.require(result['failure_present']==(folder/'failure.json').exists(),'failure capture inconsistent')
        preflight=c.read(folder/'preflight.json') if (folder/'preflight.json').exists() else None
        preflight_ok=preflight is not None and preflight['status']=='PASS'
        c.require(result['runtime_preflight_passed'] is preflight_ok,'runtime preflight state inconsistent')
        if preflight is not None:
            c.require(preflight['contract_sha256']==contract_sha and preflight['cpu_only_verified'] is True,'preflight contract/CPU proof')
            loaded=c.read(folder/'model-loaded.json')
            c.require(preflight['model_loaded_sha256']==c.file_sha(folder/'model-loaded.json'),'loaded runtime hash')
            c.require(c.full_digest(loaded['digest'])==model_digest and type(loaded['size_vram']) is int and loaded['size_vram']==0 and type(loaded['context_length']) is int and loaded['context_length']==8192,'loaded model CPU identity')
            c.require((folder/'probe-request.json').read_bytes()==c.canonical(c.probe_request(approved,tasks)).encode(),'preflight probe request drift')
            attempt=c.read(folder/'probe-attempt.json')
            c.require(type(attempt['attempt_count']) is int and attempt['attempt_count']==1 and attempt['qualification_task'] is False and attempt['contract_sha256']==contract_sha and attempt['request_sha256']==c.file_sha(folder/'probe-request.json'),'probe attempt lineage')
            probe=c.decode_stream((folder/'probe-response.ndjson').read_bytes())
            c.require(type(probe.get('eval_count')) is int and 0<probe['eval_count']<=128,'probe budget')
            rebuilt=c.projection(probe,16,preflight['projection']['remaining_job_seconds'])
            c.require(c.canonical(rebuilt)==c.canonical(preflight['projection']),'throughput estimate drift')
            c.require(preflight_ok is rebuilt['fits'],'throughput gate result mismatch')
        qualified=preflight_ok and not result['failure_present'] and scores['qualification_passed']
        c.require(result['qualification_passed'] is qualified,'worker qualification flag inconsistent')
        status='PUBLIC_CALIBRATION_PASSED' if qualified else ('PUBLIC_CALIBRATION_FAILED' if not result['failure_present'] else 'INCOMPLETE_OR_PREFLIGHT_STOP')
        c.require(result['status']==status,'result terminal status inconsistent')
        if qualified:
            c.require((folder/'worker-finished.json').is_file(),'qualification finalization receipt missing')
    if (folder/'worker-finished.json').exists():
        finished=c.read(folder/'worker-finished.json')
        c.require(finished['result_sha256']==c.file_sha(folder/'result.json'),'finalized result identity')
        c.require(finished['application_exit_code']==(0 if qualified else 76),'application result exit inconsistent')
    publication=c.read(folder/'telemetry-publication.json') if (folder/'telemetry-publication.json').exists() else {}
    telemetry=c.read(folder/'telemetry-receipt.json') if (folder/'telemetry-receipt.json').exists() else {}
    recovered=c.read(folder/'telemetry-recovery.json') if (folder/'telemetry-recovery.json').exists() else {}
    if recovered:
        c.require(recovered['scheduler_terminal_verified'] is True and recovered['job_id']==submission['job_id'],'terminal telemetry recovery identity')
        c.require(recovered['result_sha256']==(c.file_sha(folder/'result.json') if result else None),'terminal telemetry source result')
    telemetry_ok=(publication.get('exit_code')==0 or recovered.get('exit_code')==0) and telemetry.get('status') in ('COMPLETED','FAILED') and telemetry.get('slurm_job_id')==submission['job_id']
    return {'schema':'alice.mc10d.qwen.verified-public-summary.v1','calibration_id':c.CALIBRATION_ID,'source_job_id':'575089','infrastructure_preflight_sha256':c.file_sha(folder/'infrastructure-preflight.json'),'terminal_telemetry_snapshot_id':publication.get('snapshot_id'),'terminal_telemetry_url':publication.get('ledger_url'),'run_id':c.RUN_ID,'job_id':submission['job_id'],'status':status,'model_tag':c.TAG,'model_digest':model_digest,'approved_amendment_sha256':c.APPROVED_SHA,'owner_approval_sha256':c.APPROVAL_SHA,'package_sha256':package_sha,'effective_contract_sha256':contract_sha,'evidence_manifest_sha256':c.file_sha(folder/'EVIDENCE_MANIFEST.json'),'raw_result_sha256':c.file_sha(folder/'result.json') if result is not None else None,'scores':scores,'qualification_passed':qualified,'calibration_only':True,'final_telemetry_published':telemetry_ok,'no_authority':c.NO_AUTHORITY,'raw_rationales_published':False,'private_payloads_published':False,'preserved_counts':{'replacements':63,'deferred':1,'candidate_pool':287},'frozen_main':c.MAIN,
            'infrastructure_policy_sha256':c.file_sha(c.BASE/'authority/infra_recovery.json'),
            'closed_parent_run_ids':[p['run_id'] for p in infra_recovery.policy()['parents']],
            'installed_publisher_sha256':infra_recovery.policy()['installed_helper_sha256'],
            'publisher_journals_sha256':c.file_sha(folder/'publisher-journals.zip') if (folder/'publisher-journals.zip').exists() else None}
