"""Deterministic public release build and fresh-extraction verification; no remote compute."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'package/ALICE_MC10D_QWEN_PROVIDER_REVISION_v1.0.4'
OUTPUT=ROOT/'dist'
OUTPUT.mkdir(parents=True,exist_ok=True)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source=PACKAGE/'source/ALICE_QWEN_PUBLIC_RESULT_c926d9e355dd.zip'
assert sha(source)=='c926d9e355dd6ec83ad18fd78f35265c664e5ea6f317871c20d7af6b0f8d8f5f'
with zipfile.ZipFile(source) as prior:
    for name in ('approval.json','approved.json','draft.json','runtime_policy.json','source_binding_policy.json','tasks.json'):
        assert (PACKAGE/'authority'/name).read_bytes()==prior.read('authority/'+name),name
    previous=json.loads(prior.read('package-manifest.json'))
    prompt_hash=next(item['sha256'] for item in previous['files'] if item['path']=='frozen_prompt.py')
    assert sha(PACKAGE/'frozen_prompt.py')==prompt_hash


# The repair is infrastructure-only. Verify original scientific authority and
# every unchanged v102 execution component against embedded immutable sources.
rules=json.loads((PACKAGE/'authority/revision_policy.json').read_bytes())
trace=PACKAGE/'source/ALICE_QWEN_TRACE_05d0a01649cb.zip'
assert sha(trace)==rules['trace_zip_sha256']
with zipfile.ZipFile(trace) as z:
    assert set(z.namelist())=={'report.json','DIAGNOSTIC_MANIFEST.json'}
    raw=z.read('report.json')
    assert hashlib.sha256(raw).hexdigest()==rules['trace_report_sha256']
    diagnostic=json.loads(raw)
    assert diagnostic['read_only'] and diagnostic['local_controller_states_unchanged']
    assert not diagnostic['compute_job_submitted_by_diagnostic'] and not diagnostic['inference_requested_by_diagnostic']
    assert diagnostic['remote']['observed_at']==rules['trace_observed_at']
    assert diagnostic['local_state_before']['current']['sha256']==rules['old_local_state_sha256']
old_zip=PACKAGE/'source'/(rules['old_package_name']+'.zip')
assert sha(old_zip)==rules['old_package_sha256']
unchanged=('worker.py','secure_https.py','telemetry_snapshots.py','publish_context.py',
           'frozen_prompt.py','authority/request_hashes_v100.json',
           'authority/approval.json','authority/approved.json','authority/draft.json',
           'authority/tasks.json','authority/runtime_policy.json','authority/source_binding_policy.json')
with zipfile.ZipFile(old_zip) as z:
    assert hashlib.sha256(z.read(rules['old_package_name']+'/PACKAGE_MANIFEST.json')).hexdigest()==rules['old_package_manifest_sha256']
    for name in unchanged:
        assert (PACKAGE/name).read_bytes()==z.read(rules['old_package_name']+'/'+name),name
original=PACKAGE/'source/ALICE_MC10D_QWEN_PUBLIC_QUALIFICATION_v1.0.0.zip'
assert sha(original)=='bc7c76c29a4f0fcfbd4567a610737ec5bb2052e7165677562959ac4dca7c959e'

provider_trace=PACKAGE/'source/ALICE_MAGNOLIA_PROVIDER_TRACE_15f4edccafa3.zip'
assert sha(provider_trace)==rules['provider_trace_zip_sha256']
with zipfile.ZipFile(provider_trace) as z:
    doc=json.loads(z.read('report.json'))
    em=json.loads(z.read('EVIDENCE_MANIFEST.json'))
    for item in em['files']:
        raw=z.read(item['path']); assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    assert doc['local_inventories_equal'] and doc['remote']['inventories_equal']
    assert doc['remote']['observed_at']==rules['provider_trace_observed_at']
    assert doc['local_inventory_after']['files']['tools/alice-astra/qwen-fallback-a2/controller-state.json']['sha256']==rules['old_local_state_sha256']
    assert doc['remote']['inventory_after']['files']['runs/'+rules['execution_run_id']+'/run.json']['sha256']==rules['old_descriptor_sha256']

files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json']
manifest={'schema':'alice.mc10d.qwen.package-manifest.v1','files':[{'path':p.relative_to(PACKAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
(PACKAGE/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
files.append(PACKAGE/'PACKAGE_MANIFEST.json')
archive=OUTPUT/(PACKAGE.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(files):
        info=zipfile.ZipInfo(PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix(),date_time=(2026,9,7,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
        z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
launcher=OUTPUT/'Start-ALICEAstraQwenQualificationV104.ps1'
text=(ROOT/'Start-ALICEAstraQwenQualificationV104.template.ps1').read_text(encoding='utf-8').replace('__ZIP_SHA256__',sha(archive).upper())
launcher.write_bytes(text.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
with tempfile.TemporaryDirectory(prefix='alice-qwen-release-') as temp:
    root=Path(temp)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.infolist())<100 and sum(i.file_size for i in z.infolist())<4*1024*1024
        z.extractall(root/'extracted')
    package=root/'extracted'/PACKAGE.name
    compile_result=subprocess.run([sys.executable,'-m','compileall','-q',str(package)],capture_output=True,text=True)
    assert compile_result.returncode==0,compile_result.stderr
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(package),'-p','selftest*.py','-v'],capture_output=True,text=True,timeout=180)
    (OUTPUT/'BUILD_SELFTEST.log').write_text(tests.stdout+tests.stderr,encoding='utf-8',newline='\n')
    assert tests.returncode==0,tests.stderr
    count=int(re.search(r'Ran (\d+) tests',tests.stderr).group(1))
    assert count==66 and '\nOK\n' in tests.stderr,tests.stderr
    # Test the exact literal Python snippet carried in the PowerShell launcher.
    ps_snippet=text.split("$VerifyCode = @'\n",1)[1].split("\n'@",1)[0]
    checked=subprocess.run([sys.executable,'-c',ps_snippet,str(package)],capture_output=True,text=True)
    assert checked.returncode==0,checked.stderr
    # Test the SSH bootstrap's literal Python hash/extraction code, including repeat verification.
    shell=(package/'bootstrap.sh').read_text()
    boot_snippet=shell.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
    bootroot=root/'remote-extraction'
    for _ in range(2):
        checked=subprocess.run([sys.executable,'-c',boot_snippet,str(archive),sha(archive),str(bootroot)],capture_output=True,text=True)
        assert checked.returncode==0,checked.stderr
    if shutil.which('bash'):
        subprocess.run(['bash','-n',str(package/'bootstrap.sh')],check=True)
        env=os.environ.copy();env['PYTHONPATH']=str(package)
        generated=subprocess.run([sys.executable,'-c','import remote_agent; print(remote_agent.slurm_script("1"*64))'],capture_output=True,text=True,env=env,check=True)
        job=root/'job.sbatch';job.write_text(generated.stdout,encoding='utf-8',newline='\n')
        subprocess.run(['bash','-n',str(job)],check=True)
    receipt={'schema':'alice.mc10d.qwen.public-package-build-receipt.v1','package':archive.name,'package_sha256':sha(archive),'package_bytes':archive.stat().st_size,'launcher':launcher.name,'launcher_sha256':sha(launcher),'package_manifest_sha256':sha(PACKAGE/'PACKAGE_MANIFEST.json'),'files_in_zip':len(files),'owner_approval_sha256':sha(PACKAGE/'authority/approval.json'),'approved_amendment_sha256':sha(PACKAGE/'authority/approved.json'),'owner_approval_context_commit':'591dbc0781b1df8e11f0eea7d743952724722bae','python_version':sys.version,'offline_tests':count,'offline_tests_passed':True,'build_test_log_sha256':sha(OUTPUT/'BUILD_SELFTEST.log'),'fresh_zip_extraction_tested':True,'compile_gate_passed':True,'powershell_embedded_verifier_tested':True,'ssh_bootstrap_python_tested_twice':True,'bash_syntax_checked':shutil.which('bash') is not None,'native_powershell_executed':False,'actual_magnolia_ssh_executed':False,'actual_model_digest_resolved':False,'actual_qwen_inference_executed':False,'remote_compute_jobs_submitted':0,'synthetic_tests_are_qualification_evidence':False,'frozen_main':'0abaed85873c3f8de04765847eb7700b0e20433f'}
    receipt.update({'schema':'alice.mc10d.qwen.pre-submission-provider-revision-build.v1.0.4','execution_run_id':'alice-qwen38-a2-c926d9e355dd','calibration_id':'alice-qwen38-a1-8e7a384a745496f4','source_job_id':'575089','source_result_zip_sha256':'c926d9e355dd6ec83ad18fd78f35265c664e5ea6f317871c20d7af6b0f8d8f5f','scientific_authority_bytes_unchanged':True,'original_probe_and_16_request_hashes_verified':True,'immutable_ledger_lifecycle_tested':True,'pre_submission_tls_failure_blocks_sbatch_tested':True,'source_a1_modified':False,'existing_a2_revision_executed_on_magnolia':False})
    receipt.update({'revision_id':rules['revision_id'],'superseded_package_sha256':rules['old_package_sha256'],
                    'host_trace_zip_sha256':rules['trace_zip_sha256'],'host_trace_report_sha256':rules['trace_report_sha256'],
                    'actual_host_observation_at':rules['trace_observed_at'],'diagnostic_native_windows_tests_reported_passed':6,
                    'real_original_collector_exercised':True,'real_package_run_origin_boundary_exercised':True,
                    'five_interrupted_revision_write_positions_tested':True,'lost_revision_acknowledgement_then_one_job_tested':True,
                    'old_launcher_refuses_revised_descriptor_tested':True,'new_execution_identity_created':False,
                    'same_existing_a2_execution':True,'v102_unchanged_files':{name:sha(PACKAGE/name) for name in unchanged},
                    'local_state_fixture_scope':'Exact owner-returned state and unfinished v103 intent from provider trace',
                    'live_scheduler_and_tls_scope':'Actual uploaded diagnostic observation; revision and job behavior tested offline; live v104 pending'})
    receipt.update({'provider_trace_zip_sha256':rules['provider_trace_zip_sha256'], 'provider_observed_at':rules['provider_trace_observed_at'], 'actual_provider_receipts_replayed':True, 'prior_v103_intent_preserved_and_bound':True, 'shared_scheduler_adapter':True, 'raw_scheduler_journal_before_assertions':True})
    (OUTPUT/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt,indent=2))
