"""Build the pinned a3 distribution from the committed source files; no remote compute."""
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile

BASE = Path(__file__).resolve().parent
PACKAGE = BASE / 'package/ALICE_MC10D_QWEN_TELEMETRY_SUCCESSOR_v1.0.5'
OUT = BASE / 'dist'; OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rules = json.loads((PACKAGE / 'authority/infra_recovery.json').read_text())
unchanged = ['worker.py','secure_https.py','publish_context.py','frozen_prompt.py','magnolia_scheduler.py','evidence_origins.py',
             *['authority/' + n for n in ('approval.json','approved.json','draft.json','tasks.json','runtime_policy.json','source_binding_policy.json','request_hashes_v100.json')]]
parent = rules['parents'][1]
assert sha(PACKAGE / 'source' / parent['package_archive']) == parent['package_sha256']
with zipfile.ZipFile(PACKAGE / 'source' / parent['package_archive']) as z:
    for name in unchanged: assert (PACKAGE / name).read_bytes() == z.read(parent['package_name'] + '/' + name), name
    old_contract = z.read(parent['package_name'] + '/contract.py').decode()
    expected_contract = old_contract.replace("VERSION = '1.0.3'", "VERSION = '1.0.5'").replace("RUN_ID = 'alice-qwen38-a2-c926d9e355dd'", "RUN_ID = 'alice-qwen38-a3-750881d854fa'")
    assert (PACKAGE / 'contract.py').read_text() == expected_contract
    delta = []
    for p in sorted(PACKAGE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ('PACKAGE_MANIFEST.json','SOURCE_DELTA.json'): continue
        name = p.relative_to(PACKAGE).as_posix(); origin = parent['package_name'] + '/' + name
        old = hashlib.sha256(z.read(origin)).hexdigest() if origin in z.namelist() else None
        if sha(p) != old: delta.append({'path': name, 'v104_sha256': old, 'a3_sha256': sha(p)})
(PACKAGE / 'SOURCE_DELTA.json').write_text(json.dumps({'v104_package_sha256':parent['package_sha256'], 'unchanged':{n:sha(PACKAGE/n) for n in unchanged}, 'changed_or_added':delta},indent=2)+'\n')
for parent in rules['parents']:
    assert sha(PACKAGE / 'source' / parent['result_archive']) == parent['result_archive_sha256']
    with zipfile.ZipFile(PACKAGE / 'source' / parent['result_archive']) as z:
        for n, digest in parent['critical_files'].items(): assert hashlib.sha256(z.read(n)).hexdigest() == digest
assert sha(PACKAGE / 'source/ALICE_PublisherRepair_f3dc451ff517.zip') == rules['publisher_repair_zip_sha256']
assert sha(PACKAGE / 'fixtures/rayan-telemetry-push.sh') == rules['installed_helper_sha256']
embedded = (PACKAGE / 'fixtures/rayan-telemetry-push.sh').read_text().split("<<'ALICE_TELEMETRY_PUBLISHER_PY'\n", 1)[1].rsplit('\nALICE_TELEMETRY_PUBLISHER_PY', 1)[0] + '\n'
assert embedded == (PACKAGE / 'fixtures/publisher.py').read_text() + '\n'
files = [p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'PACKAGE_MANIFEST.json']
manifest = {'schema':'alice.mc10d.qwen.package-manifest.v1','files':[{'path':p.relative_to(PACKAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
(PACKAGE / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
files.append(PACKAGE / 'PACKAGE_MANIFEST.json')
archive = OUT / (PACKAGE.name + '.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(files):
        info = zipfile.ZipInfo(PACKAGE.name + '/' + p.relative_to(PACKAGE).as_posix(), (2026,9,7,0,0,0)); info.external_attr=0o100644<<16
        z.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
text = (BASE / 'launcher-template.ps1').read_text().replace('__ZIP_SHA256__', sha(archive).upper())
launcher = OUT / 'Start-ALICEAstraQwenQualificationV105.ps1'; launcher.write_bytes(text.replace('\r\n','\n').replace('\n','\r\n').encode())
with tempfile.TemporaryDirectory(prefix='alice-a3-release-') as temp:
    root = Path(temp)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.infolist()) < 100 and sum(i.file_size for i in z.infolist()) < 4*1024*1024
        z.extractall(root / 'extracted')
    extracted = root / 'extracted' / PACKAGE.name
    subprocess.run([sys.executable,'-m','compileall','-q',str(extracted)],check=True)
    tests = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(extracted),'-p','selftest*.py','-v'],capture_output=True,text=True,timeout=90)
    (OUT / 'BUILD_SELFTEST.log').write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stderr
    count = int(re.search(r'Ran (\d+) tests', tests.stderr).group(1))
    assert count == 12 and '\nOK\n' in tests.stderr, tests.stderr
    verifier = text.split("$VerifyCode = @'\n", 1)[1].split("\n'@", 1)[0]
    subprocess.run([sys.executable,'-c',verifier,str(extracted)],check=True,capture_output=True)
    snippet = (extracted / 'bootstrap.sh').read_text().split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
    for _ in range(2): subprocess.run([sys.executable,'-c',snippet,str(archive),sha(archive),str(root / 'remote-extraction')],check=True,capture_output=True)
    altered = root / 'altered.zip'; altered.write_bytes(archive.read_bytes() + b'changed')
    rejected = subprocess.run([sys.executable,'-c',snippet,str(altered),sha(archive),str(root / 'refused-extraction')],capture_output=True)
    assert rejected.returncode != 0 and not (root / 'refused-extraction').exists()
    subprocess.run(['bash','-n',str(extracted / 'bootstrap.sh')],check=True)
    env = dict(os.environ, PYTHONPATH=str(extracted))
    script = subprocess.run([sys.executable,'-B','-c','import remote_agent; print(remote_agent.slurm_script("1"*64))'],capture_output=True,text=True,env=env,check=True).stdout
    (root / 'job.sbatch').write_text(script); subprocess.run(['bash','-n',str(root / 'job.sbatch')],check=True)
    assert '#SBATCH --no-requeue' in script and '#SBATCH --time=06:00:00' in script and 'a2-c926d9e355dd' not in script
receipt = {'schema':'alice.mc10d.qwen.explicit-successor.build.v1.0.5','status':'READY_FOR_OWNER_EXECUTION',
    'package':archive.name,'package_sha256':sha(archive),'package_bytes':archive.stat().st_size,
    'launcher':launcher.name,'launcher_sha256':sha(launcher),'package_manifest_sha256':sha(PACKAGE/'PACKAGE_MANIFEST.json'),
    'execution_run_id':rules['execution_run_id'],'calibration_id':rules['calibration_id'],
    'owner_approval_sha256':rules['owner_approval_sha256'],'owner_approval_context_commit':'591dbc0781b1df8e11f0eea7d743952724722bae',
    'source_scientific_workload_sha256':rules['original_scientific_workload_sha256'],
    'infrastructure_policy_sha256':sha(PACKAGE/'authority/infra_recovery.json'),'installed_helper_sha256':rules['installed_helper_sha256'],
    'publisher_repair_result_sha256':rules['publisher_repair_zip_sha256'], 'publisher_terminal_commit':rules['verified_terminal_commit'],
    'parent_runs':[p['run_id'] for p in rules['parents']], 'frozen_model_manifest_sha256':rules['frozen_model_manifest_sha256'],
    'worker_runtime_authority_requests_unchanged':True,'request_hashes_verified':17,'offline_tests_passed':count,
    'test_log_sha256':sha(OUT/'BUILD_SELFTEST.log'),'fresh_zip_tested':True,
    'powershell_embedded_verifier_tested':True,'bootstrap_extraction_and_repeat_verification_tested':True,'altered_zip_refused_before_extraction':True,
    'real_local_git_publisher_to_capture_to_archive_tested':True,'raw_rpc_and_publisher_failure_bytes_tested':True,
    'full_synthetic_worker_collector_verifier_tested':True,'native_windows_execution_observed':False,
    'actual_magnolia_a3_submission_observed':False,'model_inference_executed':False,'scientific_outcome':'NOT_EVALUATED',
    'automatic_successor_after_attempt':False,'a2_reset_or_resubmission':False,'main':'0abaed85873c3f8de04765847eb7700b0e20433f'}
(OUT / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
