"""Build the transport repair while preserving the exact approved workload ZIP."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,tempfile,zipfile

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'package/ALICE_MC10D_QWEN_WINDOWS_TRANSPORT_REPAIR_v1.0.1'
OUTPUT=ROOT/'dist';OUTPUT.mkdir(exist_ok=True)
WORKLOAD=PACKAGE/'workload/ALICE_MC10D_QWEN_PUBLIC_QUALIFICATION_v1.0.0.zip'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(WORKLOAD)=='bc7c76c29a4f0fcfbd4567a610737ec5bb2052e7165677562959ac4dca7c959e'
files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json']
manifest={'schema':'alice.mc10d.qwen.windows-transport-package.v1.0.1','files':[{'path':p.relative_to(PACKAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
(PACKAGE/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
files.append(PACKAGE/'PACKAGE_MANIFEST.json')
archive=OUTPUT/(PACKAGE.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(files):
        info=zipfile.ZipInfo(PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix(),date_time=(2026,9,6,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
        z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
launcher=OUTPUT/'Start-ALICEAstraQwenQualificationV101.ps1'
text=(ROOT/'Start-ALICEAstraQwenQualificationV101.template.ps1').read_text().replace('__ZIP_SHA256__',sha(archive).upper())
launcher.write_bytes(text.replace('\r\n','\n').replace('\n','\r\n').encode())
with tempfile.TemporaryDirectory(prefix='alice-v101-release-') as temp:
    destination=Path(temp)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        z.extractall(destination)
    package=destination/PACKAGE.name
    result=subprocess.run([sys.executable,'-m','compileall','-q',str(package)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    tests=subprocess.run([sys.executable,str(package/'selftest_transport.py')],capture_output=True,text=True,timeout=90)
    (OUTPUT/'BUILD_SELFTEST.log').write_text(tests.stdout+tests.stderr,encoding='utf-8',newline='\n')
    assert tests.returncode==0,tests.stderr
    assert 'Ran 11 tests' in tests.stderr and '\nOK\n' in tests.stderr,tests.stderr
    snippet=text.split("$VerifyCode = @'\n",1)[1].split("\n'@",1)[0]
    verify=subprocess.run([sys.executable,'-c',snippet,str(package)],capture_output=True,text=True)
    assert verify.returncode==0,verify.stderr
    env=__import__('os').environ.copy();env['PYTHONPATH']=str(package)
    verify=subprocess.run([sys.executable,'-c','from pathlib import Path; import runner; p,a=runner.prepare_workload(Path(__import__("sys").argv[1])); import compileall; assert compileall.compile_dir(str(p),quiet=1) ',str(destination/'legacy-compile')],capture_output=True,text=True,env=env)
    assert verify.returncode==0,verify.stderr
receipt={'schema':'alice.mc10d.qwen.transport-repair-build.v1.0.1','package':archive.name,'package_sha256':sha(archive),'package_bytes':archive.stat().st_size,'launcher':launcher.name,'launcher_sha256':sha(launcher),'package_manifest_sha256':sha(PACKAGE/'PACKAGE_MANIFEST.json'),'workload_sha256':sha(WORKLOAD),'workload_manifest_sha256':'b1d94af265eaac40d29314481358aa0f2a37b072bfa8c5542409fd641387175c','run_id':'alice-qwen38-a1-8e7a384a745496f4','files_in_repair_zip':len(files),'repair_tests_passed':11,'fresh_zip_extraction_tested':True,'repair_and_frozen_workload_compiled':True,'powershell_embedded_verifier_tested':True,'actual_posix_shell_regression_test_passed':True,'native_windows_v100_tests_reported_passed':35,'native_windows_v100_elapsed_seconds':69.568,'source_terminal_sha256':'78b19e9e4f53a977547b639b839ba9988628eade5d970984f7961cfa72fa24ff','source_terminal_qwen_exit':74,'source_terminal_job_submission_reached':False,'native_windows_v101_executed':False,'actual_magnolia_transfer_after_repair_verified':False,'new_model_inference_executed':False,'frozen_workload_bytes_changed':False,'controller_run_identity_changed':False,'scientific_profile_changed':False,'python_version':sys.version,'test_log_sha256':sha(OUTPUT/'BUILD_SELFTEST.log')}
(OUTPUT/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt,indent=2))
