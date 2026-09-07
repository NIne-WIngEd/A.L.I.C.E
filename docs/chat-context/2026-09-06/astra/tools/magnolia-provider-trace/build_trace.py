"""Deterministic diagnostic build; no remote observation or compute."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent
PKG=ROOT/'package/ALICE_MAGNOLIA_PROVIDER_TRACE_v1.0.0'
DIST=ROOT/'dist'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    DIST.mkdir(exist_ok=True)
    files=[p for p in sorted(PKG.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json']
    manifest={'schema':'alice.provider-trace.package-manifest.v1','files':[{'path':p.relative_to(PKG).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (PKG/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    files.append(PKG/'PACKAGE_MANIFEST.json')
    archive=DIST/(PKG.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files):
            info=zipfile.ZipInfo(PKG.name+'/'+p.relative_to(PKG).as_posix(),(2026,9,7,0,0,0))
            info.create_system=3;info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    launcher=DIST/'Start-ALICEMagnoliaProviderTraceV100.ps1'
    template=(ROOT/'Start-ALICEMagnoliaProviderTraceV100.template.ps1').read_text()
    assert "$VerifyCode" not in template and 'collect_provider_trace.py' in template
    launcher.write_bytes(template.replace('@ZIP_SHA@',sha(archive).upper()).replace('\r\n','\n').replace('\n','\r\n').encode())
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;z.extractall(tmp)
        clean=Path(tmp)/PKG.name
        checked=subprocess.run([sys.executable,'-B',str(clean/'collect_provider_trace.py'),'--self-check'],capture_output=True,text=True,timeout=30)
        assert checked.returncode==0,checked.stderr
        tested=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(clean),'-p','selftest_provider_trace.py','-v'],capture_output=True,text=True,timeout=90)
        (DIST/'BUILD_SELFTEST.log').write_text(checked.stdout+tested.stdout+tested.stderr,encoding='utf-8')
        assert tested.returncode==0,tested.stderr
        count=int(re.search(r'Ran (\d+) tests',tested.stderr).group(1));assert count==9
        sys.path.insert(0,str(clean));import collect_provider_trace
        policy=json.loads((clean/'trace_policy.json').read_bytes());script=collect_provider_trace.remote_script(policy)
        checked=subprocess.run(['bash','-n'],input=script,capture_output=True,timeout=15)
        assert checked.returncode==0,checked.stderr
        # The emitted Python body is compiled by remote_script itself.
    receipt={'schema':'alice.magnolia.provider-trace-build.v1','archive':archive.name,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
      'launcher':launcher.name,'launcher_sha256':sha(launcher),'package_manifest_sha256':sha(PKG/'PACKAGE_MANIFEST.json'),
      'tests_passed':count,'test_log_sha256':sha(DIST/'BUILD_SELFTEST.log'),'fresh_zip_tested':True,'exact_self_check_command_tested':True,
      'literal_remote_python_compiled':True,'literal_remote_bash_syntax_checked':True,'native_windows_execution':False,'actual_magnolia_observation':False,
      'commands_recorded':len(policy['commands']),'scheduler_command_timeout_seconds':20,'ssh_timeout_seconds':300,
      'raw_scheduler_streams_preserved':True,'local_and_remote_before_after_inventory':True,'new_qwen_qualification_package':False,
      'existing_qwen_release_modified':False,'compute_jobs_submitted':0,'remote_mutations':False,'telemetry_publication':False,
      'model_inference':False,'execution_authority_granted':False,'context_parent':'c06da32bca63c9c1334594e4aa146c08241f66de'}
    (DIST/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
