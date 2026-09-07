from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,zipfile

BASE=Path(__file__).resolve().parent
PACKAGE=BASE/'package/ALICE_MAGNOLIA_RUNTIME_ROUTE_v1.0.0'
OUT=BASE/'dist';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(PACKAGE/'collect_transport.py')=='385b835deaf2fa5283f8288fc88fea9ff5d4d0d63bbda5b0895bab23105b254d'
files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json']
(PACKAGE/'PACKAGE_MANIFEST.json').write_text(json.dumps({'files':[{'path':p.relative_to(PACKAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]},indent=2)+'\n')
files.append(PACKAGE/'PACKAGE_MANIFEST.json')
archive=OUT/(PACKAGE.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(files):
        info=zipfile.ZipInfo(PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix(),(2026,9,7,0,0,0));info.external_attr=0o100644<<16
        z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
text=(BASE/'launcher-template.ps1').read_text().replace('__ZIP_SHA256__',sha(archive).upper())
launcher=OUT/'Start-ALICEMagnoliaRuntimeRouteV100.ps1';launcher.write_bytes(text.replace('\r\n','\n').replace('\n','\r\n').encode())
with tempfile.TemporaryDirectory(prefix='alice-runtime-route-build-') as temp:
    root=Path(temp)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;z.extractall(root)
    extracted=root/PACKAGE.name
    subprocess.run([sys.executable,'-m','compileall','-q',str(extracted)],check=True)
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(extracted),'-p','selftest*.py','-v'],capture_output=True,text=True,timeout=30)
    (OUT/'BUILD_SELFTEST.log').write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0 and 'Ran 3 tests' in tests.stderr and '\nOK\n' in tests.stderr,tests.stderr
    code=text.split("$VerifyCode = @'\n",1)[1].split("\n'@",1)[0]
    subprocess.run([sys.executable,'-c',code,str(extracted)],check=True,capture_output=True)
    probe=subprocess.run([sys.executable,'-B','-c','import inspect_runtime; print(inspect_runtime.remote_script().decode())'],cwd=extracted,capture_output=True,check=True)
    shell=root/'remote-input.sh';shell.write_bytes(probe.stdout);subprocess.run(['bash','-n',str(shell)],check=True)
rules=json.loads((PACKAGE/'inspection-policy.json').read_text())
receipt={'schema':'alice.magnolia.runtime-route.build.v1','status':'READY_FOR_READ_ONLY_SURVEY',
    'package':archive.name,'package_sha256':sha(archive),'package_bytes':archive.stat().st_size,
    'launcher':launcher.name,'launcher_sha256':sha(launcher),'package_manifest_sha256':sha(PACKAGE/'PACKAGE_MANIFEST.json'),
    'source_result_zip_sha256':rules['source_result_zip_sha256'],'source_run_id':rules['source_run_id'],
    'commands':6,'cpu_runtime_objects':len(rules['runtime_elf_files']),'tests_passed':3,'fresh_zip_tested':True,
    'transport_source_reused_sha256':sha(PACKAGE/'collect_transport.py'),'literal_stdin_program_compiled':True,
    'read_only':True,'actual_magnolia_survey_observed':False,'new_execution_identity_prepared':False,
    'native_windows_execution_observed':False,'next_action':'READ_EXISTING_RUNTIME_AND_AVAILABLE_ROUTES'}
(OUT/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
