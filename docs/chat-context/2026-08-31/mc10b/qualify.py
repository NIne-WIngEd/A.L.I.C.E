#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent

def req(c: bool, m: str) -> None:
    if not c:
        raise RuntimeError(m)

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest().upper()

def main() -> int:
    checks=[]
    def ok(n): checks.append(n+'=true')
    req(sys.version_info >= (3,11), f'Python 3.11+ required, running {sys.version.split()[0]}')
    req(not any(p.is_symlink() for p in ROOT.iterdir()), 'package symlinks forbidden')
    req(not any(p.is_dir() for p in ROOT.iterdir()), 'package subdirectories forbidden')
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    req(manifest.get('artifact_id')=='alice.MC10B.full-einf-frontier.execution-package.v1.1.0','package manifest id')
    expected=set(manifest['expected_package_files'])
    actual={p.name for p in ROOT.iterdir() if p.is_file()}
    req(actual==expected, f'exact package file set mismatch expected={sorted(expected)} actual={sorted(actual)}')
    ok('exact_package_file_set')

    sums={}
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        a=line.split('  ',1);req(len(a)==2,'SHA256SUMS format')
        req(a[1] not in sums,'duplicate SHA256SUMS entry '+a[1]);sums[a[1]]=a[0].upper()
    req(set(sums)==expected-{'SHA256SUMS.txt'},'SHA256SUMS exact file set')
    for n,h in sums.items(): req(sha(ROOT/n)==h,'SHA mismatch '+n)
    for n,h in manifest['frozen_artifact_sha256'].items(): req(sums.get(n)==h.upper(),'frozen artifact drift '+n)
    ok('sha256_and_frozen_bytes')

    for p in sorted(ROOT.glob('*.py')):
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
    ok('python_syntax_in_memory')

    for p in sorted(ROOT.glob('*.json')):
        json.loads(p.read_text(encoding='utf-8'))
    for p in sorted(ROOT.glob('*.jsonl')):
        for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
            if line.strip():
                obj=json.loads(line); req(isinstance(obj,dict),f'{p.name} line {i} not object')
    ok('json_and_jsonl_parse')

    ps=(ROOT/'Start-ALICEMC10BFullEInfGeneration-v1.1.0.ps1').read_text(encoding='utf-8')
    req('[ValidateRange(120,600)]' in ps,'PowerShell generation window range')
    req('mc10b_full_controller.py' in ps and ps.count('& $Python @ArgsList')==1,'PowerShell single controller delegation')
    req('mc10b-private-input.bin' not in ps,'stale private blob literal in PowerShell')
    req(ps.count('{')==ps.count('}') and ps.count('(')==ps.count(')'),'PowerShell delimiter sanity')
    ok('powershell_static_contract')

    env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
    proc=subprocess.run([sys.executable,str(ROOT/'selftest_mc10b_full_v110.py')],cwd=str(ROOT),env=env,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=180)
    print(proc.stdout.rstrip())
    req(proc.returncode==0,'selftest return code')
    req('selftest_check_count=16' in proc.stdout,'selftest expected check count')
    ok('deep_regression_selftest_16')

    req(not (ROOT/'__pycache__').exists(),'qualification created __pycache__')
    req(not list(ROOT.rglob('*.pyc')),'qualification created pyc')
    ok('no_generated_cache_artifacts')

    print('\n'.join(checks))
    print(f'package_qualification_check_count={len(checks)}')
    print('ALICE_MC10B_FULL_EINF_FRONTIER_V1_1_0_PACKAGE_QUALIFIED=true')
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print('PACKAGE_QUALIFICATION_FAILED='+repr(e))
        raise
