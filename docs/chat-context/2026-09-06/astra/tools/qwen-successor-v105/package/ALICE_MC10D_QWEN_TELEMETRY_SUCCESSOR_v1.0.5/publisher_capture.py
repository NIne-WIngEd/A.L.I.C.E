"""Preserve actual publisher output and its raw Git journals with the owning run."""
import base64
import os
from pathlib import Path
import re
import subprocess
import zipfile

import contract as c
import infra_recovery


def stream(raw):
    raw = raw or b''
    return {'bytes': len(raw), 'sha256': c.sha(raw), 'base64': base64.b64encode(raw).decode()}


def invoke(run, item, folder):
    helper = c.ROOT / 'bin/rayan-telemetry-push.sh'
    argv = [str(helper), item['snapshot_id'], 'MC10D_QWEN_PUBLIC_CALIBRATION_V1', item['status'], item['slurm_job_id'],
            str(folder / 'application.log'), str(folder / 'manifest.json'), str(folder / 'receipt.json')]
    record = {'snapshot_id': item['snapshot_id'], 'argv': argv, 'started_at': c.now(), 'exit_code': None}
    stdout = stderr = b''; code = 75; markers = []
    try:
        infra_recovery.checked_file(helper, infra_recovery.policy()['installed_helper_sha256'])
        record['helper_sha256'] = c.file_sha(helper)
        result = subprocess.run(argv, capture_output=True, timeout=90)
        stdout, stderr = result.stdout, result.stderr
        code = result.returncode
        record.update(status='RETURNED', exit_code=code)
        matches = re.findall(rb'(?m)^TELEMETRY_RECEIPT_DIR=([^\r\n]+)$', stdout)
        if len(matches) == 1:
            journal = Path(matches[0].decode('utf-8'))
            expected = c.ROOT / 'telemetry/publisher-receipts' / item['snapshot_id']
            c.require(journal.parent == expected and re.fullmatch('[0-9a-f]{32}', journal.name), 'Publisher journal location differs')
            report = c.read(journal / 'result.json')
            c.require(report['run_id'] == item['snapshot_id'] and report['exit_code'] == result.returncode, 'Publisher result identity differs')
            if code == 0: c.require(report['published'] is True, 'Publisher did not verify publication')
            record.update(journal=journal.as_posix(), result_sha256=c.file_sha(journal / 'result.json'))
        elif code == 0:
            raise c.Stop('Successful helper output lacks one raw publisher receipt')
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.output or b'', exc.stderr or b''
        record['status'] = 'TIMEOUT'; markers = ['TimeoutExpired']; code = 75
    except (OSError, ValueError, c.Stop) as exc:
        record.update(status='BOUNDARY_STOP', error_class=type(exc).__name__, message=str(exc))
        markers = [type(exc).__name__]; code = 75
    finally:
        record.update(stdout=stream(stdout), stderr=stream(stderr), effective_exit_code=code, finished_at=c.now())
        with (Path(run) / 'publisher-invocations.jsonl').open('ab') as out:
            out.write(c.canonical(record).encode()); out.flush(); os.fsync(out.fileno())
    return code, markers


def evidence_paths(run):
    run = Path(run); invocations = run / 'publisher-invocations.jsonl'
    if not invocations.is_file(): return {}
    records = [c.strict(x) for x in invocations.read_bytes().splitlines()]
    files = {}
    for snapshot in sorted({r['snapshot_id'] for r in records}):
        c.require(re.fullmatch(re.escape(c.RUN_ID) + r'-t-[0-9a-f]{16}', snapshot), 'Foreign publisher snapshot')
        root = c.ROOT / 'telemetry/publisher-receipts' / snapshot
        if not root.exists(): continue
        c.require(root.is_dir() and not root.is_symlink(), 'Linked publisher root')
        for journal in sorted(root.iterdir()):
            c.require(journal.is_dir() and not journal.is_symlink() and re.fullmatch('[0-9a-f]{32}', journal.name), 'Unexpected publisher journal directory')
            for p in sorted(journal.iterdir()):
                if not re.fullmatch(r'(result\.json|git-[0-9]+\.(json|stdout\.bin|stderr\.bin))', p.name): continue
                c.require(p.is_file() and not p.is_symlink(), 'Non-regular publisher journal member')
                files[snapshot + '/' + journal.name + '/' + p.name] = p
    c.require(len(files) <= 20000 and sum(p.stat().st_size for p in files.values()) <= 64 * 1024 * 1024, 'Publisher journal export exceeds explicit limit; originals retained')
    manifest = {'schema': 'alice.mc10d.qwen.publisher-journals.v1', 'run_id': c.RUN_ID,
                'files': [{'path': n, 'bytes': p.stat().st_size, 'sha256': c.file_sha(p)} for n, p in sorted(files.items())]}
    tmp = run / 'publisher-journals.zip.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, p in sorted(files.items()): z.writestr(n, p.read_bytes())
        z.writestr('MANIFEST.json', c.canonical(manifest))
    os.replace(tmp, run / 'publisher-journals.zip')
    return {'publisher-invocations.jsonl': invocations, 'publisher-journals.zip': run / 'publisher-journals.zip'}


def verify_evidence(folder):
    folder = Path(folder)
    if not (folder / 'publisher-invocations.jsonl').exists():
        c.require(not (folder / 'telemetry-publication.json').exists(), 'Publisher capture missing')
        return
    records = [c.strict(x) for x in (folder / 'publisher-invocations.jsonl').read_bytes().splitlines()]
    with zipfile.ZipFile(folder / 'publisher-journals.zip') as z:
        names = z.namelist()
        c.require(len(names) == len(set(names)) and len(names) <= 20001 and sum(i.file_size for i in z.infolist()) <= 68 * 1024 * 1024, 'Publisher journal archive bounds')
        manifest = c.strict(z.read('MANIFEST.json'))
        c.require(manifest['run_id'] == c.RUN_ID, 'Publisher journal archive identity')
        members = {x['path']: x for x in manifest['files']}
        c.require(len(members) == len(manifest['files']) and set(members) | {'MANIFEST.json'} == set(names), 'Publisher journal membership')
        for n, item in members.items():
            c.safe_relative(n); raw = z.read(n)
            c.require(len(raw) == item['bytes'] and c.sha(raw) == item['sha256'], 'Publisher journal member bytes differ')
            if re.search(r'/git-[0-9]+\.json$', n):
                command = c.strict(raw)
                for s in ('stdout', 'stderr'):
                    other = n[:-5] + '.' + s + '.bin'
                    c.require(other in members, 'Publisher raw stream absent')
                    data = z.read(other)
                    c.require(len(data) == command[s + '_bytes'] and c.sha(data) == command[s + '_sha256'], 'Publisher raw stream hash differs')
        for r in records:
            c.require(re.fullmatch(re.escape(c.RUN_ID) + r'-t-[0-9a-f]{16}', r['snapshot_id']), 'Publisher invocation run differs')
            for s in ('stdout', 'stderr'):
                raw = base64.b64decode(r[s]['base64'], validate=True)
                c.require(len(raw) == r[s]['bytes'] and c.sha(raw) == r[s]['sha256'], 'Publisher outer stream differs')
            if r.get('journal'):
                path = Path(r['journal']); expected = c.ROOT / 'telemetry/publisher-receipts' / r['snapshot_id']
                c.require(path.parent.as_posix() == expected.as_posix(), 'Publisher journal origin differs')
                name = r['snapshot_id'] + '/' + path.name + '/result.json'
                c.require(name in members and c.sha(z.read(name)) == r['result_sha256'], 'Publisher result capture differs')
                result = c.strict(z.read(name))
                c.require(result['run_id'] == r['snapshot_id'] and result['exit_code'] == r['exit_code'], 'Publisher receipt execution differs')
                if r['effective_exit_code'] == 0:
                    c.require(result['published'] is True and r.get('helper_sha256') == infra_recovery.policy()['installed_helper_sha256'], 'Unverified publisher success')
            else: c.require(r['effective_exit_code'] != 0, 'Publisher success without raw receipt')
    if (folder / 'telemetry-publication.json').exists():
        pub = c.read(folder / 'telemetry-publication.json')
        c.require(records and records[-1]['snapshot_id'] == pub['snapshot_id'] and records[-1]['effective_exit_code'] == pub['exit_code'], 'Latest telemetry disagrees with captured invocation')
