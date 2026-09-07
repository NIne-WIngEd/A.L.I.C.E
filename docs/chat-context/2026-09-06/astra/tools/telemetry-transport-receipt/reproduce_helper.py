"""Exercise the uploaded helper against disposable local Git repositories."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'source/rayan-telemetry-push.sh'
GIT = shutil.which('git')

def run(argv, **kw):
    p = subprocess.run(argv, capture_output=True, **kw)
    if p.returncode:
        raise RuntimeError((argv, p.returncode, p.stderr.decode(errors='replace')))
    return p

def main():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == 'ba4482728c09313aede5031e682f68e8007c79c9cfa084d3da4b11aa984d0830'
    findings = []
    with tempfile.TemporaryDirectory(prefix='alice-helper-repro-') as td:
        root = Path(td)
        fakebin = root / 'fakebin'; fakebin.mkdir()
        for name, body in {
            'git': '#!/bin/bash\nif [ "$1" = fetch ] && [ "$ALICE_REPRO_MODE" = fetch ]; then echo FETCH_SENTINEL >&2; exit 128; fi\nif [ "$1" = push ] && [ "$ALICE_REPRO_MODE" = push ]; then echo PUSH_SENTINEL >&2; exit 128; fi\nexec ' + GIT + ' "$@"\n',
            'sleep': '#!/bin/bash\nexit 0\n',
            'date': '#!/bin/bash\nprintf "%s\\n" "$ALICE_REPRO_DATE"\n',
        }.items():
            p = fakebin / name; p.write_text(body); p.chmod(0o700)
        helper = root / 'helper.sh'
        assert source.count(b'ROOT="$HOME/rayan-compute"') == 1
        helper.write_bytes(source.replace(b'ROOT="$HOME/rayan-compute"', b'ROOT="$ALICE_REPRO_ROOT"'))
        for mode in ['fetch', 'push', 'timestamp']:
            case = root / mode; case.mkdir()
            ledger = case / 'telemetry/ledger'
            ledger.parent.mkdir(parents=True)
            remote = case / 'remote.git'
            run([GIT, 'init', '--bare', str(remote)])
            run([GIT, 'init', str(ledger)])
            for key, value in [('user.name', 'ALICE local fixture'), ('user.email', 'fixture@example.invalid')]:
                run([GIT, '-C', str(ledger), 'config', key, value])
            (ledger / 'README').write_text('disposable local fixture\n')
            run([GIT, '-C', str(ledger), 'add', 'README'])
            run([GIT, '-C', str(ledger), 'commit', '-m', 'fixture'])
            run([GIT, '-C', str(ledger), 'branch', '-M', 'main'])
            run([GIT, '-C', str(ledger), 'remote', 'add', 'origin', str(remote)])
            run([GIT, '-C', str(ledger), 'push', 'origin', 'main'])
            payload = case / 'payload'; payload.mkdir()
            for n in ['application.log', 'manifest.json', 'receipt.json']:
                (payload / n).write_text('{}\n')
            env = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ['PATH'],
                ALICE_REPRO_ROOT=str(case), ALICE_REPRO_MODE=mode,
                ALICE_REPRO_DATE='2026-09-07T00:00:00+00:00')
            argv = ['bash', str(helper), 'fixture-immutable', 'FIXTURE', 'FAILED', '1'] + [str(payload / n) for n in ['application.log', 'manifest.json', 'receipt.json']]
            first = subprocess.run(argv, env=env, capture_output=True, timeout=15)
            if mode != 'timestamp':
                assert first.returncode == 74 and b'SENTINEL' not in first.stdout + first.stderr
                findings.append({'case': mode + '_failure', 'exit': 74, 'raw_git_error_lost_inside_helper': True})
            else:
                assert first.returncode == 0, first.stderr
                before = (ledger / 'runs/fixture-immutable/bundle.sha256').read_bytes()
                env['ALICE_REPRO_DATE'] = '2026-09-07T00:00:01+00:00'
                second = subprocess.run(argv, env=env, capture_output=True, timeout=15)
                assert second.returncode == 73 and b'RUN_ID_COLLISION' in second.stderr
                assert before == (ledger / 'runs/fixture-immutable/bundle.sha256').read_bytes()
                findings.append({'case': 'same_payload_later_timestamp', 'first_exit': 0,
                    'retry_exit': 73, 'original_remote_bundle_preserved': True,
                    'three_input_payloads_unchanged': True})
    result = {'source_sha256': hashlib.sha256(source).hexdigest(),
        'scope': 'Disposable local Git only. Root path remapped; sleep suppressed; date controlled. Fetch/push errors explicitly injected. No claim these injected errors occurred on Magnolia.',
        'findings': findings}
    (BASE / 'HELPER_REPRODUCTION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
