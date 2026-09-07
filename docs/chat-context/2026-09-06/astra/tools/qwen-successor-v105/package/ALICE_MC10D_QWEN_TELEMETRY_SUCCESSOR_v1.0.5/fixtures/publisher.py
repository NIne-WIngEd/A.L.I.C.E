"""Seven-argument immutable publisher. Compatible with Magnolia Git 1.8.3.1."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

CHUNK_BYTES = 45000000

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()
def write_json(path, value): path.write_text(json.dumps(value, indent=2) + '\n')

class Stop(Exception):
    def __init__(self, code, message): self.code, self.message = code, message

def require(ok, code, message):
    if not ok: raise Stop(code, message)

def file_map(folder, exclude=()):
    result = {}
    for p in sorted(folder.rglob('*')):
        require(not p.is_symlink(), 73, 'Linked spool member refused')
        if p.is_file() and p.relative_to(folder).as_posix() not in exclude:
            result[p.relative_to(folder).as_posix()] = digest(p)
    return result

def bundle(folder):
    return ''.join(value + '  ./' + name + '\n' for name, value in file_map(folder, ('bundle.sha256',)).items()).encode()

class Git:
    def __init__(self, ledger, journal, wrapper, deadline):
        self.ledger, self.journal, self.deadline = ledger, journal, deadline
        self.env = dict(os.environ, GIT_SSH=str(wrapper), GIT_SSH_VARIANT='ssh', GIT_TERMINAL_PROMPT='0')
        for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE'):
            self.env.pop(key, None)
        self.sequence = 0

    def call(self, *args, input=None, check=True):
        self.sequence += 1
        prefix = self.journal / ('git-%03d' % self.sequence)
        record = {'argv': ['git', *args], 'cwd': str(self.ledger), 'started_at': now()}
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, 74, 'Publisher time budget exhausted')
        stdout = stderr = b''; code = None
        try:
            process = subprocess.Popen(['git', *args], cwd=self.ledger, env=self.env,
                stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                stdout, stderr = process.communicate(input, timeout=min(15, remaining))
                code = process.returncode
                record['status'] = 'RETURNED'
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                stdout, stderr = process.communicate()
                record['status'] = 'TIMEOUT'
        except OSError as exc:
            record.update(status='OS_ERROR', error=str(exc))
        finally:
            prefix.with_suffix('.stdout.bin').write_bytes(stdout)
            prefix.with_suffix('.stderr.bin').write_bytes(stderr)
            record.update(exit_code=code, finished_at=now(),
                stdout_bytes=len(stdout), stderr_bytes=len(stderr),
                stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest())
            write_json(prefix.with_suffix('.json'), record)
        require(code is not None, 74, 'Git timed out or could not start; raw receipt preserved')
        if check: require(code == 0, 74, 'Git failed: ' + ' '.join(args) + '; raw receipt preserved')
        return code, stdout

    def text(self, *args): return self.call(*args)[1].decode('ascii').strip()

def prepare_spool(candidate, spool, run, stage, status, job, log, manifest, receipt):
    candidate.mkdir()
    shutil.copyfile(manifest, candidate / 'manifest.json')
    shutil.copyfile(receipt, candidate / 'receipt.json')
    size = log.stat().st_size
    if size <= CHUNK_BYTES: shutil.copyfile(log, candidate / 'application.log')
    else:
        parts = candidate / 'log-parts'; parts.mkdir()
        with log.open('rb') as stream:
            number = 0
            while True:
                data = stream.read(CHUNK_BYTES)
                if not data: break
                require(number < 10000, 69, 'Log exceeds original four-digit part capacity')
                (parts / ('application.log.part.%04d' % number)).write_bytes(data)
                number += 1
        (parts / 'SHA256SUMS').write_text(''.join(h + '  ' + name + '\n' for name, h in file_map(parts).items()))
    expected = {'schema': 'rayan.compute.telemetry.v1', 'run_id': run, 'stage': stage,
        'status': status, 'provider': 'magnolia', 'slurm_job_id': job, 'log_bytes': size}
    # Existing old-helper spool metadata is immutable evidence too. Never
    # regenerate published_at merely because this invocation is a retry.
    if spool.exists() or spool.is_symlink():
        require(spool.is_dir() and not spool.is_symlink(), 73, 'Invalid existing spool')
        require(file_map(spool, ('telemetry.json', 'bundle.sha256')) == file_map(candidate), 73, 'RUN_ID_COLLISION: pending payload differs')
        metadata = json.loads((spool / 'telemetry.json').read_bytes())
        require(all(metadata.get(k) == v for k, v in expected.items()), 73, 'RUN_ID_COLLISION: pending metadata differs')
        require((spool / 'bundle.sha256').read_bytes() == bundle(spool), 73, 'Cached bundle hash differs')
        shutil.rmtree(candidate)
    else:
        write_json(candidate / 'telemetry.json', dict(expected, published_at=now()))
        (candidate / 'bundle.sha256').write_bytes(bundle(candidate))
        os.replace(candidate, spool)
    pattern = re.compile(rb'GOCSPX-|github_pat_|ghp_|-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for name in file_map(spool):
        tail = b''
        with (spool / name).open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                require(pattern.search(tail + chunk) is None, 70, 'TELEMETRY_SECRET_SCAN_FAILED')
                tail = chunk[-128:]

def snapshot_tree(git, base, dest):
    raw = git.call('ls-tree', '-r', '-z', base, '--', dest + '/')[1]
    result = {}
    for row in raw.split(b'\0'):
        if not row: continue
        meta, path = row.split(b'\t', 1)
        mode, kind, oid = meta.decode('ascii').split()
        path = path.decode('utf-8')
        require(mode == '100644' and kind == 'blob' and path.startswith(dest + '/'), 73, 'Unexpected existing snapshot tree')
        result[path[len(dest) + 1:]] = oid
    return result

def publish(argv, root=None):
    require(len(argv) == 7, 64, 'usage: RUN_ID STAGE STATUS JOB_ID LOG MANIFEST RECEIPT')
    run, stage, status, job, log, manifest, receipt = argv
    require(re.fullmatch(r'[A-Za-z0-9._-]+', run) is not None and run not in ('.', '..'), 65, 'Invalid run ID')
    require(re.fullmatch(r'[A-Za-z0-9._-]+', stage) is not None, 66, 'Invalid stage')
    require(status in ('COMPLETED', 'FAILED', 'CANCELLED', 'RUNNING'), 67, 'Invalid status')
    require(re.fullmatch(r'[0-9]+', job) is not None, 68, 'Invalid job ID')
    log, manifest, receipt = map(Path, (log, manifest, receipt))
    require(all(p.is_file() and not p.is_symlink() for p in (log, manifest, receipt)), 69, 'Missing or linked input')
    root = Path(root) if root is not None else Path.home() / 'rayan-compute'
    ledger, spool = root / 'telemetry/ledger', root / 'telemetry/spool' / run
    journal = root / 'telemetry/publisher-receipts' / run / uuid.uuid4().hex
    journal.mkdir(parents=True, mode=0o700)
    lock = root / 'telemetry/push.lock'
    report = {'schema': 'rayan.compute.publisher-receipt.v1', 'run_id': run, 'started_at': now(),
        'journal': str(journal), 'published': False, 'worktree_or_local_branch_reset': False}
    owned = False; code = 74; deadline = time.monotonic() + 70
    try:
        for _ in range(20):
            try: lock.mkdir(mode=0o700); owned = True; break
            except FileExistsError: time.sleep(0.25)
        require(owned, 71, 'Publisher lock busy; it was not removed')
        require((ledger / '.git').is_dir(), 72, 'Missing ledger clone')
        spool.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        prepare_spool(journal / 'candidate', spool, run, stage, status, job, log, manifest, receipt)
        report['spool_sha256'] = file_map(spool)
        git = Git(ledger, journal, root / 'bin/rayan-github-ssh', deadline)
        git.env['GIT_INDEX_FILE'] = str(journal / 'private-index')
        dest = 'runs/' + run
        expected = {name: git.text('hash-object', '-w', str(spool / name)) for name in file_map(spool)}
        for attempt in range(1, 4):
            report['attempt'] = attempt
            # A destination refspec is required on Git 1.8.3.1. Do not rely on
            # opportunistic tracking-ref updates introduced in Git 1.8.4.
            git.call('fetch', 'origin', 'refs/heads/main:refs/remotes/origin/main')
            base = git.text('rev-parse', 'FETCH_HEAD')
            require(base == git.text('rev-parse', 'refs/remotes/origin/main'), 74, 'Fetched and tracking commits differ')
            report['fetched_base'] = base
            existing = snapshot_tree(git, base, dest)
            require(all(expected.get(name) == value for name, value in existing.items()), 73, 'RUN_ID_COLLISION: existing ledger bytes differ')
            if existing == expected:
                report.update(published=True, idempotent=True, verified_commit=base); code = 0; break
            # A private index preserves the shared checkout, local main and any
            # uncommitted files. The new commit has the freshly fetched parent.
            git.call('read-tree', base)
            for name, oid in expected.items():
                git.call('update-index', '--add', '--cacheinfo', '100644', oid, dest + '/' + name)
            tree = git.text('write-tree')
            commit = git.text('commit-tree', tree, '-p', base, '-m', 'telemetry: ' + run + ' ' + status)
            report['proposed_commit'] = commit
            pushed, _ = git.call('push', 'origin', commit + ':refs/heads/main', check=False)
            # Refetch on every attempt, including a lost push acknowledgement.
            # A competing append becomes the next parent; no force push is used.
            if pushed != 0: continue
            git.call('fetch', 'origin', 'refs/heads/main:refs/remotes/origin/main')
            verified = git.text('rev-parse', 'FETCH_HEAD')
            require(snapshot_tree(git, verified, dest) == expected, 75, 'Published snapshot did not verify')
            report.update(published=True, idempotent=False, verified_commit=verified); code = 0; break
        require(code == 0, 74, 'Git publication retry budget exhausted; raw receipts preserved')
    except Stop as exc:
        code = exc.code; report['error'] = exc.message
    except Exception as exc:
        code = 74; report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if owned:
            try: lock.rmdir()
            except OSError as exc: report['lock_release_error'] = str(exc)
        report.update(exit_code=code, finished_at=now())
        write_json(journal / 'result.json', report)
        print('TELEMETRY_RECEIPT_DIR=' + str(journal), flush=True)
        if report.get('error'): print(report['error'], file=sys.stderr, flush=True)
    if code == 0:
        print('RAYAN_GITHUB_TELEMETRY_PUSHED=true\nrun_id=' + run + '\nstatus=' + status)
        if report.get('idempotent'): print('telemetry_idempotent=true')
    return code

if __name__ == '__main__':
    os.umask(0o077)
    try: sys.exit(publish(sys.argv[1:]))
    except Stop as exc:
        print(exc.message, file=sys.stderr); sys.exit(exc.code)
