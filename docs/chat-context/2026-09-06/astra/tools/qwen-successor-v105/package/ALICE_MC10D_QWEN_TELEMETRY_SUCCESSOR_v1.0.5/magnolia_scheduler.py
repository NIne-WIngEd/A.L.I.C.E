"""Shared Magnolia scheduler boundary, based on the owner's raw provider receipt."""
import base64
import os
import subprocess

import contract as c

USER_QUEUE = ['squeue', '-h', '-u', 'mxrayan', '-o', '%A|%j|%T']
START = '2026-09-06'
LAST = None
LIMIT = 512 * 1024


def stream(raw):
    raw = raw or b''
    return {'bytes': len(raw), 'sha256': c.sha(raw),
            'base64': base64.b64encode(raw[:LIMIT]).decode(),
            'text': raw[:LIMIT].decode('utf-8', 'replace'), 'truncated': len(raw) > LIMIT}


def command(argv, check=True):
    """Durably retain each result before interpreting success or failure."""
    global LAST
    record = {'argv': list(argv), 'observed_at': c.now()}
    try:
        p = subprocess.run(argv, capture_output=True, timeout=90)
        record.update(exit_code=p.returncode, status='RETURNED', stdout=stream(p.stdout), stderr=stream(p.stderr))
    except subprocess.TimeoutExpired as exc:
        record.update(exit_code=None, status='TIMEOUT', stdout=stream(exc.output), stderr=stream(exc.stderr))
    except OSError as exc:
        record.update(exit_code=None, status='EXEC_ERROR', stdout=stream(b''), stderr=stream(str(exc).encode()))
    record['finished_at'] = c.now()
    journal = c.ROOT / 'runs' / c.RUN_ID / 'scheduler-commands.jsonl'
    c.require(journal.parent.is_dir() and not journal.is_symlink(), 'Scheduler journal location differs')
    with journal.open('ab') as out:
        out.write(c.canonical(record).encode()); out.flush(); os.fsync(out.fileno())
    LAST = record
    c.require(record['status'] == 'RETURNED', 'Scheduler ' + record['status'] + '; raw receipt retained: ' + str(journal))
    c.require(not record['stdout']['truncated'] and not record['stderr']['truncated'], 'Scheduler output exceeds capture limit; interpretation refused')
    try:
        result = subprocess.CompletedProcess(argv, p.returncode, p.stdout.decode('utf-8'), p.stderr.decode('utf-8'))
    except UnicodeDecodeError as exc:
        raise c.Stop('Scheduler output encoding invalid; raw receipt retained') from exc
    if check:
        c.require(result.returncode == 0, 'Scheduler command failed: ' + argv[0] + '; ' + result.stderr.strip()[:1200])
    return result


def rows(text, fields):
    result = []
    for line in text.splitlines():
        if not line.strip(): continue
        row = line.strip().split('|')
        if row[-1] == '': row.pop()
        c.require(len(row) == fields and row[0].isdigit() and bool(row[1]) and bool(row[2]),
                  'Malformed scheduler row; absence is unknown')
        result.append(row)
    return result


def active_rows(run_command):
    result = run_command(list(USER_QUEUE), check=False)
    c.require(result.returncode == 0, 'Source queue query failed; absence is unknown; ' + result.stderr.strip()[:1200])
    return rows(result.stdout, 3)


def accounting_argv(job):
    c.require(isinstance(job, str) and job.isdigit(), 'Invalid scheduler job identity')
    return ['sacct', '-X', '-n', '-P', '--starttime', START, '-j', job,
            '--format', 'JobIDRaw,JobName%100,State,ExitCode']


def historical_rows(run_command, job):
    result = run_command(accounting_argv(job), check=False)
    c.require(result.returncode == 0, 'Accounting query failed; terminal state is unknown; ' + result.stderr.strip()[:1200])
    return rows(result.stdout, 4)


def find_submitted(run_command):
    active = active_rows(run_command)
    result = run_command(['sacct', '-X', '-n', '-P', '-u', 'mxrayan', '--starttime', START,
                          '--name', c.RUN_ID, '--format', 'JobIDRaw,JobName%100,State'], check=False)
    c.require(result.returncode == 0, 'Current accounting query failed; absence is unknown; ' + result.stderr.strip()[:1200])
    found = {row[0] for row in active + rows(result.stdout, 3) if row[1] == c.RUN_ID}
    c.require(len(found) <= 1, 'Multiple scheduler jobs for the same approved run; review required')
    return next(iter(found)) if found else None


def job_state(run_command, job, run_id):
    active = [row for row in active_rows(run_command) if row[0] == job]
    c.require(len(active) <= 1, 'Ambiguous active scheduler identity')
    if active:
        c.require(active[0][1] == run_id, 'Scheduler job identity mismatch')
        return active[0][2].split()[0].rstrip('+'), True
    historical = [row for row in historical_rows(run_command, job) if row[0] == job]
    c.require(len(historical) <= 1, 'Ambiguous historical scheduler identity')
    if historical:
        c.require(historical[0][1] == run_id, 'Accounting job identity mismatch')
        return historical[0][2].split()[0].rstrip('+'), False
    return 'UNKNOWN', False
