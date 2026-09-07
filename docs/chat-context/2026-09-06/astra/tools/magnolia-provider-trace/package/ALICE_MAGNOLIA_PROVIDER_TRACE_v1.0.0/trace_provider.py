"""Read-only observations. No workload import, filesystem write or scheduler mutation."""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time

MAX_FILE = 2 * 1024 * 1024
MAX_STREAM = 512 * 1024
MAX_ENTRIES = 64


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def byte_stream(raw, limit=MAX_STREAM):
    raw = b'' if raw is None else raw
    if isinstance(raw, str): raw = raw.encode('utf-8')
    kept = raw[:limit]
    return {'bytes': len(raw), 'sha256': sha(raw), 'base64': base64.b64encode(kept).decode('ascii'),
            'text': kept.decode('utf-8', 'replace'), 'truncated': len(raw) > limit,
            'retained_bytes': len(kept)}


def command_receipt(spec, command=subprocess.run):
    """Capture raw bytes before anyone interprets a command's meaning."""
    result = {'name': spec['name'], 'argv': list(spec['argv']), 'observed_at': now()}
    try:
        p = command(spec['argv'], capture_output=True, timeout=20)
        result.update(exit_code=p.returncode, status='RETURNED', stdout=byte_stream(p.stdout), stderr=byte_stream(p.stderr))
    except subprocess.TimeoutExpired as exc:
        result.update(exit_code=None, status='TIMEOUT', stdout=byte_stream(exc.output), stderr=byte_stream(exc.stderr))
    except OSError as exc:
        result.update(exit_code=None, status='EXEC_ERROR', error_class=type(exc).__name__,
                      error_message=str(exc), stdout=byte_stream(b''), stderr=byte_stream(b''))
    result['finished_at'] = now()
    result['raw_streams_complete'] = not result['stdout']['truncated'] and not result['stderr']['truncated']
    result['query_succeeded'] = result['status'] == 'RETURNED' and result['exit_code'] == 0
    return result


def rooted(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(x in ('..', '.') for x in rel.parts) or '\\' in relative:
        raise ValueError('Unexpected observation path')
    return Path(root).joinpath(*rel.parts)


def guard(root, path):
    """Do not follow a changed link/junction into an unrelated directory."""
    relative = Path(path).relative_to(root)
    candidate = Path(root)
    for part in ('', *relative.parts[:-1]):
        if part: candidate /= part
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(candidate, 'is_junction', lambda: False)():
            return 'LINKED_PARENT'
        if not stat.S_ISDIR(info.st_mode): return 'NON_DIRECTORY_PARENT'
    return None


def file_fact(root, relative, expected=None, raw=False):
    p = rooted(root, relative)
    result = {'path': p.as_posix(), 'expected_sha256': expected}
    try:
        bad = guard(Path(root), p)
        if bad: return {**result, 'status': bad}
        info = p.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(p, 'is_junction', lambda: False)():
            return {**result, 'status': 'LINKED_FILE'}
        if not stat.S_ISREG(info.st_mode): return {**result, 'status': 'NON_REGULAR_FILE'}
        if info.st_size > MAX_FILE: return {**result, 'status': 'OVER_READ_LIMIT', 'bytes': info.st_size}
        with p.open('rb') as source: data = source.read(MAX_FILE + 1)
        if len(data) > MAX_FILE: return {**result, 'status': 'GREW_OVER_READ_LIMIT', 'bytes_before': info.st_size}
        result.update(status='PRESENT', bytes=len(data), sha256=sha(data), changed_size_during_read=len(data) != info.st_size)
        if expected is not None: result['expected_hash_matches'] = result['sha256'] == expected
        if raw:
            result['base64'] = base64.b64encode(data).decode('ascii')
            result['text'] = data.decode('utf-8', 'replace')
        return result
    except FileNotFoundError:
        return {**result, 'status': 'MISSING'}
    except OSError as exc:
        return {**result, 'status': 'READ_ERROR', 'error_class': type(exc).__name__, 'error_message': str(exc)}


def directory_fact(root, relative):
    p = rooted(root, relative)
    result = {'path': p.as_posix(), 'entries': []}
    try:
        bad = guard(Path(root), p)
        if bad: return {**result, 'status': bad}
        info = p.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(p, 'is_junction', lambda: False)(): return {**result, 'status': 'LINKED_DIRECTORY'}
        if not stat.S_ISDIR(info.st_mode): return {**result, 'status': 'NON_DIRECTORY'}
        with os.scandir(p) as entries:
            for item in entries:
                if len(result['entries']) == MAX_ENTRIES:
                    return {**result, 'status': 'ENTRY_LIMIT', 'truncated': True}
                meta = item.stat(follow_symlinks=False)
                kind = 'link' if stat.S_ISLNK(meta.st_mode) else 'directory' if stat.S_ISDIR(meta.st_mode) else 'file' if stat.S_ISREG(meta.st_mode) else 'other'
                result['entries'].append({'name': item.name, 'kind': kind})
        result['entries'].sort(key=lambda x: x['name'])
        return {**result, 'status': 'PRESENT', 'truncated': False}
    except FileNotFoundError: return {**result, 'status': 'MISSING'}
    except OSError as exc: return {**result, 'status': 'READ_ERROR', 'error_class': type(exc).__name__, 'error_message': str(exc)}


def inventory(root, mode, policy):
    spec = policy[mode + '_inventory']
    files, directories = {}, {}
    for entry in spec['files']:
        files[entry['path']] = file_fact(root, entry['path'], entry.get('expected_sha256'), entry.get('raw', False))
    for group in spec['groups']:
        relative = group['path']; listing = directory_fact(root, relative); directories[relative] = listing
        # Missing, linked, incomplete and unreadable inventories remain distinct.
        for entry in listing['entries']:
            if entry['kind'] != 'directory': continue
            sub = relative + '/' + entry['name']; sub_listing = directory_fact(root, sub); directories[sub] = sub_listing
            for name in group['files']:
                path = sub + '/' + name
                files[path] = file_fact(root, path, policy.get('expected_files', {}).get(mode + ':' + path), group['raw'])
    problems = [name for name, value in {**files, **directories}.items()
                if value['status'] not in ('PRESENT', 'MISSING') or value.get('changed_size_during_read')]
    return {'files': files, 'directories': directories, 'inventory_complete': not problems, 'unresolved_paths': problems}


def observe_remote(policy, root=None, command=subprocess.run):
    root = Path(policy['remote_root']) if root is None else Path(root)
    report = {'schema': 'alice.magnolia.read-only-provider-observation.v1', 'read_only': True,
              'source_id': policy['source_id'], 'current_id': policy['current_id'], 'observed_at': now(),
              'python_executable': sys.executable, 'python_version': sys.version,
              'execution_authority_granted': False}
    report['inventory_before'] = inventory(root, 'remote', policy)
    # No failed comparison aborts later collection or destroys raw diagnostics.
    report['commands'] = [command_receipt(spec, command) for spec in policy['commands']]
    report['inventory_after'] = inventory(root, 'remote', policy)
    report['inventories_equal'] = report['inventory_before'] == report['inventory_after']
    report['provider_command_failures'] = sum(not r['query_succeeded'] for r in report['commands'])
    report['all_raw_streams_complete'] = all(r['raw_streams_complete'] for r in report['commands'])
    report['finished_at'] = now()
    report['next_action'] = 'RETURN_RAW_RECEIPT_FOR_REVIEW; no repair or execution authority inferred'
    return report
