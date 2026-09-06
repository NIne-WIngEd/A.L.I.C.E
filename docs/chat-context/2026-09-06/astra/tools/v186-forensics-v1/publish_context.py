"""Append a reviewed public forensic summary using an isolated Git index."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import mc10d_forensics as f

REMOTE = 'https://github.com/NIne-WIngEd/A.L.I.C.E.git'
BRANCH = 'alice-context'
FROZEN_MAIN = '0abaed85873c3f8de04765847eb7700b0e20433f'
PREFIX = 'docs/chat-context/2026-09-06/astra/forensics'


class PublishError(RuntimeError):
    pass


def git(repo, *args, data=None, check=True):
    env = os.environ.copy()
    env.update({'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'})
    # Isolate from a caller's custom index/worktree without changing their shell.
    for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES'):
        env.pop(key, None)
    try:
        result = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', str(repo), *args],
                                input=data, capture_output=True, timeout=120, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublishError('Git unavailable or timed out; collection is preserved') from exc
    if check and result.returncode:
        # Avoid echoing credential-helper output or an authenticated remote URL.
        raise PublishError('Git ' + args[0] + ' failed with exit ' + str(result.returncode))
    return result


def text(repo, *args):
    return git(repo, *args).stdout.decode('utf-8').strip()


def append_commit(remote, payloads, message):
    """No checkout, no force push, append-only content paths; bounded race retry."""
    with tempfile.TemporaryDirectory(prefix='alice-context-publish-') as temp:
        repo = Path(temp)
        git(repo, 'init', '--quiet')
        git(repo, 'remote', 'add', 'origin', remote)
        git(repo, 'config', 'user.name', 'ALICE offline forensics')
        git(repo, 'config', 'user.email', 'alice-forensics@users.noreply.github.com')
        for attempt in range(3):
            print('context_publish_stage=fetch attempt=' + str(attempt + 1), flush=True)
            git(repo, 'fetch', '--quiet', '--depth=1', '--filter=blob:none', 'origin', 'refs/heads/' + BRANCH)
            parent = text(repo, 'rev-parse', 'FETCH_HEAD')
            git(repo, 'read-tree', parent)
            changed = False
            for path, content in sorted(payloads.items()):
                f.require(path.startswith(PREFIX + '/') and '..' not in Path(path).parts, 'publication path outside forensic scope')
                blob = git(repo, 'hash-object', '-w', '--stdin', data=content).stdout.decode().strip()
                old = git(repo, 'rev-parse', '--verify', parent + ':' + path, check=False)
                previous = old.stdout.decode().strip() if old.returncode == 0 else None
                if previous == blob:
                    continue
                # All evidence paths are content-addressed; only the explicit latest pointer changes.
                if previous and not path.endswith('/LATEST_V186_GLM.json'):
                    raise PublishError('Existing immutable forensic path differs; refusing overwrite')
                git(repo, 'update-index', '--add', '--cacheinfo', '100644', blob, path)
                changed = True
            if not changed:
                return {'status': 'ALREADY_PRESENT', 'branch': BRANCH, 'commit': parent, 'force_push': False}
            tree = text(repo, 'write-tree')
            commit = git(repo, 'commit-tree', tree, '-p', parent, data=message.encode('utf-8')).stdout.decode().strip()
            print('context_publish_stage=push', flush=True)
            pushed = git(repo, 'push', '--porcelain', 'origin', commit + ':refs/heads/' + BRANCH, check=False)
            if pushed.returncode == 0:
                observed = text(repo, 'ls-remote', 'origin', 'refs/heads/' + BRANCH).split()[0]
                if observed != commit:
                    # Another fast-forward may have followed ours; verify ancestry rather than claiming failure.
                    git(repo, 'fetch', '--quiet', '--depth=64', 'origin', 'refs/heads/' + BRANCH)
                    f.require(git(repo, 'merge-base', '--is-ancestor', commit, 'FETCH_HEAD', check=False).returncode == 0, 'published commit not present on remote branch')
                return {'status': 'PUBLISHED', 'branch': BRANCH, 'commit': commit, 'verified_remote_head': observed, 'force_push': False}
            reason = (pushed.stderr + pushed.stdout).lower()
            if not any(value in reason for value in (b'non-fast-forward', b'fetch first', b'failed to update ref', b'stale info')):
                raise PublishError('Git push denied or unavailable; existing collection is preserved')
        raise PublishError('Context branch advanced repeatedly; collection is preserved for a later publish')


def public_payloads(run):
    payloads = {key: f.plain_file(run / 'raw' / name) for key, name in f.FILES.items()}
    tasks, spec = f.load_authority()
    report = f.analyze(payloads, tasks, spec)
    # Recompute public content from the fixed schema; never publish raw rationales,
    # arbitrary local Markdown, local paths, lifecycle blobs or credential logs.
    content = f.canonical(report).encode('utf-8')
    identity = f.digest(content)
    directory = PREFIX + '/v186-glm-' + f.RUN_ID + '/' + identity[:16]
    pointer = {'artifact_id': 'alice.mc10d.public-forensics-pointer.v1', 'run_id': f.RUN_ID,
               'status': report['status'], 'summary_path': directory + '/summary.json',
               'summary_sha256': identity, 'qualification_granted': False}
    return {directory + '/summary.json': content,
            directory + '/summary.md': f.render_report(report).encode('utf-8'),
            PREFIX + '/LATEST_V186_GLM.json': f.canonical(pointer).encode('utf-8')}


def publish(run, repo_root):
    f.require(shutil.which('git') is not None, 'Git is not installed')
    f.require(repo_root.is_dir(), 'recorded local repository not found')
    origin = text(repo_root, 'remote', 'get-url', 'origin')
    normalized = origin.lower().removesuffix('.git').removesuffix('/')
    f.require(normalized in ('https://github.com/nine-winged/a.l.i.c.e', 'git@github.com:nine-winged/a.l.i.c.e', 'ssh://git@github.com/nine-winged/a.l.i.c.e'), 'local origin is not the ALICE repository')
    # Use the user's configured transport, including SSH, after exact repository validation.
    remote_main = text(repo_root, 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    f.require(remote_main == FROZEN_MAIN, 'remote main changed; refresh context before publishing this pinned package')
    result = append_commit(origin, public_payloads(run), 'context: preserve offline v186 GLM failure analysis\n')
    observed_main = text(repo_root, 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    result.update({'repository': 'NIne-WIngEd/A.L.I.C.E', 'main_before': remote_main, 'main_after': observed_main,
                   'main_unchanged': observed_main == remote_main, 'raw_responses_published': False})
    f.require(result['main_unchanged'], 'main advanced during publication; review the recorded commit')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--repo-root', type=Path, default=Path(r'C:\A.L.I.C.E-main'))
    args = p.parse_args()
    try:
        result = publish(args.run_dir, args.repo_root)
        print(f.canonical(result), end='')
        return 0
    except Exception as exc:
        print('CONTEXT_PUBLICATION_PENDING: ' + str(exc)[:600] + '. Existing outputs remain intact.', file=sys.stderr)
        return 75


if __name__ == '__main__':
    sys.exit(main())
