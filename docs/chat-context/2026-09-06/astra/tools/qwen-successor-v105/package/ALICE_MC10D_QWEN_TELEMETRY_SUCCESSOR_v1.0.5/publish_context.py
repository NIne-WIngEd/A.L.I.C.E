"""Publish independently rebuilt public Qwen metadata with an isolated Git index."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import contract as f
import evidence
BRANCH='alice-context'
FROZEN_MAIN=f.MAIN
PREFIX='docs/chat-context/2026-09-06/astra/qwen-public'

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
        git(repo, 'config', 'user.name', 'ALICE public calibration')
        git(repo, 'config', 'user.email', 'alice-calibration@users.noreply.github.com')
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
                if previous and not path.endswith('/LATEST_QWEN_PUBLIC.json'):
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


def validate_repo(repo_root):
    f.require(shutil.which('git') is not None, 'Git is not installed')
    f.require(repo_root.is_dir(), 'recorded local repository not found')
    origin=text(repo_root,'remote','get-url','origin')
    normalized=origin.lower().removesuffix('.git').removesuffix('/')
    f.require(normalized in ('https://github.com/nine-winged/a.l.i.c.e','git@github.com:nine-winged/a.l.i.c.e','ssh://git@github.com/nine-winged/a.l.i.c.e'), 'local origin is not the ALICE repository')
    observed=text(repo_root,'ls-remote','origin','refs/heads/main').split()[0]
    f.require(observed==FROZEN_MAIN, 'remote main changed; refresh the pinned run context')
    return origin


def public_payloads(folder,package_sha):
    summary=evidence.analyze(folder,package_sha)
    raw=f.canonical(summary).encode()
    identity=f.sha(raw)
    directory=PREFIX+'/'+f.RUN_ID+'/'+identity[:16]
    pointer={'schema':'alice.mc10d.qwen.public-pointer.v1','run_id':f.RUN_ID,'summary_path':directory+'/summary.json','summary_sha256':identity,'status':summary['status'],'qualification_passed':summary['qualification_passed'],'calibration_only':True,'four_family_binding_created':False,'breadth_v104_eligible':False}
    return {directory+'/summary.json':raw,PREFIX+'/LATEST_QWEN_PUBLIC.json':f.canonical(pointer).encode()},summary


def publish(folder,package_sha,repo_root):
    origin=validate_repo(repo_root)
    payloads,summary=public_payloads(folder,package_sha)
    result=append_commit(origin,payloads,'context: preserve independently verified Qwen public calibration result\n')
    observed=text(repo_root,'ls-remote','origin','refs/heads/main').split()[0]
    f.require(observed==FROZEN_MAIN,'main changed during publication; review recorded context commit')
    result.update(main_before=FROZEN_MAIN,main_after=observed,main_unchanged=True,raw_responses_published=False)
    return result,summary
