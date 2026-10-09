"""Authenticated append-only public fit history across readonly session roots.

The external digest and provider mount/version custody remain explicit inputs.
No readonly claim or accepted byte cache is inferred from a directory name.
"""
from __future__ import annotations

import os
from pathlib import Path

from cognitive_kernel.canonical import canonical_sha256, require_sha256
from . import train_v1_formation_specialist as shared

SCHEMA = 'mfm-public-conditional-foundation-history-v1'
SESSION_SCHEMA = 'mfm-public-conditional-foundation-session-v1'
ENTRY_SCHEMA = 'mfm-public-conditional-foundation-history-entry-v1'


def _trainer():
    from . import train_public_conditional_foundation
    return train_public_conditional_foundation


def require_readonly_root(path):
    root = _trainer()._safe_root(path)
    if not hasattr(os, 'statvfs') or not os.statvfs(root).f_flag & os.ST_RDONLY:
        raise ValueError('historical public source requires an actual readonly filesystem mount')
    return root


def _files(root, paths):
    return [{'path': p.relative_to(root).as_posix(), 'size': p.stat().st_size, 'sha256': shared._digest(p)}
            for p in sorted(paths)]


def _verify_files(root, rows):
    names = set()
    for row in rows:
        relative = row['path']
        if not isinstance(relative, str) or relative.startswith('/') or '\\' in relative or \
                any(part in {'', '.', '..'} for part in relative.split('/')) or relative in names:
            raise ValueError('public history file path is foreign or repeated')
        names.add(relative)
        path = _trainer()._safe_root(root / relative)
        if not path.is_file() or root not in path.parents or path.stat().st_size != row['size'] or \
                shared._digest(path) != require_sha256(row['sha256'], 'public history file digest'):
            raise ValueError('public history source bytes differ')


def _sealed(row):
    return row.get('record_sha256') == canonical_sha256({k: v for k, v in row.items() if k != 'record_sha256'})


class History:
    """Accepted source map and immutable complete index, never copied tensors."""
    def __init__(self, record, roots, *, root=None, latest_path=None):
        self.record, self.roots, self.root, self.latest_path = record, roots, root, latest_path

    @property
    def latest_checkpoint(self):
        entry = self.record['entries'][-1]
        return self.roots[entry['session_index']] / entry['path']

    def authorize_latest(self, checkpoint, binding):
        path = _trainer()._safe_root(checkpoint)
        if self.record['binding'] != binding or path != self.latest_checkpoint:
            raise ValueError('resume requires latest authenticated global public checkpoint')
        receipt = _trainer().verify_checkpoint(path, binding=binding)
        if receipt != self.record['entries'][-1]['receipt']:
            raise ValueError('latest public checkpoint differs from accepted history')
        return receipt

    def append(self, checkpoint):
        trainer = _trainer()
        path = trainer._safe_root(checkpoint)
        if self.root is None or path.parent != self.root:
            raise ValueError('history append requires this session owned checkpoint')
        receipt = trainer.verify_checkpoint(path, binding=self.record['binding'])
        entries = self.record['entries']
        if entries and receipt['cursor']['step'] <= entries[-1]['receipt']['cursor']['step']:
            if path == self.latest_checkpoint and receipt == entries[-1]['receipt']:
                return self.latest_path
            raise ValueError('public checkpoint history must append strictly increasing completed steps')
        entry = shared._seal({'schema': ENTRY_SCHEMA, 'session_index': self.record['session_index'],
            'path': path.name, 'receipt': receipt, 'files': [
                {'path': path.name + '/' + name, 'size': (path / name).stat().st_size, 'sha256': digest}
                for name, digest in [('checkpoint.json', shared._digest(path / 'checkpoint.json')),
                                     ('optimizer.pt', receipt['optimizer_sha256']),
                                     ('specialist.safetensors', receipt['specialist_sha256'])]],
            'previous_entry_sha256': entries[-1]['record_sha256'] if entries else None})
        record = {k: v for k, v in self.record.items() if k != 'record_sha256'}
        record['entries'] = [*entries, entry]
        destination = self.root / f"history-{receipt['cursor']['step']:08d}.json"
        self.record = shared._write_new(destination, record)
        self.latest_path = destination
        return destination


def start(root, *, manifest, curriculum, binding, session_index, allocation_ref, parent=None):
    trainer = _trainer()
    root = trainer._safe_root(root)
    if manifest.get('purpose') != 'full-roster-public-fit' or not _sealed(manifest) or not _sealed(curriculum) or \
            manifest['record_sha256'] != binding['run'] or type(session_index) is not int or session_index < 0 or \
            not isinstance(allocation_ref, str) or not allocation_ref.strip():
        raise ValueError('public fit history purpose, recipe or session identity differs')
    if parent is None:
        if session_index != 0:
            raise ValueError('fresh public fit history must start session zero')
        seed = shared._read_sealed(root / 'seed-control.json', trainer.SEED_SCHEMA)
        construction = shared._read_sealed(root / 'construction.json', 'mfm-public-conditional-foundation-construction-v1')
        if seed['record_sha256'] != binding['seed'] or seed['run_sha256'] != binding['run'] or \
                seed['optimizer_steps'] != 0 or seed['specialist_config'] != manifest['specialist_config'] or \
                seed['initialization'] != trainer.INITIALIZATION or construction['run_sha256'] != binding['run'] or \
                not construction['control_updates_restored']:
            raise ValueError('fresh public fit seed or restored construction binding differs')
        paths = [root / name for name in ('seed-control.json', 'seed-control.safetensors',
                                         'seed-generation.json', 'construction.json')]
        controls = root / 'construction-controls'
        for path in sorted(controls.glob('checkpoint-*')):
            trainer.verify_checkpoint(path, binding=binding)
            paths.extend(path.iterdir())
        if {p.name for p in controls.iterdir()} != {'checkpoint-00000000', 'checkpoint-00000001'}:
            raise ValueError('public fit original construction controls differ')
        origin = {'seed': seed, 'construction': construction, 'files': _files(root, paths)}
        entries, sessions, roots = [], [], {}
        start_cursor = {'epoch': 0, 'next_case': 0, 'step': 0}
    else:
        if parent.record['run'] != manifest or parent.record['curriculum'] != curriculum or \
                parent.record['binding'] != binding or session_index != parent.record['session_index'] + 1 or \
                root in parent.roots.values() or any(root in p.parents or p in root.parents for p in parent.roots.values()):
            raise ValueError('public continuation session or immutable recipe differs')
        origin, entries, sessions = parent.record['origin'], parent.record['entries'], parent.record['sessions']
        roots = dict(parent.roots)
        start_cursor = dict(entries[-1]['receipt']['cursor'])
        if start_cursor['epoch'] == binding['epochs']:
            raise ValueError('completed prescribed public fit cannot resume training')
    session = shared._write_new(root / 'session.json', {'schema': SESSION_SCHEMA,
        'run_sha256': binding['run'], 'session_index': session_index, 'allocation_ref': allocation_ref,
        'parent_history_sha256': parent.record['record_sha256'] if parent else None,
        'parent_entry_count': len(entries), 'start_cursor': start_cursor})
    for name, row, schema in [('run.json', manifest, trainer.RUN_SCHEMA),
                              ('curriculum.json', curriculum, trainer.CURRICULUM_SCHEMA)]:
        if (root / name).exists():
            if shared._read_sealed(root / name, schema) != row:
                raise ValueError('public session immutable metadata differs')
        else:
            shared._write_new(root / name, {k: v for k, v in row.items() if k != 'record_sha256'})
    roots[session_index] = root
    record = {'schema': SCHEMA, 'purpose': 'full-roster-public-fit', 'binding': binding,
        'run': manifest, 'curriculum': curriculum, 'origin': origin, 'session_index': session_index,
        'sessions': [*sessions, session], 'entries': list(entries)}
    return History(record, roots, root=root)


def load(path, expected_sha256, *, sources, manifest, curriculum, binding):
    """Accept all historical bytes once; no history is moved into working."""
    trainer = _trainer()
    path = trainer._safe_root(path)
    record = shared._read_sealed(path, SCHEMA)
    if record['record_sha256'] != require_sha256(expected_sha256, 'expected public history') or \
            record.get('purpose') != 'full-roster-public-fit' or manifest.get('purpose') != 'full-roster-public-fit' or \
            record['run'] != manifest or record['curriculum'] != curriculum or record['binding'] != binding or \
            not _sealed(manifest) or not _sealed(curriculum):
        raise ValueError('public source history immutable pin, purpose or recipe differs')
    index = record['session_index']
    if type(index) is not int or index < 0 or any(type(i) is not int for i in sources) or set(sources) != set(range(index + 1)) or \
            len(record['sessions']) != index + 1 or not record['entries']:
        raise ValueError('public source history requires every explicit prior session root')
    roots = {i: require_readonly_root(p) for i, p in sources.items()}
    if len(set(roots.values())) != len(roots) or any(a in b.parents for a in roots.values() for b in roots.values() if a != b):
        raise ValueError('public history source roots overlap')
    entries = record['entries']
    if entries[-1]['session_index'] != index:
        raise ValueError('public history latest session has no completed checkpoint')
    if path != roots[index] / f"history-{entries[-1]['receipt']['cursor']['step']:08d}.json":
        raise ValueError('public source history path is foreign or stale')
    previous, last_step, last_session = None, -1, 0
    for entry in entries:
        if not _sealed(entry) or entry.get('schema') != ENTRY_SCHEMA or entry['previous_entry_sha256'] != previous or \
                type(entry['session_index']) is not int or not last_session <= entry['session_index'] <= index:
            raise ValueError('public checkpoint global append-only chain differs')
        receipt = entry['receipt']
        trainer._cursor(receipt['cursor'], train_cases=binding['train_cases'], batch_size=binding['batch_size'], epochs=binding['epochs'])
        step = receipt['cursor']['step']
        if receipt['binding'] != binding or step <= last_step or (last_step == -1 and step != 0) or \
                entry['path'] != f'checkpoint-{step:08d}' or not _sealed(receipt):
            raise ValueError('public checkpoint global cursor or binding differs')
        root = roots[entry['session_index']]
        _verify_files(root, entry['files'])
        checkpoint = trainer._safe_root(root / entry['path'])
        actual = shared._read_sealed(checkpoint / 'checkpoint.json', trainer.CHECKPOINT_SCHEMA)
        files = {Path(r['path']).name: r for r in entry['files']}
        if actual != receipt or {r['path'] for r in entry['files']} != \
                {entry['path'] + '/' + name for name in ('specialist.safetensors', 'optimizer.pt', 'checkpoint.json')} or \
                set(p.name for p in checkpoint.iterdir()) != set(files) or \
                files['specialist.safetensors']['sha256'] != receipt['specialist_sha256'] or \
                files['optimizer.pt']['sha256'] != receipt['optimizer_sha256']:
            raise ValueError('public source checkpoint inventory differs')
        previous, last_step, last_session = entry['record_sha256'], step, entry['session_index']
    for i, root in roots.items():
        trainer._validate_run_paths(root)
        session = record['sessions'][i]
        if session != shared._read_sealed(root / 'session.json', SESSION_SCHEMA) or session['session_index'] != i or \
                session['run_sha256'] != binding['run'] or not session['allocation_ref'].strip():
            raise ValueError('public immutable source session identity differs')
        if shared._read_sealed(root / 'run.json', trainer.RUN_SCHEMA) != manifest or \
                shared._read_sealed(root / 'curriculum.json', trainer.CURRICULUM_SCHEMA) != curriculum:
            raise ValueError('public historical run or curriculum differs')
        expected_paths = {e['path'] for e in entries if e['session_index'] == i}
        if {p.name for p in root.glob('checkpoint-*')} != expected_paths:
            raise ValueError('public source history omits a checkpoint or contains a future checkpoint')
        count = session['parent_entry_count']
        if type(count) is not int or count != len([e for e in entries if e['session_index'] < i]):
            raise ValueError('public source history parent prefix count differs')
        if i == 0:
            if session['parent_history_sha256'] is not None or session['start_cursor'] != {'epoch': 0, 'next_case': 0, 'step': 0}:
                raise ValueError('public fresh history cannot import a construction parent')
        else:
            parent = shared._read_sealed(roots[i - 1] / f"history-{session['start_cursor']['step']:08d}.json", SCHEMA)
            if parent['record_sha256'] != session['parent_history_sha256'] or parent['entries'] != entries[:count] or \
                    parent['sessions'] != record['sessions'][:i] or parent['origin'] != record['origin'] or \
                    parent['run'] != manifest or parent['curriculum'] != curriculum or parent['binding'] != binding or \
                    session['start_cursor'] != entries[count - 1]['receipt']['cursor']:
                raise ValueError('public source history immutable ancestor prefix differs')
    origin = record['origin']
    origin_paths = {'seed-control.json', 'seed-control.safetensors', 'seed-generation.json', 'construction.json'} | \
        {f'construction-controls/checkpoint-{step:08d}/{name}' for step in (0, 1)
         for name in ('specialist.safetensors', 'optimizer.pt', 'checkpoint.json')}
    if {row['path'] for row in origin['files']} != origin_paths or len(origin['files']) != len(origin_paths):
        raise ValueError('public history original seed and construction inventory is incomplete')
    _verify_files(roots[0], origin['files'])
    if shared._read_sealed(roots[0] / 'seed-control.json', trainer.SEED_SCHEMA) != origin['seed'] or \
            origin['seed']['record_sha256'] != binding['seed'] or origin['seed']['run_sha256'] != binding['run'] or \
            origin['seed']['optimizer_steps'] != 0 or origin['seed']['specialist_config'] != manifest['specialist_config'] or \
            origin['seed']['initialization'] != trainer.INITIALIZATION or \
            shared._read_sealed(roots[0] / 'construction.json', 'mfm-public-conditional-foundation-construction-v1') != origin['construction'] or \
            not origin['construction']['control_updates_restored'] or origin['construction']['run_sha256'] != binding['run']:
        raise ValueError('public original fit seed or construction binding differs')
    control = trainer.verify_checkpoint(roots[0] / 'construction-controls' / 'checkpoint-00000001', binding=binding)
    original = trainer.verify_checkpoint(roots[0] / 'construction-controls' / 'checkpoint-00000000', binding=binding)
    if control['record_sha256'] != origin['construction']['control_checkpoint_sha256'] or \
            original['specialist_sha256'] != origin['seed']['specialist_sha256'] or \
            entries[0]['receipt']['specialist_sha256'] != origin['seed']['specialist_sha256']:
        raise ValueError('public original seed, construction controls or fit-zero lineage differs')
    return History(record, roots, latest_path=path)
