"""Export complete public conditional inputs through frozen Gemma source states.

Only the separately sealed source-input roster is opened. No human references,
decoder inputs, labels, teacher corpus or optimizer enters the backbone. Exact
processed inputs own distinct artifacts, even when their meanings are equivalent.
Each source is published atomically; only a verified complete bank is usable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import errno
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, require_sha256
from . import preflight_public_conditional_material as census_code
from . import train_v1_formation_specialist as shared


SOURCE_SCHEMA = 'mfm-public-conditional-source-input-v1'
EXPORT_SCHEMA = 'mfm-public-conditional-feature-export-v1'
FEATURE_SCHEMA = 'mfm-public-conditional-feature-source-v1'
PROGRESS_SCHEMA = 'mfm-public-conditional-feature-progress-v1'
BANK_SCHEMA = 'mfm-public-conditional-feature-bank-v1'
QUARANTINE_SCHEMA = 'mfm-public-conditional-feature-quarantine-v1'
SOURCE_FIELDS = {'schema', 'task', 'source_key', 'source_ids', 'attention_mask', 'split',
                 'case_ids', 'semantic_input_sha256s', 'source_payload_sha256s'}
STAGING_NAME = re.compile(r'\.staging-([0-9a-f]{32})\Z')
RECORD_NAME = re.compile(r'\.record-([0-9a-f]{32})\Z')
QUARANTINE_NAME = re.compile(r'(source|record)-([0-9a-f]{32})\Z')


def _within(root: Path, member: str) -> Path:
    if not member or '\\' in member or any(p in ('', '.', '..') or ':' in p for p in member.split('/')):
        raise ValueError('feature custody path escapes its root')
    path = root / member
    if Path(member).is_absolute() or any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('feature custody path or link escapes its root')
    return path


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field in public feature custody')
        result[key] = value
    return result


def _json(payload):
    return json.loads(payload, object_pairs_hook=_object)


def _read_sealed(path: Path, schema: str) -> dict:
    row = _json(path.read_bytes())
    if type(row) is not dict or row.get('schema') != schema:
        raise ValueError('public feature receipt schema differs')
    seal = row.get('record_sha256')
    if canonical_sha256({k: v for k, v in row.items() if k != 'record_sha256'}) != seal:
        raise ValueError('public feature receipt seal differs')
    return row


def _write_record(path: Path, body: dict, *, replace: bool = False) -> dict:
    if path.exists() and not replace:
        raise ValueError('immutable public feature receipt already exists')
    sealed = shared._seal(body)
    temporary = _within(path.parent, '.record-' + uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(canonical_json_bytes(sealed) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    # Failed/unpublished bytes remain for verified same-export quarantine.
    os.replace(temporary, path)
    return sealed


@contextmanager
def worker_lock(root: Path):
    """Nonblocking OS lock; abrupt worker exit releases it automatically."""
    path = _within(Path(root), 'worker.lock')
    if path.exists() and not path.is_file():
        raise ValueError('public feature worker lock is not a regular file')
    with path.open('a+b') as stream:
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise ValueError('public feature worker is already running') from exc
            raise ValueError('public feature worker lock is unavailable') from exc
        try:
            stream.seek(0)
            stream.write(b'0')
            stream.truncate(1)
            stream.flush()
            path.chmod(0o600)
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _scratch_inventory(root: Path, path: Path, kind: str) -> list[dict]:
    path = _within(root, path.relative_to(root).as_posix())
    if kind == 'record':
        if not path.is_file():
            raise ValueError('worker record scratch is not a regular file')
        return [{'path': '', 'size': path.stat().st_size, 'sha256': shared._digest(path)}]
    if not path.is_dir():
        raise ValueError('worker source scratch is not a directory')
    files = []
    for child in sorted(path.iterdir()):
        if child.name not in {'features.safetensors', 'source.json'} and not RECORD_NAME.fullmatch(child.name):
            raise ValueError('worker scratch has an unexpected path')
        child = _within(root, child.relative_to(root).as_posix())
        if not child.is_file():
            raise ValueError('worker scratch has a link or nested directory')
        files.append({'path': child.name, 'size': child.stat().st_size, 'sha256': shared._digest(child)})
    return files


def _quarantine_entries(root: Path) -> list[dict]:
    parent = _within(root, 'quarantine')
    if not parent.exists():
        return []
    if not parent.is_dir():
        raise ValueError('worker quarantine is not a custody directory')
    entries = []
    for path in sorted(parent.iterdir()):
        match = QUARANTINE_NAME.fullmatch(path.name)
        if not match:
            raise ValueError('worker quarantine has an unexpected path')
        kind, token = match.groups()
        entries.append({'quarantine_path': path.relative_to(root).as_posix(),
                        'original_path': ('sources/.staging-' if kind == 'source' else '.record-') + token,
                        'kind': kind, 'files': _scratch_inventory(root, path, kind)})
    return entries


def _verify_quarantine(root: Path, export: dict, *, allow_new: bool = False):
    entries = _quarantine_entries(root)
    path = _within(root, 'quarantine.json')
    if not path.exists():
        if entries and not allow_new:
            raise ValueError('worker quarantine custody log is missing')
        return None, entries
    record = _read_sealed(path, QUARANTINE_SCHEMA)
    prior = record.get('entries')
    if record.get('export_sha256') != export['record_sha256'] or type(prior) is not list or \
            record.get('uncommitted_payloads_never_accepted') is not True or \
            len({e.get('quarantine_path') for e in prior}) != len(prior) or \
            any(e not in entries for e in prior) or (not allow_new and prior != entries):
        raise ValueError('worker quarantine custody digest or binding differs')
    return record, entries


def recover_scratch(root: Path, sources: list[dict], export: dict):
    """Retain recognized uncommitted bytes; never promote them as features.

    The caller holds worker_lock. Atomic same-root renames can precede the
    custody-log publication; a later locked recovery inventories those retained
    names and finishes the log. Previously logged bytes must still match.
    """
    if _read_sealed(_within(root, 'export.json'), EXPORT_SCHEMA) != export:
        raise ValueError('worker scratch recovery export binding differs')
    prior, _ = _verify_quarantine(root, export, allow_new=True)
    pending = []
    root_names = {'export.json', 'sources', 'progress.json', 'bank.json', 'worker.lock',
                  'quarantine', 'quarantine.json'}
    for path in root.iterdir():
        if path.name in root_names:
            _within(root, path.name)
            continue
        match = RECORD_NAME.fullmatch(path.name)
        if not match:
            raise ValueError('public feature custody has an unlisted artifact')
        _scratch_inventory(root, path, 'record')
        pending.append((path, 'record-' + match.group(1)))
    parent = _within(root, 'sources')
    keys = {row['source_key'] for row in sources}
    if parent.exists():
        for path in parent.iterdir():
            if path.name in keys:
                _within(root, path.relative_to(root).as_posix())
                continue
            match = STAGING_NAME.fullmatch(path.name)
            if not match:
                raise ValueError('public feature custody has an unlisted source')
            _scratch_inventory(root, path, 'source')
            pending.append((path, 'source-' + match.group(1)))
    if pending:
        quarantine = _within(root, 'quarantine')
        quarantine.mkdir(mode=0o700, exist_ok=True)
        for source, name in pending:
            target = _within(root, 'quarantine/' + name)
            if target.exists():
                raise ValueError('worker quarantine refuses to overwrite retained scratch')
            source.rename(target)
    entries = _quarantine_entries(root)
    if entries or prior is not None:
        if prior is None or prior['entries'] != entries:
            prior = _write_record(_within(root, 'quarantine.json'), {
                'schema': QUARANTINE_SCHEMA, 'export_sha256': export['record_sha256'],
                'entries': entries, 'uncommitted_payloads_never_accepted': True,
                'qualified_for_product': False}, replace=prior is not None)
        print(json.dumps({'public_conditional_feature_quarantine_sha256': prior['record_sha256'],
                          'retained_uncommitted_payloads': len(entries)}), flush=True)
    return prior


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def _identities(values, name, *, digests=False):
    if type(values) is not list or not values or any(type(v) is not str or not v for v in values):
        raise ValueError('public source ' + name + ' ancestry is missing')
    if values != sorted(set(values)):
        raise ValueError('public source ' + name + ' ancestry repeats or is unordered')
    if digests:
        for value in values:
            if require_sha256(value, name) != value:
                raise ValueError('public source ancestry digest is not canonical')


def _assert_roster(census, sources):
    from hashlib import sha256
    digest = sha256()
    for row in sources:
        digest.update(canonical_json_bytes(row) + b'\n')
    if len(sources) != census['unique_source_inputs'] or digest.hexdigest() != census['source_inputs_sha256']:
        raise ValueError('complete processed public source roster differs from census')


def read_census(directory: Path, expected_sha256: str) -> tuple[dict, list[dict]]:
    """Validate the complete target-free roster without opening encoded pairs."""
    directory = Path(directory)
    require_sha256(expected_sha256, 'census_sha256')
    census = _read_sealed(_within(directory, 'receipt.json'), census_code.SCHEMA)
    if census['record_sha256'] != expected_sha256:
        raise ValueError('external public census digest differs')
    if census.get('task') != census_code.TASK or any(census.get(k) is not False for k in
            ('private_evidence_opened', 'model_weights_loaded', 'optimizer_created',
             'authority_written', 'qualified_for_product', 'final_payloads_opened')):
        raise ValueError('census is not the public source-only conditional preparation')
    for field in ('source_inputs_sha256', 'prepared_clone_receipt_sha256', 'prepared_base_weight_sha256',
                  'material_receipt_sha256', 'material_receipt_file_sha256', 'source_manifest_sha256'):
        require_sha256(census.get(field), field)
    if census.get('code_sha256') != shared._digest(Path(census_code.__file__)) or \
            census.get('source_processor_helper_sha256') != shared._digest(Path(shared.__file__)) or \
            census.get('processor_template_sha256') != canonical_template_digest():
        raise ValueError('public census processor or code binding differs')
    vocabulary = _positive(census.get('foundation_vocabulary'), 'foundation vocabulary')
    _positive(census.get('base_hidden_size'), 'base hidden size')
    maximum = _positive(census.get('max_source_tokens'), 'complete source budget')
    count = _positive(census.get('unique_source_inputs'), 'unique source inputs')
    source_path = _within(directory, 'source-inputs.jsonl')
    if shared._digest(source_path) != census['source_inputs_sha256']:
        raise ValueError('public source input file digest differs')
    sources = []
    keys, cases, semantic_splits = set(), set(), {}
    splits = Counter()
    with source_path.open('rb') as stream:
        for line in stream:
            row = _json(line)
            if type(row) is not dict or set(row) != SOURCE_FIELDS:
                raise ValueError('source-only input fields differ; targets are forbidden')
            if row['schema'] != SOURCE_SCHEMA or row['task'] != census_code.TASK:
                raise ValueError('public source schema or task differs')
            if row['split'] not in {'train', 'development'}:
                raise ValueError('public source split cannot open FINAL')
            ids, mask = row['source_ids'], row['attention_mask']
            if type(ids) is not list or not 0 < len(ids) <= maximum or any(type(v) is not int or not 0 <= v < vocabulary for v in ids):
                raise ValueError('complete public source IDs exceed geometry or vocabulary')
            if type(mask) is not list or len(mask) != len(ids) or any(type(v) is not int or v != 1 for v in mask):
                raise ValueError('complete public source attention mask differs')
            key = canonical_sha256({'task': census_code.TASK, 'source_ids': ids})
            if row['source_key'] != key:
                raise ValueError('public processed source key differs')
            if key in keys:
                raise ValueError('duplicate processed public source input')
            keys.add(key)
            _identities(row['case_ids'], 'case IDs')
            _identities(row['semantic_input_sha256s'], 'semantic input', digests=True)
            _identities(row['source_payload_sha256s'], 'original payload', digests=True)
            if cases.intersection(row['case_ids']):
                raise ValueError('public case identity belongs to multiple processed sources')
            cases.update(row['case_ids'])
            splits[row['split']] += len(row['case_ids'])
            for semantic in row['semantic_input_sha256s']:
                if semantic_splits.setdefault(semantic, row['split']) != row['split']:
                    raise ValueError('public semantic input crosses source splits')
            sources.append(row)
    statistics = census.get('statistics', {})
    if len(sources) != count or len(cases) != statistics.get('rows') or dict(splits) != statistics.get('split_counts'):
        raise ValueError('complete public source/case/split roster differs from census')
    _assert_roster(census, sources)
    return census, sources


def canonical_template_digest():
    from hashlib import sha256
    return sha256(shared.SOURCE_TEMPLATE.encode()).hexdigest()


def prepare_export(root: Path, census: dict, prepared: dict, *, torch_version: str,
                   transformers_version: str, resume: bool = False,
                   directory_reserved: bool = False) -> dict:
    """Bind the exporter to the rehashed role clone and immutable census."""
    if prepared.get('schema') != shared.CLONE_SCHEMA or prepared.get('role') != 'mfm' or \
            prepared.get('receipt_sha256') != census['prepared_clone_receipt_sha256']:
        raise ValueError('prepared role clone differs from public census')
    weights = [row.get('sha256') for row in prepared.get('files', []) if row.get('path') == 'model.safetensors']
    if weights != [census['prepared_base_weight_sha256']]:
        raise ValueError('prepared base weight differs from public census')
    if torch_version != census.get('torch_version') or transformers_version != census.get('transformers_version'):
        raise ValueError('public feature runtime differs from source census')
    binding = {'schema': EXPORT_SCHEMA, 'census_sha256': census['record_sha256'], 'census': census,
               'source_inputs_sha256': census['source_inputs_sha256'],
               'prepared_clone_receipt_sha256': prepared['receipt_sha256'],
               'prepared_base_weight_sha256': weights[0], 'exporter_sha256': shared._digest(Path(__file__)),
               'torch_version': torch_version, 'transformers_version': transformers_version,
               'representation': 'base.model.last_hidden_state', 'backbone_dtype': 'bfloat16',
               'backbone_device': 'cpu', 'base_frozen': True, 'language_head_used': False,
               'target_visible_to_backbone': False, 'private_evidence_opened': False,
               'optimizer_created': False, 'authority_written': False, 'final_payloads_opened': False,
               'qualified_for_product': False}
    root = Path(root)
    _within(root, 'export.json')
    if resume:
        if not root.is_dir():
            raise ValueError('public feature resume needs existing custody')
        record = _read_sealed(_within(root, 'export.json'), EXPORT_SCHEMA)
        if record != shared._seal(binding):
            raise ValueError('public feature export binding differs on resume')
        return record
    if directory_reserved:
        # The fresh CLI reserves this new directory exclusively before locking;
        # no previous export or unrelated custody may occupy it.
        if not root.is_dir() or {p.name for p in root.iterdir()} != {'worker.lock'} or \
                not _within(root, 'worker.lock').is_file():
            raise ValueError('fresh public feature custody is not exclusively reserved')
    else:
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
    return _write_record(_within(root, 'export.json'), binding)


def _validate_tensors(tensors, row, width):
    import torch
    if set(tensors) != {'states', 'input_ids', 'attention_mask'}:
        raise ValueError('public feature tensor fields differ')
    states, ids, mask = (tensors[k] for k in ('states', 'input_ids', 'attention_mask'))
    if states.dtype != torch.bfloat16 or ids.dtype != torch.int64 or mask.dtype != torch.int64 or \
            list(states.shape) != [1, len(row['source_ids']), width] or ids.shape != mask.shape or \
            list(ids.shape) != [1, len(row['source_ids'])] or not torch.isfinite(states).all():
        raise ValueError('public feature shape, dtype or finite geometry differs')
    if not torch.equal(ids, torch.tensor([row['source_ids']], dtype=torch.long)) or \
            not torch.equal(mask, torch.tensor([row['attention_mask']], dtype=torch.long)):
        raise ValueError('public feature input or mask differs from source census')


def read_source(root: Path, row: dict, export_digest: str, width: int):
    from safetensors.torch import load_file
    directory = _within(root, 'sources/' + row['source_key'])
    receipt = _read_sealed(_within(directory, 'source.json'), FEATURE_SCHEMA)
    path = _within(directory, 'features.safetensors')
    if shared._digest(path) != receipt.get('features_sha256'):
        raise ValueError('public feature tensor file digest differs')
    if receipt.get('source_input') != row or receipt.get('export_sha256') != export_digest or \
            receipt.get('serialized_tensors_exact') is not True or receipt.get('target_visible_to_backbone') is not False or \
            {p.name for p in directory.iterdir()} != {'source.json', 'features.safetensors'}:
        raise ValueError('public feature source receipt binding differs')
    tensors = load_file(str(path), device='cpu')
    _validate_tensors(tensors, row, width)
    if receipt.get('shape') != list(tensors['states'].shape):
        raise ValueError('public feature receipt geometry differs')
    return receipt, tensors


def _write_source(root, row, encoded, states, export_digest, width):
    import torch
    from safetensors.torch import load_file, save_file
    parent = _within(root, 'sources')
    parent.mkdir(mode=0o700, exist_ok=True)
    directory = _within(parent, row['source_key'])
    if directory.exists():
        raise ValueError('immutable public source feature already exists')
    tensors = {'states': states.detach().cpu().contiguous(),
               'input_ids': encoded['input_ids'].detach().cpu().contiguous(),
               'attention_mask': encoded['attention_mask'].detach().cpu().contiguous()}
    _validate_tensors(tensors, row, width)
    staging = _within(parent, '.staging-' + uuid4().hex)
    staging.mkdir(mode=0o700)
    path = _within(staging, 'features.safetensors')
    save_file(tensors, str(path))
    path.chmod(0o600)
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    restored = load_file(str(path), device='cpu')
    if any(not torch.equal(tensors[k], restored[k]) for k in tensors):
        raise ValueError('public feature serialization changed representations')
    receipt = _write_record(_within(staging, 'source.json'), {
        'schema': FEATURE_SCHEMA, 'source_input': row, 'export_sha256': export_digest,
        'features_sha256': shared._digest(path), 'shape': list(states.shape),
        'serialized_tensors_exact': True, 'target_visible_to_backbone': False})
    staging.rename(directory)
    return receipt


def _source_rows(root, sources, export):
    expected = {row['source_key'] for row in sources}
    parent = _within(root, 'sources')
    if parent.exists() and any(not p.is_dir() or p.name not in expected or p.is_symlink() for p in parent.iterdir()):
        raise ValueError('public feature custody has an unlisted or incomplete source')
    if any(p.name not in {'export.json', 'sources', 'progress.json', 'bank.json', 'worker.lock',
                         'quarantine', 'quarantine.json'} or p.is_symlink() for p in root.iterdir()):
        raise ValueError('public feature custody has an unlisted artifact')
    result = []
    missing = False
    for row in sources:
        if (parent / row['source_key']).exists():
            if missing:
                raise ValueError('public feature custody is not a complete ordered prefix')
            receipt, tensors = read_source(root, row, export['record_sha256'], export['census']['base_hidden_size'])
            del tensors
            result.append({'source_key': row['source_key'], 'source_receipt_sha256': receipt['record_sha256']})
        else:
            missing = True
    return result


def _source_chain(rows, export_digest):
    chain = export_digest
    for row in rows:
        chain = canonical_sha256({'previous': chain, **row})
    return chain


def _verify_progress(root, export, rows):
    path = _within(root, 'progress.json')
    if not path.exists():
        return None
    progress = _read_sealed(path, PROGRESS_SCHEMA)
    completed = progress.get('completed')
    total = export['census']['unique_source_inputs']
    if type(completed) is not int or not 0 <= completed <= len(rows) or \
            progress.get('export_sha256') != export['record_sha256'] or progress.get('total') != total or \
            progress.get('complete') is not (completed == total) or \
            progress.get('source_chain_sha256') != _source_chain(rows[:completed], export['record_sha256']):
        raise ValueError('public feature progress differs from committed custody')
    return progress


def _publish_progress(root, export, completed, total, chain):
    # Constant-size hash-chain snapshots avoid rewriting the whole growing bank
    # after every source. Source receipts retain the independently checked rows.
    progress = _write_record(_within(root, 'progress.json'), {
        'schema': PROGRESS_SCHEMA, 'export_sha256': export['record_sha256'],
        'source_chain_sha256': chain, 'completed': completed, 'total': total,
        'complete': completed == total}, replace=True)
    print(json.dumps({'public_conditional_feature_progress_sha256': progress['record_sha256'],
                      'completed': completed, 'total': total}), flush=True)
    return progress


def verify_bank(root: Path, census: dict, sources: list[dict], export: dict) -> dict:
    _assert_roster(census, sources)
    if _read_sealed(_within(root, 'export.json'), EXPORT_SCHEMA) != export or export.get('census') != census:
        raise ValueError('public feature bank export binding differs')
    rows = _source_rows(root, sources, export)
    progress = _verify_progress(root, export, rows)
    quarantine, _ = _verify_quarantine(root, export)
    bank = _read_sealed(_within(root, 'bank.json'), BANK_SCHEMA)
    if len(rows) != len(sources) or bank.get('complete') is not True or bank.get('sources') != rows or \
            bank.get('export_sha256') != export['record_sha256'] or bank.get('census_sha256') != census['record_sha256'] or \
            bank.get('source_inputs_sha256') != census['source_inputs_sha256'] or bank.get('unique_source_inputs') != len(sources) or \
            progress is None or progress.get('complete') is not True or bank.get('progress_sha256') != progress['record_sha256'] or \
            bank.get('quarantine_sha256') != (quarantine['record_sha256'] if quarantine else None) or \
            bank.get('final_payloads_opened') is not False or bank.get('qualified_for_product') is not False:
        raise ValueError('complete public feature bank roster or binding differs')
    return bank


def run_export(root: Path, census: dict, sources: list[dict], export: dict, base, *,
               stop_after_seconds: float, clock=perf_counter, started: float | None = None) -> dict:
    with worker_lock(Path(root)):
        return _run_export_locked(root, census, sources, export, base,
                                  stop_after_seconds=stop_after_seconds, clock=clock, started=started)


def _run_export_locked(root: Path, census: dict, sources: list[dict], export: dict, base, *,
                       stop_after_seconds: float, clock=perf_counter, started: float | None = None) -> dict:
    """Publish one complete source at a time; a partial bank is never complete."""
    import torch
    if not math.isfinite(stop_after_seconds) or not 0 < stop_after_seconds <= 8 * 3600:
        raise ValueError('public feature time budget must be positive and at most eight hours')
    started = clock() if started is None else started
    root = Path(root)
    _assert_roster(census, sources)
    if export.get('census') != census or _read_sealed(_within(root, 'export.json'), EXPORT_SCHEMA) != export:
        raise ValueError('public source export binding differs')
    quarantine = recover_scratch(root, sources, export)
    rows = _source_rows(root, sources, export)
    _verify_progress(root, export, rows)
    if (root / 'bank.json').exists():
        bank = verify_bank(root, census, sources, export)
        return {'complete': True, 'completed': len(rows), 'public_conditional_feature_bank_sha256': bank['record_sha256']}
    width = census['base_hidden_size']
    if getattr(base.config, 'model_type', None) != 'gemma4_unified' or not hasattr(base, 'model') or \
            base.config.text_config.hidden_size != width or base.config.text_config.vocab_size != census['foundation_vocabulary']:
        raise ValueError('public source backbone geometry differs from census')
    base.requires_grad_(False).eval()
    if any(p.dtype != torch.bfloat16 or p.device.type != 'cpu' or p.requires_grad for p in base.parameters()):
        raise ValueError('public source backbone dtype, placement or frozen state differs')
    committed = {row['source_key']: row for row in rows}
    chain = _source_chain(rows, export['record_sha256'])
    for row in sources:
        if row['source_key'] in committed:
            continue
        if clock() - started >= stop_after_seconds:
            break
        encoded = {'input_ids': torch.tensor([row['source_ids']], dtype=torch.long, device='cpu'),
                   'attention_mask': torch.tensor([row['attention_mask']], dtype=torch.long, device='cpu')}
        with torch.no_grad():
            states = base.model(**encoded, use_cache=False, return_dict=True).last_hidden_state
        receipt = _write_source(root, row, encoded, states, export['record_sha256'], width)
        del states, encoded
        committed[row['source_key']] = {'source_key': row['source_key'], 'source_receipt_sha256': receipt['record_sha256']}
        chain = canonical_sha256({'previous': chain, **committed[row['source_key']]})
        _publish_progress(root, export, len(committed), len(sources), chain)
    complete = len(committed) == len(sources)
    progress = _publish_progress(root, export, len(committed), len(sources), chain)
    result = {'complete': complete, 'completed': len(committed), 'total': len(sources),
              'public_conditional_feature_export_sha256': export['record_sha256'],
              'public_conditional_feature_progress_sha256': progress['record_sha256'],
              'qualified_for_product': False, 'seconds': clock() - started}
    if complete:
        rows = _source_rows(root, sources, export)
        bank = _write_record(_within(root, 'bank.json'), {
            'schema': BANK_SCHEMA, 'export_sha256': export['record_sha256'],
            'census_sha256': census['record_sha256'], 'source_inputs_sha256': census['source_inputs_sha256'],
            'sources': rows, 'unique_source_inputs': len(sources), 'complete': True,
            'progress_sha256': progress['record_sha256'],
            'quarantine_sha256': quarantine['record_sha256'] if quarantine else None,
            'final_payloads_opened': False, 'qualified_for_product': False})
        verify_bank(root, census, sources, export)
        result['public_conditional_feature_bank_sha256'] = bank['record_sha256']
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census-dir', type=Path, required=True)
    parser.add_argument('--census-sha256', required=True)
    parser.add_argument('--prepared-base-dir', type=Path, required=True)
    parser.add_argument('--prepared-base-receipt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-seconds', type=float, default=8 * 3600)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if not math.isfinite(args.stop_after_seconds) or not 0 < args.stop_after_seconds <= 8 * 3600:
        raise ValueError('public feature time budget must be positive and at most eight hours')
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'DO_NOT_TRACK': '1', 'WANDB_DISABLED': 'true'})
    shared._require_private_network_isolation()
    started = perf_counter()
    if not args.resume:
        args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    with worker_lock(args.output_dir):
        census, sources = read_census(args.census_dir, args.census_sha256)
        prepared = shared._prepared_base(args)
        import torch
        import transformers
        from transformers import AutoModelForMultimodalLM
        torch.set_num_threads(12)
        export = prepare_export(args.output_dir, census, prepared, torch_version=torch.__version__,
                                transformers_version=transformers.__version__, resume=args.resume,
                                directory_reserved=not args.resume)
        print(json.dumps({'public_conditional_feature_export_sha256': export['record_sha256']}), flush=True)
        recover_scratch(args.output_dir, sources, export)
        existing = _source_rows(args.output_dir, sources, export)
        _verify_progress(args.output_dir, export, existing)
        if (args.output_dir / 'bank.json').exists():
            bank = verify_bank(args.output_dir, census, sources, export)
            print(json.dumps({'public_conditional_feature_bank_sha256': bank['record_sha256'],
                              'complete': True, 'qualified_for_product': False}), flush=True)
            return
        base = AutoModelForMultimodalLM.from_pretrained(
            prepared['snapshot_path'], trust_remote_code=False, local_files_only=True,
            use_safetensors=True, dtype=torch.bfloat16, device_map={'': 'cpu'}, low_cpu_mem_usage=True)
        result = _run_export_locked(args.output_dir, census, sources, export, base,
                                    stop_after_seconds=args.stop_after_seconds, started=started)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
