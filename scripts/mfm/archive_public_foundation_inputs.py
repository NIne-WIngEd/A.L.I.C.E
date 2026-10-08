"""Archive the complete pinned public foundation inputs without loading a model.

The CLI accepts only the complete production roster and its existing licensed
material ZIP. A library fixture mode verifies real smaller banks but explicitly
labels them as reduced evidence. Failed scratch is retained; existing output is
never overwritten. Local archive readback does not establish durable custody.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
from time import perf_counter
from uuid import uuid4
import zipfile

from cognitive_kernel.canonical import canonical_json_bytes, require_sha256
from . import export_public_conditional_features as features
from . import preflight_public_conditional_material as census_code
from . import public_conditional_training as reader
from . import train_v1_formation_specialist as shared

MANIFEST_SCHEMA = 'mfm-public-foundation-input-manifest-v1'
RECEIPT_SCHEMA = 'mfm-public-foundation-input-archive-v1'
LEGAL_SHA256 = '857f472bd04e52208579ad3b195a6e317086cafa03cd9ae09c13416567ada512'
LEGAL_BYTES = 24_358_812
CHUNK_BYTES = 4 * 1024 * 1024
EMPTY_SHA256 = sha256(b'').hexdigest()
CENSUS_FILES = {'receipt.json', 'encoded-pairs.jsonl', 'source-inputs.jsonl'}
FALSE_FLAGS = ('private_evidence_opened', 'model_weights_loaded', 'optimizer_created',
               'authority_written', 'final_payloads_opened', 'qualified_for_product',
               'target_visible_to_backbone', 'learned_capability_established',
               'durable_custody_verified', 'new_source_acquisition')
WINDOWS_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                    *(f'LPT{i}' for i in range(1, 10))}


def _member_name(name: str, *, directory: bool = False) -> str:
    if type(name) is not str or not name or '\\' in name or name.startswith('/'):
        raise ValueError('unsafe archive relative path')
    if directory:
        if not name.endswith('/'):
            raise ValueError('unsafe archive directory name')
        name = name[:-1]
    parts = name.split('/')
    if any(p in {'', '.', '..'} or p[-1:] in {'.', ' '} or
           p.split('.')[0].upper() in WINDOWS_RESERVED or
           any(ord(c) < 32 or c in ':<>"|?*' for c in p) for p in parts):
        raise ValueError('unsafe archive relative path')
    return name + ('/' if directory else '')


def _no_links(path: Path):
    for item in (path, *path.parents):
        try:
            observed = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(observed.st_mode) or getattr(observed, 'st_file_attributes', 0) & \
                getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
            raise ValueError('unsafe custody link or reparse point')


def _regular(path: Path):
    _no_links(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('unsafe custody file is not regular')


def _stream_digest(stream):
    digest, size = sha256(), 0
    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _inventory(root: Path, prefix: str, allowed: set[str]) -> tuple[list[dict], dict[str, Path]]:
    _no_links(root)
    if not root.is_dir():
        raise ValueError('unsafe custody root is not a directory')
    rows = [{'path': prefix + '/', 'size': 0, 'sha256': EMPTY_SHA256, 'kind': 'directory'}]
    paths, seen = {}, set()
    def visit(directory):
        for child in sorted(directory.iterdir()):
            relative = child.relative_to(root).as_posix()
            _no_links(child)
            # The reader permits omission of this operational-only lock.
            if prefix == 'bank' and relative == 'worker.lock':
                _regular(child)
                continue
            directory_entry = stat.S_ISDIR(child.lstat().st_mode)
            member = _member_name(relative + ('/' if directory_entry else ''), directory=directory_entry)
            if member not in allowed:
                raise ValueError('unsafe or unknown public archive custody path: ' + member)
            seen.add(member)
            target = prefix + '/' + member
            if directory_entry:
                rows.append({'path': target, 'size': 0, 'sha256': EMPTY_SHA256, 'kind': 'directory'})
                visit(child)
            else:
                _regular(child)
                prior = _identity(child)
                with child.open('rb') as stream:
                    size, digest = _stream_digest(stream)
                if _identity(child) != prior or size != prior[2]:
                    raise ValueError('archive input changed during inventory')
                rows.append({'path': target, 'size': size, 'sha256': digest, 'kind': 'file'})
                paths[target] = child
    visit(root)
    if seen != allowed:
        raise ValueError('public archive custody roster is incomplete')
    return rows, paths


def _bank_members(prepared) -> set[str]:
    allowed = {'export.json', 'bank.json', 'progress.json', 'sources/'}
    for row in prepared.sources:
        prefix = 'sources/' + row['source_key'] + '/'
        allowed.update({prefix, prefix + 'source.json', prefix + 'features.safetensors'})
    quarantine, entries = features._verify_quarantine(prepared.feature_bank, prepared.export)
    if quarantine is not None:
        allowed.add('quarantine.json')
    if (prepared.feature_bank / 'quarantine').exists():
        allowed.add('quarantine/')
    for entry in entries:
        path = entry['quarantine_path']
        if entry['kind'] == 'source':
            allowed.add(path + '/')
            allowed.update(path + '/' + row['path'] for row in entry['files'])
        else:
            allowed.add(path)
    return allowed


def _zip_members(archive):
    entries, namespace = archive.infolist(), {}
    for entry in entries:
        _member_name(entry.filename, directory=entry.is_dir())
        if entry.orig_filename != entry.filename:
            raise ValueError('unsafe archive name encoding')
        key = entry.filename.rstrip('/').casefold()
        if key in namespace:
            raise ValueError('duplicate archive member name')
        namespace[key] = entry.is_dir()
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if mode not in {0, stat.S_IFREG, stat.S_IFDIR} or entry.flag_bits & 1:
            raise ValueError('unsafe linked, special or encrypted archive entry')
        if mode == stat.S_IFDIR and not entry.is_dir() or mode == stat.S_IFREG and entry.is_dir():
            raise ValueError('unsafe archive member kind')
    for key in namespace:
        parts = key.split('/')
        if any(namespace.get('/'.join(parts[:i])) is False for i in range(1, len(parts))):
            raise ValueError('unsafe archive file/directory collision')
    return entries


def _legal_archive(path: Path, expected_sha256: str):
    require_sha256(expected_sha256, 'legal material archive external digest')
    _regular(path)
    if shared._digest(path) != expected_sha256:
        raise ValueError('external legal material archive digest differs')
    with zipfile.ZipFile(path, 'r') as archive:
        entries = _zip_members(archive)
        if not entries:
            raise ValueError('legal material archive is empty')
        for entry in entries:
            with archive.open(entry, 'r') as stream:
                size, _ = _stream_digest(stream)  # EOF verifies the original CRC.
            if size != entry.file_size:
                raise ValueError('legal material archive byte count differs')


def _counts(prepared):
    groups = {'train': set(), 'development': set()}
    labels, tokens = Counter(), Counter()
    for pair in prepared.pairs:
        groups[pair['split']].add(pair['semantic_input_sha256'])
        tokens[pair['split']] += len(pair['labels'])
        if pair['split'] == 'train':
            labels.update(pair['labels'])
    return {'pairs': len(prepared.pairs),
            'pairs_by_split': {'train': len(prepared.train), 'development': len(prepared.development)},
            'semantic_groups_by_split': {k: len(v) for k, v in groups.items()},
            'unique_source_inputs': len(prepared.sources), 'label_tokens_by_split': dict(tokens),
            'train_positive_label_types': len(labels), 'base_hidden_size': prepared.census['base_hidden_size'],
            'foundation_vocabulary': prepared.census['foundation_vocabulary']}


def _production_roster(counts):
    expected = {'pairs': 40868, 'pairs_by_split': {'train': 36597, 'development': 4271},
                'semantic_groups_by_split': {'train': 4356, 'development': 506},
                'unique_source_inputs': 4862, 'base_hidden_size': 3840,
                'foundation_vocabulary': 262144, 'train_positive_label_types': 3341}
    return all(counts.get(key) == value for key, value in expected.items())


def _zip_info(row):
    entry = zipfile.ZipInfo(row['path'], date_time=(1980, 1, 1, 0, 0, 0))
    entry.create_system = 3
    is_directory = row['kind'] == 'directory'
    entry.external_attr = ((stat.S_IFDIR | 0o700) if is_directory else (stat.S_IFREG | 0o600)) << 16
    if is_directory:
        entry.external_attr |= 0x10
    entry.compress_type = zipfile.ZIP_STORED if is_directory or row['path'].startswith('legal-material/') \
        else zipfile.ZIP_DEFLATED
    # Custom ZipInfo does not inherit ZipFile.compresslevel in open(). This
    # spelling remains compatible with Python before the 3.13 public property.
    entry._compresslevel = 1
    entry.file_size = row['size']
    return entry


def _write_zip(path: Path, rows: list[dict], paths: dict[str, Path], manifest_bytes: bytes):
    with zipfile.ZipFile(path, 'x', allowZip64=True, compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for row in rows:
            with archive.open(_zip_info(row), 'w', force_zip64=True) as target:
                if row['kind'] == 'directory':
                    continue
                if row['path'] == 'manifest.json':
                    target.write(manifest_bytes)
                    continue
                original = paths[row['path']]
                _regular(original)
                before = _identity(original)
                digest, size = sha256(), 0
                with original.open('rb') as source:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b''):
                        target.write(chunk)
                        size += len(chunk)
                        digest.update(chunk)
                if size != row['size'] or digest.hexdigest() != row['sha256'] or _identity(original) != before:
                    raise ValueError('archive input changed during packaging')
    with path.open('r+b') as stream:
        stream.flush()
        os.fsync(stream.fileno())


def verify_archive(path: Path, files: list[dict]):
    """Read every byte through ZIP CRC checking and compare independent hashes."""
    with zipfile.ZipFile(path, 'r') as archive:
        entries = _zip_members(archive)
        expected = {row['path']: row for row in files}
        if len(expected) != len(files) or {row.filename for row in entries} != set(expected):
            raise ValueError('archive readback roster differs')
        for entry in entries:
            row = expected[entry.filename]
            if row['kind'] not in {'file', 'directory'} or entry.is_dir() is not (row['kind'] == 'directory'):
                raise ValueError('archive readback member kind differs')
            with archive.open(entry, 'r') as stream:
                size, digest = _stream_digest(stream)
            if size != row['size'] or size != entry.file_size or digest != row['sha256']:
                raise ValueError('archive readback size or SHA256 differs')


def _all_inputs(prepared, census_dir, legal_path, legal_digest):
    census_rows, census_paths = _inventory(census_dir, 'census', CENSUS_FILES)
    bank_rows, bank_paths = _inventory(prepared.feature_bank, 'bank', _bank_members(prepared))
    _legal_archive(legal_path, legal_digest)
    legal_name = 'legal-material/sha256-' + legal_digest + '.zip'
    rows = [*census_rows, *bank_rows,
            {'path': 'legal-material/', 'size': 0, 'sha256': EMPTY_SHA256, 'kind': 'directory'},
            {'path': legal_name, 'size': legal_path.stat().st_size, 'sha256': legal_digest, 'kind': 'file'}]
    paths = {**census_paths, **bank_paths, legal_name: legal_path}
    return sorted(rows, key=lambda row: row['path']), paths


def _fsync_directory(path):
    if hasattr(os, 'O_DIRECTORY'):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_exclusive(staged, target):
    # Same filesystem link is atomic and refuses an existing target on both OSes.
    # Retain the staged hard link as evidence; it does not duplicate payload bytes.
    os.link(staged, target)
    _fsync_directory(target.parent)


def archive_inputs(census_dir: Path, census_sha256: str, feature_bank: Path,
                   feature_bank_sha256: str, legal_material_archive: Path,
                   legal_material_archive_sha256: str, output_dir: Path, *,
                   require_full_roster: bool = True) -> dict:
    """Fixture mode is a library diagnostic; the production CLI never exposes it."""
    if type(require_full_roster) is not bool:
        raise ValueError('archive production roster requirement must be boolean')
    census_dir, feature_bank = Path(census_dir), Path(feature_bank)
    legal_path, output_dir = Path(legal_material_archive), Path(output_dir)
    for path in (census_dir, feature_bank, legal_path, output_dir):
        _no_links(path)
    if output_dir.exists():
        raise ValueError('public archive output already exists; preserve original scratch')
    destination = output_dir.resolve()
    for original in (census_dir.resolve(), feature_bank.resolve(), legal_path.resolve()):
        if destination.is_relative_to(original) or original.is_relative_to(destination):
            raise ValueError('archive output overlaps original custody inputs')
    started = perf_counter()
    with features.worker_lock(feature_bank):
        prepared = reader.load_public_training(census_dir, census_sha256, feature_bank, feature_bank_sha256)
        counts = _counts(prepared)
        full_roster = _production_roster(counts)
        if require_full_roster and not full_roster:
            raise ValueError('complete production roster required before public archive write')
        if require_full_roster and (legal_material_archive_sha256 != LEGAL_SHA256 or
                                    legal_path.stat().st_size != LEGAL_BYTES):
            raise ValueError('production legal material archive external pin or bytes differ')
        rows, paths = _all_inputs(prepared, census_dir, legal_path, legal_material_archive_sha256)
        manifest = shared._seal({
            'schema': MANIFEST_SCHEMA, 'census_sha256': census_sha256,
            'feature_bank_sha256': feature_bank_sha256, 'export_sha256': prepared.export['record_sha256'],
            'progress_sha256': prepared.bank['progress_sha256'],
            'quarantine_sha256': prepared.bank.get('quarantine_sha256'),
            'legal_material_archive_sha256': legal_material_archive_sha256,
            'counts': counts, 'full_production_roster': full_roster, 'source_only': False,
            'licensed_material_included_unchanged': True, 'files': rows,
            'code_sha256': {name: shared._digest(Path(module.__file__)) for name, module in
                            (('archive_worker', sys.modules[__name__]), ('public_training_reader', reader),
                             ('feature_exporter', features), ('census_preflight', census_code),
                             ('shared_formation', shared))},
            'runtime': {'python_version': platform.python_version(),
                        'torch_version': importlib.metadata.version('torch'),
                        'safetensors_version': importlib.metadata.version('safetensors'),
                        'platform': platform.platform()},
            **dict.fromkeys(FALSE_FLAGS, False)})
        manifest_bytes = canonical_json_bytes(manifest) + b'\n'
        files = [*rows, {'path': 'manifest.json', 'size': len(manifest_bytes),
                         'sha256': sha256(manifest_bytes).hexdigest(), 'kind': 'file'}]
        output_dir.mkdir(mode=0o700, exist_ok=False)
        staged = output_dir / ('.archive-' + uuid4().hex + '.zip')
        _write_zip(staged, files, paths, manifest_bytes)
        verify_archive(staged, files)
        # The OS worker lock protects the producer. Detect all other input changes
        # as well, including exact receipt bytes, legal ZIP and unexpected extras.
        checked = reader.load_public_training(census_dir, census_sha256, feature_bank, feature_bank_sha256)
        checked_rows, _ = _all_inputs(checked, census_dir, legal_path, legal_material_archive_sha256)
        if checked_rows != rows or _counts(checked) != counts:
            raise ValueError('archive input changed before publication')
        archive_digest = shared._digest(staged)
        final_name = 'sha256-' + archive_digest + '.zip'
        result = shared._seal({
            **{key: value for key, value in manifest.items() if key not in {'schema', 'record_sha256', 'files'}},
            'schema': RECEIPT_SCHEMA, 'manifest_record_sha256': manifest['record_sha256'],
            'manifest_file_sha256': sha256(manifest_bytes).hexdigest(), 'files': files,
            'archive_sha256': archive_digest, 'archive_bytes': staged.stat().st_size,
            'archive_filename': final_name, 'archive_crc_sha256_readback_verified': True,
            'original_inputs_reverified_before_publication': True, 'complete': True,
            'elapsed_seconds': perf_counter() - started})
        receipt_staged = output_dir / ('.receipt-' + uuid4().hex + '.json')
        with receipt_staged.open('xb') as stream:
            stream.write(canonical_json_bytes(result) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        _publish_exclusive(staged, output_dir / final_name)
        _publish_exclusive(receipt_staged, output_dir / 'receipt.json')
        return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census-dir', type=Path, required=True)
    parser.add_argument('--census-sha256', required=True)
    parser.add_argument('--feature-bank', type=Path, required=True)
    parser.add_argument('--feature-bank-sha256', required=True)
    parser.add_argument('--legal-material-archive', type=Path, required=True)
    parser.add_argument('--legal-material-archive-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    result = archive_inputs(args.census_dir, args.census_sha256, args.feature_bank,
                            args.feature_bank_sha256, args.legal_material_archive,
                            args.legal_material_archive_sha256, args.output_dir)
    print(json.dumps({key: result[key] for key in ('archive_sha256', 'archive_bytes',
          'archive_filename', 'record_sha256', 'counts', 'elapsed_seconds')}, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    main()
