"""Pinned public-only Kaggle transport into unchanged full construction.

This module is embedded verbatim in the generated Kaggle script. It provisions
only the approved exact offline Transformers overlay, with no network installs,
base-weight loading, producer resizing, or fitting.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from email.parser import BytesParser
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import traceback
from uuid import uuid4
import zipfile

SCHEMA = 'mfm-kaggle-full-public-construction-transport-v1'
MANIFEST_NAME = 'mfm-construction-transport.json'
METADATA_NAMES = ('.gitattributes', 'README.md', 'config.json', 'generation_config.json',
                  'processor_config.json', 'tokenizer.json', 'tokenizer_config.json')
INTEGER_FIELDS = ('epochs', 'batch_size', 'save_every_steps', 'seed', 'logit_chunk_tokens', 'generation_tokens')
FLOAT_FIELDS = ('learning_rate', 'weight_decay', 'beta1', 'beta2', 'adam_eps')
LEGAL_SHA256 = '857f472bd04e52208579ad3b195a6e317086cafa03cd9ae09c13416567ada512'
TRANSFORMERS_WHEEL = {'name': 'transformers', 'version': '5.17.0',
    'filename': 'transformers-5.17.0-py3-none-any.whl', 'bytes': 12295140,
    'sha256': '78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801',
    'source_url': 'https://files.pythonhosted.org/packages/e8/d0/c502b60d684adbd98a8dc7d5bb866842772b816ac4354e4608be240041ae/transformers-5.17.0-py3-none-any.whl'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def seal(value):
    if 'record_sha256' in value:
        raise ValueError('cannot replace an existing seal')
    return {**value, 'record_sha256': sha256(canonical(value)).hexdigest()}


def pin(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('explicit lowercase SHA256 pin required')
    return value


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('pinned input must be a regular file')
    result = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def relative_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('unsafe archive path')
    pieces = name.split('/')
    if name.startswith('/') or any(piece in {'', '.', '..'} for piece in pieces):
        raise ValueError('unsafe archive path')
    return PurePosixPath(name)


@contextmanager
def opened_archive(path, format_name):
    if format_name == 'zip':
        with zipfile.ZipFile(path) as archive:
            yield archive
    elif format_name == 'tar':
        with tarfile.open(path, 'r:*') as archive:
            yield archive
    else:
        raise ValueError('only explicit zip or tar formats are supported')


def archive_entries(archive, format_name):
    rows, files, names, directories = [], {}, set(), []
    members = archive.infolist() if format_name == 'zip' else archive.getmembers()
    for member in members:
        name = member.orig_filename if format_name == 'zip' else member.name
        directory = member.is_dir() if format_name == 'zip' else member.isdir()
        clean_name = name[:-1] if directory and name.endswith('/') else name
        relative_name(clean_name)
        if clean_name in names:
            raise ValueError('duplicate archive entry')
        names.add(clean_name)
        if format_name == 'zip':
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR) or member.flag_bits & 1:
                raise ValueError('archive links, special or encrypted files are forbidden')
            if directory != (file_type == stat.S_IFDIR) and file_type != 0:
                raise ValueError('archive entry type differs')
            size = member.file_size
        else:
            if not (member.isfile() or directory):
                raise ValueError('archive links or special files are forbidden')
            size = member.size
        if directory:
            directories.append(clean_name)
            continue
        if any(parent.as_posix() in files for parent in PurePosixPath(clean_name).parents if parent.as_posix() != '.'):
            raise ValueError('archive file shadows a directory')
        if any(existing.startswith(clean_name + '/') for existing in names):
            raise ValueError('archive file shadows a directory')
        if type(size) is not int or size < 0:
            raise ValueError('invalid archive file size')
        rows.append({'path': clean_name, 'size': size})
        files[clean_name] = member
    if not rows:
        raise ValueError('empty archive cannot carry construction')
    return sorted(rows, key=lambda row: row['path']), files, sorted(directories)


def describe_archive(path, format_name):
    archive_sha256 = digest(path)
    with opened_archive(path, format_name) as archive:
        entries, _, directories = archive_entries(archive, format_name)
    return {'format': format_name, 'sha256': archive_sha256, 'bytes': Path(path).stat().st_size,
            'entries': entries, 'directories': directories, 'uncompressed_bytes': sum(row['size'] for row in entries)}


def extract_verified(path, destination, descriptor):
    destination = Path(destination)
    if digest(path) != pin(descriptor['sha256']) or Path(path).stat().st_size != descriptor['bytes']:
        raise ValueError('mounted archive byte pin differs')
    if destination.exists() or destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError('archive extraction requires a new owned directory')
    with opened_archive(path, descriptor['format']) as archive:
        entries, files, directories = archive_entries(archive, descriptor['format'])
        if entries != descriptor['entries'] or directories != descriptor['directories'] or sum(row['size'] for row in entries) != descriptor['uncompressed_bytes']:
            raise ValueError('archive roster differs from pinned transport')
        destination.mkdir(mode=0o700)
        for name in directories:
            destination.joinpath(*relative_name(name).parts).mkdir(parents=True, exist_ok=True, mode=0o700)
        for row in entries:
            target = destination.joinpath(*relative_name(row['path']).parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.parent.is_symlink() or target.exists():
                raise ValueError('unsafe extraction target')
            member = files[row['path']]
            stream = archive.open(member) if descriptor['format'] == 'zip' else archive.extractfile(member)
            with stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
            if target.stat().st_size != row['size']:
                raise ValueError('extracted file size differs')


def read_manifest(path, expected_pin):
    value = json.loads(Path(path).read_bytes())
    if type(value) is not dict:
        raise ValueError('transport manifest is not an object')
    body = {key: item for key, item in value.items() if key != 'record_sha256'}
    if value.get('schema') != SCHEMA or value.get('record_sha256') != pin(expected_pin) or seal(body) != value:
        raise ValueError('transport manifest seal differs')
    return value


def locate_manifest(input_root, expected_pin):
    paths = sorted(Path(input_root).rglob(MANIFEST_NAME))
    if len(paths) != 1 or paths[0].is_symlink():
        raise ValueError('exactly one transport manifest is required')
    read_manifest(paths[0], expected_pin)
    return paths[0]


def validate_recipe(recipe):
    if type(recipe) is not dict or set(recipe) != set(INTEGER_FIELDS + FLOAT_FIELDS):
        raise ValueError('all measurement fields must be explicitly specified')
    for name in INTEGER_FIELDS:
        if type(recipe[name]) is not int or (name != 'seed' and recipe[name] < 1):
            raise ValueError('invalid explicit integer recipe field')
    if recipe['batch_size'] >= 36597 or recipe['generation_tokens'] > 8192:
        raise ValueError('measurement requires different next batch and full position contract')
    for name in FLOAT_FIELDS:
        if type(recipe[name]) not in (float, int) or not math.isfinite(recipe[name]):
            raise ValueError('nonfinite measurement optimizer field')
    if recipe['learning_rate'] <= 0 or recipe['weight_decay'] < 0 or recipe['adam_eps'] <= 0 or \
            not 0 <= recipe['beta1'] < 1 or not 0 <= recipe['beta2'] < 1:
        raise ValueError('invalid explicit AdamW recipe')
    return recipe


def verify_dependencies(required, actual):
    if type(required) is not dict or required.get('transformers') != '5.17.0':
        raise ValueError('actual census AutoProcessor Transformers 5.17.0 must be pinned')
    for name, version in required.items():
        if not isinstance(version, str) or not version or actual.get(name) != version:
            raise ValueError(f'required runtime dependency differs: {name} expected {version}, actual {actual.get(name)}')


def verify_dependency_manifest(record):
    fields = {'schema', 'python_specifier', 'base_requirements', 'wheels', 'forbidden_replacement', 'record_sha256'}
    if type(record) is not dict or set(record) != fields or record['schema'] != 'mfm-kaggle-offline-dependency-overlay-v1':
        raise ValueError('offline dependency manifest schema or fields differ')
    if seal({key: value for key, value in record.items() if key != 'record_sha256'}) != record or \
            record['python_specifier'] != '>=3.10.0' or record['wheels'] != [TRANSFORMERS_WHEEL] or \
            record['forbidden_replacement'] != ['torch', 'torchvision', 'torchaudio']:
        raise ValueError('only the original pinned Transformers wheel may be provisioned')
    if type(record['base_requirements']) is not list or not record['base_requirements'] or \
            any(not isinstance(value, str) or not value for value in record['base_requirements']):
        raise ValueError('original wheel base requirements are required')
    return pin(record['record_sha256'])


def validate_base_requirements(requirements, actual, python_version, python_specifier):
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    if python_version not in SpecifierSet(python_specifier):
        raise ValueError('actual Python is incompatible with the pinned wheel')
    normalized = {canonicalize_name(name): value for name, value in actual.items()}
    for value in requirements:
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        version = normalized.get(canonicalize_name(requirement.name))
        if requirement.url or version is None or version not in requirement.specifier:
            raise ValueError(f'actual base dependency does not satisfy pinned wheel: {value}; actual {version}')


def wheel_preflight(root, record, actual):
    from packaging.requirements import Requirement
    from packaging.tags import parse_tag, sys_tags
    verify_dependency_manifest(record)
    row = record['wheels'][0]
    path = Path(root) / row['filename']
    if digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
        raise ValueError('offline wheel original publisher byte pin differs')
    describe_archive(path, 'zip')
    with zipfile.ZipFile(path) as archive:
        metadata_bytes = archive.read('transformers-5.17.0.dist-info/METADATA')
        metadata = BytesParser().parsebytes(metadata_bytes)
        wheel = BytesParser().parsebytes(archive.read('transformers-5.17.0.dist-info/WHEEL'))
    requirements = [value for value in metadata.get_all('Requires-Dist', [])
                    if not Requirement(value).marker or Requirement(value).marker.evaluate({'extra': ''})]
    if metadata['Name'] != 'transformers' or metadata['Version'] != '5.17.0' or \
            metadata['Requires-Python'] != record['python_specifier'] or requirements != record['base_requirements']:
        raise ValueError('actual wheel metadata differs from the sealed dependency recipe')
    tags = set().union(*(parse_tag(tag) for tag in wheel.get_all('Tag', [])))
    if not tags or not tags.intersection(sys_tags()):
        raise ValueError('original wheel tags are incompatible with the actual Python/ABI/platform')
    validate_base_requirements(requirements, actual, platform.python_version(), record['python_specifier'])
    return {'wheel_sha256': row['sha256'], 'wheel_metadata_sha256': sha256(metadata_bytes).hexdigest(),
            'requires_python': metadata['Requires-Python'], 'requires_dist': requirements,
            'wheel_tags': sorted(str(tag) for tag in tags), 'base_requirements_passed': True,
            'torch_cuda_replaced': False}


def overlay_install_argv(overlay, wheels):
    return [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
            '--disable-pip-version-check', '--target', str(overlay)] + [str(path) for path in wheels]


def trainer_environment(code, overlay):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join((str(overlay), str(Path(code) / 'src'), str(code))),
               PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PIP_NO_INDEX='1')
    return env


def provision_overlay(root, overlay, record, actual, result_root):
    if overlay.exists() or overlay.is_symlink():
        raise ValueError('offline overlay requires a new owned target')
    preflight = wheel_preflight(root, record, actual)
    write_record(result_root / 'dependency-preflight.json', seal({'schema': 'mfm-kaggle-offline-dependency-preflight-v1',
        'dependency_manifest_sha256': record['record_sha256'], 'actual_python': platform.python_version(), **preflight}))
    env = dict(os.environ, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    argv = overlay_install_argv(overlay, [root / row['filename'] for row in record['wheels']])
    with (result_root / 'offline-overlay-install.log').open('x', encoding='utf-8') as stream:
        process = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f'pinned offline overlay exited {process.returncode}; original install log retained')
    sys.path.insert(0, str(overlay))
    importlib.invalidate_caches()
    after = package_versions()
    verify_dependencies({'transformers': '5.17.0'}, after)
    for name in ('torch', 'torchvision', 'torchaudio'):
        if after.get(name) != actual.get(name):
            raise ValueError('offline overlay changed a protected Torch/CUDA dependency')
    import transformers
    if transformers.__version__ != '5.17.0' or not Path(transformers.__file__).resolve().is_relative_to(overlay.resolve()):
        raise ValueError('actual Transformers import does not resolve to the pinned owned overlay')
    import torchvision
    import PIL
    files = file_inventory(overlay)
    write_record(result_root / 'dependency-overlay.json', seal({'schema': 'mfm-kaggle-offline-dependency-overlay-execution-v1',
        'dependency_manifest_sha256': record['record_sha256'], 'installation_argv': argv,
        'actual_package_versions': after, 'transformers_module': str(Path(transformers.__file__).resolve()),
        'torchvision_import_version': torchvision.__version__, 'pillow_import_version': PIL.__version__,
        'overlay_files': files, 'overlay_inventory_sha256': sha256(canonical(files)).hexdigest(),
        'internet_installer': False, 'torch_cuda_replaced': False}))
    return after


def verify_material_binding(record, expected):
    if type(record) is not dict or record.get('schema') != 'mfm-public-foundation-input-manifest-v1':
        raise ValueError('full input manifest schema differs')
    body = {key: value for key, value in record.items() if key != 'record_sha256'}
    if seal(body) != record:
        raise ValueError('full input manifest seal differs')
    for name in ('census_sha256', 'feature_bank_sha256'):
        if record.get(name) != pin(expected[name]):
            raise ValueError('full input manifest external pin differs')
    if record.get('legal_material_archive_sha256') != LEGAL_SHA256 or record.get('full_production_roster') is not True or \
            record.get('source_only') is not False or record.get('licensed_material_included_unchanged') is not True:
        raise ValueError('full public material role or legal binding differs')
    counts = record.get('counts', {})
    for name, value in {'pairs': 40868, 'pairs_by_split': {'train': 36597, 'development': 4271},
                        'semantic_groups_by_split': {'train': 4356, 'development': 506},
                        'unique_source_inputs': 4862, 'base_hidden_size': 3840,
                        'foundation_vocabulary': 262144}.items():
        if counts.get(name) != value:
            raise ValueError('full public material count/geometry differs')
    for name in ('private_evidence_opened', 'model_weights_loaded', 'optimizer_created', 'authority_written',
                 'final_payloads_opened', 'qualified_for_product', 'target_visible_to_backbone',
                 'learned_capability_established', 'new_source_acquisition'):
        if record.get(name) is not False:
            raise ValueError('full public input boundary flag differs')
    return pin(record['record_sha256'])


def verify_extracted_material(root, transport):
    name = str(relative_name(transport['material_manifest_name']))
    record = json.loads((root / name).read_bytes())
    if verify_material_binding(record, transport['pins']) != transport['material_manifest_sha256']:
        raise ValueError('extracted full input manifest binding differs')
    descriptor = transport['archives']['material']
    rows = record.get('files')
    if type(rows) is not list or any(type(row) is not dict or set(row) != {'path', 'size', 'sha256', 'kind'} for row in rows):
        raise ValueError('full input file roster is invalid')
    roster = [{'path': row['path'], 'size': row['size'], 'kind': 'file'} for row in descriptor['entries'] if row['path'] != name]
    roster += [{'path': path + '/', 'size': 0, 'kind': 'directory'} for path in descriptor['directories']]
    if sorted(({key: row[key] for key in ('path', 'size', 'kind')} for row in rows), key=lambda row: row['path']) != sorted(roster, key=lambda row: row['path']):
        raise ValueError('full input manifest roster differs from exact archive')
    for row in rows:
        path = root.joinpath(*relative_name(row['path'].rstrip('/') if row['kind'] == 'directory' else row['path']).parts)
        expected = pin(row['sha256'])
        if row['kind'] == 'directory':
            if not path.is_dir() or path.is_symlink() or expected != sha256(b'').hexdigest():
                raise ValueError('full input directory custody differs')
        elif path.stat().st_size != row['size'] or digest(path) != expected:
            raise ValueError('full input file custody differs')
    return record['record_sha256']


def trainer_argv(manifest, *, material, metadata, output, allocation_ref):
    recipe = validate_recipe(manifest['recipe'])
    layout = manifest['layout']
    census = Path(material).joinpath(*relative_name(layout['census']).parts)
    bank = Path(material).joinpath(*relative_name(layout['bank']).parts)
    argv = ['-m', 'scripts.mfm.train_public_conditional_foundation',
            '--census-dir', str(census), '--feature-bank', str(bank), '--output-dir', str(output),
            '--prepared-base-dir', str(Path(metadata) / 'processor-metadata'),
            '--prepared-base-receipt', str(Path(metadata) / 'mfm-clone-576486.json'),
            '--provider', 'Kaggle-free-public', '--allocation-ref', allocation_ref, '--device', 'cuda:0']
    for name in ('census_sha256', 'feature_bank_sha256', 'prepared_clone_sha256', 'config_sha256'):
        argv.extend(['--' + name.replace('_', '-'), pin(manifest['pins'][name])])
    for name in INTEGER_FIELDS + FLOAT_FIELDS:
        argv.extend(['--' + name.replace('_', '-'), str(recipe[name])])
    return argv + ['--measure-only']


def write_record(path, record):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def package_versions():
    versions = {}
    for name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy', 'Pillow', 'torchvision',
                 'torchaudio', 'librosa', 'sentencepiece', 'huggingface-hub', 'packaging', 'PyYAML',
                 'regex', 'typer', 'tqdm', 'pip'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_inventory(root):
    return [{'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in sorted(Path(root).rglob('*')) if path.is_file()]


def run(expected_pin):
    started = time.monotonic()
    manifest_path = locate_manifest('/kaggle/input', expected_pin)
    manifest = read_manifest(manifest_path, expected_pin)
    if manifest['purpose'] != 'construction-measurement-only':
        raise ValueError('transport only authorizes full construction measurement')
    allocation_ref = manifest['kernel_id'] + '/allocation-' + uuid4().hex
    result_root = Path('/kaggle/working') / ('mfm-full-construction-' + expected_pin[:16])
    result_root.mkdir(mode=0o700, exist_ok=False)
    try:
        versions = package_versions()
        inventory = {'schema': 'mfm-kaggle-actual-runtime-v1', 'transport_sha256': expected_pin,
                     'allocation_ref': allocation_ref, 'python': platform.python_version(),
                     'platform': platform.platform(), 'package_versions': versions,
                     'working_disk': dict(zip(('total', 'used', 'free'), shutil.disk_usage(result_root))),
                     'private_evidence_opened': False, 'base_weights_loaded': False,
                     'saved_output_quota_verified': False}
        write_record(result_root / 'runtime-before-imports.json', seal(inventory))
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        temp_parent = Path('/kaggle/temp')
        temp_parent.mkdir(exist_ok=True)
        extraction_root = temp_parent / ('mfm-construction-' + uuid4().hex)
        extraction_root.mkdir(mode=0o700)
        needed = sum(row['uncompressed_bytes'] for row in manifest['archives'].values())
        if shutil.disk_usage(extraction_root).free <= needed:
            raise ValueError('actual extraction filesystem lacks input bytes')
        locations = {}
        for role, descriptor in manifest['archives'].items():
            blob = manifest_path.parent / descriptor['filename']
            relative_name(descriptor['filename'])
            locations[role] = extraction_root / role
            extract_verified(blob, locations[role], descriptor)
        dependency_root = locations['dependencies']
        dependency_record = json.loads((dependency_root / 'dependency-manifest.json').read_bytes())
        if verify_dependency_manifest(dependency_record) != manifest['dependency_manifest_sha256'] or \
                {path.name for path in dependency_root.iterdir()} != {'dependency-manifest.json', TRANSFORMERS_WHEEL['filename']}:
            raise ValueError('mandatory dependency archive binding/roster differs')
        overlay = extraction_root / 'owned-python-overlay'
        inventory['overlay_package_versions'] = provision_overlay(dependency_root, overlay, dependency_record, versions, result_root)
        verify_dependencies(manifest['requirements'], inventory['overlay_package_versions'])
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise ValueError('actual CUDA allocation required; no CPU fallback')
        inventory['actual_devices'] = [{'ordinal': index, 'name': torch.cuda.get_device_name(index),
            'capability': list(torch.cuda.get_device_capability(index)),
            'total_memory_bytes': torch.cuda.get_device_properties(index).total_memory}
            for index in range(torch.cuda.device_count())]
        inventory['selected_device'] = 'cuda:0'
        inventory['pooled_memory_or_distributed_execution'] = False
        write_record(result_root / 'runtime-actual-allocation.json', seal(inventory))
        material_manifest = verify_extracted_material(locations['material'], manifest)
        write_record(result_root / 'input-custody.json', seal({'schema': 'mfm-kaggle-full-input-custody-v1',
            'transport_sha256': expected_pin, 'material_manifest_sha256': material_manifest,
            'all_extracted_archive_files_verified': True, 'base_weights_loaded': False, 'private_evidence_opened': False}))
        metadata = locations['metadata']
        if {path.name for path in metadata.iterdir()} != set(METADATA_NAMES) | {'mfm-clone-576486.json', 'subset-manifest.json'}:
            raise ValueError('metadata subset archive roster differs')
        processor_dir = metadata / 'processor-metadata'
        processor_dir.mkdir(mode=0o700)
        for name in METADATA_NAMES:
            (metadata / name).rename(processor_dir / name)
        code = locations['code']
        for row in manifest['source_files']:
            path = code.joinpath(*relative_name(row['path']).parts)
            if digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
                raise ValueError('source file custody differs')
        argv = [sys.executable] + trainer_argv(manifest, material=locations['material'], metadata=metadata,
                                              output=result_root / 'trainer', allocation_ref=allocation_ref)
        env = trainer_environment(code, overlay)
        write_record(result_root / 'trainer-invocation.json', seal({'schema': 'mfm-kaggle-public-invocation-v1',
            'transport_sha256': expected_pin, 'argv': argv, 'source_revision': manifest['source_revision'],
            'cublas_workspace_config': env['CUBLAS_WORKSPACE_CONFIG'], 'mode': 'full-construction-measurement-only'}))
        with (result_root / 'trainer.log').open('x', encoding='utf-8') as stream:
            process = subprocess.run(argv, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        outputs = file_inventory(result_root / 'trainer') if (result_root / 'trainer').exists() else []
        receipt = seal({'schema': 'mfm-kaggle-full-construction-execution-v1', 'transport_sha256': expected_pin,
            'allocation_ref': allocation_ref, 'trainer_exit_code': process.returncode,
            'seconds': time.monotonic() - started, 'trainer_outputs': outputs,
            'trainer_output_bytes': sum(row['bytes'] for row in outputs),
            'working_disk_after': dict(zip(('total', 'used', 'free'), shutil.disk_usage(result_root))),
            'saved_output_quota_verified': False, 'fit_started': False, 'learned_capability_claimed': False,
            'durable_custody': False})
        write_record(result_root / 'construction-execution.json', receipt)
        if process.returncode != 0:
            raise RuntimeError(f'full construction trainer exited {process.returncode}; original trainer.log retained')
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return receipt
    except Exception as error:
        write_record(result_root / 'launcher-failure.json', seal({'schema': 'mfm-kaggle-construction-failure-v1',
            'transport_sha256': expected_pin, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'seconds': time.monotonic() - started,
            'fit_started': False, 'durable_custody': False, 'original_evidence_preserved': True}))
        raise
