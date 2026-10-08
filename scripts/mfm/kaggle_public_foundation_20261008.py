"""Prepare a private exact-byte Kaggle package; this command never uploads.

Only public construction inputs and code are eligible. Root supplies completed
bank and durable-restore pins, and decides when to execute this preparation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from . import kaggle_public_foundation_20261008_runtime as runtime

LEGAL_ARCHIVE = 'legal-material/sha256-857f472bd04e52208579ad3b195a6e317086cafa03cd9ae09c13416567ada512.zip'


def git_bytes(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True).stdout


def source_archive(repo, revision, destination):
    if re.fullmatch('[0-9a-f]{40}', revision) is None or git_bytes(repo, 'rev-parse', revision + '^{commit}').decode().strip() != revision:
        raise ValueError('an exact source commit is required')
    tree = git_bytes(repo, 'ls-tree', '-r', '--name-only', revision).decode().splitlines()
    context = {'README.md', 'SECURITY.md', '.gitattributes'}
    names = sorted(name for name in tree if (name.endswith('.py') and name.startswith(('src/cognitive_kernel/', 'scripts/mfm/'))) or
                   name in context or (re.fullmatch(r'(LICENSE|NOTICE|COPYING)(\..*)?', name.rsplit('/', 1)[-1], re.IGNORECASE) and
                                       ('/' not in name or name.startswith(('src/cognitive_kernel/', 'scripts/mfm/')))))
    if 'scripts/mfm/train_public_conditional_foundation.py' not in names:
        raise ValueError('source revision lacks full public trainer')
    files = []
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            runtime.relative_name(name)
            contents = git_bytes(repo, 'show', revision + ':' + name)
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, contents)
            files.append({'path': name, 'bytes': len(contents), 'sha256': sha256(contents).hexdigest()})
    return files


def copy_archive(source, destination, format_name, expected_pin):
    if runtime.digest(source) != runtime.pin(expected_pin):
        raise ValueError('externally supplied archive SHA256 differs')
    if destination.exists() or destination.is_symlink():
        raise ValueError('package input cannot be overwritten')
    with Path(source).open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
    descriptor = runtime.describe_archive(destination, format_name)
    if descriptor['sha256'] != expected_pin:
        raise ValueError('copied package archive differs')
    return {**descriptor, 'filename': destination.name}


def prepare(args):
    if args.timeout_seconds is not None and (type(args.timeout_seconds) is not int or args.timeout_seconds <= 0):
        raise ValueError('custom execution timeout must be an explicit positive integer')
    if not re.fullmatch('mkrayanyan/[a-z0-9]+(?:-[a-z0-9]+)*', args.dataset_id) or \
            not re.fullmatch('mkrayanyan/[a-z0-9]+(?:-[a-z0-9]+)*', args.kernel_id):
        raise ValueError('explicit authenticated owner dataset/kernel identity required')
    if args.dataset_id == args.kernel_id:
        raise ValueError('dataset and kernel identities must differ')
    for name in ('census_sha256', 'feature_bank_sha256', 'prepared_clone_sha256', 'config_sha256', 'durable_receipt_sha256'):
        runtime.pin(getattr(args, name))
    recipe = {name: getattr(args, name) for name in runtime.INTEGER_FIELDS + runtime.FLOAT_FIELDS}
    runtime.validate_recipe(recipe)
    layout = {'census': args.census_subdir, 'bank': args.bank_subdir}
    for name in layout.values():
        runtime.relative_name(name)
    runtime.relative_name(args.material_manifest_name)
    if '/' in args.material_manifest_name:
        raise ValueError('material manifest must remain at archive root')
    if layout['census'] == layout['bank'] or any(a.startswith(b + '/') for a, b in ((layout['census'], layout['bank']), (layout['bank'], layout['census']))):
        raise ValueError('census and bank need disjoint archive directories')
    destination = Path(args.output_dir)
    if destination.exists() or destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError('package requires a new owned output directory')
    destination.mkdir(mode=0o700)
    dataset = destination / 'dataset'
    kernel = destination / 'kernel'
    dataset.mkdir(mode=0o700)
    kernel.mkdir(mode=0o700)
    material = copy_archive(args.material_archive, dataset / 'public-material.mfmblob', args.material_format, args.material_archive_sha256)
    metadata = copy_archive(args.metadata_archive, dataset / 'processor-metadata.mfmblob', 'zip', args.metadata_archive_sha256)
    dependencies = copy_archive(args.dependency_archive, dataset / 'offline-dependencies.mfmblob', 'zip', args.dependency_archive_sha256)
    if {row['path'] for row in dependencies['entries']} != {'dependency-manifest.json', runtime.TRANSFORMERS_WHEEL['filename']}:
        raise ValueError('mandatory Transformers-only dependency archive roster differs')
    with zipfile.ZipFile(dataset / dependencies['filename']) as archive:
        dependency_manifest = json.loads(archive.read('dependency-manifest.json'))
    dependency_manifest_sha256 = runtime.verify_dependency_manifest(dependency_manifest)
    material_names = [row['path'] for row in material['entries']]
    allowed_files = {args.material_manifest_name, LEGAL_ARCHIVE}
    if any(name not in allowed_files and not any(name.startswith(prefix + '/') for prefix in layout.values()) for name in material_names):
        raise ValueError('material archive contains an unlisted public input root')
    if not any(name.startswith(layout['census'] + '/') for name in material_names) or \
            not any(name.startswith(layout['bank'] + '/') for name in material_names):
        raise ValueError('material archive lacks requested full census/bank layout')
    with runtime.opened_archive(dataset / material['filename'], material['format']) as archive:
        _, members, _ = runtime.archive_entries(archive, material['format'])
        if args.material_manifest_name not in members:
            raise ValueError('complete sealed material manifest is missing')
        member = members[args.material_manifest_name]
        stream = archive.open(member) if material['format'] == 'zip' else archive.extractfile(member)
        with stream:
            input_manifest = json.loads(stream.read())
    material_manifest_sha256 = runtime.verify_material_binding(input_manifest, {'census_sha256': args.census_sha256, 'feature_bank_sha256': args.feature_bank_sha256})
    if any(PureName(name) in {'model.safetensors', 'pytorch_model.bin'} for name in material_names):
        raise ValueError('base model weight payload is forbidden')
    if {row['path'] for row in metadata['entries']} != set(runtime.METADATA_NAMES) | {'mfm-clone-576486.json', 'subset-manifest.json'}:
        raise ValueError('original metadata ZIP asset roster differs')
    source_path = dataset / 'reviewed-source.mfmblob'
    files = source_archive(args.source_root, args.source_revision, source_path)
    code = {**runtime.describe_archive(source_path, 'zip'), 'filename': source_path.name}
    launcher = Path(runtime.__file__).read_text(encoding='utf-8')
    manifest = runtime.seal({'schema': runtime.SCHEMA, 'purpose': 'construction-measurement-only',
        'dataset_id': args.dataset_id, 'dataset_version_attachment': 'fresh-immutable-id-exact-byte-enforced',
        'kernel_id': args.kernel_id, 'source_revision': args.source_revision, 'source_files': files,
        'source_legal_context': {'included_paths': [row['path'] for row in files if not row['path'].endswith('.py')],
            'license_file_present': any(re.fullmatch(r'LICENSE(\..*)?', row['path'].rsplit('/', 1)[-1], re.IGNORECASE) for row in files),
            'new_license_or_redistribution_grant_created': False},
        'launcher_sha256': sha256(launcher.encode()).hexdigest(),
        'pins': {name: getattr(args, name) for name in ('census_sha256', 'feature_bank_sha256', 'prepared_clone_sha256', 'config_sha256')},
        'durable_input_receipt_sha256': args.durable_receipt_sha256,
        'archives': {'material': material, 'metadata': metadata, 'code': code, 'dependencies': dependencies}, 'layout': layout,
        'material_manifest_name': args.material_manifest_name, 'material_manifest_sha256': material_manifest_sha256,
        'dependency_manifest_sha256': dependency_manifest_sha256,
        'recipe': recipe,
        'requirements': {'transformers': '5.17.0'},
        'boundary': {'full_source_roster': 4862, 'full_train_pairs': 36597, 'full_development_pairs': 4271,
            'source_features_only': True, 'base_weights_included': False, 'private_material_included': False,
            'final_payloads_included': False, 'model_code_changed': False, 'fit_started': False}})
    runtime.write_record(dataset / runtime.MANIFEST_NAME, manifest)
    runtime.write_record(dataset / 'dataset-metadata.json', {'id': args.dataset_id,
        'title': args.dataset_id.split('/')[1].replace('-', ' ')[:50], 'licenses': [{'name': 'other'}],
        'description': 'Private transport of full admitted public E2E conditional foundation material, frozen source-only states, exact code and Gemma metadata. Original E2E CC-BY-SA-4.0, source code terms and Gemma terms/notices remain applicable. No base model weights or personal material. Input archives and manifests enforce exact bytes.'})
    main = launcher + '\n\nif __name__ == "__main__":\n    run(' + repr(manifest['record_sha256']) + ')\n'
    (kernel / 'main.py').write_text(main, encoding='utf-8', newline='\n')
    runtime.write_record(kernel / 'kernel-metadata.json', {'id': args.kernel_id,
        'title': args.kernel_id.split('/')[1].replace('-', ' '), 'code_file': 'main.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'docker_image_pinning_type': 'original', 'dataset_sources': [args.dataset_id],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
    launch = ['kaggle', 'kernels', 'push', '-p', str(kernel)]
    if args.timeout_seconds is not None:
        launch += ['-t', str(args.timeout_seconds)]
    launch += ['--accelerator', 'NvidiaTeslaT4']
    commands = {'create_private_dataset': ['kaggle', 'datasets', 'create', '-p', str(dataset), '-t', '-r', 'skip'],
                'read_dataset_status': ['kaggle', 'datasets', 'status', args.dataset_id, '--format', 'json'],
                'launch_full_construction': launch,
                'read_kernel_status': ['kaggle', 'kernels', 'status', args.kernel_id],
                'retrieve_outputs': ['kaggle', 'kernels', 'output', args.kernel_id, '-p', str(destination / 'returned-output')]}
    receipt = runtime.seal({'schema': 'mfm-kaggle-full-construction-preparation-v1',
        'transport_sha256': manifest['record_sha256'], 'dataset_id': args.dataset_id, 'kernel_id': args.kernel_id,
        'kernel_main_sha256': runtime.digest(kernel / 'main.py'), 'commands': commands,
        'uploaded': False, 'submitted': False, 'model_code_changed': False})
    runtime.write_record(destination / 'preparation.json', receipt)
    return receipt


def PureName(name):
    return runtime.relative_name(name).name


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'output-dir', 'material-archive', 'metadata-archive', 'dependency-archive'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-revision', 'material-archive-sha256', 'metadata-archive-sha256', 'dependency-archive-sha256',
                 'census-sha256', 'feature-bank-sha256', 'prepared-clone-sha256', 'config-sha256',
                 'durable-receipt-sha256', 'dataset-id', 'kernel-id', 'census-subdir', 'bank-subdir', 'material-manifest-name'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--material-format', choices=('zip', 'tar'), required=True)
    parser.add_argument('--timeout-seconds', type=int,
                        help='Optional custom run limit, justified by the reviewed launch policy; omission uses the provider default.')
    for name in runtime.INTEGER_FIELDS:
        parser.add_argument('--' + name.replace('_', '-'), type=int, required=True)
    for name in runtime.FLOAT_FIELDS:
        parser.add_argument('--' + name.replace('_', '-'), type=float, required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(prepare(arguments()), sort_keys=True))
