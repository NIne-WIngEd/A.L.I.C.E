"""Prepare tiny reviewed fit code; never upload or read large bank archives."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from . import kaggle_public_foundation_20261008 as construction
from . import kaggle_public_foundation_fit_20261008_runtime as runtime

PREPARATION_SCHEMA = 'mfm-kaggle-full-fit-preparation-v1'
REQUIRED_CODE = {'scripts/mfm/kaggle_public_foundation_fit_20261008.py',
    'scripts/mfm/kaggle_public_foundation_fit_20261008_runtime.py',
    'scripts/mfm/kaggle_public_foundation_20261008_runtime.py',
    'scripts/mfm/train_public_conditional_foundation.py', 'scripts/mfm/public_checkpoint_history.py'}


def main_source(transport_sha256):
    runtime.common.pin(transport_sha256)
    shared = Path(runtime.common.__file__).read_text(encoding='utf-8')
    fit = Path(runtime.__file__).read_text(encoding='utf-8')
    return ('import types\nMFM_COMMON = types.ModuleType("mfm_pinned_common")\n'
            + 'exec(compile(' + repr(shared) + ', "mfm-pinned-common", "exec"), MFM_COMMON.__dict__)\n'
            + 'exec(compile(' + repr(fit) + ', "mfm-pinned-fit", "exec"), globals())\n'
            + '\nif __name__ == "__main__":\n    run(' + repr(transport_sha256) + ')\n')


def kernel_metadata(kernel_id, dataset_id, input_dataset_id, kernel_sources):
    runtime.owned_id(kernel_id); runtime.owned_id(dataset_id)
    if input_dataset_id != runtime.INPUT_DATASET or len({kernel_id, dataset_id, input_dataset_id}) != 3:
        raise ValueError('new private code/kernel must reuse the exact original input dataset')
    if type(kernel_sources) is not list or len(set(kernel_sources)) != len(kernel_sources):
        raise ValueError('explicit unique prior source identities required')
    for identity in kernel_sources:
        runtime.owned_id(identity)
    return {'id': kernel_id, 'title': kernel_id.split('/')[1].replace('-', ' ')[:50], 'code_file': 'main.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4', 'docker_image_pinning_type': 'original',
        'dataset_sources': [input_dataset_id, dataset_id], 'competition_sources': [], 'model_sources': [], 'kernel_sources': kernel_sources}


def assert_executing_source(repo, revision):
    # Package preparation must execute the reviewed exact Git bytes too.
    for module in (runtime, runtime.common, construction):
        relative = 'scripts/mfm/' + Path(module.__file__).name
        expected = construction.git_bytes(repo, 'show', revision + ':' + relative)
        if Path(module.__file__).read_bytes() != expected:
            raise ValueError('executing packager/runtime differs from the exact reviewed source commit')
    expected = construction.git_bytes(repo, 'show', revision + ':scripts/mfm/' + Path(__file__).name)
    if Path(__file__).read_bytes() != expected:
        raise ValueError('executing fit packager differs from the exact source commit')


def prepare(args):
    common = runtime.common
    runtime.owned_id(args.dataset_id); runtime.owned_id(args.kernel_id)
    inputs = runtime.accepted_input_manifest(args.input_manifest)
    session = json.loads(Path(args.session_json).read_bytes()) if args.session_json else \
        {'stop_step': args.stop_step, 'session_index': 0, 'resume_history': None, 'source_roots': []}
    recipe = {name: getattr(args, name) for name in common.INTEGER_FIELDS + common.FLOAT_FIELDS}
    runtime.validate_program({'purpose': 'full-roster-public-fit', 'recipe': recipe, 'session': session})
    if re.fullmatch('[0-9a-f]{40}', args.source_revision) is None:
        raise ValueError('exact full Git source revision required')
    assert_executing_source(args.source_root, args.source_revision)
    kernel_sources = [row['kernel_id'] for row in session['source_roots']]
    metadata = kernel_metadata(args.kernel_id, args.dataset_id, runtime.INPUT_DATASET, kernel_sources)
    destination = Path(args.output_dir)
    if destination.exists() or destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError('tiny package requires a new owned directory')
    destination.mkdir(mode=0o700)
    dataset, kernel = destination / 'dataset', destination / 'kernel'
    dataset.mkdir(mode=0o700); kernel.mkdir(mode=0o700)
    code_path = dataset / 'reviewed-fit-source.mfmblob'
    files = construction.source_archive(args.source_root, args.source_revision, code_path)
    if not REQUIRED_CODE.issubset({row['path'] for row in files}):
        raise ValueError('exact committed source lacks the fit/history bootstrap')
    code = {**common.describe_archive(code_path, 'zip'), 'filename': code_path.name}
    manifest = common.seal({'schema': runtime.SCHEMA, 'purpose': 'full-roster-public-fit',
        'dataset_id': args.dataset_id, 'kernel_id': args.kernel_id, 'source_revision': args.source_revision,
        'input_dataset': {'id': runtime.INPUT_DATASET, 'numeric_id': 12455297, 'version_number': 1, 'is_private': True,
            'manifest_name': common.MANIFEST_NAME, 'manifest_file_sha256': runtime.INPUT_MANIFEST_FILE_SHA256,
            'manifest_sha256': runtime.INPUT_MANIFEST_SHA256},
        'source_files': files, 'code_archive': code, 'recipe': recipe, 'session': session,
        'source_legal_context': {'included_paths': [row['path'] for row in files if not row['path'].endswith('.py')],
            'new_license_or_redistribution_grant_created': False},
        'launcher_sources': {'common_sha256': common.digest(Path(runtime.common.__file__)),
            'fit_sha256': common.digest(Path(runtime.__file__))},
        'boundary': {'full_train_pairs': 36597, 'full_development_pairs': 4271, 'full_parameter_count': 267542272,
            'source_features_only': True, 'base_weights_included': False, 'private_material_included': False,
            'fit_started': False, 'durable_custody': False, 'learned_capability_claimed': False}})
    common.write_record(dataset / runtime.MANIFEST_NAME, manifest)
    common.write_record(dataset / 'dataset-metadata.json', {'id': args.dataset_id,
        'title': args.dataset_id.split('/')[1].replace('-', ' ')[:50], 'licenses': [{'name': 'other'}],
        'description': 'Private exact public source capsule for the fixed full-roster conditional foundation fit. Existing public source notices and terms apply. Reuses admitted source-only input features; no base weights or personal payloads.'})
    (kernel / 'main.py').write_text(main_source(manifest['record_sha256']), encoding='utf-8', newline='\n')
    common.write_record(kernel / 'kernel-metadata.json', metadata)
    inventory = common.file_inventory(destination)
    receipt = common.seal({'schema': PREPARATION_SCHEMA, 'dataset_id': args.dataset_id, 'input_dataset_id': runtime.INPUT_DATASET,
        'kernel_id': args.kernel_id, 'kernel_main_sha256': common.digest(kernel / 'main.py'),
        'transport_sha256': manifest['record_sha256'], 'transport_file_sha256': common.digest(dataset / runtime.MANIFEST_NAME),
        'source_revision': args.source_revision, 'dataset_dir': str(dataset.resolve()), 'kernel_dir': str(kernel.resolve()),
        'manifest_filename': runtime.MANIFEST_NAME, 'files': inventory, 'recipe': recipe, 'session': session,
        'storage_forecast': runtime.storage_plan(manifest, inputs, code_bytes=code['uncompressed_bytes']),
        'custom_runtime_timeout_sent': False, 'uploaded': False, 'submitted': False, 'fit_started': False,
        'large_input_archives_opened': False, 'bank_rebuilt': False, 'learned_capability_claimed': False})
    common.write_record(destination / 'preparation.json', receipt)
    return receipt


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'output-dir', 'input-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-revision', 'dataset-id', 'kernel-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--session-json', type=Path)
    parser.add_argument('--stop-step', type=int, default=6864)
    for name in runtime.common.INTEGER_FIELDS:
        parser.add_argument('--' + name.replace('_', '-'), type=int, required=True)
    for name in runtime.common.FLOAT_FIELDS:
        parser.add_argument('--' + name.replace('_', '-'), type=float, required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(prepare(arguments()), sort_keys=True))
