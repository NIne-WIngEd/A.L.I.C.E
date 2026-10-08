"""Public transport and invocation controls; no model or learning simulation."""
import importlib
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest
import zipfile
import tarfile
import io
import hashlib
import subprocess
import shutil
import sys
from types import SimpleNamespace


def isolated_trainer_help(code, *, fallback=None):
    """Run real exported code; owned modules may only resolve inside the capsule.

    External installed libraries remain real. No trainer/model module is stubbed,
    and --help exits before data reads, model construction or training.
    """
    program = r'''
import hashlib, importlib.abc, importlib.machinery, importlib.resources
import json, pathlib, runpy, sys
code = pathlib.Path(sys.argv[1]).resolve()
roots = {'scripts', 'alice_memory', 'alice_vault', 'alice_conversation',
         'alice_information', 'alice_evolution', 'alice_capability_profiles',
         'cognitive_kernel', 'product_family'} | {
    p.name if p.is_dir() else p.stem for p in (code / 'src').iterdir()}
assert not roots.intersection(sys.modules), 'owned project module was preloaded'
sys.path[:0] = [str(code / 'src'), str(code)]
if len(sys.argv) > 2:
    sys.path.append(sys.argv[2])
class CapsuleOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in roots:
            return None
        search = [str(code / 'src'), str(code)] if path is None else list(path)
        assert all(pathlib.Path(p).resolve().is_relative_to(code) for p in search)
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None:
            raise ModuleNotFoundError('No module named ' + repr(fullname), name=fullname)
        if spec.origin not in (None, 'namespace'):
            assert pathlib.Path(spec.origin).resolve().is_relative_to(code)
        return spec
sys.meta_path.insert(0, CapsuleOnly())
sys.argv = ['scripts.mfm.train_public_conditional_foundation', '--help']
try:
    runpy.run_module('scripts.mfm.train_public_conditional_foundation', run_name='__main__', alter_sys=True)
except SystemExit as error:
    if error.code != 0:
        raise
origins = {}
for name, module in sorted(sys.modules.items()):
    if name.split('.')[0] in roots:
        origin = getattr(module, '__file__', None)
        if origin is not None:
            origin = pathlib.Path(origin).resolve()
            assert origin.is_relative_to(code), (name, str(origin))
            origins[name] = origin.relative_to(code).as_posix()
resource = importlib.resources.files('cognitive_kernel').joinpath('canonical.py').read_bytes()
print('CAPSULE_IMPORT_PROOF=' + json.dumps({'origins': origins,
    'canonical_resource_sha256': hashlib.sha256(resource).hexdigest(),
    'unused_vault_imported': 'alice_vault' in sys.modules,
    'unused_conversation_imported': 'alice_conversation' in sys.modules}, sort_keys=True))
'''
    argv = [sys.executable, '-I', '-B', '-c', program, str(code)]
    if fallback is not None:
        argv.append(str(fallback))
    return subprocess.run(argv, cwd=code, capture_output=True, text=True)


class KaggleConstructionTransportTests(unittest.TestCase):
    def module(self):
        name = 'scripts.mfm.kaggle_public_foundation_20261008_runtime'
        self.assertIsNotNone(importlib.util.find_spec(name), 'full-construction transport runtime is missing')
        return importlib.import_module(name)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, entries):
        path = self.root / 'input.blob'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, value in entries:
                member = zipfile.ZipInfo(name)
                member.filename = name
                archive.writestr(member, value)
        return path

    def recipe(self):
        return {'epochs': 1, 'batch_size': 1, 'save_every_steps': 1, 'seed': 20261008,
                'logit_chunk_tokens': 16, 'generation_tokens': 128,
                'learning_rate': 0.0001, 'weight_decay': 0.0,
                'beta1': 0.9, 'beta2': 0.999, 'adam_eps': 1e-8}

    def test_verified_archive_extracts_exact_bytes_to_new_owned_directory(self):
        m = self.module()
        archive = self.archive([('census/census.json', b'{"public":true}'), ('bank/sources/a/features.safetensors', b'fixture')])
        descriptor = m.describe_archive(archive, 'zip')
        destination = self.root / 'extracted'
        m.extract_verified(archive, destination, descriptor)
        self.assertEqual((destination / 'census/census.json').read_bytes(), b'{"public":true}')
        self.assertEqual((destination / 'bank/sources/a/features.safetensors').read_bytes(), b'fixture')
        self.assertEqual(descriptor['uncompressed_bytes'], 22)
        with self.assertRaises(ValueError):
            m.extract_verified(archive, destination, descriptor)

    def test_changed_archive_is_rejected_before_output_directory_creation(self):
        m = self.module()
        archive = self.archive([('data.json', b'good')])
        descriptor = m.describe_archive(archive, 'zip')
        archive.write_bytes(archive.read_bytes() + b'changed')
        destination = self.root / 'out'
        with self.assertRaises(ValueError):
            m.extract_verified(archive, destination, descriptor)
        self.assertFalse(destination.exists())

    def test_archive_escape_duplicate_and_symlink_rejected(self):
        m = self.module()
        for name in ('../escape', '/absolute', 'C:/drive', 'a\\escape', 'a/../escape', 'a//escape'):
            with self.subTest(name=name):
                archive = self.archive([(name, b'invalid')])
                with self.assertRaises(ValueError):
                    m.describe_archive(archive, 'zip')
        with self.assertWarns(UserWarning):
            archive = self.archive([('same', b'a'), ('same', b'b')])
        with self.assertRaises(ValueError):
            m.describe_archive(archive, 'zip')
        with zipfile.ZipFile(archive, 'w') as output:
            member = zipfile.ZipInfo('link')
            member.create_system = 3
            member.external_attr = (0o120777 << 16)
            output.writestr(member, '../escape')
        with self.assertRaises(ValueError):
            m.describe_archive(archive, 'zip')

    def test_tar_regular_files_work_and_hardlinks_rejected(self):
        m = self.module()
        archive = self.root / 'input.tarblob'
        with tarfile.open(archive, 'w:gz') as output:
            member = tarfile.TarInfo('bank/file.json')
            member.size = 2
            output.addfile(member, io.BytesIO(b'{}'))
        descriptor = m.describe_archive(archive, 'tar')
        m.extract_verified(archive, self.root / 'tar-out', descriptor)
        self.assertEqual((self.root / 'tar-out/bank/file.json').read_bytes(), b'{}')
        with tarfile.open(archive, 'w') as output:
            member = tarfile.TarInfo('hardlink')
            member.type = tarfile.LNKTYPE
            member.linkname = 'elsewhere'
            output.addfile(member)
        with self.assertRaises(ValueError):
            m.describe_archive(archive, 'tar')

    def test_empty_custody_directories_are_preserved(self):
        m = self.module()
        archive = self.root / 'directories.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('bank/quarantine/', b'')
            output.writestr('bank/bank.json', b'{}')
        descriptor = m.describe_archive(archive, 'zip')
        m.extract_verified(archive, self.root / 'out', descriptor)
        self.assertTrue((self.root / 'out/bank/quarantine').is_dir())

    def test_material_binding_rejects_incomplete_or_changed_sealed_manifest(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'verify_material_binding'), 'sealed full material binding validator is missing')
        body = self.material_binding_body()
        expected = {'census_sha256': '1' * 64, 'feature_bank_sha256': '2' * 64}
        sealed = m.seal(body)
        self.assertEqual(m.verify_material_binding(sealed, expected), sealed['record_sha256'])
        for change in ({'full_production_roster': False}, {'census_sha256': '8' * 64}, {'authority_written': True},
                       {'counts': {**body['counts'], 'unique_source_inputs': 144}}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    m.verify_material_binding(m.seal({**body, **change}), expected)
        with self.assertRaises(ValueError):
            m.verify_material_binding({**sealed, 'source_only': True}, expected)

    def material_binding_body(self):
        # Literal contract-envelope fixture; it provides no real bank and never
        # reaches the unchanged full-material trainer or establishes learning.
        return {'schema': 'mfm-public-foundation-input-manifest-v1', 'census_sha256': '1' * 64,
            'feature_bank_sha256': '2' * 64, 'legal_material_archive_sha256': '857f472bd04e52208579ad3b195a6e317086cafa03cd9ae09c13416567ada512',
            'counts': {'pairs': 40868, 'pairs_by_split': {'train': 36597, 'development': 4271},
                'semantic_groups_by_split': {'train': 4356, 'development': 506}, 'unique_source_inputs': 4862,
                'base_hidden_size': 3840, 'foundation_vocabulary': 262144}, 'full_production_roster': True,
            'source_only': False, 'licensed_material_included_unchanged': True,
            'private_evidence_opened': False, 'model_weights_loaded': False, 'optimizer_created': False,
            'authority_written': False, 'final_payloads_opened': False, 'qualified_for_product': False,
            'target_visible_to_backbone': False, 'learned_capability_established': False,
            'new_source_acquisition': False}

    def test_single_pinned_manifest_required_among_mounted_inputs(self):
        m = self.module()
        folder = self.root / 'input'
        folder.mkdir()
        first = folder / 'dataset-a'
        first.mkdir()
        record = m.seal({'schema': m.SCHEMA, 'purpose': 'construction-measurement-only'})
        path = first / 'mfm-construction-transport.json'
        path.write_text(json.dumps(record))
        self.assertEqual(m.locate_manifest(folder, record['record_sha256']), path)
        (first / 'mfm-construction-transport.json').write_text(json.dumps({**record, 'purpose': 'changed'}))
        with self.assertRaises(ValueError):
            m.locate_manifest(folder, record['record_sha256'])
        path.write_text(json.dumps(record))
        second = folder / 'dataset-b'
        second.mkdir()
        (second / path.name).write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            m.locate_manifest(folder, record['record_sha256'])

    def test_recipe_requires_explicit_finite_fields_and_forbids_fit_or_resume(self):
        m = self.module()
        self.assertEqual(m.validate_recipe(self.recipe()), self.recipe())
        for field, bad in [('epochs', 0), ('batch_size', 36597), ('learning_rate', float('nan')),
                           ('beta2', 1.0), ('generation_tokens', 8193), ('seed', True)]:
            with self.subTest(field=field):
                recipe = self.recipe()
                recipe[field] = bad
                with self.assertRaises(ValueError):
                    m.validate_recipe(recipe)
        for extra in ('resume_checkpoint', 'measure_only', 'device'):
            with self.assertRaises(ValueError):
                m.validate_recipe({**self.recipe(), extra: 'unexpected'})

    def test_invocation_preserves_full_pins_and_forces_measurement_cuda(self):
        m = self.module()
        manifest = {'recipe': self.recipe(), 'pins': {'census_sha256': '1' * 64,
                    'feature_bank_sha256': '2' * 64, 'prepared_clone_sha256': '3' * 64,
                    'config_sha256': '4' * 64}, 'kernel_id': 'mkrayanyan/full-construction',
                    'layout': {'census': 'census', 'bank': 'bank'}}
        argv = m.trainer_argv(manifest, material=Path('/input/material'), metadata=Path('/input/meta'),
                              output=Path('/output/full'), allocation_ref='kernel-run-actual')
        self.assertEqual(argv[:3], ['-m', 'scripts.mfm.train_public_conditional_foundation', '--census-dir'])
        self.assertEqual(Path(argv[argv.index('--feature-bank') + 1]).as_posix(), '/input/material/bank')
        self.assertEqual(argv[argv.index('--feature-bank-sha256') + 1], '2' * 64)
        self.assertEqual(argv[argv.index('--device') + 1], 'cuda:0')
        self.assertEqual(argv[argv.index('--provider') + 1], 'Kaggle-free-public')
        self.assertEqual(argv[argv.index('--allocation-ref') + 1], 'kernel-run-actual')
        self.assertIn('--measure-only', argv)
        self.assertNotIn('--resume-checkpoint', argv)
        self.assertEqual(Path(argv[argv.index('--prepared-base-receipt') + 1]).as_posix(), '/input/meta/mfm-clone-576486.json')

    def test_runtime_version_mismatch_blocks_without_substitution(self):
        m = self.module()
        m.verify_dependencies({'transformers': '5.17.0'}, {'transformers': '5.17.0', 'torch': 'actual'})
        with self.assertRaises(ValueError):
            m.verify_dependencies({'transformers': '5.17.0'}, {'transformers': '5.16.0'})

    def test_owned_overlay_path_is_inherited_by_real_trainer_subprocess(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'trainer_environment'), 'owned overlay subprocess environment is missing')
        overlay, code = self.root / 'overlay', self.root / 'code'
        overlay.mkdir()
        (code / 'src').mkdir(parents=True)
        (overlay / 'transport_priority_fixture.py').write_text('VALUE = "owned-overlay"\n')
        (code / 'src/transport_priority_fixture.py').write_text('VALUE = "wrong-source"\n')
        env = m.trainer_environment(code, overlay)
        result = subprocess.run([__import__('sys').executable, '-c',
                                 'import transport_priority_fixture as f; print(f.VALUE)'],
                                env=env, capture_output=True, check=True, text=True)
        self.assertEqual(result.stdout.strip(), 'owned-overlay')
        self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')

    def test_dependency_manifest_allows_only_original_transformers_wheel(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'verify_dependency_manifest'), 'offline dependency manifest validator is missing')
        body = self.dependency_body()
        record = m.seal(body)
        self.assertEqual(m.verify_dependency_manifest(record), record['record_sha256'])
        for row in ({**body['wheels'][0], 'name': 'torch'}, {**body['wheels'][0], 'sha256': '0' * 64},
                    {**body['wheels'][0], 'filename': '../transformers.whl'}):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    m.verify_dependency_manifest(m.seal({**body, 'wheels': [row]}))

    def dependency_body(self):
        return {'schema': 'mfm-kaggle-offline-dependency-overlay-v1', 'python_specifier': '>=3.10.0',
            'base_requirements': ['huggingface-hub<2.0,>=1.5.0', 'numpy>=1.17', 'packaging>=20.0',
                'PyYAML>=5.1', 'regex>=2025.10.22', 'tokenizers<0.24,>=0.23.1', 'typer', 'safetensors>=0.8.0', 'tqdm>=4.60'],
            'forbidden_replacement': ['torch', 'torchvision', 'torchaudio'],
            'wheels': [{'name': 'transformers', 'version': '5.17.0', 'filename': 'transformers-5.17.0-py3-none-any.whl',
                'bytes': 12295140, 'sha256': '78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801',
                'source_url': 'https://files.pythonhosted.org/packages/e8/d0/c502b60d684adbd98a8dc7d5bb866842772b816ac4354e4608be240041ae/transformers-5.17.0-py3-none-any.whl'}]}

    def test_dependency_missing_base_requirement_blocks_and_pip_argv_has_no_index_or_dependencies(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'validate_base_requirements'), 'offline wheel base requirement validator is missing')
        with self.assertRaises(ValueError):
            m.validate_base_requirements(['tokenizers>=0.23.1,<0.24'], {'tokenizers': '0.22.0'}, '3.13.1', '>=3.10.0')
        with self.assertRaises(ValueError):
            m.validate_base_requirements(['tokenizers>=0.23.1,<0.24'], {'tokenizers': '0.23.1'}, '3.9.1', '>=3.10.0')
        m.validate_base_requirements(['tokenizers>=0.23.1,<0.24'], {'tokenizers': '0.23.1'}, '3.13.1', '>=3.10.0')
        argv = m.overlay_install_argv(Path('/owned/overlay'), [Path('/input/transformers.whl')])
        self.assertEqual(argv[1:6], ['-m', 'pip', 'install', '--no-index', '--no-deps'])
        self.assertIn('--target', argv)
        self.assertNotIn('--upgrade', argv)

    def test_package_builder_module_exists(self):
        name = 'scripts.mfm.kaggle_public_foundation_20261008'
        self.assertIsNotNone(importlib.util.find_spec(name), 'public construction packager is missing')

    def package_inputs(self):
        repo = self.root / 'repo'
        repo.mkdir()
        for name, value in [('scripts/mfm/__init__.py', b''),
                            ('scripts/mfm/train_public_conditional_foundation.py', b'"""Fixture code; never executed."""\n'),
                            ('src/cognitive_kernel/__init__.py', b'"""Fixture source."""\n'),
                            ('LICENSE', b'Fixture legal notice, only for this transport test.\n'),
                            ('README.md', b'Fixture repository readme.\n')]:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        def git(*argv):
            return subprocess.run(['git', '-C', str(repo), *argv], check=True, capture_output=True).stdout.decode().strip()
        git('init', '-q')
        git('add', '.')
        git('-c', 'user.name=Transport Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'source fixture')
        revision = git('rev-parse', 'HEAD')
        (repo / 'scripts/mfm/train_public_conditional_foundation.py').write_bytes(b'changed working bytes\n')
        (repo / 'src/cognitive_kernel/untracked.py').write_bytes(b'untracked\n')
        material = self.root / 'material.zip'
        with zipfile.ZipFile(material, 'w') as archive:
            archive.writestr('census/census.json', b'{}')
            archive.writestr('bank/bank.json', b'{}')
            archive.writestr('full-public-input-manifest.json', json.dumps(self.module().seal(self.material_binding_body())))
        metadata = self.root / 'metadata.zip'
        with zipfile.ZipFile(metadata, 'w') as archive:
            for name in ('.gitattributes', 'README.md', 'config.json', 'generation_config.json',
                         'processor_config.json', 'tokenizer.json', 'tokenizer_config.json',
                         'mfm-clone-576486.json', 'subset-manifest.json'):
                archive.writestr(name, b'{}')
        dependencies = self.root / 'dependency.zip'
        with zipfile.ZipFile(dependencies, 'w') as archive:
            archive.writestr('dependency-manifest.json', json.dumps(self.module().seal(self.dependency_body())))
            # Envelope-only fixture; wheel_preflight would reject these bytes.
            # No test installs this file or invokes the full trainer.
            archive.writestr('transformers-5.17.0-py3-none-any.whl', b'not an installed wheel')
        return SimpleNamespace(source_root=repo, source_revision=revision, output_dir=self.root / 'package',
            material_archive=material, material_archive_sha256=hashlib.sha256(material.read_bytes()).hexdigest(),
            material_format='zip', metadata_archive=metadata,
            metadata_archive_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
            dependency_archive=dependencies, dependency_archive_sha256=hashlib.sha256(dependencies.read_bytes()).hexdigest(),
            census_sha256='1' * 64, feature_bank_sha256='2' * 64, prepared_clone_sha256='3' * 64,
            config_sha256='4' * 64, durable_receipt_sha256='5' * 64,
            dataset_id='mkrayanyan/full-public-transport-fixture', kernel_id='mkrayanyan/full-construction-fixture',
            census_subdir='census', bank_subdir='bank', material_manifest_name='full-public-input-manifest.json',
            timeout_seconds=3600, **self.recipe())

    def test_real_git_packaging_preserves_commit_bytes_and_generates_private_launch(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        receipt = m.prepare(args)
        package = args.output_dir
        self.assertFalse(receipt['uploaded'])
        self.assertFalse(receipt['submitted'])
        manifest = self.module().read_manifest(package / 'dataset/mfm-construction-transport.json', receipt['transport_sha256'])
        with zipfile.ZipFile(package / 'dataset/reviewed-source.mfmblob') as archive:
            self.assertEqual(archive.read('scripts/mfm/train_public_conditional_foundation.py'), b'"""Fixture code; never executed."""\n')
            self.assertNotIn('src/cognitive_kernel/untracked.py', archive.namelist())
            self.assertIn('LICENSE', archive.namelist(), 'source legal notice was omitted')
            self.assertEqual(archive.read('LICENSE'), b'Fixture legal notice, only for this transport test.\n')
            self.assertEqual(archive.read('README.md'), b'Fixture repository readme.\n')
        self.assertEqual(manifest['source_revision'], args.source_revision)
        self.assertEqual(manifest['recipe']['batch_size'], 1)
        kernel = json.loads((package / 'kernel/kernel-metadata.json').read_bytes())
        self.assertTrue(kernel['is_private'])
        self.assertFalse(kernel['enable_internet'])
        self.assertEqual(kernel['dataset_sources'], [args.dataset_id])
        self.assertEqual(kernel['model_sources'], [])
        self.assertNotIn('--public', receipt['commands']['create_private_dataset'])
        self.assertEqual(receipt['commands']['launch_full_construction'][-2:], ['--accelerator', 'NvidiaTeslaT4'])
        self.assertEqual(receipt['commands']['launch_full_construction'][-4:-2], ['-t', '3600'])
        compile((package / 'kernel/main.py').read_bytes(), 'main.py', 'exec')
        with self.assertRaises(ValueError):
            m.prepare(args)

    def test_source_bundle_preserves_sibling_packages_parent_initializers_resources_and_notices(self):
        # The prior cognitive_kernel-only selection loses a required sibling.
        # Exact Git bytes, regular parents and resource lookup are independent
        # fixture expectations; working/untracked data must not enter the ZIP.
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        entries = {
            'scripts/__init__.py': b'PARENT_INITIALIZED = True\n',
            'src/alice_memory/__init__.py': b'PACKAGE_INITIALIZED = True\n',
            'src/alice_memory/schema.py': b'VALUE = "committed sibling"\n',
            'src/alice_memory/resources/schema.json': b'{"resource":"committed"}\n',
            'src/alice_memory/NOTICE': b'Fixture sibling package notice.\n',
            'src/alice_capability_profiles.py': b'VALUE = "committed root module"\n',
        }
        for name, contents in entries.items():
            path = args.source_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        m.git_bytes(args.source_root, 'add', '--', *entries)
        m.git_bytes(args.source_root, '-c', 'user.name=Transport Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'complete public package fixture')
        revision = m.git_bytes(args.source_root, 'rev-parse', 'HEAD').decode().strip()
        (args.source_root / 'src/alice_memory/schema.py').write_bytes(b'wrong working sibling\n')
        (args.source_root / 'src/alice_memory/private-untracked.json').write_bytes(b'not eligible')
        archive_path = self.root / 'complete-source.zip'
        m.source_archive(args.source_root, revision, archive_path)
        code = self.root / 'fixture-code'
        with zipfile.ZipFile(archive_path) as archive:
            for name, contents in entries.items():
                self.assertIn(name, archive.namelist(), name + ' missing from source capsule')
                self.assertEqual(archive.read(name), contents)
            self.assertNotIn('src/alice_memory/private-untracked.json', archive.namelist())
            archive.extractall(code)
        result = subprocess.run([sys.executable, '-I', '-B', '-c',
            'import sys,pathlib; sys.path[:0]=[sys.argv[1]+"/src",sys.argv[1]]; '
            'import scripts,alice_memory.schema; from importlib.resources import files; '
            'assert scripts.PARENT_INITIALIZED and alice_memory.PACKAGE_INITIALIZED; '
            'assert alice_memory.schema.VALUE=="committed sibling"; '
            'assert files("alice_memory").joinpath("resources/schema.json").read_bytes()==b\'{"resource":"committed"}\\n\'',
            str(code)], cwd=code, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_rejects_invalid_execution_timeout_before_writing(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        args.timeout_seconds = 0
        with self.assertRaises(ValueError):
            m.prepare(args)
        self.assertFalse(args.output_dir.exists())

    def test_package_defaults_to_provider_execution_limit(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        args.timeout_seconds = None
        receipt = m.prepare(args)
        self.assertEqual(receipt['commands']['launch_full_construction'],
                         ['kaggle', 'kernels', 'push', '-p', str(args.output_dir / 'kernel'), '--accelerator', 'NvidiaTeslaT4'])
        self.assertFalse(receipt['uploaded'])
        self.assertFalse(receipt['submitted'])

    def test_cli_custom_execution_timeout_is_optional(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        argv = [item for key, value in vars(args).items() if key != 'timeout_seconds'
                for item in ('--' + key.replace('_', '-'), str(value))]
        self.assertIsNone(m.arguments(argv).timeout_seconds)
        self.assertEqual(m.arguments(argv + ['--timeout-seconds', '3600']).timeout_seconds, 3600)

    def test_package_rejects_nonpositive_or_noninteger_opt_in_before_writing(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        for value in (0, -1, True, 1.5, '3600'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    m.prepare(SimpleNamespace(timeout_seconds=value))
        self.assertFalse((self.root / 'package').exists())

    def test_package_rejects_unlisted_material(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        with zipfile.ZipFile(args.material_archive, 'a') as archive:
            archive.writestr('private/unrelated.json', b'synthetic forbidden fixture')
        args.material_archive_sha256 = hashlib.sha256(args.material_archive.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            m.prepare(args)


class KaggleRealSourceBootstrapTests(unittest.TestCase):
    """CPU bootstrap only: no processor/data/model or hosted operation executes."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.repo = Path(__file__).resolve().parents[2]
        cls.packager = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        cls.revision = cls.packager.git_bytes(cls.repo, 'rev-parse', 'HEAD').decode().strip()
        cls.archive = cls.root / 'actual-git-source.zip'
        cls.files = cls.packager.source_archive(cls.repo, cls.revision, cls.archive)
        cls.code = cls.root / 'actual-code'
        with zipfile.ZipFile(cls.archive) as archive:
            archive.extractall(cls.code)

    def test_real_trainer_help_from_exact_export_without_checkout_fallback(self):
        result = isolated_trainer_help(self.code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--measure-only', result.stdout)
        proof = json.loads(result.stdout.split('CAPSULE_IMPORT_PROOF=')[1])
        self.assertEqual(proof['origins']['alice_memory.store'], 'src/alice_memory/store.py')
        self.assertEqual(proof['origins']['alice_memory.migrations'], 'src/alice_memory/migrations.py')
        expected = self.packager.git_bytes(self.repo, 'show', self.revision + ':src/cognitive_kernel/canonical.py')
        self.assertEqual(proof['canonical_resource_sha256'], hashlib.sha256(expected).hexdigest())
        self.assertFalse(proof['unused_vault_imported'])
        self.assertFalse(proof['unused_conversation_imported'])
        tree = self.packager.git_bytes(self.repo, 'ls-tree', '-r', '--name-only', self.revision).decode().splitlines()
        with zipfile.ZipFile(self.archive) as archive:
            for name in tree:
                if name.startswith('src/') or name in ('scripts/__init__.py', 'scripts/mfm/__init__.py'):
                    self.assertIn(name, archive.namelist(), 'tracked source/resource/parent was omitted: ' + name)
                    self.assertEqual(archive.read(name), self.packager.git_bytes(self.repo, 'show', self.revision + ':' + name))

    def test_omitted_transitive_source_cannot_be_supplied_by_checkout(self):
        positive = isolated_trainer_help(self.code)
        self.assertEqual(positive.returncode, 0, positive.stdout + positive.stderr)
        for name, missing in [('schema.py', 'alice_memory.schema'), ('store.py', 'alice_memory.store'),
                              ('migrations.py', 'alice_memory.migrations')]:
            with self.subTest(missing=missing):
                code = self.root / ('missing-' + name)
                shutil.copytree(self.code, code)
                (code / 'src/alice_memory' / name).unlink()
                # Deliberately offer the real checkout as an alternative path;
                # isolation must still expose the capsule's missing dependency.
                result = isolated_trainer_help(code, fallback=self.repo / 'src')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ModuleNotFoundError: No module named ' + repr(missing), result.stderr)
                self.assertNotIn('CAPSULE_IMPORT_PROOF=', result.stdout)


if __name__ == '__main__':
    unittest.main()
