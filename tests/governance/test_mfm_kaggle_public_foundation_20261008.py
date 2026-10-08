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
from types import SimpleNamespace


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
        compile((package / 'kernel/main.py').read_bytes(), 'main.py', 'exec')
        with self.assertRaises(ValueError):
            m.prepare(args)

    def test_package_rejects_invalid_execution_timeout_before_writing(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        args.timeout_seconds = 0
        with self.assertRaises(ValueError):
            m.prepare(args)
        self.assertFalse(args.output_dir.exists())

    def test_package_rejects_unlisted_material(self):
        m = importlib.import_module('scripts.mfm.kaggle_public_foundation_20261008')
        args = self.package_inputs()
        with zipfile.ZipFile(args.material_archive, 'a') as archive:
            archive.writestr('private/unrelated.json', b'synthetic forbidden fixture')
        args.material_archive_sha256 = hashlib.sha256(args.material_archive.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            m.prepare(args)


if __name__ == '__main__':
    unittest.main()
