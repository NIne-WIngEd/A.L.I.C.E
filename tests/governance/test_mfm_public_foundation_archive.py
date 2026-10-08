"""Real tiny bank custody/ZIP tests; these do not prove full-bank custody."""
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import os
import stat
import struct
import subprocess
import unittest
from unittest.mock import patch
import warnings
import zipfile
import zlib

import torch

from scripts.mfm import export_public_conditional_features as features
from scripts.mfm import public_conditional_training as reader
from scripts.mfm import train_v1_formation_specialist as shared
import test_mfm_public_conditional_training as fixtures


class PublicFoundationArchiveTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PublicConditionalTrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.fixture.root
        self.output = self.root / 'archive-output'
        self.legal = self.root / 'original-legal.zip'
        with zipfile.ZipFile(self.legal, 'x') as archive:
            archive.writestr('material/ATTRIBUTION.txt', b'original attribution\n')
            archive.writestr('material/CC-BY-SA-4.0-legalcode.txt', b'original legal notice\n')
            archive.writestr('census/receipt.json', b'{"original":true}\n')
        self.legal_pin = shared._digest(self.legal)

    def module(self):
        name = 'scripts.mfm.archive_public_foundation_inputs'
        self.assertIsNotNone(importlib.util.find_spec(name), 'public foundation archive worker is missing')
        return importlib.import_module(name)

    def archive(self, **kwargs):
        return self.module().archive_inputs(
            self.fixture.directory, self.fixture.pin,
            self.fixture.bank_dir, self.fixture.bank_pin,
            self.legal, self.legal_pin, self.output, require_full_roster=False, **kwargs)

    def arguments(self):
        return ['--census-dir', str(self.fixture.directory), '--census-sha256', self.fixture.pin,
                '--feature-bank', str(self.fixture.bank_dir), '--feature-bank-sha256', self.fixture.bank_pin,
                '--legal-material-archive', str(self.legal), '--legal-material-archive-sha256', self.legal_pin,
                '--output-dir', str(self.output)]

    def add_retained_quarantine(self):
        # Real exporter recovery preserves failed publication bytes and binding.
        prepared = self.fixture.load()
        token = '1' * 32
        retained = self.fixture.bank_dir / ('.record-' + token)
        retained.write_bytes(b'retained failed publication\x00\xff')
        empty = self.fixture.bank_dir / 'sources' / ('.staging-' + '2' * 32)
        empty.mkdir()
        # The incomplete scratch is intentionally introduced after the prior
        # control read; recovery is the producer's actual quarantine path.
        with features.worker_lock(self.fixture.bank_dir):
            quarantine = features.recover_scratch(self.fixture.bank_dir, list(prepared.sources), prepared.export)
        body = {k: v for k, v in prepared.bank.items() if k != 'record_sha256'}
        body['quarantine_sha256'] = quarantine['record_sha256']
        bank = features._write_record(self.fixture.bank_dir / 'bank.json', body, replace=True)
        self.fixture.bank_pin = bank['record_sha256']
        return 'bank/quarantine/record-' + token

    def test_roundtrip_preserves_every_original_byte_and_independent_reader(self):
        # Break caught: omission/recompression corruption or wrong custody paths.
        before = self.fixture.load()
        payload = reader.payload(before, before.train[1], device='cpu')
        result = self.archive()
        archive_path = self.output / result['archive_filename']
        self.assertEqual(archive_path.name, 'sha256-' + shared._digest(archive_path) + '.zip')
        self.assertEqual(result['archive_sha256'], shared._digest(archive_path))
        restored = self.root / 'restored'
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertNotIn('bank/worker.lock', names)
            legal_name = 'legal-material/sha256-' + self.legal_pin + '.zip'
            self.assertEqual(archive.read(legal_name), self.legal.read_bytes())
            self.assertEqual(archive.getinfo(legal_name).compress_type, zipfile.ZIP_STORED)
            original_export = (self.fixture.bank_dir / 'export.json').read_bytes()
            self.assertEqual(archive.getinfo('bank/export.json').compress_size,
                             len(zlib.compress(original_export, level=1, wbits=-15)))
            for row in result['files']:
                content = archive.read(row['path'])
                self.assertEqual(len(content), row['size'])
                self.assertEqual(sha256(content).hexdigest(), row['sha256'])
            self.assertEqual(set(names), {row['path'] for row in result['files']})
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['files'], [r for r in result['files'] if r['path'] != 'manifest.json'])
            archive.extractall(restored)
        for prefix, directory in (('census', self.fixture.directory), ('bank', self.fixture.bank_dir)):
            for original in directory.rglob('*'):
                if original.is_file() and original.name != 'worker.lock':
                    self.assertEqual((restored / prefix / original.relative_to(directory)).read_bytes(), original.read_bytes())
        after = reader.load_public_training(restored / 'census', self.fixture.pin,
                                            restored / 'bank', self.fixture.bank_pin)
        actual = reader.payload(after, after.train[1], device='cpu')
        for key in payload:
            self.assertTrue(torch.equal(payload[key], actual[key]))
        self.assertEqual(result['counts']['pairs'], 4)
        self.assertEqual(result['counts']['pairs_by_split'], {'train': 3, 'development': 1})
        self.assertEqual(result['counts']['semantic_groups_by_split'], {'train': 2, 'development': 1})
        self.assertFalse(result['full_production_roster'])
        self.assertFalse(result['qualified_for_product'])
        self.assertFalse(result['model_weights_loaded'])
        self.assertFalse(result['optimizer_created'])
        self.assertFalse(result['target_visible_to_backbone'])
        self.assertFalse(result['durable_custody_verified'])
        self.assertTrue(result['archive_crc_sha256_readback_verified'])
        sealed = features._read_sealed(self.output / 'receipt.json', self.module().RECEIPT_SCHEMA)
        self.assertEqual(sealed, result)

    def test_quarantine_binding_bytes_and_empty_directories_survive(self):
        # Break caught: treating retained failures as disposable operations.
        retained_name = self.add_retained_quarantine()
        result = self.archive()
        restored = self.root / 'restored-quarantine'
        with zipfile.ZipFile(self.output / result['archive_filename']) as archive:
            self.assertEqual(archive.read(retained_name), b'retained failed publication\x00\xff')
            self.assertIn('bank/quarantine/source-' + '2' * 32 + '/', archive.namelist())
            archive.extractall(restored)
        prepared = reader.load_public_training(restored / 'census', self.fixture.pin,
                                               restored / 'bank', self.fixture.bank_pin)
        self.assertEqual(result['quarantine_sha256'], prepared.bank['quarantine_sha256'])

    def test_cli_refuses_reduced_roster_before_output_creation(self):
        # Break caught: accepting a small diagnostic as the production archive.
        with self.assertRaisesRegex(ValueError, 'complete production roster'):
            self.module().main(self.arguments())
        self.assertFalse(self.output.exists())

    def test_corrupt_bank_and_wrong_external_pins_refuse_before_write(self):
        # Break caught: packaging self-consistent or corrupted untrusted inputs.
        for attribute in ('pin', 'bank_pin'):
            with self.subTest(attribute=attribute):
                original = getattr(self.fixture, attribute)
                setattr(self.fixture, attribute, '0' * 64)
                with self.assertRaises(ValueError):
                    self.archive()
                setattr(self.fixture, attribute, original)
                self.assertFalse(self.output.exists())
        original_legal = self.legal_pin
        self.legal_pin = '0' * 64
        with self.assertRaisesRegex(ValueError, 'legal.*digest'):
            self.archive()
        self.legal_pin = original_legal
        self.assertFalse(self.output.exists())
        tensor = next(self.fixture.bank_dir.glob('sources/*/features.safetensors'))
        with tensor.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'tensor file digest'):
            self.archive()
        self.assertFalse(self.output.exists())

    def test_unknown_extra_is_never_packaged(self):
        # Break caught: silently accepting files outside the validated roster.
        for directory in (self.fixture.directory, self.fixture.bank_dir):
            extra = directory / 'private-extra.bin'
            extra.write_bytes(b'not public')
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                self.archive()
            extra.unlink()
            self.assertFalse(self.output.exists())

    def test_symlink_is_never_packaged(self):
        target = self.root / 'outside.bin'
        target.write_bytes(b'outside')
        link = self.fixture.directory / 'linked.bin'
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest('symlinks unavailable: ' + str(error))
        with self.assertRaisesRegex(ValueError, 'link|reparse|unsafe'):
            self.archive()
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows junction control')
    def test_junction_root_is_never_followed(self):
        # Native junctions do not require symlink privileges on Windows.
        junction = self.root / 'junction-census'
        completed = subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), str(self.fixture.directory)],
                                   capture_output=True, text=True, check=False)
        if completed.returncode:
            self.skipTest('junction creation unavailable: ' + completed.stderr)
        # Both resolved paths have been established inside this fixture root.
        self.assertTrue(junction.absolute().is_relative_to(self.root.absolute()))
        self.assertTrue(junction.resolve().is_relative_to(self.root.resolve()))
        self.addCleanup(junction.rmdir)
        with self.assertRaisesRegex(ValueError, 'link|reparse|unsafe'):
            self.module().archive_inputs(junction, self.fixture.pin,
                self.fixture.bank_dir, self.fixture.bank_pin, self.legal, self.legal_pin,
                self.output, require_full_roster=False)
        self.assertFalse(self.output.exists())

    def test_unsafe_duplicate_or_linked_legal_zip_refuses_before_write(self):
        # Break caught: preserving an unsafe nested archive without checking it.
        for mode in ('traversal', 'duplicate', 'link'):
            bad = self.root / (mode + '.zip')
            with warnings.catch_warnings(), zipfile.ZipFile(bad, 'x') as archive:
                warnings.simplefilter('ignore', UserWarning)
                if mode == 'traversal':
                    archive.writestr('../outside', b'bad')
                elif mode == 'duplicate':
                    archive.writestr('notice', b'first')
                    archive.writestr('notice', b'second')
                else:
                    entry = zipfile.ZipInfo('notice')
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                    archive.writestr(entry, b'../outside')
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'unsafe|duplicate|link'):
                self.module().archive_inputs(self.fixture.directory, self.fixture.pin,
                    self.fixture.bank_dir, self.fixture.bank_pin, bad, shared._digest(bad), self.output,
                    require_full_roster=False)
            self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        # Break caught: retrying against existing custody destroys prior bytes.
        self.output.mkdir()
        sentinel = self.output / 'failed-scratch.bin'
        sentinel.write_bytes(b'preserve original failure')
        with self.assertRaises((ValueError, FileExistsError)):
            self.archive()
        self.assertEqual(sentinel.read_bytes(), b'preserve original failure')
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_changed_input_after_packaging_retains_failed_scratch_without_publication(self):
        # Real file mutation between write and publication; validator stays real.
        module = self.module()
        original_write = module._write_zip
        source = self.fixture.directory / 'encoded-pairs.jsonl'
        def changed_after_write(*args, **kwargs):
            result = original_write(*args, **kwargs)
            with source.open('ab') as stream:
                stream.write(b'changed')
            return result
        with patch.object(module, '_write_zip', changed_after_write):
            with self.assertRaisesRegex(ValueError, 'changed|digest'):
                self.archive()
        self.assertTrue(self.output.exists())
        self.assertTrue(list(self.output.glob('.archive-*.zip')))
        self.assertFalse(list(self.output.glob('sha256-*.zip')))
        self.assertFalse((self.output / 'receipt.json').exists())

    def test_archive_readback_rejects_altered_bytes_and_duplicate_names(self):
        # Break caught: CRC alone or name lookup misses modified/duplicate data.
        result = self.archive()
        module = self.module()
        # Flip one actual stored payload byte without updating its ZIP CRC.
        crc_bad = self.root / 'corrupt-crc.zip'
        archive_path = self.output / result['archive_filename']
        crc_bad.write_bytes(archive_path.read_bytes())
        with zipfile.ZipFile(archive_path) as archive:
            offset = archive.getinfo('legal-material/sha256-' + self.legal_pin + '.zip').header_offset
        with crc_bad.open('r+b') as stream:
            stream.seek(offset + 26)
            name_length, extra_length = struct.unpack('<HH', stream.read(4))
            stream.seek(offset + 30 + name_length + extra_length)
            original = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaises(zipfile.BadZipFile):
            module.verify_archive(crc_bad, result['files'])
        bad = self.root / 'altered.zip'
        with zipfile.ZipFile(bad, 'x') as archive:
            for row in result['files']:
                archive.writestr(row['path'], b'changed' if row['path'] == 'manifest.json' else b'')
        with self.assertRaises(ValueError):
            module.verify_archive(bad, result['files'])
        duplicate = self.root / 'duplicate-readback.zip'
        with warnings.catch_warnings(), zipfile.ZipFile(duplicate, 'x') as archive:
            warnings.simplefilter('ignore', UserWarning)
            archive.writestr('same', b'one')
            archive.writestr('same', b'two')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.verify_archive(duplicate, [])


if __name__ == '__main__':
    unittest.main()
