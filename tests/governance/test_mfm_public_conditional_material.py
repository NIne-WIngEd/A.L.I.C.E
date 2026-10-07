"""Public fixtures exercise original-byte, grant, grouping and publication custody."""
from copy import deepcopy
import csv
from hashlib import sha256
from importlib import import_module
import io
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / 'configs/mfm/e2e_conditional_material_v1.json'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow(('mr', 'ref'))
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def fixture(rows=None):
    rows = rows or [('name[Café North], food[French]', 'Café North serves French food.  \r\n'),
                    ('food[French], name[Café North]', 'A French café called Café North.'),
                    ('name[Café North], food[French]', 'Café North serves French food.  \r\n'),
                    ('name[South], near[Station, West]', 'South is near Station, West.')]
    pin = json.loads(MANIFEST.read_bytes())
    train = csv_bytes(rows)
    readme = b'Original authors\nLicense\nCC4.0-BY-SA\nhttps://creativecommons.org/licenses/by-sa/4.0/\n'
    legal = b'Attribution-ShareAlike 4.0 International\npublic fixture notice\n'
    for name, raw in (('train', train), ('readme', readme), ('legal_notice', legal)):
        pin[name]['sha256'], pin[name]['bytes'] = sha256(raw).hexdigest(), len(raw)
    pin['train']['rows'] = len(rows)
    return pin, train, readme, legal


class ConditionalMaterialTests(unittest.TestCase):
    def worker(self):
        try:
            return import_module('scripts.mfm.prepare_public_conditional_material')
        except ModuleNotFoundError as exc:
            self.fail(f'Public conditional preparation API is missing: {exc.name}')

    def test_all_alternative_references_group_without_target_or_order_in_split(self):
        worker = self.worker()
        pin, train, readme, legal = fixture()
        records, summary = worker.prepare_records(pin, train, readme, legal)
        self.assertEqual((summary['source_csv_rows'], summary['rows'], summary['duplicate_rows']),
                         (4, 3, 1))
        cafe = [row for row in records if 'Café North' in row['source_text']]
        self.assertEqual(len({row['semantic_input_sha256'] for row in cafe}), 1)
        self.assertEqual(len({row['split'] for row in cafe}), 1)
        self.assertEqual([row['semantic_reference_count'] for row in cafe], [2, 2])
        reversed_rows = list(reversed(list(csv.reader(io.StringIO(train.decode('utf-8'), newline='')))[1:]))
        changed = deepcopy(pin)
        reordered = csv_bytes(reversed_rows)
        changed['train'].update(sha256=sha256(reordered).hexdigest(), bytes=len(reordered))
        again, _ = worker.prepare_records(changed, reordered, readme, legal)
        view = lambda rows: {row['case_id']: (row['semantic_input_sha256'], row['split']) for row in rows}
        self.assertEqual(view(records), view(again))
        altered = csv_bytes([(row['source_text'], 'A different human reference.') for row in records])
        changed['train'].update(sha256=sha256(altered).hexdigest(), bytes=len(altered), rows=3)
        new, _ = worker.prepare_records(changed, altered, readme, legal)
        self.assertEqual({row['source_text']: row['split'] for row in records},
                         {row['source_text']: row['split'] for row in new})

    def test_utf8_line_endings_and_literal_text_are_preserved_and_bound(self):
        worker = self.worker()
        original = '  Café "North" <eos>  \r\nβ\n'
        pin, train, readme, legal = fixture([('name[Café North], food[French]', original)])
        records, _ = worker.prepare_records(pin, train, readme, legal)
        row = records[0]
        self.assertEqual(row['target_text'].encode('utf-8'), original.encode('utf-8'))
        self.assertEqual(row['target_payload_sha256'], sha256(original.encode('utf-8')).hexdigest())
        self.assertEqual(row['source_payload_sha256'],
                         sha256(b'name[Caf\xc3\xa9 North], food[French]').hexdigest())
        self.assertEqual(row['source_rows'], [1])
        self.assertNotIn('formation_decision', row)
        self.assertNotIn('world_truth', row)

    def test_known_input_hash_buckets_reserve_development_reference_family(self):
        worker = self.worker()
        pin, train, readme, legal = fixture([
            ('name[North]', 'North is a restaurant.'),
            ('name[Fixture20]', 'The place is called Fixture20.'),
            ('name[Fixture20]', 'Fixture20 is its name.'),
        ])
        rows, summary = worker.prepare_records(pin, train, readme, legal)
        actual = {row['source_text']: (row['semantic_input_sha256'], row['split'],
                                      row['semantic_reference_count']) for row in rows}
        self.assertEqual(actual, {
            'name[North]': ('f149988e6a236e238a5b26953ac80d168f758b0f6e2544134ee00dcf9804d21f', 'train', 1),
            'name[Fixture20]': ('4f42048d9a6e9c1e13c772234547b5bd22801c4e0a9d8efe3505c6c5a53b3606', 'development', 2),
        })
        self.assertEqual(summary['split_counts'], {'train': 1, 'development': 2})
        self.assertEqual(summary['split_semantic_input_counts'], {'train': 1, 'development': 1})

    def test_parser_handles_slot_order_commas_and_rejects_unsupported_or_ambiguous_input(self):
        worker = self.worker()
        self.assertEqual(worker.semantic_mr(' food[French] , name[Café North], near[Station, West] '),
                         (('food', 'French'), ('name', 'Café North'), ('near', 'Station, West')))
        for bad in ('name[North], name[South]', 'secret[North]', 'name[]',
                    'name[North] junk', 'name[North', 'name[[North]]', ',name[North]',
                    'name[North],', 'name[North], unknown[South]'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                worker.semantic_mr(bad)

    def test_wrong_payload_pin_or_notice_rejects_before_records(self):
        worker = self.worker()
        pin, train, readme, legal = fixture()
        for raws in ((train+b' ', readme, legal), (train, readme+b' ', legal),
                     (train, readme, legal+b' ')):
            with self.subTest(kind=raws), self.assertRaises(ValueError):
                worker.prepare_records(pin, *raws)
        missing_grant = b'No grant here\n'
        bad = deepcopy(pin)
        bad['readme'].update(sha256=sha256(missing_grant).hexdigest(), bytes=len(missing_grant))
        with self.assertRaises(ValueError):
            worker.prepare_records(bad, train, missing_grant, legal)

    def test_nontrain_or_foreign_revision_manifest_is_rejected(self):
        worker = self.worker()
        pin, train, readme, legal = fixture()
        for field, value in (('filename', 'devset.csv'), ('url', pin['reserved_official_files'][0]['url'])):
            bad = deepcopy(pin)
            bad['train'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.prepare_records(bad, train, readme, legal)
        bad = deepcopy(pin)
        bad['upstream_revision'] = '0'*40
        with self.assertRaises(ValueError):
            worker.prepare_records(bad, train, readme, legal)

    def test_declared_row_count_and_exact_csv_fields_are_enforced(self):
        worker = self.worker()
        pin, train, readme, legal = fixture()
        wrong = deepcopy(pin)
        wrong['train']['rows'] += 1
        with self.assertRaises(ValueError):
            worker.prepare_records(wrong, train, readme, legal)
        for raw in (b'MR,ref\r\nname[North],North\r\n',
                    b'mr,ref,teacher\r\nname[North],North,gold\r\n',
                    b'mr,ref\r\nname[North],North,extra\r\n',
                    b'mr,ref\r\nname[North]\r\n'):
            wrong = deepcopy(pin)
            wrong['train'].update(sha256=sha256(raw).hexdigest(), bytes=len(raw), rows=1)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                worker.prepare_records(wrong, raw, readme, legal)

    def test_real_publication_has_original_notices_self_digest_and_external_pin(self):
        worker = self.worker()
        pin, train, readme, legal = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'manifest.json'
            config.write_bytes(encoded(pin))
            cache = root/'cache'
            cache.mkdir()
            for key, raw in (('train', train), ('readme', readme), ('legal_notice', legal)):
                (cache/pin[key]['filename']).write_bytes(raw)
            output = root/'prepared'
            result = worker.stage_material(config, sha256(config.read_bytes()).hexdigest(), cache, output)
            receipt_raw = (output/'receipt.json').read_bytes()
            receipt = json.loads(receipt_raw)
            self.assertTrue(receipt['conditional_training_eligible'])
            self.assertFalse(receipt['formation_training_eligible'])
            self.assertFalse(receipt['qualified'])
            self.assertFalse(receipt['authority'])
            self.assertEqual(receipt['material_role'], 'conditional_meaning_to_text_only')
            self.assertEqual((output/'README.md').read_bytes(), readme)
            self.assertEqual((output/'CC-BY-SA-4.0-legalcode.txt').read_bytes(), legal)
            self.assertEqual(receipt['pairs_sha256'], sha256((output/'pairs.jsonl').read_bytes()).hexdigest())
            self.assertEqual((output/'receipt.sha256').read_text().strip(), sha256(receipt_raw).hexdigest())
            body = dict(receipt)
            self_digest = body.pop('receipt_sha256')
            self.assertEqual(self_digest, sha256(encoded(body)).hexdigest())
            self.assertEqual(result['receipt_file_sha256'], sha256(receipt_raw).hexdigest())
            self.assertEqual(sum(receipt['split_counts'].values()), 3)
            self.assertEqual(sum(receipt['split_semantic_input_counts'].values()), 2)
            self.assertEqual(receipt['authors'], ['Jekaterina Novikova', 'Ondrej Dusek', 'Verena Rieser'])
            with self.assertRaises(FileExistsError):
                worker.stage_material(config, sha256(config.read_bytes()).hexdigest(), cache, output)

    def test_bad_external_config_pin_and_cached_tamper_publish_nothing(self):
        worker = self.worker()
        pin, train, readme, legal = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'manifest.json'
            config.write_bytes(encoded(pin))
            cache = root/'cache'
            cache.mkdir()
            for key, raw in (('train', train), ('readme', readme), ('legal_notice', legal)):
                (cache/pin[key]['filename']).write_bytes(raw)
            output = root/'prepared'
            with self.assertRaises(ValueError):
                worker.stage_material(config, '0'*64, cache, output)
            self.assertFalse(output.exists())
            (cache/'trainset.csv').write_bytes(train+b' ')
            with self.assertRaises(ValueError):
                worker.stage_material(config, sha256(config.read_bytes()).hexdigest(), cache, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
