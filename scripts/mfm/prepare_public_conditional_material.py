"""Prepare pinned E2E human MR-to-text material; no formation or model authority.

The CLI needs an external manifest pin. Only the original training CSV is read;
official development/test files remain reserved. Existing outputs are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import urllib.request


SLOTS = ('name', 'eatType', 'food', 'priceRange', 'customer rating', 'area',
         'familyFriendly', 'near')
SOURCE_SCHEMA = 'mfm-public-conditional-source-pin-v1'
PAIR_SCHEMA = 'mfm-public-conditional-pair-v1'
RECEIPT_SCHEMA = 'mfm-public-conditional-material-receipt-v1'
ROLE = 'conditional_meaning_to_text_only'
SPLIT_RULE = 'semantic_mr_sha256_mod10_0_development_v1'
REPOSITORY = 'https://github.com/tuetschek/e2e-dataset'
RAW_PREFIX = 'https://raw.githubusercontent.com/tuetschek/e2e-dataset/'
LICENSE_URL = 'https://creativecommons.org/licenses/by-sa/4.0/'
AUTHORS = ['Jekaterina Novikova', 'Ondrej Dusek', 'Verena Rieser']


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected a lowercase SHA-256 pin')
    return value


def _checked(raw, reference):
    if not isinstance(raw, bytes):
        raise ValueError('payload custody requires original bytes')
    if (type(reference.get('bytes')) is not int or reference['bytes'] <= 0 or
            len(raw) != reference['bytes'] or sha256(raw).hexdigest() != _digest(reference['sha256'])):
        raise ValueError('public payload differs from pinned bytes/hash')
    return raw


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != SOURCE_SCHEMA:
        raise ValueError('unsupported public conditional manifest')
    if (manifest.get('material_role') != ROLE or manifest.get('upstream_repository') != REPOSITORY or
            manifest.get('license') != 'CC-BY-SA-4.0' or manifest.get('license_url') != LICENSE_URL or
            manifest.get('authors') != AUTHORS or manifest.get('semantic_slots') != list(SLOTS) or
            manifest.get('split_rule') != SPLIT_RULE):
        raise ValueError('manifest differs from the reviewed E2E source contract')
    for key, expected in (('conditional_training_eligible', True), ('formation_training_eligible', False),
                          ('qualified', False), ('authority', False)):
        if manifest.get(key) is not expected:
            raise ValueError('conditional material cannot assert formation/qualification/authority')
    for key in ('citation', 'rights_review', 'privacy_origin_limits'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError('manifest must retain attribution and review limits')
    revision = manifest.get('upstream_revision')
    if not isinstance(revision, str) or re.fullmatch('[0-9a-f]{40}', revision) is None:
        raise ValueError('upstream revision must be an exact Git commit')
    for key, filename, url in (
        ('train', 'trainset.csv', RAW_PREFIX+revision+'/trainset.csv'),
        ('readme', 'README.md', RAW_PREFIX+revision+'/README.md'),
        ('legal_notice', 'CC-BY-SA-4.0-legalcode.txt', LICENSE_URL+'legalcode.txt'),
    ):
        reference = manifest.get(key)
        if not isinstance(reference, dict) or reference.get('filename') != filename or reference.get('url') != url:
            raise ValueError('only exact pinned training and original notices can be prepared')
        _digest(reference.get('sha256'))
        if type(reference.get('bytes')) is not int or reference['bytes'] <= 0:
            raise ValueError('missing exact public payload size')
    if type(manifest['train'].get('rows')) is not int or manifest['train']['rows'] <= 0:
        raise ValueError('missing complete training row count')
    reserved = manifest.get('reserved_official_files')
    expected = {'devset.csv', 'testset.csv', 'testset_w_refs.csv'}
    if not isinstance(reserved, list) or len(reserved) != 3 or {row.get('filename') for row in reserved} != expected:
        raise ValueError('official development/test exclusions must remain complete')
    for row in reserved:
        if row.get('url') != RAW_PREFIX+revision+'/'+row['filename']:
            raise ValueError('reserved file lineage differs')
        _digest(row.get('sha256'))


def semantic_mr(text):
    """Parse the eight-slot grammar, preserving every value and raw caller text.

    Commas inside bracketed values are data. Whitespace around slot delimiters
    is syntax; slot order is immaterial. Unknown, repeated or malformed slots
    cannot silently create a different grouping or be dropped.
    """
    if not isinstance(text, str) or not text.strip() or '\x00' in text:
        raise ValueError('meaning representation must be nonempty text')
    cursor, values = 0, {}
    while cursor < len(text):
        match = re.match(r'\s*([^\[\],]+?)\s*\[', text[cursor:])
        if match is None:
            raise ValueError('malformed E2E slot syntax')
        slot = match.group(1).strip()
        cursor += match.end()
        end = text.find(']', cursor)
        if end < 0:
            raise ValueError('unclosed E2E slot')
        value = text[cursor:end]
        if slot not in SLOTS or slot in values or not value or '[' in value:
            raise ValueError('unknown, repeated or ambiguous E2E slot/value')
        values[slot] = value
        cursor = end+1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        if text[cursor] != ',' or not text[cursor+1:].strip():
            raise ValueError('trailing or unsupported E2E slot syntax')
        cursor += 1
    return tuple(sorted(values.items()))


def prepare_records(manifest, train_raw, readme_raw, legal_raw):
    """Validate full source bytes before returning any language-only records."""
    validate_manifest(manifest)
    for key, raw in (('train', train_raw), ('readme', readme_raw), ('legal_notice', legal_raw)):
        _checked(raw, manifest[key])
    readme = readme_raw.decode('utf-8', errors='strict')
    legal = legal_raw.decode('utf-8', errors='strict')
    if 'CC4.0-BY-SA' not in readme or LICENSE_URL not in readme:
        raise ValueError('the original README must retain its explicit CC BY-SA grant')
    if 'Attribution-ShareAlike 4.0 International' not in legal:
        raise ValueError('the pinned legal notice must identify the reviewed license')
    reader = csv.reader(io.StringIO(train_raw.decode('utf-8', errors='strict'), newline=''), strict=True)
    if next(reader, None) != ['mr', 'ref']:
        raise ValueError('only original mr/ref training fields are supported')
    pairs, original_rows = {}, 0
    for original_rows, fields in enumerate(reader, start=1):
        if len(fields) != 2 or not fields[1]:
            raise ValueError('each original row must contain exactly mr and nonempty ref')
        source, target = fields
        semantic = sha256(canonical_bytes(semantic_mr(source))).hexdigest()
        identity = sha256(canonical_bytes({'source_text': source, 'target_text': target})).hexdigest()
        if identity in pairs:
            if (pairs[identity]['source_text'], pairs[identity]['target_text']) != (source, target):
                raise ValueError('pair digest identity collision')
            pairs[identity]['source_rows'].append(original_rows)
            continue
        pairs[identity] = {
            'schema': PAIR_SCHEMA, 'case_id': 'e2e-'+identity,
            'source_text': source, 'target_text': target,
            'source_payload_sha256': sha256(source.encode('utf-8')).hexdigest(),
            'target_payload_sha256': sha256(target.encode('utf-8')).hexdigest(),
            'source_file_sha256': manifest['train']['sha256'],
            'semantic_input_sha256': semantic,
            'split': 'development' if int(semantic, 16) % 10 == 0 else 'train',
            'source_row': original_rows, 'source_rows': [original_rows],
            'upstream_revision': manifest['upstream_revision'],
        }
    if original_rows != manifest['train']['rows']:
        raise ValueError('complete training CSV row count differs from the manifest')
    counts = Counter(row['semantic_input_sha256'] for row in pairs.values())
    records = [pairs[identity] for identity in sorted(pairs)]
    for row in records:
        row['semantic_reference_count'] = counts[row['semantic_input_sha256']]
    groups = {row['semantic_input_sha256']: row['split'] for row in records}
    split_counts = {name: sum(row['split'] == name for row in records) for name in ('train', 'development')}
    group_counts = {name: sum(split == name for split in groups.values()) for name in ('train', 'development')}
    summary = {'source_csv_rows': original_rows, 'rows': len(records),
               'duplicate_rows': original_rows-len(records),
               'unique_semantic_inputs': len(groups), 'split_counts': split_counts,
               'split_semantic_input_counts': group_counts,
               'source_train_sha256': manifest['train']['sha256']}
    return records, summary


def _read_json(raw):
    def no_duplicates(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate manifest key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=no_duplicates)


def _new_file(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)


def stage_material(manifest_path, manifest_sha256, cache_dir, output_dir):
    """Stage real pinned bytes; publish a receipt only after complete validation.

    A cache mismatch is an error, not permission to replace retained evidence.
    Downloaded content is checked before its exclusive cache publication.
    """
    manifest_path, cache_dir, output_dir = map(Path, (manifest_path, cache_dir, output_dir))
    manifest_raw = manifest_path.read_bytes()
    if sha256(manifest_raw).hexdigest() != _digest(manifest_sha256):
        raise ValueError('external manifest pin differs')
    manifest = _read_json(manifest_raw)
    validate_manifest(manifest)
    if output_dir.exists():
        raise FileExistsError('preserve existing material output; choose a new directory')
    raw = {}
    for key in ('train', 'readme', 'legal_notice'):
        reference = manifest[key]
        path = cache_dir/reference['filename']
        if path.exists():
            raw[key] = _checked(path.read_bytes(), reference)
        else:
            request = urllib.request.Request(reference['url'], headers={'User-Agent': 'MFM-pinned-public-material/1'})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw[key] = _checked(response.read(reference['bytes']+1), reference)
            cache_dir.mkdir(parents=True, exist_ok=True)
            _new_file(path, raw[key])
    records, summary = prepare_records(manifest, raw['train'], raw['readme'], raw['legal_notice'])
    pairs_raw = b''.join(canonical_bytes(row)+b'\n' for row in records)
    attribution = ('E2E Dataset\nAuthors: '+', '.join(manifest['authors'])+'\n'+manifest['citation']+
                   '\nSource: '+manifest['upstream_repository']+'\nRevision: '+manifest['upstream_revision']+
                   '\nLicense: CC BY-SA 4.0 '+LICENSE_URL+
                   '\nChanges: CSV fields serialized without text normalization; exact pairs deduplicated;'
                   ' semantic MR grouping and training/development assignment added.\n'
                   'Official development/test excluded; language-only teaching, no formation authority.\n').encode('utf-8')
    files = {'pairs.jsonl': pairs_raw, 'README.md': raw['readme'],
             'CC-BY-SA-4.0-legalcode.txt': raw['legal_notice'],
             'source-manifest.json': manifest_raw, 'ATTRIBUTION.txt': attribution}
    receipt = {
        'schema': RECEIPT_SCHEMA, 'material_role': ROLE,
        'conditional_training_eligible': True, 'formation_training_eligible': False,
        'qualified': False, 'authority': False,
        'manifest_sha256': manifest_sha256, 'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'upstream_repository': manifest['upstream_repository'], 'upstream_revision': manifest['upstream_revision'],
        'license': manifest['license'], 'license_url': LICENSE_URL, 'authors': manifest['authors'],
        'citation': manifest['citation'], 'rights_review': manifest['rights_review'],
        'privacy_origin_limits': manifest['privacy_origin_limits'],
        'split_rule': SPLIT_RULE, 'semantic_slots': list(SLOTS),
        'semantic_reference_count_rule': 'unique_exact_source_target_pairs_per_semantic_input',
        'reserved_official_files': manifest['reserved_official_files'], 'official_splits_used': False,
        'independent_mfm_final': False, 'pairs_sha256': sha256(pairs_raw).hexdigest(),
        'files': {name: {'bytes': len(data), 'sha256': sha256(data).hexdigest()} for name, data in files.items()},
        **summary,
    }
    receipt['receipt_sha256'] = sha256(canonical_bytes(receipt)).hexdigest()
    receipt_raw = canonical_bytes(receipt)+b'\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        _new_file(output_dir/name, data)
    _new_file(output_dir/'receipt.json', receipt_raw)
    actual_digest = sha256(receipt_raw).hexdigest()
    _new_file(output_dir/'receipt.sha256', (actual_digest+'\n').encode('ascii'))
    return {**receipt, 'receipt_file_sha256': actual_digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    receipt = stage_material(args.manifest, args.manifest_sha256, args.cache_dir, args.output_dir)
    print(json.dumps({key: receipt[key] for key in ('source_csv_rows', 'rows', 'duplicate_rows',
          'unique_semantic_inputs', 'split_counts', 'split_semantic_input_counts', 'pairs_sha256',
          'receipt_file_sha256')}, sort_keys=True))


if __name__ == '__main__':
    main()
