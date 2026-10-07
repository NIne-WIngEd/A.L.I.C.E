"""Complete public conditional-source/target census, without weights or fitting.

This auxiliary material is not personal evidence or a formation target. The
original processor template is reused explicitly; the conditional task wrapper
is new and versioned. The receipt never qualifies language or memory behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from scripts.mfm.train_v1_formation_specialist import (
    SOURCE_TEMPLATE, _escaped_text, _prepared_base, _require_private_network_isolation,
)
from scripts.mfm.prepare_public_conditional_material import validate_manifest, semantic_mr

TASK = 'conditional_meaning_to_text'
SCHEMA = 'mfm-public-conditional-processor-census-v1'


def digest(path):
    with Path(path).open('rb') as stream:
        result = sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def encode_pair(processor, row, max_source_tokens, max_target_tokens, vocabulary_size):
    for name in ('source', 'target'):
        text = row.get(name + '_text')
        if type(text) is not str or not text or sha256(text.encode('utf-8')).hexdigest() != row.get(name + '_payload_sha256'):
            raise ValueError(name + ' original text digest differs')
    if any(type(n) is not int or n < 1 for n in (max_source_tokens, max_target_tokens, vocabulary_size)):
        raise ValueError('positive source/target budgets and vocabulary required')
    tokenizer = processor.tokenizer
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if any(type(token) is not int or not 0 <= token < vocabulary_size for token in (bos, eos)) or bos == eos:
        raise ValueError('valid distinct BOS/EOS required')
    rendered = 'task=' + TASK + '\njson_text=' + _escaped_text(row['source_text'])
    payload = processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'text', 'text': rendered}]}],
        chat_template=SOURCE_TEMPLATE, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt', processor_kwargs={'do_sample_frames': False})
    tokens = payload.get('input_ids'); mask = payload.get('attention_mask')
    if tokens is None or tokens.ndim != 2 or tokens.shape[0] != 1 or not 0 < tokens.shape[1] <= max_source_tokens:
        raise ValueError('complete source exceeds budget or lacks tokens; do not truncate')
    if mask is None or mask.shape != tokens.shape or not bool((mask == 1).all()):
        raise ValueError('single complete source requires an aligned unpadded mask')
    source_ids = tokens[0].tolist()
    target = canonical_json_bytes({'value_text': row['target_text']}).decode('utf-8')
    body = tokenizer.encode(target, add_special_tokens=False)
    if not body or len(body) + 1 > max_target_tokens:
        raise ValueError('complete target exceeds budget; do not truncate')
    if any(type(token) is not int or not 0 <= token < vocabulary_size for token in [*source_ids, *body]):
        raise ValueError('processor token exceeds prepared vocabulary')
    if bos in body or eos in body:
        raise ValueError('target contains interior reserved BOS/EOS; cannot terminate faithfully')
    if tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target:
        raise ValueError('canonical human target does not round-trip through prepared tokenizer')
    return {'case_id': row['case_id'], 'split': row['split'], 'source_ids': source_ids,
            'decoder_inputs': [bos, *body], 'labels': [*body, eos]}


def token_statistics(records):
    frequencies = Counter(); source_counts = Counter(); split_counts = Counter()
    source_tokens = Counter(); target_tokens = Counter(); seen = set()
    source_max = target_max = pairs_max = 0
    for record in records:
        case_id = record['case_id']; split = record['split']
        if case_id in seen:
            raise ValueError('duplicate conditional case')
        seen.add(case_id)
        if split not in {'train', 'development'}:
            raise ValueError('conditional census cannot open FINAL')
        source = record['source_ids']; inputs = record['decoder_inputs']; labels = record['labels']
        if not source or not labels or len(inputs) != len(labels) or inputs[1:] != labels[:-1]:
            raise ValueError('incomplete or unshifted conditional sequence')
        split_counts[split] += 1; source_tokens[split] += len(source); target_tokens[split] += len(labels)
        source_max = max(source_max, len(source)); target_max = max(target_max, len(labels))
        if split == 'train':
            frequencies.update(labels); source_counts.update(source)
            pairs_max = max(pairs_max, len(source) * len(labels))
    if not seen:
        raise ValueError('empty conditional census')
    return {'rows': len(seen), 'split_counts': dict(split_counts),
            'source_tokens': dict(source_tokens), 'target_tokens': dict(target_tokens),
            'development_label_tokens': target_tokens['development'],
            'largest_source_tokens': source_max, 'largest_target_tokens': target_max,
            'largest_train_source_target_pairs': pairs_max,
            'train_positive_label_counts': {str(k): v for k,v in sorted(frequencies.items())},
            'train_source_token_counts': {str(k): v for k,v in sorted(source_counts.items())},
            'train_positive_label_types': len(frequencies),
            'train_label_types_once': sum(n == 1 for n in frequencies.values()),
            'train_label_types_2_to_9': sum(2 <= n <= 9 for n in frequencies.values())}


def source_inputs(records):
    """Whitelist observed inputs into a separate target-free export contract.

    Alternative human references share a feature only when the actual processed
    input is identical. Semantic equivalence alone does not imply identical
    token order or contextual hidden states.
    """
    groups = {}
    for row in records:
        ids = row['source_ids']
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('invalid processed public source IDs')
        key = canonical_sha256({'task': TASK, 'source_ids': ids})
        item = groups.setdefault(key, {'schema': 'mfm-public-conditional-source-input-v1',
            'task': TASK, 'source_key': key, 'source_ids': ids, 'split': row['split'],
            'attention_mask': [1] * len(ids), 'case_ids': set(),
            'semantic_input_sha256s': set(), 'source_payload_sha256s': set()})
        if row['split'] not in {'train', 'development'} or item['split'] != row['split']:
            raise ValueError('same processed input crosses teaching/development splits')
        item['case_ids'].add(row['case_id'])
        item['semantic_input_sha256s'].add(row['semantic_input_sha256'])
        item['source_payload_sha256s'].add(row['source_payload_sha256'])
    return [{name: sorted(value) if isinstance(value, set) else value for name,value in groups[key].items()}
            for key in sorted(groups)]


def material_rows(directory, expected_receipt_file_sha256, manifest_path, manifest_sha256):
    directory = Path(directory)
    if digest(directory / 'receipt.json') != expected_receipt_file_sha256 or digest(manifest_path) != manifest_sha256:
        raise ValueError('conditional material/manifest external file pin differs')
    receipt = json.loads((directory / 'receipt.json').read_text('utf-8'))
    sealed = {key: value for key,value in receipt.items() if key != 'receipt_sha256'}
    if sha256(canonical_json_bytes(sealed)).hexdigest() != receipt.get('receipt_sha256'):
        raise ValueError('conditional material self seal differs')
    scope = {'schema': 'mfm-public-conditional-material-receipt-v1',
             'material_role': 'conditional_meaning_to_text_only',
             'conditional_training_eligible': True, 'formation_training_eligible': False,
             'qualified': False, 'authority': False, 'official_splits_used': False,
             'independent_mfm_final': False}
    if any(type(receipt.get(key)) is not type(value) or receipt.get(key) != value for key,value in scope.items()):
        raise ValueError('conditional material scope or authority differs')
    manifest = json.loads(Path(manifest_path).read_text('utf-8'))
    validate_manifest(manifest)
    for key in ('upstream_revision', 'upstream_repository', 'license', 'license_url',
                'authors', 'reserved_official_files', 'semantic_slots', 'split_rule'):
        if receipt.get(key) != manifest[key]:
            raise ValueError('conditional origin/rights/split binding differs')
    if receipt.get('source_train_sha256') != manifest['train']['sha256'] or receipt.get('source_csv_rows') != manifest['train']['rows']:
        raise ValueError('conditional original training source binding differs')
    expected_files = {'pairs.jsonl', 'README.md', 'CC-BY-SA-4.0-legalcode.txt', 'ATTRIBUTION.txt', 'source-manifest.json'}
    if type(receipt.get('files')) is not dict or set(receipt['files']) != expected_files:
        raise ValueError('conditional material notice/custody roster differs')
    for name, item in receipt['files'].items():
        path = directory / name
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('conditional retained material or notice differs')
    original_notices = {'README.md': manifest['readme']['sha256'],
                        'CC-BY-SA-4.0-legalcode.txt': manifest['legal_notice']['sha256'],
                        'source-manifest.json': manifest_sha256}
    if any(receipt['files'][name]['sha256'] != expected for name,expected in original_notices.items()):
        raise ValueError('original grant or source notice does not match its pinned parent')
    if receipt.get('manifest_sha256') != manifest_sha256 or digest(directory / 'pairs.jsonl') != receipt.get('pairs_sha256'):
        raise ValueError('conditional pairs/manifest binding differs')
    rows = [json.loads(line) for line in (directory / 'pairs.jsonl').read_text('utf-8').splitlines()]
    if len(rows) != receipt.get('rows') or Counter(row['split'] for row in rows) != receipt.get('split_counts'):
        raise ValueError('complete conditional row/split roster differs')
    groups = Counter(row['semantic_input_sha256'] for row in rows); splits = {}; seen_cases = set(); seen_pairs = set()
    for row in rows:
        if row.get('schema') != 'mfm-public-conditional-pair-v1' or row['split'] not in {'train', 'development'}:
            raise ValueError('unsupported public conditional record')
        group = row['semantic_input_sha256']
        if row['case_id'] in seen_cases or (row['source_payload_sha256'], row['target_payload_sha256']) in seen_pairs:
            raise ValueError('duplicate conditional case/pair identity')
        seen_cases.add(row['case_id']); seen_pairs.add((row['source_payload_sha256'], row['target_payload_sha256']))
        for name in ('source', 'target'):
            if sha256(row[name+'_text'].encode('utf-8')).hexdigest() != row[name+'_payload_sha256']:
                raise ValueError('conditional original field digest differs')
        expected_group = canonical_sha256(semantic_mr(row['source_text']))
        pair_hash = canonical_sha256({'source_text': row['source_text'], 'target_text': row['target_text']})
        if (group != expected_group or row['case_id'] != 'e2e-'+pair_hash or
                row['source_file_sha256'] != manifest['train']['sha256'] or row['upstream_revision'] != manifest['upstream_revision']):
            raise ValueError('conditional semantic/input/pair/source identity differs')
        if row['split'] != ('development' if int(group,16)%10 == 0 else 'train'):
            raise ValueError('conditional target-independent split differs')
        if row.get('semantic_reference_count') != groups[group]:
            raise ValueError('conditional semantic reference mass differs')
        if splits.setdefault(group, row['split']) != row['split']:
            raise ValueError('semantic input crosses teaching/development splits')
    return rows, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material-dir', type=Path, required=True)
    parser.add_argument('--material-receipt-file-sha256', required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--prepared-base-dir', type=Path, required=True)
    parser.add_argument('--prepared-base-receipt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-source-tokens', type=int, default=8192)
    parser.add_argument('--max-target-tokens', type=int, default=8192)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError('census output already exists; do not replace a receipt')
    _require_private_network_isolation()
    rows, material = material_rows(args.material_dir, args.material_receipt_file_sha256,
                                  args.source_manifest, args.source_manifest_sha256)
    prepared = _prepared_base(args)
    import transformers
    import torch
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(prepared['snapshot_path'], trust_remote_code=False, local_files_only=True)
    config = json.loads((Path(prepared['snapshot_path']) / 'config.json').read_text('utf-8'))
    vocabulary = config['text_config']['vocab_size']
    args.output_dir.mkdir(mode=0o700)
    encoded_path = args.output_dir / 'encoded-pairs.jsonl'
    started = time.monotonic()
    observed_inputs = []
    def encoded_records():
        with encoded_path.open('x', encoding='utf-8', newline='\n') as stream:
            for index, row in enumerate(rows):
                encoded = encode_pair(processor, row, args.max_source_tokens, args.max_target_tokens, vocabulary)
                encoded.update(semantic_input_sha256=row['semantic_input_sha256'],
                               semantic_reference_count=row['semantic_reference_count'],
                               source_payload_sha256=row['source_payload_sha256'])
                stream.write(canonical_json_bytes(encoded).decode('utf-8') + '\n')
                observed_inputs.append({key: encoded[key] for key in (
                    'case_id', 'split', 'source_ids', 'semantic_input_sha256', 'source_payload_sha256')})
                if (index+1) % 2000 == 0:
                    print(json.dumps({'processed_public_pairs': index+1}), flush=True)
                yield encoded
    statistics = token_statistics(encoded_records())
    inputs = source_inputs(observed_inputs)
    source_path = args.output_dir / 'source-inputs.jsonl'
    with source_path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in inputs:
            stream.write(canonical_json_bytes(row).decode('utf-8') + '\n')
    receipt = {'schema': SCHEMA, 'task': TASK, 'material_receipt_sha256': material['receipt_sha256'],
               'material_receipt_file_sha256': args.material_receipt_file_sha256,
               'source_manifest_sha256': args.source_manifest_sha256,
               'prepared_clone_receipt_sha256': prepared['receipt_sha256'],
               'prepared_base_weight_sha256': next(item['sha256'] for item in prepared['files'] if item['path'] == 'model.safetensors'),
               'base_hidden_size': config['text_config']['hidden_size'],
               'foundation_vocabulary': vocabulary, 'transformers_version': transformers.__version__,
               'torch_version': torch.__version__, 'encoded_pairs_sha256': digest(encoded_path),
               'source_inputs_sha256': digest(source_path), 'unique_source_inputs': len(inputs),
               'processor_template_sha256': sha256(SOURCE_TEMPLATE.encode()).hexdigest(),
               'task_wrapper': 'task=conditional_meaning_to_text\\njson_text=<escaped original MR>',
               'target_wrapper': {'value_text': '<unchanged human reference>'},
               'code_sha256': digest(__file__), 'source_processor_helper_sha256': digest(Path(__file__).with_name('train_v1_formation_specialist.py')),
               'max_source_tokens': args.max_source_tokens, 'max_target_tokens': args.max_target_tokens,
               'statistics': statistics, 'seconds': time.monotonic() - started,
               'private_evidence_opened': False, 'model_weights_loaded': False,
               'optimizer_created': False, 'authority_written': False, 'qualified_for_product': False,
               'independent_gold': False, 'final_payloads_opened': False}
    receipt['record_sha256'] = sha256(canonical_json_bytes(receipt)).hexdigest()
    (args.output_dir / 'receipt.json').write_bytes(canonical_json_bytes(receipt) + b'\n')
    print(json.dumps({'public_conditional_census_sha256': receipt['record_sha256'],
                      'rows': statistics['rows'], 'qualified_for_product': False}), flush=True)


if __name__ == '__main__':
    main()
