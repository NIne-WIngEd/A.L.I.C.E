"""Whole v1.6 target geometry against an exact complete J6 source packet.

Field and literal choices are separately authored, FILE-pinned operands. No
choice is projected from geometry, and this module grants no material rights.
"""
from copy import deepcopy
from hashlib import sha256

from cognitive_kernel.canonical import canonical_sha256, require_sha256
from cognitive_kernel import formation_learning_v16 as learning
from . import formation_source_alignment as source_api
from . import formation_target_alignment as target_api
from . import formation_field_responsibility as field_api
from . import train_v16_formation_specialist as target_ids_api
from . import formation_paired_teaching_v6 as paired

PAIR = 'mfm-paired-full-source-j6-joint-target-geometry-v1'
CHOICES = 'mfm-j6-external-field-and-literal-choices-v1'


def _choice_packet(raw, expected, frozen, target):
    choice = paired.pinned(raw, expected, CHOICES)
    paired.need(set(choice) == {'schema', 'input_freeze_sha256',
        'target_record_sha256', 'author_id', 'author_provenance_file_sha256',
        'field_annotations', 'literal_declarations',
        'whole_literal_inventory_complete', 'record_sha256'} and
        choice['input_freeze_sha256'] == frozen['record_sha256'] and
        choice['target_record_sha256'] == target['record_sha256'] and
        type(choice['author_id']) is str and bool(choice['author_id']) and
        type(choice['field_annotations']) is list and
        type(choice['literal_declarations']) is list and
        choice['whole_literal_inventory_complete'] is True,
        'external whole choice packet differs')
    require_sha256(choice['author_provenance_file_sha256'],
                   'independently pinned author provenance FILE')
    return choice


def field_rows(geometry, frozen, annotations):
    policy = field_api.normative_policy()
    candidates = frozen['source']['meaning']['candidate_inventory']
    rows = []
    for node in geometry['nodes']:
        rules = [rule for rule in policy['rules']
                 if field_api._match(node['path'], rule['path'])]
        paired.need(len(rules) == 1 and node['kind'] in rules[0]['kinds'],
                    'unknown normative target path/type')
        rule = rules[0]
        eligible, matching = [], []
        if rule['candidate_path'] is not None and node['kind'] == 'string':
            eligible = [i for i, candidate in enumerate(candidates)
                if rule['candidate_path'] in candidate['eligible_paths']]
            matching = [i for i in eligible
                if candidates[i]['ref_id'] == node['value']]
        rows.append({'pointer': node['pointer'], 'kind': node['kind'],
            'responsibility': rule['responsibility'],
            'candidate_path': rule['candidate_path'],
            'eligible_candidate_indices': eligible,
            'matching_candidate_indices': matching,
            'candidate_status': ('not_applicable' if rule['candidate_path'] is None
                or node['kind'] == 'null' else 'missing' if not matching else
                'ambiguous' if len(matching) > 1 else 'unique'),
            'annotation_status': 'unannotated'})
    by_pointer = {row['pointer']: row for row in rows}
    seen = set()
    for annotation in annotations:
        paired.need(type(annotation) is dict and
                    annotation.get('pointer') in by_pointer and
                    annotation['pointer'] not in seen,
                    'invalid or duplicate authored pointer')
        pointer = annotation['pointer']
        seen.add(pointer)
        row = by_pointer[pointer]
        paired.need(annotation.get('mode') == row['responsibility'],
                    'annotation cannot reclassify normative field')
        if annotation['mode'] == 'external_selection':
            paired.need(set(annotation) == {'pointer', 'mode',
                'candidate_ordinal'} and
                type(annotation['candidate_ordinal']) is int and
                annotation['candidate_ordinal'] in
                    row['matching_candidate_indices'],
                'authored ordinal is not eligible exact identity')
            row['selected_candidate_ordinal'] = annotation['candidate_ordinal']
        else:
            paired.need(set(annotation) == {'pointer', 'mode'} and
                        annotation['mode'] == 'semantic_authoring',
                        'unsupported authored field annotation')
        row['annotation_status'] = 'authored'
    paired.need(all(row['annotation_status'] == 'authored'
        for row in rows if row['responsibility'] in
            ('semantic_authoring', 'external_selection') and
            not (row['responsibility'] == 'external_selection' and
                 row['kind'] == 'null')),
        'whole authored field inventory or ordinal missing')
    return rows


def literal_rows(geometry, tokens, frozen, declarations):
    nodes = {node['pointer']: node for node in geometry['nodes']}
    source = frozen['source']
    alignment = source['alignment']
    opened = [(item['ref_id'], __import__('base64').b64decode(
        item['payload_base64'], validate=True))
        for item in source['caller']['opened_sources']]
    sources = {item['ref_id']: item for item in alignment['sources']}
    seen, rows = set(), []
    for declaration in declarations:
        paired.need(type(declaration) is dict and set(declaration) ==
            {'pointer', 'target_value_utf8', 'source_occurrences'},
            'literal declaration shape differs')
        node = nodes.get(declaration['pointer'])
        paired.need(node is not None and len(node['path']) == 4 and
            node['path'][0] == 'proposals' and
            node['path'][2:] == ['proposal', 'value_text'],
            'only explicit value_text literal segments supported')
        span = declaration['target_value_utf8']
        paired.need(type(span) is list and len(span) == 2 and
            all(type(value) is int for value in span),
            'invalid literal UTF8 span')
        raw = node['value'].encode('utf-8')
        boundaries = {0, len(raw), *(item['value_utf8'][0]
            for item in node['characters'])}
        paired.need(0 <= span[0] < span[1] <= len(raw) and
            all(value in boundaries for value in span),
            'literal splits original UTF8 atom')
        key = (declaration['pointer'], *span)
        paired.need(key not in seen, 'duplicate authored literal')
        seen.add(key)
        needle = raw[slice(*span)]
        occurrences = []
        for ref, data in opened:
            source_row = sources[ref]
            bounds = {item['original_utf8'][0]: item['rendered_chars'][0]
                for item in source_row['characters']}
            bounds[len(data)] = source_row['rendered_content_chars'][1]
            start = data.find(needle)
            while start >= 0:
                end = start + len(needle)
                paired.need(start in bounds and end in bounds,
                            'source literal splits UTF8 atom')
                a, b = bounds[start], bounds[end]
                indices = [item['token_index'] for item in alignment['tokens']
                    if item['source_ref'] == ref and
                    item['observed_source_eligible'] and
                    item['rendered_chars'][0] < b and
                    item['rendered_chars'][1] > a]
                occurrences.append({'source_ref': ref,
                    'original_utf8': [start, end], 'rendered_chars': [a, b],
                    'supporting_token_indices': indices})
                start = data.find(needle, start + 1)
        paired.need(bool(occurrences),
                    'declared literal has no original source alternative')
        selected = declaration['source_occurrences']
        paired.need(type(selected) is list and bool(selected) and
            len({canonical_sha256(item) for item in selected}) == len(selected),
            'whole authored copy alternatives missing or duplicate')
        for item in selected:
            paired.need(type(item) is dict and set(item) ==
                {'source_ref', 'original_utf8'} and
                type(item['source_ref']) is str and
                type(item['original_utf8']) is list and
                len(item['original_utf8']) == 2 and
                all(type(value) is int for value in item['original_utf8']) and
                any(item == {'source_ref': option['source_ref'],
                             'original_utf8': option['original_utf8']}
                    for option in occurrences),
                'authored whole copy tuple is not an original alternative')
        atoms = [item for item in node['characters']
            if span[0] <= item['value_utf8'][0] <
               item['value_utf8'][1] <= span[1]]
        a, b = atoms[0]['canonical_chars'][0], atoms[-1]['canonical_chars'][1]
        body_indices = [item['body_index'] for item in tokens
            if not item['special_mask'] and
            item['canonical_chars'][0] < b and
            item['canonical_chars'][1] > a]
        rows.append({'pointer': declaration['pointer'],
            'target_value_utf8': span, 'literal_utf8_hex': needle.hex(),
            'canonical_chars': [a, b], 'target_body_indices': body_indices,
            'occurrences': occurrences, 'selected_occurrences': selected,
            'token_id_copy_claimed': False})
    return rows


def _core(frozen, target, choice_sha, choice_raw, choice, geometry, inputs, labels,
          tokens, maximum, fields, literals):
    return {'schema': PAIR, 'input_freeze_sha256': frozen['record_sha256'],
        'target_record_sha256': target['record_sha256'],
        'original_target_file_sha256': target['original_target_file_sha256'],
        'choice_file_sha256': choice_sha,
        'original_choice_hex': choice_raw.hex(),
        'author_provenance_file_sha256': choice['author_provenance_file_sha256'],
        'source_file_sha256': frozen['source_file_sha256'],
        'source_cache': deepcopy(frozen['source_cache']),
        'native_batch': deepcopy(frozen['native_batch']),
        'processor_binding': deepcopy(frozen['processor_binding']),
        'canonical_output_sha256': target['canonical_output_sha256'],
        'max_target_tokens': maximum, 'geometry': geometry,
        'decoder_inputs': inputs, 'labels': labels, 'tokens': tokens,
        'scaffolding': {'bos': {'decoder_input_index': 0,
            'token_id': inputs[0]}, 'eos': {'label_index': len(tokens),
            'query_index': len(tokens), 'token_id': labels[-1]}},
        'policy': field_api.normative_policy(), 'fields': fields,
        'choice_packet': choice, 'literal_segments': literals,
        'learning_qualified': False, 'authority_granted': False,
        'processor_reexecuted_for_target': True}


def align_target(j5, v6, processor, frozen, target, *,
                 expected_input_sha256, expected_source_file_sha256,
                 migration_reader, choice_reader, expected_choice_file_sha256,
                 max_target_tokens):
    paired.validate_input(j5, v6, frozen,
        expected_input_sha256=expected_input_sha256,
        expected_source_file_sha256=expected_source_file_sha256,
        migration_reader=migration_reader)
    example, output = paired.require_complete_labels(frozen, target)
    paired.need(type(max_target_tokens) is int and max_target_tokens > 0,
                'explicit complete target budget required')
    choice_raw = choice_reader()
    choice = _choice_packet(choice_raw, expected_choice_file_sha256,
                            frozen, target)
    annotation_raw = bytes.fromhex(target['raw_annotation_hex'])
    annotation = paired.pinned(annotation_raw,
        target['annotation_file_sha256'], paired.ANNOTATION_V2)
    selected = {row['pointer']: row['selected_candidate_ordinal']
        for row in annotation['authored_selection_fields']}
    chosen = {row['pointer']: row['candidate_ordinal']
        for row in choice['field_annotations']
        if row['mode'] == 'external_selection'}
    paired.need(selected == chosen and
        annotation['literal_copy_segments_declared'] ==
            choice['literal_declarations'],
        'external choice FILE differs from author annotation')
    binding = source_api.processor_binding(processor)
    paired.need(canonical_sha256(binding) ==
                canonical_sha256(frozen['processor_binding']),
                'actual target processor differs from frozen source')
    geometry = target_api.canonical_field_geometry(output,
        expected_target_sha256=target['canonical_output_sha256'])
    text = bytes.fromhex(geometry['canonical_utf8_hex']).decode('utf-8')
    try:
        with source_api._processor_execution_settings(processor):
            inputs, labels = target_ids_api._target_ids(
                processor.tokenizer, example, max_target_tokens)
            observed = processor.tokenizer(text, add_special_tokens=False,
                padding=False, truncation=False,
                return_offsets_mapping=True, return_attention_mask=True,
                return_special_tokens_mask=True)
            tokens = target_api._token_rows(geometry, observed['input_ids'],
                observed['attention_mask'], observed['offset_mapping'],
                observed['special_tokens_mask'])
            paired.need(observed['input_ids'] == inputs[1:] == labels[:-1],
                        'complete original target ID sequence differs')
    finally:
        paired.need(canonical_sha256(source_api.processor_binding(processor)) ==
                    canonical_sha256(binding),
                    'processor semantic binding changed during target')
    fields = field_rows(geometry, frozen, choice['field_annotations'])
    literals = literal_rows(geometry, tokens, frozen,
                            choice['literal_declarations'])
    return paired.seal(_core(frozen, target, expected_choice_file_sha256,
        choice_raw, choice, geometry, inputs, labels, tokens, max_target_tokens,
        fields, literals))


def validate_pair(pair, frozen, target, *, expected_pair_sha256):
    require_sha256(expected_pair_sha256, 'whole J6 pair')
    paired.need(type(pair) is dict and pair.get('schema') == PAIR and
        pair.get('record_sha256') == expected_pair_sha256 and
        canonical_sha256({k: v for k, v in pair.items()
            if k != 'record_sha256'}) == expected_pair_sha256,
        'whole J6 pair seal differs')
    example, output = paired.require_complete_labels(frozen, target)
    geometry = target_api.canonical_field_geometry(output,
        expected_target_sha256=target['canonical_output_sha256'])
    choice_raw = bytes.fromhex(pair['original_choice_hex'])
    choice = _choice_packet(choice_raw, pair['choice_file_sha256'],
                            frozen, target)
    paired.need(choice['record_sha256'] == canonical_sha256({k: v
        for k, v in choice.items() if k != 'record_sha256'}) and
        choice['input_freeze_sha256'] == frozen['record_sha256'] and
        choice['target_record_sha256'] == target['record_sha256'] and
        pair['choice_packet'] == choice,
        'whole choice packet seal/ancestry differs')
    annotation = paired.pinned(bytes.fromhex(target['raw_annotation_hex']),
        target['annotation_file_sha256'], paired.ANNOTATION_V2)
    selected = {row['pointer']: row['selected_candidate_ordinal']
        for row in annotation['authored_selection_fields']}
    chosen = {row['pointer']: row['candidate_ordinal']
        for row in choice['field_annotations']
        if row['mode'] == 'external_selection'}
    paired.need(selected == chosen and
        annotation['literal_copy_segments_declared'] ==
            choice['literal_declarations'],
        'whole choice and author annotation differ')
    rows = pair['tokens']
    rebuilt = target_api._token_rows(geometry,
        [row['token_id'] for row in rows],
        [row['attention_mask'] for row in rows],
        [row['canonical_chars'] for row in rows],
        [row['special_mask'] for row in rows])
    inputs, labels, maximum = (pair['decoder_inputs'], pair['labels'],
                               pair['max_target_tokens'])
    paired.need(type(maximum) is int and maximum > 0 and
        type(inputs) is list and type(labels) is list and
        len(inputs) == len(labels) == len(rows) + 1 <= maximum and
        all(type(value) is int and value >= 0 for value in inputs + labels) and
        rows == rebuilt and inputs[1:] == labels[:-1] ==
            [row['token_id'] for row in rows],
        'whole target decoder shift/geometry differs')
    tokenizer = frozen['processor_binding']['tokenizer_wrapper']
    paired.need(inputs[0] == tokenizer['bos_token_id'] and
                labels[-1] == tokenizer['eos_token_id'],
                'exact source-bound BOS/EOS differs')
    fields = field_rows(geometry, frozen, choice['field_annotations'])
    literals = literal_rows(geometry, rows, frozen,
                            choice['literal_declarations'])
    expected = paired.seal(_core(frozen, target, pair['choice_file_sha256'],
        choice_raw, choice, geometry, inputs, labels, rows, maximum,
        fields, literals))
    paired.need(canonical_sha256(pair) == canonical_sha256(expected),
                'whole J6 target/reference/copy packet differs')
    return deepcopy(pair)
