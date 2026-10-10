"""One complete target-free text view with original/candidate/field geometry.

Supplied custody and causal position are observed, not learned use or authority.
Native evidence, missing record families and semantic entailment stay separate.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from cognitive_kernel import canonical
from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes, canonical_sha256
from . import formation_caller_input as caller_api, formation_caller_meaning as meaning_api
from . import formation_source_alignment as source, formation_target_alignment as geometry_api
from . import train_v1_formation_specialist as shared

SCHEMA = 'mfm-joint-caller-input-alignment-v1'
IDENTITY_SCHEMA = 'mfm-joint-source-identity-v1'
TASK = 'formation-complete-caller-input-v1'
PROFILE = 'text-causal-joint-v1'


def _require(condition, message):
    if not condition:
        raise CognitiveKernelContractError(message)


def _seal(core):
    return {**deepcopy(core), 'record_sha256': canonical_sha256(core)}


def _identity(row, expected):
    source._pinned_record(row, expected, IDENTITY_SCHEMA)
    _require(set(row) == {'schema', 'repository', 'revision', 'source_receipt_sha256',
                         'source_frozen', 'precision', 'record_sha256'}, 'source identity fields differ')
    _require(all(type(row[k]) is str and row[k] and '\x00' not in row[k] for k in ('repository', 'revision'))
             and row['source_frozen'] is True and row['precision'] == 'bf16', 'frozen source identity differs')
    canonical.require_sha256(row['source_receipt_sha256'], 'external frozen source receipt FILE')


def joint_code_binding():
    return {'schema': 'mfm-joint-input-code-binding-v1', 'task': TASK, 'profile': PROFILE,
        'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'meaning_code_sha256': sha256(Path(meaning_api.__file__).read_bytes()).hexdigest(),
        'caller_code_sha256': sha256(Path(caller_api.__file__).read_bytes()).hexdigest(),
        'source_geometry_code_sha256': sha256(Path(source.__file__).read_bytes()).hexdigest(),
        'canonical_geometry_code_sha256': sha256(Path(geometry_api.__file__).read_bytes()).hexdigest(),
        'template_sha256': sha256(shared.SOURCE_TEMPLATE.encode()).hexdigest()}


def _compose(atoms, start, end, content_end):
    _require(0 <= start <= end <= len(atoms), 'canonical field exceeds block')
    return [atoms[start]['rendered_chars'][0] if start < len(atoms) else content_end,
            atoms[end - 1]['rendered_chars'][1] if end > start else
            (atoms[start]['rendered_chars'][0] if start < len(atoms) else content_end)]


def _render(caller, meaning, caller_sha256, meaning_sha256):
    context, opened, registry = caller_api.caller_input_from_record(caller)
    _require(all(r.modality in ('text', 'code', 'structured') for r in context.base.evidence),
             'native evidence requires its own complete caller adapter')
    semantic = meaning_api.semantic_view(meaning, caller, meaning_sha256=meaning_sha256, caller_sha256=caller_sha256)
    values = [{'section': 'task_context_snapshot', 'task': TASK, 'profile': PROFILE,
        'context': context.record(), 'snapshot': meaning['snapshot']}]
    values += [{'section': 'observation', 'source_ref': ref, 'content': raw.decode('utf-8', 'strict')}
               for ref, raw in opened]
    values += [{'section': 'candidate_meaning', **row} for row in semantic]
    values += [{'section': 'candidate_index', 'candidates': [
        {'slot': slot, 'ref_id': r['ref_id'], 'kind': r['kind'], 'origin': r['origin'],
         'namespace': r['namespace'], 'eligible_paths': r['eligible_paths']}
        for slot, r in enumerate(meaning['candidate_inventory'])]}]
    blocks, fields, originals, position = [], [], [], 0
    for block_index, value in enumerate(values):
        raw = canonical_json_bytes(value); text = raw.decode('utf-8')
        quoted = shared._escaped_text(text); block_text = 'json_text=' + quoted
        start = position + len('json_text=') + 1
        atoms, end = source._character_map(text, start, ascii_source=True)
        geometry = geometry_api.canonical_field_geometry(value, expected_target_sha256=sha256(raw).hexdigest())
        nodes = {n['pointer']: n for n in geometry['nodes']}
        slot = value.get('slot')
        content = nodes.get('/content')
        body_pin = value.get('body_file_sha256')
        if body_pin is not None:
            _require(content is not None and sha256(raw[slice(*content['canonical_utf8'])]).hexdigest() == body_pin,
                     'supplied body differs from complete rendered content')
        spans = [*geometry['lexemes'], *({**n, 'role': 'container'} for n in geometry['nodes']
                                       if n['kind'] in ('object', 'array'))]
        for span in spans:
            node = nodes[span['pointer']]
            field = {'block_index': block_index, 'section': value['section'], 'candidate_slot': slot,
                'pointer': span['pointer'], 'role': span['role'],
                'kind': 'string' if span['role'] == 'key' else node['kind'],
                'canonical_chars': span['canonical_chars'], 'canonical_utf8': span['canonical_utf8'],
                'rendered_chars': _compose(atoms, *span['canonical_chars'], end),
                'body_file_sha256': None, 'body_original_utf8': None}
            if span['role'] == 'value':
                field['value'] = deepcopy(node['value'])
                if node['kind'] == 'string':
                    field['characters'] = [{'value_utf8': a['value_utf8'],
                        'rendered_chars': _compose(atoms, *a['canonical_chars'], end)} for a in node['characters']]
            if body_pin is not None and span['pointer'].startswith('/content') and (
                    content['canonical_utf8'][0] <= span['canonical_utf8'][0] <= span['canonical_utf8'][1] <= content['canonical_utf8'][1]):
                field.update(body_file_sha256=body_pin, body_original_utf8=[
                    offset - content['canonical_utf8'][0] for offset in span['canonical_utf8']])
            fields.append(field)
        if value['section'] == 'observation':
            source_index = block_index - 1; ref, original = opened[source_index]
            _require(content is not None and content['kind'] == 'string', 'original content is not a canonical string')
            characters = [{'original_utf8': a['value_utf8'],
                'rendered_chars': _compose(atoms, *a['canonical_chars'], end)} for a in content['characters']]
            boundary = _compose(atoms, content['canonical_chars'][0] + 1, content['canonical_chars'][1] - 1, end)
            originals.append({'ref_id': ref, 'modality': context.base.evidence[source_index].modality,
                'content_sha256': sha256(original).hexdigest(), 'original_bytes': len(original),
                'rendered_content_chars': boundary, 'characters': characters, 'block_index': block_index})
        blocks.append({'section': value['section'], 'candidate_slot': slot, 'json_string': quoted,
            'block_utf8_sha256': sha256(raw).hexdigest(), 'rendered_chars': [position, position + len(block_text)],
            'canonical_geometry_sha256': geometry['record_sha256']})
        position += len(block_text) + 1
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'json_text=' + b['json_string']} for b in blocks]}]
    return messages, ''.join('json_text=' + b['json_string'] + '\n' for b in blocks), blocks, fields, originals


def _joins(rendered, originals, ids, masks, offsets, specials, special_ids, fields, blocks, inventory):
    tokens = source._tokens(rendered, originals, ids, masks, offsets, specials, special_ids)
    for index, (row, token, special) in enumerate(zip(tokens, ids, specials)):
        row.update(token_id=token, attention_mask=masks[index], special_mask=special,
                   field_indices=[], candidate_slots=[])
    # Index overlapping offsets without assuming one token per character or
    # monotone byte offsets. Repeated byte-token offsets remain distinct tokens.
    valid = sorted((r['rendered_chars'][0], r['rendered_chars'][1], r['token_index']) for r in tokens
        if r['rendered_chars'][0] < r['rendered_chars'][1] and not r['special_mask'] and r['token_id'] not in special_ids)
    starts, max_ends, largest = [], [], 0
    for a, b, _ in valid:
        starts.append(a); largest = max(largest, b); max_ends.append(largest)
    def overlaps(span):
        a, b = span
        return sorted(j for x, y, j in valid[bisect_right(max_ends, a):bisect_left(starts, b)] if y > a)
    for index, field in enumerate(fields):
        field['supporting_token_indices'] = overlaps(field['rendered_chars'])
        for token_index in field['supporting_token_indices']:
            tokens[token_index]['field_indices'].append(index)
    candidates = []
    footer_index = len(blocks) - 1
    for slot, candidate in enumerate(inventory):
        meaning_block = next(b for b in blocks if b['section'] == 'candidate_meaning' and b['candidate_slot'] == slot)
        meaning_indices = overlaps(meaning_block['rendered_chars'])
        index_fields = [f for f in fields if f['block_index'] == footer_index and
                        f['pointer'] == '/candidates/' + str(slot) and f['role'] == 'container']
        _require(len(index_fields) == 1, 'complete typed index geometry differs')
        index_indices = index_fields[0]['supporting_token_indices']
        candidates.append({'slot': slot, 'ref_id': candidate['ref_id'], 'kind': candidate['kind'],
            'origin': candidate['origin'], 'namespace': candidate['namespace'],
            'eligible_paths': deepcopy(candidate['eligible_paths']), 'meaning_token_indices': meaning_indices,
            'index_token_indices': index_indices, 'independent_observation': False})
        for token_index in sorted(set(meaning_indices + index_indices)):
            tokens[token_index]['candidate_slots'].append(slot)
    return tokens, candidates


def joint_source_batch_with_alignment(processor, caller, meaning, *, caller_sha256, meaning_sha256,
                                     processor_binding, source_identity, source_identity_sha256, max_source_tokens):
    """Process the full supplied input only; refuse loss of any whole-view token."""
    meaning_api.caller_meaning_from_record(meaning, caller, meaning_sha256=meaning_sha256, caller_sha256=caller_sha256)
    _identity(source_identity, source_identity_sha256)
    _require(type(max_source_tokens) is int and max_source_tokens > 0, 'explicit complete source budget required')
    binding = source.processor_binding(processor)
    _require(binding == processor_binding, 'externally bound processor differs')
    messages, assembled, blocks, fields, originals = _render(caller, meaning, caller_sha256, meaning_sha256)
    rendered = processor.apply_chat_template(messages, chat_template=shared.SOURCE_TEMPLATE,
        tokenize=False, add_generation_prompt=False)
    _require(type(rendered) is str and rendered == assembled, 'actual complete joint template differs')
    with source._processor_execution_settings(processor):
        batch = processor.apply_chat_template(messages, chat_template=shared.SOURCE_TEMPLATE,
            tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt',
            processor_kwargs={'do_sample_frames': False})
        _require(all(k in batch and hasattr(batch[k], 'tolist') and len(batch[k].shape) == 2 and
                     batch[k].shape[0] == 1 for k in ('input_ids', 'attention_mask')), 'complete processor batch missing')
        ids, masks = batch['input_ids'][0].tolist(), batch['attention_mask'][0].tolist()
        _require(0 < len(ids) <= max_source_tokens and len(ids) == len(masks), 'complete source budget overflow or shape differs')
        encoded = processor.apply_chat_template(messages, chat_template=shared.SOURCE_TEMPLATE,
            tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt',
            processor_kwargs={'do_sample_frames': False, 'return_offsets_mapping': True,
                              'return_special_tokens_mask': True, 'return_attention_mask': True})
        expected = {'input_ids': (1, len(ids)), 'attention_mask': (1, len(ids)),
            'offset_mapping': (1, len(ids), 2), 'special_tokens_mask': (1, len(ids))}
        _require(all(k in encoded and hasattr(encoded[k], 'tolist') and tuple(encoded[k].shape) == shape
                     for k, shape in expected.items()), 'actual complete processor geometry malformed')
        _require(encoded['input_ids'][0].tolist() == ids and encoded['attention_mask'][0].tolist() == masks,
                 'actual full processor IDs/masks differ from offset observation')
    _require(source.processor_binding(processor) == binding, 'processor binding changed during joint observation')
    offsets, specials = encoded['offset_mapping'][0].tolist(), encoded['special_tokens_mask'][0].tolist()
    tokens, candidates = _joins(rendered, originals, ids, masks, offsets, specials,
                               set(processor.tokenizer.all_special_ids), fields, blocks, meaning['candidate_inventory'])
    codes = joint_code_binding()
    view_key = canonical_sha256({'task': TASK, 'profile': PROFILE, 'codes': codes, 'source_identity': source_identity,
        'processor_binding': binding, 'rendered_utf8_sha256': sha256(rendered.encode()).hexdigest(),
        'input_ids': ids, 'attention_mask': masks, 'max_source_tokens': max_source_tokens})
    provenance_key = canonical_sha256({'caller_sha256': caller_sha256, 'meaning_sha256': meaning_sha256,
        'candidate_inventory_sha256': canonical_sha256(meaning['candidate_inventory']),
        'snapshot_sha256': meaning['snapshot']['record_sha256'], 'source_identity_sha256': source_identity_sha256})
    return batch, _seal({'schema': SCHEMA, 'task': TASK, 'profile': PROFILE, 'code_binding': codes,
        'caller_sha256': caller_sha256, 'meaning_sha256': meaning_sha256, 'processor_binding': binding,
        'source_identity': source_identity, 'max_source_tokens': max_source_tokens,
        'candidate_inventory_sha256': canonical_sha256(meaning['candidate_inventory']),
        'rendered_text': rendered, 'rendered_utf8_sha256': sha256(rendered.encode()).hexdigest(),
        'input_ids': ids, 'attention_mask': masks, 'blocks': blocks, 'fields': fields,
        'sources': originals, 'candidates': candidates, 'tokens': tokens,
        'cache': {'processed_view_sha256': view_key, 'provenance_sha256': provenance_key},
        'text_attention': 'causal_full_or_sliding_as_configured', 'teacher_accessed': False,
        'native_locator_alignment': False, 'learning_qualified': False, 'authority_granted': False})


def verify_joint_alignment(processor, caller, meaning, sidecar, *, alignment_sha256, **kwargs):
    """Rebuild every join and cache binding against actual inputs/processor."""
    source._pinned_record(sidecar, alignment_sha256, SCHEMA)
    _, actual = joint_source_batch_with_alignment(processor, caller, meaning, **kwargs)
    _require(actual == sidecar, 'actual complete joint alignment differs')
    return actual
