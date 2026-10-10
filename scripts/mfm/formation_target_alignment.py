"""Complete canonical target geometry beside the unchanged v16 target producer.

Offsets describe one complete target encoding, never an isolated-field token
encoding or a teacher-derived input. External pins supply custody, not truth,
write authority, native alignment or learned field decisions.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from cognitive_kernel import canonical, formation_learning_v16 as learning
from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes, canonical_sha256
from . import formation_source_alignment as source
from . import train_v16_formation_specialist as v16

GEOMETRY_SCHEMA = 'mfm-canonical-target-field-geometry-v1'
SCHEMA = 'mfm-v16-target-alignment-v1'
PROFILE = 'output_v16'


def _require(condition, message):
    if not condition:
        raise CognitiveKernelContractError(message)


def _seal(core):
    return {**deepcopy(core), 'record_sha256': canonical_sha256(core)}


def _pin(row, expected, schema):
    canonical.require_sha256(expected, 'external alignment digest')
    _require(type(row) is dict and row.get('schema') == schema and row.get('record_sha256') == expected,
             'externally bound target record differs')
    _require(canonical_sha256({k: v for k, v in row.items() if k != 'record_sha256'}) == expected,
             'target record seal differs')


def _material(value):
    if type(value) is dict:
        _require(all(type(k) is str for k in value), 'canonical object keys must be strings')
        for key, item in value.items():
            key.encode('utf-8', 'strict'); _material(item)
    elif type(value) is list:
        for item in value:
            _material(item)
    elif type(value) is str:
        value.encode('utf-8', 'strict')
    elif type(value) is float:
        _require(math.isfinite(value), 'canonical number must be finite')
    else:
        _require(value is None or type(value) in (int, bool), 'unsupported canonical target material')


def _prefix(text):
    result = [0]
    for char in text:
        result.append(result[-1] + len(char.encode('utf-8')))
    return result


def _pointer(path):
    return ''.join('/' + str(part).replace('~', '~0').replace('/', '~1') for part in path)


def canonical_field_geometry(record, *, expected_target_sha256):
    """Scan authoritative serializer output; scalar types and all syntax survive."""
    canonical.require_sha256(expected_target_sha256, 'external canonical target digest')
    try:
        _material(record)
        raw = canonical_json_bytes(record)
    except (UnicodeError, TypeError, ValueError) as exc:
        raise CognitiveKernelContractError('invalid canonical target material') from exc
    _require(sha256(raw).hexdigest() == expected_target_sha256, 'canonical target byte pin differs')
    text = raw.decode('utf-8'); prefix = _prefix(text)
    nodes, lexemes = [], []
    position = 0

    def ranges(start, end):
        return {'canonical_chars': [start, end], 'canonical_utf8': [prefix[start], prefix[end]]}

    def lexeme(pointer, role, piece):
        nonlocal position
        start = position
        _require(text.startswith(piece, position), 'canonical lexical correspondence differs')
        position += len(piece)
        lexemes.append({'pointer': pointer, 'role': role, **ranges(start, position)})

    def scan(value, path):
        nonlocal position
        pointer = _pointer(path); start = position
        kind = ('object' if type(value) is dict else 'array' if type(value) is list else
                'string' if type(value) is str else 'boolean' if type(value) is bool else
                'integer' if type(value) is int else 'number' if type(value) is float else 'null')
        node = {'pointer': pointer, 'path': deepcopy(path), 'kind': kind}
        nodes.append(node)
        if kind == 'object':
            lexeme(pointer, 'open', '{')
            for index, key in enumerate(sorted(value)):
                child = _pointer([*path, key])
                if index:
                    lexeme(pointer, 'comma', ',')
                lexeme(child, 'key', canonical_json_bytes(key).decode('utf-8'))
                lexeme(child, 'colon', ':')
                scan(value[key], [*path, key])
            lexeme(pointer, 'close', '}')
        elif kind == 'array':
            lexeme(pointer, 'open', '[')
            for index, item in enumerate(value):
                if index:
                    lexeme(pointer, 'comma', ',')
                scan(item, [*path, index])
            lexeme(pointer, 'close', ']')
        else:
            node['value'] = deepcopy(value)
            lexeme(pointer, 'value', canonical_json_bytes(value).decode('utf-8'))
            if kind == 'string':
                at, value_byte = start + 1, 0
                atoms = []
                for char in value:
                    piece = canonical_json_bytes(char).decode('utf-8')[1:-1]
                    end = at + len(piece); value_end = value_byte + len(char.encode('utf-8'))
                    _require(text[at:end] == piece, 'string atom correspondence differs')
                    atoms.append({'value_utf8': [value_byte, value_end], **ranges(at, end)})
                    at, value_byte = end, value_end
                _require(at == position - 1, 'complete canonical string atoms differ')
                node['characters'] = atoms
        node.update(ranges(start, position))

    scan(record, [])
    _require(position == len(text), 'canonical scanner left unconsumed material')
    _require(b''.join(raw[slice(*row['canonical_utf8'])] for row in lexemes) == raw,
             'complete canonical syntax reconstruction differs')
    return _seal({'schema': GEOMETRY_SCHEMA, 'target_sha256': expected_target_sha256,
                  'canonical_utf8_hex': raw.hex(), 'nodes': nodes, 'lexemes': lexemes})


def target_code_binding():
    return {'schema': 'mfm-target-alignment-code-binding-v1',
            'profile': PROFILE, 'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'canonical_sha256': sha256(Path(canonical.__file__).read_bytes()).hexdigest(),
            'learning_codec_sha256': sha256(Path(learning.__file__).read_bytes()).hexdigest(),
            'v16_target_sha256': sha256(Path(v16.__file__).read_bytes()).hexdigest()}


def _token_rows(geometry, ids, masks, offsets, specials):
    text = bytes.fromhex(geometry['canonical_utf8_hex']).decode('utf-8'); prefix = _prefix(text)
    _require(all(type(v) in (list, tuple) for v in (ids, masks, offsets, specials)) and
             len(ids) == len(masks) == len(offsets) == len(specials), 'complete target observation lengths differ')
    spans = [*geometry['lexemes'], *({'pointer': n['pointer'], 'role': 'container',
                                    'canonical_chars': n['canonical_chars']}
                                   for n in geometry['nodes'] if n['kind'] in ('object', 'array'))]
    rows = []
    for j, (token, mask, offset, special) in enumerate(zip(ids, masks, offsets, specials)):
        _require(type(token) is int and token >= 0 and type(mask) is int and mask == 1 and
                 type(special) is int and special in (0, 1), 'invalid actual target ID or mask')
        _require(type(offset) in (list, tuple) and len(offset) == 2 and all(type(v) is int for v in offset)
                 and 0 <= offset[0] <= offset[1] <= len(text), 'invalid actual target offset')
        a, b = offset; links = []
        for span in spans:
            start, end = span['canonical_chars']; lo, hi = max(a, start), min(b, end)
            if lo < hi:
                links.append({'pointer': span['pointer'], 'role': span['role'],
                              'canonical_chars': [lo, hi], 'canonical_utf8': [prefix[lo], prefix[hi]],
                              'fully_contained': start <= a and b <= end})
        rows.append({'body_index': j, 'label_index': j, 'query_index': j, 'decoder_input_index': j + 1,
                     'token_id': token, 'attention_mask': mask, 'special_mask': special,
                     'canonical_chars': [a, b], 'canonical_utf8': [prefix[a], prefix[b]], 'links': links})
    return rows


def target_batch_with_alignment(processor, caller, example, *, caller_sha256, source_alignment,
                                source_alignment_sha256, processor_binding, target_sha256,
                                max_source_tokens, max_target_tokens):
    """Observe complete actual IDs/offsets and compare the unchanged v16 shift."""
    _require(type(max_source_tokens) is int and max_source_tokens > 0 and
             type(max_target_tokens) is int and max_target_tokens > 0, 'explicit positive token budgets required')
    context, opened, _ = source._caller(caller, caller_sha256)
    _require(context.record() == example.context.record() and opened == tuple(example.opened_sources),
             'target example differs from independently frozen caller')
    actual_binding = source.processor_binding(processor)
    _require(actual_binding == processor_binding, 'external target processor binding differs')
    verified = source.verify_source_alignment(processor, caller, source_alignment,
        caller_sha256=caller_sha256, alignment_sha256=source_alignment_sha256,
        processor_binding=processor_binding, max_source_tokens=max_source_tokens)
    record = learning.supervised_output_record_v16(example)
    geometry = canonical_field_geometry(record, expected_target_sha256=target_sha256)
    text = bytes.fromhex(geometry['canonical_utf8_hex']).decode('utf-8')
    try:
        with source._processor_execution_settings(processor):
            inputs, labels = v16._target_ids(processor.tokenizer, example, max_target_tokens)
            observed = processor.tokenizer(text, add_special_tokens=False, padding=False, truncation=False,
                return_offsets_mapping=True, return_attention_mask=True, return_special_tokens_mask=True)
            rows = _token_rows(geometry, observed['input_ids'], observed['attention_mask'],
                               observed['offset_mapping'], observed['special_tokens_mask'])
            _require(observed['input_ids'] == labels[:-1] and inputs[1:] == labels[:-1],
                     'actual complete target IDs differ from unchanged v16 producer')
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, CognitiveKernelContractError):
            raise
        raise CognitiveKernelContractError('malformed complete target observation') from exc
    finally:
        _require(source.processor_binding(processor) == actual_binding, 'processor binding changed during target observation')
    n = len(rows)
    sidecar = _seal({'schema': SCHEMA, 'profile': PROFILE, 'caller_sha256': caller_sha256,
        'context_sha256': context.content_digest(), 'source_alignment_sha256': source_alignment_sha256,
        'candidate_inventory_sha256': verified['candidate_inventory_sha256'],
        'processor_binding': actual_binding, 'max_source_tokens': max_source_tokens,
        'max_target_tokens': max_target_tokens, 'target_sha256': target_sha256, 'geometry': geometry,
        'decoder_inputs': inputs, 'labels': labels, 'tokens': rows,
        'scaffolding': {'bos': {'decoder_input_index': 0, 'token_id': inputs[0]},
                        'eos': {'label_index': n, 'query_index': n, 'token_id': labels[-1]}},
        'target_code_binding': target_code_binding(), 'learning_qualified': False, 'authority_granted': False})
    return deepcopy(inputs), deepcopy(labels), sidecar


def validate_target_alignment_record(sidecar, *, target_alignment_sha256):
    """Validate pinned metadata and recompute geometry; actual reload is separate."""
    try:
        _pin(sidecar, target_alignment_sha256, SCHEMA)
        expected_keys = {'schema', 'profile', 'caller_sha256', 'context_sha256', 'source_alignment_sha256',
            'candidate_inventory_sha256', 'processor_binding', 'max_source_tokens', 'max_target_tokens',
            'target_sha256', 'geometry', 'decoder_inputs', 'labels', 'tokens', 'scaffolding',
            'target_code_binding', 'learning_qualified', 'authority_granted', 'record_sha256'}
        _require(set(sidecar) == expected_keys and sidecar['profile'] == PROFILE and
                 sidecar['target_code_binding'] == target_code_binding() and
                 sidecar['learning_qualified'] is False and sidecar['authority_granted'] is False,
                 'unsupported target profile or code binding')
        for key in ('caller_sha256', 'context_sha256', 'source_alignment_sha256', 'candidate_inventory_sha256', 'target_sha256'):
            canonical.require_sha256(sidecar[key], key)
        _require(all(type(sidecar[k]) is int and sidecar[k] > 0 for k in ('max_source_tokens', 'max_target_tokens')),
                 'invalid target budgets')
        geometry = sidecar['geometry']; raw = bytes.fromhex(geometry['canonical_utf8_hex'])
        record = json.loads(raw.decode('utf-8'))
        _require(canonical_json_bytes(record) == raw and
                 canonical_field_geometry(record, expected_target_sha256=sidecar['target_sha256']) == geometry,
                 'target geometry differs from complete canonical record')
        rows = sidecar['tokens']
        _require(type(rows) is list and type(sidecar['decoder_inputs']) is list and type(sidecar['labels']) is list,
                 'target sequences must be arrays')
        inputs, labels = sidecar['decoder_inputs'], sidecar['labels']
        _require(len(inputs) == len(labels) == len(rows) + 1 <= sidecar['max_target_tokens'] and
                 all(type(v) is int and v >= 0 for v in inputs + labels), 'invalid complete decoder sequence')
        recomputed = _token_rows(geometry, [r['token_id'] for r in rows], [r['attention_mask'] for r in rows],
                                [r['canonical_chars'] for r in rows], [r['special_mask'] for r in rows])
        _require(rows == recomputed and inputs[1:] == labels[:-1] == [r['token_id'] for r in rows],
                 'target token positions or complete shift differ')
        n = len(rows)
        _require(sidecar['scaffolding'] == {'bos': {'decoder_input_index': 0, 'token_id': inputs[0]},
            'eos': {'label_index': n, 'query_index': n, 'token_id': labels[-1]}} and
            inputs[0] == sidecar['processor_binding']['tokenizer_wrapper']['bos_token_id'] and
            labels[-1] == sidecar['processor_binding']['tokenizer_wrapper']['eos_token_id'], 'target scaffolding differs')
        return deepcopy(record)
    except (KeyError, TypeError, ValueError, UnicodeError, IndexError) as exc:
        if isinstance(exc, CognitiveKernelContractError):
            raise
        raise CognitiveKernelContractError('malformed target alignment record') from exc


def verify_target_alignment(processor, caller, example, sidecar, *, target_alignment_sha256, **kwargs):
    validate_target_alignment_record(sidecar, target_alignment_sha256=target_alignment_sha256)
    _, _, actual = target_batch_with_alignment(processor, caller, example, **kwargs)
    _require(actual == sidecar, 'actual complete target alignment differs')
    return actual
