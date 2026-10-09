"""Caller-bound original text geometry beside the unchanged formation processor.

This target-free sidecar is a construction interface, not a learned pointer or
an authority grant. Native locator geometry and full target-field tokenization
remain separate responsibilities. Original text is never normalized to make a
copy label fit. External digests bind custody; they do not authenticate callers.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_json_bytes, canonical_sha256, require_sha256,
)
from cognitive_kernel import formation_multimodal as media
from .formation_caller_input import caller_input_from_record, selection_candidates
from . import train_v1_formation_specialist as shared
from . import train_v16_formation_specialist as v16

SCHEMA = 'mfm-text-source-alignment-v1'
LITERAL_SCHEMA = 'mfm-original-literal-alignment-v1'
_TEXT = frozenset({'text', 'code', 'structured'})


def _require(value, message):
    if not value:
        raise CognitiveKernelContractError(message)


def _seal(core):
    return {**core, 'record_sha256': canonical_sha256(core)}


def _pinned_record(record, expected, schema):
    _require(type(record) is dict and record.get('schema') == schema,
             'unsupported alignment record')
    require_sha256(expected, 'external record digest')
    _require(record.get('record_sha256') == expected and expected == canonical_sha256(
        {k: v for k, v in record.items() if k != 'record_sha256'}),
        'externally bound record differs')


def _caller(record, expected):
    require_sha256(expected, 'caller digest')
    _require(record.get('record_sha256') == expected, 'externally bound caller differs')
    return caller_input_from_record(record)


def processor_binding(processor):
    """Fingerprint the actual tokenizer configuration and responsible source.

    Production consumers supply a previously frozen external binding. Computing
    this value at consumption time and trusting it alone provides no custody.
    """
    tokenizer = processor.tokenizer
    _require(getattr(tokenizer, 'is_fast', False) is True and
             hasattr(tokenizer, 'backend_tokenizer'), 'actual fast tokenizer required')
    wrapper = {name: getattr(tokenizer, name, None) for name in (
        'bos_token_id', 'eos_token_id', 'unk_token_id', 'pad_token_id',
        'all_special_ids', 'padding_side', 'truncation_side', 'model_max_length',
        'clean_up_tokenization_spaces', 'add_bos_token', 'add_eos_token',
        'split_special_tokens', 'legacy')}
    return {'schema': 'mfm-source-alignment-processor-binding-v1',
        'processor_class': type(processor).__module__ + '.' + type(processor).__qualname__,
        'tokenizer_class': type(tokenizer).__module__ + '.' + type(tokenizer).__qualname__,
        'tokenizer_backend_sha256': sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        'tokenizer_wrapper': wrapper,
        'transformers_version': version('transformers'), 'tokenizers_version': version('tokenizers'),
        'source_template_sha256': sha256(shared.SOURCE_TEMPLATE.encode()).hexdigest(),
        'source_v16_sha256': sha256(Path(v16.__file__).read_bytes()).hexdigest(),
        'source_renderer_sha256': sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        'media_builder_sha256': sha256(Path(media.__file__).read_bytes()).hexdigest(),
        'alignment_code_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}


@contextmanager
def _processor_execution_settings(processor):
    """Restore only the backend modes that Transformers changes per call.

    The full serialized backend stays in the external binding. No semantic
    tokenizer node is excluded, normalized or replaced to make reload pass.
    """
    backend = processor.tokenizer.backend_tokenizer
    padding, truncation = deepcopy(backend.padding), deepcopy(backend.truncation)
    try:
        yield
    finally:
        if padding is None:
            backend.no_padding()
        else:
            backend.enable_padding(**padding)
        if truncation is None:
            backend.no_truncation()
        else:
            backend.enable_truncation(**truncation)


def _character_map(text, rendered_start, *, ascii_source):
    atoms, pieces = [], []
    original_byte = 0
    rendered_position = rendered_start
    for char in text:
        encoded = json.dumps(char, ensure_ascii=ascii_source)[1:-1]
        if ascii_source:
            encoded = encoded.replace('<', '\\u003c')
        original_end = original_byte + len(char.encode('utf-8'))
        # Source offsets are Python character offsets; canonical output offsets
        # are UTF-8 bytes. Neither is silently substituted for original bytes.
        rendered_end = rendered_position + (len(encoded) if ascii_source else len(encoded.encode('utf-8')))
        atoms.append({'original_utf8': [original_byte, original_end],
                      'rendered_chars' if ascii_source else 'canonical_utf8':
                      [rendered_position, rendered_end]})
        pieces.append(encoded); original_byte = original_end; rendered_position = rendered_end
    expected = (shared._escaped_text(text) if ascii_source else
                canonical_json_bytes(text).decode('utf-8'))[1:-1]
    _require(''.join(pieces) == expected, 'character escape composition differs')
    return atoms, rendered_position


def _render(context, opened):
    _require(all(ref.modality in _TEXT for ref in context.base.evidence),
             'native evidence requires its own locator alignment adapter')
    with media.formation_context_media_messages(context.base, opened) as messages:
        _require(len(messages) == 1 and messages[0]['role'] == 'user', 'source message layout differs')
        content = messages[0]['content']
        _require(len(content) == 1 + 2 * len(opened) and
                 all(item['type'] == 'text' for item in content), 'source block layout differs')
        instruction = v16.SOURCE_INSTRUCTION + '\n' + canonical_json_bytes({
            'objective': v16.OBJECTIVE_VERSION_V16, 'context': context.record()}).decode('utf-8')
        blocks, sources, position = [], [], 0
        for index, item in enumerate(content):
            text = instruction if index == 0 else item['text']
            block = 'json_text=' + shared._escaped_text(text)
            if index >= 2 and index % 2 == 0:
                source_index = (index - 2) // 2
                ref = context.base.evidence[source_index]
                source_ref, raw = opened[source_index]
                _require(text.encode('utf-8') == raw and ref.ref_id == source_ref,
                         'original source block differs')
                start = position + len('json_text=') + 1
                characters, end = _character_map(text, start, ascii_source=True)
                sources.append({'ref_id': source_ref, 'modality': ref.modality,
                    'content_sha256': sha256(raw).hexdigest(), 'original_bytes': len(raw),
                    'rendered_content_chars': [start, end], 'characters': characters})
            blocks.append(block); position += len(block) + 1
        versioned = [{'role': 'user', 'content': [{'type': 'text', 'text': block} for block in blocks]}]
        return versioned, ''.join(block + '\n' for block in blocks), sources


def _tokens(rendered, sources, ids, masks, offsets, specials, special_ids):
    _require(len(ids) == len(masks) == len(offsets) == len(specials), 'offset/ID lengths differ')
    starts = [row['rendered_content_chars'][0] for row in sources]
    geometry = [(row, [a['rendered_chars'][0] for a in row['characters']],
                 [a['rendered_chars'][1] for a in row['characters']]) for row in sources]
    result = []
    for index, (token, mask, offset, special) in enumerate(zip(ids, masks, offsets, specials)):
        _require(type(token) is int and token >= 0 and type(mask) is int and mask == 1 and
                 type(special) is int and special in (0, 1), 'invalid actual token or mask')
        _require(type(offset) in (tuple, list) and len(offset) == 2 and
                 all(type(x) is int for x in offset), 'invalid actual character offsets')
        start, end = offset
        _require(0 <= start <= end <= len(rendered), 'actual offset exceeds rendered source')
        body = {'token_index': index, 'rendered_chars': [start, end], 'source_ref': None,
                'original_utf8': None, 'observed_source_eligible': False,
                'full_character_boundaries': False}
        source_index = bisect_right(starts, start) - 1
        if start < end and not special and token not in special_ids and source_index >= 0:
            row, atom_starts, atom_ends = geometry[source_index]
            if end <= row['rendered_content_chars'][1]:
                lo, hi = bisect_right(atom_ends, start), bisect_left(atom_starts, end)
                atoms = row['characters'][lo:hi]
                if atoms:
                    body.update(source_ref=row['ref_id'], observed_source_eligible=True,
                        original_utf8=[atoms[0]['original_utf8'][0], atoms[-1]['original_utf8'][1]],
                        full_character_boundaries=start == atoms[0]['rendered_chars'][0] and
                            end == atoms[-1]['rendered_chars'][1])
        result.append(body)
    return result


def text_source_batch_with_alignment(processor, caller, *, caller_sha256,
                                     processor_binding, max_source_tokens):
    """Return the actual unchanged processor batch and a target-free sidecar."""
    context, opened, registry = _caller(caller, caller_sha256)
    _require(type(max_source_tokens) is int and max_source_tokens > 0, 'full source budget required')
    actual_binding = globals()['processor_binding'](processor)
    _require(actual_binding == processor_binding, 'externally bound processor differs')
    messages, assembled, sources = _render(context, opened)
    rendered = processor.apply_chat_template(messages, chat_template=shared.SOURCE_TEMPLATE,
        tokenize=False, add_generation_prompt=False)
    _require(type(rendered) is str and rendered == assembled, 'actual source template rendering differs')
    with _processor_execution_settings(processor):
        batch = v16.source_batch_v16(processor, context, opened, max_source_tokens)
        ids, masks = batch['input_ids'][0].tolist(), batch['attention_mask'][0].tolist()
        # Observe offsets through the same processor path as the unchanged batch.
        # Its default special-token policy is part of processing; a standalone
        # tokenizer call cannot substitute for that policy or shift its positions.
        encoded = processor.apply_chat_template(messages, chat_template=shared.SOURCE_TEMPLATE,
            tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt',
            processor_kwargs={'do_sample_frames': False, 'return_offsets_mapping': True,
                              'return_special_tokens_mask': True, 'return_attention_mask': True})
        expected_shapes = {'input_ids': (1, len(ids)), 'attention_mask': (1, len(ids)),
            'offset_mapping': (1, len(ids), 2), 'special_tokens_mask': (1, len(ids))}
        _require(all(name in encoded and hasattr(encoded[name], 'tolist') and
                     tuple(encoded[name].shape) == shape for name, shape in expected_shapes.items()),
                 'actual processor omitted or malformed aligned IDs/masks/offsets/specials')
        _require(encoded['input_ids'][0].tolist() == ids and encoded['attention_mask'][0].tolist() == masks,
                 'actual processor IDs/masks differ from offset-bearing rendered tokenization')
    _require(globals()['processor_binding'](processor) == actual_binding,
             'processor binding changed during source observation')
    candidates = selection_candidates(context, opened, registry)
    return batch, _seal({'schema': SCHEMA, 'caller_sha256': caller_sha256,
        'context_sha256': context.content_digest(), 'processor_binding': actual_binding,
        'candidate_inventory_sha256': canonical_sha256(candidates),
        'rendered_text': rendered, 'rendered_utf8_sha256': sha256(rendered.encode()).hexdigest(),
        'input_ids': ids, 'attention_mask': masks, 'sources': sources,
        'tokens': _tokens(rendered, sources, ids, masks, encoded['offset_mapping'][0].tolist(),
                          encoded['special_tokens_mask'][0].tolist(), set(processor.tokenizer.all_special_ids)),
        'native_locator_alignment': False, 'learning_qualified': False,
        'authority_granted': False})


def verify_source_alignment(processor, caller, sidecar, *, caller_sha256,
                            alignment_sha256, processor_binding, max_source_tokens):
    """Reload against actual inputs/processor; resealing altered geometry fails."""
    _pinned_record(sidecar, alignment_sha256, SCHEMA)
    _, actual = text_source_batch_with_alignment(processor, caller, caller_sha256=caller_sha256,
        processor_binding=processor_binding, max_source_tokens=max_source_tokens)
    _require(sidecar == actual, 'actual original-source alignment differs')
    return actual


def literal_alignment(caller, sidecar, literal, *, caller_sha256, alignment_sha256):
    """Relate a later authored literal to all exact original occurrences.

    This consumes the externally frozen sidecar; initial admission/reload uses
    verify_source_alignment with the actual processor. It produces no full
    target-field token IDs or semantics. Source token support is contextual:
    selecting a partial escape does not authorize copying its token ID.
    """
    context, opened, registry = _caller(caller, caller_sha256)
    _pinned_record(sidecar, alignment_sha256, SCHEMA)
    _require(sidecar['caller_sha256'] == caller_sha256 and sidecar['context_sha256'] == context.content_digest()
        and sidecar['candidate_inventory_sha256'] == canonical_sha256(selection_candidates(context, opened, registry)),
        'literal consumer input ancestry differs')
    _, rendered, sources = _render(context, opened)
    _require(sidecar['rendered_text'] == rendered and sidecar['sources'] == sources,
             'literal original geometry differs')
    _require(type(literal) is str and bool(literal), 'nonempty authored literal required')
    needle = literal.encode('utf-8', 'strict')
    output_atoms, _ = _character_map(literal, 1, ascii_source=False)
    occurrences = []
    for row, (ref, raw) in zip(sources, opened):
        boundaries = {a['original_utf8'][0]: a['rendered_chars'][0] for a in row['characters']}
        boundaries[row['original_bytes']] = row['rendered_content_chars'][1]
        position = raw.find(needle)
        while position >= 0:
            end = position + len(needle)
            _require(position in boundaries and end in boundaries, 'literal is not at original UTF-8 boundaries')
            start_char, end_char = boundaries[position], boundaries[end]
            supporting = [token for token in sidecar['tokens'] if token['source_ref'] == ref and
                token['observed_source_eligible'] and token['rendered_chars'][0] < end_char and
                token['rendered_chars'][1] > start_char]
            covered = start_char
            for a, b in sorted(token['rendered_chars'] for token in supporting):
                if a > covered:
                    break
                covered = max(covered, b)
            occurrences.append({'source_ref': ref, 'content_sha256': row['content_sha256'],
                'original_utf8': [position, end], 'rendered_chars': [start_char, end_char],
                'supporting_token_indices': [token['token_index'] for token in supporting],
                'complete_eligible_token_coverage': covered >= end_char})
            position = raw.find(needle, position + 1)
    return _seal({'schema': LITERAL_SCHEMA, 'caller_sha256': caller_sha256,
        'alignment_sha256': alignment_sha256, 'literal_utf8_sha256': sha256(needle).hexdigest(),
        'canonical_literal_utf8_hex': canonical_json_bytes(literal).hex(),
        'canonical_literal_characters': output_atoms, 'occurrences': occurrences,
        'semantic_paraphrase_alignment': False, 'native_locator_alignment': False,
        'learning_qualified': False, 'authority_granted': False})
