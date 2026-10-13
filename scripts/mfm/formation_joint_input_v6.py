"""Complete text-native source geometry for a proposed full-source J6 profile.

This is source-only construction. It neither changes the original J5 budget nor
claims Gemma forward capacity, material rights, target access or learned fit.
"""
from copy import deepcopy
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import torch

from cognitive_kernel.canonical import canonical_sha256, require_sha256

SCHEMA = 'mfm-joint-caller-input-alignment-v6'
FROZEN = 'mfm-full-source-text-construction-freeze-v6'
TASK = 'formation-complete-native-text-source-v6'
PROFILE = 'text-causal-complete-semantic-canonical-parts-native-v1'
NATIVE_FIELDS = ('input_ids', 'attention_mask', 'mm_token_type_ids')
OFFSET_FIELDS = (*NATIVE_FIELDS, 'offset_mapping', 'special_tokens_mask')


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _seal(row):
    return {**deepcopy(row), 'record_sha256': canonical_sha256(row)}


def _pinned_j5(j5, expected_file_sha256):
    require_sha256(expected_file_sha256, 'original J5 source FILE')
    _need(j5.__name__ == 'scripts.mfm.formation_joint_input_v5' and
          sha256(Path(j5.__file__).read_bytes()).hexdigest() == expected_file_sha256,
          'exact original J5 source FILE differs')
    return j5


def code_binding(j5, *, j5_file_sha256):
    _pinned_j5(j5, j5_file_sha256)
    return _seal({'schema': 'mfm-joint-input-code-binding-v6', 'task': TASK,
        'profile': PROFILE, 'v6_source_file_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_j5_file_sha256': j5_file_sha256,
        'original_j5_code_binding': j5.joint_code_binding_v5(),
        'source_only': True, 'capacity_admitted': False,
        'authority_granted': False})


def validate_native_batch_pair(default, offset, *, proposed_max_source_tokens):
    """Check the exact observed text-native return, including its third field."""
    _need(type(proposed_max_source_tokens) is int and proposed_max_source_tokens > 0,
          'explicit proposed complete source budget required')
    _need(isinstance(default, Mapping) and isinstance(offset, Mapping) and
          set(default) == set(NATIVE_FIELDS) and set(offset) == set(OFFSET_FIELDS),
          'exact native text processor field roster differs')
    ids = default['input_ids']
    _need(type(ids) is torch.Tensor and ids.ndim == 2 and ids.shape[0] == 1 and
          ids.device.type == 'cpu' and ids.dtype == torch.int64,
          'complete native input IDs shape/dtype differs')
    length = ids.shape[1]
    _need(0 < length <= proposed_max_source_tokens, 'complete source budget overflow')
    for key in NATIVE_FIELDS:
        a, b = default[key], offset[key]
        _need(type(a) is torch.Tensor and type(b) is torch.Tensor and
              a.dtype == b.dtype == torch.int64 and a.device.type == b.device.type == 'cpu' and
              tuple(a.shape) == tuple(b.shape) == (1, length),
              'complete native field shape/dtype differs')
        _need(torch.equal(a, b), 'default and offset native fields differ')
    _need(bool(torch.all(default['attention_mask'] == 1)),
          'complete native attention mask differs')
    _need(bool(torch.all(default['input_ids'] >= 0)) and
          bool(torch.all(default['mm_token_type_ids'] == 0)),
          'complete native text field values differ')
    _need(type(offset['offset_mapping']) is torch.Tensor and
          offset['offset_mapping'].dtype == torch.int64 and
          offset['offset_mapping'].device.type == 'cpu' and
          tuple(offset['offset_mapping'].shape) == (1, length, 2) and
          type(offset['special_tokens_mask']) is torch.Tensor and
          offset['special_tokens_mask'].dtype == torch.int64 and
          offset['special_tokens_mask'].device.type == 'cpu' and
          tuple(offset['special_tokens_mask'].shape) == (1, length),
          'complete offset geometry shape/dtype differs')
    return {'source_tokens': length,
            'native_fields': {key: default[key][0].tolist() for key in NATIVE_FIELDS},
            'offset_mapping': offset['offset_mapping'][0].tolist(),
            'special_tokens_mask': offset['special_tokens_mask'][0].tolist()}


def _validate_saved_native_fields(fields, batch, *, source_tokens,
                                  proposed_max_source_tokens):
    """Check the complete serialized text-native input and its exact join."""
    _need(type(proposed_max_source_tokens) is int and
          proposed_max_source_tokens > 0 and type(source_tokens) is int and
          0 < source_tokens <= proposed_max_source_tokens,
          'saved complete J6 source budget differs')
    _need(type(fields) is dict and set(fields) == set(NATIVE_FIELDS) and
          type(batch) is dict and set(batch) == set(NATIVE_FIELDS),
          'saved complete J6 native field roster differs')
    for key in NATIVE_FIELDS:
        row_values = fields[key]
        batch_values = batch[key]
        _need(type(row_values) is list and len(row_values) == source_tokens and
              type(batch_values) is list and len(batch_values) == 1 and
              type(batch_values[0]) is list and
              len(batch_values[0]) == source_tokens and
              all(type(value) is int for value in row_values) and
              all(type(value) is int for value in batch_values[0]),
              'saved complete J6 native field type/shape differs')
        if key == 'input_ids':
            valid = all(value >= 0 for value in row_values)
        elif key == 'attention_mask':
            valid = all(value == 1 for value in row_values)
        else:
            valid = all(value == 0 for value in row_values)
        _need(valid and canonical_sha256(row_values) ==
              canonical_sha256(batch_values[0]),
              'saved complete J6 native field values differ')
    return source_tokens


def _alignment(j5, caller, meaning, rendered, blocks, fields, originals, parts,
               native, tokens, candidates, *, caller_sha256, meaning_sha256,
               render_policy, processor_binding, source_identity,
               proposed_max_source_tokens, j5_file_sha256):
    codes = code_binding(j5, j5_file_sha256=j5_file_sha256)
    ids = native['native_fields']['input_ids']
    mask = native['native_fields']['attention_mask']
    mm = native['native_fields']['mm_token_type_ids']
    rp = render_policy['record_sha256']
    candidate_sha = canonical_sha256(meaning['candidate_inventory'])
    text_sha = sha256(rendered.encode('utf-8')).hexdigest()
    view = canonical_sha256({'task': TASK, 'profile': PROFILE, 'codes': codes,
        'render_policy_sha256': rp, 'source_identity': source_identity,
        'processor_binding': processor_binding, 'rendered_utf8_sha256': text_sha,
        'native_fields': native['native_fields'],
        'proposed_max_source_tokens': proposed_max_source_tokens})
    provenance = canonical_sha256({'caller_sha256': caller_sha256,
        'meaning_sha256': meaning_sha256, 'render_policy_sha256': rp,
        'code_binding_sha256': codes['record_sha256'],
        'candidate_inventory_sha256': candidate_sha,
        'claim_snapshot_sha256': meaning['claim_snapshot']['record_sha256'],
        'episode_snapshot_sha256': meaning['episode_snapshot']['record_sha256'],
        'task_snapshot_sha256': meaning['task_snapshot']['record_sha256'],
        'source_identity_sha256': source_identity['record_sha256'],
        'original_part_inventory_sha256': canonical_sha256(parts)})
    return _seal({'schema': SCHEMA, 'task': TASK, 'profile': PROFILE,
        'code_binding': codes, 'render_policy': render_policy,
        'caller_sha256': caller_sha256, 'meaning_sha256': meaning_sha256,
        'processor_binding': processor_binding, 'source_identity': source_identity,
        'proposed_max_source_tokens': proposed_max_source_tokens,
        'candidate_inventory_sha256': candidate_sha,
        'rendered_text': rendered, 'rendered_utf8_sha256': text_sha,
        'source_tokens': native['source_tokens'],
        'input_ids': ids, 'attention_mask': mask, 'mm_token_type_ids': mm,
        'blocks': blocks, 'fields': fields, 'sources': originals,
        'prior_original_parts': parts, 'candidates': candidates, 'tokens': tokens,
        'cache': {'processed_view_sha256': view, 'provenance_sha256': provenance},
        'text_attention': 'causal_full_or_sliding_as_configured',
        'teacher_accessed': False, 'native_locator_alignment': False,
        'construction_only': True, 'capacity_admitted': False,
        'learning_qualified': False, 'authority_granted': False})


def joint_source_batch_with_alignment_v6(j5, processor, caller, meaning, *,
        caller_sha256, meaning_sha256, render_policy, processor_binding,
        source_identity, proposed_max_source_tokens, j5_file_sha256):
    """One default plus one offset processor pass over the entire source."""
    _pinned_j5(j5, j5_file_sha256)
    j5._policy(render_policy, render_policy['record_sha256'])
    j5.meaning_api.caller_meaning_v3_from_record(meaning, caller,
        meaning_sha256=meaning_sha256, caller_sha256=caller_sha256)
    j5._identity(source_identity, source_identity['record_sha256'])
    _need(canonical_sha256(j5.source.processor_binding(processor)) ==
          canonical_sha256(processor_binding), 'exact source processor binding differs')
    messages, assembled, blocks, fields, originals, parts = j5._render(
        caller, meaning, caller_sha256, meaning_sha256, render_policy,
        render_policy['record_sha256'])
    rendered = processor.apply_chat_template(messages,
        chat_template=j5.shared.SOURCE_TEMPLATE, tokenize=False,
        add_generation_prompt=False)
    _need(type(rendered) is str and rendered == assembled,
          'actual complete semantic template differs')
    with j5.source._processor_execution_settings(processor):
        default = processor.apply_chat_template(messages,
            chat_template=j5.shared.SOURCE_TEMPLATE, tokenize=True,
            add_generation_prompt=False, return_dict=True, return_tensors='pt',
            processor_kwargs={'do_sample_frames': False})
        offset = processor.apply_chat_template(messages,
            chat_template=j5.shared.SOURCE_TEMPLATE, tokenize=True,
            add_generation_prompt=False, return_dict=True, return_tensors='pt',
            processor_kwargs={'do_sample_frames': False,
                'return_offsets_mapping': True, 'return_special_tokens_mask': True,
                'return_attention_mask': True})
        native = validate_native_batch_pair(default, offset,
            proposed_max_source_tokens=proposed_max_source_tokens)
    _need(canonical_sha256(j5.source.processor_binding(processor)) ==
          canonical_sha256(processor_binding), 'processor changed during complete source pass')
    tokens, candidates = j5._joins(rendered, originals,
        native['native_fields']['input_ids'], native['native_fields']['attention_mask'],
        native['offset_mapping'], native['special_tokens_mask'],
        set(processor.tokenizer.all_special_ids), fields, blocks,
        meaning['candidate_inventory'])
    row = _alignment(j5, caller, meaning, rendered, blocks, fields, originals,
        parts, native, tokens, candidates, caller_sha256=caller_sha256,
        meaning_sha256=meaning_sha256, render_policy=render_policy,
        processor_binding=processor_binding, source_identity=source_identity,
        proposed_max_source_tokens=proposed_max_source_tokens,
        j5_file_sha256=j5_file_sha256)
    return default, row


def validate_saved_alignment(j5, caller, meaning, row, batch, *,
                             j5_file_sha256, proposed_max_source_tokens):
    """Rebuild saved source geometry without a processor or target."""
    _pinned_j5(j5, j5_file_sha256)
    _need(type(row) is dict and row.get('schema') == SCHEMA and
          row.get('record_sha256') == canonical_sha256({k: v for k, v in row.items()
              if k != 'record_sha256'}), 'saved J6 alignment seal differs')
    _validate_saved_native_fields({key: row[key] for key in NATIVE_FIELDS},
        batch, source_tokens=row['source_tokens'],
        proposed_max_source_tokens=proposed_max_source_tokens)
    _need(canonical_sha256(row['code_binding']) ==
          canonical_sha256(code_binding(j5, j5_file_sha256=j5_file_sha256)) and
          type(row['proposed_max_source_tokens']) is int and
          row['proposed_max_source_tokens'] == proposed_max_source_tokens,
          'saved J6 code/budget differs')
    j5._policy(row['render_policy'], row['render_policy']['record_sha256'])
    j5._identity(row['source_identity'], row['source_identity']['record_sha256'])
    messages, rendered, blocks, fields, originals, parts = j5._render(
        caller, meaning, row['caller_sha256'], row['meaning_sha256'],
        row['render_policy'], row['render_policy']['record_sha256'])
    _need(row['rendered_text'] == rendered, 'saved complete rendered text differs')
    _need(len(row['tokens']) == row['source_tokens'] and
          len(row['mm_token_type_ids']) == row['source_tokens'],
          'saved J6 complete native length differs')
    offsets = [token['rendered_chars'] for token in row['tokens']]
    specials = [token['special_mask'] for token in row['tokens']]
    tokens, candidates = j5._joins(rendered, originals, row['input_ids'],
        row['attention_mask'], offsets, specials,
        set(row['processor_binding']['tokenizer_wrapper']['all_special_ids']),
        fields, blocks, meaning['candidate_inventory'])
    native = {'source_tokens': len(row['input_ids']),
        'native_fields': {key: row[key] for key in NATIVE_FIELDS}}
    rebuilt = _alignment(j5, caller, meaning, rendered, blocks, fields,
        originals, parts, native, tokens, candidates,
        caller_sha256=row['caller_sha256'], meaning_sha256=row['meaning_sha256'],
        render_policy=row['render_policy'],
        processor_binding=row['processor_binding'],
        source_identity=row['source_identity'],
        proposed_max_source_tokens=proposed_max_source_tokens,
        j5_file_sha256=j5_file_sha256)
    _need(canonical_sha256(row) == canonical_sha256(rebuilt),
          'saved complete J6 field/token/cache geometry differs')
    return row


def freeze_source(j5, caller, meaning, alignment, batch, *,
                  j5_file_sha256, proposed_max_source_tokens, cut_id,
                  expected_cut_sha256):
    """Source-only complete input for later capacity measurement, no admission."""
    validate_saved_alignment(j5, caller, meaning, alignment, batch,
        j5_file_sha256=j5_file_sha256,
        proposed_max_source_tokens=proposed_max_source_tokens)
    episode = meaning['episode_snapshot']['historical_prefix_proof']
    task = meaning['task_snapshot']['historical_prefix_proof']
    require_sha256(expected_cut_sha256, 'external original cut proof')
    _need(type(cut_id) is str and bool(cut_id) and
          episode['world'] == task['world'] and episode['cut'] == task['cut'] and
          episode['cut']['record_sha256'] == expected_cut_sha256,
          'original world/cut proof differs')
    return _seal({'schema': FROZEN, 'cut_id': cut_id,
        'world_sha256': episode['world']['world_sha256'],
        'cut_sha256': episode['cut']['record_sha256'],
        'caller': caller, 'meaning': meaning, 'alignment': alignment,
        'native_batch': batch,
        'proposed_max_source_tokens': proposed_max_source_tokens,
        'construction_only': True, 'capacity_admitted': False,
        'target_accessed': False, 'training_authorized': False,
        'current_service_authenticated': False})


def validate_frozen_source(j5, row, *, j5_file_sha256,
                           proposed_max_source_tokens):
    _need(type(row) is dict and row.get('schema') == FROZEN and
          row.get('record_sha256') == canonical_sha256({k: v for k, v in row.items()
              if k != 'record_sha256'}), 'saved J6 source freeze seal differs')
    rebuilt = freeze_source(j5, row['caller'], row['meaning'],
        row['alignment'], row['native_batch'], j5_file_sha256=j5_file_sha256,
        proposed_max_source_tokens=proposed_max_source_tokens,
        cut_id=row['cut_id'], expected_cut_sha256=row['cut_sha256'])
    _need(canonical_sha256(row) == canonical_sha256(rebuilt),
          'saved J6 source freeze differs')
    return row
