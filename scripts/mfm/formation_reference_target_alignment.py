"""Complete reference-decision lowering and actual whole-record query geometry.

Original complete pairs remain immutable. This is construction metadata, not
corpus admission, authenticated authorship, or learned selection evidence.
"""
from copy import deepcopy
from hashlib import sha256

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel import formation_decision_v16 as local, formation_learning_v16 as learning
from cognitive_kernel import formation_reference_decision as references
from . import formation_joint_target_alignment as original, formation_target_alignment as geometry_api
from . import formation_source_alignment as source, formation_paired_teaching as paired
from . import formation_joint_target_alignment_v6 as j6_target

VIEW_SCHEMA = 'mfm-formation-reference-target-view-v1'
ALIGNMENT_SCHEMA = 'mfm-formation-reference-decision-token-alignment-v1'
PROFILES = {
    'mfm-paired-supplied-input-v1': ('mfm-joint-caller-input-alignment-v4', 'mfm-paired-supplied-joint-target-geometry-v1'),
    'mfm-paired-supplied-j5-input-v1': ('mfm-joint-caller-input-alignment-v5', 'mfm-paired-supplied-j5-joint-target-geometry-v1'),
    j6_target.paired.INPUT: ('mfm-joint-caller-input-alignment-v6', j6_target.PAIR),
}


def _seal(row):
    return {**deepcopy(row), 'record_sha256': canonical_sha256(row)}


def source_parts(frozen_input):
    """Select actual source locations; full original/material admission is upstream."""
    if frozen_input.get('schema') != j6_target.paired.INPUT:
        return frozen_input['caller'], frozen_input['joint']
    need = references._require
    saved = frozen_input['source']
    references._pinned(saved, saved['record_sha256'])
    need(saved.get('schema') == 'mfm-full-source-text-construction-freeze-v6' and
         sha256(canonical_json_bytes(saved) + b'\n').hexdigest() == frozen_input['source_file_sha256'],
         'original nested J6 source FILE differs')
    caller, joint = saved['caller'], saved['alignment']
    references._pinned(caller, caller['record_sha256'])
    references._pinned(joint, joint['record_sha256'])
    need(joint.get('schema') == 'mfm-joint-caller-input-alignment-v6' and
         canonical_sha256(frozen_input['source_cache']) == canonical_sha256(joint['cache']) and
         canonical_sha256(frozen_input['processor_binding']) == canonical_sha256(joint['processor_binding']) and
         canonical_sha256(frozen_input['native_batch']) == canonical_sha256(saved['native_batch']) and
         canonical_sha256(saved['native_batch']) == canonical_sha256({key: [joint[key]]
             for key in ('input_ids', 'attention_mask', 'mm_token_type_ids')}),
         'original nested J6 native/cache/processor join differs')
    return caller, joint


def j6_choices(complete_pair, frozen_input):
    """Bind the exact original authored choice bytes before metadata lowering."""
    need = references._require
    raw = bytes.fromhex(complete_pair['original_choice_hex'])
    choice = j6_target.paired.pinned(raw, complete_pair['choice_file_sha256'], j6_target.CHOICES)
    need(canonical_sha256(choice) == canonical_sha256(complete_pair['choice_packet']) and
         choice['input_freeze_sha256'] == frozen_input['record_sha256'] and
         choice['target_record_sha256'] == complete_pair['target_record_sha256'] and
         choice['author_provenance_file_sha256'] == complete_pair['author_provenance_file_sha256'] and
         choice['whole_literal_inventory_complete'] is True and
         complete_pair['source_file_sha256'] == frozen_input['source_file_sha256'] and
         canonical_sha256(complete_pair['source_cache']) == canonical_sha256(frozen_input['source_cache']) and
         canonical_sha256(complete_pair['native_batch']) == canonical_sha256(frozen_input['native_batch']),
         'original J6 choice FILE/source/ancestry join differs')
    return choice


def reference_decision_view(example, complete_pair, frozen_input, *, pair_sha256, frozen_input_sha256,
                            inventory, local_uncertainty_refs=()):
    """Consume explicitly supplied complete pairs without inferring any ordinal.

    The original profile's whole-input/admission reader remains required; these
    pinned metadata and semantic joins do not admit a private J5 corpus.
    """
    need = references._require
    references._pinned(frozen_input, frozen_input_sha256); references._pinned(complete_pair, pair_sha256)
    references._inventory(inventory)
    profile = PROFILES.get(frozen_input.get('schema'))
    need(profile is not None, 'original frozen-input profile differs')
    caller, joint = source_parts(frozen_input)
    need(joint['schema'] == profile[0] == inventory['source_profile'] and
         complete_pair.get('schema') == profile[1], 'original frozen-input/pair/joint profile differs')
    need(complete_pair['input_freeze_sha256'] == frozen_input_sha256 and
         caller['record_sha256'] == inventory['binding']['caller_sha256'] and
         joint['candidate_inventory_sha256'] == inventory['binding']['candidate_inventory_sha256'] and
         joint['cache'] == {k: inventory['binding'][k] for k in ('processed_view_sha256', 'provenance_sha256')} and
         complete_pair['processor_binding'] == joint['processor_binding'] == inventory['processor_binding'],
         'original pair caller/processed-view/provenance/processor binding differs')
    caller_context, opened, _ = references.caller_api.caller_input_from_record(caller)
    need(example.context.record() == caller_context.record() and example.opened_sources == opened,
         'complete target example differs from frozen caller')
    record = learning.supervised_output_record_v16(example)
    raw = canonical_json_bytes(record); target_sha = sha256(raw).hexdigest()
    geometry = geometry_api.canonical_field_geometry(record, expected_target_sha256=target_sha)
    need(complete_pair['canonical_output_sha256'] == target_sha and complete_pair['geometry'] == geometry,
         'complete original target geometry differs')
    rows = complete_pair['tokens']
    rebuilt = geometry_api._token_rows(geometry, [r['token_id'] for r in rows], [r['attention_mask'] for r in rows],
        [r['canonical_chars'] for r in rows], [r['special_mask'] for r in rows])
    inputs, labels = complete_pair['decoder_inputs'], complete_pair['labels']
    wrapper = inventory['processor_binding']['tokenizer_wrapper']
    need(rows == rebuilt and inputs[1:] == labels[:-1] == [r['token_id'] for r in rows] and
         len(inputs) == len(labels) == len(rows) + 1 and inputs[0] == wrapper['bos_token_id'] and
         labels[-1] == wrapper['eos_token_id'], 'complete original target shift differs')
    if frozen_input['schema'] == j6_target.paired.INPUT:
        choice = j6_choices(complete_pair, frozen_input)
        normative = j6_target.field_rows(geometry, frozen_input, choice['field_annotations'])
    else:
        normative = original.field_rows(geometry, frozen_input, complete_pair['annotations'])
    need(complete_pair['fields'] == normative, 'complete original field/explicit ordinal annotations differ')
    decision = local.decision_record(example, local_uncertainty_refs=local_uncertainty_refs)
    decision['schema'] = references.SCHEMA
    authored = []
    for row in normative:
        if row['responsibility'] != 'external_selection':
            continue
        value = references._get(decision, row['pointer'])
        if value is None or type(value) is dict:
            continue  # Only explicit local namespaces were introduced by the local compiler.
        need(row['annotation_status'] == 'authored' and type(row.get('selected_candidate_ordinal')) is int,
             'external reference requires an explicit authored ordinal annotation')
        ordinal = row['selected_candidate_ordinal']
        need(ordinal in row['matching_candidate_indices'] and
             inventory['candidate_inventory'][ordinal]['ref_id'] == value,
             'authored reference does not match original caller identity')
        references._set(decision, row['pointer'], {'caller_reference': True})
        authored.append({'pointer': row['pointer'], 'candidate_ordinal': ordinal})
    leaves = references.reference_requests(decision, inventory)
    need({r['pointer'] for r in leaves} == {r['pointer'] for r in authored}, 'complete reference lowering differs')
    return _seal({'schema': VIEW_SCHEMA, 'decision': decision, 'binding': deepcopy(inventory['binding']),
        'source_profile': {'input_schema': frozen_input['schema'], 'joint_schema': profile[0], 'pair_schema': profile[1]},
        'processor_binding': deepcopy(inventory['processor_binding']), 'original_pair_sha256': pair_sha256,
        'original_frozen_input_sha256': frozen_input_sha256, 'original_target_sha256': target_sha,
        'reference_leaves': leaves, 'authored_ordinals': sorted(authored, key=lambda r: r['pointer']),
        'training_authorized': False, 'learning_qualified': False, 'authority_granted': False})


def _align(processor, decision, leaves, binding, *, max_target_tokens, mode, generated_ids=None):
    need = references._require
    need(type(max_target_tokens) is int and max_target_tokens > 0, 'complete reference target budget is invalid')
    raw = canonical_json_bytes(decision); text = raw.decode('utf-8')
    geometry = geometry_api.canonical_field_geometry(decision, expected_target_sha256=sha256(raw).hexdigest())
    with source._processor_execution_settings(processor):
        observed = processor.tokenizer(text, add_special_tokens=False, padding=False, truncation=False,
            return_offsets_mapping=True, return_attention_mask=True, return_special_tokens_mask=True)
    ids = observed['input_ids']
    need(type(ids) is list and len(ids) + 1 <= max_target_tokens, 'complete reference target budget overflow')
    if mode == 'generated':
        need(type(generated_ids) is list and all(type(i) is int for i in generated_ids) and ids == generated_ids,
             'actual generated token IDs differ from whole decoded-wire identity')
        need(processor.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text,
             'actual generated text differs from complete canonical decision')
    tokens = geometry_api._token_rows(geometry, ids, observed['attention_mask'], observed['offset_mapping'], observed['special_tokens_mask'])
    nodes = {row['pointer']: row for row in geometry['nodes']}
    requests = []
    for leaf in leaves:
        node = nodes[leaf['pointer']]; a, b = node['canonical_chars']
        overlaps = [row['body_index'] for row in tokens if not row['special_mask'] and
                    row['canonical_chars'][0] < b and row['canonical_chars'][1] > a]
        need(bool(overlaps), 'reference operand has no actual whole-record token query')
        j = min(overlaps)
        requests.append({**deepcopy(leaf), 'query_index': j, 'decoder_input_index': j + 1})
    bos, eos = processor.tokenizer.bos_token_id, processor.tokenizer.eos_token_id
    need(type(bos) is int and type(eos) is int, 'reference tokenizer lacks explicit BOS/EOS')
    result = {'schema': ALIGNMENT_SCHEMA, 'mode': mode, 'binding': deepcopy(binding), 'geometry': geometry,
        'body_token_ids': list(ids), 'decoder_inputs': [bos, *ids], 'tokens': tokens, 'requests': requests,
        'eos_query_index': len(ids), 'query_conditioning': 'completed-decision-conditioned',
        'learning_qualified': False, 'authority_granted': False}
    if mode == 'teacher_forced':
        result['labels'] = [*ids, eos]
    else:
        result['generated_ids'] = list(generated_ids)
    return _seal(result)


def align_reference_decision_target(processor, view, *, view_sha256, max_target_tokens):
    references._pinned(view, view_sha256)
    references._require(view.get('schema') == VIEW_SCHEMA and source.processor_binding(processor) == view['processor_binding'],
                        'reference target processor/view binding differs')
    result = _align(processor, view['decision'], view['reference_leaves'], view['binding'],
                    max_target_tokens=max_target_tokens, mode='teacher_forced')
    references._require(source.processor_binding(processor) == view['processor_binding'], 'reference processor changed')
    return result


def align_generated_reference_decision(processor, decision, generated_ids, inventory, *, max_target_tokens, eos_observed):
    references._inventory(inventory)
    references._require(eos_observed is True and source.processor_binding(processor) == inventory['processor_binding'],
                        'generated reference decision requires complete EOS and pinned processor')
    leaves = references.reference_requests(decision, inventory)
    result = _align(processor, decision, leaves, inventory['binding'], max_target_tokens=max_target_tokens,
                    mode='generated', generated_ids=generated_ids)
    references._require(source.processor_binding(processor) == inventory['processor_binding'], 'reference processor changed')
    return result
