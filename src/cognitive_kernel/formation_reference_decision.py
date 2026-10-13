"""Typed caller-reference lowering into the complete original v1.6 consumer.

Caller snapshots and masks establish mechanical eligibility, not truth, current
rights or memory-write authority. This module neither repairs identity strings
nor changes semantic values. Claim must still reread live authority.
"""
from copy import deepcopy

from .canonical import CognitiveKernelContractError, canonical_sha256, require_sha256
from . import formation_decision_v16 as local
from .formation_semantics_v16 import validate_formation_grounding_v16
from scripts.mfm import formation_caller_input as caller_api
from scripts.mfm import formation_field_responsibility as fields

SCHEMA = 'mfm-formation-decision-v1.6-caller-references-v1'
BINDING_FIELDS = {'caller_sha256', 'candidate_inventory_sha256', 'processed_view_sha256', 'provenance_sha256'}
JOINT_SCHEMAS = {'mfm-joint-caller-input-alignment-v4', 'mfm-joint-caller-input-alignment-v5',
                 'mfm-joint-caller-input-alignment-v6'}
CANDIDATE_FIELDS = {'slot', 'ref_id', 'kind', 'origin', 'namespace', 'eligible_paths',
                    'meaning_token_indices', 'index_token_indices', 'independent_observation'}


def _require(condition, message):
    if not condition:
        raise CognitiveKernelContractError(message)


def _binding(binding):
    _require(type(binding) is dict and set(binding) == BINDING_FIELDS, 'reference binding fields differ')
    for key, value in binding.items():
        _require(require_sha256(value, key) == value, 'reference binding must use canonical SHA256')
    return binding


def _pinned(row, expected):
    _require(require_sha256(expected, 'external reference record pin') == expected, 'noncanonical external pin')
    _require(type(row) is dict and row.get('record_sha256') == expected and
             canonical_sha256({k: v for k, v in row.items() if k != 'record_sha256'}) == expected,
             'reference original record pin or seal differs')


def _inventory(inventory):
    _require(type(inventory) is dict and set(inventory) == {'binding', 'source_profile', 'candidate_inventory',
             'candidates', 'processor_binding'}, 'reference inventory packet differs')
    _binding(inventory['binding'])
    _require(inventory['source_profile'] in JOINT_SCHEMAS and type(inventory['candidate_inventory']) is list and
             canonical_sha256(inventory['candidate_inventory']) == inventory['binding']['candidate_inventory_sha256'],
             'reference inventory profile or original digest differs')
    original, joined = inventory['candidate_inventory'], inventory['candidates']
    _require(type(joined) is list and len(joined) == len(original), 'reference complete candidate roster differs')
    for slot, (candidate, row) in enumerate(zip(original, joined)):
        _require(type(row) is dict and set(row) == CANDIDATE_FIELDS and type(row['slot']) is int and row['slot'] == slot,
                 'reference global candidate slot differs')
        _require(all(row[k] == candidate[k] for k in ('ref_id', 'kind', 'origin', 'namespace', 'eligible_paths')) and
                 row['independent_observation'] is False, 'reference candidate role/path identity differs')
    return inventory


def bind_reference_inventory(caller, joint_alignment, *, caller_sha256, joint_alignment_sha256, expected_binding):
    """Bind already supplied joint metadata to the actual input-only caller roster.

    Whole J4/J5/J6 source/feature admission remains with its original reader; this
    metadata join does not authenticate a service or manufacture processed data.
    """
    _binding(expected_binding); _pinned(caller, caller_sha256); _pinned(joint_alignment, joint_alignment_sha256)
    _require(joint_alignment.get('schema') in JOINT_SCHEMAS, 'unsupported original joint reference profile')
    context, opened, registry = caller_api.caller_input_from_record(caller)
    original = caller_api.selection_candidates(context, opened, registry)
    actual = {'caller_sha256': caller_sha256, 'candidate_inventory_sha256': canonical_sha256(original),
              **joint_alignment['cache']}
    _require(actual == expected_binding and joint_alignment['caller_sha256'] == caller_sha256 and
             joint_alignment['candidate_inventory_sha256'] == actual['candidate_inventory_sha256'],
             'reference caller/inventory/processed-view/provenance binding differs')
    inventory = {'binding': deepcopy(actual), 'source_profile': joint_alignment['schema'],
        'candidate_inventory': original, 'candidates': deepcopy(joint_alignment['candidates']),
        'processor_binding': deepcopy(joint_alignment['processor_binding'])}
    _inventory(inventory)
    ids, masks, tokens = joint_alignment['input_ids'], joint_alignment['attention_mask'], joint_alignment['tokens']
    _require(type(ids) is list and type(masks) is list and type(tokens) is list and len(ids) == len(masks) == len(tokens),
             'reference complete source token roster differs')
    for row in inventory['candidates']:
        for key in ('meaning_token_indices', 'index_token_indices'):
            indices = row[key]
            _require(type(indices) is list and bool(indices) and len(indices) == len(set(indices)) and
                     all(type(i) is int and 0 <= i < len(ids) and masks[i] == 1 and
                         tokens[i]['token_index'] == i and tokens[i]['token_id'] == ids[i] and
                         tokens[i]['attention_mask'] == 1 and tokens[i]['special_mask'] == 0 and
                         row['slot'] in tokens[i]['candidate_slots'] for i in indices),
                     'reference candidate source-token join differs')
    return inventory


def _pointer(path):
    return ''.join('/' + str(key).replace('~', '~0').replace('/', '~1') for key in path)


def _get(record, pointer):
    value = record
    for key in pointer.split('/')[1:]:
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if type(value) is list else value[key]
    return value


def _set(record, pointer, value):
    parent, _, key = pointer.rpartition('/')
    owner = _get(record, parent) if parent else record
    key = key.replace('~1', '/').replace('~0', '~')
    owner[int(key) if type(owner) is list else key] = value


def reference_requests(decision, inventory):
    """Derive requests only from generated marker leaves and normative paths."""
    _inventory(inventory)
    _require(type(decision) is dict and set(decision) == {'schema', 'proposals', 'dispositions'} and
             decision['schema'] == SCHEMA and type(decision['proposals']) is list and type(decision['dispositions']) is list,
             'unsupported complete reference decision')
    rules = [r for r in fields.normative_policy()['rules'] if r['responsibility'] == 'external_selection']
    requests = []
    def visit(value, path):
        matches = [r for r in rules if fields._match(path, r['path'])]
        rule = matches[0] if len(matches) == 1 else None
        marker = type(value) is dict and 'caller_reference' in value
        if marker:
            _require(set(value) == {'caller_reference'} and value['caller_reference'] is True and rule is not None,
                     'reference marker is malformed or has unsupported semantic path')
            eligible = [i for i, c in enumerate(inventory['candidate_inventory']) if rule['candidate_path'] in c['eligible_paths']]
            _require(bool(eligible), 'reference request has no eligible caller candidate')
            requests.append({'pointer': _pointer(path), 'candidate_path': rule['candidate_path'],
                             'eligible_candidate_indices': eligible})
            return
        if rule is not None:
            if value is None:
                _require('null' in rule['kinds'], 'mandatory reference cannot be null')
                return
            tags = {'local_uncertainty'} if path[-1] == 'uncertainty_ref' else \
                {'local_proposal', 'local_value', 'local_uncertainty'} if rule['candidate_path'].endswith(('.contradicts', '.target_refs')) else set()
            _require(type(value) is dict and len(value) == 1 and next(iter(value)) in tags and
                     type(next(iter(value.values()))) is int and next(iter(value.values())) > 0,
                     'external reference must be an explicit marker; free identity repair is forbidden')
            return
        if type(value) is dict:
            for key, child in value.items():
                visit(child, [*path, key])
        elif type(value) is list:
            for index, child in enumerate(value):
                visit(child, [*path, index])
    visit(decision, [])
    return sorted(requests, key=lambda row: row['pointer'])


def compile_reference_decision(context, opened_sources, decision, predictions, *, caller, caller_sha256,
        joint_alignment, joint_alignment_sha256, expected_binding, artifact_sha256, inference_run_id):
    """Resolve the actual predicted set, then parse AND ground the whole output."""
    inventory = bind_reference_inventory(caller, joint_alignment, caller_sha256=caller_sha256,
        joint_alignment_sha256=joint_alignment_sha256, expected_binding=expected_binding)
    supplied_context, supplied_opened, _ = caller_api.caller_input_from_record(caller)
    _require(context.record() == supplied_context.record() and tuple(opened_sources) == supplied_opened,
             'reference consumer actual context or opened bytes differ')
    _require(type(predictions) is dict and set(predictions) == {'binding', 'rows'} and
             _binding(predictions['binding']) == inventory['binding'] and type(predictions['rows']) is list,
             'reference prediction binding or packet differs')
    requests = {row['pointer']: row for row in reference_requests(decision, inventory)}
    chosen = {}
    for row in predictions['rows']:
        _require(type(row) is dict and set(row) == {'pointer', 'candidate_ordinal'} and
                 type(row['pointer']) is str and row['pointer'] in requests and row['pointer'] not in chosen,
                 'reference prediction set contains missing/extra/duplicate request')
        ordinal = row['candidate_ordinal']
        _require(type(ordinal) is int and ordinal in requests[row['pointer']]['eligible_candidate_indices'],
                 'predicted global caller ordinal is ineligible or malformed')
        chosen[row['pointer']] = ordinal
    _require(set(chosen) == set(requests), 'reference prediction set is incomplete')
    resolved = deepcopy(decision)
    for pointer, ordinal in chosen.items():
        _set(resolved, pointer, inventory['candidate_inventory'][ordinal]['ref_id'])
    resolved['schema'] = local.SCHEMA
    bundle = local.compile_decision(context, resolved, artifact_sha256=artifact_sha256, inference_run_id=inference_run_id)
    validate_formation_grounding_v16(context, bundle, tuple(opened_sources))
    return bundle
