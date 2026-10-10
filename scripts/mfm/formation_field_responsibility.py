"""Normative field roles and independently declared exact-literal connections.

Eligibility is mechanical preparation, never an inferred teacher choice,
entailment judgment or authority grant. The original-output profile retains
local IDs; the separate existing decision compiler owns typed local allocation.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from cognitive_kernel import canonical, formation_contracts, formation_semantics_v16 as semantics
from cognitive_kernel import formation_decision_v16 as decision
from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_sha256
from . import formation_caller_input as inputs
from . import formation_source_alignment as source
from . import formation_target_alignment as target

POLICY_SCHEMA = 'mfm-v16-field-responsibility-policy-v1'
SCHEMA = 'mfm-v16-field-responsibilities-v1'
ANNOTATION_SCHEMA = 'mfm-authored-field-responsibilities-v1'
DECLARATION_SCHEMA = 'mfm-authored-literal-segments-v1'
LITERAL_SCHEMA = 'mfm-declared-literal-segment-alignment-v1'


def _require(condition, message):
    if not condition:
        raise CognitiveKernelContractError(message)


def _seal(core):
    return {**deepcopy(core), 'record_sha256': canonical_sha256(core)}


def _pin(row, expected, schema):
    canonical.require_sha256(expected, 'external responsibility digest')
    _require(type(row) is dict and row.get('schema') == schema and row.get('record_sha256') == expected,
             'external responsibility record differs')
    _require(canonical_sha256({k: v for k, v in row.items() if k != 'record_sha256'}) == expected,
             'responsibility record seal differs')


def normative_policy():
    """Explicit exhaustive original-v16 paths/types/roles; consumers pin this."""
    rules = []
    def add(path, kinds, role, candidate_path=None):
        rules.append({'path': path, 'kinds': kinds, 'responsibility': role, 'candidate_path': candidate_path})
    protocol = 'protocol_structure'; semantic = 'semantic_authoring'; selection = 'external_selection'
    p = ['proposals', '*']; base = [*p, 'proposal']; d = ['dispositions', '*']
    for path, kinds in (([], ['object']), (['schema'], ['string']), (['proposals'], ['array']),
                        (p, ['object']), (base, ['object']), (['dispositions'], ['array']), (d, ['object'])):
        add(path, kinds, protocol)
    for name in ('proposal_id', 'value_ref'):
        add([*base, name], ['string'], 'local_bookkeeping')
    for name in ('kind', 'domain', 'epistemic_status', 'value_text', 'temporal_granularity'):
        add([*base, name], ['string'], semantic)
    for name in ('valid_from', 'valid_to', 'disposition_scope_ref'):
        add([*base, name], ['string', 'null'], semantic)
    add([*base, 'confidence'], ['number', 'integer', 'null'], semantic)
    add([*p, 'sensitivity_hint'], ['string', 'null'], semantic)
    add([*p, 'episode'], ['object', 'null'], semantic)
    for name in ('event_start', 'event_end'):
        add([*p, 'episode', name], ['string', 'null'], semantic)
    add([*base, 'subject_ref'], ['string'], selection, 'proposals[].proposal.subject_ref')
    add([*base, 'uncertainty_ref'], ['string', 'null'], selection, 'proposals[].proposal.uncertainty_ref')
    add([*p, 'relationship_counterpart_ref'], ['string', 'null'], selection, 'proposals[].relationship_counterpart_ref')
    for path, candidate in (([*base, 'evidence_refs'], 'proposals[].proposal.evidence_refs'),
                            ([*base, 'contradicts'], 'proposals[].proposal.contradicts'),
                            ([*base, 'target_refs'], 'proposals[].proposal.target_refs'),
                            ([*p, 'episode', 'participant_refs'], 'proposals[].episode.participant_refs'),
                            ([*p, 'episode', 'scene_refs'], 'proposals[].episode.scene_refs'),
                            ([*p, 'mission_target_refs'], 'proposals[].mission_target_refs'),
                            ([*p, 'workspace_target_refs'], 'proposals[].workspace_target_refs'),
                            ([*d, 'evidence_refs'], 'dispositions[].evidence_refs'),
                            ([*d, 'target_refs'], 'dispositions[].target_refs')):
        add(path, ['array'], protocol); add([*path, '*'], ['string'], selection, candidate)
    a = [*base, 'anchors', '*']
    add([*base, 'anchors'], ['array'], protocol); add(a, ['object'], protocol)
    add([*a, 'ref_id'], ['string'], selection, 'proposals[].proposal.anchors[].ref_id')
    for name in ('start_byte', 'end_byte'):
        add([*a, name], ['integer', 'null'], 'source_coordinates')
    add([*a, 'locator'], ['string', 'null'], 'native_locator')
    for name in ('scope_ref', 'action'):
        add([*d, name], ['string'], semantic)
    modules = {'code': Path(__file__), 'base_contract': Path(formation_contracts.__file__),
               'v16_contract': Path(semantics.__file__), 'decision_contract': Path(decision.__file__),
               'caller': Path(inputs.__file__), 'target_alignment': Path(target.__file__)}
    return _seal({'schema': POLICY_SCHEMA, 'profile': target.PROFILE, 'contract_version': semantics.SCHEMA_VERSION,
        'output_schema': semantics.OUTPUT_SCHEMA_V16,
        'contract_binding': {name + '_sha256': sha256(path.read_bytes()).hexdigest() for name, path in modules.items()},
        'rules': rules})


def _match(path, pattern):
    return len(path) == len(pattern) and all(type(part) is int if rule == '*' else
        type(part) is str and part == rule for part, rule in zip(path, pattern))


def _validated(target_alignment, caller, target_alignment_sha256, caller_sha256):
    record = target.validate_target_alignment_record(target_alignment, target_alignment_sha256=target_alignment_sha256)
    context, opened, registry = source._caller(caller, caller_sha256)
    candidates = inputs.selection_candidates(context, opened, registry)
    _require(target_alignment['caller_sha256'] == caller_sha256 and
             target_alignment['context_sha256'] == context.content_digest() and
             target_alignment['candidate_inventory_sha256'] == canonical_sha256(candidates),
             'responsibility caller/target ancestry differs')
    bundle = semantics.bundle_v16_from_output(context, record, artifact_sha256=target_alignment['target_sha256'],
                                               inference_run_id='target-field-construction')
    semantics.validate_formation_grounding_v16(context, bundle, opened)
    return record, candidates


def field_responsibilities(target_alignment, caller, *, target_alignment_sha256, caller_sha256, policy, policy_sha256):
    """Retain all eligible/matching typed alternatives; never choose implicitly."""
    try:
        _pin(policy, policy_sha256, POLICY_SCHEMA)
        _require(policy == normative_policy(), 'supplied policy differs from normative complete v16 path/type/role contract')
        _, candidates = _validated(target_alignment, caller, target_alignment_sha256, caller_sha256)
        rows = []
        for node in target_alignment['geometry']['nodes']:
            rules = [rule for rule in policy['rules'] if _match(node['path'], rule['path'])]
            _require(len(rules) == 1 and node['kind'] in rules[0]['kinds'], 'unknown normative target path or type')
            rule = rules[0]; eligible, matching = [], []
            if rule['candidate_path'] is not None and node['kind'] == 'string':
                eligible = [i for i, c in enumerate(candidates) if rule['candidate_path'] in c['eligible_paths']]
                matching = [i for i in eligible if candidates[i]['ref_id'] == node['value']]
            state = ('not_applicable' if rule['candidate_path'] is None or node['kind'] == 'null' else
                     'missing' if not matching else 'ambiguous' if len(matching) > 1 else 'unique')
            rows.append({'pointer': node['pointer'], 'kind': node['kind'], 'responsibility': rule['responsibility'],
                'value_state': 'absent' if node['kind'] == 'null' else 'present',
                'candidate_path': rule['candidate_path'], 'eligible_candidate_indices': eligible,
                'matching_candidate_indices': matching, 'candidate_status': state, 'annotation_status': 'unannotated'})
        return _seal({'schema': SCHEMA, 'profile': target.PROFILE, 'caller_sha256': caller_sha256,
            'target_alignment_sha256': target_alignment_sha256, 'target_sha256': target_alignment['target_sha256'],
            'policy_sha256': policy_sha256, 'candidate_inventory_sha256': canonical_sha256(candidates),
            'candidates': candidates, 'fields': rows, 'protocol_lexemes': target_alignment['geometry']['lexemes'],
            'annotations_sha256': None, 'learning_qualified': False, 'authority_granted': False})
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, CognitiveKernelContractError):
            raise
        raise CognitiveKernelContractError('malformed field responsibility input') from exc


def _authored(row, expected, schema, target_alignment, caller_sha256, extra):
    _pin(row, expected, schema)
    keys = {'schema', 'target_alignment_sha256', 'target_sha256', 'caller_sha256',
            'author', 'provenance', 'entries', 'record_sha256', *extra}
    _require(set(row) == keys and row['target_alignment_sha256'] == target_alignment['record_sha256'] and
             row['target_sha256'] == target_alignment['target_sha256'] and row['caller_sha256'] == caller_sha256,
             'authored record ancestry differs')
    _require(all(type(row[k]) is str and bool(row[k].strip()) and '\x00' not in row[k] for k in ('author', 'provenance')),
             'independent author and provenance required')
    _require(type(row['entries']) is list, 'authored entries must be an array')


def bind_authored_responsibilities(target_alignment, caller, *, annotations, annotations_sha256, **kwargs):
    """Bind supplied choices; provenance strings do not authenticate an author."""
    try:
        result = field_responsibilities(target_alignment, caller, **kwargs)
        _authored(annotations, annotations_sha256, ANNOTATION_SCHEMA, target_alignment, kwargs['caller_sha256'], {'policy_sha256'})
        _require(annotations['policy_sha256'] == kwargs['policy_sha256'], 'authored policy binding differs')
        fields = {row['pointer']: row for row in result['fields']}; seen = set()
        for entry in annotations['entries']:
            _require(type(entry) is dict and type(entry.get('pointer')) is str and entry['pointer'] in fields and
                     entry['pointer'] not in seen, 'invalid or duplicate authored field pointer')
            seen.add(entry['pointer']); row = fields[entry['pointer']]
            _require(entry.get('mode') == row['responsibility'], 'authored mode cannot reclassify a normative field')
            if entry['mode'] == 'external_selection':
                _require(set(entry) == {'pointer', 'mode', 'candidate_index'} and type(entry['candidate_index']) is int and
                         entry['candidate_index'] in row['matching_candidate_indices'], 'authored reference choice is not an eligible exact caller identity')
                row['selected_candidate_index'] = entry['candidate_index']
            else:
                _require(set(entry) == {'pointer', 'mode'} and entry['mode'] == 'semantic_authoring',
                         'this annotation interface admits semantic authoring or eligible external selection only')
            row['annotation_status'] = 'authored'
        result.pop('record_sha256'); result['annotations_sha256'] = annotations_sha256
        result['author'] = annotations['author']; result['provenance'] = annotations['provenance']
        return _seal(result)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CognitiveKernelContractError):
            raise
        raise CognitiveKernelContractError('malformed authored responsibilities') from exc


def align_declared_literal_segments(caller, source_alignment, target_alignment, declarations, *, caller_sha256,
                                   source_alignment_sha256, target_alignment_sha256, declarations_sha256):
    """Join explicit decoded-value segments to every exact original occurrence."""
    try:
        _validated(target_alignment, caller, target_alignment_sha256, caller_sha256)
        source._pinned_record(source_alignment, source_alignment_sha256, source.SCHEMA)
        _require(target_alignment['source_alignment_sha256'] == source_alignment_sha256 and
                 target_alignment['processor_binding'] == source_alignment['processor_binding'], 'literal source/target ancestry differs')
        _authored(declarations, declarations_sha256, DECLARATION_SCHEMA, target_alignment, caller_sha256, {'source_alignment_sha256'})
        _require(declarations['source_alignment_sha256'] == source_alignment_sha256, 'declared source binding differs')
        nodes = {n['pointer']: n for n in target_alignment['geometry']['nodes']}; seen = set(); segments = []
        prefix = target._prefix(bytes.fromhex(target_alignment['geometry']['canonical_utf8_hex']).decode('utf-8'))
        for entry in declarations['entries']:
            _require(type(entry) is dict and set(entry) == {'pointer', 'target_value_utf8', 'source_occurrences'} and
                     type(entry['pointer']) is str and entry['pointer'] in nodes, 'unsupported authored literal declaration')
            node = nodes[entry['pointer']]; span = entry['target_value_utf8']
            _require(node['kind'] == 'string' and _match(node['path'], ['proposals', '*', 'proposal', 'value_text']),
                     'only warranted value-text segments admit this literal interface; native locators require their own adapter')
            _require(type(span) is list and len(span) == 2 and all(type(v) is int for v in span) and
                     0 <= span[0] < span[1] <= len(node['value'].encode('utf-8')), 'invalid target literal UTF-8 segment')
            key = (entry['pointer'], *span); _require(key not in seen, 'duplicate authored literal segment'); seen.add(key)
            boundaries = {0, len(node['value'].encode('utf-8'))}
            boundaries.update(v for atom in node['characters'] for v in atom['value_utf8'])
            _require(all(v in boundaries for v in span), 'target literal splits a Unicode UTF-8 atom')
            raw = node['value'].encode('utf-8')[slice(*span)]; literal = raw.decode('utf-8')
            mechanical = source.literal_alignment(caller, source_alignment, literal, caller_sha256=caller_sha256,
                                                   alignment_sha256=source_alignment_sha256)
            occurrences = mechanical['occurrences']; _require(bool(occurrences), 'declared literal has no exact original source occurrence')
            selected = entry['source_occurrences']
            if selected is not None:
                _require(type(selected) is list and bool(selected), 'selected occurrences must be a nonempty explicit array or null')
                unique = set()
                for choice in selected:
                    _require(type(choice) is dict and set(choice) == {'source_ref', 'original_utf8'} and
                             type(choice['source_ref']) is str and type(choice['original_utf8']) is list and
                             len(choice['original_utf8']) == 2 and all(type(v) is int for v in choice['original_utf8']),
                             'malformed selected literal occurrence')
                    identity = (choice['source_ref'], *choice['original_utf8'])
                    _require(identity not in unique and any(choice['source_ref'] == o['source_ref'] and
                        choice['original_utf8'] == o['original_utf8'] for o in occurrences), 'selected literal occurrence is not a mechanical alternative')
                    unique.add(identity)
            atoms = [deepcopy(a) for a in node['characters'] if span[0] <= a['value_utf8'][0] and a['value_utf8'][1] <= span[1]]
            begin, end = atoms[0]['canonical_chars'][0], atoms[-1]['canonical_chars'][1]
            byte_begin, byte_end = atoms[0]['canonical_utf8'][0], atoms[-1]['canonical_utf8'][1]
            links = []
            for token in target_alignment['tokens']:
                a, b = token['canonical_chars']; lo, hi = max(begin, a), min(end, b)
                if lo < hi:
                    links.append({'body_index': token['body_index'], 'label_index': token['label_index'],
                        'query_index': token['query_index'], 'decoder_input_index': token['decoder_input_index'],
                        'token_id': token['token_id'], 'canonical_chars': [lo, hi], 'canonical_utf8': [prefix[lo], prefix[hi]],
                        'fully_contained_in_segment': begin <= a and b <= end,
                        'whole_character_atoms': all(not (a < atom['canonical_chars'][1] and b > atom['canonical_chars'][0]) or
                            a <= atom['canonical_chars'][0] and atom['canonical_chars'][1] <= b for atom in atoms)})
            covered = begin
            for a, b in sorted(row['canonical_chars'] for row in links):
                if a > covered:
                    break
                covered = max(covered, b)
            segments.append({'pointer': entry['pointer'], 'target_value_utf8': span, 'literal_utf8_hex': raw.hex(),
                'literal_utf8_sha256': sha256(raw).hexdigest(), 'canonical_chars': [begin, end],
                'canonical_utf8': [byte_begin, byte_end], 'target_characters': atoms, 'target_token_links': links,
                'complete_target_token_coverage': covered >= end, 'occurrences': occurrences,
                'selected_occurrences': selected, 'original_literal_alignment_sha256': mechanical['record_sha256']})
        return _seal({'schema': LITERAL_SCHEMA, 'profile': target.PROFILE, 'caller_sha256': caller_sha256,
            'source_alignment_sha256': source_alignment_sha256, 'target_alignment_sha256': target_alignment_sha256,
            'declarations_sha256': declarations_sha256, 'target_sha256': target_alignment['target_sha256'],
            'author': declarations['author'], 'provenance': declarations['provenance'], 'segments': segments,
            'token_id_copy_qualified': False, 'semantic_entailment_qualified': False,
            'native_locator_alignment': False, 'learning_qualified': False, 'authority_granted': False})
    except (KeyError, TypeError, ValueError, UnicodeError, IndexError) as exc:
        if isinstance(exc, CognitiveKernelContractError):
            raise
        raise CognitiveKernelContractError('malformed declared literal segment input') from exc
