"""Verified connected membership derived from source-complete authoring.

The caller must hold the externally pinned, complete authoring context and
verify the original compiler receipt before calling. This pure adapter does
not read files, alter the compiler collection/receipt, migrate families, change
source kinds or grant authority. Its copied view adds only already authored
voice IDs with verified original support. It must be followed by the unchanged
construct_personality.verify_reference_families and allocation validators.

Derivation and collection membership are distinct provenance relations:
https://www.w3.org/TR/prov-o/#wasDerivedFrom
https://www.w3.org/TR/prov-o/#hadMember
No ontology or new source contract is introduced here.
"""
from copy import deepcopy
from hashlib import sha256

from scripts.eipm.gemma_n0.construct_personality import canonical


_FLAGS = ('acceptance_authority', 'private_gradient_authorized',
    'historical_authority_granted', 'training_authorized', 'source_admission')
_ORIGINAL_FORBIDDEN = ('Alice_lived_memory', 'autobiographical_recall_allowed',
    'historical_Elaina_truth', 'generator_has_acceptance_authority', 'model_training_authority')
_DECLARATIONS = frozenset({'training_authority', 'historical_truth_allowed',
    'alice_lived_memory', 'autobiographical_recall_allowed', 'owner_final_review_required',
    'runtime_behavioral_prior_allowed', 'support_mode', 'source_coverage_bin'})
_SUPPORT_FIELDS = (('identity_supporting_e0_ids', 'identity_supporting_E0_unit_ids', 'E0'),
    ('supporting_curated_einf_ids', 'supporting_curated_EINF_ids', 'EINF'))


def _refuse():
    raise ValueError('connected voice reference membership refused')


def _id(value):
    if (type(value) is not str or not value or len(value) > 512
            or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)):
        _refuse()
    return value


def _ids(value):
    if type(value) is not list:
        _refuse()
    values = [_id(item) for item in value]
    if len(set(values)) != len(values):
        _refuse()
    return values


def _body_sha(row):
    body = row.get('original_source_record')
    if type(body) is not dict:
        _refuse()
    digest = sha256(canonical(body)).hexdigest()
    if digest != row.get('original_source_record_sha256'):
        _refuse()
    return digest


def connected_voice_reference_families(compiled_families, original_rows, *, expected_voice_count=96):
    """Return a separate copied view; preserve every original compiler member.

    ASYN_VOICE is a workspace subtype whose pinned authoring row explicitly
    declares the existing ASYN_TARGETED consumer. Its ID must be the original
    proposal ID and must never be presented as historical compiler membership.
    Support modes/coverage bins are checked only as administrative lineage;
    no teaching or calibration labels are made from them.
    """
    if (type(compiled_families) is not list or not compiled_families
            or type(original_rows) is not dict or not original_rows
            or type(expected_voice_count) is not int or expected_voice_count < 1):
        _refuse()
    families, membership = {}, {}
    for family in compiled_families:
        if type(family) is not dict:
            _refuse()
        fid = _id(family.get('family_id'))
        if fid in families or family.get('split') not in {'train', 'dev', 'test'}:
            _refuse()
        members = _ids(family.get('record_ids'))
        if not members or any(rid in membership for rid in members):
            _refuse()
        families[fid] = family
        membership.update({rid: fid for rid in members})
    voices = {}
    for rid, row in original_rows.items():
        _id(rid)
        if type(row) is not dict or type(row.get('draft_workitem')) is not dict:
            _refuse()
        work = row['draft_workitem']
        if work.get('record_id') != rid:
            _refuse()
        if work.get('source_kind') == 'ASYN_VOICE':
            if rid in membership:
                _refuse()
            voices[rid] = row
        elif rid not in membership:
            _refuse()
    if (len(voices) != expected_voice_count
            or set(original_rows) != set(membership) | set(voices)):
        _refuse()
    shared_family = None
    for rid, row in voices.items():
        work = row['draft_workitem']
        fid = _id(work.get('source_family_id'))
        if (fid not in families or families[fid]['split'] != 'train'
                or (shared_family is not None and fid != shared_family)
                or row.get('schema') != 'alice-personality-source-complete-case-authoring-item-v1'
                or row.get('state') != 'CASE_AUTHORING_REQUIRED'
                or work.get('schema') != 'alice-personality-source-linked-curriculum-workitem-v1'
                or work.get('state') != 'SOURCE_LINKED_DRAFT'
                or row.get('source_kind_for_future_document') != 'ASYN_TARGETED'
                or work.get('evidence_category') != 'A-SYN'
                or 'original_source_family_id' not in work or work['original_source_family_id'] is not None
                or work.get('original_split') != 'train'
                or any(value.get(flag) is not False for value in (row, work) for flag in _FLAGS)):
            _refuse()
        shared_family = fid
        digest = _body_sha(row)
        body = row['original_source_record']
        if (body.get('proposal_id') != rid or body.get('provenance_class') != 'A-SYN'
                or digest != work.get('voice_source_record_sha256')
                or any(body.get(flag) is not False for flag in _ORIGINAL_FORBIDDEN)
                or body.get('owner_final_review_required') is not True
                or type(body.get('runtime_behavioral_prior_allowed')) is not bool):
            _refuse()
        decl = work.get('source_declarations')
        if (type(decl) is not dict or set(decl) != _DECLARATIONS
                or any(decl[name] is not False for name in ('training_authority',
                    'historical_truth_allowed', 'alice_lived_memory', 'autobiographical_recall_allowed'))
                or decl['owner_final_review_required'] is not True
                or type(decl['runtime_behavioral_prior_allowed']) is not bool
                or decl['runtime_behavioral_prior_allowed'] != body['runtime_behavioral_prior_allowed']
                or 'support_mode' not in body or decl['support_mode'] != body['support_mode']
                or 'n0_voice_competency' not in body or decl['source_coverage_bin'] != body['n0_voice_competency']):
            _refuse()
        links = work.get('evidence_links')
        if type(links) is not dict or set(links) != {item[0] for item in _SUPPORT_FIELDS}:
            _refuse()
        for link_field, body_field, kind in _SUPPORT_FIELDS:
            supports = _ids(links[link_field])
            if supports != _ids(body.get(body_field)) or (kind == 'E0' and not supports):
                _refuse()
            for sid in supports:
                support = original_rows.get(sid)
                if membership.get(sid) != fid or support is None:
                    _refuse()
                draft = support['draft_workitem']
                if (draft.get('source_kind') != kind or support.get('source_kind_for_future_document') != kind
                        or draft.get('original_source_family_id') != fid
                        or draft.get('source_family_id') != fid or draft.get('original_split') != 'train'):
                    _refuse()
                _body_sha(support)
                original = support['original_source_record']
                id_fields = ('unit_id', 'substrate_id') if kind == 'E0' else ('curated_id', 'source_proposal_id')
                if (next((original[field] for field in id_fields if field in original), None) != sid
                        or original.get('provenance_class') != ('E0' if kind == 'E0' else 'E-INF')):
                    _refuse()
    connected = deepcopy(compiled_families)
    for family in connected:
        if family['family_id'] == shared_family:
            family['record_ids'].extend(sorted(voices))
    return connected
