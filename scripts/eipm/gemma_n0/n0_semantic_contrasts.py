"""Target-free current-context contrasts for authentic frozen-feature sessions.

This helper edits no historical evidence or identity authority. It does not
establish whether new wording is truthful/coherent or supply teaching targets.
The caller reviews that meaning, retains derivative ancestry and extracts new
texts in a genuinely new source-verified session. Joint-model failures remain
unlocalized; packaging success cannot approve N0 or a personality.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import re

from src.alice_personality.identity.codec import CognitiveFrameDocument, canonical
from src.alice_personality.identity.contracts import IdentityError

_CURRENT_ROLES = {'situation','selected_spans','relationship','goals_constraints',
                  'stakes_consequences','social_emotional'}
_CONTEXT_VIEWS = {'context','host','relationship','self'}
_CONTEXT_SOURCES = {'HOST','RELATIONSHIP','ASELF'}
_KINDS = {'paraphrase','irrelevant_social','neutral_extra_text','decisive_condition'}


def _require(condition, message):
    if not condition:
        raise IdentityError(message)


def _hash(value):
    return sha256(canonical(value)).hexdigest()


def _text(value):
    _require(type(value) is str and bool(value.strip()), 'contrast text and ancestry must be explicit')
    return value


def _replacements(value, entries, pointer):
    value = {} if value is None else value
    _require(type(value) is dict, 'contrast replacements must explicitly map existing pointers to text')
    original = {pointer(entry):entry.text for entry in entries}
    _require(set(value) <= set(original), 'contrast replacement names an unknown pointer')
    for key, text in value.items():
        _text(key)
        _text(text)
        _require(text != original[key], 'contrast replacement must make an actual text change')
    return value


def _binding(original, variant, *, parent_case_id, parent_family_id, parent_split,
             variant_id, comparison_kind):
    original.validate()
    variant.validate()
    for value in (parent_case_id,parent_family_id,variant_id):
        _text(value)
    _require(parent_split in {'TRAIN','DEV','CAL'}, 'contrasts retain the original development partition')
    _require(comparison_kind in _KINDS, 'unsupported context contrast meaning')
    before, after = original.to_mapping(), variant.to_mapping()
    for key in ('candidates','relations','labels','edges','schema'):
        _require(canonical(before[key]) == canonical(after[key]), 'contrast changed fixed candidate or authority structure')
    fixed = original.to_mapping()
    field_changes, concept_changes, source_changes = [], [], []
    _require(len(original.fields) == len(variant.fields), 'contrast cannot add or remove fields')
    for index, (old,new) in enumerate(zip(original.fields,variant.fields)):
        old_meta, new_meta = asdict(old), asdict(new)
        old_meta.pop('text')
        new_meta.pop('text')
        _require(old_meta == new_meta, 'contrast changed field identity, role, order or availability')
        if old.text != new.text:
            _require(old.available and old.semantic_role in _CURRENT_ROLES,
                'contrast cannot replace absent, historical-time or identity fields')
            field_changes.append({'field_id':old.field_id,'semantic_role':old.semantic_role,
                'original_text_sha256':_hash(old.text),'text_sha256':_hash(new.text)})
            fixed['fields'][index]['text'] = None
    _require(len(original.concepts) == len(variant.concepts), 'contrast cannot add or remove concepts')
    for index, (old,new) in enumerate(zip(original.concepts,variant.concepts)):
        _require(old.record == new.record, 'contrast changed concept records, views or order')
        if old.text != new.text:
            _require(old.record.view in _CONTEXT_VIEWS, 'contrast cannot replace identity concept meaning')
            concept_changes.append({'concept_id':old.record.concept_id,'view':old.record.view,
                'record_sha256':_hash(asdict(old.record)),
                'original_text_sha256':_hash(old.text),'text_sha256':_hash(new.text)})
            fixed['concepts'][index]['text'] = None
    _require(len(original.sources) == len(variant.sources), 'contrast cannot add or remove sources')
    for index, (old,new) in enumerate(zip(original.sources,variant.sources)):
        _require(old.record == new.record, 'contrast changed source records, lineage, permission or order')
        if old.text != new.text:
            record = old.record
            _require(record.source_kind in _CONTEXT_SOURCES and record.context_allowed
                and not any((record.identity_core_allowed,record.historical_truth_allowed,
                             record.alice_lived_memory,record.autobiographical_recall_allowed)),
                'contrast cannot replace historical, identity or lived-experience source text')
            source_changes.append({'record_id':record.record_id,'source_kind':record.source_kind,
                'record_sha256':_hash(asdict(record)),
                'original_text_sha256':_hash(old.text),'text_sha256':_hash(new.text)})
            fixed['sources'][index]['text'] = None
    _require(bool(field_changes or concept_changes or source_changes), 'contrast must change current context text')
    result = {'schema':'alice-personality-authentic-context-contrast-binding-v1',
        'parent':{'case_id':parent_case_id,'family_id':parent_family_id,'split':parent_split},
        'variant_id':variant_id,'comparison_kind':comparison_kind,
        'parent_document_sha256':original.sha256,'document_sha256':variant.sha256,
        'unchanged_payload_sha256':_hash(fixed),'changed_fields':field_changes,
        'changed_context_concepts':concept_changes,'changed_context_sources':source_changes,
        'evaluation_use':'development_contrast_of_existing_parent',
        'qualification':'UNQUALIFIED','independent_qualification':False,
        'n0_approved':False,'personality_approved':False,'source_acceptance_authority':False,
        'experience_event':False,'targets_in_input':False,
        'new_source_verified_feature_session_required':True,
        'failure_attribution':'unresolved_across_provider_readout_n1_n2_n3_consumer'}
    result['binding_sha256'] = _hash(result)
    return result


def build_context_contrast(original: CognitiveFrameDocument, *, replacements,
        parent_case_id, parent_family_id, parent_split, variant_id, comparison_kind,
        contextual_concepts=None, contextual_sources=None):
    """Replace explicitly named current texts; retain all records and ancestry.

    Contextual source-body changes are hypothetical current-episode derivatives,
    never modifications to identity evidence, lived events or source records.
    Inputs/targets must still receive primary review outside this function.
    """
    _require(type(original) is CognitiveFrameDocument, 'contrast needs an actual cognitive document')
    original.validate()
    fields = _replacements(replacements,original.fields,lambda entry:entry.field_id)
    concepts = _replacements(contextual_concepts,original.concepts,lambda entry:entry.record.concept_id)
    sources = _replacements(contextual_sources,original.sources,lambda entry:entry.record.record_id)
    variant = replace(original,
        fields=tuple(replace(entry,text=fields[entry.field_id]) if entry.field_id in fields else entry
                     for entry in original.fields),
        concepts=tuple(replace(entry,text=concepts[entry.record.concept_id])
                       if entry.record.concept_id in concepts else entry for entry in original.concepts),
        sources=tuple(replace(entry,text=sources[entry.record.record_id])
                      if entry.record.record_id in sources else entry for entry in original.sources))
    binding = _binding(original,variant,parent_case_id=parent_case_id,
        parent_family_id=parent_family_id,parent_split=parent_split,
        variant_id=variant_id,comparison_kind=comparison_kind)
    return variant,binding


def verify_context_contrast(original, variant, binding, *, expected_binding_sha256):
    """Verify the exact external binding and actual invariant document structure.

    An external hash is not a signature or independent semantic adjudication.
    The caller must get it from the reviewed proposal, not an unreviewed output.
    """
    _require(type(original) is CognitiveFrameDocument and type(variant) is CognitiveFrameDocument,
        'contrast verification needs actual cognitive documents')
    _require(type(binding) is dict and type(expected_binding_sha256) is str
        and re.fullmatch('[0-9a-f]{64}',expected_binding_sha256) is not None,
        'contrast requires an explicit external binding pin')
    body = {key:value for key,value in binding.items() if key != 'binding_sha256'}
    _require(binding.get('binding_sha256') == expected_binding_sha256 == _hash(body),
        'contrast binding differs from its external reviewed proposal')
    parent = binding.get('parent')
    _require(type(parent) is dict and set(parent) == {'case_id','family_id','split'},
        'contrast parent ancestry is incomplete')
    expected = _binding(original,variant,parent_case_id=parent['case_id'],
        parent_family_id=parent['family_id'],parent_split=parent['split'],
        variant_id=binding.get('variant_id'),comparison_kind=binding.get('comparison_kind'))
    _require(canonical(binding) == canonical(expected), 'contrast declared changes or fixed evidence differ')
    return binding
