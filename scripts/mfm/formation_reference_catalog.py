"""Explicit caller inventory for a future learned typed reference selector.

This contract opens no payloads, consults no gold to build its catalog, infers
no registration from raw text, and grants no read/write or factual authority.
It changes only declared reference fields of the existing decision wire format.
Indices are training/consumer handles, not evidence of native pointer learning.
The original compiler, exact grounding and current independent Claim gate remain
necessary after decoding. A caller must supply complete authorized inventories;
unknown references remain contract gaps rather than a default catalog entry.

The encoded envelope is caller transport metadata, NEVER a learned target.
``learnable_reference_decision`` exposes only the indexed decision content.
The caller retains its request's catalog digest independently of model output
and uses ``bind_reference_decision`` before resolving the result. Current MFM
training does not use this separately versioned, unintegrated foundation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_json_bytes, canonical_sha256,
    normalize_identifier_sequence, require_identifier, require_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_decision_v16 import SCHEMA as DECISION_SCHEMA
from cognitive_kernel.formation_semantics_v16 import FormationContextV16, TARGET_KINDS


CATALOG_SCHEMA = 'mfm-explicit-reference-catalog-v1'
SELECTION_SCHEMA = 'mfm-formation-reference-selection-v1'
REFERENCE_KINDS = TARGET_KINDS | frozenset({'evidence', 'claim', 'episode', 'uncertainty', 'scope'})
MEMORY_REFERENCE_KINDS = REFERENCE_KINDS - {'scope'}
LOCAL_KINDS = frozenset({'local_proposal', 'local_value', 'local_uncertainty'})
_BASE_FIELDS = frozenset({
    'kind', 'domain', 'subject_ref', 'evidence_refs', 'value_text', 'anchors',
    'epistemic_status', 'valid_from', 'valid_to', 'confidence', 'uncertainty_ref',
    'contradicts', 'target_refs', 'disposition_scope_ref', 'temporal_granularity',
})
_PROPOSAL_FIELDS = frozenset({
    'proposal', 'sensitivity_hint', 'episode', 'relationship_counterpart_ref',
    'mission_target_refs', 'workspace_target_refs',
})


def _identifier(value, name):
    if require_identifier(value, name) != value:
        raise CognitiveKernelContractError(f'{name} must be canonical')


def _record(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise CognitiveKernelContractError(f'{name} has unsupported or missing fields')
    return value


def _array(value, name):
    if type(value) is not list:
        raise CognitiveKernelContractError(f'{name} must be an array')
    return value


@dataclass(frozen=True)
class ReferenceCandidate:
    """Caller-declared snapshot identity; its description does not establish truth."""

    ref_id: str
    kind: str
    scope: ProductHostScope
    authority_namespace_id: str
    description: str = ''

    def validate(self):
        _identifier(self.ref_id, 'candidate ref_id')
        if self.kind not in REFERENCE_KINDS:
            raise CognitiveKernelContractError('unsupported reference candidate kind')
        if not isinstance(self.scope, ProductHostScope):
            raise CognitiveKernelContractError('candidate needs explicit product-host scope')
        self.scope.validate()
        _identifier(self.authority_namespace_id, 'candidate authority_namespace_id')
        if type(self.description) is not str or '\x00' in self.description:
            raise CognitiveKernelContractError('candidate description must be text')

    def record(self):
        self.validate()
        return {'ref_id': self.ref_id, 'kind': self.kind,
                'scope': self.scope.metadata_record(),
                'authority_namespace_id': self.authority_namespace_id,
                'description': self.description}


@dataclass(frozen=True)
class ReferenceCatalog:
    scope: ProductHostScope
    authority_namespace_id: str
    context_digest: str
    entries: tuple[ReferenceCandidate, ...]

    def validate(self):
        if not isinstance(self.scope, ProductHostScope):
            raise CognitiveKernelContractError('catalog needs explicit product-host scope')
        self.scope.validate()
        _identifier(self.authority_namespace_id, 'catalog authority_namespace_id')
        if require_sha256(self.context_digest, 'catalog context_digest') != self.context_digest:
            raise CognitiveKernelContractError('catalog context digest must be canonical')
        if type(self.entries) is not tuple:
            raise CognitiveKernelContractError('catalog entries must be an immutable roster')
        seen = set()
        for entry in self.entries:
            if not isinstance(entry, ReferenceCandidate):
                raise CognitiveKernelContractError('catalog contains an unsupported candidate')
            entry.validate()
            if entry.scope != self.scope or entry.authority_namespace_id != self.authority_namespace_id:
                raise CognitiveKernelContractError('foreign-scope reference candidate')
            if entry.ref_id in seen:
                raise CognitiveKernelContractError('duplicate catalog reference identity')
            seen.add(entry.ref_id)
        if self.entries != tuple(sorted(self.entries, key=lambda entry: (entry.kind, entry.ref_id))):
            raise CognitiveKernelContractError('catalog roster must be canonical')

    def record(self):
        self.validate()
        return {'schema': CATALOG_SCHEMA, 'scope': self.scope.metadata_record(),
                'authority_namespace_id': self.authority_namespace_id,
                'context_digest': self.context_digest,
                'entries': [entry.record() for entry in self.entries]}

    def content_digest(self):
        return canonical_sha256(self.record())


class ReferenceInventoryGap(CognitiveKernelContractError):
    """Missing explicit caller inventory, not permission to fabricate an identity."""

    def __init__(self, gaps):
        self.gaps = tuple(deepcopy(gaps))
        super().__init__('reference inventory gaps: ' + '; '.join(
            f"{gap['path']} -> {gap['ref_id']} ({','.join(gap['expected_kinds'])})"
            for gap in self.gaps))


def build_reference_catalog(context, *, opened_source_refs, candidates):
    """Build only from context registrations and explicit caller input.

    ``opened_source_refs`` is the caller's exact opened roster, not a read
    operation. This function does not verify source custody or admit snapshots.
    Additional evidence/claim/episode/uncertainty/scope entries must come from
    an authorized caller inventory. Entity/scene/mission/workspace registrations
    come exclusively from ``context.targets``; descriptions cannot invent them.
    No decision, target bundle, teacher labels or raw text is accepted here.
    """
    if not isinstance(context, FormationContextV16):
        raise CognitiveKernelContractError('catalog requires a v1.6 registered context')
    context.validate()
    if type(opened_source_refs) not in (tuple, list):
        raise CognitiveKernelContractError('opened source roster must be explicit')
    opened = tuple(opened_source_refs)
    if normalize_identifier_sequence(opened, 'opened source refs') != opened:
        raise CognitiveKernelContractError('opened source roster must be canonical')
    if set(opened) != {source.ref_id for source in context.base.evidence}:
        raise CognitiveKernelContractError('opened source roster differs from complete context evidence')
    if type(candidates) not in (tuple, list):
        raise CognitiveKernelContractError('additional candidate inventory must be explicit')
    scope, namespace = context.base.scope, context.base.authority_namespace_id
    entries = [ReferenceCandidate(ref, 'evidence', scope, namespace) for ref in opened]
    entries.extend(ReferenceCandidate(target.ref_id, target.kind, target.scope,
                                      target.authority_namespace_id) for target in context.targets)
    for candidate in candidates:
        if not isinstance(candidate, ReferenceCandidate):
            raise CognitiveKernelContractError('caller inventory needs typed reference candidates')
        candidate.validate()
        if candidate.kind in TARGET_KINDS:
            raise CognitiveKernelContractError('target candidate needs existing context registration')
        entries.append(candidate)
    catalog = ReferenceCatalog(scope, namespace, context.content_digest(),
                               tuple(sorted(entries, key=lambda entry: (entry.kind, entry.ref_id))))
    catalog.validate()
    return catalog


def _local(value, allowed, path, proposal_count, uncertainty_count):
    if type(value) is not dict or len(value) != 1:
        raise CognitiveKernelContractError(f'{path} is not a typed local reference')
    kind, index = next(iter(value.items()))
    limit = uncertainty_count if kind == 'local_uncertainty' else proposal_count
    if kind not in allowed or type(index) is not int or not 1 <= index <= limit:
        raise CognitiveKernelContractError(f'{path} has an invalid local reference')
    return deepcopy(value)


def _transform_decision(decision, reference):
    """Traverse declared reference paths only; every other field stays unchanged."""
    _record(decision, {'schema', 'proposals', 'dispositions'}, 'decision')
    if decision['schema'] != DECISION_SCHEMA:
        raise CognitiveKernelContractError('unsupported source decision schema')
    row = deepcopy(decision)
    proposals = _array(row['proposals'], 'decision proposals')
    dispositions = _array(row['dispositions'], 'decision dispositions')
    uncertainty_slots = set()
    for item in proposals:
        _record(item, _PROPOSAL_FIELDS, 'decision proposal')
        base = _record(item['proposal'], _BASE_FIELDS, 'proposal semantic fields')
        value = base['uncertainty_ref']
        if type(value) is dict and 'local_uncertainty' in value:
            index = value['local_uncertainty']
            if set(value) != {'local_uncertainty'} or type(index) is not int or index < 1:
                raise CognitiveKernelContractError('invalid local uncertainty declaration')
            uncertainty_slots.add(index)
    if uncertainty_slots != set(range(1, len(uncertainty_slots) + 1)):
        raise CognitiveKernelContractError('new uncertainty slots must be contiguous')

    def ref(value, kinds, path, *, optional=False, locals=frozenset()):
        if value is None and optional:
            return None
        if type(value) is dict and set(value).intersection(LOCAL_KINDS):
            return _local(value, locals, path, len(proposals), len(uncertainty_slots))
        return reference(value, frozenset(kinds), path)

    def refs(values, kinds, path, *, locals=frozenset()):
        values = _array(values, path)
        result = [ref(value, kinds, f'{path}.{index}', locals=locals)
                  for index, value in enumerate(values)]
        fingerprints = [canonical_json_bytes(value) for value in result]
        if len(fingerprints) != len(set(fingerprints)):
            raise CognitiveKernelContractError(f'{path} contains duplicate references')
        return result

    for index, item in enumerate(proposals):
        path = f'proposals.{index}'
        base = item['proposal']
        for field, kinds, optional, local_kinds in (
            ('subject_ref', {'entity'}, False, frozenset()),
            ('uncertainty_ref', {'uncertainty'}, True, {'local_uncertainty'}),
            ('disposition_scope_ref', {'scope'}, True, frozenset()),
        ):
            base[field] = ref(base[field], kinds, f'{path}.proposal.{field}',
                              optional=optional, locals=local_kinds)
        for field, kinds, local_kinds in (
            ('evidence_refs', {'evidence'}, frozenset()),
            ('contradicts', MEMORY_REFERENCE_KINDS, LOCAL_KINDS),
            ('target_refs', MEMORY_REFERENCE_KINDS, LOCAL_KINDS),
        ):
            base[field] = refs(base[field], kinds, f'{path}.proposal.{field}', locals=local_kinds)
        for anchor_index, anchor in enumerate(_array(base['anchors'], f'{path}.proposal.anchors')):
            _record(anchor, {'ref_id', 'start_byte', 'end_byte', 'locator'}, 'evidence anchor')
            anchor['ref_id'] = ref(anchor['ref_id'], {'evidence'},
                                    f'{path}.proposal.anchors.{anchor_index}.ref_id')
        if item['episode'] is not None:
            episode = _record(item['episode'], {'event_start', 'event_end', 'participant_refs',
                                                'scene_refs'}, 'episode semantics')
            for field, kind in (('participant_refs', 'entity'), ('scene_refs', 'scene')):
                episode[field] = refs(episode[field], {kind}, f'{path}.episode.{field}')
        item['relationship_counterpart_ref'] = ref(item['relationship_counterpart_ref'], {'entity'},
                                                   f'{path}.relationship_counterpart_ref', optional=True)
        for field, kind in (('mission_target_refs', 'mission'), ('workspace_target_refs', 'workspace')):
            item[field] = refs(item[field], {kind}, f'{path}.{field}')
    for index, item in enumerate(dispositions):
        path = f'dispositions.{index}'
        _record(item, {'scope_ref', 'action', 'evidence_refs', 'target_refs'}, 'disposition')
        item['scope_ref'] = ref(item['scope_ref'], {'scope'}, f'{path}.scope_ref')
        item['evidence_refs'] = refs(item['evidence_refs'], {'evidence'}, f'{path}.evidence_refs')
        item['target_refs'] = refs(item['target_refs'], MEMORY_REFERENCE_KINDS,
                                   f'{path}.target_refs', locals=LOCAL_KINDS)
    canonical_json_bytes(row)
    return row


def _encoding(decision, catalog):
    catalog.validate()
    by_id = {}
    counts = {}
    for entry in catalog.entries:
        counts[entry.kind] = counts.get(entry.kind, 0) + 1
        by_id[entry.ref_id] = (entry.kind, counts[entry.kind])
    gaps = []

    def encode(value, kinds, path):
        _identifier(value, path)
        candidate = by_id.get(value)
        if candidate is None:
            gaps.append({'path': path, 'ref_id': value, 'expected_kinds': sorted(kinds)})
            return value  # Only for reporting; an encoded output is never returned with gaps.
        kind, index = candidate
        if kind not in kinds:
            raise CognitiveKernelContractError(f'{path} references wrong candidate type {kind}')
        return {'catalog_ref': {'kind': kind, 'index': index}}

    return _transform_decision(decision, encode), gaps


def reference_inventory_gaps(decision, catalog):
    """Report missing declared references without changing or growing the catalog."""
    _, gaps = _encoding(decision, catalog)
    return gaps


def encode_reference_decision(decision, catalog):
    """Encode a caller-bound transport envelope; do not tokenize it as a target.

    A future learning consumer must use ``learnable_reference_decision`` to
    exclude the per-context checksum and caller binding from prediction.
    """
    encoded, gaps = _encoding(decision, catalog)
    if gaps:
        raise ReferenceInventoryGap(gaps)
    return {'schema': SELECTION_SCHEMA, 'catalog_digest': catalog.content_digest(),
            'decision': encoded}


def decode_reference_decision(selection, catalog):
    """Resolve an externally caller-bound envelope; binding is never a model choice."""
    catalog.validate()
    _record(selection, {'schema', 'catalog_digest', 'decision'}, 'reference selection')
    if selection['schema'] != SELECTION_SCHEMA or selection['catalog_digest'] != catalog.content_digest():
        raise CognitiveKernelContractError('reference selection schema or catalog binding differs')
    by_kind = {}
    for entry in catalog.entries:
        by_kind.setdefault(entry.kind, []).append(entry.ref_id)

    def decode(value, kinds, path):
        _record(value, {'catalog_ref'}, path)
        token = _record(value['catalog_ref'], {'kind', 'index'}, f'{path}.catalog_ref')
        kind, index = token['kind'], token['index']
        if type(kind) is not str or kind not in kinds:
            raise CognitiveKernelContractError(f'{path} selects an unknown or wrong reference type')
        candidates = by_kind.get(kind, ())
        if type(index) is not int or not 1 <= index <= len(candidates):
            raise CognitiveKernelContractError(f'{path} selects an invalid reference index')
        return candidates[index - 1]

    return _transform_decision(selection['decision'], decode)


def learnable_reference_decision(selection, catalog):
    """Expose only valid index/semantic content, without caller transport metadata.

    Validate the original external binding before extracting a detached copy.
    Catalog/context checksums, candidate descriptions and authority metadata
    are not prediction targets. The caller retains that binding separately.
    """
    decode_reference_decision(selection, catalog)
    return deepcopy(selection['decision'])


def bind_reference_decision(decision, catalog, *, expected_catalog_digest):
    """Attach the caller's retained request binding to model-produced index content.

    ``expected_catalog_digest`` must come from the caller's immutable request,
    never from a learned output. Recomputing a replacement binding after the
    catalog changes is not an allowed substitute. Validate typed choices and
    the retained digest before returning an envelope for the existing codec.
    This grants no factual or write authority and performs no output repair.
    """
    if require_sha256(expected_catalog_digest, 'expected catalog digest') != expected_catalog_digest:
        raise CognitiveKernelContractError('caller catalog digest must be canonical')
    selection = {'schema': SELECTION_SCHEMA, 'catalog_digest': expected_catalog_digest,
                 'decision': deepcopy(decision)}
    decode_reference_decision(selection, catalog)
    return selection
