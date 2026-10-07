"""Separate input custody for future full-semantic formation construction.

Only caller context, exact opened bytes and explicit read-only metadata enter
this boundary. It reads no teacher, payload path, authority service or model.
Digests bind supplied inputs; they do not authenticate the caller, grant rights,
establish truth, or prove that a snapshot is current. Candidate path flags are
caller-bound type constraints, not learned decisions or authority permissions.
The original formation compiler and independent policy/registration checks
remain necessary. This manifest is not integrated with the older catalog.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from copy import deepcopy
from hashlib import sha256

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_sha256, require_identifier, require_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_semantics_v16 import FormationContextV16, context_v16_from_record


SCHEMA = 'mfm-formation-caller-input-v1'
REGISTRY_SCHEMA = 'mfm-readonly-reference-registry-v1'
REGISTRY_KINDS = frozenset({'claim', 'episode', 'uncertainty', 'evidence'})
REGISTRY_STATUSES = frozenset({'present', 'withheld', 'revoked', 'deleted'})
_SCOPE_KEYS = {'product_id', 'host_instance_id', 'schema_version', 'encryption_domain'}
_REGISTRY_KEYS = {'schema', 'host_scope', 'namespace', 'generation', 'records',
                  'records_sha256', 'registry_sha256'}
_ENTRY_KEYS = {'ref_id', 'kind', 'record_sha256', 'status', 'description'}
_INPUT_KEYS = {'schema', 'context', 'context_sha256', 'opened_sources',
               'registry_snapshot', 'record_sha256'}
_GOVERNANCE_PATHS = (
    'proposals[].proposal.contradicts', 'proposals[].proposal.target_refs',
    'dispositions[].target_refs',
)
_CITATION_PATHS = (
    'proposals[].proposal.evidence_refs', 'proposals[].proposal.anchors[].ref_id',
    'dispositions[].evidence_refs',
)


def _record(value, keys, name):
    if type(value) is not dict or set(value) != keys:
        raise CognitiveKernelContractError(f'{name} has unsupported or missing fields')
    return value


def _identifier(value, name):
    if require_identifier(value, name) != value:
        raise CognitiveKernelContractError(f'{name} must be canonical')


def _digest(value, name):
    if require_sha256(value, name) != value:
        raise CognitiveKernelContractError(f'{name} must be canonical')


def _registry(snapshot, context):
    if snapshot is None:
        return None
    row = _record(snapshot, _REGISTRY_KEYS, 'read-only registry')
    if row['schema'] != REGISTRY_SCHEMA:
        raise CognitiveKernelContractError('unsupported read-only registry schema')
    scope_row = _record(row['host_scope'], _SCOPE_KEYS, 'registry host_scope')
    scope = ProductHostScope.create(**scope_row)
    if scope.metadata_record() != scope_row:
        raise CognitiveKernelContractError('registry host_scope must be canonical')
    _identifier(row['namespace'], 'registry namespace')
    _identifier(row['generation'], 'registry generation')
    if scope != context.base.scope or row['namespace'] != context.base.authority_namespace_id:
        raise CognitiveKernelContractError('foreign host scope or namespace in read-only registry')
    if type(row['records']) is not list:
        raise CognitiveKernelContractError('registry records must be an ordered array')
    seen = set()
    for value in row['records']:
        entry = _record(value, _ENTRY_KEYS, 'registry record')
        _identifier(entry['ref_id'], 'registry ref_id')
        if type(entry['kind']) is not str or entry['kind'] not in REGISTRY_KINDS:
            raise CognitiveKernelContractError('unsupported read-only registry kind')
        if type(entry['status']) is not str or entry['status'] not in REGISTRY_STATUSES:
            raise CognitiveKernelContractError('unsupported read-only registry status')
        _digest(entry['record_sha256'], 'registry record_sha256')
        if type(entry['description']) is not str or '\x00' in entry['description']:
            raise CognitiveKernelContractError('registry description must be text without NUL')
        if entry['ref_id'] in seen:
            raise CognitiveKernelContractError('duplicate read-only registry identity')
        seen.add(entry['ref_id'])
    _digest(row['records_sha256'], 'registry records_sha256')
    _digest(row['registry_sha256'], 'registry registry_sha256')
    if canonical_sha256(row['records']) != row['records_sha256']:
        raise CognitiveKernelContractError('read-only registry records digest mismatch')
    if canonical_sha256({key: value for key, value in row.items()
                         if key != 'registry_sha256'}) != row['registry_sha256']:
        raise CognitiveKernelContractError('read-only registry snapshot digest mismatch')
    return deepcopy(row)


def _validated_inputs(context, opened_sources, registry_snapshot):
    if not isinstance(context, FormationContextV16):
        raise CognitiveKernelContractError('caller input requires FormationContextV16')
    context.validate()
    if type(opened_sources) not in (tuple, list):
        raise CognitiveKernelContractError('opened sources must be an explicit ordered sequence')
    opened = []
    for value in opened_sources:
        if type(value) not in (tuple, list) or len(value) != 2:
            raise CognitiveKernelContractError('opened source must contain ref_id and exact bytes')
        ref_id, raw = value
        _identifier(ref_id, 'opened source ref_id')
        if type(raw) is not bytes:
            raise CognitiveKernelContractError('opened source payload must be exact bytes')
        opened.append((ref_id, raw))
    if tuple(ref for ref, _ in opened) != tuple(ref.ref_id for ref in context.base.evidence):
        raise CognitiveKernelContractError('opened sources must follow complete context evidence order')
    for ref, (_, raw) in zip(context.base.evidence, opened):
        if sha256(raw).hexdigest() != ref.content_digest:
            raise CognitiveKernelContractError('opened source bytes differ from context content digest')
    registry = _registry(registry_snapshot, context)
    evidence = {ref.ref_id: ref for ref in context.base.evidence}
    targets = {target.ref_id: target for target in context.targets}
    subjects = {ref.subject_ref for ref in context.base.evidence if ref.subject_ref is not None}
    if set(evidence) & set(targets):
        raise CognitiveKernelContractError('opened evidence and registered target identity collision')
    for subject in subjects:
        if subject in evidence or (subject in targets and targets[subject].kind != 'entity'):
            raise CognitiveKernelContractError('source subject has an incompatible identity collision')
    for entry in registry['records'] if registry is not None else ():
        ref_id = entry['ref_id']
        if ref_id in targets or ref_id in subjects:
            raise CognitiveKernelContractError('read-only registry has an incompatible identity collision')
        if ref_id in evidence and (entry['kind'] != 'evidence' or entry['status'] != 'present' or
                                   entry['record_sha256'] != evidence[ref_id].content_digest):
            raise CognitiveKernelContractError('registry/opened evidence collision needs same present byte digest')
    return tuple(opened), registry


def caller_input_record(context, opened_sources, registry_snapshot=None):
    """Seal exact ordered inputs without changing the existing v1.6 context.

    Registry records_sha256 hashes its ordered records. registry_sha256 hashes
    every snapshot field except itself, including the canonical generation ID.
    For evidence overlapping an opened source, record_sha256 must equal that
    source's original-byte content digest and status must be present. This does
    not fetch a prior record or assert that any other record digest is truthful.
    """
    opened, registry = _validated_inputs(context, opened_sources, registry_snapshot)
    core = {'schema': SCHEMA, 'context': context.record(),
            'context_sha256': context.content_digest(),
            'opened_sources': [{'ref_id': ref, 'content_sha256': sha256(raw).hexdigest(),
                                'payload_base64': b64encode(raw).decode('ascii')}
                               for ref, raw in opened],
            'registry_snapshot': registry}
    return {**core, 'record_sha256': canonical_sha256(core)}


def caller_input_from_record(value):
    """Verify the manifest and return (context, exact opened bytes, registry).

    The digest must also be pinned by the consuming caller if external custody
    matters: a self-consistent replacement is not authenticated by this API.
    Returned registry dictionaries are copies, not mutable input aliases.
    """
    row = _record(value, _INPUT_KEYS, 'caller input')
    if row['schema'] != SCHEMA:
        raise CognitiveKernelContractError('unsupported caller-input schema')
    _digest(row['record_sha256'], 'caller input record_sha256')
    if canonical_sha256({key: value for key, value in row.items()
                         if key != 'record_sha256'}) != row['record_sha256']:
        raise CognitiveKernelContractError('caller-input manifest digest mismatch')
    try:
        context = context_v16_from_record(row['context'])
    except (TypeError, KeyError, AttributeError) as exc:
        raise CognitiveKernelContractError('caller input contains malformed v1.6 context') from exc
    _digest(row['context_sha256'], 'caller input context_sha256')
    if context.content_digest() != row['context_sha256']:
        raise CognitiveKernelContractError('caller-input context digest mismatch')
    if type(row['opened_sources']) is not list:
        raise CognitiveKernelContractError('caller-input opened sources must be an ordered array')
    opened = []
    for value in row['opened_sources']:
        source = _record(value, {'ref_id', 'content_sha256', 'payload_base64'}, 'opened source')
        _digest(source['content_sha256'], 'opened content_sha256')
        if type(source['payload_base64']) is not str:
            raise CognitiveKernelContractError('opened source payload_base64 must be text')
        try:
            raw = b64decode(source['payload_base64'], validate=True)
        except (Base64Error, ValueError) as exc:
            raise CognitiveKernelContractError('opened source payload_base64 is invalid') from exc
        if b64encode(raw).decode('ascii') != source['payload_base64']:
            raise CognitiveKernelContractError('opened source payload_base64 must be canonical')
        if sha256(raw).hexdigest() != source['content_sha256']:
            raise CognitiveKernelContractError('opened source payload digest mismatch')
        opened.append((source['ref_id'], raw))
    expected = caller_input_record(context, tuple(opened), row['registry_snapshot'])
    if expected != row:
        raise CognitiveKernelContractError('caller input is not a canonical record')
    return context, tuple(opened), deepcopy(expected['registry_snapshot'])


def selection_candidates(context, opened_sources, registry_snapshot=None):
    """Return input-only typed candidates in caller order with path masks.

    Opened evidence, source-declared subjects, context targets, then explicit
    registry records retain their supplied order and provenance. Subject/entity
    overlap is two distinct typed roles; only a context registration gives
    participant/counterpart eligibility. Metadata-only evidence never gains a
    citation or anchor path. Semantic disposition scopes are not inventoried.
    """
    _, registry = _validated_inputs(context, opened_sources, registry_snapshot)
    records = {row['ref_id']: row for row in registry['records']} if registry else {}
    scope, namespace = context.base.scope.metadata_record(), context.base.authority_namespace_id
    rows = []

    def candidate(ref_id, kind, origin, paths, provenance, *, opened=False):
        rows.append({'ref_id': ref_id, 'kind': kind, 'origin': origin,
                     'host_scope': deepcopy(scope), 'namespace': namespace,
                     'eligible_paths': list(paths), 'citation_eligible': opened,
                     'anchor_eligible': opened, 'provenance': deepcopy(provenance)})

    for ref in context.base.evidence:
        provenance = {'content_sha256': ref.content_digest, 'source_metadata': ref.metadata_record()}
        if ref.ref_id in records:
            provenance.update(registry_record=records[ref.ref_id],
                              registry_generation=registry['generation'],
                              registry_sha256=registry['registry_sha256'])
        candidate(ref.ref_id, 'evidence', 'opened_source', (*_CITATION_PATHS, *_GOVERNANCE_PATHS),
                  provenance, opened=True)
    # Aggregate repeated declarations of one subject without inventing an entity.
    subjects = {}
    for ref in context.base.evidence:
        if ref.subject_ref is not None:
            subjects.setdefault(ref.subject_ref, []).append(ref)
    for ref_id, sources in subjects.items():
        candidate(ref_id, 'source_subject', 'source_subject', ('proposals[].proposal.subject_ref',),
                  {'supporting_evidence_refs': [source.ref_id for source in sources],
                   'source_metadata': [source.metadata_record() for source in sources]})
    target_paths = {
        'entity': ('proposals[].proposal.subject_ref', 'proposals[].episode.participant_refs',
                   'proposals[].relationship_counterpart_ref'),
        'scene': ('proposals[].episode.scene_refs',),
        'mission': ('proposals[].mission_target_refs',),
        'workspace': ('proposals[].workspace_target_refs',),
    }
    for target in context.targets:
        candidate(target.ref_id, target.kind, 'context_target',
                  (*target_paths[target.kind], *_GOVERNANCE_PATHS),
                  {'registration': target.record(),
                   'supporting_evidence_refs': list(target.supporting_evidence_refs)})
    for entry in registry['records'] if registry is not None else ():
        if entry['ref_id'] in {ref.ref_id for ref in context.base.evidence}:
            continue  # Consistent opened evidence was merged, retaining both provenances.
        extra = ('proposals[].proposal.uncertainty_ref',) if entry['kind'] == 'uncertainty' else ()
        candidate(entry['ref_id'], entry['kind'], 'readonly_registry', (*extra, *_GOVERNANCE_PATHS),
                  {'generation': registry['generation'], 'registry_sha256': registry['registry_sha256'],
                   'record': entry})
    return rows
