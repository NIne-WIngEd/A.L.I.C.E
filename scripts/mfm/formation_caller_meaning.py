"""Caller-supplied meaning beside frozen caller v1, without teacher or live reads.

Records bind supplied custody; they authenticate no service and grant no rights.
Descriptions, absent bodies and validated immutable Claim snapshots stay distinct.
"""
from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path

from cognitive_kernel import canonical, claim_contracts as claims, memory_contracts as memory
from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes, canonical_sha256
from cognitive_kernel.contracts import ProductHostScope
from . import formation_caller_input as caller_api

SCHEMA = 'mfm-formation-caller-meaning-v1'
SNAPSHOT_SCHEMA = 'mfm-caller-semantic-read-snapshot-v1'
POLICY_SCHEMA = 'mfm-caller-meaning-view-policy-v1'
PROFILE = 'text-causal-joint-v1'
FAMILIES = ('opened-source-v1', 'source-subject-v1', 'absent-body-v1',
            'registered-description-v1', 'claim-snapshot-v1')
SECTIONS = ('task_context_snapshot', 'observations', 'candidate_meanings', 'candidate_index')
_BODY_KEYS = {'slot', 'availability', 'family', 'body_base64', 'body_bytes', 'body_sha256'}
_AVAILABILITY = {'present', 'metadata_only', 'unavailable', 'withheld', 'revoked', 'deleted'}


def _require(condition, message):
    if not condition:
        raise CognitiveKernelContractError(message)


def _dict(value, keys, name):
    _require(type(value) is dict and set(value) == set(keys), name + ' has unsupported fields')
    return value


def _seal(core):
    return {**deepcopy(core), 'record_sha256': canonical_sha256(core)}


def _pin(value, expected, schema):
    canonical.require_sha256(expected, 'external meaning digest')
    _require(type(value) is dict and value.get('schema') == schema and value.get('record_sha256') == expected,
             'externally bound meaning record differs')
    _require(canonical_sha256({k: v for k, v in value.items() if k != 'record_sha256'}) == expected,
             'meaning record seal differs')


def _scope(value):
    _dict(value, {'product_id', 'host_instance_id', 'schema_version', 'encryption_domain'}, 'scope')
    result = ProductHostScope(**value); result.validate()
    _require(result.metadata_record() == value, 'noncanonical meaning scope')
    return result


def _list(value, name):
    _require(type(value) is list, name + ' must be an array')
    return tuple(value)


def _envelope(value):
    _dict(value, {f.name for f in fields(memory.MemoryUnitEnvelope)}, 'memory envelope')
    args = deepcopy(value); args['scope'] = _scope(args['scope'])
    for name in ('causal_parents', 'source_records', 'supersedes', 'superseded_by'):
        args[name] = _list(args[name], name)
    result = memory.MemoryUnitEnvelope(**args); result.validate()
    _require(result.metadata_record() == value, 'memory envelope round trip differs')
    return result


def _tagged(value):
    _dict(value, {'type_tag', 'value', 'value_sha256'}, 'typed Claim value')
    result = claims.CanonicalTaggedValue(value['type_tag'], canonical_json_bytes(value['value']).decode('utf-8'),
                                         value['value_sha256'])
    result.validate()
    _require(result.metadata_record() == value, 'typed Claim value round trip differs')
    return result


def _qualifier(value):
    _dict(value, {'key', 'value', 'qualifier_sha256'}, 'Claim qualifier')
    result = claims.ClaimQualifier(value['key'], _tagged(value['value']), value['qualifier_sha256'])
    result.validate()
    _require(result.metadata_record() == value, 'Claim qualifier round trip differs')
    return result


def _claim_record(cls, value):
    _dict(value, {f.name for f in fields(cls)}, 'Claim record')
    args = deepcopy(value); args['envelope'] = _envelope(args['envelope'])
    if cls is claims.ClaimIdentity:
        for name in ('canonical_subject', 'canonical_value'):
            args[name] = _tagged(args[name])
        args['semantic_scope'] = _list(args['semantic_scope'], 'semantic_scope')
    elif cls is claims.ClaimVersion:
        args['value'] = _tagged(args['value'])
        for name in ('evidence_relation_ids', 'correction_of'):
            args[name] = _list(args[name], name)
    if cls in (claims.ClaimIdentity, claims.ClaimVersion):
        args['qualifiers'] = tuple(_qualifier(v) for v in _list(args['qualifiers'], 'qualifiers'))
    result = cls(**args); result.validate()
    _require(result.metadata_record() == value, 'Claim record round trip differs')
    return result


def _body(row):
    raw_fields = [row[name] for name in ('body_base64', 'body_bytes', 'body_sha256')]
    if all(value is None for value in raw_fields):
        return None
    _require(all(value is not None for value in raw_fields) and type(row['body_base64']) is str and
             type(row['body_bytes']) is int and row['body_bytes'] > 0, 'body custody fields differ')
    canonical.require_sha256(row['body_sha256'], 'body FILE digest')
    try:
        raw = b64decode(row['body_base64'], validate=True)
        _require(b64encode(raw).decode('ascii') == row['body_base64'] and
                 len(raw) == row['body_bytes'] and sha256(raw).hexdigest() == row['body_sha256'],
                 'body original byte pin differs')
        value = json.loads(raw.decode('utf-8', 'strict'))
        _require(canonical_json_bytes(value) == raw, 'body must be exact canonical UTF-8 material')
    except (Base64Error, UnicodeError, ValueError, TypeError) as exc:
        raise CognitiveKernelContractError('malformed supplied body') from exc
    return value


def normative_view_policy():
    """Closed code/role/family contract; an external pin is required at use."""
    return _seal({'schema': POLICY_SCHEMA, 'profile': PROFILE, 'families': list(FAMILIES),
        'section_order': list(SECTIONS), 'text_attention': 'causal', 'teacher_input': False,
        'authority': False, 'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'caller_code_sha256': sha256(Path(caller_api.__file__).read_bytes()).hexdigest(),
        'canonical_code_sha256': sha256(Path(canonical.__file__).read_bytes()).hexdigest(),
        'claim_code_sha256': sha256(Path(claims.__file__).read_bytes()).hexdigest(),
        'memory_code_sha256': sha256(Path(memory.__file__).read_bytes()).hexdigest(),
        'claim_schema_version': claims.CLAIM_CONTRACT_SCHEMA_VERSION,
        'memory_schema_version': memory.MEMORY_CONTRACT_SCHEMA_VERSION})


def _snapshot(row, expected, context, registry):
    _pin(row, expected, SNAPSHOT_SCHEMA)
    _dict(row, {'schema', 'scope', 'namespace', 'generation', 'authority_generation', 'as_of',
                'source_position', 'read_service_id', 'read_receipt_sha256', 'policy_sha256', 'record_sha256'}, 'snapshot')
    _require(_scope(row['scope']) == context.base.scope and row['namespace'] == context.base.authority_namespace_id,
             'snapshot scope or namespace differs')
    for name in ('namespace', 'generation', 'read_service_id'):
        _require(canonical.require_identifier(row[name], name) == row[name], 'noncanonical snapshot identifier')
    for name in ('authority_generation', 'source_position'):
        _require(type(row[name]) is int and row[name] >= 0, 'snapshot integer differs')
    _require(canonical.normalize_timestamp(row['as_of'], 'snapshot as_of') == row['as_of'], 'snapshot clock differs')
    for name in ('read_receipt_sha256', 'policy_sha256'):
        canonical.require_sha256(row[name], name)
    _require(registry is None or registry['generation'] == row['generation'], 'registry snapshot generation differs')


def _meaning(row, candidate, snapshot, context):
    _dict(row, _BODY_KEYS, 'candidate meaning')
    _require(type(row['slot']) is int and row['slot'] >= 0 and row['family'] in FAMILIES and
             row['availability'] in _AVAILABILITY, 'unsupported meaning family or availability')
    body = _body(row); family = row['family']; origin = candidate['origin']
    registry = candidate['provenance'].get('record') if origin == 'readonly_registry' else None
    if registry is not None and registry['status'] != 'present':
        _require(row['availability'] == registry['status'] and body is None, 'withheld lifecycle cannot gain content')
    if family == 'opened-source-v1':
        _require(origin == 'opened_source' and row['availability'] == 'present' and body is None,
                 'opened source meaning role differs')
    elif family == 'source-subject-v1':
        _require(origin == 'source_subject' and row['availability'] == 'metadata_only' and body is None,
                 'source subject meaning role differs')
    elif family == 'absent-body-v1':
        _require(origin in ('context_target', 'readonly_registry') and row['availability'] != 'present' and body is None,
                 'absent body must remain absent')
    elif family == 'registered-description-v1':
        _require(origin == 'context_target' and row['availability'] == 'present', 'description role differs')
        _dict(body, {'schema', 'scope', 'namespace', 'ref_id', 'kind', 'description', 'read_receipt_sha256'}, 'description')
        _require(body['schema'] == 'mfm-registered-target-description-v1' and
            _scope(body['scope']) == context.base.scope and body['namespace'] == candidate['namespace'] and
            body['ref_id'] == candidate['ref_id'] and body['kind'] == candidate['kind'] and
            body['read_receipt_sha256'] == snapshot['read_receipt_sha256'], 'description custody or identity differs')
        _require(type(body['description']) is str and '\x00' not in body['description'], 'description must be exact text')
    elif family == 'claim-snapshot-v1':
        _require(origin == 'readonly_registry' and candidate['kind'] == 'claim' and
                 registry['status'] == 'present' and row['availability'] == 'present', 'Claim body role differs')
        _dict(body, {'schema', 'identity', 'version', 'projection'}, 'Claim snapshot')
        _require(body['schema'] == 'mfm-claim-snapshot-meaning-v1', 'Claim family schema differs')
        identity = _claim_record(claims.ClaimIdentity, body['identity'])
        version = _claim_record(claims.ClaimVersion, body['version'])
        projection = _claim_record(claims.CurrentClaimProjection, body['projection'])
        projection.assert_projects(identity, version)
        _require(identity.claim_id == candidate['ref_id'] and registry['record_sha256'] == version.version_sha256,
                 'registry does not bind supplied Claim version')
        for record in (identity, version, projection):
            _require(record.envelope.scope == context.base.scope and
                     record.envelope.authority_namespace_id == candidate['namespace'], 'foreign Claim body')
            _require(record.envelope.deletion_state == 'active', 'inactive Claim body cannot be supplied as present')
        _require(projection.deletion_state == 'active' and
                 projection.authority_generation == snapshot['authority_generation'] and
                 projection.source_position <= snapshot['source_position'], 'Claim snapshot lifecycle or read boundary differs')
    return body


def caller_meaning_record(caller, *, caller_sha256, snapshot, snapshot_sha256,
                          candidate_meanings, view_policy, view_policy_sha256):
    """Validate independently supplied input-only records; never fetch or infer."""
    canonical.require_sha256(caller_sha256, 'caller external digest')
    _require(type(caller) is dict and caller.get('record_sha256') == caller_sha256, 'caller external binding differs')
    context, opened, registry = caller_api.caller_input_from_record(caller)
    _snapshot(snapshot, snapshot_sha256, context, registry)
    _pin(view_policy, view_policy_sha256, POLICY_SCHEMA)
    _require(view_policy == normative_view_policy(), 'normative view policy differs')
    candidates = caller_api.selection_candidates(context, opened, registry)
    _require(type(candidate_meanings) is list and len(candidate_meanings) == len(candidates), 'complete meaning inventory required')
    try:
        for slot, (row, candidate) in enumerate(zip(candidate_meanings, candidates)):
            _require(type(row) is dict and row.get('slot') == slot and type(row.get('slot')) is int,
                     'meaning slots must follow normative inventory order')
            _meaning(row, candidate, snapshot, context)
    except (TypeError, KeyError, AttributeError, UnicodeError, ValueError) as exc:
        if isinstance(exc, CognitiveKernelContractError):
            raise
        raise CognitiveKernelContractError('malformed supplied meaning') from exc
    return _seal({'schema': SCHEMA, 'caller': {'record_sha256': caller_sha256,
        'context_sha256': context.content_digest(), 'caller_code_sha256': sha256(Path(caller_api.__file__).read_bytes()).hexdigest()},
        'snapshot': snapshot, 'candidate_inventory': candidates,
        'candidate_meanings': candidate_meanings, 'view_policy': view_policy})


def caller_meaning_from_record(record, caller, *, meaning_sha256, caller_sha256):
    _pin(record, meaning_sha256, SCHEMA)
    _dict(record, {'schema', 'caller', 'snapshot', 'candidate_inventory', 'candidate_meanings', 'view_policy', 'record_sha256'}, 'companion')
    actual = caller_meaning_record(caller, caller_sha256=caller_sha256, snapshot=record['snapshot'],
        snapshot_sha256=record['snapshot']['record_sha256'], candidate_meanings=record['candidate_meanings'],
        view_policy=record['view_policy'], view_policy_sha256=record['view_policy']['record_sha256'])
    _require(record == actual, 'complete meaning companion differs')
    return actual


def semantic_view(record, caller, *, meaning_sha256, caller_sha256):
    """Project validated content with descriptive/source/authority provenance kept."""
    verified = caller_meaning_from_record(record, caller, meaning_sha256=meaning_sha256, caller_sha256=caller_sha256)
    result = []
    for candidate, entry in zip(verified['candidate_inventory'], verified['candidate_meanings']):
        body = _body(entry)
        provenance_class = ('validated_claim_snapshot' if entry['family'] == 'claim-snapshot-v1' else
                            'registered_description' if entry['family'] == 'registered-description-v1' else
                            'opened_source_metadata' if entry['family'] == 'opened-source-v1' else
                            'source_subject_metadata' if entry['family'] == 'source-subject-v1' else 'metadata_only')
        result.append({'slot': entry['slot'], 'candidate': deepcopy(candidate),
            'availability': entry['availability'], 'family': entry['family'],
            'provenance_class': provenance_class, 'content': body,
            'body_file_sha256': entry['body_sha256']})
    return result
