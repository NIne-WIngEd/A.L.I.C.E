"""Finite supplied causal-input contracts; no authority write, grant or target read."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from hashlib import sha256
import json
from pathlib import Path

from cognitive_kernel import canonical, claim_contracts as claims, contracts, experience, ledger, ledger_store
from cognitive_kernel import claim_authority_prototype as authority, formation_sources, memory_contracts
from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes, canonical_sha256
from . import formation_caller_meaning as meaning


def require(ok, message):
    if not ok:
        raise CognitiveKernelContractError(message)


def exact(value, keys, name):
    require(type(value) is dict and set(value) == set(keys), name + ' fields differ')
    return value


def identifier(value):
    require(canonical.require_identifier(value, 'world identifier') == value, 'noncanonical world identifier')
    return value


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'world integer differs')
    return value


def digest(value):
    require(canonical.require_sha256(value, 'world digest') == value, 'noncanonical world digest')
    return value


def timestamp(value):
    require(canonical.normalize_timestamp(value, 'world timestamp') == value, 'noncanonical world timestamp')
    return value


def seal(value, key='record_sha256'):
    return {**deepcopy(value), key: canonical_sha256(value)}


def pin(value, expected, schema, key='record_sha256'):
    digest(expected)
    require(type(value) is dict and value.get('schema') == schema and value.get(key) == expected,
            'world external pin differs')
    require(canonical_sha256({k: v for k, v in value.items() if k != key}) == expected, 'world seal differs')


@dataclass(frozen=True)
class ConstructionLimits:
    max_records: int
    max_artifact_bytes: int
    max_opened_bytes: int

    def __post_init__(self):
        for value in (self.max_records, self.max_artifact_bytes, self.max_opened_bytes):
            integer(value, 1)


def artifact_ref(ref, limits):
    exact(ref, {'artifact_id', 'bytes', 'file_sha256', 'media_type'}, 'world artifact')
    identifier(ref['artifact_id']); integer(ref['bytes']); digest(ref['file_sha256'])
    require(ref['media_type'] in ('application/json', 'application/octet-stream'), 'world artifact type differs')
    require(ref['bytes'] <= limits.max_artifact_bytes, 'complete world artifact exceeds explicit limit')
    return deepcopy(ref)


def read_bytes(ref, read, limits):
    artifact_ref(ref, limits)
    raw = read(deepcopy(ref))
    require(type(raw) is bytes and len(raw) == ref['bytes'] and sha256(raw).hexdigest() == ref['file_sha256'],
            'original artifact bytes differ')
    return raw


def parse_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'duplicate original JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode('utf-8', 'strict'), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        require(type(value) is dict and canonical_json_bytes(value) == raw, 'original JSON is not canonical UTF-8')
    except (UnicodeError, ValueError, TypeError) as exc:
        raise CognitiveKernelContractError('malformed original world JSON') from exc
    return value


def read_json(ref, read, limits):
    return parse_json(read_bytes(ref, read, limits))


def world_contract_binding():
    modules = (canonical, contracts, experience, ledger, ledger_store, authority, claims,
               formation_sources, memory_contracts, meaning)
    return seal({'schema': 'mfm-world-contract-binding-v1', 'profile': 'supplied-claim-prefix-v1',
                 'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                 'construction_code': {name:sha256(Path(__file__).with_name(name+'.py').read_bytes()).hexdigest()
                    for name in ('formation_world_replay','formation_world_caller')},
                 'dependencies': {m.__name__: sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules},
                 'unsupported_full_bodies': ['episode', 'uncertainty', 'mission', 'workspace', 'native'],
                 'authority_write': False, 'permission_granted': False})


def binding(value, expected):
    pin(value, expected, 'mfm-world-contract-binding-v1')
    require(value == world_contract_binding(), 'normative world contract differs')


def scope(value):
    return meaning._scope(value)


def registration(value):
    from cognitive_kernel.formation_contracts import FormationEvidenceRef
    exact(value, {'evidence','object_ref','registration_sha256'}, 'registration')
    record=value['evidence']; exact(record,{f.name for f in fields(FormationEvidenceRef)},'original evidence')
    args=deepcopy(record); args['scope']=scope(args['scope'])
    args['parent_refs']=meaning._list(args['parent_refs'],'source parent_refs')
    evidence=FormationEvidenceRef(**args); evidence.validate()
    result=formation_sources.RegisteredFormationSource.create(evidence=evidence,object_ref=value['object_ref'])
    require(evidence.metadata_record()==record and result.registration_sha256==value['registration_sha256'],
            'original registration normalizes/differs')
    return result


def append_request(value):
    exact(value, {f.name for f in fields(authority.ClaimAuthorityAppendRequest)}, 'original append request')
    args = deepcopy(value)
    args['identity'] = meaning._claim_record(claims.ClaimIdentity, args['identity'])
    for key in ('version_envelope', 'projection_envelope'):
        args[key] = meaning._envelope(args[key])
    args['value'] = meaning._tagged(args['value'])
    args['qualifiers'] = tuple(meaning._qualifier(v) for v in meaning._list(args['qualifiers'], 'qualifiers'))
    for key in ('evidence_relation_ids', 'correction_of'):
        args[key] = meaning._list(args[key], key)
    req = authority.ClaimAuthorityAppendRequest(**args); req.validate()
    rebuilt = {}
    for f in fields(req):
        v = getattr(req, f.name)
        if hasattr(v, 'metadata_record'): v = v.metadata_record()
        elif type(v) is tuple: v = [x.metadata_record() if hasattr(x, 'metadata_record') else x for x in v]
        rebuilt[f.name] = v
    require(rebuilt == value, 'original append request normalizes')
    return req


def governance(ref, read, limits, expected_scope, namespace):
    value = read_json(ref, read, limits)
    exact(value, {'schema', 'scope', 'namespace', 'decision_id', 'issuer_id', 'authority_basis_ref', 'decision',
                  'subject_ref', 'expected_current_claim_version_id', 'request_digest', 'source_refs',
                  'origin', 'record_sha256'}, 'supplied governance')
    pin(value, value['record_sha256'], 'mfm-world-supplied-governance-v1')
    require(scope(value['scope']) == expected_scope and value['namespace'] == namespace and
            value['origin'] == 'research_explicit_records', 'supplied governance scope/origin differs')
    for name in ('decision_id', 'issuer_id', 'authority_basis_ref', 'subject_ref'): identifier(value[name])
    require(value['decision'] in ('accepted', 'deny', 'defer', 'present', 'withheld', 'revoked', 'deleted'),
            'unknown supplied verdict')
    if value['expected_current_claim_version_id'] is not None: identifier(value['expected_current_claim_version_id'])
    if value['request_digest'] is not None: digest(value['request_digest'])
    require(type(value['source_refs']) is list and len(set(value['source_refs'])) == len(value['source_refs']),
            'supplied governance source array differs')
    for ref_id in value['source_refs']: identifier(ref_id)
    return value


def claim_transition(payload, read, limits, expected_scope, namespace, profile):
    exact(payload, {'action', 'governance_ref', 'append_request', 'append_receipt', 'identity', 'version', 'projection'},
          'Claim transition')
    require(payload['action'] in ('append', 'correct', 'invalidate', 'delete'), 'unknown Claim action')
    originals = {key: read_bytes(payload[key], read, limits) for key in ('identity', 'version', 'projection')}
    records = {key: parse_json(raw) for key, raw in originals.items()}
    identity = meaning._claim_record(claims.ClaimIdentity, records['identity'])
    version = meaning._claim_record(claims.ClaimVersion, records['version'])
    projection = meaning._claim_record(claims.CurrentClaimProjection, records['projection'])
    projection.assert_projects(identity, version)
    req = append_request(read_json(payload['append_request'], read, limits))
    receipt_row = read_json(payload['append_receipt'], read, limits)
    exact(receipt_row, {f.name for f in fields(authority.ClaimAuthorityAppendReceipt)}, 'original append receipt')
    for name in ('store_sequence', 'version_sequence', 'projection_generation'): integer(receipt_row[name], 1)
    require(type(receipt_row['idempotent_replay']) is bool, 'receipt replay flag differs')
    receipt = authority.ClaimAuthorityAppendReceipt(**receipt_row); receipt.validate()
    require(authority.ClaimAuthorityAppendReceipt.create(**{k: v for k, v in receipt_row.items() if k != 'receipt_sha256'})
            .metadata_record() == receipt_row, 'receipt normalizes')
    governing = governance(payload['governance_ref'], read, limits, expected_scope, namespace)
    require(governing['decision'] == 'accepted' and governing['subject_ref'] == identity.claim_id and
            governing['expected_current_claim_version_id'] == req.expected_current_claim_version_id and
            governing['request_digest'] == req.request_digest, 'governance/request agreement differs')
    for obj in (identity, version, projection):
        require(obj.envelope.scope == expected_scope and obj.envelope.authority_namespace_id == namespace,
                'foreign original Claim')
    require(req.identity == identity and receipt.authority_id == profile['authority_id'] and
            receipt.claim_id == identity.claim_id and receipt.request_digest == req.request_digest and
            receipt.claim_version_id == version.claim_version_id and receipt.projection_id == projection.projection_id,
            'original Claim identities differ')
    expected_version = claims.ClaimVersion.create(envelope=req.version_envelope, claim_version_id=req.version_envelope.record_id,
        claim_id=identity.claim_id, version_sequence=receipt.version_sequence, store_sequence=receipt.store_sequence,
        event_stream_position=None, value=req.value, qualifiers=req.qualifiers, authority_class=req.authority_class,
        confidence=req.confidence, adjudication_state=req.adjudication_state, evidence_relation_ids=req.evidence_relation_ids,
        conflict_set_id=req.conflict_set_id, correction_of=req.correction_of, request_digest=req.request_digest)
    expected_projection = claims.CurrentClaimProjection.create(envelope=req.projection_envelope,
        projection_id=req.projection_envelope.record_id, claim_id=identity.claim_id,
        current_claim_version_id=version.claim_version_id, authority_generation=profile['authority_generation'],
        projection_generation=receipt.projection_generation, adjudication_state=version.adjudication_state,
        validity_state=req.validity_state, conflict_state=req.conflict_state, deletion_state=req.deletion_state,
        source_position=receipt.store_sequence)
    require(version == expected_version and projection == expected_projection, 'original transition result differs')
    require((payload['action'] != 'delete' or req.deletion_state != 'active') and
            (payload['action'] != 'invalidate' or req.validity_state != 'current') and
            (payload['action'] != 'correct' or bool(req.correction_of)), 'declared action disagrees with result')
    return {'identity': identity, 'version': version, 'projection': projection, 'request': req,
            'receipt': receipt, 'governance': governing, 'originals': originals}
