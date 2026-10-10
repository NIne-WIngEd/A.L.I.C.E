"""Real scoped records for independent caller meaning; no live reads or learning."""
from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
import importlib
import unittest

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, CognitiveKernelContractError
from cognitive_kernel.claim_contracts import CanonicalTaggedValue, ClaimIdentity, ClaimVersion, CurrentClaimProjection
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope
from cognitive_kernel.formation_semantics_v16 import RegisteredFormationTarget, context_v16_from_record
from scripts.mfm import formation_caller_input as caller_api
from tests.governance import test_mfm_source_alignment as source_fixture

def sealed(core):
    return {**deepcopy(core), 'record_sha256': canonical_sha256(core)}

def envelope(scope, namespace, record_id, record_type, role, source_records=()):
    return MemoryUnitEnvelope.create(scope=scope, record_id=record_id, record_type=record_type,
        authority_namespace_id=namespace, host_or_cluster_id=scope.host_instance_id,
        authority_role=role, deployment_profile='single_workstation',
        created_at='2026-08-06T04:00:00Z', valid_from='2026-08-06T04:00:00Z', valid_to=None,
        transaction_time='2026-08-06T04:00:00Z', logical_clock=1, generation=0,
        state='committed', data_classification='ordinary', retention_class='authoritative_source',
        deletion_state='active', provenance_digest='a' * 64, content_digest='b' * 64,
        writer='fictional-authority', workflow_or_request_id='fictional-read',
        idempotency_namespace='fictional', idempotency_key=record_id, source_records=source_records)

def claim_body(scope, namespace, value=None):
    if value is None:
        value = {'literal/~': 'Café "A" \\ <|image|> 😀', 'count': 2, 'weight': 0.5,
                 'flag': True, 'missing': None, 'list': ['ana', 'ana']}
    tagged = CanonicalTaggedValue.create(type_tag='map', value=value)
    identity = ClaimIdentity.create(envelope=envelope(scope, namespace, 'claim-1', 'claim_identity', 'claim_authority'),
        claim_id='claim-1', canonical_subject=CanonicalTaggedValue.create(type_tag='identifier', value='owner'),
        canonical_predicate='stored_preference', canonical_value=tagged, semantic_scope=('fictional',))
    version = ClaimVersion.create(envelope=envelope(scope, namespace, 'version-1', 'claim_version', 'claim_authority', ('claim-1',)),
        claim_version_id='version-1', claim_id='claim-1', version_sequence=1, store_sequence=17,
        event_stream_position=42, value=tagged, authority_class='owner_attested', confidence=0.5,
        adjudication_state='disputed', evidence_relation_ids=('binding-1',), conflict_set_id='conflict-1', request_digest='c' * 64)
    projection = CurrentClaimProjection.create(
        envelope=envelope(scope, namespace, 'projection-1', 'current_claim_projection', 'registered_projection', ('claim-1', 'version-1')),
        projection_id='projection-1', claim_id='claim-1', current_claim_version_id='version-1',
        authority_generation=2, projection_generation=3, adjudication_state='disputed',
        validity_state='historical', conflict_state='open', deletion_state='active', source_position=17)
    return {'schema': 'mfm-claim-snapshot-meaning-v1', 'identity': identity.metadata_record(),
            'version': version.metadata_record(), 'projection': projection.metadata_record()}

def body_entry(slot, family, body=None, availability='present'):
    raw = canonical_json_bytes(body) if body is not None else None
    return {'slot': slot, 'availability': availability, 'family': family,
            'body_base64': b64encode(raw).decode('ascii') if raw is not None else None,
            'body_bytes': len(raw) if raw is not None else None,
            'body_sha256': sha256(raw).hexdigest() if raw is not None else None}

def inputs(value=None):
    base = source_fixture.caller(('Café "A" \\ <|image|> 😀 banana banana',))
    context, opened, _ = caller_api.caller_input_from_record(base)
    from dataclasses import replace
    targets = tuple(RegisteredFormationTarget(ref, kind, context.base.scope,
        context.base.authority_namespace_id, ('source-0',)) for ref, kind in
        [('owner', 'entity'), ('scene-1', 'scene'), ('mission-1', 'mission'), ('workspace-1', 'workspace')])
    context = replace(context, targets=targets)
    body = claim_body(context.base.scope, context.base.authority_namespace_id, value)
    registry_core = {'schema': caller_api.REGISTRY_SCHEMA, 'host_scope': context.base.scope.metadata_record(),
        'namespace': context.base.authority_namespace_id, 'generation': 'generation-2', 'records': [
            {'ref_id': 'claim-1', 'kind': 'claim', 'record_sha256': body['version']['version_sha256'],
             'status': 'present', 'description': 'A stored assertion, not an independent observation.'},
            {'ref_id': 'episode-1', 'kind': 'episode', 'record_sha256': 'd' * 64,
             'status': 'present', 'description': 'Episode metadata; full original body unavailable.'},
            {'ref_id': 'uncertainty-1', 'kind': 'uncertainty', 'record_sha256': 'e' * 64,
             'status': 'present', 'description': 'Known uncertainty identity; no fetched body.'},
            {'ref_id': 'withheld-source', 'kind': 'evidence', 'record_sha256': 'f' * 64,
             'status': 'withheld', 'description': 'Withheld record metadata.'}]}
    registry = {**registry_core, 'records_sha256': canonical_sha256(registry_core['records'])}
    registry['registry_sha256'] = canonical_sha256(registry)
    caller = caller_api.caller_input_record(context, opened, registry)
    snapshot = sealed({'schema': 'mfm-caller-semantic-read-snapshot-v1',
        'scope': context.base.scope.metadata_record(), 'namespace': context.base.authority_namespace_id,
        'generation': 'generation-2', 'authority_generation': 2,
        'as_of': '2026-08-06T05:00:00.000000Z', 'source_position': 42,
        'read_service_id': 'fictional-read-service', 'read_receipt_sha256': '1' * 64, 'policy_sha256': '2' * 64})
    candidates = caller_api.selection_candidates(context, opened, registry)
    entries = []
    for slot, candidate in enumerate(candidates):
        if candidate['origin'] == 'opened_source':
            entry = body_entry(slot, 'opened-source-v1')
        elif candidate['origin'] == 'source_subject':
            entry = body_entry(slot, 'source-subject-v1', availability='metadata_only')
        elif candidate['origin'] == 'context_target':
            description = {'schema': 'mfm-registered-target-description-v1',
                'scope': context.base.scope.metadata_record(), 'namespace': context.base.authority_namespace_id,
                'ref_id': candidate['ref_id'], 'kind': candidate['kind'],
                'description': candidate['kind'] + ' description Café "A" <|image|>',
                'read_receipt_sha256': snapshot['read_receipt_sha256']}
            entry = body_entry(slot, 'registered-description-v1', description)
        elif candidate['ref_id'] == 'claim-1':
            entry = body_entry(slot, 'claim-snapshot-v1', body)
        else:
            entry = body_entry(slot, 'absent-body-v1', availability=
                'withheld' if candidate['ref_id'] == 'withheld-source' else 'metadata_only')
        entries.append(entry)
    return caller, snapshot, entries, body

class CallerMeaningTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('scripts.mfm.formation_caller_meaning')

    def build(self, value=None):
        api = self.api(); caller, snapshot, entries, body = inputs(value)
        policy = api.normative_view_policy()
        kwargs = {'caller_sha256': caller['record_sha256'], 'snapshot': snapshot,
                  'snapshot_sha256': snapshot['record_sha256'], 'candidate_meanings': entries,
                  'view_policy': policy, 'view_policy_sha256': policy['record_sha256']}
        record = api.caller_meaning_record(caller, **kwargs)
        return api, caller, record, kwargs, body

    def test_complete_input_retains_typed_roles_and_missing_originals(self):
        api, caller, record, kwargs, body = self.build()
        roles = [(r['ref_id'], r['kind'], r['origin']) for r in record['candidate_inventory']]
        self.assertEqual(roles[:3], [('source-0', 'evidence', 'opened_source'),
            ('owner', 'source_subject', 'source_subject'), ('owner', 'entity', 'context_target')])
        self.assertEqual(len(record['candidate_meanings']), 10)
        self.assertEqual(record['candidate_meanings'][-1]['availability'], 'withheld')
        self.assertIsNone(record['candidate_meanings'][-2]['body_base64'])
        self.assertEqual(api.caller_meaning_from_record(record, caller,
            meaning_sha256=record['record_sha256'], caller_sha256=caller['record_sha256']), record)
        record['candidate_inventory'][0]['eligible_paths'].append('forged')
        self.assertNotIn('forged', caller_api.selection_candidates(*caller_api.caller_input_from_record(caller))[0]['eligible_paths'])

    def test_teacher_access_is_unnecessary_and_candidate_meaning_is_independent(self):
        from unittest.mock import patch
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',
                   side_effect=AssertionError('teacher must remain inaccessible')):
            _, _, record, _, _ = self.build()
        self.assertEqual(record['candidate_meanings'][2]['family'], 'registered-description-v1')
        self.assertFalse(any('target' in key for key in record if key != 'candidate_inventory'))

    def test_claim_adapter_keeps_actual_values_lifecycle_and_all_hashes(self):
        api, caller, record, kwargs, body = self.build()
        actual = api.semantic_view(record, caller, meaning_sha256=record['record_sha256'], caller_sha256=caller['record_sha256'])
        claim = actual[6]['content']
        self.assertEqual(claim, body)
        self.assertEqual(claim['version']['value']['value']['count'], 2)
        self.assertIsNone(claim['version']['value']['value']['missing'])
        self.assertEqual(claim['projection']['conflict_state'], 'open')
        self.assertEqual(claim['projection']['validity_state'], 'historical')

    def test_bad_claim_hash_is_not_repaired_by_reconstruction(self):
        api = self.api(); caller, snapshot, entries, body = inputs()
        body['version']['value']['value_sha256'] = '0' * 64
        entries[6] = body_entry(6, 'claim-snapshot-v1', body)
        policy = api.normative_view_policy()
        with self.assertRaises(CognitiveKernelContractError):
            api.caller_meaning_record(caller, caller_sha256=caller['record_sha256'], snapshot=snapshot,
                snapshot_sha256=snapshot['record_sha256'], candidate_meanings=entries,
                view_policy=policy, view_policy_sha256=policy['record_sha256'])

    def test_claim_current_version_scope_generation_and_read_position_are_checked(self):
        api, caller, _, kwargs, body = self.build()
        for mutate in ('version', 'scope', 'generation', 'position'):
            changed = deepcopy(kwargs)
            if mutate in ('generation', 'position'):
                changed['snapshot']['authority_generation' if mutate == 'generation' else 'source_position'] = 1
                changed['snapshot'] = sealed({k: v for k, v in changed['snapshot'].items() if k != 'record_sha256'})
                changed['snapshot_sha256'] = changed['snapshot']['record_sha256']
            else:
                altered = deepcopy(body)
                if mutate == 'version':
                    altered['projection']['current_claim_version_id'] = 'wrong-version'
                else:
                    altered['identity']['envelope']['scope']['host_instance_id'] = 'foreign-host'
                changed['candidate_meanings'][6] = body_entry(6, 'claim-snapshot-v1', altered)
            with self.subTest(mutation=mutate), self.assertRaises(CognitiveKernelContractError):
                api.caller_meaning_record(caller, **changed)

    def test_withheld_record_cannot_gain_body_or_present_status(self):
        api, caller, _, kwargs, body = self.build()
        for entry in (body_entry(9, 'claim-snapshot-v1', body), body_entry(9, 'absent-body-v1')):
            changed = deepcopy(kwargs); changed['candidate_meanings'][9] = entry
            with self.assertRaises(CognitiveKernelContractError):
                api.caller_meaning_record(caller, **changed)

    def test_unknown_family_role_promotion_and_slot_omission_refuse(self):
        api, caller, _, kwargs, _ = self.build()
        for mutate in ('family', 'subject', 'slot', 'omit', 'policy'):
            changed = deepcopy(kwargs)
            if mutate == 'family': changed['candidate_meanings'][7]['family'] = 'invented-episode-adapter'
            elif mutate == 'subject': changed['candidate_meanings'][1] = deepcopy(changed['candidate_meanings'][2])
            elif mutate == 'slot': changed['candidate_meanings'][2]['slot'] = 1
            elif mutate == 'omit': changed['candidate_meanings'].pop()
            else:
                changed['view_policy']['families'].append('scope-as-entity')
                changed['view_policy'] = sealed({k: v for k, v in changed['view_policy'].items() if k != 'record_sha256'})
                changed['view_policy_sha256'] = changed['view_policy']['record_sha256']
            with self.subTest(mutation=mutate), self.assertRaises(CognitiveKernelContractError):
                api.caller_meaning_record(caller, **changed)

    def test_supplied_description_identity_and_body_file_pin_are_required(self):
        api, caller, _, kwargs, _ = self.build()
        for mutate in ('body-pin', 'identity', 'extra', 'noncanonical'):
            from base64 import b64decode
            import json
            changed = deepcopy(kwargs); entry = changed['candidate_meanings'][2]
            if mutate == 'body-pin': entry['body_sha256'] = '0' * 64
            else:
                body = json.loads(b64decode(entry['body_base64']))
                if mutate == 'identity': body['ref_id'] = 'not-owner'
                elif mutate == 'extra': body['expected_action'] = 'correct'
                if mutate == 'noncanonical':
                    raw = json.dumps(body, ensure_ascii=False, indent=2).encode()
                    entry.update(body_base64=b64encode(raw).decode(), body_bytes=len(raw), body_sha256=sha256(raw).hexdigest())
                else: changed['candidate_meanings'][2] = body_entry(2, 'registered-description-v1', body)
            with self.subTest(mutation=mutate), self.assertRaises(CognitiveKernelContractError):
                api.caller_meaning_record(caller, **changed)
