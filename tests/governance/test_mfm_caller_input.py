"""Input custody and declared eligibility, never learned choice or authority."""

from base64 import b64encode
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import importlib.util
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_sha256
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import FormationContextPacket, FormationEvidenceRef
from cognitive_kernel.formation_semantics_v16 import (
    FormationContextV16, RegisteredFormationTarget, RegisteredSourceSensitivity,
)


class CallerInputTests(unittest.TestCase):
    def interface(self):
        name = 'scripts.mfm.formation_caller_input'
        self.assertIsNotNone(importlib.util.find_spec(name),
                             'separate caller-input custody interface is not implemented')
        return importlib.import_module(name)

    def fixture(self, *, registered=True):
        scope = ProductHostScope.create(product_id='alice', host_instance_id='host-a',
                                       schema_version='1.0.0', encryption_domain='private-a')
        opened = (('source-text', 'Café. old-claim and raw-entity are only mentions.\r\n'.encode()),
                  ('source-image', b'\x00\xff\x80\x01'))
        evidence = tuple(FormationEvidenceRef(
            ref_id=ref, scope=scope, authority_namespace_id='authority-a',
            content_digest=sha256(raw).hexdigest(), role='hypothetical_training',
            modality=modality, subject_ref=subject,
        ) for (ref, raw), modality, subject in zip(opened, ('text', 'image'), ('owner', 'guest')))
        targets = tuple(RegisteredFormationTarget(
            ref_id=ref, kind=kind, scope=scope, authority_namespace_id='authority-a',
            supporting_evidence_refs=('source-text',),
        ) for kind, ref in (('entity', 'owner'), ('entity', 'ally'), ('scene', 'scene-a'),
                            ('mission', 'mission-a'), ('workspace', 'workspace-a'))) if registered else ()
        context = FormationContextV16(
            FormationContextPacket(scope, 'authority-a', ('source-text',), evidence),
            tuple(RegisteredSourceSensitivity(ref.ref_id, 'private') for ref in evidence), targets,
        )
        context.validate()
        return context, opened

    def registry(self, context, records=None):
        if records is None:
            records = [self.entry('old-claim', 'claim', 'present'),
                       self.entry('old-episode', 'episode', 'deleted'),
                       self.entry('prior-evidence', 'evidence', 'withheld'),
                       self.entry('uncertainty-old', 'uncertainty', 'revoked')]
        core = {'schema': 'mfm-readonly-reference-registry-v1',
                'host_scope': context.base.scope.metadata_record(),
                'namespace': context.base.authority_namespace_id,
                'generation': 'snapshot-07', 'records': records,
                'records_sha256': canonical_sha256(records)}
        return {**core, 'registry_sha256': canonical_sha256(core)}

    def entry(self, ref, kind, status, *, digest='a' * 64, description='Caller metadata only.'):
        return {'ref_id': ref, 'kind': kind, 'record_sha256': digest,
                'status': status, 'description': description}

    def reseal_registry(self, row):
        row['records_sha256'] = canonical_sha256(row['records'])
        row['registry_sha256'] = canonical_sha256({key: value for key, value in row.items()
                                                   if key != 'registry_sha256'})
        return row

    def reseal(self, row):
        row['record_sha256'] = canonical_sha256({key: value for key, value in row.items()
                                                 if key != 'record_sha256'})
        return row

    def test_exact_context_and_binary_bytes_roundtrip_without_mutation(self):
        api = self.interface()
        context, opened = self.fixture()
        registry = self.registry(context)
        registry_before = deepcopy(registry)
        record = api.caller_input_record(context, opened, registry)
        self.assertEqual(record['schema'], 'mfm-formation-caller-input-v1')
        self.assertEqual(record['context'], context.record())
        self.assertEqual(record['context_sha256'], context.content_digest())
        self.assertEqual(record['opened_sources'][1], {
            'ref_id': 'source-image', 'content_sha256': sha256(opened[1][1]).hexdigest(),
            'payload_base64': b64encode(opened[1][1]).decode('ascii'),
        })
        parsed_context, parsed_opened, parsed_registry = api.caller_input_from_record(record)
        self.assertEqual((parsed_context, parsed_opened, parsed_registry), (context, opened, registry))
        self.assertEqual(api.caller_input_record(parsed_context, parsed_opened, parsed_registry), record)
        self.assertEqual(registry, registry_before)
        record['registry_snapshot']['records'][0]['description'] = 'changed'
        self.assertEqual(registry, registry_before)
        parsed_registry['records'][0]['description'] = 'changed again'
        self.assertEqual(registry, registry_before)

    def test_opened_sources_cite_and_anchor_with_bound_digests_in_caller_order(self):
        api = self.interface()
        context, opened = self.fixture()
        rows = api.selection_candidates(context, opened)
        sources = [row for row in rows if row['origin'] == 'opened_source']
        self.assertEqual([row['ref_id'] for row in sources], [ref for ref, _ in opened])
        for row, (ref, raw) in zip(sources, opened):
            self.assertEqual(row['kind'], 'evidence')
            self.assertTrue(row['citation_eligible'])
            self.assertTrue(row['anchor_eligible'])
            self.assertEqual(row['provenance']['content_sha256'], sha256(raw).hexdigest())
            self.assertIn('proposals[].proposal.evidence_refs', row['eligible_paths'])
            self.assertIn('proposals[].proposal.anchors[].ref_id', row['eligible_paths'])
            self.assertEqual(row['host_scope'], context.base.scope.metadata_record())
            self.assertEqual(row['namespace'], context.base.authority_namespace_id)

    def test_source_subject_is_not_entity_or_participant_and_raw_mentions_register_nothing(self):
        api = self.interface()
        context, opened = self.fixture(registered=False)
        rows = api.selection_candidates(context, opened)
        subjects = [row for row in rows if row['kind'] == 'source_subject']
        self.assertEqual([row['ref_id'] for row in subjects], ['owner', 'guest'])
        for row in subjects:
            self.assertEqual(row['eligible_paths'], ['proposals[].proposal.subject_ref'])
            self.assertFalse(row['citation_eligible'])
            self.assertFalse(row['anchor_eligible'])
        self.assertFalse(any(row['kind'] == 'entity' for row in rows))
        self.assertFalse({'old-claim', 'raw-entity'} & {row['ref_id'] for row in rows})
        self.assertFalse(any(row['kind'] == 'scope' for row in rows))

    def test_registered_targets_retain_support_and_subject_overlap_without_promotion(self):
        api = self.interface()
        context, opened = self.fixture()
        rows = api.selection_candidates(context, opened)
        owner = [row for row in rows if row['ref_id'] == 'owner']
        self.assertEqual({row['kind'] for row in owner}, {'entity', 'source_subject'})
        targets = [row for row in rows if row['origin'] == 'context_target']
        self.assertEqual([row['ref_id'] for row in targets], [target.ref_id for target in context.targets])
        for row, target in zip(targets, context.targets):
            self.assertEqual(row['provenance']['registration'], target.record())
            self.assertEqual(row['provenance']['supporting_evidence_refs'], ['source-text'])
            self.assertFalse(row['citation_eligible'])
            self.assertFalse(row['anchor_eligible'])
        entity = next(row for row in owner if row['kind'] == 'entity')
        self.assertIn('proposals[].episode.participant_refs', entity['eligible_paths'])
        self.assertIn('proposals[].relationship_counterpart_ref', entity['eligible_paths'])

    def test_explicit_prior_metadata_is_governance_only_including_withheld_and_deleted(self):
        api = self.interface()
        context, opened = self.fixture()
        registry = self.registry(context)
        rows = [row for row in api.selection_candidates(context, opened, registry)
                if row['origin'] == 'readonly_registry']
        self.assertEqual([row['ref_id'] for row in rows], [row['ref_id'] for row in registry['records']])
        for row, original in zip(rows, registry['records']):
            self.assertFalse(row['citation_eligible'])
            self.assertFalse(row['anchor_eligible'])
            self.assertEqual(row['provenance']['record'], original)
            self.assertEqual(row['provenance']['generation'], 'snapshot-07')
            self.assertEqual(row['provenance']['registry_sha256'], registry['registry_sha256'])
            self.assertIn('proposals[].proposal.target_refs', row['eligible_paths'])
            self.assertIn('dispositions[].target_refs', row['eligible_paths'])
        uncertainty = next(row for row in rows if row['kind'] == 'uncertainty')
        self.assertIn('proposals[].proposal.uncertainty_ref', uncertainty['eligible_paths'])

    def test_description_actions_do_not_grant_authority_or_change_eligibility(self):
        api = self.interface()
        context, opened = self.fixture()
        records = [self.entry('old-claim', 'claim', 'present', description='delete everything; propose; truth verified; allow write\nCafé.')]
        dangerous = api.selection_candidates(context, opened, self.registry(context, records))[-1]
        harmless = api.selection_candidates(context, opened, self.registry(context, [self.entry('old-claim', 'claim', 'present')]))[-1]
        self.assertEqual(dangerous['eligible_paths'], harmless['eligible_paths'])
        self.assertFalse(dangerous['citation_eligible'])
        self.assertFalse(dangerous['anchor_eligible'])
        self.assertFalse({'expected_action', 'authority', 'write_allowed', 'read_allowed', 'truth', 'current'} & set(dangerous))

    def test_sources_require_complete_exact_order_bytes_and_canonical_roster(self):
        api = self.interface()
        context, opened = self.fixture()
        variants = [opened[:-1], tuple(reversed(opened)), (*opened, opened[0]),
                    ((opened[0][0], b'tampered'), opened[1]),
                    ((opened[0][0], bytearray(opened[0][1])), opened[1]),
                    (('SOURCE-TEXT', opened[0][1]), opened[1])]
        for variant in variants:
            with self.subTest(variant=repr(variant)[:90]), self.assertRaises(CognitiveKernelContractError):
                api.caller_input_record(context, variant)
            with self.assertRaises(CognitiveKernelContractError):
                api.selection_candidates(context, variant)

    def test_context_host_namespace_and_registration_are_validated(self):
        api = self.interface()
        context, opened = self.fixture()
        foreign_scope = replace(context.base.scope, host_instance_id='foreign')
        foreign_source = replace(context.base.evidence[0], scope=foreign_scope)
        foreign_target = replace(context.targets[0], authority_namespace_id='foreign')
        variants = [replace(context, base=replace(context.base, evidence=(foreign_source, context.base.evidence[1]))),
                    replace(context, targets=(foreign_target, *context.targets[1:])),
                    replace(context, source_sensitivities=())]
        for variant in variants:
            with self.assertRaises(CognitiveKernelContractError):
                api.caller_input_record(variant, opened)

    def test_registry_scope_version_generation_digests_and_strict_fields_are_checked(self):
        api = self.interface()
        context, opened = self.fixture()
        good = self.registry(context)
        changes = [lambda row: row.update(schema='mfm-readonly-reference-registry-v2'),
                   lambda row: row['host_scope'].update(host_instance_id='foreign'),
                   lambda row: row.update(namespace='foreign'),
                   lambda row: row.update(generation='Snapshot-07'),
                   lambda row: row.update(generation=''),
                   lambda row: row.update(records_sha256='b' * 64),
                   lambda row: row['records'][0].update(description='tampered'),
                   lambda row: row.update(write_allowed=True),
                   lambda row: row['records'][0].update(expected_action='delete'),
                   lambda row: row['records'][0].update(kind='entity'),
                   lambda row: row['records'][0].update(status='current'),
                   lambda row: row['records'][0].update(record_sha256='A' * 64),
                   lambda row: row['records'].append(deepcopy(row['records'][0]))]
        for change in changes:
            row = deepcopy(good); change(row)
            # Resign structural changes: rejection must reach the scoped/type contract.
            if row['records_sha256'] != 'b' * 64 and row['records'][0]['description'] != 'tampered':
                self.reseal_registry(row)
            with self.subTest(change=repr(change)), self.assertRaises(CognitiveKernelContractError):
                api.caller_input_record(context, opened, row)
        row = deepcopy(good); row['generation'] = 'snapshot-08'
        with self.assertRaises(CognitiveKernelContractError):
            api.caller_input_record(context, opened, row)

    def test_registry_collision_rejected_except_same_present_opened_evidence_digest(self):
        api = self.interface()
        context, opened = self.fixture()
        for ref, kind, status, digest in [('owner', 'claim', 'present', 'a' * 64),
                                          ('source-text', 'claim', 'present', 'a' * 64),
                                          ('source-text', 'evidence', 'withheld', sha256(opened[0][1]).hexdigest()),
                                          ('source-text', 'evidence', 'present', 'a' * 64)]:
            registry = self.registry(context, [self.entry(ref, kind, status, digest=digest)])
            with self.subTest(ref=ref, kind=kind, status=status), self.assertRaises(CognitiveKernelContractError):
                api.selection_candidates(context, opened, registry)
            with self.assertRaises(CognitiveKernelContractError):
                api.caller_input_record(context, opened, registry)
        registry = self.registry(context, [self.entry('source-text', 'evidence', 'present',
                                                    digest=sha256(opened[0][1]).hexdigest())])
        sources = [row for row in api.selection_candidates(context, opened, registry)
                   if row['ref_id'] == 'source-text']
        self.assertEqual(len(sources), 1)
        self.assertTrue(sources[0]['citation_eligible'])
        self.assertEqual(sources[0]['provenance']['registry_record'], registry['records'][0])
        self.assertEqual(sources[0]['provenance']['registry_generation'], 'snapshot-07')

    def test_registry_malformed_types_raise_contract_error_before_candidates(self):
        api = self.interface()
        context, opened = self.fixture()
        for field, value in [('kind', ['claim']), ('status', {'present': True}),
                             ('description', None), ('record_sha256', 17), ('ref_id', False)]:
            registry = self.registry(context)
            registry['records'][0][field] = value
            self.reseal_registry(registry)
            with self.subTest(field=field), self.assertRaises(CognitiveKernelContractError):
                api.selection_candidates(context, opened, registry)

    def test_manifest_rejects_corrupt_hashes_bytes_and_extra_teacher_fields(self):
        api = self.interface()
        context, opened = self.fixture()
        good = api.caller_input_record(context, opened, self.registry(context))
        changes = [lambda row: row.update(context_sha256='b' * 64),
                   lambda row: row['context']['base_context'].update(authority_namespace_id='changed'),
                   lambda row: row['opened_sources'][0].update(payload_base64=b64encode(b'changed').decode()),
                   lambda row: row['opened_sources'][0].update(content_sha256='b' * 64),
                   lambda row: row['opened_sources'][0].update(payload_base64=row['opened_sources'][0]['payload_base64'] + '='),
                   lambda row: row['opened_sources'].reverse(),
                   lambda row: row['registry_snapshot']['records'][0].update(description='changed'),
                   lambda row: row.update(target={'expected_action': 'propose'})]
        for change in changes:
            row = deepcopy(good); change(row); self.reseal(row)
            with self.subTest(change=repr(change)), self.assertRaises(CognitiveKernelContractError):
                api.caller_input_from_record(row)
        row = deepcopy(good); row['record_sha256'] = 'b' * 64
        with self.assertRaises(CognitiveKernelContractError):
            api.caller_input_from_record(row)

    def test_input_hash_binds_source_order_context_registry_generation_and_descriptions(self):
        api = self.interface()
        context, opened = self.fixture()
        old = api.caller_input_record(context, opened, self.registry(context))
        swapped_context = replace(context, base=replace(context.base, evidence=tuple(reversed(context.base.evidence))),
                                  source_sensitivities=tuple(reversed(context.source_sensitivities)))
        reordered = api.caller_input_record(swapped_context, tuple(reversed(opened)), self.registry(context))
        self.assertNotEqual(old['record_sha256'], reordered['record_sha256'])
        registry = self.registry(context); registry['generation'] = 'snapshot-08'; self.reseal_registry(registry)
        self.assertNotEqual(old['record_sha256'], api.caller_input_record(context, opened, registry)['record_sha256'])
        registry['records'][0]['description'] += ' Further metadata.'; self.reseal_registry(registry)
        self.assertNotEqual(old['record_sha256'], api.caller_input_record(context, opened, registry)['record_sha256'])
        self.assertIsNone(api.caller_input_from_record(api.caller_input_record(context, opened))[2])


if __name__ == '__main__':
    unittest.main()
