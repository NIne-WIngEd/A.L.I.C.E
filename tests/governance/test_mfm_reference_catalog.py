"""Caller inventory/codec contracts; no fixture proves learned reference choice."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import importlib.util
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import FormationContextPacket, FormationEvidenceRef
from cognitive_kernel.formation_decision_v16 import SCHEMA, compile_decision
from cognitive_kernel.formation_semantics_v16 import (
    FormationContextV16, RegisteredFormationTarget, RegisteredSourceSensitivity,
)


class ReferenceCatalogTests(unittest.TestCase):
    def interface(self):
        name = 'scripts.mfm.formation_reference_catalog'
        self.assertIsNotNone(importlib.util.find_spec(name), 'reference catalog is not implemented')
        return importlib.import_module(name)

    def context(self):
        scope = ProductHostScope.create(product_id='alice', host_instance_id='host-a',
                                       schema_version='1.0.0', encryption_domain='private-a')
        evidence = tuple(FormationEvidenceRef(
            ref_id=ref, scope=scope, authority_namespace_id='authority-a',
            content_digest=sha256(payload).hexdigest(), role='hypothetical_training',
            modality='text', subject_ref='host',
        ) for ref, payload in (('source-a', b'Fictional event.'),
                               ('source-b', b'claim-not-registered is only raw text.')))
        targets = tuple(RegisteredFormationTarget(
            ref_id=ref, kind=kind, scope=scope, authority_namespace_id='authority-a',
            supporting_evidence_refs=('source-a',),
        ) for kind, ref in (('entity', 'host'), ('entity', 'ally'),
                            ('scene', 'scene-a'), ('mission', 'mission-a'),
                            ('workspace', 'workspace-a')))
        return FormationContextV16(
            FormationContextPacket(scope, 'authority-a', ('source-a',), evidence),
            tuple(RegisteredSourceSensitivity(e.ref_id, 'private') for e in evidence),
            targets,
        )

    def candidates(self, interface, context):
        return tuple(interface.ReferenceCandidate(
            ref_id=ref, kind=kind, scope=context.base.scope,
            authority_namespace_id=context.base.authority_namespace_id,
            description=description,
        ) for kind, ref, description in (
            ('claim', 'old-claim', 'Caller read snapshot of an earlier stored claim.'),
            ('claim', 'distractor-claim', 'An unrelated stored claim.'),
            ('episode', 'old-episode', 'Caller read snapshot of an earlier episode.'),
            ('evidence', 'prior-evidence', 'Explicit prior-memory evidence inventory.'),
            ('uncertainty', 'shared-uncertainty', 'Existing unresolved group.'),
            ('scope', 'memory', 'Explicit authoring scope candidate.'),
            ('scope', 'transient_detail', 'Explicit alternative scope candidate.'),
        ))

    def catalog(self, *, candidates=None):
        interface = self.interface()
        context = self.context()
        if candidates is None:
            candidates = self.candidates(interface, context)
        return interface.build_reference_catalog(
            context, opened_source_refs=('source-a', 'source-b'), candidates=candidates,
        )

    def decision(self):
        base = {
            'kind': 'episode', 'domain': 'host', 'subject_ref': 'host',
            'evidence_refs': ['source-a'], 'value_text': 'An event; claim-not-registered is text.',
            'anchors': [{'ref_id': 'source-a', 'start_byte': 0, 'end_byte': 9, 'locator': None}],
            'epistemic_status': 'hypothetical', 'valid_from': None, 'valid_to': None,
            'confidence': 0.4, 'uncertainty_ref': 'shared-uncertainty',
            'contradicts': ['old-claim'],
            'target_refs': ['old-episode', 'prior-evidence', {'local_proposal': 1},
                            {'local_value': 1}, {'local_uncertainty': 1}],
            'disposition_scope_ref': 'memory', 'temporal_granularity': 'instant',
        }
        first = {
            'proposal': base, 'sensitivity_hint': 'private',
            'episode': {'event_start': '2026-01-01T00:00:00.000000Z', 'event_end': None,
                        'participant_refs': ['host', 'ally'], 'scene_refs': ['scene-a']},
            'relationship_counterpart_ref': None, 'mission_target_refs': ['mission-a'],
            'workspace_target_refs': ['workspace-a'],
        }
        second = deepcopy(first)
        second.update(episode=None, relationship_counterpart_ref='ally')
        second['proposal'].update(kind='relationship_norm', domain='relationship',
                                  value_text='An explicitly fictional negotiated norm.',
                                  uncertainty_ref={'local_uncertainty': 1})
        return {'schema': SCHEMA, 'proposals': [first, second],
                'dispositions': [{'scope_ref': 'memory', 'action': 'propose',
                                  'evidence_refs': ['source-a'],
                                  'target_refs': ['old-claim', {'local_uncertainty': 1}]},
                                 {'scope_ref': 'transient_detail', 'action': 'abstain',
                                  'evidence_refs': ['source-b'], 'target_refs': []}]}

    def test_lossless_full_reference_roundtrip_preserves_semantics_and_local_allocations(self):
        interface = self.interface()
        catalog = self.catalog()
        original = self.decision()
        snapshot = deepcopy(original)
        encoded = interface.encode_reference_decision(original, catalog)
        self.assertEqual(encoded['decision']['proposals'][0]['proposal']['evidence_refs'],
                         [{'catalog_ref': {'kind': 'evidence', 'index': 2}}])
        self.assertEqual(encoded['decision']['proposals'][0]['proposal']['subject_ref'],
                         {'catalog_ref': {'kind': 'entity', 'index': 2}})
        self.assertEqual(encoded['decision']['proposals'][0]['proposal']['value_text'],
                         'An event; claim-not-registered is text.')
        self.assertEqual(encoded['decision']['proposals'][1]['proposal']['uncertainty_ref'],
                         {'local_uncertainty': 1})
        self.assertEqual(interface.decode_reference_decision(encoded, catalog), snapshot)
        self.assertEqual(original, snapshot)
        compiled = compile_decision(self.context(), interface.decode_reference_decision(encoded, catalog),
                                    artifact_sha256='a' * 64, inference_run_id='reference-contract')
        self.assertEqual(compiled.proposals[0].base.uncertainty_ref, 'shared-uncertainty')
        self.assertIn(compiled.proposals[1].base.uncertainty_ref, compiled.proposals[0].base.target_refs)

    def test_catalog_uses_only_explicit_inventory_and_keeps_distractors(self):
        catalog = self.catalog()
        refs = {entry.ref_id for entry in catalog.entries}
        self.assertIn('distractor-claim', refs)
        self.assertNotIn('claim-not-registered', refs)
        self.assertEqual(len(refs), 14)

    def test_catalog_binding_is_independent_of_decision_and_inventory_order(self):
        interface = self.interface()
        context = self.context()
        inventory = self.candidates(interface, context)
        a = self.catalog(candidates=inventory)
        b = interface.build_reference_catalog(context, opened_source_refs=('source-b', 'source-a'),
                                             candidates=tuple(reversed(inventory)))
        self.assertEqual(a.record(), b.record())
        original = self.decision()
        changed = deepcopy(original)
        changed['proposals'][0]['proposal']['value_text'] = 'Different teacher wording.'
        interface.encode_reference_decision(original, a)
        interface.encode_reference_decision(changed, a)
        self.assertEqual(a.record(), b.record())

    def test_learnable_target_contains_indices_without_caller_checksum_metadata(self):
        interface = self.interface()
        self.assertTrue(hasattr(interface, 'learnable_reference_decision'),
                        'explicit learnable-target/transport split is not implemented')
        catalog = self.catalog()
        envelope = interface.encode_reference_decision(self.decision(), catalog)
        target = interface.learnable_reference_decision(envelope, catalog)
        self.assertEqual(set(target), {'schema', 'proposals', 'dispositions'})
        self.assertEqual(target['proposals'][0]['proposal']['evidence_refs'],
                         [{'catalog_ref': {'kind': 'evidence', 'index': 2}}])
        serialized = canonical_json_bytes(target)
        self.assertNotIn(catalog.content_digest().encode(), serialized)
        self.assertNotIn(catalog.context_digest.encode(), serialized)
        self.assertNotIn(b'catalog_digest', serialized)
        self.assertNotIn(b'context_digest', serialized)
        self.assertNotIn(b'authority_namespace_id', serialized)
        target['proposals'][0]['proposal']['value_text'] = 'Changed target copy.'
        self.assertEqual(envelope['decision']['proposals'][0]['proposal']['value_text'],
                         'An event; claim-not-registered is text.')

    def test_caller_rebinds_model_indices_with_its_retained_catalog_digest(self):
        interface = self.interface()
        self.assertTrue(hasattr(interface, 'bind_reference_decision'),
                        'explicit caller transport binding is not implemented')
        catalog = self.catalog()
        envelope = interface.encode_reference_decision(self.decision(), catalog)
        target = interface.learnable_reference_decision(envelope, catalog)
        rebound = interface.bind_reference_decision(target, catalog,
                             expected_catalog_digest=envelope['catalog_digest'])
        self.assertEqual(rebound, envelope)
        self.assertEqual(interface.decode_reference_decision(rebound, catalog), self.decision())
        with self.assertRaises(CognitiveKernelContractError):
            interface.bind_reference_decision(target, catalog, expected_catalog_digest='f' * 64)
        with self.assertRaises(CognitiveKernelContractError):
            interface.bind_reference_decision(envelope, catalog,
                                              expected_catalog_digest=envelope['catalog_digest'])
        changed = deepcopy(envelope)
        changed['catalog_digest'] = 'f' * 64
        with self.assertRaises(CognitiveKernelContractError):
            interface.learnable_reference_decision(changed, catalog)

    def test_semantic_target_does_not_change_when_only_transport_catalog_metadata_changes(self):
        interface = self.interface()
        self.assertTrue(hasattr(interface, 'learnable_reference_decision'),
                        'explicit learnable-target/transport split is not implemented')
        context = self.context()
        original = self.candidates(interface, context)
        a = self.catalog(candidates=original)
        b = self.catalog(candidates=(replace(original[0], description='Revised caller description.'),
                                     *original[1:]))
        self.assertNotEqual(a.content_digest(), b.content_digest())
        first = interface.encode_reference_decision(self.decision(), a)
        second = interface.encode_reference_decision(self.decision(), b)
        self.assertEqual(interface.learnable_reference_decision(first, a),
                         interface.learnable_reference_decision(second, b))
        with self.assertRaises(CognitiveKernelContractError):
            interface.bind_reference_decision(interface.learnable_reference_decision(first, a), b,
                                              expected_catalog_digest=first['catalog_digest'])

    def test_unknown_inventory_is_a_reported_gap_without_auto_registration(self):
        interface = self.interface()
        context = self.context()
        catalog = self.catalog(candidates=tuple(c for c in self.candidates(interface, context)
                                               if c.kind not in ('scope', 'claim')))
        before = catalog.record()
        gaps = interface.reference_inventory_gaps(self.decision(), catalog)
        self.assertIn(('proposals.0.proposal.contradicts.0', 'old-claim'),
                      {(gap['path'], gap['ref_id']) for gap in gaps})
        self.assertIn(('dispositions.0.scope_ref', 'memory'),
                      {(gap['path'], gap['ref_id']) for gap in gaps})
        with self.assertRaises(interface.ReferenceInventoryGap):
            interface.encode_reference_decision(self.decision(), catalog)
        self.assertEqual(catalog.record(), before)

    def test_duplicate_identity_rejected_even_with_a_different_type(self):
        interface = self.interface()
        context = self.context()
        candidates = self.candidates(interface, context)
        for duplicate in (candidates[0], replace(candidates[0], kind='episode'),
                          replace(candidates[0], ref_id='source-a')):
            with self.subTest(duplicate=duplicate), self.assertRaises(CognitiveKernelContractError):
                self.catalog(candidates=(*candidates, duplicate))

    def test_foreign_scope_or_authority_inventory_rejected(self):
        interface = self.interface()
        context = self.context()
        original = self.candidates(interface, context)[0]
        for changed in (replace(original, scope=replace(original.scope, host_instance_id='host-b')),
                        replace(original, authority_namespace_id='authority-b')):
            with self.subTest(changed=changed), self.assertRaises(CognitiveKernelContractError):
                self.catalog(candidates=(changed,))

    def test_opened_sources_need_exact_nonduplicated_context_roster(self):
        interface = self.interface()
        for roster in (('source-a',), ('source-a', 'source-b', 'unknown-source'),
                       ('source-a', 'source-b', 'source-a')):
            with self.subTest(roster=roster), self.assertRaises(CognitiveKernelContractError):
                interface.build_reference_catalog(self.context(), opened_source_refs=roster, candidates=())

    def test_caller_cannot_invent_entity_target_registration(self):
        interface = self.interface()
        context = self.context()
        entity = interface.ReferenceCandidate('invented-entity', 'entity', context.base.scope,
                                               context.base.authority_namespace_id)
        with self.assertRaises(CognitiveKernelContractError):
            self.catalog(candidates=(entity,))

    def test_wrong_reference_type_rejected_at_its_declared_field(self):
        interface = self.interface()
        for field, value in (('subject_ref', 'old-claim'), ('evidence_refs', ['host']),
                             ('uncertainty_ref', 'scene-a'), ('disposition_scope_ref', 'old-claim')):
            decision = self.decision()
            decision['proposals'][0]['proposal'][field] = value
            with self.subTest(field=field), self.assertRaises(CognitiveKernelContractError):
                interface.encode_reference_decision(decision, self.catalog())

    def test_index_decoder_rejects_unknown_wrong_type_or_tampered_catalog(self):
        interface = self.interface()
        catalog = self.catalog()
        encoded = interface.encode_reference_decision(self.decision(), catalog)
        for token in ({'catalog_ref': {'kind': 'evidence', 'index': 0}},
                      {'catalog_ref': {'kind': 'evidence', 'index': True}},
                      {'catalog_ref': {'kind': 'evidence', 'index': 999}},
                      {'catalog_ref': {'kind': 'entity', 'index': 1}},
                      {'catalog_ref': {'kind': 'unknown', 'index': 1}},
                      {'catalog_ref': {'kind': 'evidence', 'index': 2}, 'extra': True},
                      'source-a'):
            changed = deepcopy(encoded)
            changed['decision']['proposals'][0]['proposal']['evidence_refs'][0] = token
            with self.subTest(token=token), self.assertRaises(CognitiveKernelContractError):
                interface.decode_reference_decision(changed, catalog)
        changed = deepcopy(encoded)
        changed['catalog_digest'] = 'f' * 64
        with self.assertRaises(CognitiveKernelContractError):
            interface.decode_reference_decision(changed, catalog)
        other_context = replace(self.context(), base=replace(self.context().base,
                               scope=replace(self.context().base.scope, host_instance_id='host-b')))
        # A context whose own evidence belongs to another scope is rejected before catalog creation.
        with self.assertRaises(CognitiveKernelContractError):
            interface.build_reference_catalog(other_context, opened_source_refs=('source-a', 'source-b'),
                                              candidates=())

    def test_local_indices_are_validated_without_allocating_global_identities(self):
        interface = self.interface()
        for local in ({'local_proposal': 3}, {'local_value': True}, {'local_uncertainty': 2},
                      {'local_uncertainty': 0}, {'local_unknown': 1}):
            changed = self.decision()
            changed['proposals'][0]['proposal']['target_refs'].append(local)
            with self.subTest(local=local), self.assertRaises(CognitiveKernelContractError):
                interface.encode_reference_decision(changed, self.catalog())

    def test_invalid_shape_extra_fields_and_unknown_schema_are_not_repaired(self):
        interface = self.interface()
        for mutate in (
            lambda row: row.update(schema='unsupported'),
            lambda row: row.update(permission_to_write=True),
            lambda row: row['proposals'][0]['proposal'].update(proposal_id='model-assigned'),
            lambda row: row['proposals'][0]['proposal'].update(evidence_refs='source-a'),
            lambda row: row['proposals'][0].update(relationship_counterpart_ref={'local_value': 1}),
        ):
            changed = self.decision()
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(CognitiveKernelContractError):
                interface.encode_reference_decision(changed, self.catalog())


if __name__ == '__main__':
    unittest.main()
