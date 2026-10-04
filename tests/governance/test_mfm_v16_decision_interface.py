"""Test producer/consumer contracts; these fixtures do not test learned ability."""
from copy import deepcopy
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record, model_input_sha256_v16
from scripts.mfm.build_v16_authoring_seed import CASES, build_one


class DecisionInterfaceTests(unittest.TestCase):
    def interface(self):
        from cognitive_kernel import formation_decision_v16 as interface
        return interface

    def example(self, case_id):
        row = build_one(next(s for s in CASES if s['case_id'] == case_id))
        return row, learning_example_v16_from_record(row, split=row['split'])

    def test_local_id_renaming_does_not_change_the_learning_target(self):
        row, original = self.example('v16-seed-02')
        changed = deepcopy(row)
        changed['case_id'] = 'different-author-bookkeeping'
        for i, p in enumerate(changed['target']['proposals']):
            p['proposal']['proposal_id'] = f'other-proposal-{i}'
            p['proposal']['value_ref'] = f'other-value-{i}'
        renamed = learning_example_v16_from_record(changed, split=changed['split'])
        self.assertEqual(model_input_sha256_v16(original), model_input_sha256_v16(renamed))
        self.assertNotEqual(original.target.output_record(), renamed.target.output_record())
        self.assertEqual(self.interface().decision_record(original), self.interface().decision_record(renamed))

    def test_existing_shared_uncertainty_retains_its_identity(self):
        row, _ = self.example('v16-seed-18')
        first = row['target']['proposals'][0]
        first['proposal']['uncertainty_ref'] = 'registered-uncertainty-17'
        second = deepcopy(first)
        second['proposal'].update(proposal_id='second-proposal', value_ref='second-value')
        row['target']['proposals'].append(second)
        example = learning_example_v16_from_record(row, split=row['split'])
        decision = self.interface().decision_record(example)
        compiled = self.interface().compile_decision(example.context, decision,
            artifact_sha256='a' * 64, inference_run_id='contract-check')
        self.assertEqual([p.base.uncertainty_ref for p in compiled.proposals],
                         ['registered-uncertainty-17'] * 2)

    def test_explicit_new_shared_uncertainty_is_one_local_identity(self):
        row, _ = self.example('v16-seed-18')
        first = row['target']['proposals'][0]
        local = first['proposal']['uncertainty_ref']
        second = deepcopy(first)
        second['proposal'].update(proposal_id='second-proposal', value_ref='second-value',
                                 target_refs=[local])
        row['target']['proposals'].append(second)
        example = learning_example_v16_from_record(row, split=row['split'])
        decision = self.interface().decision_record(example, local_uncertainty_refs=(local,))
        self.assertEqual(decision['proposals'][0]['proposal']['uncertainty_ref'], {'local_uncertainty': 1})
        compiled = self.interface().compile_decision(example.context, decision,
            artifact_sha256='a' * 64, inference_run_id='contract-check')
        self.assertEqual(compiled.proposals[0].base.uncertainty_ref, compiled.proposals[1].base.uncertainty_ref)
        self.assertEqual(compiled.proposals[1].base.target_refs, (compiled.proposals[0].base.uncertainty_ref,))

    def test_compilation_preserves_all_semantics_external_refs_and_uncertainty(self):
        interface = self.interface()
        for spec in CASES:
            row, example = self.example(spec['case_id'])
            decision = interface.decision_record(example)
            compiled = interface.compile_decision(example.context, decision,
                artifact_sha256='a' * 64, inference_run_id='contract-check')
            # Compare the complete decision again, not just a count or schema.
            self.assertEqual(interface.decision_from_bundle(compiled), decision)
            interface.validate_formation_grounding_v16(example.context, compiled, example.opened_sources)
            for original, actual in zip(example.target.proposals, compiled.proposals):
                self.assertEqual(actual.base.contradicts, original.base.contradicts)
                self.assertEqual(actual.base.target_refs, original.base.target_refs)
                self.assertEqual(actual.base.uncertainty_ref is not None, original.base.uncertainty_ref is not None)

    def test_compiler_never_repairs_bad_evidence_subject_fields_or_extra_keys(self):
        interface = self.interface()
        _, example = self.example('v16-seed-02')
        original = interface.decision_record(example)
        for mutate in (
            lambda d: d.update(permission_to_write=True),
            lambda d: d['proposals'][0]['proposal'].update(proposal_id='model-assigned-id'),
            lambda d: d['proposals'][0]['proposal'].update(evidence_refs=['foreign-source']),
            lambda d: d['proposals'][0].update(uncertainty_requested='yes'),
        ):
            changed = deepcopy(original)
            mutate(changed)
            with self.subTest(value=changed), self.assertRaises((CognitiveKernelContractError, ValueError)):
                bundle = interface.compile_decision(example.context, changed,
                    artifact_sha256='a' * 64, inference_run_id='contract-check')
                interface.validate_formation_grounding_v16(example.context, bundle, example.opened_sources)

    def test_memory_snapshot_supplies_missing_refs_without_reading_target_labels(self):
        from scripts.mfm import v16_prior_memory_inputs as inputs
        row, example = self.example('v16-seed-06')
        extended = inputs.augment_public_seed_example(example)
        self.assertEqual(example.target.output_record(), extended.target.output_record())
        self.assertNotEqual(model_input_sha256_v16(example), model_input_sha256_v16(extended))
        snapshot = extended.opened_sources[-1][1]
        self.assertIn(b'calendar-meeting-claim', snapshot)
        self.assertIn(b'calendar-location-claim', snapshot)  # alternative pointer, not a sole-answer cue
        self.assertNotIn(b'correction_request', snapshot)
        self.assertNotIn(b'expected_dispositions', snapshot)
        changed = deepcopy(row)
        changed['target']['proposals'][0]['proposal']['value_text'] = 'A different teacher paraphrase.'
        alternative = learning_example_v16_from_record(changed, split=changed['split'])
        self.assertEqual(inputs.augment_public_seed_example(alternative).opened_sources[-1], extended.opened_sources[-1])

    def test_prior_memory_is_scoped_registered_input_not_a_new_experience_or_authority(self):
        from scripts.mfm import v16_prior_memory_inputs as inputs
        for case_id in ('v16-seed-06', 'v16-seed-11', 'v16-seed-18'):
            _, example = self.example(case_id)
            changed = inputs.augment_public_seed_example(example)
            changed.validate()
            self.assertEqual(example.context.base.experience_refs, changed.context.base.experience_refs)
            self.assertEqual(changed.context.base.evidence[-1].scope, example.context.base.scope)
            self.assertEqual(changed.context.base.evidence[-1].role, 'tool_or_action_observation')
            self.assertEqual(changed.context.base.evidence[-1].modality, 'structured')
            self.assertNotIn(changed.context.base.evidence[-1].ref_id, changed.context.base.experience_refs)
        _, unaffected = self.example('v16-seed-05')
        self.assertIs(inputs.augment_public_seed_example(unaffected), unaffected)


if __name__ == '__main__':
    unittest.main()
