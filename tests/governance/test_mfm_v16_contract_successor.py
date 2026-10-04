import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record
from scripts.mfm.build_v16_authoring_seed import CASES, build_one
from scripts.mfm import prepare_v16_contract_successor as preparation
from scripts.mfm.v16_prior_memory_inputs import augment_public_seed_example


class SuccessorPreparationTests(unittest.TestCase):
    def example(self, case_id):
        row = build_one(next(spec for spec in CASES if spec['case_id'] == case_id))
        return learning_example_v16_from_record(row, split=row['split'])

    def test_unavailable_existing_memory_refs_become_visible_input(self):
        for case_id, expected in (
            ('v16-seed-06', ['calendar-meeting-claim']),
            ('v16-seed-11', ['current-study-preference', 'quiet-study-preference']),
            ('v16-seed-18', ['archive-interview-date']),
        ):
            original = self.example(case_id)
            self.assertEqual(preparation.reference_input_gaps(original), expected)
            self.assertEqual(preparation.reference_input_gaps(augment_public_seed_example(original)), [])
        self.assertEqual(preparation.reference_input_gaps(self.example('v16-seed-05')), [])

    def test_issued_subset_roster_never_requires_an_unadmitted_fixture(self):
        examples = tuple(self.example(c) for c in ('v16-seed-02', 'v16-seed-06', 'v16-seed-11'))
        self.assertEqual(preparation.expected_augmentation_roster(examples), {'v16-seed-06', 'v16-seed-11'})
        self.assertNotIn('v16-seed-18', preparation.expected_augmentation_roster(examples))

    def test_privacy_and_parent_verification_precede_output_or_new_source_reads(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'new-view'
            with patch.object(preparation.diagnostic, 'prepare_diagnostic', side_effect=CognitiveKernelContractError('boundary rejected')):
                with self.assertRaisesRegex(CognitiveKernelContractError, 'boundary rejected'):
                    preparation.prepare(SimpleNamespace(output_dir=output))
            self.assertFalse(output.exists())

    def test_reference_gap_requires_complete_identity_and_includes_uncertainty(self):
        example = self.example('v16-seed-06')
        from scripts.mfm.v16_prior_memory_inputs import attach_prior_memory_snapshot
        changed = attach_prior_memory_snapshot(example, (('calendar-meeting-claim-archived', 'Other record.'),))
        self.assertEqual(preparation.reference_input_gaps(changed), ['calendar-meeting-claim'])
        row = build_one(next(s for s in CASES if s['case_id'] == 'v16-seed-18'))
        row['target']['proposals'][0]['proposal']['uncertainty_ref'] = 'existing-uncertainty-17'
        changed = augment_public_seed_example(learning_example_v16_from_record(row, split=row['split']))
        self.assertEqual(preparation.reference_input_gaps(changed), ['existing-uncertainty-17'])

    def test_source_runtime_must_match_parent_and_record_actual_placement(self):
        export = {'torch_version': '2.7.1+cu118', 'transformers_version': '5.17.0'}
        actual = preparation.verify_source_runtime(export, '2.7.1+cu118', '5.17.0')
        self.assertEqual(actual['source_dtype'], 'bfloat16')
        self.assertEqual(actual['source_device'], 'cpu')
        with self.assertRaises(CognitiveKernelContractError):
            preparation.verify_source_runtime(export, '2.14.1+cpu', '5.17.0')

    def test_literal_identity_audit_does_not_decode_native_binary_sources(self):
        from dataclasses import replace
        from hashlib import sha256
        from scripts.mfm.v16_prior_memory_inputs import attach_prior_memory_snapshot
        from cognitive_kernel.formation_semantics_v16 import bundle_v16_from_output
        from base64 import b64decode
        example = attach_prior_memory_snapshot(self.example('v16-seed-06'), (('unrelated-record', 'Other.'),))
        raw = b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7S0AAAAASUVORK5CYII=')
        evidence = replace(example.context.base.evidence[-1], modality='image', content_digest=sha256(raw).hexdigest())
        context = replace(example.context, base=replace(example.context.base,
            evidence=(*example.context.base.evidence[:-1], evidence)))
        target = bundle_v16_from_output(context, example.target.output_record(),
            artifact_sha256=example.target.base.model_artifact_digest,
            inference_run_id=example.target.base.inference_run_id)
        changed = replace(example, context=context, target=target,
            opened_sources=(*example.opened_sources[:-1], (evidence.ref_id, raw)))
        changed.validate()
        self.assertEqual(preparation.reference_input_gaps(changed), ['calendar-meeting-claim'])

    def test_full_decision_shift_roundtrip_and_overflow(self):
        class CharacterTokenizer:
            bos_token_id, eos_token_id = 2, 1
            def encode(self, text, **kwargs):
                return [ord(char) + 3 for char in text]
            def decode(self, ids, **kwargs):
                return ''.join(chr(value - 3) for value in ids)
        example = self.example('v16-seed-18')
        inputs, labels = preparation.target_ids(CharacterTokenizer(), example, 16384)
        self.assertEqual(inputs[0], 2)
        self.assertEqual(labels[-1], 1)
        self.assertEqual(inputs[1:], labels[:-1])
        with self.assertRaisesRegex(CognitiveKernelContractError, 'never truncate'):
            preparation.target_ids(CharacterTokenizer(), example, len(inputs) - 1)


if __name__ == '__main__':
    unittest.main()
