"""Public semantic/custody controls for the isolated Magnolia comparison."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.eipm.gemma_n0 import magnolia_paired_update_comparison_20261020 as api


class PairedUpdateWorkerTests(unittest.TestCase):
    def test_public_progress_has_no_per_case_or_per_document_identifiers(self):
        import ast
        from pathlib import Path
        source = ast.parse(Path(api.__file__).read_text(encoding='utf-8'))
        forbidden = {'case_id', 'document_id', 'batch_sha256', 'case_sha256', 'document_sha256'}
        progress_records = []
        for call in ast.walk(source):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'print':
                for value in ast.walk(call):
                    if isinstance(value, ast.Dict):
                        fields = {key.value for key in value.keys if isinstance(key, ast.Constant)}
                        self.assertFalse(fields & forbidden)
                        if 'stage' in fields:
                            progress_records.append(fields)
        self.assertGreater(len(progress_records), 5)

    def test_guard_refusal_precedes_any_path_access(self):
        class Refused(Exception):
            pass
        class Namespace:
            def _p2_guard(self):
                raise Refused()
            def _bounded_path(self, *args, **kwargs):
                raise AssertionError('access before owner namespace guard')
        with patch.object(api.Path, 'stat', side_effect=AssertionError('stat before guard')), \
                patch.object(api.Path, 'open', side_effect=AssertionError('open before guard')):
            with self.assertRaises(Refused):
                api.run(Namespace())

    def pair(self):
        from tests.personality.test_identity_codec import document
        from src.alice_personality.identity.codec import SourceEntry
        from src.alice_personality.identity.contracts import SourceRecord
        from src.alice_personality.teaching.cases import ReviewedTeachingCase, NamedTarget, TargetFact, CASE_LINEAGE_SCHEMA
        from src.alice_personality.teaching.lineage import ConstructionLineage
        doc = document()
        doc = replace(doc, candidates=tuple(replace(c, authority_allowed=True) for c in doc.candidates))
        pins = {name: sha256(name.encode()).hexdigest() for name in
                ('source_package', 'compiled_receipt', 'prepared_receipt', 'annotation_protocol', 'review_record')}
        cases = []
        for branch in (0, 1):
            family = f'context-{branch}'
            source = SourceEntry(f'Public hypothetical condition {branch}.',
                SourceRecord(f'host-{branch}', family, 'HOST', 'HOST', False, True, False, False, False, False))
            current = replace(doc, sources=doc.sources+(source,),
                fields=tuple(replace(f, text=f'Public fictional condition {branch}.') if f.semantic_role == 'situation'
                             else f for f in doc.fields),
                candidates=doc.candidates if branch == 0 else tuple(reversed(doc.candidates)))
            fact = lambda pointer, value: TargetFact((pointer,), value, 0. if value is not None else None,
                                                    value is not None, (), ())
            a, b = [entry.candidate_id for entry in doc.candidates]
            targets = (NamedTarget('preferences', 'closed_distribution',
                                  (fact(a, 1.-branch), fact(b, float(branch)))),
                       NamedTarget('co_valid_probabilities', 'independent',
                                   (fact(a, 1. if branch == 0 else None), fact(b, 1. if branch == 1 else None))),
                       NamedTarget('contraindications', 'independent', (fact(a, None), fact(b, None))))
            lineage = ConstructionLineage((doc.sources[0].record.source_family_id,), (family,), (f'episode-{branch}',))
            case = ReviewedTeachingCase(f'case-{branch}', 'n2', 'TRAIN', 'governed_identity_targets', current,
                tuple(entry.record.source_family_id for entry in current.sources), pins, 'reviewed_source_case',
                'public-syntactic-test-not-admission', 'Public semantic pairing fixture only.', targets,
                CASE_LINEAGE_SCHEMA, lineage)
            case.validate()
            cases.append(case)
        allocation = {'supervision_parents': {'episode-0': [], 'episode-1': ['episode-0']}}
        return cases, allocation

    def test_pair_uses_semantics_and_reviewed_ancestry_despite_reversed_presentation(self):
        cases, allocation = self.pair()
        self.assertEqual(api.source_pairs(cases, allocation, expected_pairs=1), [(cases[0], cases[1])])
        self.assertNotEqual(cases[0].document.candidates, cases[1].document.candidates)

    def test_matching_identifiers_cannot_hide_changed_candidate_meaning(self):
        cases, allocation = self.pair()
        child = cases[1]
        changed = replace(child.document.candidates[0], text='A different public proposed action.')
        cases[1] = replace(child, document=replace(child.document,
                                                  candidates=(changed, *child.document.candidates[1:])))
        with self.assertRaisesRegex(ValueError, 'candidate meanings'):
            api.source_pairs(cases, allocation, expected_pairs=1)

    def test_matching_candidates_cannot_hide_changed_source_meaning_or_ancestry(self):
        cases, allocation = self.pair()
        child = cases[1]
        changed = replace(child.document.sources[0], text='The source now asserts a different policy.')
        cases[1] = replace(child, document=replace(child.document, sources=(changed, *child.document.sources[1:])))
        with self.assertRaisesRegex(ValueError, 'source meanings'):
            api.source_pairs(cases, allocation, expected_pairs=1)
        cases, allocation = self.pair()
        allocation['supervision_parents']['episode-1'] = []
        with self.assertRaisesRegex(ValueError, 'coverage'):
            api.source_pairs(cases, allocation, expected_pairs=1)

    def test_group_control_retains_every_occurrence_and_each_endpoint_weight(self):
        cases = lambda prefix, n: [SimpleNamespace(case_id=f'{prefix}-{i}', sha256=f'{i:064x}') for i in range(n)]
        broader = cases('broader', 80)
        remainder = {name: cases(name, size) for name, size in (('conditional', 80), ('original', 40), ('voice', 16))}
        lookup = {(name, case.case_id): sha256((name+case.case_id).encode()).hexdigest()
                  for name, members in [('broader', broader), *remainder.items()] for case in members}
        groups, permutation = api.matched_group_geometry(list(zip(broader[::2], broader[1::2])), remainder, lookup)
        self.assertEqual([len(g) for g in groups], [6]*16+[5]*24)
        control = [group[:1]+[groups[permutation[i]][1]]+group[2:] for i, group in enumerate(groups)]
        weights = lambda rows: {item: 1./len(group) for group in rows for item in group}
        self.assertEqual(weights(groups), weights(control))
        self.assertEqual(len(weights(groups)), 216)
        self.assertTrue(all(group[1] != control[i][1] for i, group in enumerate(groups)))
        self.assertTrue(all(group[2:] == control[i][2:] for i, group in enumerate(groups)))

    def test_paired_correctness_preserves_unknowns_and_tests_preference_direction(self):
        cases, _ = self.pair()
        a, b = [entry.candidate_id for entry in cases[0].document.candidates]
        def decision(ids, preference, co):
            return {'candidate_ids': [ids], 'preferences': [preference],
                    'co_valid_probabilities': [co], 'contraindications': [[.5, .5]]}
        baseline = {c.case_id: decision([a, b], [.5, .5], [.4, .4]) for c in cases}
        candidate = {cases[0].case_id: decision([a, b], [.8, .2], [.8, .1]),
                     cases[1].case_id: decision([b, a], [.8, .2], [.8, .1])}
        result = api.paired_region_comparison([(cases[0], cases[1])], baseline, candidate)['fields']
        self.assertEqual(result['co_valid_probabilities']['available_binary_facts'], 2)
        self.assertEqual(result['co_valid_probabilities']['repairs'], 2)
        self.assertEqual(result['contraindications'].get('available_binary_facts', 0), 0)
        self.assertEqual(result['preferences']['after_both_contexts_correct'], 1)
        self.assertEqual(result['preferences']['pair_repairs'], 1)
        self.assertGreater(result['preferences']['after_mean_target_signed_condition_delta'], 0)

    def test_actual_native_capture_refuses_packet_mismatch_and_cleans_hook(self):
        import torch
        from src.alice_personality.identity import IdentityModel, IdentityModelConfig
        from src.alice_personality.identity.codec import fingerprint
        from tests.personality.test_identity_teaching import _frame
        torch.set_num_threads(1)
        model = IdentityModel(IdentityModelConfig(4, 3, learned_width=8, attention_heads=2))
        frame = _frame(model.config)
        before = fingerprint(model.state_dict())
        hooks = len(model.n2._forward_hooks)
        packet, decision, scalars, digest = api.capture_candidate(model, frame)
        self.assertEqual(digest, fingerprint(packet))
        self.assertEqual(len(scalars[0][0]), 6)
        self.assertIn('native_latent', decision)
        with self.assertRaisesRegex(ValueError, 'reproduction'):
            api.capture_candidate(model, frame, expected_packet_sha256='0'*64)
        self.assertEqual(len(model.n2._forward_hooks), hooks)
        self.assertEqual(before, fingerprint(model.state_dict()))

    def test_exact_export_readback_preserves_adamw_integer_keys_and_tensors(self):
        import io
        import torch
        from src.alice_personality.identity.codec import fingerprint
        from src.alice_personality.identity.contracts import IdentityError
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.AdamW([parameter], lr=.0003)
        parameter.square().mean().backward()
        optimizer.step()
        state = {'optimizer_state': optimizer.state_dict(), 'model_state': {'weight': parameter.detach().clone()},
                 'qualification': 'UNQUALIFIED', 'native_checkpoint': False}
        # Original mismatch remains explicit: the packet fingerprint codec is
        # correctly strict about string keys; optimizer storage has integer IDs.
        with self.assertRaisesRegex(IdentityError, 'string keys'):
            fingerprint(state)
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        api.verify_exported_state(state, restored)
        restored['optimizer_state']['state'][0]['exp_avg'].add_(.01)
        with self.assertRaisesRegex(ValueError, 'tensor changed'):
            api.verify_exported_state(state, restored)

    def test_binary_joint_correctness_requires_observations_in_both_contexts(self):
        cases, _ = self.pair()
        from src.alice_personality.teaching.cases import TargetFact
        unknown = lambda fact: TargetFact(fact.pointers, None, None, False, (), ())
        cases[1] = replace(cases[1], targets=tuple(replace(target, facts=tuple(map(unknown, target.facts)))
                           if target.field == 'co_valid_probabilities' else target for target in cases[1].targets))
        a, b = [entry.candidate_id for entry in cases[0].document.candidates]
        predictions = {c.case_id: {'candidate_ids': [[a, b]], 'preferences': [[.5, .5]],
                                  'co_valid_probabilities': [[.9, .9]], 'contraindications': [[.5, .5]]} for c in cases}
        result = api.paired_region_comparison([(cases[0], cases[1])], predictions, predictions)['fields']
        self.assertEqual(result['co_valid_probabilities']['pairs_with_available_facts_in_both_contexts'], 0)
        self.assertEqual(result['co_valid_probabilities']['after_both_contexts_correct'], 0)

    def test_actual_two_arm_export_is_loadable_and_complete(self):
        from pathlib import Path
        import tempfile
        import torch
        from tests.personality.test_paired_learning_experiment import PairedLearningExperimentTests
        torch.set_num_threads(1)
        fixture = PairedLearningExperimentTests()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory).resolve()/'experimental'
            result = fixture.run_trial(destination=destination)
            restored = torch.load(destination/'experimental.pt', weights_only=True)
            api.verify_exported_state(result, restored)
            self.assertEqual(restored['recipe']['phase'], 'n2')
            self.assertFalse(restored['resumable'])
            self.assertEqual(set(restored['arms']), {'matched', 'control'})
            self.assertFalse((destination/'resume.pt').exists())

    def test_native_production_validator_checks_actual_geometry(self):
        from tests.personality.test_identity_teaching import _frame, _production_fixture
        from src.alice_personality.identity import IdentityModelConfig
        from src.alice_personality.teaching.trainer import _production
        from src.alice_personality.teaching import TeachingError
        frame = _frame(IdentityModelConfig(4, 3, learned_width=8, attention_heads=2))
        binding, session, kwargs = _production_fixture(frame)
        _production(frame, binding, session, **kwargs)
        binding['source_geometry'] = [49, 3840]
        binding['binding_sha256'] = sha256(api.canonical({k: v for k, v in binding.items() if k != 'binding_sha256'})).hexdigest()
        session['frame_binding_sha256'] = [binding['binding_sha256']]
        session['receipt_sha256'] = sha256(api.canonical({k: v for k, v in session.items() if k != 'receipt_sha256'})).hexdigest()
        with self.assertRaisesRegex(TeachingError, 'exact frame provenance'):
            _production(frame, binding, session, **kwargs)


if __name__ == '__main__':
    unittest.main()
