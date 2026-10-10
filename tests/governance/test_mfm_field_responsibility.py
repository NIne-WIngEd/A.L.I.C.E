"""Declared responsibilities and exact segment joins; no inferred teacher labels."""
from copy import deepcopy
import importlib
import importlib.util
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_sha256
from cognitive_kernel import formation_decision_v16 as decision
from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record
from scripts.mfm.formation_caller_input import selection_candidates
from tests.governance import test_mfm_target_alignment as target_tests
from tests.governance import test_mfm_caller_input as caller_tests
from tests.governance import test_mfm_v16_decision_interface as decision_tests
reseal = target_tests.reseal


class FieldResponsibilityTests(unittest.TestCase):
    def api(self):
        name = 'scripts.mfm.formation_field_responsibility'
        self.assertIsNotNone(importlib.util.find_spec(name), 'exhaustive normative field responsibility is not implemented')
        return importlib.import_module(name)

    def built(self, **kw):
        api = self.api()
        parts = target_tests.ActualTargetTests().built(**kw)
        _, _, caller, _, _, _, _, _, _, target = parts
        policy = api.normative_policy()
        kwargs = {'target_alignment_sha256': target['record_sha256'], 'caller_sha256': caller['record_sha256'],
                  'policy': policy, 'policy_sha256': policy['record_sha256']}
        result = api.field_responsibilities(target, caller, **kwargs)
        return api, parts, policy, kwargs, result

    def authored(self, target, caller, policy, entries):
        return reseal({'schema': 'mfm-authored-field-responsibilities-v1',
            'target_alignment_sha256': target['record_sha256'], 'target_sha256': target['target_sha256'],
            'caller_sha256': caller['record_sha256'], 'policy_sha256': policy['record_sha256'],
            'author': 'public-fixture-author', 'provenance': 'explicit-public-construction-control', 'entries': entries})

    def declaration(self, target, caller, source, entries):
        return reseal({'schema': 'mfm-authored-literal-segments-v1',
            'target_alignment_sha256': target['record_sha256'], 'target_sha256': target['target_sha256'],
            'caller_sha256': caller['record_sha256'], 'source_alignment_sha256': source['record_sha256'],
            'author': 'public-fixture-author', 'provenance': 'explicit-public-construction-control', 'entries': entries})

    def test_exhaustive_fields_protocol_and_unannotated_semantics_are_separate(self):
        _, parts, _, _, result = self.built()
        target = parts[-1]; rows = {r['pointer']: r for r in result['fields']}
        self.assertEqual(set(rows), {n['pointer'] for n in target['geometry']['nodes']})
        prefix = '/proposals/0/proposal/'
        for field in ('kind', 'domain', 'epistemic_status', 'value_text', 'valid_from', 'valid_to',
                      'confidence', 'temporal_granularity', 'disposition_scope_ref'):
            self.assertEqual(rows[prefix + field]['responsibility'], 'semantic_authoring')
        self.assertEqual(rows[prefix + 'proposal_id']['responsibility'], 'local_bookkeeping')
        self.assertEqual(rows[prefix + 'value_ref']['responsibility'], 'local_bookkeeping')
        self.assertEqual(rows[prefix + 'anchors/0/start_byte']['responsibility'], 'source_coordinates')
        self.assertEqual(rows[prefix + 'anchors/0/locator']['responsibility'], 'native_locator')
        self.assertEqual(rows['/dispositions/0/scope_ref']['responsibility'], 'semantic_authoring')
        self.assertEqual(rows['']['responsibility'], 'protocol_structure')
        self.assertTrue(all(r['annotation_status'] == 'unannotated' for r in result['fields']))
        self.assertFalse(result['authority_granted']); self.assertFalse(result['learning_qualified'])
        self.assertEqual(result['protocol_lexemes'], target['geometry']['lexemes'])

    def test_policy_cannot_promote_scopes_native_locators_subjects_or_unknown_types(self):
        api, parts, policy, kwargs, _ = self.built()
        target, caller = parts[-1], parts[2]
        for mutate in (lambda row: row['rules'][0].update(responsibility='external_selection'),
                       lambda row: row.update(contract_version='1.7.0'),
                       lambda row: row['contract_binding'].update(caller_sha256='0' * 64),
                       lambda row: row['rules'].append({'path': ['invented'], 'kinds': ['string'], 'responsibility': 'external_selection', 'candidate_path': 'invented'})):
            changed = deepcopy(policy); mutate(changed); reseal(changed)
            with self.subTest(mutate=mutate), self.assertRaises(CognitiveKernelContractError):
                api.field_responsibilities(target, caller, **{**kwargs, 'policy': changed, 'policy_sha256': changed['record_sha256']})
        with self.assertRaises(CognitiveKernelContractError):
            api.field_responsibilities(target, caller, **{**kwargs, 'caller_sha256': '0' * 64})

    def test_same_id_subject_candidates_remain_ambiguous_without_authored_choice(self):
        api, parts, policy, kwargs, result = self.built()
        target, caller = parts[-1], parts[2]
        subject = next(r for r in result['fields'] if r['pointer'] == '/proposals/0/proposal/subject_ref')
        self.assertEqual(subject['candidate_status'], 'ambiguous')
        self.assertEqual(len(subject['matching_candidate_indices']), 2)
        self.assertNotIn('selected_candidate_index', subject)
        participant = next(r for r in result['fields'] if r['pointer'] == '/proposals/0/episode/participant_refs/0')
        self.assertEqual(len(participant['matching_candidate_indices']), 1)
        self.assertTrue(set(participant['eligible_candidate_indices']) < set(subject['eligible_candidate_indices']))
        choice = subject['matching_candidate_indices'][1]
        annotation = self.authored(target, caller, policy, [{'pointer': subject['pointer'], 'mode': 'external_selection', 'candidate_index': choice}])
        bound = api.bind_authored_responsibilities(target, caller, annotations=annotation,
            annotations_sha256=annotation['record_sha256'], **kwargs)
        chosen = next(r for r in bound['fields'] if r['pointer'] == subject['pointer'])
        self.assertEqual(chosen['selected_candidate_index'], choice)
        self.assertEqual(chosen['matching_candidate_indices'], subject['matching_candidate_indices'])
        bad = self.authored(target, caller, policy, [{'pointer': participant['pointer'], 'mode': 'external_selection',
            'candidate_index': subject['matching_candidate_indices'][0]}])
        with self.assertRaises(CognitiveKernelContractError):
            api.bind_authored_responsibilities(target, caller, annotations=bad, annotations_sha256=bad['record_sha256'], **kwargs)

    def test_authored_provenance_pins_modes_missing_candidates_and_duplicates_refuse(self):
        api, parts, policy, kwargs, result = self.built()
        target, caller = parts[-1], parts[2]; pointer = '/proposals/0/proposal/value_text'
        good = self.authored(target, caller, policy, [{'pointer': pointer, 'mode': 'semantic_authoring'}])
        bound = api.bind_authored_responsibilities(target, caller, annotations=good, annotations_sha256=good['record_sha256'], **kwargs)
        self.assertEqual(sum(r['annotation_status'] == 'authored' for r in bound['fields']), 1)
        changes = (lambda r: r.update(author=''), lambda r: r.update(target_sha256='0' * 64),
                   lambda r: r.update(provenance=''), lambda r: r['entries'].append(deepcopy(r['entries'][0])),
                   lambda r: r['entries'][0].update(mode='external_selection', candidate_index=0),
                   lambda r: r['entries'][0].update(pointer='/proposals/0/proposal/uncertainty_ref', mode='external_selection', candidate_index=0))
        for change in changes:
            bad = deepcopy(good); change(bad); reseal(bad)
            with self.subTest(change=change), self.assertRaises(CognitiveKernelContractError):
                api.bind_authored_responsibilities(target, caller, annotations=bad, annotations_sha256=bad['record_sha256'], **kwargs)
        self.assertFalse(any('selected_candidate_index' in r for r in result['fields']))

    def test_input_registry_provenance_never_becomes_citation_or_scope(self):
        api = self.api(); helper = caller_tests.CallerInputTests(); context, opened = helper.fixture()
        rows = selection_candidates(context, opened, helper.registry(context))
        self.assertFalse(any(r['kind'] == 'scope' for r in rows))
        for row in rows:
            if row['origin'] == 'readonly_registry':
                self.assertFalse(row['citation_eligible']); self.assertFalse(row['anchor_eligible'])
                self.assertEqual(row['provenance']['generation'], 'snapshot-07')
        policy = api.normative_policy()
        for rule in policy['rules']:
            if rule['path'][-1:] in (['scope_ref'], ['disposition_scope_ref'], ['locator']):
                self.assertIsNone(rule['candidate_path'])

    def test_declared_utf8_literal_keeps_all_repeated_and_overlapping_occurrences(self):
        api, parts, _, _, _ = self.built(text='Café "A" \\ 😀 banana banana', value='Café "A" \\ 😀 banana')
        caller, sidecar, target = parts[2], parts[5], parts[-1]
        value = 'Café "A" \\ 😀 banana'; raw = value.encode(); begin = len('Café "A" \\ 😀 '.encode())
        declaration = self.declaration(target, caller, sidecar,
            [{'pointer': '/proposals/0/proposal/value_text', 'target_value_utf8': [begin, len(raw)], 'source_occurrences': None}])
        joined = api.align_declared_literal_segments(caller, sidecar, target, declaration,
            caller_sha256=caller['record_sha256'], source_alignment_sha256=sidecar['record_sha256'],
            target_alignment_sha256=target['record_sha256'], declarations_sha256=declaration['record_sha256'])
        segment = joined['segments'][0]
        self.assertEqual(len(segment['occurrences']), 2); self.assertIsNone(segment['selected_occurrences'])
        self.assertTrue(segment['target_token_links']); self.assertTrue(segment['complete_target_token_coverage'])
        self.assertEqual(bytes.fromhex(segment['literal_utf8_hex']), b'banana')
        self.assertFalse(joined['token_id_copy_qualified']); self.assertFalse(joined['semantic_entailment_qualified'])
        api, parts, _, _, _ = self.built(text='banana', value='ana')
        caller, sidecar, target = parts[2], parts[5], parts[-1]
        declaration = self.declaration(target, caller, sidecar,
            [{'pointer': '/proposals/0/proposal/value_text', 'target_value_utf8': [0, 3], 'source_occurrences': None}])
        joined = api.align_declared_literal_segments(caller, sidecar, target, declaration,
            caller_sha256=caller['record_sha256'], source_alignment_sha256=sidecar['record_sha256'],
            target_alignment_sha256=target['record_sha256'], declarations_sha256=declaration['record_sha256'])
        self.assertEqual([r['original_utf8'] for r in joined['segments'][0]['occurrences']], [[1, 4], [3, 6]])

    def test_literal_escape_atoms_mixed_tokens_selected_occurrence_and_bad_boundaries(self):
        api, parts, _, _, _ = self.built(text='é "é" é', value='é "é"')
        caller, sidecar, target = parts[2], parts[5], parts[-1]
        entry = {'pointer': '/proposals/0/proposal/value_text', 'target_value_utf8': [3, 7],
                 'source_occurrences': [{'source_ref': caller['opened_sources'][0]['ref_id'], 'original_utf8': [3, 7]}]}
        declaration = self.declaration(target, caller, sidecar, [entry])
        kw = {'caller_sha256': caller['record_sha256'], 'source_alignment_sha256': sidecar['record_sha256'],
              'target_alignment_sha256': target['record_sha256']}
        joined = api.align_declared_literal_segments(caller, sidecar, target, declaration,
            declarations_sha256=declaration['record_sha256'], **kw)
        segment = joined['segments'][0]
        self.assertEqual(len(segment['target_characters']), 3)
        self.assertEqual(segment['selected_occurrences'], entry['source_occurrences'])
        api, mixed_parts, _, _, _ = self.built(text='ab ab', value='ab', merge_pair=('a', 'b'))
        mc, ms, mt = mixed_parts[2], mixed_parts[5], mixed_parts[-1]
        md = self.declaration(mt, mc, ms, [{'pointer': '/proposals/0/proposal/value_text',
            'target_value_utf8': [0, 1], 'source_occurrences': None}])
        mj = api.align_declared_literal_segments(mc, ms, mt, md, caller_sha256=mc['record_sha256'],
            source_alignment_sha256=ms['record_sha256'], target_alignment_sha256=mt['record_sha256'], declarations_sha256=md['record_sha256'])
        self.assertTrue(any(not r['fully_contained_in_segment'] for r in mj['segments'][0]['target_token_links']))
        for change in (lambda r: r['entries'][0].update(target_value_utf8=[1, 7]),
                       lambda r: r['entries'][0].update(pointer='/proposals/0/proposal/anchors/0/locator'),
                       lambda r: r['entries'][0]['source_occurrences'][0].update(original_utf8=[0, 7]),
                       lambda r: r.update(source_alignment_sha256='0' * 64)):
            bad = deepcopy(declaration); change(bad); reseal(bad)
            with self.subTest(change=change), self.assertRaises(CognitiveKernelContractError):
                api.align_declared_literal_segments(caller, sidecar, target, bad, declarations_sha256=bad['record_sha256'], **kw)

    def test_template_only_and_absent_literal_are_not_authored_source_copy(self):
        api, parts, _, _, _ = self.built(text='Only observed text.', value='json_text=')
        caller, sidecar, target = parts[2], parts[5], parts[-1]
        declaration = self.declaration(target, caller, sidecar,
            [{'pointer': '/proposals/0/proposal/value_text', 'target_value_utf8': [0, 10], 'source_occurrences': None}])
        with self.assertRaises(CognitiveKernelContractError):
            api.align_declared_literal_segments(caller, sidecar, target, declaration,
                caller_sha256=caller['record_sha256'], source_alignment_sha256=sidecar['record_sha256'],
                target_alignment_sha256=target['record_sha256'], declarations_sha256=declaration['record_sha256'])

    def test_original_output_and_explicit_shared_local_decision_remain_distinct(self):
        self.api(); helper = decision_tests.DecisionInterfaceTests()
        row, _ = helper.example('v16-seed-18'); first = row['target']['proposals'][0]
        local = first['proposal']['uncertainty_ref']; second = deepcopy(first)
        second['proposal'].update(proposal_id='second-proposal', value_ref='second-value', target_refs=[local])
        row['target']['proposals'].append(second)
        example = learning_example_v16_from_record(row, split=row['split'])
        wire = decision.decision_record(example, local_uncertainty_refs=(local,))
        compiled = decision.compile_decision(example.context, wire, artifact_sha256='a' * 64, inference_run_id='field-control')
        newlocal = compiled.proposals[0].base.uncertainty_ref
        self.assertEqual(decision.decision_from_bundle(compiled, local_uncertainty_refs=(newlocal,)), wire)
        self.assertEqual(compiled.proposals[1].base.target_refs, (newlocal,))
        decision.validate_formation_grounding_v16(example.context, compiled, example.opened_sources)
        for change in (lambda r: r.update(permission_to_write=True),
                       lambda r: r['proposals'][0]['proposal'].update(uncertainty_ref={'local_uncertainty': 3}),
                       lambda r: r['proposals'][1]['proposal'].update(target_refs=[{'local_value': 3}])):
            bad = deepcopy(wire); change(bad)
            with self.assertRaises(CognitiveKernelContractError):
                decision.compile_decision(example.context, bad, artifact_sha256='a' * 64, inference_run_id='field-control')


if __name__ == '__main__':
    unittest.main()
