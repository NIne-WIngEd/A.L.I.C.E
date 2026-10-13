"""Complete synthetic reference-output controls; no training or learned claim."""
from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.util
import unittest

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, CognitiveKernelContractError
from cognitive_kernel import formation_decision_v16 as old
from cognitive_kernel import formation_learning_v16 as L
from cognitive_kernel.formation_contracts import FormationProposal, FormationEvidenceAnchor
from cognitive_kernel.formation_semantics_v16 import FormationProposalV16
from scripts.mfm import formation_joint_input_v4 as J, formation_source_alignment as S
from scripts.mfm import formation_paired_teaching as P, formation_joint_target_alignment as G
from scripts.mfm import formation_target_alignment as T
from tests.governance.test_mfm_world_caller_v3 import build
from tests.governance.test_mfm_joint_input import source_identity
from tests.governance.test_mfm_paired_teaching import split, seal
from tests.governance.test_mfm_source_alignment import Processor


class NativeProcessor(Processor):
    """Real small ByteLevel tokenizer with the native BOS2/EOS1 convention."""
    def __init__(self):
        super().__init__()
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        vocab = {char: i + 3 for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        vocab.update({'<pad>': 0, '</s>': 1, '<s>': 2, '<unk>': 259})
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>')


class ReferenceDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, _, _, _, cls.bundle = build()
        cls.processor = NativeProcessor()
        identity = source_identity()
        policy = J.normative_joint_render_policy_v4()
        caller, meaning = cls.bundle['caller'], cls.bundle['meaning']
        batch, cls.joint = J.joint_source_batch_with_alignment_v4(cls.processor, caller, meaning,
            caller_sha256=caller['record_sha256'], meaning_sha256=meaning['record_sha256'],
            render_policy=policy, render_policy_sha256=policy['record_sha256'],
            processor_binding=S.processor_binding(cls.processor), source_identity=identity,
            source_identity_sha256=identity['record_sha256'], max_source_tokens=200000)
        cls.expected = dict(caller_sha256=caller['record_sha256'], meaning_sha256=meaning['record_sha256'],
            joint_sha256=cls.joint['record_sha256'], batch_sha256=canonical_sha256({k: batch[k].tolist() for k in ('input_ids', 'attention_mask')}),
            joint_code_binding=J.joint_code_binding_v4(), processor_binding=S.processor_binding(cls.processor),
            source_identity=identity, max_source_tokens=200000)
        cls.frozen = P.freeze_input(caller, meaning, cls.joint,
            {k: batch[k].tolist() for k in ('input_ids', 'attention_mask')}, expected=cls.expected, split_row=split())
        cls.binding = {'caller_sha256': caller['record_sha256'],
            'candidate_inventory_sha256': cls.joint['candidate_inventory_sha256'], **cls.joint['cache']}
        cls._pair_cache = {}

    def api(self):
        names = ('cognitive_kernel.formation_reference_decision', 'scripts.mfm.formation_reference_target_alignment')
        for name in names:
            self.assertIsNotNone(importlib.util.find_spec(name), 'native reference output path is missing')
        return tuple(importlib.import_module(name) for name in names)

    def inventory(self, C):
        return C.bind_reference_inventory(self.bundle['caller'], self.joint,
            caller_sha256=self.binding['caller_sha256'], joint_alignment_sha256=self.joint['record_sha256'],
            expected_binding=self.binding)

    def complete_pair(self, *, local_uncertainty=False):
        if local_uncertainty in self._pair_cache:
            return deepcopy(self._pair_cache[local_uncertainty])
        context, opened, _ = J.caller_api.caller_input_from_record(self.bundle['caller'])
        candidates = self.bundle['meaning']['candidate_inventory']
        # Explicit fixture-author choices from inputs, not inferred from a target.
        evidence = next(i for i, c in enumerate(candidates) if c['kind'] == 'evidence' and c['citation_eligible'])
        subject = next(i for i, c in enumerate(candidates) if c['kind'] == 'source_subject')
        mission = next(i for i, c in enumerate(candidates) if 'proposals[].mission_target_refs' in c['eligible_paths'])
        workspace = next(i for i, c in enumerate(candidates) if 'proposals[].workspace_target_refs' in c['eligible_paths'])
        ref = candidates[evidence]['ref_id']; raw = dict(opened)[ref]
        base = FormationProposal('fixture-proposal', 'claim', 'host', candidates[subject]['ref_id'],
            'fixture-value', (ref,), value_text='Café "A" \\ 😀\n' + raw.decode('utf-8'),
            anchors=(FormationEvidenceAnchor(ref, 0, len(raw)),), epistemic_status='hypothetical',
            confidence=0.625, uncertainty_ref='fixture-new-uncertainty' if local_uncertainty else None,
            disposition_scope_ref='fixture-semantic-scope')
        proposal = FormationProposalV16(base, context.source_sensitivities[0].minimum,
            mission_target_refs=(candidates[mission]['ref_id'],), workspace_target_refs=(candidates[workspace]['ref_id'],))
        authored = {'schema': L.TARGET_SCHEMA_V16, 'context': context.record(), 'source_ids': [r for r, _ in opened],
            'proposals': [proposal.record()], 'dispositions': [{'scope_ref': 'fixture-semantic-scope', 'action': 'propose',
                'evidence_refs': [ref], 'target_refs': []}],
            'adjudications': {k: 'present' if k in {'sensitivity', 'mission', 'workspace'} else 'negative' for k in L.FULL_ROLE_DIMENSIONS}}
        choices = {'/proposals/0/proposal/subject_ref': subject,
            '/proposals/0/proposal/evidence_refs/0': evidence, '/proposals/0/proposal/anchors/0/ref_id': evidence,
            '/proposals/0/mission_target_refs/0': mission, '/proposals/0/workspace_target_refs/0': workspace,
            '/dispositions/0/evidence_refs/0': evidence}
        annotations = [{'pointer': pointer, 'mode': 'external_selection', 'candidate_ordinal': index}
                       for pointer, index in choices.items()]
        raw_target = canonical_json_bytes(authored)
        accepted = P.admit_target(self.frozen, expected_freeze_sha256=self.frozen['record_sha256'], expected=self.expected,
            target_reader=lambda: raw_target, expected_target_file_sha256=sha256(raw_target).hexdigest())
        pair = G.align_target(self.processor, self.frozen, accepted, expected=self.expected,
            expected_freeze_sha256=self.frozen['record_sha256'], expected_target_record_sha256=accepted['record_sha256'],
            max_target_tokens=20000, annotations=annotations)
        example = P.admitted(accepted, self.frozen, accepted['record_sha256'])
        self._pair_cache[local_uncertainty] = (example, pair, choices)
        return deepcopy(self._pair_cache[local_uncertainty])

    def view(self, C, A, *, local_uncertainty=False):
        example, pair, choices = self.complete_pair(local_uncertainty=local_uncertainty)
        inventory = self.inventory(C)
        view = A.reference_decision_view(example, pair, self.frozen,
            pair_sha256=pair['record_sha256'], frozen_input_sha256=self.frozen['record_sha256'],
            inventory=inventory, local_uncertainty_refs=('fixture-new-uncertainty',) if local_uncertainty else ())
        return example, pair, choices, inventory, view

    def compile(self, C, example, decision, choices, **kwargs):
        packet = {'binding': deepcopy(self.binding), 'rows': [{'pointer': k, 'candidate_ordinal': v} for k, v in choices.items()]}
        return C.compile_reference_decision(example.context, example.opened_sources, decision, packet,
            caller=self.bundle['caller'], caller_sha256=self.binding['caller_sha256'], joint_alignment=self.joint,
            joint_alignment_sha256=self.joint['record_sha256'], expected_binding=self.binding,
            artifact_sha256='a' * 64, inference_run_id='mechanical-reference-control', **kwargs)

    def test_predicted_references_reach_complete_grounded_v16_and_originals_unchanged(self):
        C, A = self.api(); example, pair, choices = self.complete_pair(); inventory = self.inventory(C)
        original_pair, original_frozen = canonical_json_bytes(pair), canonical_json_bytes(self.frozen)
        view = A.reference_decision_view(example, pair, self.frozen,
            pair_sha256=pair['record_sha256'], frozen_input_sha256=self.frozen['record_sha256'], inventory=inventory)
        compiled = self.compile(C, example, view['decision'], choices)
        expected = old.decision_record(example)
        self.assertEqual(old.decision_from_bundle(compiled), expected)
        self.assertEqual(compiled.proposals[0].base.value_text, example.target.proposals[0].base.value_text)
        self.assertEqual(compiled.proposals[0].base.confidence, 0.625)
        self.assertEqual(compiled.proposals[0].base.uncertainty_ref, None)
        self.assertEqual(compiled.base.dispositions[0].scope_ref, 'fixture-semantic-scope')
        self.assertEqual(canonical_json_bytes(pair), original_pair)
        self.assertEqual(canonical_json_bytes(self.frozen), original_frozen)
        self.assertEqual({r['pointer']: r['candidate_ordinal'] for r in view['authored_ordinals']}, choices)
        self.assertEqual([r['slot'] for r in inventory['candidates']], list(range(len(inventory['candidates']))))

    def test_changed_prediction_changes_only_the_resolved_identity(self):
        C, A = self.api(); example, _, choices, inventory, view = self.view(C, A)
        pointer = '/dispositions/0/evidence_refs/0'
        other = next(i for i, row in enumerate(inventory['candidate_inventory'])
            if 'dispositions[].evidence_refs' in row['eligible_paths'] and i != choices[pointer])
        changed = {**choices, pointer: other}
        compiled = self.compile(C, example, view['decision'], changed)
        expected = old.decision_record(example)
        expected['dispositions'][0]['evidence_refs'] = [inventory['candidate_inventory'][other]['ref_id']]
        self.assertEqual(old.decision_from_bundle(compiled), expected)
        self.assertEqual(compiled.base.dispositions[0].evidence_refs,
            (inventory['candidate_inventory'][other]['ref_id'],))

    def test_missing_or_wrong_role_authored_selection_cannot_be_inferred(self):
        C, A = self.api(); example, pair, _ = self.complete_pair(); inventory = self.inventory(C)
        bad = deepcopy(pair); bad['annotations'] = []
        bad['fields'] = G.field_rows(bad['geometry'], self.frozen, []); bad = seal({k: v for k, v in bad.items() if k != 'record_sha256'})
        with self.assertRaisesRegex(CognitiveKernelContractError, 'authored|annotation'):
            A.reference_decision_view(example, bad, self.frozen, pair_sha256=bad['record_sha256'],
                frozen_input_sha256=self.frozen['record_sha256'], inventory=inventory)
        bad = deepcopy(pair); bad['annotations'][0]['candidate_ordinal'] = True
        bad = seal({k: v for k, v in bad.items() if k != 'record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):
            A.reference_decision_view(example, bad, self.frozen, pair_sha256=bad['record_sha256'],
                frozen_input_sha256=self.frozen['record_sha256'], inventory=inventory)
        bad = deepcopy(pair)
        bad['annotations'][0]['candidate_ordinal'] = next(i for i, row in enumerate(inventory['candidate_inventory'])
            if 'proposals[].proposal.subject_ref' not in row['eligible_paths'])
        bad = seal({k: v for k, v in bad.items() if k != 'record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):
            A.reference_decision_view(example, bad, self.frozen, pair_sha256=bad['record_sha256'],
                frozen_input_sha256=self.frozen['record_sha256'], inventory=inventory)

    def test_complete_new_geometry_has_causal_query_and_actual_generated_id_guard(self):
        C, A = self.api(); _, pair, _, inventory, view = self.view(C, A)
        aligned = A.align_reference_decision_target(self.processor, view, view_sha256=view['record_sha256'], max_target_tokens=20000)
        text = canonical_json_bytes(view['decision']).decode('utf-8')
        observed = self.processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        start = text.index('"subject_ref":{"caller_reference":true}') + len('"subject_ref":')
        expected_j = next(i for i, (a, b) in enumerate(observed['offset_mapping']) if a < start + 1 and b > start)
        request = next(r for r in aligned['requests'] if r['pointer'] == '/proposals/0/proposal/subject_ref')
        self.assertEqual(request['query_index'], expected_j)
        self.assertEqual(request['decoder_input_index'], expected_j + 1)
        self.assertEqual(aligned['decoder_inputs'][0], 2); self.assertEqual(aligned['labels'][-1], 1)
        self.assertEqual(aligned['decoder_inputs'][1:], observed['input_ids'])
        self.assertNotEqual(aligned['geometry'], pair['geometry'])
        generated = A.align_generated_reference_decision(self.processor, view['decision'], observed['input_ids'], inventory,
            max_target_tokens=20000, eos_observed=True)
        self.assertEqual(generated['requests'], aligned['requests'])
        bad_ids = list(observed['input_ids']); bad_ids[0] += 1
        with self.assertRaisesRegex(CognitiveKernelContractError, 'generated|identity|IDs|token'):
            A.align_generated_reference_decision(self.processor, view['decision'], bad_ids, inventory,
                max_target_tokens=20000, eos_observed=True)
        with self.assertRaises(CognitiveKernelContractError):
            A.align_generated_reference_decision(self.processor, view['decision'], observed['input_ids'], inventory,
                max_target_tokens=20000, eos_observed=False)

    def test_binding_predictions_wrong_paths_free_ids_and_boolean_ordinals_refuse(self):
        C, A = self.api(); example, _, choices, _, view = self.view(C, A)
        for mutate in (
            lambda p: p['binding'].update(caller_sha256='f' * 64),
            lambda p: p['binding'].update(candidate_inventory_sha256='f' * 64),
            lambda p: p['binding'].update(processed_view_sha256='f' * 64),
            lambda p: p['binding'].update(provenance_sha256='f' * 64),
            lambda p: p['rows'].pop(),
            lambda p: p['rows'].append(deepcopy(p['rows'][0])),
            lambda p: p['rows'][0].update(candidate_ordinal=True),
            lambda p: p['rows'][0].update(candidate_ordinal=len(self.joint['candidates'])),
        ):
            packet = {'binding': deepcopy(self.binding), 'rows': [{'pointer': k, 'candidate_ordinal': v} for k, v in choices.items()]}
            mutate(packet)
            with self.subTest(packet=packet), self.assertRaises(CognitiveKernelContractError):
                C.compile_reference_decision(example.context, example.opened_sources, view['decision'], packet,
                    caller=self.bundle['caller'], caller_sha256=self.binding['caller_sha256'], joint_alignment=self.joint,
                    joint_alignment_sha256=self.joint['record_sha256'], expected_binding=self.binding,
                    artifact_sha256='a' * 64, inference_run_id='mechanical-negative')
        for mutate in (
            lambda d: d['proposals'][0]['proposal'].update(subject_ref='free-model-identity'),
            lambda d: d['proposals'][0]['proposal'].update(value_text={'caller_reference': True}),
        ):
            bad = deepcopy(view['decision']); mutate(bad)
            with self.assertRaises(CognitiveKernelContractError): self.compile(C, example, bad, choices)

    def test_original_profile_and_pair_pin_cannot_be_relabelled(self):
        C, A = self.api(); example, pair, _ = self.complete_pair(); inventory = self.inventory(C)
        for mutate in (
            lambda p: p.update(schema='mfm-paired-supplied-j5-joint-target-geometry-v1'),
            lambda p: p.update(input_freeze_sha256='f' * 64),
        ):
            bad = deepcopy(pair); mutate(bad)
            bad = seal({k: v for k, v in bad.items() if k != 'record_sha256'})
            with self.assertRaisesRegex(CognitiveKernelContractError, 'profile|binding'):
                A.reference_decision_view(example, bad, self.frozen, pair_sha256=bad['record_sha256'],
                    frozen_input_sha256=self.frozen['record_sha256'], inventory=inventory)

    def test_optional_null_has_no_request_but_present_empty_role_refuses(self):
        C, A = self.api(); example, _, choices, inventory, view = self.view(C, A)
        path = 'proposals[].relationship_counterpart_ref'
        self.assertFalse(any(path in c['eligible_paths'] for c in inventory['candidate_inventory']))
        self.assertFalse(any(r['candidate_path'] == path for r in view['reference_leaves']))
        self.assertIsNone(self.compile(C, example, view['decision'], choices).proposals[0].relationship_counterpart_ref)
        bad = deepcopy(view['decision'])
        bad['proposals'][0]['relationship_counterpart_ref'] = {'caller_reference': True}
        with self.assertRaisesRegex(CognitiveKernelContractError, 'no eligible'):
            C.reference_requests(bad, inventory)

    def test_explicit_local_uncertainty_and_optional_null_are_preserved(self):
        C, A = self.api(); example, _, choices, _, view = self.view(C, A, local_uncertainty=True)
        self.assertEqual(view['decision']['proposals'][0]['proposal']['uncertainty_ref'], {'local_uncertainty': 1})
        self.assertFalse(any(r['pointer'].endswith('/uncertainty_ref') for r in view['reference_leaves']))
        compiled = self.compile(C, example, view['decision'], choices)
        self.assertIsNotNone(compiled.proposals[0].base.uncertainty_ref)
        self.assertIsNone(compiled.proposals[0].relationship_counterpart_ref)

    def test_registered_but_unsupported_selected_target_is_rejected_by_real_grounding(self):
        C, A = self.api(); example, _, choices, inventory, view = self.view(C, A)
        # Same actual caller and joint source. Both identities are eligible, but
        # the newly selected observation does not support the selected mission.
        self.compile(C, example, view['decision'], choices)
        mission = choices['/proposals/0/mission_target_refs/0']
        target = next(t for t in example.context.targets if t.ref_id == inventory['candidate_inventory'][mission]['ref_id'])
        other = next(i for i, c in enumerate(inventory['candidate_inventory'])
            if 'proposals[].proposal.evidence_refs' in c['eligible_paths'] and c['ref_id'] not in target.supporting_evidence_refs)
        changed = {**choices, '/proposals/0/proposal/evidence_refs/0': other,
            '/proposals/0/proposal/anchors/0/ref_id': other}
        decision = deepcopy(view['decision'])
        raw = dict(example.opened_sources)[inventory['candidate_inventory'][other]['ref_id']]
        decision['proposals'][0]['proposal']['anchors'][0]['end_byte'] = len(raw)
        with self.assertRaisesRegex(CognitiveKernelContractError, 'support|cited'):
            self.compile(C, example, decision, changed)


if __name__ == '__main__':
    unittest.main()
