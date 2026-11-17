"""Public actual-model mechanics; no private teaching or qualification evidence."""
import copy
from dataclasses import fields, replace
from hashlib import sha256
import importlib
import importlib.util
import json
import unittest

import torch

from src.alice_personality.identity import IdentityModel, IdentityModelConfig
from src.alice_personality.identity.codec import ConceptEntry, GraphEdge, SourceEntry, fingerprint
from src.alice_personality.identity.contracts import ConceptRecord, IdentityError, SourceRecord
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.teaching import IdentityObjectives, TeachingError
from src.alice_personality.teaching.cases import NamedTarget, ReviewedTeachingCase, TargetFact
from src.alice_personality.teaching.contracts import canonical
from src.alice_personality.teaching.materialize import materialize_case
from tests.personality.test_identity_codec import Backbone, Tokenizer, document
from tests.personality.test_reviewed_teaching_cases import PINS


NAME = 'scripts.eipm.gemma_n0.conditional_teaching'
CHANNELS = ('concept_applicability', 'concept_uncertainty',
            'concept_source_support', 'concept_source_disconfirmation')


def public_fixture():
    """Admitted public facts and a real producer frame with forbidden coordinates."""
    config = IdentityModelConfig(8, 3, learned_width=16, attention_heads=4, graph_layers=1, readout_layers=1)
    doc = document()
    source = doc.sources[0].record.record_id
    concepts = tuple(ConceptEntry('Public concept ' + str(i), ConceptRecord('concept-' + str(i), 'identity', i == 0))
                     for i in range(4))
    host = SourceEntry('Public host context', SourceRecord('host', doc.sources[0].record.source_family_id,
                       'HOST', 'HOST', False, True, False, False, False, False))
    doc = replace(doc, concepts=concepts, sources=(*doc.sources, host),
                  candidates=tuple(replace(c, authority_allowed=True) for c in doc.candidates),
                  edges=tuple(GraphEdge(source, c.record.concept_id, doc.relations[0].entry_id, 'identity_support')
                              for c in concepts))
    targets = []
    for field in CHANNELS:
        relation = 'source_' in field
        facts = []
        for i in range(4):
            pointers = ('concept-' + str(i), source) if relation else ('concept-' + str(i),)
            available = i != 2
            value = (1. if field in ('concept_applicability', 'concept_source_support') else 0.) if i == 0 else .25
            facts.append(TargetFact(pointers, value if available else None,
                (1. if i == 3 else .2) if available else None, available, (source,), ()))
        targets.append(NamedTarget(field, 'independent', tuple(facts)))
    common = dict(split='TRAIN', source_class='public_mechanical_fixture', document=doc,
                  source_family_ids=(doc.sources[0].record.source_family_id,), artifact_sha256=PINS,
                  annotation_origin='mechanical_fixture', review_id='public-review', annotation_note='Public mechanics only.')
    n1 = ReviewedTeachingCase(case_id='n1-public', phase='n1', targets=tuple(targets), **common)
    n2 = ReviewedTeachingCase(case_id='n2-public', phase='n2', targets=(NamedTarget('preferences', 'closed_distribution',
        tuple(TargetFact((c.candidate_id,), value, 0., True, (source,), ())
              for c, value in zip(doc.candidates, (.2, .8)))),), **common)
    producer = FrozenFrameProducer(Tokenizer(), Backbone(), config, _fixture=True)
    produced = producer.produce(doc)
    session = producer.close_and_verify()
    def materialize(case):
        return materialize_case(case, produced, session, config, expected_case_sha256=case.sha256,
            expected_artifact_sha256=PINS, family_splits={common['source_family_ids'][0]: 'TRAIN'}).batch
    return config, n1, n2, materialize(n1), materialize(n2), produced.binding, materialize


class ConditionalTeachingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(NAME), 'conditional teaching helper is missing')
        self.api = importlib.import_module(NAME)
        torch.manual_seed(61)
        (config, self.n1, self.n2, self.b1, self.b2, self.binding, self.materialize) = public_fixture()
        self.model = IdentityModel(config)
        self.objectives = IdentityObjectives(config.learned_width).requires_grad_(False)
        self.model.set_phase('n2')
        self.record = self.build()

    def build(self, **changes):
        b1 = changes.pop('n1_batch', self.b1)
        reviews = {b1.fingerprint(): {'split': b1.split, 'case_ids': list(b1.case_ids),
                                    'source_family_ids': [list(row) for row in b1.source_family_ids]}}
        args = dict(n1_case=self.n1, n2_case=self.n2, n1_batch=b1, n2_batch=self.b2,
                    frame_binding=self.binding, reviewed_n1_batches=reviews)
        args.update(changes)
        return self.api.build_correction_record(**args)

    def forward(self, record=None, **kwargs):
        record = self.record if record is None else record
        return self.api.forward_with_correction(self.model, self.b2, record,
            expected_record_sha256=self.api.record_sha256(record), **kwargs)

    def loss_and_gradients(self, packet):
        loss, _, _ = self.objectives(packet, self.b2)
        loss.backward()
        return float(loss.detach()), {n: p.grad.detach().clone() for n, p in self.model.named_parameters() if p.grad is not None}

    def test_record_retains_original_provenance_and_filters_only_eligible_facts(self):
        self.assertEqual(json.loads(json.dumps(self.record))['n1_case'], self.n1.to_mapping())
        values = self.api.validate_correction_record(self.record, self.b2,
            expected_sha256=self.api.record_sha256(self.record))
        source = self.n1.document.sources[0].record.record_id
        self.assertEqual(values['concept_applicability'], {('concept-0',): 1., ('concept-1',): .25})
        self.assertEqual(values['concept_source_support'], {('concept-0', source): 1., ('concept-1', source): .25})
        self.assertEqual(set(values), set(CHANNELS))
        self.assertEqual(self.record['n1_batch_sha256'], self.b1.fingerprint())
        self.assertEqual(self.record['n2_batch_sha256'], self.b2.fingerprint())

    def test_sham_native_outputs_and_all_gradients_are_exact(self):
        natural = self.model(self.b2.frame)
        expected_packet = fingerprint(natural)
        loss, gradients = self.loss_and_gradients(natural)
        self.model.zero_grad(set_to_none=True)
        sham = self.forward(apply_correction=False)
        actual_loss, actual_gradients = self.loss_and_gradients(sham)
        self.assertEqual(expected_packet, fingerprint(sham))
        self.assertEqual(loss, actual_loss)
        self.assertEqual(gradients.keys(), actual_gradients.keys())
        for name in gradients:
            self.assertTrue(torch.equal(gradients[name], actual_gradients[name]), name)

    def test_same_value_correction_preserves_logits_packet_and_gradients(self):
        natural = self.model(self.b2.frame)
        expected = fingerprint(natural)
        loss, gradients = self.loss_and_gradients(natural)
        targets = []
        for target in self.n1.targets:
            value = getattr(natural.representation, target.field)
            facts = tuple(replace(f, value=float(value[(0, i, 0) if 'source_' in target.field else (0, i)]))
                          if f.available else f for i, f in enumerate(target.facts))
            targets.append(replace(target, facts=facts))
        case = replace(self.n1, targets=tuple(targets))
        record = self.build(n1_case=case, n1_batch=self.materialize(case))
        self.model.zero_grad(set_to_none=True)
        actual = self.forward(record)
        actual_loss, actual_gradients = self.loss_and_gradients(actual)
        self.assertEqual(expected, fingerprint(actual))
        self.assertEqual(loss, actual_loss)
        for name in gradients:
            self.assertTrue(torch.equal(gradients[name], actual_gradients[name]), name)

    def test_corrected_native_forward_trains_n2_only_and_preserves_rich_state(self):
        natural = self.model(self.b2.frame)
        _, natural_gradients = self.loss_and_gradients(natural)
        self.model.zero_grad(set_to_none=True)
        observed = []
        handle = self.model.n1.register_forward_hook(lambda _m, _a, out: observed.append(out))
        try:
            packet = self.forward()
        finally:
            handle.remove()
        original, query = observed[0]
        patched = packet.representation
        self.assertFalse(torch.equal(packet.latent_candidates, natural.latent_candidates))
        self.assertEqual(float(patched.concept_applicability[0, 0]), 1.)
        self.assertEqual(float(patched.concept_uncertainty[0, 0]), 0.)
        for field in fields(original):
            if field.name not in CHANNELS and field.name not in tuple(n + '_logits' for n in CHANNELS):
                self.assertIs(getattr(patched, field.name), getattr(original, field.name))
        for name in CHANNELS:
            self.assertTrue(torch.equal(getattr(patched, name)[:, 2:], getattr(original, name)[:, 2:]))
            self.assertTrue(torch.isfinite(getattr(patched, name + '_logits')).all())
            if 'source_' in name:
                self.assertTrue(torch.equal(getattr(patched, name)[:, :, 1:], getattr(original, name)[:, :, 1:]))
        _, gradients = self.loss_and_gradients(packet)
        self.assertTrue(gradients)
        self.assertTrue(all(name.startswith('n2.') for name in gradients))
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in gradients.values()))
        self.assertTrue(any(not torch.equal(gradients[name], natural_gradients[name]) for name in gradients))
        self.assertFalse(query.requires_grad)
        for bank in (self.b2.frame.query, self.b2.frame.sources, self.b2.frame.concepts, self.b2.frame.candidates):
            self.assertFalse(bank.states.requires_grad)
            self.assertIsNone(bank.states.grad)

    def test_partial_subsets_partition_complete_concept_groups_deterministically(self):
        packets = {subset: self.forward(subset=subset) for subset in ('full', 'first_half', 'second_half')}
        original = self.model(self.b2.frame).representation
        groups = []
        for subset in ('first_half', 'second_half'):
            representation = packets[subset].representation
            selected = set()
            for i in (0, 1):
                changed = [not torch.equal(getattr(representation, name)[:, i], getattr(original, name)[:, i])
                           for name in CHANNELS]
                self.assertTrue(all(changed) or not any(changed))
                if all(changed):
                    selected.add(i)
            groups.append(selected)
            self.assertEqual(fingerprint(packets[subset]), fingerprint(self.forward(subset=subset)))
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(groups[0] | groups[1], {0, 1})
        self.assertFalse(groups[0] & groups[1])

    def test_external_pin_case_frame_batch_and_review_mismatches_are_refused(self):
        with self.assertRaises((TeachingError, IdentityError)):
            self.api.validate_correction_record(self.record, self.b2, expected_sha256='0' * 64)
        for change in ({'case_ids': ('different',)}, {'frame': replace(self.b2.frame,
                       candidate_context_mask=~self.b2.frame.candidate_context_mask)}):
            with self.subTest(change=tuple(change)), self.assertRaises((TeachingError, IdentityError)):
                self.api.validate_correction_record(self.record, replace(self.b2, **change),
                    expected_sha256=self.api.record_sha256(self.record))
        for change in ({'reviewed_n1_batches': {}}, {'n1_case': replace(self.n1, split='DEV')},
                       {'n1_batch': replace(self.b1, targets=self.b2.targets)},
                       {'frame_binding': {**self.binding, 'frame_sha256': '0' * 64}}):
            with self.subTest(change=tuple(change)), self.assertRaises((TeachingError, IdentityError)):
                self.build(**change)

    def test_repinned_unknown_and_forbidden_corrections_are_refused(self):
        for pointer in ('unknown', 'host'):
            record = copy.deepcopy(self.record)
            target = record['n1_case']['targets'][2]
            target['facts'][0]['pointers'][1] = pointer
            record['n1_case_sha256'] = sha256(canonical(record['n1_case'])).hexdigest()
            receipt = record['n1_materialization_binding']
            receipt['case_sha256'] = record['n1_case_sha256']
            receipt['receipt_sha256'] = sha256(canonical({k: v for k, v in receipt.items()
                                                        if k != 'receipt_sha256'})).hexdigest()
            # Even a caller supplying a new record pin cannot create authority.
            with self.subTest(pointer=pointer), self.assertRaises((TeachingError, IdentityError)):
                self.api.validate_correction_record(record, self.b2, expected_sha256=self.api.record_sha256(record))

    def test_repinned_n1_batch_digest_is_checked_against_original_materialization(self):
        record = copy.deepcopy(self.record)
        record['n1_batch_sha256'] = '0' * 64
        with self.assertRaises(TeachingError):
            self.api.validate_correction_record(record, self.b2, expected_sha256=self.api.record_sha256(record))

    def test_modified_native_mask_is_refused_and_existing_hook_survives(self):
        def change_mask(_module, _args, output):
            representation, query = output
            mask = representation.concept_mask.clone()
            mask[0, 0] = False
            return replace(representation, concept_mask=mask), query
        handle = self.model.n1.register_forward_hook(change_mask)
        try:
            before = tuple(self.model.n1._forward_hooks)
            with self.assertRaises(TeachingError):
                self.forward()
            self.assertEqual(before, tuple(self.model.n1._forward_hooks))
        finally:
            handle.remove()

    def test_invalid_sham_record_and_trainable_n1_or_n3_are_refused(self):
        record = copy.deepcopy(self.record)
        record['n2_case_id'] = 'wrong-case'
        with self.assertRaises((TeachingError, IdentityError)):
            self.forward(record, apply_correction=False)
        for module in (self.model.n1, self.model.n3):
            parameter = next(module.parameters())
            parameter.requires_grad_(True)
            with self.assertRaises((TeachingError, IdentityError)):
                self.forward()
            parameter.requires_grad_(False)
        self.model.set_phase('inference')
        self.forward().validate()

    def test_exception_removes_only_our_hook_and_next_forward_is_natural(self):
        baseline = fingerprint(self.model(self.b2.frame))
        state = fingerprint(self.model.state_dict())
        before = tuple(self.model.n1._forward_hooks)
        def fail(_module, _args, _output):
            raise RuntimeError('public downstream failure')
        handle = self.model.n2.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'public downstream failure'):
                self.forward()
        finally:
            handle.remove()
        self.assertEqual(before, tuple(self.model.n1._forward_hooks))
        self.assertEqual(state, fingerprint(self.model.state_dict()))
        self.assertEqual(baseline, fingerprint(self.model(self.b2.frame)))


if __name__ == '__main__':
    unittest.main()
