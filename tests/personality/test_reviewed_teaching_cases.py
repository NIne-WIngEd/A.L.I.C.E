"""Public construction mechanics; no learned ALICE or owner-fidelity claim."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.alice_personality.identity import IdentityModel, IdentityModelConfig
from src.alice_personality.identity.codec import SourceEntry, decision_to_mapping, fingerprint
from src.alice_personality.identity.contracts import HEAD_FAMILIES, LabelSpec, SourceRecord
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.teaching import IdentityObjectives, IdentityTrainer, TeachingError, TrainingRecipe
from src.alice_personality.teaching.cases import NamedTarget, ReviewedTeachingCase, TargetFact
from src.alice_personality.teaching.materialize import materialize_case
from src.alice_personality.teaching.contracts import canonical
from tests.personality.test_identity_codec import Backbone, Tokenizer, document
from tests.personality.test_identity_teaching import _admission


PINS = {name: sha256(name.encode()).hexdigest() for name in
        ('source_package', 'compiled_receipt', 'annotation_protocol', 'review_record')}
PINS['source_package'] = sha256(b'public mechanical source bytes; no identity payload').hexdigest()
PINS['compiled_receipt'] = sha256(canonical({'schema': 'public-mechanical-source-receipt-v1',
    'training_authority_granted': False, 'acceptance_authority': False}) + b'\n').hexdigest()


def fact(*pointers, value=1.0, uncertainty=0.0, available=True):
    return TargetFact(tuple(pointers), value if available else None,
                      uncertainty if available else None, available, (), ())


class ReviewedCaseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)
        self.config = IdentityModelConfig(8, 3, learned_width=16, attention_heads=4)
        doc = document()
        self.doc = replace(doc, candidates=(doc.candidates[0], replace(doc.candidates[1], authority_allowed=True)))
        self.a, self.b = (entry.candidate_id for entry in self.doc.candidates)
        self.family = self.doc.sources[0].record.source_family_id
        self.source = self.doc.sources[0].record.record_id
        self.concept = self.doc.concepts[0].record.concept_id

    def case(self, targets=None, *, phase='n2', doc=None, split=None):
        return ReviewedTeachingCase('public-construction-case', phase,
            split or ('CALIBRATION_TRAIN' if phase == 'calibration' else 'TRAIN'),
            'public_mechanical_fixture', doc or self.doc, (self.family,), PINS,
            'mechanical_fixture', 'public-mechanics-review', 'Fixture values exercise binding, not identity gold.',
            targets or (NamedTarget('preferences', 'closed_distribution',
                       (fact(self.a, value=0.4), fact(self.b, value=0.6))),))

    def build(self, case, *, produced=None, session=None, expected=None, families=None):
        if produced is None:
            producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.config, _fixture=True)
            produced = producer.produce(case.document)
            session = producer.close_and_verify()
        return materialize_case(case, produced, session, self.config,
            expected_case_sha256=expected or case.sha256, expected_artifact_sha256=PINS,
            family_splits=families or {self.family: case.split})

    def test_roundtrip_strict_schema_and_targets_never_enter_provider_text(self):
        case = self.case()
        restored = ReviewedTeachingCase.from_json(canonical(case.to_mapping()))
        self.assertEqual(restored.sha256, case.sha256)
        tokenizer = Tokenizer()
        producer = FrozenFrameProducer(tokenizer, Backbone(), self.config, _fixture=True)
        produced = producer.produce(case.document)
        self.build(case, produced=produced, session=producer.close_and_verify())
        self.assertNotIn(case.review_id, '\n'.join(tokenizer.seen))
        self.assertNotIn(case.annotation_note, '\n'.join(tokenizer.seen))
        raw = case.to_mapping()
        raw['preferred_candidate'] = self.a
        with self.assertRaises(TeachingError):
            ReviewedTeachingCase.from_json(json.dumps(raw))
        with self.assertRaises(TeachingError):
            ReviewedTeachingCase.from_json('{"schema":"x","schema":"y"}')

    def test_pointer_resolution_survives_candidate_reordering(self):
        case = self.case()
        first = self.build(case)
        changed = replace(case, document=replace(self.doc, candidates=tuple(reversed(self.doc.candidates))))
        second = self.build(changed)
        torch.testing.assert_close(first.batch.targets['preferences'].values,
                                   second.batch.targets['preferences'].values.flip(-1))
        self.assertNotEqual(first.receipt['case_sha256'], second.receipt['case_sha256'])
        self.assertEqual(first.batch.materialization_binding, first.receipt)

    def test_partial_comparison_preserves_unknown_rival_and_zero_discrimination_count(self):
        case = self.case((NamedTarget('preferences', 'partial_distribution', (fact(self.a),)),))
        result = self.build(case)
        target = result.batch.targets['preferences']
        self.assertEqual(target.available.tolist(), [[True, False]])
        self.assertTrue(torch.isnan(target.values[0, 1]))
        self.assertTrue(torch.isnan(target.uncertainty[0, 1]))
        self.assertEqual(result.receipt['coverage']['preferences']['discriminating_rows'], 0)
        with self.assertRaisesRegex(TeachingError, 'closed'):
            self.build(replace(case, targets=(replace(case.targets[0], mode='closed_distribution'),)))

    def test_unknown_duplicate_unauthorized_pointers_and_bad_probabilities_refused(self):
        for targets in (
            (NamedTarget('preferences', 'partial_distribution', (fact('absent'),)),),
            (NamedTarget('preferences', 'partial_distribution', (fact(self.a), fact(self.a))),),
            (NamedTarget('preferences', 'partial_distribution', (fact(self.a, value=-1),)),),
            (NamedTarget('preferences', 'independent', (fact(self.a),)),),
            (NamedTarget('owner_fidelity_uncertainty', 'independent', (fact(self.a),)),),
        ):
            with self.subTest(targets=targets), self.assertRaises(TeachingError):
                self.build(self.case(targets))
        case = self.case(doc=document())
        with self.assertRaises(TeachingError):
            self.build(case)

    def test_external_review_frame_session_and_family_bindings_are_not_self_approval(self):
        case = self.case()
        with self.assertRaises(TeachingError):
            self.build(case, expected='0' * 64)
        with self.assertRaisesRegex(TeachingError, 'family'):
            self.build(case, families={self.family: 'CALIBRATION_TRAIN'})
        with self.assertRaises(TeachingError):
            self.build(replace(case, split='FINAL'))
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.config, _fixture=True)
        produced = producer.produce(case.document)
        with self.assertRaises(TeachingError):
            self.build(case, produced=produced, session={})
        session = producer.close_and_verify()
        tampered = replace(produced, frame=replace(produced.frame,
                          candidate_context_mask=~produced.frame.candidate_context_mask))
        with self.assertRaises(TeachingError):
            self.build(case, produced=tampered, session=session)
        forged = dict(session, frame_binding_sha256=[])
        forged['receipt_sha256'] = sha256(canonical({k: v for k, v in forged.items() if k != 'receipt_sha256'})).hexdigest()
        with self.assertRaises(TeachingError):
            self.build(case, produced=produced, session=forged)

    def test_pairs_preserve_unreviewed_reverse_and_reject_contradictions(self):
        case = self.case((NamedTarget('candidate_pair_probabilities', 'independent',
                                  (fact(self.a, self.b, value=0.5),)),))
        target = self.build(case).batch.targets['candidate_pair_probabilities']
        self.assertTrue(target.available[0, 0, 1])
        self.assertFalse(target.available[0, 1, 0])
        self.assertTrue(torch.isnan(target.values[0, 1, 0]))
        bad = replace(case, targets=(replace(case.targets[0], facts=(
            fact(self.a, self.b, value=0.7), fact(self.b, self.a, value=0.7))),))
        with self.assertRaises(TeachingError):
            self.build(bad)

    def test_materialized_cases_run_actual_admitted_n1_n2_and_separate_n3_steps(self):
        model = IdentityModel(self.config)
        n1_targets = (
            NamedTarget('concept_applicability', 'independent', (fact(self.concept, value=0.8),)),
            NamedTarget('concept_uncertainty', 'independent', (fact(self.concept, value=0.2),)),
            NamedTarget('concept_source_support', 'independent', (fact(self.concept, self.source, value=0.9),)),
            NamedTarget('concept_source_disconfirmation', 'independent', (fact(self.concept, self.source, value=0.1),)),
        )
        for phase, targets in [('n1', n1_targets), ('n2', self.case().targets), ('calibration', (
            NamedTarget('owner_fidelity_uncertainty', 'independent', (fact(self.a, value=0.6),)),))]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                built = self.build(self.case(targets, phase=phase))
                recipe = TrainingRecipe(phase, 0.002, 1.0, 37)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                review = {'reviewed_case_artifact_sha256': {built.receipt['case_sha256']: PINS}}
                admission = _admission(directory, model, recipe, [built.batch], reviewed_changes=review)
                trainer = IdentityTrainer(model, recipe, admission,
                                          first_batch=built.batch)
                trainer.step(built.batch)
                changed = [name for name, parameter in model.named_parameters()
                           if not torch.equal(before[name], parameter)]
                self.assertTrue(changed)
                self.assertTrue(all(name.startswith({'n1': 'n1.', 'n2': 'n2.', 'calibration': 'n3.'}[phase])
                                    for name in changed), changed)
                self.assertFalse(built.batch.frame.sources.states.requires_grad)
                if phase == 'n1':
                    self.assertTrue(any(name.startswith('n1.conditional_concept.') for name in changed))
                    self.assertTrue(any(name.startswith('n1.conditional_source_key.') for name in changed))
                if phase == 'n2':
                    for prefix in ('n2.conditional_source_value.', 'n2.conditional_relation_embedding.', 'n2.conditional_embedding.'):
                        self.assertTrue(any(name.startswith(prefix) for name in changed), prefix)
                checkpoint = Path(directory).resolve() / 'checkpoint'
                manifest = trainer.save_checkpoint(checkpoint)
                resumed_model = IdentityModel(self.config)
                resumed_model.load_state_dict(before)
                resumed = IdentityTrainer(resumed_model, recipe, admission, first_batch=built.batch)
                resumed.resume_checkpoint(checkpoint,
                    expected_manifest_sha256=manifest['manifest_sha256'])
                self.assertEqual(resumed.steps, trainer.steps)
                for name, parameter in resumed_model.named_parameters():
                    torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)
        decoded = decision_to_mapping(model(built.batch.frame))
        self.assertIn('conditional_concepts', decoded)
        self.assertEqual(decoded['qualification'], 'UNQUALIFIED')

    def test_sparse_conditional_loss_unknown_positions_have_no_gradient(self):
        extra = SourceEntry('Public mechanical synthetic policy, not historical evidence.',
            SourceRecord('unknown-support-source', self.family, 'ASYN_DIRECT', 'A-SYN',
                         True, True, False, False, False, True))
        doc = replace(self.doc, sources=(*self.doc.sources, extra))
        case = self.case((NamedTarget('concept_source_support', 'independent',
                            (fact(self.concept, self.source, value=0.3),)),), phase='n1', doc=doc)
        built = self.build(case)
        model = IdentityModel(self.config)
        model.set_phase('n1')
        packet = model(built.batch.frame)
        logits = packet.representation.concept_source_support_logits
        logits.retain_grad()
        loss, _, _ = IdentityObjectives(self.config.learned_width)(packet, built.batch)
        loss.backward()
        self.assertNotEqual(float(logits.grad[0, 0, 0]), 0)
        self.assertEqual(float(logits.grad[0, 0, 1]), 0)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.n1.parameters()))

    def test_conditional_learning_signals_are_consumed_by_n2_without_authority_changes(self):
        built = self.build(self.case())
        model = IdentityModel(self.config)
        first = model(built.batch.frame)
        with torch.no_grad():
            model.n1.conditional_concept[-1].bias[0].add_(5)
        second = model(built.batch.frame)
        self.assertFalse(torch.equal(first.representation.concept_applicability,
                                     second.representation.concept_applicability))
        self.assertFalse(torch.equal(first.latent_candidates, second.latent_candidates))
        torch.testing.assert_close(first.representation.concept_mask, second.representation.concept_mask)
        torch.testing.assert_close(first.representation.concept_source_mask, second.representation.concept_source_mask)
        self.assertEqual(first.source_records, second.source_records)

    def test_all_n2_and_calibration_families_materialize_with_explicit_unknown_memory(self):
        value_labels = (*self.doc.labels['values'], LabelSpec('second-value', 'another value',
                        'A second public mechanical value descriptor', 'categorical'))
        doc = replace(self.doc, labels={**self.doc.labels, 'values': value_labels})
        for phase in ('n2', 'calibration'):
            targets = [NamedTarget('preferences', 'closed_distribution',
                                  (fact(self.a, value=0.5), fact(self.b, value=0.5)))]
            for name in ('co_valid_probabilities', 'evidence_sufficiency', 'uncertainty', 'contraindications', 'failure_tail_risk'):
                targets.append(NamedTarget(name, 'independent', (fact(self.a, value=0.4),)))
            for family in HEAD_FAMILIES:
                if family == 'voice' and phase == 'n2':
                    continue
                targets.append(NamedTarget('head:' + family,
                    'closed_distribution' if family == 'stance' else 'independent',
                    tuple(fact(self.a, spec.label_id, value=1.0 if family == 'stance' else 0.4)
                          for spec in doc.labels[family])))
            targets.extend((NamedTarget('candidate_pair_probabilities', 'independent',
                                       (fact(self.a, self.b, value=0.5),)),
                            NamedTarget('value_tradeoff_probabilities', 'independent',
                                       (fact(self.a, value_labels[0].label_id, value_labels[1].label_id, value=0.7),)),
                            NamedTarget('voice_control_confidence', 'independent',
                                       (fact(self.a, doc.labels['voice'][0].label_id, value=0.3),))))
            if phase == 'n2':
                targets.extend((NamedTarget('voice_control_values', 'independent',
                                            (fact(self.a, doc.labels['voice'][0].label_id, value=0.6),)),
                    NamedTarget('concept_pointers', 'closed_distribution', (fact(self.a, self.concept),)),
                    NamedTarget('evidence_pointers', 'closed_distribution', (fact(self.a, self.source),)),
                    NamedTarget('historical_evidence_pointers', 'closed_distribution', (fact(self.a, self.source),)),
                    NamedTarget('alice_experience_pointers', 'partial_distribution',
                                (fact(self.a, self.source, available=False),))))
            else:
                targets.append(NamedTarget('owner_fidelity_uncertainty', 'independent', (fact(self.a, value=0.7),)))
            built = self.build(self.case(tuple(targets), phase=phase, doc=doc))
            loss, losses, _ = IdentityObjectives(self.config.learned_width)(IdentityModel(self.config)(built.batch.frame), built.batch)
            self.assertEqual(set(losses), {target.field for target in targets})
            self.assertTrue(torch.isfinite(loss))
            if phase == 'n2':
                self.assertFalse(built.batch.targets['alice_experience_pointers'].available.any())

    def test_resealed_frame_with_document_pointer_or_authority_change_is_refused(self):
        case = self.case()
        producer = FrozenFrameProducer(Tokenizer(), Backbone(), self.config, _fixture=True)
        produced = producer.produce(case.document)
        session = producer.close_and_verify()
        changed = replace(produced.frame, source_records=((replace(produced.frame.source_records[0][0],
                           historical_truth_allowed=False),),))
        binding = {**produced.binding, 'frame_sha256': fingerprint(changed)}
        binding['binding_sha256'] = sha256(canonical({k: v for k, v in binding.items() if k != 'binding_sha256'})).hexdigest()
        session = {**session, 'frame_binding_sha256': [binding['binding_sha256']]}
        session['receipt_sha256'] = sha256(canonical({k: v for k, v in session.items() if k != 'receipt_sha256'})).hexdigest()
        with self.assertRaisesRegex(TeachingError, 'document'):
            self.build(case, produced=replace(produced, frame=changed, binding=binding), session=session)

    def test_source_specific_conditional_relations_reach_n2_even_when_means_are_identical(self):
        extra = SourceEntry('A distinct public mechanical evidence source with different content.',
                            replace(self.doc.sources[0].record, record_id='second-source'))
        built = self.build(self.case(doc=replace(self.doc, sources=(*self.doc.sources, extra))))
        model = IdentityModel(self.config)
        with torch.no_grad():
            model.n1.conditional_source_key.weight.mul_(10)
        first = model(built.batch.frame)
        handle = model.n1.conditional_source_key.register_forward_hook(lambda _module, _args, output: output.flip(1))
        try:
            second = model(built.batch.frame)
        finally:
            handle.remove()
        torch.testing.assert_close(first.representation.concept_source_support,
                                   second.representation.concept_source_support.flip(-1))
        self.assertFalse(torch.equal(first.representation.concept_source_support,
                                     second.representation.concept_source_support))
        self.assertFalse(torch.equal(first.latent_candidates, second.latent_candidates))

    def test_materialized_source_compiler_and_annotation_pins_must_match_admission(self):
        for broken in ('source_package', 'compiled_receipt', 'annotation_protocol', 'review_record', 'missing_annotation_roles'):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as directory:
                built = self.build(self.case())
                model = IdentityModel(self.config)
                recipe = TrainingRecipe('n2', 0.002, 1.0, 37)
                if broken == 'missing_annotation_roles':
                    review_pins = {key: value for key, value in PINS.items() if key not in ('annotation_protocol', 'review_record')}
                    receipt = {**built.receipt, 'artifact_sha256': review_pins}
                    receipt['receipt_sha256'] = sha256(canonical({k: v for k, v in receipt.items() if k != 'receipt_sha256'})).hexdigest()
                    changed = replace(built.batch, materialization_binding=receipt)
                elif broken in ('annotation_protocol', 'review_record'):
                    review_pins = {**PINS, broken: '0' * 64}
                    changed = built.batch
                else:
                    receipt = {**built.receipt, 'artifact_sha256': {**PINS, broken: '0' * 64}}
                    receipt['receipt_sha256'] = sha256(canonical({k: v for k, v in receipt.items() if k != 'receipt_sha256'})).hexdigest()
                    changed = replace(built.batch, materialization_binding=receipt)
                    review_pins = receipt['artifact_sha256']
                admission = _admission(directory, model, recipe, [changed], reviewed_changes={
                    'reviewed_case_artifact_sha256': {built.receipt['case_sha256']: review_pins}})
                with self.assertRaisesRegex(TeachingError, 'materializ'):
                    IdentityTrainer(model, recipe, admission, first_batch=changed)


if __name__ == '__main__':
    unittest.main()
