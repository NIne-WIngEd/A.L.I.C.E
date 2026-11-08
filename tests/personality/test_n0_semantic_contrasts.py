"""Authentic contrast packaging; fictional producer does not qualify Gemma."""
from dataclasses import replace
from hashlib import sha256
import importlib
import unittest

import torch

from src.alice_personality.identity.codec import ConceptEntry, SourceEntry, canonical
from src.alice_personality.identity.contracts import ConceptRecord, IdentityError, IdentityModelConfig, SourceRecord
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from tests.personality import test_identity_codec as fixtures

try:
    api = importlib.import_module('scripts.eipm.gemma_n0.n0_semantic_contrasts')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.n0_semantic_contrasts':
        raise
    api = None


class N0SemanticContrastTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(api, 'The target-free authentic context contrast helper is missing')
        self.original = fixtures.document()
        self.options = dict(parent_case_id='fictional-train-parent',
            parent_family_id='fictional-family', parent_split='TRAIN',
            variant_id='fictional-context-paraphrase', comparison_kind='paraphrase')

    def contrast(self, **changes):
        options = dict(self.options)
        options.update(changes)
        return api.build_context_contrast(self.original,
            replacements={'situation-id': 'A fictional person considers passing along an unrelated rumor.'}, **options)

    def test_changes_only_current_text_and_real_producer_retains_source_and_candidates(self):
        original_sha = self.original.sha256
        variant, binding = self.contrast()
        first, second = self.original.to_mapping(), variant.to_mapping()
        for key in ('sources','concepts','candidates','relations','labels','edges','schema'):
            self.assertEqual(canonical(first[key]), canonical(second[key]))
        for old, new in zip(self.original.fields, variant.fields):
            self.assertEqual((old.field_id,old.semantic_role,old.name,old.available),
                             (new.field_id,new.semantic_role,new.name,new.available))
            if old.field_id != 'situation-id':
                self.assertEqual(old, new)
        self.assertEqual(self.original.sha256, original_sha)
        self.assertNotEqual(variant.sha256, original_sha)
        config = IdentityModelConfig(8,3,learned_width=8,attention_heads=2)
        producer = FrozenFrameProducer(fixtures.Tokenizer(),fixtures.Backbone(),config,_fixture=True)
        old_frame = producer.produce(self.original)
        new_frame = producer.produce(variant)
        self.assertTrue(torch.equal(old_frame.frame.sources.states,new_frame.frame.sources.states))
        self.assertTrue(torch.equal(old_frame.frame.candidates.states,new_frame.frame.candidates.states))
        self.assertFalse(torch.equal(old_frame.frame.query.states,new_frame.frame.query.states))
        self.assertEqual(new_frame.binding['document_sha256'],binding['document_sha256'])
        result = api.verify_context_contrast(self.original,variant,binding,
            expected_binding_sha256=binding['binding_sha256'])
        self.assertEqual(result,binding)
        self.assertNotIn('targets',variant.to_mapping())
        self.assertNotIn('targets',binding)

    def test_retains_parent_partition_and_cannot_assert_independent_or_qualified_evidence(self):
        _, binding = self.contrast()
        self.assertEqual(binding['parent'],{'case_id':'fictional-train-parent',
            'family_id':'fictional-family','split':'TRAIN'})
        self.assertEqual(binding['comparison_kind'],'paraphrase')
        self.assertEqual(binding['qualification'],'UNQUALIFIED')
        for key in ('independent_qualification','n0_approved','personality_approved',
                    'source_acceptance_authority','experience_event','targets_in_input'):
            self.assertFalse(binding[key])
        self.assertEqual(binding['evaluation_use'],'development_contrast_of_existing_parent')
        self.assertEqual(binding['failure_attribution'],'unresolved_across_provider_readout_n1_n2_n3_consumer')
        with self.assertRaises(IdentityError):
            self.contrast(parent_split='FINAL')

    def test_refuses_identity_time_unknown_unavailable_and_noop_edits(self):
        for replacements in ({'source_identity-id':'Overwritten origin'},
                             {'evidence_time_uncertainty-id':'Overwritten time'},
                             {'unknown-field':'Something'},
                             {'social_emotional-id':'Previously absent cue'},
                             {}, {'situation-id':next(f.text for f in self.original.fields if f.field_id=='situation-id')}):
            with self.subTest(fields=list(replacements)):
                with self.assertRaises(IdentityError):
                    api.build_context_contrast(self.original,replacements=replacements,**self.options)

    def test_verifier_rejects_changed_evidence_candidates_and_undeclared_field_text(self):
        variant, binding = self.contrast()
        source = replace(variant.sources[0],text='A different fictional source policy.')
        candidate = replace(variant.candidates[0],text='A changed candidate response.')
        fields = tuple(replace(f,text='A different situation.') if f.field_id=='situation-id' else f for f in variant.fields)
        for changed in (replace(variant,sources=(source,)),
                        replace(variant,candidates=(candidate,*variant.candidates[1:])),
                        replace(variant,fields=fields)):
            with self.subTest(document=changed.sha256):
                with self.assertRaises(IdentityError):
                    api.verify_context_contrast(self.original,changed,binding,
                        expected_binding_sha256=binding['binding_sha256'])

    def test_resealed_claim_cannot_make_source_or_identity_field_edit_eligible(self):
        variant, binding = self.contrast()
        field = next(f for f in variant.fields if f.field_id=='source_identity-id')
        changed = replace(variant,fields=tuple(replace(f,available=True,text='A fabricated identity.')
            if f is field else f for f in variant.fields))
        forged = dict(binding,document_sha256=changed.sha256)
        forged.pop('binding_sha256')
        forged['binding_sha256'] = sha256(canonical(forged)).hexdigest()
        with self.assertRaises(IdentityError):
            api.verify_context_contrast(self.original,changed,forged,
                expected_binding_sha256=forged['binding_sha256'])

    def test_verifier_rejects_changed_external_pin_and_ancestry(self):
        variant, binding = self.contrast()
        with self.assertRaises(IdentityError):
            api.verify_context_contrast(self.original,variant,binding,expected_binding_sha256='0'*64)
        forged = dict(binding,parent={'case_id':'another-parent','family_id':'another-family','split':'DEV'})
        with self.assertRaises(IdentityError):
            api.verify_context_contrast(self.original,variant,forged,
                expected_binding_sha256=binding['binding_sha256'])

    def test_coherent_context_edit_preserves_identity_and_all_context_records(self):
        self.original = replace(self.original,concepts=(*self.original.concepts,
            ConceptEntry('Current fictional permission remains active.',
                ConceptRecord('current-relationship','relationship'))),
            fields=tuple(replace(f,available=True,text='Current fictional permission remains active.')
                if f.semantic_role in ('selected_spans','relationship') else f for f in self.original.fields))
        variant, binding = api.build_context_contrast(self.original,
            replacements={'situation-id':'The fictional permission is now withdrawn.',
                'selected_spans-id':'The fictional permission is now withdrawn.',
                'relationship-id':'The fictional permission is now withdrawn.'},
            contextual_concepts={'current-relationship':'Current fictional permission is withdrawn.'},
            **dict(self.options,comparison_kind='decisive_condition'))
        self.assertEqual(variant.concepts[0],self.original.concepts[0])
        self.assertEqual(variant.concepts[1].record,self.original.concepts[1].record)
        self.assertEqual(variant.concepts[1].text,'Current fictional permission is withdrawn.')
        self.assertEqual(binding['changed_context_concepts'][0]['concept_id'],'current-relationship')
        self.assertEqual(binding['changed_context_concepts'][0]['view'],'relationship')
        self.assertEqual(api.verify_context_contrast(self.original,variant,binding,
            expected_binding_sha256=binding['binding_sha256']),binding)
        with self.assertRaises(IdentityError):
            api.build_context_contrast(self.original,replacements={'situation-id':'A new condition.'},
                contextual_concepts={'opaque-concept-id':'Overwritten identity.'},**self.options)

    def test_only_explicit_current_context_source_text_can_change(self):
        current = SourceEntry('Current fictional permission remains active.',
            SourceRecord('current-source','fictional-family','RELATIONSHIP','RELATIONSHIP',
                False,True,False,False,False,False))
        self.original = replace(self.original,sources=(*self.original.sources,current))
        variant, binding = api.build_context_contrast(self.original,
            replacements={'situation-id':'Current fictional permission is withdrawn.'},
            contextual_sources={'current-source':'Current fictional permission is withdrawn.'},
            **dict(self.options,comparison_kind='decisive_condition'))
        self.assertEqual(variant.sources[0],self.original.sources[0])
        self.assertEqual(variant.sources[1].record,current.record)
        self.assertEqual(variant.sources[1].text,'Current fictional permission is withdrawn.')
        self.assertEqual(binding['changed_context_sources'][0]['record_id'],'current-source')
        self.assertEqual(binding['changed_context_sources'][0]['source_kind'],'RELATIONSHIP')
        self.assertEqual(api.verify_context_contrast(self.original,variant,binding,
            expected_binding_sha256=binding['binding_sha256']),binding)
        with self.assertRaises(IdentityError):
            api.build_context_contrast(self.original,replacements={'situation-id':'A new condition.'},
                contextual_sources={'opaque-source-id':'Changed historical identity.'},**self.options)
        lived = SourceEntry('A fictional Alice event retained as lived.',
            SourceRecord('lived-source','fictional-family','AEXP','A-EXP',
                False,True,False,True,True,False))
        original = replace(self.original,sources=(*self.original.sources,lived))
        with self.assertRaises(IdentityError):
            api.build_context_contrast(original,replacements={'situation-id':'A new condition.'},
                contextual_sources={'lived-source':'Changed lived event.'},**self.options)
