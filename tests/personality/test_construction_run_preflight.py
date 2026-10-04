"""Public fictional run preflight; never private source or compute admission."""
from dataclasses import replace
from hashlib import sha256
import unittest

from scripts.eipm.gemma_n0.construct_personality import document_text_banks, token_cost_profile, verify_reference_families
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.identity.contracts import IdentityModelConfig
from tests.personality.test_identity_codec import document, Tokenizer, Backbone


class RunPreflightTests(unittest.TestCase):
    def test_profile_uses_exact_actual_producer_semantic_texts_and_deduplicates(self):
        doc=document()
        tokenizer=Tokenizer()
        p=FrozenFrameProducer(tokenizer,Backbone(),IdentityModelConfig(8,3,16,4),_fixture=True)
        p.produce(doc)
        texts={text for bank in document_text_banks(doc).values() for text in bank}
        self.assertEqual(texts,set(tokenizer.seen))
        profile=token_cost_profile([doc,doc],Tokenizer(),layers=3,width=8,max_tokens=64)
        self.assertEqual(profile['unique_entries'],len(texts))
        self.assertEqual(profile['unique_source_tokens'],4*len(texts))
        self.assertEqual(profile['state_value_bytes'],len(texts)*4*3*8*2)
        self.assertEqual(profile['document_count'],2)
        self.assertFalse(profile['measured_peak_memory'])

    def test_actual_compiled_family_and_original_permissions_cannot_be_redeclared(self):
        doc=document();entry=doc.sources[0];rid=entry.record.record_id;family=entry.record.source_family_id
        original={'provenance_class':'E0','claim_subject_partitions':['source_person:elaina'],
            'loss_mask':{'conditional_identity':True,'excluded_from_identity_loss':False}}
        row={'draft_workitem':{'record_id':rid,'source_family_id':family,'original_split':'train',
            'source_declarations':{'historical_truth_allowed':True,'runtime_behavioral_prior_allowed':False}},
            'source_kind_for_future_document':'E0','original_source_record':original}
        from scripts.eipm.gemma_n0.construct_personality import canonical
        body=sha256(canonical(original)).hexdigest();row['original_source_record_sha256']=body
        protocol={'source_body_sha256':{rid:body},'source_render_sha256':{rid:sha256(entry.text.encode()).hexdigest()}}
        allocation={'reference_family_ids':[family],'family_splits':{family:'TRAIN'}}
        compiled=[{'family_id':family,'split':'train','record_ids':[rid]}]
        verify_reference_families(doc,allocation,compiled,{rid:row},protocol)
        with self.assertRaises(ValueError):
            verify_reference_families(doc,allocation,[],{rid:row},protocol)
        original['loss_mask']['excluded_from_identity_loss']=True
        body=sha256(canonical(original)).hexdigest();row['original_source_record_sha256']=body;protocol['source_body_sha256'][rid]=body
        with self.assertRaises(ValueError):
            verify_reference_families(doc,allocation,compiled,{rid:row},protocol)
