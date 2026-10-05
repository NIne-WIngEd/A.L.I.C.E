import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch
from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError
from scripts.mfm import analyze_v16_decision_fit as analysis


class LanguageAnalysisTests(unittest.TestCase):
    def test_field_spans_follow_nested_scalars_and_unicode_escaping(self):
        body={'dispositions':[{'evidence_refs':['source-α'], 'action':'defer'}],
              'proposals':[{'proposal':{'value_text':'Line\n"é"', 'uncertainty_ref':None, 'confidence':0.4},
                            'anchors':[{'start_byte':0,'end_byte':2}]}], 'schema':'test'}
        text, spans=analysis.scalar_spans(body)
        self.assertEqual(text.encode(),canonical_json_bytes(body))
        excerpts={kind:[text[start:end] for start,end,k in spans if k==kind] for kind in {s[2] for s in spans}}
        self.assertIn('"source-α"',excerpts['reference_identity'])
        self.assertIn('null',excerpts['value'])
        self.assertIn('0.4',excerpts['value'])
        self.assertEqual(excerpts['value_text'],['"Line\\n\\"é\\""'])

    def test_unseen_positive_label_is_not_tokenizer_oov_and_copy_presence_is_only_membership(self):
        groups=['reference_identity','value_text','key','eos']
        counts=analysis.exposure_counts([11,12,13,2],groups,Counter({13:100,2:432}),{11,13})
        self.assertEqual(counts['reference_identity']['no_positive_train_corpus_occurrence'],1)
        self.assertEqual(counts['reference_identity']['no_positive_train_corpus_occurrence_present_in_source'],1)
        self.assertEqual(counts['value_text']['no_positive_train_corpus_occurrence_present_in_source'],0)
        self.assertEqual(counts['key']['no_positive_train_corpus_occurrence'],0)
        with self.assertRaises(CognitiveKernelContractError):
            analysis.exposure_counts([11],groups,Counter(),set())

    def test_frequency_strata_preserve_weighted_field_metrics_in_same_forward(self):
        payload={'labels':SimpleNamespace(flatten=lambda:SimpleNamespace(tolist=lambda:[11,12,13]))}
        measured={'tokens':3, 'groups':{
            'reference_identity|0':{'tokens':1,'mean_loss':8.,'accuracy':0.},
            'reference_identity|1-9':{'tokens':2,'mean_loss':2.,'accuracy':0.5}}}
        with patch.object(analysis.parent_analysis,'measure',return_value=measured) as forward:
            row=analysis.measure_with_exposure(None,payload,['reference_identity']*3,Counter({12:2,13:4}))
        forward.assert_called_once_with(None,payload,['reference_identity|0','reference_identity|1-9','reference_identity|1-9'])
        self.assertEqual(row['groups']['reference_identity']['mean_loss'],4.)
        self.assertAlmostEqual(row['groups']['reference_identity']['accuracy'],1/3)
        self.assertEqual(row['groups_by_positive_train_corpus_frequency']['reference_identity']['0']['tokens'],1)

    def test_privacy_admission_precedes_new_component_or_output_reads(self):
        with patch.object(analysis.source,'load',side_effect=CognitiveKernelContractError('privacy rejected')):
            with self.assertRaisesRegex(CognitiveKernelContractError,'privacy rejected'):
                analysis.run(SimpleNamespace())


if __name__=='__main__':
    unittest.main()
