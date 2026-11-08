"""Exercise actual worker admission and behavioral reporting with public inputs."""
import ast
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import unittest

from src.alice_personality.identity.codec import canonical
from src.alice_personality.identity.consumer import RenderedResponse
from scripts.eipm.gemma_n0 import n0_semantic_contrasts as contrast
from tests.personality.test_identity_codec import document

WORKER=Path(__file__).resolve().parents[2]/'scripts/eipm/gemma_n0/magnolia_n0_authentic_20261011.py'

def worker_functions():
    if not WORKER.exists():
        raise AssertionError('Authentic N0 worker admission is not implemented')
    tree=ast.parse(WORKER.read_bytes())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)
           and node.name in ('verify_contrast_inputs','directional_report')]
    namespace={'sha256':sha256,'canonical':canonical}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(WORKER),'exec'),namespace)
    return namespace

class AuthenticN0WorkerTests(unittest.TestCase):
    def setUp(self):
        self.functions=worker_functions()
        self.original=document()
        self.variant,self.binding=contrast.build_context_contrast(self.original,
            replacements={'situation-id':'A fictional person considers disclosing the unrelated rumor.'},
            parent_case_id='parent',parent_family_id='family',parent_split='TRAIN',
            variant_id='changed',comparison_kind='paraphrase')
        self.control={'schema':'alice-personality-original-n0-contrast-control-v1',
            'variant_id':'original','parent_case_id':'parent',
            'parent_document_sha256':self.original.sha256,'document_sha256':self.original.sha256,
            'parent_split':'TRAIN','independent_qualification':False,'n0_approved':False}
        self.control['binding_sha256']=sha256(canonical(self.control)).hexdigest()
        self.rows=[{'document_id':'original','parent_document_id':'parent','comparison_kind':'original',
                    'split':'TRAIN','document':self.original.to_mapping()},
                   {'document_id':'changed','parent_document_id':'parent','comparison_kind':'paraphrase',
                    'split':'TRAIN','document':self.variant.to_mapping()}]
        self.rows=json.loads(canonical(self.rows))
        self.bindings={'original':self.control,'changed':self.binding}
        self.index={'document_sha256':{'original':self.original.sha256,'changed':self.variant.sha256}}

    def admit(self,rows=None,bindings=None):
        return self.functions['verify_contrast_inputs']({'parent':self.original},
            self.rows if rows is None else rows,self.bindings if bindings is None else bindings,
            self.index,contrast)

    def test_original_and_actual_changes_admitted_but_relabelled_baseline_refused(self):
        self.assertEqual(set(self.admit()),{'original','changed'})
        rows=deepcopy(self.rows);rows[0]['document']=self.variant.to_mapping()
        with self.assertRaises((AssertionError,ValueError)):
            self.admit(rows=rows)
        rows=deepcopy(self.rows);rows[1]['split']='FINAL'
        with self.assertRaises((AssertionError,ValueError)):
            self.admit(rows=rows)

    def test_target_injection_and_changed_candidate_cannot_reach_source_producer(self):
        rows=deepcopy(self.rows);rows[1]['targets']={'preferences':[1,0]}
        with self.assertRaises((AssertionError,ValueError)):
            self.admit(rows=rows)
        rows=deepcopy(self.rows);rows[1]['document']['candidates'][0]['text']='Changed response meaning'
        with self.assertRaises((AssertionError,ValueError)):
            self.admit(rows=rows)

    def test_reports_every_wrong_governing_region_without_repairing_output(self):
        candidate_ids=['a','b','c']
        decision={'co_valid_probabilities':[[.8,.9,.1]],'contraindications':[[.1,.2,.9]],
                  'preferences':[[.6,.3,.1]]}
        targets={'review':{'candidate_regions':[{'candidate_id':identifier,
            'co_valid_region':'NEGATIVE','contraindication_region':'POSITIVE'} for identifier in candidate_ids]}}
        before=canonical(decision)
        result=self.functions['directional_report'](candidate_ids,decision,targets)
        self.assertEqual(result['region_errors'],4)
        self.assertEqual(result['region_comparisons'],6)
        self.assertEqual(canonical(decision),before)
        self.assertFalse(result['independent_qualification'])

    def test_actual_text_callback_declares_absent_artifact_and_preserves_intent(self):
        tree=ast.parse(WORKER.read_bytes())
        callbacks=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='verbatim']
        self.assertEqual(len(callbacks),1)
        namespace={'RenderedResponse':RenderedResponse}
        exec(compile(ast.fix_missing_locations(ast.Module(body=callbacks,type_ignores=[])),str(WORKER),'exec'),namespace)
        envelope={'preferred_candidate_ids':['a'],'response_text':'Fictional candidate wording',
            'preference_selection':{'intents':[{'candidate_id':'a','voice_controls':{'tempo':.3}}]}}
        before=canonical(envelope)
        rendered=namespace['verbatim'](envelope)
        self.assertIsInstance(rendered,RenderedResponse)
        self.assertEqual(rendered.text,envelope['response_text'])
        self.assertEqual(rendered.delivery,{'tempo':.3})
        self.assertIsNone(rendered.artifact_sha256)
        self.assertEqual(canonical(envelope),before)

if __name__=='__main__':unittest.main()
