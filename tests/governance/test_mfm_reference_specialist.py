"""Actual tiny native decoder/head mechanics; no learned selection claim."""
import copy
from hashlib import sha256
import importlib
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from cognitive_kernel.canonical import canonical_sha256
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from cognitive_kernel.formation_v1_specialist import SpecialistConfig


SUBJECT='proposals[].proposal.subject_ref'
EVIDENCE='proposals[].proposal.evidence_refs'
GOVERNANCE=['proposals[].proposal.contradicts','proposals[].proposal.target_refs','dispositions[].target_refs']
CITATIONS=[EVIDENCE,'proposals[].proposal.anchors[].ref_id','dispositions[].evidence_refs']


class ReferenceSpecialistTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=SpecialistConfig(4,19,8,1,2,16,0,2,1,3)
        self.parent=FormationDecisionSpecialist(self.config).eval()
        self.parent_path=self.root/'actual-local-parent.safetensors'
        save_file(self.parent.state_dict(),str(self.parent_path))
        self.parent_pin=sha256(self.parent_path.read_bytes()).hexdigest()
        self.config_pin=canonical_sha256(self.config.record())
        self.binding=dict(caller_sha256='a'*64,candidate_inventory_sha256='b'*64,
            processed_view_sha256='c'*64,provenance_sha256='d'*64)
        self.states=torch.arange(24,dtype=torch.float32).reshape(1,6,4)/10
        self.mask=torch.tensor([[1,1,1,1,1,0]])
        self.inputs=torch.tensor([[2,3,4,5,6]])
        self.candidates=[
            dict(slot=0,ref_id='source-a',kind='evidence',origin='opened_source',namespace='owner-memory',
                eligible_paths=CITATIONS+GOVERNANCE,meaning_token_indices=[0,1,1],index_token_indices=[2],independent_observation=False),
            dict(slot=1,ref_id='same-person',kind='source_subject',origin='source_subject',namespace='owner-memory',
                eligible_paths=[SUBJECT],meaning_token_indices=[3],index_token_indices=[4],independent_observation=False),
            dict(slot=2,ref_id='same-person',kind='entity',origin='context_target',namespace='owner-memory',
                eligible_paths=[SUBJECT,'proposals[].episode.participant_refs','proposals[].relationship_counterpart_ref']+GOVERNANCE,
                meaning_token_indices=[1,3],index_token_indices=[0,4],independent_observation=False)]
        self.requests=[dict(pointer='/proposals/0/proposal/subject_ref',candidate_path=SUBJECT,
            eligible_candidate_indices=[1,2],query_index=1,decoder_input_index=2),
            dict(pointer='/proposals/0/proposal/evidence_refs/0',candidate_path=EVIDENCE,
            eligible_candidate_indices=[0],query_index=2,decoder_input_index=3)]
        self.eligibility=torch.tensor([[False,True,True],[True,False,False]])

    def module(self):
        name='cognitive_kernel.formation_reference_specialist'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing native MFM reference producer')
        return importlib.import_module(name)

    def model(self):
        return self.module().FormationReferenceSpecialist.from_parent_safetensors(self.config,self.parent_path,
            parent_sha256=self.parent_pin,config_sha256=self.config_pin).eval()

    def packets(self,model,mode='generated',candidates=None):
        hidden=model.decision_states(base_states=self.states,source_mask=self.mask,input_ids=self.inputs,
            binding=self.binding,mode=mode)
        keys=model.candidate_keys(base_states=self.states,source_mask=self.mask,
            candidates=self.candidates if candidates is None else candidates,binding=self.binding)
        return hidden,keys

    def test_actual_parent_load_preserves_every_inherited_tensor_and_lexical_logits(self):
        model=self.model()
        for key,value in self.parent.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
        with torch.inference_mode():
            expected=self.parent(base_states=self.states,source_mask=self.mask,input_ids=self.inputs)
            actual=model(base_states=self.states,source_mask=self.mask,input_ids=self.inputs)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertEqual(actual.shape,(1,5,19))
        self.assertEqual(model.parent_binding['parent_weight_file_sha256'],self.parent_pin)
        self.assertEqual(model.parent_binding['reference_updates_performed_by_migration'],0)

    def test_corrupt_parent_bytes_config_keys_shapes_dtype_or_nan_refuse_without_partial_load(self):
        model=self.model()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(ValueError):
            model.load_parent_safetensors(self.parent_path,parent_sha256='0'*64,config_sha256=self.config_pin)
        with self.assertRaises(ValueError):
            model.load_parent_safetensors(self.parent_path,parent_sha256=self.parent_pin,config_sha256='0'*64)
        state=self.parent.state_dict()
        variants=[{k:v for k,v in state.items() if k!='output_bias'},dict(state,foreign=torch.ones(1)),
            dict(state,output_bias=torch.ones(18)),{k:v.to(torch.bfloat16) for k,v in state.items()},
            dict(state,output_bias=torch.full((19,),float('nan')))]
        for i,value in enumerate(variants):
            with self.subTest(variant=i):
                path=self.root/f'corrupt-{i}.safetensors';save_file(value,str(path))
                with self.assertRaises(ValueError):
                    model.load_parent_safetensors(path,parent_sha256=sha256(path.read_bytes()).hexdigest(),config_sha256=self.config_pin)
                for key,tensor in before.items():
                    torch.testing.assert_close(model.state_dict()[key],tensor,atol=0,rtol=0)

    def test_native_head_preserves_actual_cached_lexical_path_and_strict_full_state_reload(self):
        from cognitive_kernel.formation_incremental_decoder import CachedFormationDecoder
        model=self.model()
        restored=self.module().FormationReferenceSpecialist(self.config).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key],value,atol=0,rtol=0)
        with torch.inference_mode():
            cache=CachedFormationDecoder(restored,base_states=self.states,source_mask=self.mask)
            for i,token in enumerate(self.inputs[0].tolist()):
                expected=self.parent.next_token_logits(base_states=self.states,source_mask=self.mask,
                    input_ids=self.inputs[:,:i+1])
                torch.testing.assert_close(cache.step(token),expected,atol=2e-5,rtol=2e-5)
            labels=torch.tensor([[3,4,5,6,1]])
            _,expected_loss=self.parent(base_states=self.states,source_mask=self.mask,input_ids=self.inputs,labels=labels)
            _,actual_loss=restored(base_states=self.states,source_mask=self.mask,input_ids=self.inputs,labels=labels)
            torch.testing.assert_close(actual_loss,expected_loss,atol=0,rtol=0)

    def test_candidate_pooling_counts_unique_indices_separately_and_rejects_masked_source(self):
        model=self.model()
        _,keys=self.packets(model)
        deduplicated=copy.deepcopy(self.candidates)
        deduplicated[0]['meaning_token_indices']=[0,1]
        same=model.candidate_keys(base_states=self.states,source_mask=self.mask,candidates=deduplicated,binding=self.binding)
        torch.testing.assert_close(keys['keys'],same['keys'],atol=0,rtol=0)
        self.assertEqual(keys['slots'],[0,1,2])
        for field in ('meaning_token_indices','index_token_indices'):
            bad=copy.deepcopy(self.candidates);bad[0][field]=[5]
            with self.assertRaises(ValueError):
                model.candidate_keys(base_states=self.states,source_mask=self.mask,candidates=bad,binding=self.binding)

    def test_slot_preserving_permutation_changes_columns_only_and_keeps_global_prediction(self):
        model=self.model();hidden,keys=self.packets(model)
        logits=model.reference_logits(hidden,self.requests,keys,self.eligibility)
        prediction=model.predict_references(hidden,self.requests,keys,self.eligibility)
        order=[2,0,1]
        _,reordered=self.packets(model,candidates=[self.candidates[i] for i in order])
        other=model.reference_logits(hidden,self.requests,reordered,self.eligibility[:,order])
        torch.testing.assert_close(other,logits[:,order],atol=1e-6,rtol=1e-6)
        self.assertEqual(model.predict_references(hidden,self.requests,reordered,self.eligibility[:,order]),prediction)
        self.assertEqual(prediction['binding'],self.binding)
        self.assertEqual(prediction['rows'][1],{'pointer':self.requests[1]['pointer'],'candidate_ordinal':0})

    def test_same_identity_different_roles_remains_distinct_and_wrong_role_or_mask_refuses(self):
        model=self.model();hidden,keys=self.packets(model)
        self.assertEqual([row['ref_id'] for row in keys['candidates']].count('same-person'),2)
        logits=model.reference_logits(hidden,self.requests,keys,self.eligibility)
        self.assertTrue(torch.isneginf(logits[0,0]))
        self.assertTrue(torch.isfinite(logits[0,1:]).all())
        bad=self.eligibility.clone();bad[1,1]=True
        with self.assertRaises(ValueError):model.reference_logits(hidden,self.requests,keys,bad)
        wrong=copy.deepcopy(self.candidates);wrong[1]['eligible_paths']=[EVIDENCE]
        with self.assertRaises(ValueError):
            model.candidate_keys(base_states=self.states,source_mask=self.mask,candidates=wrong,binding=self.binding)
        duplicate=copy.deepcopy(self.candidates);duplicate[2]=dict(duplicate[1],slot=2)
        with self.assertRaises(ValueError):
            model.candidate_keys(base_states=self.states,source_mask=self.mask,candidates=duplicate,binding=self.binding)

    def test_exact_score_ties_keep_the_lowest_global_ordinal_across_key_permutations(self):
        model=self.model();hidden,keys=self.packets(model)
        with torch.no_grad():
            model.reference_query.weight.zero_();model.reference_query.bias.zero_()
            model.reference_path.weight.zero_()
        expected=model.predict_references(hidden,self.requests,keys,self.eligibility)
        self.assertEqual(expected['rows'][0]['candidate_ordinal'],1)
        order=[2,0,1]
        _,reordered=self.packets(model,candidates=[self.candidates[i] for i in order])
        self.assertEqual(model.predict_references(hidden,self.requests,reordered,self.eligibility[:,order]),expected)

    def test_query_uses_predictor_j_not_operand_input_j_plus_one_and_completed_state(self):
        model=self.model();hidden,keys=self.packets(model)
        self.assertEqual(hidden['conditioning'],'completed-decision-conditioned')
        self.assertEqual(hidden['completion_index'],4)
        with torch.no_grad():
            model.reference_query.weight.zero_();model.reference_query.bias.zero_()
            model.reference_query.weight[:,:8]=torch.eye(8)
            model.reference_path.weight.zero_()
        requests=[self.requests[0]];mask=self.eligibility[:1]
        scores=model.reference_logits(hidden,requests,keys,mask)
        expected=hidden['hidden'][0,1] @ keys['keys'].T/(8**.5)
        torch.testing.assert_close(scores[0,1:],expected[1:],atol=1e-6,rtol=1e-6)
        wrong=copy.deepcopy(requests);wrong[0]['decoder_input_index']=1
        with self.assertRaises(ValueError):model.reference_logits(hidden,wrong,keys,mask)
        changed=copy.deepcopy(requests);changed[0]['query_index']=2;changed[0]['decoder_input_index']=3
        shifted=model.reference_logits(hidden,changed,keys,mask)
        self.assertFalse(torch.equal(scores[0,1:],shifted[0,1:]))
        with torch.no_grad():
            model.reference_query.weight.zero_()
            model.reference_query.weight[:,8:]=torch.eye(8)
        completed=model.reference_logits(hidden,requests,keys,mask)
        expected_completion=hidden['hidden'][0,4] @ keys['keys'].T/(8**.5)
        torch.testing.assert_close(completed[0,1:],expected_completion[1:],atol=1e-6,rtol=1e-6)

    def test_stale_bindings_duplicate_requests_no_eligible_and_optional_null_refuse_or_remain_empty(self):
        model=self.model();hidden,keys=self.packets(model)
        stale=dict(keys,binding=dict(self.binding,provenance_sha256='f'*64))
        with self.assertRaises(ValueError):model.reference_logits(hidden,self.requests,stale,self.eligibility)
        with self.assertRaises(ValueError):
            model.reference_logits(hidden,[self.requests[0],self.requests[0]],keys,self.eligibility)
        impossible=copy.deepcopy(self.requests[:1]);impossible[0]['eligible_candidate_indices']=[]
        with self.assertRaises(ValueError):
            model.reference_logits(hidden,impossible,keys,torch.zeros((1,3),dtype=torch.bool))
        unavailable=[dict(self.requests[0],pointer='/proposals/0/proposal/uncertainty_ref',
            candidate_path='proposals[].proposal.uncertainty_ref',eligible_candidate_indices=[])]
        with self.assertRaises(ValueError):
            model.reference_logits(hidden,unavailable,keys,torch.zeros((1,3),dtype=torch.bool))
        empty=model.predict_references(hidden,[],keys,torch.zeros((0,3),dtype=torch.bool))
        self.assertEqual(empty,{'binding':self.binding,'rows':[]})

    def test_request_pointer_requires_a_concrete_operand_leaf_not_a_template_or_array(self):
        model=self.model();hidden,keys=self.packets(model)
        for request,mask,pointer in ((self.requests[0],self.eligibility[:1],'/proposals[]/proposal/subject_ref'),
                                    (self.requests[1],self.eligibility[1:],'/proposals/0/proposal/evidence_refs'),
                                    (self.requests[0],self.eligibility[:1],'/proposals/01/proposal/subject_ref')):
            with self.subTest(pointer=pointer):
                with self.assertRaises(ValueError):
                    model.reference_logits(hidden,[dict(request,pointer=pointer)],keys,mask)

    def test_only_explicit_authored_ordinals_form_loss_and_real_gradients_reach_native_model(self):
        model=self.model();model.train();hidden,keys=self.packets(model,mode='teacher_forced')
        authored=[{'pointer':self.requests[0]['pointer'],'candidate_ordinal':2},
                  {'pointer':self.requests[1]['pointer'],'candidate_ordinal':0}]
        logits=model.reference_logits(hidden,self.requests,keys,self.eligibility)
        expected=(torch.logsumexp(logits[0,1:],0)-logits[0,2])/2
        loss=model.selection_loss(hidden,self.requests,keys,self.eligibility,authored_ordinals=authored)
        torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-6)
        loss.backward()
        for parameter in (model.reference_query.weight,model.reference_meaning_key.weight,
                          model.reference_index_key.weight,model.source_projection.weight,model.token_embedding.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(),0)
        for wrong in ([],[{'pointer':self.requests[0]['pointer'],'candidate_ordinal':None}],
                      [dict(authored[0],candidate_ordinal=0),authored[1]]):
            with self.assertRaises(ValueError):
                model.selection_loss(hidden,self.requests,keys,self.eligibility,authored_ordinals=wrong)
        generated,_=self.packets(model,mode='generated')
        with self.assertRaises(ValueError):
            model.selection_loss(generated,self.requests,keys,self.eligibility,authored_ordinals=authored)

    def test_generated_query_and_candidate_keys_do_not_receive_labels_or_backprop_to_frozen_sources(self):
        model=self.model();source=self.states.clone().requires_grad_()
        hidden=model.decision_states(base_states=source,source_mask=self.mask,input_ids=self.inputs,
            binding=self.binding,mode='teacher_forced')
        keys=model.candidate_keys(base_states=source,source_mask=self.mask,candidates=self.candidates,binding=self.binding)
        authored=[{'pointer':self.requests[0]['pointer'],'candidate_ordinal':2},
                  {'pointer':self.requests[1]['pointer'],'candidate_ordinal':0}]
        model.selection_loss(hidden,self.requests,keys,self.eligibility,authored_ordinals=authored).backward()
        self.assertIsNone(source.grad)
        with self.assertRaises(TypeError):
            model.decision_states(base_states=self.states,source_mask=self.mask,input_ids=self.inputs,
                binding=self.binding,mode='generated',labels=torch.ones_like(self.inputs))
        with self.assertRaises(TypeError):
            model.candidate_keys(base_states=self.states,source_mask=self.mask,candidates=self.candidates,
                binding=self.binding,authored_ordinals=authored)


class ReferenceAssemblyTests(unittest.TestCase):
    """Controlled upstream logits, real native head/consumer; mechanics only."""
    @classmethod
    def setUpClass(cls):
        from tests.governance.test_mfm_reference_decision import NativeProcessor
        from tests.governance.test_mfm_world_caller_v3 import build
        from tests.governance.test_mfm_joint_input import source_identity
        from scripts.mfm import formation_joint_input_v4 as J, formation_source_alignment as S
        torch.set_num_threads(1)
        cls.processor=NativeProcessor()
        bundle=build()[5];cls.caller=bundle['caller']
        policy=J.normative_joint_render_policy_v4();identity=source_identity()
        _,cls.joint=J.joint_source_batch_with_alignment_v4(cls.processor,cls.caller,bundle['meaning'],
            caller_sha256=cls.caller['record_sha256'],meaning_sha256=bundle['meaning']['record_sha256'],
            render_policy=policy,render_policy_sha256=policy['record_sha256'],
            processor_binding=S.processor_binding(cls.processor),source_identity=identity,
            source_identity_sha256=identity['record_sha256'],max_source_tokens=200000)
        cls.binding={'caller_sha256':cls.caller['record_sha256'],
            'candidate_inventory_sha256':cls.joint['candidate_inventory_sha256'],**cls.joint['cache']}

    def setUp(self):
        from cognitive_kernel.formation_reference_specialist import FormationReferenceSpecialist
        torch.manual_seed(193)
        self.model=FormationReferenceSpecialist(SpecialistConfig(4,len(self.processor.tokenizer),8,1,2,8192,0,2,1,3))
        # Exact ties exercise the actual native scoring/address path, not a
        # mocked prediction or claim that untrained weights learned a choice.
        with torch.no_grad():
            self.model.reference_query.weight.zero_();self.model.reference_query.bias.zero_()
            self.model.reference_path.weight.zero_()
        n=len(self.joint['input_ids'])
        self.states=(torch.arange(n*4,dtype=torch.float32).reshape(1,n,4)%37/37).requires_grad_()
        self.mask=torch.tensor([self.joint['attention_mask']])
        self.args=dict(processor=self.processor,base_states=self.states,source_mask=self.mask,
            caller=self.caller,caller_sha256=self.caller['record_sha256'],joint_alignment=self.joint,
            joint_alignment_sha256=self.joint['record_sha256'],expected_binding=self.binding,
            artifact_sha256='a'*64,inference_run_id='controlled-upstream-real-head')
        self.calls=[]

    def api(self):
        method=getattr(self.model,'generate_reference_decision',None)
        self.assertIsNotNone(method,'missing actual native lexical-to-reference-to-grounding assembly')
        return method

    def stream(self,raw,*,nonfinite_at=None):
        ids=self.processor.tokenizer(raw,add_special_tokens=False)['input_ids']
        def controlled(model,*,base_states,source_mask,input_ids):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            self.assertFalse(any(module.training for module in model.modules()))
            prefix=input_ids[0].tolist();self.calls.append(prefix)
            self.assertEqual(prefix,[2,*ids[:len(prefix)-1]])
            torch.rand(1)  # Forces the method's RNG preservation control.
            if nonfinite_at is not None and len(prefix)-1==nonfinite_at:
                return torch.full((1,model.config.vocabulary_size),float('nan'))
            logits=torch.full((1,model.config.vocabulary_size),-10.)
            token=ids[len(prefix)-1] if len(prefix)-1<len(ids) else 1
            logits[0,token]=10.
            return logits
        self.model.next_token_logits=types.MethodType(controlled,self.model)
        return ids

    def record(self,*,bad_anchor=False):
        from cognitive_kernel.canonical import canonical_json_bytes
        from cognitive_kernel.formation_contracts import FormationProposal, FormationEvidenceAnchor
        from cognitive_kernel.formation_semantics_v16 import FormationProposalV16
        from cognitive_kernel import formation_reference_decision as C
        context,opened,registry=C.caller_api.caller_input_from_record(self.caller)
        candidates=C.caller_api.selection_candidates(context,opened,registry)
        evidence=next(c for c in candidates if c['origin']=='opened_source')
        subject=next(c for c in candidates if c['origin']=='source_subject')
        raw=dict(opened)[evidence['ref_id']]
        base=FormationProposal('controlled-proposal','claim','host',subject['ref_id'],'controlled-value',
            (evidence['ref_id'],),value_text='Controlled lexical mechanics; no learned selection claim.',
            anchors=(FormationEvidenceAnchor(evidence['ref_id'],0,len(raw)+int(bad_anchor)),),
            epistemic_status='hypothetical',confidence=None)
        item=FormationProposalV16(base,context.source_sensitivities[0].minimum).record()
        item['proposal'].pop('proposal_id');item['proposal'].pop('value_ref')
        for name in ('subject_ref',):item['proposal'][name]={'caller_reference':True}
        item['proposal']['evidence_refs']=[{'caller_reference':True}]
        item['proposal']['anchors'][0]['ref_id']={'caller_reference':True}
        decision={'schema':C.SCHEMA,'proposals':[item],'dispositions':[]}
        return decision,canonical_json_bytes(decision).decode('utf-8')

    def test_controlled_own_lexical_stream_reaches_actual_head_and_grounded_bundle_without_updates(self):
        _,raw=self.record();ids=self.stream(raw)
        self.model.train();self.model.source_projection.eval();self.model.decoder.layers[0].linear1.eval()
        modes=[module.training for module in self.model.modules()]
        rng=torch.random.get_rng_state().clone();before={k:v.clone() for k,v in self.model.state_dict().items()}
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=self.api()(**self.args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['validation_status'],'grounded_proposal_only')
        self.assertIsNone(result['failure_stage']);self.assertIsNone(result['validation_error'])
        self.assertEqual(result['generated_token_ids'],ids);self.assertEqual(result['raw_output_text'],raw)
        self.assertTrue(result['eos_observed']);self.assertEqual(len(self.calls),len(ids)+1)
        self.assertEqual(result['reference_predictions']['binding'],self.binding)
        from cognitive_kernel import formation_reference_decision as C
        context,opened,registry=C.caller_api.caller_input_from_record(self.caller)
        candidates=C.caller_api.selection_candidates(context,opened,registry)
        chosen={row['pointer']:row['candidate_ordinal'] for row in result['reference_predictions']['rows']}
        compiled=result['bundle'].proposals[0].base
        self.assertEqual(compiled.subject_ref,candidates[chosen['/proposals/0/proposal/subject_ref']]['ref_id'])
        self.assertEqual(compiled.evidence_refs,(candidates[chosen['/proposals/0/proposal/evidence_refs/0']]['ref_id'],))
        self.assertEqual(compiled.anchors[0].ref_id,candidates[chosen['/proposals/0/proposal/anchors/0/ref_id']]['ref_id'])
        self.assertEqual(compiled.value_text,'Controlled lexical mechanics; no learned selection claim.')
        self.assertEqual([module.training for module in self.model.modules()],modes)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
        for name,value in before.items():torch.testing.assert_close(self.model.state_dict()[name],value,atol=0,rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertIsNone(self.states.grad)

    def test_own_empty_reference_stream_still_aligns_compiles_and_grounds(self):
        from cognitive_kernel.canonical import canonical_json_bytes
        from cognitive_kernel.formation_reference_decision import SCHEMA
        raw=canonical_json_bytes({'schema':SCHEMA,'proposals':[],'dispositions':[]}).decode()
        ids=self.stream(raw);result=self.api()(**self.args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['validation_status'],'grounded_proposal_only')
        self.assertEqual(result['reference_predictions'],{'binding':self.binding,'rows':[]})
        self.assertFalse(result['bundle'].proposals);self.assertEqual(result['generated_token_ids'],ids)

    def test_capped_malformed_duplicate_nonfinite_and_noncanonical_streams_preserve_first_failure(self):
        _,canonical=self.record()
        cases=[(canonical,3,'generation'),('not-json',None,'parse'),
            ('{"schema":"first","schema":"second"}',None,'parse'),('{"n":NaN}',None,'parse'),
            (canonical+'\n',None,'alignment')]
        for raw,budget,stage in cases:
            with self.subTest(stage=stage,raw=raw[:30]):
                self.calls=[];ids=self.stream(raw);cap=budget or len(ids)+1
                result=self.api()(**self.args,max_new_tokens=cap)
                retained=ids[:cap] if budget else ids
                self.assertEqual(result['validation_status'],'invalid');self.assertIsNone(result['bundle'])
                self.assertEqual(result['generated_token_ids'],retained)
                self.assertEqual(result['raw_output_text'],self.processor.tokenizer.decode(retained,
                    skip_special_tokens=False,clean_up_tokenization_spaces=False))
                self.assertEqual(result['eos_observed'],budget is None)
                self.assertEqual(result['failure_stage'],stage);self.assertTrue(result['validation_error'])

    def test_real_grounder_refuses_generated_bad_span_without_repair(self):
        _,raw=self.record(bad_anchor=True);ids=self.stream(raw)
        result=self.api()(**self.args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['failure_stage'],'compile_ground');self.assertIsNone(result['bundle'])
        self.assertEqual(result['generated_token_ids'],ids);self.assertEqual(result['raw_output_text'],raw)
        self.assertTrue(result['eos_observed']);self.assertTrue(result['reference_predictions']['rows'])

    def test_source_processor_budget_and_fp32_preflight_refuse_before_any_lexical_call(self):
        _,raw=self.record();ids=self.stream(raw);method=self.api()
        for changes in ({'source_mask':self.mask[:,:-1]},
                        {'expected_binding':dict(self.binding,provenance_sha256='f'*64)},
                        {'max_new_tokens':self.model.config.max_target_tokens+1},
                        {'max_new_tokens':True}):
            with self.subTest(changes=list(changes)):
                result=method(**{**self.args,'max_new_tokens':len(ids)+1,**changes})
                self.assertEqual(result['failure_stage'],'preflight');self.assertEqual(result['generated_token_ids'],[])
                self.assertFalse(self.calls)
        self.model.bfloat16()
        result=method(**self.args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['failure_stage'],'preflight');self.assertFalse(self.calls)

    def test_nonfinite_own_logits_retain_emitted_prefix_and_inference_api_excludes_teaching(self):
        _,raw=self.record();ids=self.stream(raw,nonfinite_at=2);method=self.api()
        modes=[module.training for module in self.model.modules()];rng=torch.random.get_rng_state().clone()
        result=method(**self.args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['failure_stage'],'generation');self.assertEqual(result['generated_token_ids'],ids[:2])
        self.assertEqual(result['raw_output_text'],self.processor.tokenizer.decode(ids[:2],skip_special_tokens=False,
            clean_up_tokenization_spaces=False))
        self.assertFalse(result['eos_observed']);self.assertIsNone(result['bundle'])
        self.assertEqual([module.training for module in self.model.modules()],modes)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
        for name in ('decision','teacher','complete_pair','authored_ordinals'):
            with self.subTest(name=name),self.assertRaises(TypeError):
                method(**self.args,max_new_tokens=1,**{name:{}})

    def test_failure_raw_decode_is_inside_the_rng_preservation_boundary(self):
        _,raw=self.record();self.stream(raw);method=self.api()
        original=self.processor.tokenizer.decode
        def consuming_decode(*args,**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            torch.rand(1)
            return original(*args,**kwargs)
        rng=torch.random.get_rng_state().clone()
        with patch.object(self.processor.tokenizer,'decode',side_effect=consuming_decode), \
                torch.autocast('cpu',dtype=torch.bfloat16):
            result=method(**self.args,max_new_tokens=3)
        self.assertEqual(result['failure_stage'],'generation')
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)

    def test_actual_inherited_lexical_tensor_path_emits_and_retains_its_own_capped_prefix(self):
        # Real lexical tensors, controlled output bias, no upstream mock.
        token=self.processor.tokenizer('x',add_special_tokens=False)['input_ids'][0]
        with torch.no_grad():
            self.model.output_bias.fill_(-1000.);self.model.output_bias[token]=1000.
        result=self.api()(**self.args,max_new_tokens=3)
        self.assertEqual(result['generated_token_ids'],[token]*3)
        self.assertEqual(result['raw_output_text'],'xxx');self.assertFalse(result['eos_observed'])
        self.assertEqual(result['failure_stage'],'generation');self.assertIsNone(result['bundle'])

    def test_failed_wire_decode_is_retained_once_without_a_speculative_retry(self):
        _,raw=self.record();ids=self.stream(raw);method=self.api()
        modes=[module.training for module in self.model.modules()];rng=torch.random.get_rng_state().clone()
        with patch.object(self.processor.tokenizer,'decode',side_effect=ValueError('controlled wire decode failure')) as decode:
            result=method(**self.args,max_new_tokens=len(ids)+1)
        decode.assert_called_once()
        self.assertEqual(result['failure_stage'],'decode');self.assertIsNone(result['raw_output_text'])
        self.assertEqual(result['generated_token_ids'],ids);self.assertTrue(result['eos_observed'])
        self.assertIn('controlled wire decode failure',result['validation_error'])
        self.assertEqual([module.training for module in self.model.modules()],modes)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
