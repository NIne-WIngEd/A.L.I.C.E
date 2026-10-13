"""Tiny untrained owned copy mechanics; these controls do not prove learning."""
import copy
from hashlib import sha256
import importlib
import importlib.util
import math
from pathlib import Path
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file
from cognitive_kernel.canonical import canonical_sha256, canonical_json_bytes
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from cognitive_kernel.formation_reference_specialist import FormationReferenceSpecialist, TASK as REF_TASK
from cognitive_kernel.formation_v1_specialist import SpecialistConfig


class OriginalCopySpecialistTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(53)
        self.config = SpecialistConfig(4,19,8,1,2,16,0,2,1,3)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        lexical = FormationDecisionSpecialist(self.config)
        self.lexical_path = self.root/'lexical.safetensors'
        save_file(lexical.state_dict(), str(self.lexical_path))
        self.reference = FormationReferenceSpecialist.from_parent_safetensors(self.config, self.lexical_path,
            parent_sha256=self.pin(self.lexical_path), config_sha256=canonical_sha256(self.config.record()))
        self.parent_path = self.root/'reference.safetensors'
        save_file(self.reference.state_dict(), str(self.parent_path))
        self.config_path = self.root/'config.json'; self.config_path.write_bytes(canonical_json_bytes(self.config.record()))
        import cognitive_kernel.formation_reference_specialist as R
        self.provenance = dict(kind='untrained_fixture', task=REF_TASK, producer_sha256=self.pin(R.__file__),
            weight_file_sha256=self.pin(self.parent_path), config_file_sha256=self.pin(self.config_path),
            config_canonical_sha256=canonical_sha256(self.config.record()), lexical_parent_binding=self.reference.parent_binding)
        self.parent_args = dict(parent_sha256=self.pin(self.parent_path), parent_config_path=self.config_path,
            parent_config_file_sha256=self.pin(self.config_path), parent_task=REF_TASK,
            parent_provenance=self.provenance, parent_provenance_sha256=canonical_sha256(self.provenance))
        self.binding = dict(caller_sha256='a'*64,candidate_inventory_sha256='b'*64,
            processed_view_sha256='c'*64,provenance_sha256='d'*64)
        # Input-only UTF8 geometry for aéb; no target strings/tuples in inventory.
        raw = 'aéb'.encode(); offsets = [0,1,3,4]
        boundaries=[]
        for k,offset in enumerate(offsets):
            left=[] if k==0 else [k-1]; right=[] if k==3 else [k]
            boundaries.append(dict(byte_offset=offset,left_token_indices=left,right_token_indices=right,
                original_byte_window=[raw[i] if 0<=i<len(raw) else 256 for i in range(offset-4,offset+4)],
                coordinate_features=[offset/4,offset/4,float(bool(left)),0.,[0,1,2,1][k]/4,
                    [1,2,1,0][k]/4,float(not left),float(not right)]))
        sources=[dict(source_slot=0,source_ref='opened-a',content_sha256=sha256(raw).hexdigest(),
            original_bytes=4,modality='text',source_token_indices=[0,1,2],boundaries=boundaries),
            dict(source_slot=1,source_ref='opened-empty',content_sha256=sha256(b'').hexdigest(),original_bytes=0,
                modality='text',source_token_indices=[],boundaries=[dict(byte_offset=0,left_token_indices=[],
                    right_token_indices=[],original_byte_window=[256]*8,coordinate_features=[0.]*6+[1.,1.])])]
        self.inventory=dict(schema='mfm-original-copy-input-inventory-v1',reference_inventory={'binding':self.binding},
            joint_alignment_sha256='e'*64,joint_schema='mfm-joint-caller-input-alignment-v4',source_identity={},
            source_geometry={'token_count':4,'attention_mask':[1,1,1,0]},sources=sources)
        self.inventory['record_sha256']=canonical_sha256(self.inventory)
        self.states=torch.arange(16,dtype=torch.float32).reshape(1,4,4)/9
        self.mask=torch.tensor([[1,1,1,0]])
        self.inputs=torch.tensor([[2,3,4,5]])
        self.requests=[dict(pointer='/proposals/0/proposal/value_text/parts/0',query_index=1,decoder_input_index=2)]

    @staticmethod
    def pin(path): return sha256(Path(path).read_bytes()).hexdigest()

    def module(self):
        name='cognitive_kernel.formation_original_copy_specialist'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing actual native original-copy producer')
        return importlib.import_module(name)

    def model(self): return self.module().FormationOriginalCopySpecialist.from_reference_parent_safetensors(
        self.config,self.parent_path,**self.parent_args).eval()

    def packets(self,model,mode='generated',source=None):
        source=self.states if source is None else source
        hidden=model.decision_states(base_states=source,source_mask=self.mask,input_ids=self.inputs,
            binding=self.binding,mode=mode)
        keys=model.copy_keys(base_states=source,source_mask=self.mask,inventory=self.inventory)
        return hidden,keys

    def test_complete_parent_preserves_lexical_reference_and_refuses_lexical_bypass(self):
        model=self.model()
        for name,tensor in self.reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name],tensor,atol=0,rtol=0)
        with self.assertRaises(ValueError): model.load_parent_safetensors(self.lexical_path,
            parent_sha256=self.pin(self.lexical_path),config_sha256=canonical_sha256(self.config.record()))
        with self.assertRaises(ValueError): self.module().FormationOriginalCopySpecialist.from_parent_safetensors(
            self.config,self.lexical_path,parent_sha256=self.pin(self.lexical_path),config_sha256=canonical_sha256(self.config.record()))
        self.assertEqual(model.parent_binding['copy_updates_performed_by_migration'],0)

    def test_parent_migration_is_atomic_for_corruption_and_forged_ancestry(self):
        model=self.model(); before={k:v.clone() for k,v in model.state_dict().items()}
        state=self.reference.state_dict()
        variants=[{k:v for k,v in state.items() if not k.startswith('reference_')},
            dict(state,output_bias=torch.full((19,),float('nan'))),dict(state,foreign=torch.zeros(1)),
            dict(state,output_bias=torch.ones(18)),{k:v.bfloat16() for k,v in state.items()}]
        for i,variant in enumerate(variants):
            path=self.root/f'bad-{i}.safetensors'; save_file(variant,str(path))
            provenance={**self.provenance,'weight_file_sha256':self.pin(path)}
            args={**self.parent_args,'parent_sha256':self.pin(path),'parent_provenance':provenance,
                'parent_provenance_sha256':canonical_sha256(provenance)}
            with self.assertRaises(ValueError): model.load_reference_parent_safetensors(path,**args)
        for changes in ({'parent_task':'language-only'}, {'parent_provenance_sha256':'f'*64},
                        {'parent_config_file_sha256':'f'*64}):
            with self.assertRaises(ValueError): model.load_reference_parent_safetensors(self.parent_path,**{**self.parent_args,**changes})
        for k,v in before.items(): torch.testing.assert_close(model.state_dict()[k],v,atol=0,rtol=0)

    def test_parent_migration_refuses_a_destination_that_would_round_parent_fp32(self):
        model=self.model().bfloat16();before={k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(ValueError):model.load_reference_parent_safetensors(self.parent_path,**self.parent_args)
        for name,tensor in before.items():torch.testing.assert_close(model.state_dict()[name],tensor,atol=0,rtol=0)

    def test_keys_distinguish_same_context_utf8_boundaries_and_mask_empty_source(self):
        model=self.model(); hidden,keys=self.packets(model)
        self.assertEqual(keys['source_keys'].shape,(2,8)); self.assertEqual(keys['boundary_keys'][0].shape,(4,8))
        self.assertFalse(torch.equal(keys['boundary_keys'][0][1],keys['boundary_keys'][0][2]))
        source=model.copy_logits(hidden,self.requests,keys)
        self.assertEqual(source['coordinates'],[0,1]); self.assertTrue(torch.isneginf(source['logits'][0,1]))
        start=model.copy_logits(hidden,self.requests,keys,source_slot=0)
        self.assertEqual(start['coordinates'],[0,1,3,4]); self.assertTrue(torch.isneginf(start['logits'][0,3]))
        end=model.copy_logits(hidden,self.requests,keys,source_slot=0,start_byte=1)
        self.assertEqual(end['coordinates'],[0,1,3,4]); self.assertTrue(torch.isneginf(end['logits'][0,:2]).all())
        prediction=model.predict_copies(hidden,self.requests,keys)
        row=prediction['rows'][0]; self.assertEqual(row['source_slot'],0)
        self.assertIn(row['start_byte'],[0,1,3]);self.assertIn(row['end_byte'],[1,3,4])
        self.assertLess(row['start_byte'],row['end_byte'])

    def test_exact_tuple_alternatives_marginal_excludes_crossed_unselected_spans(self):
        model=self.model()
        with torch.no_grad():
            for layer in (model.copy_source_query,model.copy_start_query,model.copy_end_query):
                layer.weight.zero_();layer.bias.zero_()
        hidden,keys=self.packets(model,mode='teacher_forced')
        authored={'binding':keys['binding'],'rows':[{'pointer':self.requests[0]['pointer'],'alternatives':[
            {'source_slot':0,'start_byte':0,'end_byte':3},{'source_slot':0,'start_byte':1,'end_byte':4}]}]}
        loss=model.copy_loss(hidden,self.requests,keys,authored_copy_tuples=authored)
        self.assertAlmostEqual(loss.item(),-math.log(5/18),places=6)
        self.assertNotAlmostEqual(loss.item(),-math.log(5/9),places=5)
        self.assertIsNone(model.copy_loss(hidden,[],keys,authored_copy_tuples={'binding':keys['binding'],'rows':[]}))
        for alternatives in (None,[],[{'source_slot':0,'start_byte':2,'end_byte':4}],
            [authored['rows'][0]['alternatives'][0]]*2):
            bad=copy.deepcopy(authored);bad['rows'][0]['alternatives']=alternatives
            with self.assertRaises(ValueError):model.copy_loss(hidden,self.requests,keys,authored_copy_tuples=bad)
        with self.assertRaises(ValueError): model.predict_copies(hidden,self.requests,keys)

    def test_owned_gradients_and_frozen_source_detach_and_no_teacher_in_inference(self):
        model=self.model(); source=self.states.clone().requires_grad_()
        hidden,keys=self.packets(model,mode='teacher_forced',source=source)
        authored={'binding':keys['binding'],'rows':[{'pointer':self.requests[0]['pointer'],'alternatives':[
            {'source_slot':0,'start_byte':0,'end_byte':3}]}]}
        model.copy_loss(hidden,self.requests,keys,authored_copy_tuples=authored).backward()
        for name in ('copy_byte_embedding.weight','copy_boundary.0.weight','copy_start_query.weight',
            'copy_end_query.weight','copy_start_key.weight','copy_end_key.weight','copy_marker_query.weight',
            'source_projection.weight','token_embedding.weight'):
            grad=dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(grad.abs().sum().item(),0,name)
        self.assertIsNone(source.grad)
        generated,_=self.packets(model)
        with self.assertRaises(ValueError): model.copy_loss(generated,self.requests,keys,authored_copy_tuples=authored)
        with self.assertRaises(TypeError): model.predict_copies(generated,self.requests,keys,authored_copy_tuples=authored)

    def test_complete_input_geometry_stale_bindings_and_bad_queries_refuse(self):
        model=self.model();hidden,keys=self.packets(model)
        with self.assertRaises(ValueError):model.copy_keys(base_states=self.states[:,:3],source_mask=self.mask[:,:3],inventory=self.inventory)
        with self.assertRaises(ValueError):model.copy_keys(base_states=self.states,source_mask=torch.ones_like(self.mask),inventory=self.inventory)
        stale={**keys,'binding':copy.deepcopy(keys['binding'])};stale['binding']['reference_binding']['caller_sha256']='f'*64
        with self.assertRaises(ValueError):model.copy_logits(hidden,self.requests,stale)
        for request in (dict(self.requests[0],pointer='/proposals/01/proposal/value_text/parts/0'),
            dict(self.requests[0],decoder_input_index=1),dict(self.requests[0],query_index=True)):
            with self.assertRaises(ValueError):model.copy_logits(hidden,[request],keys)
        with self.assertRaises(ValueError):model.copy_logits(hidden,self.requests*2,keys)
        for kwargs in ({'source_slot':1},{'source_slot':0,'start_byte':2},{'start_byte':0},
                       {'source_slot':True},{'source_slot':0,'start_byte':4}):
            with self.assertRaises(ValueError):model.copy_logits(hidden,self.requests,keys,**kwargs)

    def test_unknown_joint_profile_and_changed_utf8_geometry_cannot_be_resealed_into_keys(self):
        model=self.model()
        for kind in ('joint','interior','wrong_bytes'):
            bad=copy.deepcopy(self.inventory)
            if kind=='joint':bad['joint_schema']='not-an-admitted-original-joint-profile'
            elif kind=='interior':
                b=bad['sources'][0]['boundaries'][2];b['byte_offset']=2
                b['coordinate_features'][0]=.5
            else:bad['sources'][0]['boundaries'][1]['original_byte_window'][3]=ord('z')
            bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                model.copy_keys(base_states=self.states,source_mask=self.mask,inventory=bad)

    def test_owned_source_probability_gradients_with_two_distinct_opened_sources(self):
        model=self.model();inventory=copy.deepcopy(self.inventory)
        second=copy.deepcopy(inventory['sources'][0]);second.update(source_slot=1,source_ref='opened-b',source_token_indices=[2])
        for boundary in second['boundaries']:
            if boundary['left_token_indices']:boundary['left_token_indices']=[2]
            if boundary['right_token_indices']:boundary['right_token_indices']=[2]
        inventory['sources'][1]=second
        inventory['record_sha256']=canonical_sha256({k:v for k,v in inventory.items() if k!='record_sha256'})
        hidden=model.decision_states(base_states=self.states,source_mask=self.mask,input_ids=self.inputs,
            binding=self.binding,mode='teacher_forced')
        keys=model.copy_keys(base_states=self.states,source_mask=self.mask,inventory=inventory)
        labels={'binding':keys['binding'],'rows':[{'pointer':self.requests[0]['pointer'],'alternatives':[
            {'source_slot':0,'start_byte':0,'end_byte':3}]}]}
        model.copy_loss(hidden,self.requests,keys,authored_copy_tuples=labels).backward()
        for parameter in (model.copy_source_query.weight,model.copy_source_key.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all());self.assertGreater(parameter.grad.abs().sum().item(),0)

    def test_byte_coordinate_ablation_exposes_equal_token_support_boundary_collision(self):
        model=self.model();inventory=copy.deepcopy(self.inventory)
        row=inventory['sources'][0];row['source_token_indices']=[0]
        for boundary in row['boundaries']:
            if boundary['left_token_indices']:boundary['left_token_indices']=[0]
            if boundary['right_token_indices']:boundary['right_token_indices']=[0]
        inventory['record_sha256']=canonical_sha256({k:v for k,v in inventory.items() if k!='record_sha256'})
        keys=model.copy_keys(base_states=self.states,source_mask=self.mask,inventory=inventory)
        self.assertFalse(torch.equal(keys['boundary_keys'][0][1],keys['boundary_keys'][0][2]))
        with torch.no_grad():model.copy_boundary[0].weight[:,2*self.config.width:].zero_()
        ablated=model.copy_keys(base_states=self.states,source_mask=self.mask,inventory=inventory)
        torch.testing.assert_close(ablated['boundary_keys'][0][1],ablated['boundary_keys'][0][2],atol=0,rtol=0)


class OriginalCopyAssemblyTests(unittest.TestCase):
    """Controlled own lexical choices, actual decoder/heads/consumer; no learning claim."""
    from tests.governance.test_mfm_reference_specialist import ReferenceAssemblyTests as _fixtures
    setUpClass = classmethod(_fixtures.setUpClass.__func__)
    stream = _fixtures.stream

    def setUp(self):
        from cognitive_kernel.formation_original_copy_specialist import FormationOriginalCopySpecialist
        torch.manual_seed(193)
        self.model=FormationOriginalCopySpecialist(SpecialistConfig(4,len(self.processor.tokenizer),8,1,2,8192,0,2,1,3))
        with torch.no_grad():
            self.model.reference_query.weight.zero_();self.model.reference_query.bias.zero_()
            self.model.reference_path.weight.zero_()
            for layer in (self.model.copy_source_query,self.model.copy_start_query,self.model.copy_end_query):
                layer.weight.zero_();layer.bias.zero_()
        n=len(self.joint['input_ids'])
        self.states=(torch.arange(n*4,dtype=torch.float32).reshape(1,n,4)%37/37).requires_grad_()
        self.mask=torch.tensor([self.joint['attention_mask']])
        self.calls=[]

    def api(self):
        method=getattr(self.model,'generate_original_copy_decision',None)
        self.assertIsNotNone(method,'missing actual owned lexical/copy/reference/grounding assembly')
        return method

    def args(self):
        from cognitive_kernel import formation_original_copy_decision as C
        inventory=C.bind_original_copy_inventory(self.caller,self.joint,
            caller_sha256=self.caller['record_sha256'],joint_alignment_sha256=self.joint['record_sha256'],expected_binding=self.binding)
        return dict(processor=self.processor,base_states=self.states,source_mask=self.mask,
            caller=self.caller,caller_sha256=self.caller['record_sha256'],joint_alignment=self.joint,
            joint_alignment_sha256=self.joint['record_sha256'],expected_binding=C.copy_binding(inventory),
            artifact_sha256='a'*64,inference_run_id='untrained-real-original-copy-mechanics')

    def record(self,*,bad_anchor=False,no_copy=False):
        from cognitive_kernel.canonical import canonical_json_bytes
        decision,_=self._fixtures.record(self,bad_anchor=bad_anchor)
        decision['schema']='mfm-formation-decision-v1.6-caller-references-original-spans-v1'
        decision['proposals'][0]['proposal']['value_text']={'parts':[
            {'semantic_text':'Exact first original character: '}, {'original_copy':True}, {'semantic_text':'.'}]}
        if no_copy:decision['proposals'][0]['proposal']['value_text']={'parts':[{'semantic_text':'No declared copy.'}]}
        return decision,canonical_json_bytes(decision).decode()

    def test_own_mixed_stream_real_heads_exact_bytes_compiler_and_modes_rng_no_updates(self):
        method=self.api();args=self.args();_,raw=self.record();ids=self.stream(raw)
        self.model.train();self.model.source_projection.eval();self.model.decoder.layers[0].linear1.eval()
        modes=[m.training for m in self.model.modules()];rng=torch.random.get_rng_state().clone()
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        with torch.autocast('cpu',dtype=torch.bfloat16):result=method(**args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['validation_status'],'grounded_proposal_only',result['validation_error'])
        self.assertIsNone(result['failure_stage']);self.assertEqual(result['generated_token_ids'],ids)
        self.assertEqual(result['raw_output_text'],raw);self.assertTrue(result['eos_observed'])
        self.assertEqual(result['copy_predictions']['rows'],[dict(pointer='/proposals/0/proposal/value_text/parts/1',
            source_slot=0,start_byte=0,end_byte=1)])
        from cognitive_kernel.formation_reference_decision import caller_api
        _,opened,_=caller_api.caller_input_from_record(self.caller)
        literal=opened[0][1][:1].decode()
        self.assertEqual(result['bundle'].proposals[0].base.value_text,'Exact first original character: '+literal+'.')
        self.assertEqual(result['copy_provenance'][0]['original_utf8'],[0,1])
        self.assertEqual(result['copy_provenance'][0]['value_utf8'],[32,33])
        self.assertEqual([m.training for m in self.model.modules()],modes)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
        for k,v in before.items():torch.testing.assert_close(self.model.state_dict()[k],v,atol=0,rtol=0)
        self.assertIsNone(self.states.grad);self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_zero_copy_real_assembly_and_grounding_failure_keep_all_raw_actions(self):
        method=self.api();args=self.args()
        for bad,no_copy in ((False,True),(True,False)):
            self.calls=[];_,raw=self.record(bad_anchor=bad,no_copy=no_copy);ids=self.stream(raw)
            result=method(**args,max_new_tokens=len(ids)+1)
            self.assertEqual(result['failure_stage'],'compile_ground' if bad else None)
            self.assertEqual(result['generated_token_ids'],ids);self.assertEqual(result['raw_output_text'],raw)
            self.assertTrue(result['reference_predictions']['rows'])
            self.assertEqual(len(result['copy_predictions']['rows']),0 if no_copy else 1)

    def test_actual_byte_boundary_head_copies_complete_unicode_escaped_observation(self):
        method=self.api();args=self.args();_,raw=self.record();ids=self.stream(raw)
        # Mechanics intervention on owned weights uses only the admitted endpoint
        # feature. Real boundary/end head executes; no tuple or target enters API.
        with torch.no_grad():
            for parameter in self.model.copy_boundary.parameters():parameter.zero_()
            self.model.copy_boundary[0].weight[0,2*self.model.config.width+256+7]=1.
            self.model.copy_boundary[2].weight[0,0]=1.
            self.model.copy_end_query.bias[0]=1.
            self.model.copy_end_key.weight.zero_();self.model.copy_end_key.weight[0,0]=1.
        result=method(**args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['validation_status'],'grounded_proposal_only',result['validation_error'])
        from cognitive_kernel.formation_reference_decision import caller_api
        _,opened,_=caller_api.caller_input_from_record(self.caller)
        original=opened[0][1]
        self.assertIn('Café "A" \\ 😀',original.decode())
        self.assertEqual(result['copy_predictions']['rows'][0]['end_byte'],len(original))
        self.assertEqual(result['bundle'].proposals[0].base.value_text,'Exact first original character: '+original.decode()+'.')
        self.assertEqual(result['copy_provenance'][0]['original_utf8'],[0,len(original)])

    def test_actual_generated_copy_and_reference_actions_refuse_uncited_source_without_repair(self):
        method=self.api();args=self.args();_,raw=self.record();ids=self.stream(raw)
        from cognitive_kernel import formation_original_copy_decision as C
        inventory=C.bind_original_copy_inventory(self.caller,self.joint,
            caller_sha256=self.caller['record_sha256'],joint_alignment_sha256=self.joint['record_sha256'],expected_binding=self.binding)
        with torch.no_grad():
            keys=self.model.copy_keys(base_states=self.states,source_mask=self.mask,inventory=inventory)
            # Input-only owned key difference makes source1 beat source0 through
            # the actual source head. Reference actions remain their real ties.
            delta=keys['source_keys'][1]-keys['source_keys'][0]
            self.assertGreater(delta.square().sum().item(),0)
            self.model.copy_source_query.bias.copy_(delta)
        result=method(**args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['failure_stage'],'compile_ground');self.assertIsNone(result['bundle'])
        self.assertEqual(result['copy_predictions']['rows'][0]['source_slot'],1)
        chosen={r['pointer']:r['candidate_ordinal'] for r in result['reference_predictions']['rows']}
        self.assertEqual(chosen['/proposals/0/proposal/evidence_refs/0'],0)
        self.assertEqual(chosen['/proposals/0/proposal/anchors/0/ref_id'],0)
        self.assertEqual(result['generated_token_ids'],ids);self.assertEqual(result['raw_output_text'],raw)
        self.assertTrue(result['eos_observed']);self.assertTrue(result['validation_error'])

    def test_capped_malformed_and_failed_decode_keep_first_failure_and_no_teacher_api(self):
        from unittest.mock import patch
        method=self.api();args=self.args();_,canonical=self.record()
        for raw,budget,stage in ((canonical,3,'generation'),('not-json',None,'parse'),
            ('{"schema":"a","schema":"b"}',None,'parse'),(canonical+'\n',None,'alignment')):
            self.calls=[];ids=self.stream(raw);result=method(**args,max_new_tokens=budget or len(ids)+1)
            self.assertEqual(result['failure_stage'],stage);self.assertEqual(result['generated_token_ids'],ids[:budget] if budget else ids)
            self.assertIsNone(result['bundle']);self.assertTrue(result['validation_error'])
        self.calls=[];ids=self.stream(canonical)
        with patch.object(self.processor.tokenizer,'decode',side_effect=ValueError('controlled decode refusal')) as decode:
            result=method(**args,max_new_tokens=len(ids)+1)
        decode.assert_called_once();self.assertEqual(result['failure_stage'],'decode')
        self.assertIsNone(result['raw_output_text']);self.assertEqual(result['generated_token_ids'],ids)
        for name in ('example','complete_pair','teacher','authored_copy_tuples','decision'):
            with self.assertRaises(TypeError):method(**args,max_new_tokens=1,**{name:{}})

    def test_real_owned_lexical_weights_emit_capped_prefix_without_a_generation_stub(self):
        method=self.api();args=self.args()
        token=self.processor.tokenizer('x',add_special_tokens=False)['input_ids'][0]
        with torch.no_grad():
            self.model.output_bias.fill_(-1000.);self.model.output_bias[token]=1000.
        result=method(**args,max_new_tokens=3)
        self.assertEqual(result['generated_token_ids'],[token]*3)
        self.assertEqual(result['raw_output_text'],'xxx');self.assertFalse(result['eos_observed'])
        self.assertEqual(result['failure_stage'],'generation');self.assertIsNone(result['bundle'])

    def test_preflight_foreign_binding_partial_source_and_nonfp32_refuse_before_generation(self):
        method=self.api();args=self.args();_,raw=self.record();ids=self.stream(raw)
        for changes in ({'source_mask':self.mask[:,:-1]},
            {'expected_binding':{**args['expected_binding'],'copy_inventory_sha256':'f'*64}},
            {'max_new_tokens':True},{'max_new_tokens':self.model.config.max_target_tokens+1}):
            result=method(**{**args,'max_new_tokens':len(ids)+1,**changes})
            self.assertEqual(result['failure_stage'],'preflight');self.assertEqual(result['generated_token_ids'],[])
            self.assertFalse(self.calls)
        self.model.bfloat16();result=method(**args,max_new_tokens=len(ids)+1)
        self.assertEqual(result['failure_stage'],'preflight');self.assertFalse(self.calls)


if __name__=='__main__':unittest.main()
