"""Public bounded measurements, with actual task-conditioned cross attention."""
from hashlib import sha256
from importlib import import_module, util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel.formation_sequence_specialist import FormationSequenceSpecialist


class SourceExample:
    split='train'
    case_id='v16-seed-01'
    raw='Exact é\r\n<literal>  '.encode('utf-8')
    opened_sources=(('public-source',raw),)
    context=SimpleNamespace(base=SimpleNamespace(evidence=(SimpleNamespace(
        ref_id='public-source',modality='text',content_digest=sha256(raw).hexdigest()),)))
    @property
    def target(self): raise AssertionError('source target must not access a teacher')
    def validate(self): raise AssertionError('target-reading admission is not a source helper')


class ChunkTokenizer:
    bos_token_id,eos_token_id=1,2
    def __init__(self): self.forward={};self.reverse={}
    def encode(self,text,*,add_special_tokens):
        if add_special_tokens: raise AssertionError('no automatic specials')
        answer=[]
        for i in range(0,len(text),3):
            chunk=text[i:i+3]
            if chunk not in self.forward:
                token=len(self.forward)+3;self.forward[chunk]=token;self.reverse[token]=chunk
            answer.append(self.forward[chunk])
        return answer
    def decode(self,ids,*,skip_special_tokens):
        if skip_special_tokens: raise AssertionError('roundtrip must preserve content')
        return ''.join(self.reverse[i] for i in ids)
    def __call__(self,text,*,add_special_tokens,return_offsets_mapping):
        assert return_offsets_mapping
        return {'input_ids':self.encode(text,add_special_tokens=add_special_tokens),
                'offset_mapping':[(i,min(i+3,len(text))) for i in range(0,len(text),3)]}


class SequenceFailureMetricsTests(unittest.TestCase):
    def module(self):
        name='scripts.mfm.formation_sequence_failure_metrics'
        self.assertIsNotNone(util.find_spec(name),'bounded failure metrics are missing')
        return import_module(name)

    def model_payload(self):
        torch.manual_seed(83)
        config=SpecialistConfig(6,19,8,2,2,16,0,1,2)
        model=FormationSequenceSpecialist(config).eval()
        with torch.no_grad(): model.task_embedding.weight[1].add_(.7)
        return model,{'base_states':torch.randn(1,5,6),'source_mask':torch.tensor([[1,1,1,0,0]]),
            'input_ids':torch.tensor([[1,4,7,8]]),'labels':torch.tensor([[4,7,8,2]])}

    def test_source_groups_use_real_offsets_and_source_only_exact_roundtrip(self):
        module=self.module();tokenizer=ChunkTokenizer();example=SourceExample()
        labels,groups=module.task_targets(tokenizer,example,256,'source_reconstruction')
        text,spans=module.ancestry.precise_scalar_spans(module.curriculum.lexical_source_record(example))
        self.assertEqual(tokenizer.decode(labels[:-1],skip_special_tokens=False),text)
        self.assertEqual((labels[-1],groups[-1]),(2,'eos'))
        expected=[]
        for left,right in tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']:
            overlap,kind=max((max(0,min(right,end)-max(left,start)),kind) for start,end,kind in spans)
            if kind=='value:sources.*.text': kind='source_text:sources.*.text'
            expected.append(kind if overlap else 'structure')
        self.assertEqual(groups[:-1],expected)
        self.assertIn('source_text:sources.*.text',groups)
        self.assertIn('source_reference:sources.*.ref_id',groups)

    def test_formation_delegates_original_precise_target_measurement(self):
        module=self.module();example=object();tokenizer=object()
        with patch.object(module.ancestry,'precise_target_groups',return_value=([3,2],['structure','eos'])) as original:
            self.assertEqual(module.task_targets(tokenizer,example,256,'formation'),([3,2],['structure','eos']))
        original.assert_called_once_with(tokenizer,example,256)

    def test_source_rejects_bad_offsets_incomplete_budget_and_unknown_task(self):
        module=self.module();tokenizer=ChunkTokenizer();example=SourceExample()
        for task,budget in [('source_reconstruction',1),('unregistered',256)]:
            with self.assertRaises(CognitiveKernelContractError): module.task_targets(tokenizer,example,budget,task)
        text,_=module.ancestry.precise_scalar_spans(module.curriculum.lexical_source_record(example))
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        encoded['offset_mapping'][0]=(0,999)
        with patch.object(ChunkTokenizer,'__call__',return_value=encoded):
            with self.assertRaises(CognitiveKernelContractError): module.task_targets(tokenizer,example,256,'source_reconstruction')

    def test_interventions_keep_targets_and_pair_foreign_states_with_mask(self):
        module=self.module();_,payload=self.model_payload()
        zero=module.source_intervention_payload(payload,mode='zero')
        self.assertTrue(torch.equal(zero['base_states'],torch.zeros_like(payload['base_states'])))
        self.assertIs(zero['source_mask'],payload['source_mask'])
        foreign={'base_states':torch.ones(1,3,6,dtype=torch.float64),'source_mask':torch.tensor([[1.,0.,1.]])}
        swapped=module.source_intervention_payload(payload,foreign,mode='foreign_case')
        self.assertEqual(swapped['base_states'].shape,(1,3,6))
        self.assertTrue(torch.equal(swapped['source_mask'],foreign['source_mask'].to(payload['source_mask'].dtype)))
        self.assertEqual(swapped['base_states'].dtype,payload['base_states'].dtype)
        for changed in (zero,swapped):
            self.assertIs(changed['input_ids'],payload['input_ids']);self.assertIs(changed['labels'],payload['labels'])
        self.assertTrue(payload['base_states'].abs().sum()>0)
        for mode,other in [('row_shuffle',None),('foreign_case',None),('foreign_case',{'base_states':torch.ones(1,3,7),'source_mask':torch.ones(1,3)})]:
            with self.assertRaises(CognitiveKernelContractError): module.source_intervention_payload(payload,other,mode=mode)

    def test_rank_measure_matches_actual_reference_logits_for_both_bound_tasks(self):
        module=self.module();model,payload=self.model_payload();position=2;candidate=5
        measured=[]
        for task in ('formation','source_reconstruction'):
            view=model.bind_task(task)
            with torch.inference_mode():
                logits=view.next_token_logits(base_states=payload['base_states'],source_mask=payload['source_mask'],
                                             input_ids=payload['input_ids'][:,:position+1])[0]
                probability=logits.softmax(-1);teacher=int(payload['labels'][0,position])
            row=module.rank_measure(view,payload,position,candidate=candidate);measured.append(row)
            self.assertAlmostEqual(row['teacher_probability'],probability[teacher].item(),places=7)
            self.assertEqual(row['teacher_rank'],1+int((logits>logits[teacher]).sum()))
            self.assertEqual(row['teacher_top1'],int(logits.argmax())==teacher)
            self.assertAlmostEqual(row['candidate_probability'],probability[candidate].item(),places=7)
            self.assertEqual(row['argmax_token_id'],int(logits.argmax()))
            self.assertEqual(row['dtype'],'float32')
        self.assertNotEqual(measured[0]['teacher_probability'],measured[1]['teacher_probability'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_rank_measure_rejects_non_fp32_invalid_position_or_ignored_teacher(self):
        module=self.module();model,payload=self.model_payload()
        for position,candidate in [(-1,None),(4,None),(2,19)]:
            with self.assertRaises(CognitiveKernelContractError): module.rank_measure(model.bind_task('formation'),payload,position,candidate=candidate)
        bad={**payload,'labels':payload['labels'].clone()};bad['labels'][0,2]=-100
        with self.assertRaises(CognitiveKernelContractError): module.rank_measure(model.bind_task('formation'),bad,2)
        model=model.double()
        with self.assertRaises(CognitiveKernelContractError): module.rank_measure(model.bind_task('formation'),payload,2)

    def test_intervention_rank_uses_paired_sources_and_fixed_teacher_prefix(self):
        module=self.module();model,payload=self.model_payload();view=model.bind_task('source_reconstruction')
        foreign={'base_states':torch.full((1,3,6),3.),'source_mask':torch.tensor([[1,0,1]])}
        zero=module.source_intervention_payload(payload,mode='zero')
        swapped=module.source_intervention_payload(payload,foreign,mode='foreign_case')
        measured=[]
        for changed in (payload,zero,swapped):
            result=module.rank_measure(view,changed,2);measured.append(result['teacher_probability'])
            with torch.inference_mode():
                reference=view.next_token_logits(base_states=changed['base_states'],source_mask=changed['source_mask'],
                    input_ids=payload['input_ids'][:,:3])[0].softmax(-1)[8].item()
            self.assertAlmostEqual(result['teacher_probability'],reference,places=7)
            self.assertEqual(result['teacher_token_id'],8)
        self.assertEqual(len(set(measured)),3)

    def test_nonfinite_projection_fails_before_reporting_a_rank(self):
        module=self.module();model,payload=self.model_payload()
        with torch.no_grad(): model.output_bias[3]=float('nan')
        with self.assertRaisesRegex(CognitiveKernelContractError,'finite FP32'):
            module.rank_measure(model.bind_task('formation'),payload,2)


if __name__=='__main__': unittest.main()
