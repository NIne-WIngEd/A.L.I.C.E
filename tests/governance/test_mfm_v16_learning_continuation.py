"""Continuation state/objective contracts; no synthetic capability claim."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from scripts.mfm import train_v16_decision_specialist as parent
from scripts.mfm import continue_v16_decision_learning as child


class ContinuationTests(unittest.TestCase):
    def test_privacy_admission_precedes_parent_or_child_reads(self):
        with patch.object(child.source,'load',side_effect=CognitiveKernelContractError('privacy rejected')):
            with self.assertRaisesRegex(CognitiveKernelContractError,'privacy rejected'):
                child.run(SimpleNamespace())

    def test_stop_requires_two_flat_equal_case_intervals_and_is_not_success(self):
        self.assertFalse(child.should_stop([4.,3.,2.]))
        self.assertFalse(child.should_stop([4.,3.99]))
        self.assertTrue(child.should_stop([4.,3.99,3.98]))
        self.assertTrue(child.should_stop([4.,4.1,4.2]))
        with self.assertRaises(CognitiveKernelContractError):
            child.should_stop([4.,float('nan'),3.])

    def test_terminal_stop_truthfully_distinguishes_plateau_from_completed_cap(self):
        from scripts.mfm import run_v16_budget_diagnostic as diagnostic
        run={'parent_completed_epochs':2,'parent_optimizer_steps':54,'maximum_total_epochs':8,
             'gradient_accumulation':16}
        result={'construction_schedule_completed':True,'terminal_reason':'additional_epoch_cap',
            'global_completed_epochs':8,'global_optimizer_steps':216,'additional_optimizer_steps':162,
            'all_maximum_train_epochs_completed':True}
        diagnostic.verify_terminal(run,result,432,[4.,2.,1.,.5])
        stopped={**result,'terminal_reason':'insufficient_train_progress','global_completed_epochs':6,
            'global_optimizer_steps':162,'additional_optimizer_steps':108,'all_maximum_train_epochs_completed':False}
        diagnostic.verify_terminal(run,stopped,432,[4.,3.99,3.98])
        for wrong in ({**stopped,'all_maximum_train_epochs_completed':True},
                      {**result,'global_optimizer_steps':215},
                      {**stopped,'terminal_reason':'additional_epoch_cap'}):
            with self.assertRaises(CognitiveKernelContractError): diagnostic.verify_terminal(run,wrong,432,[4.,3.99,3.98])
        with self.assertRaises(CognitiveKernelContractError): diagnostic.verify_terminal(run,stopped,432,[4.,2.,1.])

    def test_full_train_loss_matches_equal_case_gradient_objective(self):
        short=SimpleNamespace(case_id='short',split='train')
        long=SimpleNamespace(case_id='long',split='train')
        prepared=SimpleNamespace(train=(short,long))
        class Model:
            def eval(self): pass
            def __call__(self,**payload): return None,payload['test_loss']
        def payload(prepared,example,**kwargs):
            n=2 if example.case_id=='short' else 10
            return {'labels':torch.ones(1,n,dtype=torch.long),
                    'test_loss':torch.tensor(1. if n==2 else 3.)}
        prepared.device=torch.device('cpu')
        with patch.object(child.source,'payload',side_effect=payload):
            result=child.train_losses(Model(),prepared)
        self.assertEqual(result['equal_case_mean_loss'],2.)
        self.assertAlmostEqual(result['token_weighted_mean_loss'],32/12)
        self.assertEqual(result['train_cases'],2)
        prepared.train=(SimpleNamespace(case_id='bad',split='development'),)
        with self.assertRaises(CognitiveKernelContractError): child.train_losses(Model(),prepared)

    def make_parent(self,root):
        torch.manual_seed(124)
        config=SpecialistConfig(12,31,16,2,4,16,0,1,2)
        model=FormationDecisionSpecialist(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        example=SimpleNamespace(case_id='one',split='train')
        prepared=SimpleNamespace(config=config,device=torch.device('cpu'),train=(example,),
                                 view={'record_sha256':'b'*64})
        payload={'base_states':torch.randn(1,7,12),'source_mask':torch.ones(1,7,dtype=torch.long),
                 'input_ids':torch.tensor([[1,7,9]]),'labels':torch.tensor([[7,9,2]])}
        manifest={'schema':parent.RUN_SCHEMA,'objective':parent.OBJECTIVE,'initialization':parent.INITIALIZATION,
            'source_view_sha256':'b'*64,'code_binding':parent.code_binding(),'corpus_sha256':'c'*64,
            'owner_authorization_ref':'test-owner','specialist_config':config.record(),
            'specialist_dtype':'float32','device':'cpu','torch_version':torch.__version__,
            'seed':124,'learning_rate':1e-4,'gradient_accumulation':1,'epochs':2,
            'probe_only':False,'teacher_fit':True,'full_fit':False,
            'independent_gold':False,'final_payloads_opened':False,'qualified_for_product':False}
        run=parent.shared._write_new(root/'run.json',manifest)
        parent.training._seed_control(root,model,run['record_sha256'],config)
        with patch.object(parent.source,'payload',return_value=payload):
            for _ in range(2): parent.optimizer_step(model,optimizer,(example,),prepared=prepared)
        path=root/'formation-specialist.safetensors'
        save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(path))
        parent.training._checkpoint(root,specialist=model,optimizer=optimizer,
            run_digest=run['record_sha256'],epoch=1,next_case=1,step=2)
        component=parent.shared._write_new(root/'component.json',{
            'schema':parent.ARTIFACT_SCHEMA,'run_sha256':run['record_sha256'],
            'source_view_sha256':'b'*64,'specialist_config':config.record(),'probe_only':False,
            'qualified_for_product':False,'result':{'all_train_epochs_completed':True,
            'optimizer_steps':2,'trained_weights_sha256':parent.shared._digest(path)}})
        checkpoint=root/'checkpoint-00000002'
        receipt=parent.shared._read_sealed(checkpoint/'checkpoint.json',parent.training.CHECKPOINT_SCHEMA)
        args=SimpleNamespace(parent_decision_dir=root,parent_decision_sha256=component['record_sha256'],
            parent_checkpoint_sha256=receipt['record_sha256'],input_sha256='c'*64,
            owner_authorization_ref='test-owner')
        return model,optimizer,prepared,payload,args

    def test_parent_transfer_preserves_next_weights_moments_rng_and_global_cursor(self):
        with TemporaryDirectory() as d:
            root=Path(d); model,opt,p,payload,args=self.make_parent(root)
            parent_record=child.verify_parent(args,p)
            with patch.object(parent.source,'payload',return_value=payload):
                expected=parent.optimizer_step(model,opt,p.train,prepared=p)
                weights=deepcopy(model.state_dict());moments=deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
                epoch,cursor,step=child.restore_parent(parent_record,model,opt)
                observed=parent.optimizer_step(model,opt,p.train,prepared=p)
            self.assertEqual((epoch,cursor,step),(2,0,2))
            self.assertEqual(observed,expected)
            self.assertEqual(parent.cached.compare_tree(model.state_dict(),weights),0.)
            self.assertEqual(parent.cached.compare_tree(opt.state_dict(),moments),0.)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_external_checkpoint_pin_and_partial_parent_are_rejected(self):
        with TemporaryDirectory() as d:
            root=Path(d);_,_,p,_,args=self.make_parent(root)
            args.parent_checkpoint_sha256='e'*64
            with self.assertRaises(CognitiveKernelContractError): child.verify_parent(args,p)
            args.parent_checkpoint_sha256=parent.shared._read_sealed(root/'checkpoint-00000002/checkpoint.json',parent.training.CHECKPOINT_SCHEMA)['record_sha256']
            p.train=(*p.train,SimpleNamespace(case_id='extra',split='train'))
            with self.assertRaises(CognitiveKernelContractError): child.verify_parent(args,p)

    def test_interrupted_child_resumes_to_same_final_weights_as_uninterrupted(self):
        with TemporaryDirectory() as d:
            root=Path(d);parent_root=root/'parent';parent_root.mkdir()
            _,_,p,payload,args=self.make_parent(parent_root)
            p.view['stress_train_case_ids']=['one'];p.tokenizer=None
            p.development=(SimpleNamespace(case_id='dev',split='development'),)
            args.additional_epochs=2;args.save_every_steps=10;args.session_seconds=1000
            args.resume_checkpoint=None;args.output_dir=root/'uninterrupted'
            with patch.object(child.source,'load',return_value=p),\
                 patch.object(child.source,'payload',return_value=payload),\
                 patch.object(child.source.preparation,'target_ids',return_value=([1,7,9],[7,9,2])),\
                 patch.object(child.analysis,'target_groups',return_value=('',[7,9,2],['key','value','eos'])):
                complete=child.run(args)
                args.output_dir=root/'interrupted'
                clock=iter([0.,1001.,1001.])
                with patch.object(child,'perf_counter',side_effect=lambda:next(clock)):
                    paused=child.run(args)
                self.assertEqual(paused['state'],'CHECKPOINTED_INCOMPLETE')
                self.assertEqual(paused['global_optimizer_steps'],3)
                args.resume_checkpoint=Path(paused['resume_checkpoint'])
                resumed=child.run(args)
            self.assertEqual(resumed['trained_weights_sha256'],complete['trained_weights_sha256'])
            self.assertEqual(resumed['global_optimizer_steps'],4)
            self.assertEqual(resumed['terminal_reason'],'additional_epoch_cap')


if __name__=='__main__': unittest.main()
