"""Public diagnostic contracts; no private corpus or learned success claim."""
from hashlib import sha256
from importlib import import_module, util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes


class PublicExample:
    def __init__(self, case_id, split='train'):
        self.case_id, self.split = case_id, split
        raw = 'public é\r\n<literal>  '.encode('utf-8')
        self.opened_sources = (('public-source', raw),)
        ref = SimpleNamespace(ref_id='public-source', modality='text',
                              content_digest=sha256(raw).hexdigest())
        self.context = SimpleNamespace(base=SimpleNamespace(evidence=(ref,)),
                                       content_digest=lambda: 'c'*64)

    @property
    def target(self):
        raise AssertionError('free generation cannot access a teacher target')


class PublicTokenizer:
    bos_token_id, eos_token_id=1,2

    def encode(self,text,*,add_special_tokens):
        if add_special_tokens: raise AssertionError('source targets cannot add specials')
        return [ord(c) for c in text]

    def decode(self, ids, *, skip_special_tokens):
        if skip_special_tokens:
            raise AssertionError('raw decoder must preserve literal content')
        return ''.join(chr(i) for i in ids)


def public_prepared():
    base = [f'v16-seed-{i:02}' for i in (*range(1, 13), 19, 20, 21)]
    base.append('role-source-person-revocation')
    operations = ('norm-agreed', 'norm-unsettled', 'episode-mission', 'plan-outcome',
                  'procedure-revision', 'source-person', 'revocation-deletion',
                  'record-correction', 'owner-deletion', 'sensitive-boundary',
                  'assistant-self', 'hypothetical-code', 'conflicting-outside')
    ids = [*base, *(f'long-{craft}-{site}-{operation}'
                   for craft in ('atlas', 'ceramics', 'orchard', 'theatre', 'harbor', 'library', 'garden', 'radio')
                   for site in ('east', 'west', 'north', 'south') for operation in operations)]
    train = tuple(PublicExample(i) for i in ids)
    dev = tuple(PublicExample(f'public-development-{i}', 'development') for i in range(9))
    rows = [{'case_id':e.case_id, 'split':e.split, 'source_tokens':20} for e in (*train, *dev)]
    rows[-10]['source_tokens'] = 500
    lengths = {e.case_id:30 for e in train}
    lengths[ids[-2]] = 100
    receipt = {'cases':[{'case_id':i, 'lexical_target_tokens':lengths[i]} for i in ids],
               'maximum_lexical_target_tokens':100}
    config = SimpleNamespace(base_hidden_size=4, max_target_tokens=2048,
                             record=lambda: {'public':'config'})
    prepared = SimpleNamespace(train=train, development=dev, config=config,
        device='cpu', tokenizer=PublicTokenizer(), locators={e.case_id:(Path('.'), {}) for e in (*train,*dev)},
        view={'record_sha256':'v'*64, 'longest_target_tokens':1200, 'cases':rows,
              'stress_train_case_ids':[ids[0],ids[1]]})
    return prepared, receipt


class CandidateFixture:
    """Small source bytes, full public roster and real sealed receipt readers."""
    def __init__(self,module,root,*,run_change=None,result_change=None):
        import torch
        self.module=module
        self.prepared,_=public_prepared()
        self.prepared.view['record_sha256']='a'*64
        self.receipt=module.curriculum.curriculum_receipt(self.prepared.train,self.prepared.tokenizer,2048)
        fit,probe=root/'fit',root/'probe'
        fit.mkdir();probe.mkdir()
        self.args=SimpleNamespace(sequence_dir=fit,sequence_probe_dir=probe,input_sha256='b'*64,
                                  owner_authorization_ref='public-synthetic-owner')
        self.old_run={'seed':19,'learning_rate':.001,'record_sha256':'c'*64}
        self.old_component={'record_sha256':'d'*64,'result':{'trained_weights_sha256':'e'*64}}
        self.old_diag={'record_sha256':'f'*64}
        manifest={'schema':module.trainer.RUN_SCHEMA,'objective':module.curriculum.OBJECTIVE,
            'task_names':['formation','source_reconstruction'],
            'formation_objective':module.ancestry.child.parent.OBJECTIVE,'task_mixture':[.5,.5],
            'initialization':module.INITIALIZATION,'optimizer_initialization':module.trainer.OPTIMIZER_INITIALIZATION,
            'source_view_sha256':'a'*64,'code_binding':module.trainer.code_binding(),
            'curriculum_sha256':self.receipt['record_sha256'],'corpus_sha256':'b'*64,
            'owner_authorization_ref':'public-synthetic-owner','specialist_config':{'public':'config'},
            'specialist_dtype':'float32','device':'cpu','torch_version':torch.__version__,
            'seed':19,'learning_rate':.001,'gradient_accumulation':16,'epochs':4,
            'save_every_steps':10,'session_seconds':72000,
            'parent_component_sha256':'d'*64,'parent_run_sha256':'c'*64,
            'parent_weights_sha256':'e'*64,'parent_diagnostic_sha256':'f'*64,
            'parent_optimizer_loaded':False,
            'seed_control_meaning':'same-seed new-task fresh decoder; historical parent has different objective',
            'stress_train_case_ids':module.trainer.stress_ids(self.prepared,self.receipt),'probe_only':False,
            'teacher_fit':True,'full_fit':False,'independent_gold':False,
            'final_payloads_opened':False,'qualified_for_product':False}
        if run_change: manifest.update(run_change)
        probe_run=module.shared._write_new(probe/'run.json',
            {**manifest,'probe_only':True,'stress_probe_sha256':None})
        self.write_curriculum(probe)
        probe_seed=self.write_seed(probe,probe_run)
        checkpoint=self.write_checkpoint(probe,probe_run,2,0,16,b'public-probe-weights')
        wanted=manifest['stress_train_case_ids']
        selected=sorted(self.prepared.train,key=lambda e:wanted.index(e.case_id) if e.case_id in wanted else len(wanted))[:16]
        ids=[e.case_id for e in selected]
        checks={task:{'argmaxes_equal':True,'compared_greedy_prefixes':32,'max_abs_difference':0.,
                      'atol':1e-4,'rtol':1e-4} for task in ('formation','source_reconstruction')}
        probe_component=module.shared._write_new(probe/'component.json',{
            'schema':module.trainer.ARTIFACT_SCHEMA,'run_sha256':probe_run['record_sha256'],
            'source_view_sha256':'a'*64,'curriculum_sha256':self.receipt['record_sha256'],
            'specialist_config':{'public':'config'},'probe_only':True,'teacher_fit':True,
            'full_fit':False,'qualified_for_product':False,
            'result':{'backward_and_restart_passed':True,'optimizer_steps':3,
                'migration_parity':{'bit_exact':True,'case_ids':ids,'max_abs_difference':0.},
                'stress_case_ids':wanted,'accumulation_case_ids':ids,
                'checkpoint_sha256':checkpoint['record_sha256'],'task_bound_actual_cache_checks':checks,
                'model_max_abs_difference':0.,'optimizer_max_abs_difference':0.}})
        self.run=module.shared._write_new(fit/'run.json',
            {**manifest,'stress_probe_sha256':probe_component['record_sha256']})
        self.write_curriculum(fit)
        self.write_seed(fit,self.run)
        weights=b'public-trained-weights'
        (fit/'formation-specialist.safetensors').write_bytes(weights)
        weights_sha=sha256(weights).hexdigest()
        self.write_checkpoint(fit,self.run,108,3,432,weights)
        losses={e.case_id:[2.,4.] for e in self.prepared.train}
        groups=self.receipt['group_counts']
        for epoch in range(5):
            module.shared._write_new(fit/f'progress-{epoch*27:08d}.json',{
                'schema':module.trainer.PROGRESS_SCHEMA,'run_sha256':self.run['record_sha256'],
                'epoch':epoch,'optimizer_steps':epoch*27,'train_cases':432,
                'balanced_two_task_objective':3.,'complete_case_losses':losses,
                'strata':{k:{'cases':n,'formation_mean_loss':2.,'source_mean_loss':4.} for k,n in groups.items()},
                'development_read':False,'qualified_for_product':False})
        result={'optimizer_steps':108,'all_train_epochs_completed':True,'trained_weights_sha256':weights_sha,
            'diagnostic_development_teacher_forced_loss':3.,'seeded_development_teacher_forced_loss':5.,
            'development_has_no_gradients':True,'generated_ability_established':False,
            'independent_qualification':False}
        if result_change: result.update(result_change)
        self.component=module.shared._write_new(fit/'component.json',{
            'schema':module.trainer.ARTIFACT_SCHEMA,'run_sha256':self.run['record_sha256'],
            'source_view_sha256':'a'*64,'curriculum_sha256':self.receipt['record_sha256'],
            'specialist_config':{'public':'config'},'probe_only':False,'teacher_fit':True,
            'full_fit':False,'qualified_for_product':False,'result':result})
        self.args.sequence_sha256=self.component['record_sha256']

    def write_curriculum(self,root):
        return self.module.shared._write_new(root/'curriculum.json',
            {k:v for k,v in self.receipt.items() if k!='record_sha256'})

    def write_seed(self,root,run):
        (root/'seed-control.safetensors').write_bytes(b'public-same-seed-weights')
        return self.module.shared._write_new(root/'seed-control.json',{
            'schema':self.module.training.SEED_SCHEMA,
            'specialist_sha256':sha256(b'public-same-seed-weights').hexdigest(),
            'run_manifest_sha256':run['record_sha256'],'specialist_config':{'public':'config'},
            'optimizer_steps':0,'meaning':'public fixture negative control'})

    def write_checkpoint(self,root,run,step,epoch,next_case,weights):
        path=root/f'checkpoint-{step:08d}'
        path.mkdir()
        (path/'specialist.safetensors').write_bytes(weights)
        (path/'optimizer.pt').write_bytes(b'public-optimizer-custody')
        return self.module.shared._write_new(path/'checkpoint.json',{
            'schema':self.module.training.CHECKPOINT_SCHEMA,'run_manifest_sha256':run['record_sha256'],
            'specialist_sha256':sha256(weights).hexdigest(),
            'optimizer_sha256':sha256(b'public-optimizer-custody').hexdigest(),
            'epoch':epoch,'next_case':next_case,'step':step})

    def parent(self):
        return patch.object(self.module.ancestry,'verify_child',
            return_value=(self.old_run,self.old_component,self.old_diag,[],Path('public-parent'),{}))

    def rewrite(self,path,change):
        row=json.loads(path.read_text(encoding='utf-8'))
        row.pop('record_sha256')
        change(row)
        sealed=self.module.shared._seal(row)
        path.write_bytes(canonical_json_bytes(sealed))
        return sealed


class SequenceDiagnosticTests(unittest.TestCase):
    def module(self):
        name = 'scripts.mfm.run_v16_sequence_diagnostic'
        self.assertIsNotNone(util.find_spec(name), 'sequence diagnostic is missing')
        return import_module(name)

    def test_rosters_keep_all_diverse_one_complete_host_and_nine_development(self):
        module = self.module()
        prepared, receipt = public_prepared()
        rosters = module.select_rosters(prepared, receipt)
        formation = rosters['formation']
        self.assertEqual(len(formation), 38)
        self.assertEqual(sum(e.split=='development' for e in formation), 9)
        self.assertEqual(sum(e.case_id.startswith('long-atlas-east-') for e in formation), 13)
        self.assertEqual(sum(e.split=='train' and not e.case_id.startswith('long-') for e in formation), 16)
        stress = [e.case_id for e in rosters['source_reconstruction']]
        self.assertEqual(stress, ['v16-seed-01','v16-seed-02',
            'long-radio-south-hypothetical-code','long-radio-south-conflicting-outside'])
        self.assertTrue(all(e.split=='train' for e in rosters['source_reconstruction']))

    def test_roster_rejects_duplicate_or_other_train_construction(self):
        module = self.module()
        for replacement in ('v16-seed-01','long-atlas-east-unseen-action'):
            prepared, receipt = public_prepared()
            prepared.train = (*prepared.train[:-1], PublicExample(replacement))
            with self.assertRaises(CognitiveKernelContractError):
                module.select_rosters(prepared, receipt)

    def test_complete_budgets_include_eos_and_never_truncate(self):
        module = self.module()
        prepared, receipt = public_prepared()
        self.assertEqual(module.decode_budgets(prepared, receipt),
                         {'formation':2048,'source_reconstruction':100})
        for target, source in ((2049,100),(1200,2049),(1200,0)):
            prepared.view['longest_target_tokens']=target
            receipt['maximum_lexical_target_tokens']=source
            with self.assertRaises(CognitiveKernelContractError):
                module.decode_budgets(prepared, receipt)

    def test_source_requires_exact_canonical_json_and_eos(self):
        module = self.module()
        example = PublicExample('v16-seed-01')
        raw = canonical_json_bytes({'sources':[{'ref_id':'public-source',
            'text':'public é\r\n<literal>  '}]}).decode('utf-8')
        with patch.object(module.original, 'validate', side_effect=AssertionError('memory validator must not run')):
            self.assertEqual(module.validate_source(raw,True,example)['validation_status'],
                             'exact_source_reconstruction')
            self.assertEqual(module.validate_source(raw,False,example)['validation_status'],'invalid')
            equivalent = json.dumps(json.loads(raw),ensure_ascii=False,indent=2)
            self.assertEqual(module.validate_source(equivalent,True,example)['validation_status'],'invalid')
            changed = raw.replace('public é','public e')
            self.assertEqual(module.validate_source(changed,True,example)['validation_status'],'invalid')

    def test_generation_routes_the_explicit_task_view_before_comparison(self):
        module = self.module()
        prepared, _ = public_prepared()
        example = prepared.train[0]
        events=[]
        bound=object()
        model=SimpleNamespace(bind_task=lambda task: events.append(('bind',task)) or bound)
        def greedy(view,states,mask,budget):
            self.assertIs(view,bound)
            self.assertEqual((states,mask,budget),('states','mask',100))
            events.append(('generate',None))
            return [123,125],True,{'argmaxes_equal':True}
        def comparison(raw,ended,e):
            self.assertEqual(events[-1][0],'generate')
            events.append(('compare',None))
            return {'raw_output_text':raw,'eos_observed':ended,'validation_status':'invalid'}
        with patch.object(module.source.features,'read_case',return_value=({}, {'states':'states','attention_mask':'mask'})), \
             patch.object(module.source.features,'case_key',return_value='i'*64), \
             patch.object(module.source,'payload',side_effect=AssertionError('teacher payload cannot enter generation')), \
             patch.object(module.original,'greedy_fp32',side_effect=greedy), \
             patch.object(module,'validate_source',side_effect=comparison):
            row=module.generate_case(model,prepared,example,task='source_reconstruction',
                control='trained',weights_sha256='a'*64,run_id='public-run',maximum=100)
        self.assertEqual(events,[('bind','source_reconstruction'),('generate',None),('compare',None)])
        self.assertEqual(row['generated_token_ids'],[123,125])
        self.assertEqual(row['raw_output_text'],'{}')

    def test_unregistered_task_fails_before_reading_source_features(self):
        module=self.module()
        prepared,_=public_prepared()
        with patch.object(module.source.features,'read_case',side_effect=AssertionError('must reject before read')):
            with self.assertRaises(CognitiveKernelContractError):
                module.generate_case(object(),prepared,prepared.train[0],task='answer',
                    control='trained',weights_sha256='a'*64,run_id='public-run',maximum=100)

    def test_formation_retains_original_strict_validation_after_bound_generation(self):
        module=self.module()
        prepared,_=public_prepared()
        model=SimpleNamespace(bind_task=lambda task: ('bound',task))
        events=[]
        def greedy(view,states,mask,budget):
            self.assertEqual(view,('bound','formation'))
            self.assertEqual(budget,2048)
            events.append('generate')
            return [123,125],True,{'argmaxes_equal':True}
        def strict(raw,ended,example,weights,run_id):
            self.assertEqual(events,['generate'])
            self.assertEqual((raw,ended,weights,run_id),('{}',True,'a'*64,'public-run'))
            events.append('strict')
            return {'raw_output_text':raw,'eos_observed':ended,'validation_status':'invalid',
                    'proposal_count':0,'exact_teacher_decision_agreement':None}
        with patch.object(module.source.features,'read_case',return_value=({}, {'states':'states','attention_mask':'mask'})), \
             patch.object(module.source.features,'case_key',return_value='i'*64), \
             patch.object(module.original,'greedy_fp32',side_effect=greedy), \
             patch.object(module.original,'validate',side_effect=strict), \
             patch.object(module,'validate_source',side_effect=AssertionError('formation must retain its own validator')):
            row=module.generate_case(model,prepared,prepared.train[0],task='formation',
                control='trained',weights_sha256='a'*64,run_id='public-run',maximum=2048)
        self.assertEqual(events,['generate','strict'])
        self.assertEqual(row['generated_token_ids'],[123,125])

    def test_terminal_requires_all_432_cases_for_every_declared_epoch(self):
        module=self.module()
        run={'epochs':4,'gradient_accumulation':16,'save_every_steps':10,'session_seconds':72000}
        result={'optimizer_steps':108,'all_train_epochs_completed':True,
                'development_has_no_gradients':True,'generated_ability_established':False,
                'independent_qualification':False}
        self.assertEqual(module.verify_terminal(run,result,432),108)
        for field,value in (('optimizer_steps',107),('all_train_epochs_completed',False),
                            ('development_has_no_gradients',False),('generated_ability_established',True)):
            broken={**result,field:value}
            with self.assertRaises(CognitiveKernelContractError):
                module.verify_terminal(run,broken,432)
        with self.assertRaises(CognitiveKernelContractError):
            module.verify_terminal(run,result,431)
        with self.assertRaises(CognitiveKernelContractError):
            module.verify_terminal({**run,'epochs':5},result,432)

    def test_parent_chain_is_verified_before_candidate_files_are_read(self):
        module=self.module()
        prepared,_=public_prepared()
        with patch.object(module.ancestry,'verify_child',side_effect=CognitiveKernelContractError('original external parent pin differs')) as parent, \
             patch.object(module.shared,'_read_sealed',side_effect=AssertionError('candidate read before parent admission')):
            with self.assertRaisesRegex(CognitiveKernelContractError,'original external parent pin'):
                module.verify_candidate(SimpleNamespace(),prepared)
        parent.assert_called_once()

    def test_owner_admission_precedes_all_new_candidate_reads(self):
        module=self.module()
        with patch.object(module.source,'load',side_effect=CognitiveKernelContractError('owner admission rejected')) as admission, \
             patch.object(module,'verify_candidate',side_effect=AssertionError('new candidate read before owner admission')):
            with self.assertRaisesRegex(CognitiveKernelContractError,'owner admission'):
                module.run(SimpleNamespace())
        admission.assert_called_once()

    def test_family_task_metrics_remain_separate(self):
        module=self.module()
        rows=[{'task':'formation','family':'base-train','eos_observed':True,
               'validation_status':'grounded_teacher_proposal_only','proposal_count':2,
               'exact_teacher_decision_agreement':True},
              {'task':'source_reconstruction','family':'base-train','eos_observed':True,
               'validation_status':'exact_source_reconstruction','exact_source_json_agreement':True}]
        summary={}
        for row in rows: module.add_measurement(summary,row)
        self.assertEqual(set(summary),{'formation','source_reconstruction'})
        self.assertEqual(summary['formation']['base-train']['grounded_outputs'],1)
        self.assertEqual(summary['source_reconstruction']['base-train']['exact_source_reconstructions'],1)
        self.assertNotIn('all-tasks',summary)

    def test_completed_candidate_checks_real_receipts_and_capacity_parent_chain(self):
        module=self.module()
        with TemporaryDirectory() as directory:
            fixture=CandidateFixture(module,Path(directory))
            with fixture.parent() as parent:
                candidate=module.verify_candidate(fixture.args,fixture.prepared)
            parent.assert_called_once_with(fixture.args,fixture.prepared)
            self.assertEqual(candidate.checkpoint['step'],108)
            self.assertEqual(len(candidate.progress),5)
            self.assertEqual(candidate.probe['result']['optimizer_steps'],3)

    def test_resigned_different_parent_or_old_objective_still_rejected(self):
        module=self.module()
        for change in ({'parent_component_sha256':'0'*64},{'objective':'historical-one-task-objective'}):
            with TemporaryDirectory() as directory:
                fixture=CandidateFixture(module,Path(directory),run_change=change)
                with fixture.parent(),self.assertRaisesRegex(CognitiveKernelContractError,'lineage'):
                    module.verify_candidate(fixture.args,fixture.prepared)

    def test_external_sequence_pin_and_old_receipt_schema_are_not_optional(self):
        module=self.module()
        for mode in ('wrong-pin','old-schema'):
            with TemporaryDirectory() as directory:
                fixture=CandidateFixture(module,Path(directory))
                if mode=='wrong-pin': fixture.args.sequence_sha256='0'*64
                else:
                    row=fixture.rewrite(fixture.args.sequence_dir/'component.json',
                        lambda r:r.update(schema='mfm-historical-component-v1'))
                    fixture.args.sequence_sha256=row['record_sha256']
                with fixture.parent(),self.assertRaises(CognitiveKernelContractError):
                    module.verify_candidate(fixture.args,fixture.prepared)

    def test_partial_training_and_changed_lexical_receipt_are_rejected(self):
        module=self.module()
        for mode in ('partial','lexical'):
            with TemporaryDirectory() as directory:
                fixture=CandidateFixture(module,Path(directory),
                    result_change={'optimizer_steps':107} if mode=='partial' else None)
                if mode=='lexical':
                    fixture.rewrite(fixture.args.sequence_dir/'curriculum.json',
                        lambda r:r['cases'][0].update(lexical_target_token_ids_sha256='0'*64))
                with fixture.parent(),self.assertRaises(CognitiveKernelContractError):
                    module.verify_candidate(fixture.args,fixture.prepared)

    def test_missing_case_in_complete_progress_is_rejected_even_when_resealed(self):
        module=self.module()
        with TemporaryDirectory() as directory:
            fixture=CandidateFixture(module,Path(directory))
            fixture.rewrite(fixture.args.sequence_dir/'progress-00000108.json',
                lambda r:r['complete_case_losses'].pop('v16-seed-01'))
            with fixture.parent(),self.assertRaises(CognitiveKernelContractError):
                module.verify_candidate(fixture.args,fixture.prepared)

    def test_completed_checkpoint_cannot_stop_before_last_case(self):
        module=self.module()
        with TemporaryDirectory() as directory:
            fixture=CandidateFixture(module,Path(directory))
            fixture.rewrite(fixture.args.sequence_dir/'checkpoint-00000108'/'checkpoint.json',
                lambda r:r.update(next_case=416))
            with fixture.parent(),self.assertRaisesRegex(CognitiveKernelContractError,'full-train boundary'):
                module.verify_candidate(fixture.args,fixture.prepared)

    def test_actual_same_seed_state_is_checked_tensor_for_tensor(self):
        import torch
        module=self.module()
        torch.manual_seed(37)
        model=torch.nn.Linear(3,2)
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        self.assertTrue(module.verify_seed_parameters(model,state)['actual_same_seed_parameters_equal'])
        state['bias'][0]+=1
        with self.assertRaises(CognitiveKernelContractError): module.verify_seed_parameters(model,state)


if __name__=='__main__': unittest.main()
