"""Completed observation contracts; small local tensors prove no learned capability."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel.formation_conditional_task import bind_conditional_task
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from scripts.mfm import train_public_conditional_foundation as producer
from scripts.mfm import train_v1_formation_specialist as shared
from scripts.mfm.prepare_public_conditional_material import semantic_mr
import test_mfm_public_conditional_training as fixtures


class Tokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs == dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return ''.join(f'<{i}>' for i in ids)


class CompletedObservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = fixtures.PublicConditionalTrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.prepared = self.fixture.load()
        self.model = FormationDecisionSpecialist(SpecialistConfig(3, 16, 8, 1, 2, 128, 0, 2, 1, 2))
        self.view = bind_conditional_task(self.model, task=producer.TASK)

    def minimal_plan(self):
        source = next(s for s in self.prepared.sources if s['split'] == 'development')
        return dict(source_order=[source['source_key']], sources={source['source_key']: source},
            source_groups={source['source_key']: source['semantic_input_sha256s']},
            groups={source['semantic_input_sha256s'][0]: {'observed_slots': {'name': 'Synthetic'},
                'original_mrs': [{'source_text': 'name[Synthetic]'}]}},
            assignments={source['semantic_input_sha256s'][0]: 'calibration'},
            decode=dict(generation_tokens=128, bos_token_id=2, eos_token_id=1),
            bindings={'proposal_file_sha256': 'a'*64}, exposure={'known_exposures': []},
            counts={'pairs': 1, 'semantic_groups': 1, 'distinct_processed_sources': 1})

    def test_parser_has_separate_purpose_without_adam_recipe(self):
        self.assertTrue(hasattr(producer, 'observation_arguments'), 'missing completed observation parser')
        argv = []
        for name in ('census-dir','feature-bank','output-dir','prepared-base-dir','prepared-base-receipt',
                     'component-dir','completed-run-dir','seed-dir','plan-dir','exposure-path'):
            argv += ['--'+name, str(self.root/name)]
        for name in ('census-sha256','feature-bank-sha256','config-sha256','prepared-clone-sha256',
                     'component-file-sha256','run-file-sha256','curriculum-file-sha256','history-file-sha256',
                     'checkpoint-file-sha256','seed-file-sha256','final-weight-sha256','seed-weight-sha256',
                     'proposal-sha256','approval-sha256','exposure-sha256'):
            argv += ['--'+name, 'a'*64]
        argv += ['--device','cpu','--provider','fixture','--allocation-ref','local-tiny']
        args = producer.observation_arguments(argv)
        self.assertFalse(hasattr(args, 'learning_rate'))
        self.assertFalse(hasattr(args, 'epochs'))
        self.assertEqual(args.component_dir, self.root/'component-dir')

    def test_real_source_only_generation_retains_eos_empty_and_cap(self):
        self.assertTrue(hasattr(producer, 'observe_public_development'), 'missing full-roster observation consumer')
        self.model.train()
        self.model.decoder.layers[0].eval()
        modes = [m.training for m in self.model.modules()]
        rng = torch.random.get_rng_state().clone()
        # Force a real learned-head EOS argmax, without replacing generation.
        with torch.no_grad():
            self.model.output_bias.fill_(-1e6)
            self.model.output_bias[1] = 1e6
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        with patch.object(producer.reader, 'payload', side_effect=AssertionError('teacher path opened')), \
             patch.object(self.view, 'forward', side_effect=AssertionError('teacher forward opened')):
            report = producer.observe_public_development(self.view, self.prepared, Tokenizer(),
                self.minimal_plan(), output_dir=self.root, condition='opaque-a', weight_sha256='b'*64)
        row = json.loads((self.root/report['raw_file']).read_text())
        self.assertEqual(row['raw_token_ids'], [])
        self.assertTrue(row['eos_observed'])
        self.assertEqual(row['decoder_steps'], 1)
        self.assertEqual(row['raw_output_text'], '')
        self.assertEqual(row['case_ids'], ['d'])
        self.assertGreater(row['parity']['compared_greedy_prefixes'], 0)
        original_raw = (self.root/report['raw_file']).read_bytes()
        with self.assertRaisesRegex(ValueError,'overwritten'):
            producer.observe_public_development(self.view,self.prepared,Tokenizer(),self.minimal_plan(),
                output_dir=self.root,condition='opaque-a',weight_sha256='b'*64)
        self.assertEqual((self.root/report['raw_file']).read_bytes(),original_raw)
        self.assertEqual([m.training for m in self.model.modules()], modes)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for key,value in before.items():
            torch.testing.assert_close(self.model.state_dict()[key], value, atol=0, rtol=0)
        # A non-EOS special token is retained, and exhausting 128 steps is explicit.
        with torch.no_grad():
            self.model.output_bias[1] = -1e6
            self.model.output_bias[2] = 1e6
        report = producer.observe_public_development(self.view, self.prepared, Tokenizer(),
            self.minimal_plan(), output_dir=self.root, condition='opaque-b', weight_sha256='c'*64)
        row = json.loads((self.root/report['raw_file']).read_text())
        self.assertEqual(row['raw_token_ids'], [2]*128)
        self.assertFalse(row['eos_observed'])
        self.assertTrue(row['cap_exhausted'])
        self.assertEqual(row['decoder_steps'], 128)
        self.assertEqual(row['parity']['compared_greedy_prefixes'], 128)
        self.assertTrue(row['parity']['longest_source_full_actual_prefix_check'])
        self.assertEqual(report['longest_source_reference_check']['compared_greedy_prefixes'],128)

    def test_exception_retains_partial_file_and_restores_mode_rng(self):
        self.assertTrue(hasattr(producer, 'observe_public_development'), 'missing completed observation operation')
        self.model.train()
        modes = [m.training for m in self.model.modules()]
        rng = torch.random.get_rng_state().clone()
        def corrupt(**kwargs):
            torch.rand(2)
            return torch.zeros(1, 16)
        with patch.object(self.view, 'next_token_logits', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'parity|reference'):
                producer.observe_public_development(self.view, self.prepared, Tokenizer(),
                    self.minimal_plan(), output_dir=self.root, condition='opaque-c', weight_sha256='d'*64)
        self.assertEqual([m.training for m in self.model.modules()], modes)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        failures = list(self.root.glob('*failure*.json'))
        self.assertEqual(len(failures), 1)
        error = json.loads(failures[0].read_text())
        self.assertEqual(error['completed_sources'], 0)
        self.assertIn('parity', error['error'])
        self.assertTrue(list(self.root.glob('raw-*.jsonl')))

    def test_source_fanout_decodes_each_exact_input_once_without_group_collapse(self):
        self.assertTrue(hasattr(producer, 'observe_public_development'), 'missing completed observation operation')
        # Synthetic sources are separately constructed; no actual model weights or targets are read.
        plan = self.minimal_plan()
        original = plan['sources'][plan['source_order'][0]]
        second = dict(original, source_key='e'*64, case_ids=['fixture-second'])
        original = dict(original, case_ids=['d', 'fixture-reference'])
        plan['sources'] = {original['source_key']: original, second['source_key']: second}
        plan['source_order'] = list(plan['sources'])
        plan['source_groups'][second['source_key']] = original['semantic_input_sha256s']
        receipt = {'record_sha256': self.prepared.source_receipt_by_key[original['source_key']]}
        self.prepared.source_receipt_by_key[second['source_key']] = receipt['record_sha256']
        states = {'states':torch.ones(1, 2, 3), 'attention_mask':torch.ones(1, 2, dtype=torch.long)}
        with torch.no_grad():
            self.model.output_bias.fill_(-1e6); self.model.output_bias[1] = 1e6
        with patch.object(producer.reader.features, 'read_source', return_value=(receipt,states)) as read:
            report = producer.observe_public_development(self.view, self.prepared, Tokenizer(), plan,
                output_dir=self.root, condition='opaque-d', weight_sha256='e'*64)
        rows = [json.loads(line) for line in (self.root/report['raw_file']).read_text().splitlines()]
        self.assertEqual(read.call_count, 2)
        self.assertEqual([r['case_ids'] for r in rows], [['d','fixture-reference'],['fixture-second']])
        self.assertEqual(report['pair_associations'], 3)

    def test_nonfinite_reference_failure_is_preserved_without_json_nan_or_inf(self):
        with patch.object(self.view, 'next_token_logits', return_value=torch.full((1,16),float('inf'))):
            with self.assertRaisesRegex(ValueError,'parity'):
                producer.observe_public_development(self.view,self.prepared,Tokenizer(),self.minimal_plan(),
                    output_dir=self.root,condition='opaque-nonfinite',weight_sha256='f'*64)
        failure=json.loads((self.root/'observation-failure-opaque-nonfinite.json').read_text())
        self.assertFalse(failure['parity']['reference_finite'])
        self.assertIsNone(failure['parity']['max_abs_difference'])

    def test_longest_source_detects_reference_mismatch_after_first32(self):
        with torch.no_grad():
            self.model.output_bias.fill_(-1e6)
            self.model.output_bias[2]=1e6
        actual=self.view.next_token_logits
        def late_mismatch(**data):
            logits=actual(**data)
            if data['input_ids'].shape[1]==41:
                logits=logits.clone()
                logits[0,3]=2e6
            return logits
        with patch.object(self.view,'next_token_logits',side_effect=late_mismatch):
            with self.assertRaisesRegex(ValueError,'parity'):
                producer.observe_public_development(self.view,self.prepared,Tokenizer(),self.minimal_plan(),
                    output_dir=self.root,condition='opaque-late',weight_sha256='f'*64)
        failure=json.loads((self.root/'observation-failure-opaque-late.json').read_text())
        self.assertEqual(failure['parity']['compared_greedy_prefixes'],41)
        self.assertEqual(len(failure['parity']['failed_prefix_ids']),41)

    def test_decode_rejects_unbound_task_and_bf16(self):
        self.assertTrue(hasattr(producer, 'observe_public_development'), 'missing completed observation operation')
        kwargs = dict(output_dir=self.root, condition='opaque-e', weight_sha256='e'*64)
        with self.assertRaisesRegex(ValueError,'task'):
            producer.observe_public_development(self.model, self.prepared, Tokenizer(), self.minimal_plan(), **kwargs)
        self.model.to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'FP32'):
            producer.observe_public_development(self.view, self.prepared, Tokenizer(), self.minimal_plan(), **kwargs)

    def frozen_files(self):
        """Independent locally authored custody; no production files used by tests."""
        for n in range(1000):
            text = f'name[Synthetic {n}], food[Italian]'
            group = canonical_sha256(semantic_mr(text))
            component = sha256(group.encode('ascii')).hexdigest()
            partition = 'calibration' if int(sha256(('mfm-e2e-semantic-rubric-partition-v1\n'+component).encode('ascii')).hexdigest(),16)%5==0 else 'audit'
            if int(group,16)%10==0 and partition=='audit':
                break
        self.fixture.pairs[-1]['semantic_input_sha256'] = group
        self.fixture.write_census()
        self.fixture.bank_dir = self.fixture.bank_dir.parent/'observation-bank'
        self.fixture.fixture.output = self.fixture.bank_dir
        self.fixture.export_bank()
        self.prepared = self.fixture.load()
        source = next(s for s in self.prepared.sources if s['split']=='development')
        pair = self.prepared.development[0]
        path = self.root/'plan'; path.mkdir()
        def write(name, core):
            result = shared._write_new(path/name, core)
            return {'sha256':shared._digest(path/name), 'logical_sha256':result['record_sha256'], 'bytes':(path/name).stat().st_size}
        counts = {'pairs':1,'label_tokens':len(pair['labels']),'semantic_groups':1,'distinct_processed_sources':1}
        cohort = write('cohort.json', {'schema':'mfm-public-development-observation-cohort-v1',
            'census_sha256':self.prepared.census['record_sha256'], 'counts':counts,
            'pairs':[dict(case_id='d',source_key=source['source_key'],semantic_input_sha256=group,
                reference_count=1,encoded_pair_record_sha256=canonical_sha256(pair),
                source_ids_sha256=canonical_sha256(pair['source_ids']),labels_sha256=canonical_sha256(pair['labels']),
                label_tokens=len(pair['labels']),source_payload_sha256=pair['source_payload_sha256'])],
            'groups':[dict(semantic_input_sha256=group,observed_slots=dict(semantic_mr(text)),reference_count=1,
                case_ids=['d'],source_keys=[source['source_key']],original_mrs=[dict(source_key=source['source_key'],
                source_text=text,source_input_record_sha256=canonical_sha256(source),source_payload_sha256s=source['source_payload_sha256s'])])],
            'generation_source_keys':[source['source_key']], 'case_order_sha256':canonical_sha256(['d']),
            'semantic_group_order_sha256':canonical_sha256([group]),
            'generation_source_order_sha256':canonical_sha256([source['source_key']]),
            'semantic_gold_created':False,'targets_rewritten':False,'development_gradient_effect':False})
        controls = write('single-attribute-controls.json', {'schema':'mfm-observed-development-single-attribute-controls-v1',
            'cohort_file_sha256':cohort['sha256'],'cohort_logical_sha256':cohort['logical_sha256'],
            'controls':[],'counts':{'controls':0,'groups_with_controls':0,'groups_without_controls':1,
                'by_changed_slot':{'food':0,'name':0}},'unmatched_group_sha256s':[group]})
        decode = dict(generation_tokens=128,bos_token_id=2,eos_token_id=1,method='greedy_argmax',gradient=False,
            malformed_outputs_retained=True,repair_or_base_head_fallback=False,rng_and_model_mode_restored=True,
            skip_special_tokens=False,temperature_sampling=False,tokenizer_cleanup=False,source_order=[source['source_key']],
            output_unit='one output per distinct frozen development source_key per checkpoint')
        proposal = write('proposal.json', {'schema':'mfm-public-development-observation-input-proposal-v1',
            'cohort':cohort,'single_attribute_controls':controls,'accepted_bank_external_pin':self.prepared.bank['record_sha256'],
            'measurement_protocol':{'decode':decode,'ce':{'cases':1,'semantic_groups':1,'label_tokens':len(pair['labels'])}},
            'input_evidence':{'census':{'logical_sha256':self.prepared.census['record_sha256']},
                'encoded_pairs':{'sha256':self.prepared.census['encoded_pairs_sha256']},
                'source_inputs':{'sha256':self.prepared.census['source_inputs_sha256']}},
            'fit_program':{'epochs':20,'batch_size':16},'source_code':{}})
        assignment = [dict(semantic_input_sha256=group,component_sha256=component,partition=partition)]
        partition_counts = {'calibration':{'groups':0,'pairs_per_checkpoint':0,'edges':0},
            'audit':{'groups':1,'pairs_per_checkpoint':1,'edges':0}}
        rubric = write('semantic-adjudication-rubric-proposal-v2.json', {'schema':'mfm-public-development-semantic-adjudication-proposal-v2',
            'input_bindings':{'proposal.json':proposal,'cohort.json':cohort,'single-attribute-controls.json':controls},
            'accepted_bank_external_pin':self.prepared.bank['record_sha256'],
            'calibration_route_proposed':{'partition_version':'mfm-e2e-semantic-rubric-partition-v1',
                'actual_partition_counts':{'assignment_sha256':canonical_sha256(assignment),'counts':partition_counts,'cross_partition_edges':0}}})
        approval = write('root-prospective-rubric-v2-approval.json', {'schema':'mfm-root-prospective-semantic-workflow-approval-v1',
            'decision':'APPROVED_PROSPECTIVE_SOURCE_FIDELITY_OBSERVATION_WORKFLOW_ONLY',
            'proposal':{'file_sha256':rubric['sha256'],'record_sha256':rubric['logical_sha256']},
            'frozen_cohort':{'groups':1,'pairs_per_checkpoint':1,'existing_contrast_edges':0,'rewritten_targets':0},
            'source_only_partition':{'calibration_groups':0,'audit_groups':1,'calibration_pairs':0,'audit_pairs':1,
                'calibration_edges':0,'audit_edges':0,'cross_partition_edges':0},'semantic_evaluator_admitted':False,
            'automatic_score_or_threshold_admitted':False,'qualified_for_product':False})
        exposure = write('exposure.json', {'schema':'mfm-existing-development-output-exposure-ledger-v1',
            'cohort_file_sha256':cohort['sha256'],'assignment_sha256':canonical_sha256(assignment),
            'cohort_and_partition_unchanged':True,'drop_or_rebalance_performed':False,'fixed_audit_groups':1,
            'fixed_audit_outputs_both_conditions':2,'known_exposed_audit_groups_in_these_contexts':1,
            'known_exposures':[{'semantic_input_sha256':group,'case_id':'d','partition':'audit'}],
            'all_audit_outputs_untouched_claim_valid_for_these_contexts':False})
        return path, proposal['sha256'], approval['sha256'], path/'exposure.json', exposure['sha256']

    def test_frozen_source_plan_joins_original_fields_and_keeps_exposed_audit(self):
        pins = self.frozen_files()
        self.assertTrue(hasattr(producer,'read_observation_plan'), 'missing frozen plan custody')
        plan = producer.read_observation_plan(*pins, self.prepared)
        self.assertEqual(len(plan['source_order']),1)
        self.assertEqual(plan['counts']['pairs'],1)
        self.assertEqual(set(plan['assignments'].values()), {'audit'})
        self.assertEqual(plan['exposure']['known_exposed_audit_groups_in_these_contexts'],1)
        bad = list(pins); bad[1]='f'*64
        with self.assertRaisesRegex(ValueError,'FILE|file|digest|pin'):
            producer.read_observation_plan(*bad,self.prepared)
        bad = list(pins); bad[-1]='f'*64
        with self.assertRaisesRegex(ValueError,'FILE|file|digest|pin'):
            producer.read_observation_plan(*bad,self.prepared)
        self.prepared.pair_by_case['d']['source_ids']=[2,8]
        with self.assertRaisesRegex(ValueError,'pair|source|roster'):
            producer.read_observation_plan(*pins,self.prepared)

    def test_strict_actual_weight_loader_stops_corrupt_dtype_keys_before_use(self):
        self.assertTrue(hasattr(producer,'_load_observation_weights'),'missing strict observation tensor custody')
        from safetensors.torch import save_file
        path = self.root/'weights.safetensors'
        state = {k:v.detach().contiguous() for k,v in self.model.state_dict().items()}
        save_file(state,str(path))
        digest=shared._digest(path)
        producer._load_observation_weights(self.view,path,digest)
        with self.assertRaisesRegex(ValueError,'digest|bytes'):
            producer._load_observation_weights(self.view,path,'0'*64)
        path.unlink(); save_file({k:v.to(torch.bfloat16) for k,v in state.items()},str(path))
        with self.assertRaisesRegex(ValueError,'FP32|dtype'):
            producer._load_observation_weights(self.view,path,shared._digest(path))
        path.unlink(); save_file({'unrelated':torch.ones(1)},str(path))
        with self.assertRaisesRegex(ValueError,'keys|shape|state'):
            producer._load_observation_weights(self.view,path,shared._digest(path))

    def test_completed_runtime_rejects_missing_direct_bytes_before_model_construction(self):
        args=SimpleNamespace(provider='local-fixture',allocation_ref='tiny',device='cpu',
            completed_run_dir=self.root/'final',seed_dir=self.root/'seed',census_dir=self.root,
            census_sha256='a'*64,feature_bank=self.root,feature_bank_sha256='a'*64,
            plan_dir=self.root,proposal_sha256='a'*64,approval_sha256='a'*64,
            exposure_path=self.root/'exposure',exposure_sha256='a'*64,prepared_base_dir=self.root,
            prepared_base_receipt=self.root/'clone',prepared_clone_sha256='a'*64,
            final_weight_sha256='b'*64,seed_weight_sha256='c'*64)
        declared={'weight_paths':{'final':self.root/'absent-final.safetensors',
                                 'seed':self.root/'absent-seed.safetensors'}}
        with patch.object(producer.checkpoint_history,'require_readonly_root',return_value=self.root), \
             patch.object(producer.reader,'load_public_training',return_value=self.prepared), \
             patch.object(producer,'require_full_material'), \
             patch.object(producer,'read_observation_plan',return_value=self.minimal_plan()), \
             patch.object(producer,'verify_tokenizer_metadata',return_value={}), \
             patch.object(producer,'_read_completed_observation',return_value=declared), \
             patch.object(producer,'FormationDecisionSpecialist',side_effect=AssertionError('model constructed before bytes')), \
             patch.object(producer,'_load_observation_weights',side_effect=AssertionError('weight use before preflight')):
            with self.assertRaises(FileNotFoundError):
                producer.run_completed_observation(args)

    def test_completed_metadata_is_file_pinned_seed_matched_and_no_optimizer_opened(self):
        self.assertTrue(hasattr(producer,'_read_completed_observation'),'missing completed parent/control custody')
        from safetensors.torch import save_file
        final = self.root/'final'; final.mkdir()
        seed_dir = self.root/'seed'; seed_dir.mkdir()
        component_dir = final/'component'; component_dir.mkdir()
        config = self.model.config.record()
        metadata = {'asset_roster_sha256':'a'*64}
        curriculum = shared._write_new(final/'curriculum.json', {'schema':producer.CURRICULUM_SCHEMA,
            'task':producer.TASK,'objective':producer.OBJECTIVE,'sampling':producer.SAMPLER,
            'epochs':20,'batch_size':16,'seed':71,'train_cases':3,'development_cases':1,
            'development_gradient_or_selection_effect':False})
        run = shared._write_new(final/'run.json', {'schema':producer.RUN_SCHEMA,'task':producer.TASK,
            'purpose':'full-roster-public-fit','objective':producer.OBJECTIVE,'initialization':producer.INITIALIZATION,
            'specialist_config':config,'curriculum_sha256':curriculum['record_sha256'],'metadata':metadata,
            'census_sha256':self.prepared.census['record_sha256'],'feature_bank_sha256':self.prepared.bank['record_sha256'],
            'encoded_pairs_sha256':self.prepared.census['encoded_pairs_sha256'],
            'source_inputs_sha256':self.prepared.census['source_inputs_sha256'],'export_sha256':self.prepared.export['record_sha256'],
            'code':{'train_public_conditional_foundation.py':'a'*64},'seed':71,
            'base_weights_loaded':False,'private_evidence_opened':False,'final_payloads_opened':False,'qualified_for_product':False})
        save_file(self.model.state_dict(), str(seed_dir/'seed-control.safetensors'))
        seed = shared._write_new(seed_dir/'seed-control.json', {
            'schema':producer.SEED_SCHEMA,'run_sha256':run['record_sha256'],'specialist_config':config,
            'initialization':producer.INITIALIZATION,'optimizer_steps':0,
            'specialist_sha256':shared._digest(seed_dir/'seed-control.safetensors')})
        with torch.no_grad(): self.model.output_bias[1].add_(.3)
        save_file(self.model.state_dict(), str(component_dir/'specialist.safetensors'))
        weight = shared._digest(component_dir/'specialist.safetensors')
        binding = {'run':run['record_sha256'],'seed':seed['record_sha256'],'epochs':20,'batch_size':16,'train_cases':3}
        entries=[]
        for epoch in range(21):
            receipt=shared._seal({'schema':producer.CHECKPOINT_SCHEMA,'binding':binding,
                'cursor':{'epoch':epoch,'next_case':0,'step':epoch},'specialist_sha256':weight if epoch else seed['specialist_sha256'],
                'optimizer_sha256':'b'*64})
            entry=shared._seal({'schema':producer.checkpoint_history.ENTRY_SCHEMA,'session_index':0,
                'path':f'checkpoint-{epoch:08d}','receipt':receipt,'files':[],
                'previous_entry_sha256':entries[-1]['record_sha256'] if entries else None})
            entries.append(entry)
        checkpoint=final/'checkpoint-00000020';checkpoint.mkdir()
        shared._write_new(checkpoint/'checkpoint.json',{k:v for k,v in receipt.items() if k!='record_sha256'})
        history=shared._write_new(final/'history-00000020.json', {'schema':producer.checkpoint_history.SCHEMA,
            'purpose':'full-roster-public-fit','run':run,'curriculum':curriculum,'binding':binding,
            'origin':{'seed':seed,'files':[]},'entries':entries,'session_index':0,
            'sessions':[shared._seal({'schema':producer.checkpoint_history.SESSION_SCHEMA,'run_sha256':run['record_sha256'],
                'session_index':0,'allocation_ref':'tiny','parent_entry_count':0,'parent_history_sha256':None,
                'start_cursor':{'epoch':0,'next_case':0,'step':0}})]})
        shared._write_new(component_dir/'component.json', {'schema':producer.COMPONENT_SCHEMA,
            'task':producer.TASK,'objective':producer.OBJECTIVE,'run_sha256':run['record_sha256'],
            'curriculum_sha256':curriculum['record_sha256'],'seed_sha256':seed['record_sha256'],
            'checkpoint_sha256':receipt['record_sha256'],'checkpoint_lineage':[e['receipt']['record_sha256'] for e in entries],
            'checkpoint_history_sha256':history['record_sha256'],'specialist_sha256':weight,'specialist_config':config,
            'cursor':receipt['cursor'],'development_selection_effect':False,'qualified_for_product':False})
        args=SimpleNamespace(component_dir=component_dir,completed_run_dir=final,seed_dir=seed_dir,
            config_sha256=canonical_sha256(config),final_weight_sha256=weight,seed_weight_sha256=seed['specialist_sha256'])
        for name,path in [('component',component_dir/'component.json'),('run',final/'run.json'),
                          ('curriculum',final/'curriculum.json'),('history',final/'history-00000020.json'),
                          ('checkpoint',checkpoint/'checkpoint.json'),('seed',seed_dir/'seed-control.json')]:
            setattr(args,name+'_file_sha256',shared._digest(path))
        plan={'proposal':{'fit_program':{'epochs':20,'batch_size':16},
                          'source_code':{'trainer':{'git_export_sha256':'a'*64}}}}
        original_digest=shared._digest
        def metadata_only_digest(path):
            if Path(path).suffix=='.safetensors':
                raise AssertionError('tensor bytes opened during metadata admission')
            return original_digest(path)
        with patch('torch.load',side_effect=AssertionError('optimizer opened')), \
             patch.object(shared,'_digest',side_effect=metadata_only_digest):
            records=producer._read_completed_observation(args,self.prepared,metadata,plan)
        self.assertEqual(records['checkpoint']['cursor']['epoch'],20)
        producer._verify_observation_weight_bytes(records,args)
        args.seed_file_sha256='f'*64
        with self.assertRaisesRegex(ValueError,'FILE|digest'):
            producer._read_completed_observation(args,self.prepared,metadata,plan)


if __name__ == '__main__':
    unittest.main()
