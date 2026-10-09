"""Fit transport/entrypoint contracts; no hosted API or model execution.

Original seven controls RED at missing-runtime assertion on 2026-10-08,
SHA f1a6cbc112d75e27ec0722400ca2ddbcc476a3d037a9b96c6e1810f166920ad6.
Root then selected E20. Source inspection established history CLI consumes its
logical seal, and checkpoint staging is the Nth state rather than an extra one.
The updated fixtures track these actual contracts without claiming hosted fit.
"""
import ast
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
from hashlib import sha256
import os
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


class FitTransportControls(unittest.TestCase):
    def module(self):
        name='scripts.mfm.kaggle_public_foundation_fit_20261008_runtime'
        self.assertIsNotNone(importlib.util.find_spec(name),'full fit-purpose runtime is missing')
        return importlib.import_module(name)

    def recipe(self):
        return {'epochs':20,'batch_size':16,'save_every_steps':2288,'seed':20261008,'logit_chunk_tokens':16,
            'generation_tokens':128,'learning_rate':0.0001,'weight_decay':0.0,'beta1':0.9,'beta2':0.999,'adam_eps':1e-8}

    def manifest(self):
        return {'purpose':'full-roster-public-fit','recipe':self.recipe(),
            'session':{'stop_step':6864,'session_index':0,'resume_history':None,'source_roots':[]}}

    def inputs(self):
        return {'layout':{'census':'census','bank':'bank'},'pins':{'census_sha256':'1'*64,'feature_bank_sha256':'2'*64,
            'prepared_clone_sha256':'3'*64,'config_sha256':'4'*64}}

    def test_fresh_actual_argv_is_fit_purpose_with_complete_boundary_and_no_timer(self):
        m=self.module()
        argv=m.trainer_argv(self.manifest(),self.inputs(),material=Path('/verified/material'),metadata=Path('/verified/metadata'),
            output=Path('/new/owned/trainer'),allocation_ref='mkrayanyan/new-fit/allocation-fixture',history_path=None,source_roots={})
        for flag,value in [('--epochs','20'),('--batch-size','16'),('--save-every-steps','2288'),('--stop-step','6864'),('--session-index','0')]:
            self.assertEqual(argv[argv.index(flag)+1],value)
        self.assertEqual(argv[:2],['-m','scripts.mfm.train_public_conditional_foundation'])
        self.assertEqual(argv[argv.index('--output-dir')+1],str(Path('/new/owned/trainer')))
        for flag in ('--measure-only','--session-seconds','--timeout','--resume-checkpoint','--resume-history'):
            self.assertNotIn(flag,argv)

    def test_complete_epoch_stop_and_three_new_checkpoint_storage_contract_reject_invalid_recipe(self):
        m=self.module();m.validate_program(self.manifest())
        for mutate in [lambda d:d['session'].update(stop_step=6863),lambda d:d['session'].update(stop_step=True),
            lambda d:d['session'].update(stop_step=45761),lambda d:d['recipe'].update(batch_size=8),
            lambda d:d['recipe'].update(save_every_steps=1),lambda d:d['recipe'].update(session_seconds=7200),
            lambda d:d.update(purpose='construction-measurement-only')]:
            candidate=deepcopy(self.manifest());mutate(candidate)
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):m.validate_program(candidate)

    def test_resume_actual_argv_uses_new_owned_output_and_explicit_all_readonly_history_sources(self):
        m=self.module();d=self.manifest()
        d['session']={'stop_step':13728,'session_index':1,'resume_history':{'path':'fit/trainer/history-00006864.json','sha256':'5'*64,
            'record_sha256':'6'*64,'source_session_index':0},'source_roots':[{'session_index':0,'kernel_id':'mkrayanyan/prior-fit',
                'kernel_numeric_id':100,'version_number':1,'trainer_root':'fit/trainer','allocation_ref':'mkrayanyan/prior-fit/allocation-fixture'}]}
        argv=m.trainer_argv(d,self.inputs(),material=Path('/inputs/material'),metadata=Path('/inputs/metadata'),output=Path('/new/trainer'),
            allocation_ref='mkrayanyan/next-fit/allocation-fixture',history_path=Path('/readonly/fit/trainer/history-00006864.json'),source_roots={0:Path('/readonly/fit/trainer')})
        self.assertEqual(argv[argv.index('--resume-history')+1],str(Path('/readonly/fit/trainer/history-00006864.json')))
        self.assertEqual(argv[argv.index('--resume-history-sha256')+1],'6'*64)
        self.assertEqual(argv[argv.index('--history-source')+1],'0='+str(Path('/readonly/fit/trainer')))
        self.assertEqual(argv[argv.index('--output-dir')+1],str(Path('/new/trainer')))
        self.assertNotIn('--resume-checkpoint',argv)
        for change in [lambda x:x['session']['source_roots'][0].update(version_number=2),
            lambda x:x['session']['source_roots'][0].update(kernel_id='other-owner/prior-fit'),
            lambda x:x['session'].update(source_roots=[]),lambda x:x['session'].update(resume_history=None)]:
            bad=deepcopy(d);change(bad)
            with self.assertRaises(ValueError):m.validate_program(bad)

    def test_generated_embedded_runtime_bootstraps_actual_fit_module_without_model_or_checkout_fallback(self):
        m=self.module();p=importlib.import_module('scripts.mfm.kaggle_public_foundation_fit_20261008')
        main=p.main_source('a'*64)
        ast.parse(main)
        namespace={'__name__':'controlled_import','__package__':None}
        exec(compile(main,'controlled-fit-main','exec'),namespace)
        argv=namespace['trainer_argv'](self.manifest(),self.inputs(),material=Path('/material'),metadata=Path('/metadata'),
            output=Path('/output'),allocation_ref='fixture',history_path=None,source_roots={})
        self.assertIn('--stop-step',argv);self.assertNotIn('--measure-only',argv)
        self.assertEqual(namespace['common'].SCHEMA,'mfm-kaggle-full-public-construction-transport-v1')

    def test_tiny_code_dataset_metadata_reuses_existing_bank_without_retransport_or_timeout(self):
        m=self.module();p=importlib.import_module('scripts.mfm.kaggle_public_foundation_fit_20261008')
        metadata=p.kernel_metadata('mkrayanyan/full-fit-001','mkrayanyan/code-fit-001','mkrayanyan/rayan-mfm-full-inputs-imports-20261008',[])
        self.assertEqual(metadata['dataset_sources'],['mkrayanyan/rayan-mfm-full-inputs-imports-20261008','mkrayanyan/code-fit-001'])
        self.assertEqual(metadata['kernel_sources'],[])
        self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_internet'])
        self.assertNotIn('timeout',metadata);self.assertNotIn('session_timeout_seconds',metadata)
        with self.assertRaises(ValueError):p.kernel_metadata('mkrayanyan/full-fit-001','mkrayanyan/code-fit-001','other/bank',[])

    def test_readonly_source_mount_guard_rejects_plain_writable_directory(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):m.require_readonly_mount(Path(temp),Path(temp))

    def test_initial_input_guard_accepts_real_ordinary_file_and_refuses_escape_directory_and_links(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'input';root.mkdir();path=root/'public.mfmblob';path.write_bytes(b'public')
            self.assertEqual(m.confined_regular_input(path,root),path.resolve())
            outside=Path(temp)/'outside.mfmblob';outside.write_bytes(b'public')
            with self.assertRaises(ValueError):m.confined_regular_input(outside,root)
            with self.assertRaises(ValueError):m.confined_regular_input(root,root)
            # Unit-check the path link predicate without claiming Windows can
            # create a symlink or that the fixture has a real readonly mount.
            with patch.object(Path,'is_symlink',return_value=True):
                with self.assertRaises(ValueError):m.confined_regular_input(path,root)

    def test_preserved_initial_readonly_guard_candidate_is_exact_fixture(self):
        path=Path(__file__).parent/'fixtures/kaggle_fit_runtime_948d4202_before_input_guard.py.txt'
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),'948d4202b8830c03ff5850d50a9a0fb1d28405770a7ea9086810bf32459fc493')

    def test_measured_checkpoint_budget_includes_atomic_scratch_and_shared_input_filesystem(self):
        m=self.module();d=self.manifest()
        inputs={'archives':{'material':{'uncompressed_bytes':1936767012},'metadata':{'uncompressed_bytes':32210372},
            'dependencies':{'uncompressed_bytes':12296065}}}
        plan=m.storage_plan(d,inputs,code_bytes=6_000_000)
        self.assertGreater(plan['working_required_bytes'],16_000_000_000)
        self.assertGreater(plan['extraction_required_bytes'],1_987_000_000)
        self.assertLess(plan['working_required_bytes']+plan['extraction_required_bytes'],20_957_446_144)
        self.assertEqual(plan['additional_checkpoint_transaction_bytes'],0)
        self.assertEqual(plan['component_bytes'],0)
        m.validate_storage(plan,working_free=20_957_446_144,extraction_free=60_000_000_000,same_filesystem=False)
        m.validate_storage(plan,working_free=20_957_446_144,extraction_free=20_957_446_144,same_filesystem=True)
        with self.assertRaises(ValueError):m.validate_storage(plan,working_free=17_000_000_000,extraction_free=17_000_000_000,same_filesystem=True)

    def test_later_sessions_do_not_copy_origin_and_only_final_program_publishes_component(self):
        m=self.module();d=self.manifest()
        roots=[]
        for i in range(6):
            roots.append({'session_index':i,'kernel_id':f'mkrayanyan/prior-fit-{i}','kernel_numeric_id':100+i,
                'version_number':1,'trainer_root':f'fit-{i}/trainer','allocation_ref':f'mkrayanyan/prior-fit-{i}/allocation-fixture'})
        d['session']={'stop_step':45760,'session_index':6,'resume_history':{'path':'fit-5/trainer/history-00041184.json',
            'sha256':'5'*64,'record_sha256':'6'*64,'source_session_index':5},'source_roots':roots}
        inputs={'archives':{role:{'uncompressed_bytes':1} for role in ('material','metadata','dependencies')}}
        plan=m.storage_plan(d,inputs,code_bytes=1)
        self.assertEqual(plan['new_epoch_states'],2)
        self.assertEqual(plan['component_bytes'],m.MODEL_BYTES)
        self.assertEqual(plan['working_required_bytes'],2*m.CHECKPOINT_BYTES+m.MODEL_BYTES+m.METADATA_RESERVE_BYTES)
        self.assertFalse(plan['old_history_copied'])
        with self.assertRaises(ValueError):m.trainer_argv(d,self.inputs(),material=Path('/m'),metadata=Path('/p'),
            output=Path('/output'),allocation_ref='fixture',history_path=Path('/history'),source_roots={0:Path('/root')})

    def test_original_accepted_small_manifest_is_sufficient_without_any_large_blobs(self):
        m=self.module()
        source=Path(__file__).resolve().parents[3]/'kaggle-full-construction-imports-20261008/dataset/mfm-construction-transport.json'
        original=m.accepted_input_manifest(source)
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/source.name;target.write_bytes(source.read_bytes())
            self.assertEqual(m.accepted_input_manifest(target),original)
            self.assertEqual([p.name for p in Path(temp).iterdir()],[source.name])
            target.write_bytes(target.read_bytes()+b' ')
            with self.assertRaises(ValueError):m.accepted_input_manifest(target)

    def test_real_fit_manifest_external_seal_refuses_tamper_and_changed_input_identity(self):
        m=self.module();d=self.manifest()
        d.update(schema=m.SCHEMA,dataset_id='mkrayanyan/new-code',kernel_id='mkrayanyan/new-fit',
            input_dataset={'id':m.INPUT_DATASET,'numeric_id':12455297,'version_number':1,'is_private':True,
                'manifest_name':m.common.MANIFEST_NAME,'manifest_file_sha256':m.INPUT_MANIFEST_FILE_SHA256,
                'manifest_sha256':m.INPUT_MANIFEST_SHA256})
        d=m.common.seal(d)
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/m.MANIFEST_NAME;m.common.write_record(target,d)
            self.assertEqual(m.read_fit_manifest(target,d['record_sha256']),d)
            changed=deepcopy(d);changed['recipe']['epochs']=2;target.write_bytes(m.common.canonical(changed))
            with self.assertRaises(ValueError):m.read_fit_manifest(target,d['record_sha256'])
            body={k:v for k,v in d.items() if k!='record_sha256'};body['input_dataset']['version_number']=2
            changed=m.common.seal(body);target.write_bytes(m.common.canonical(changed))
            with self.assertRaises(ValueError):m.read_fit_manifest(target,changed['record_sha256'])

    def test_real_history_roster_maps_exact_allocation_and_refuses_changed_or_ambiguous_parent(self):
        m=self.module();d=self.manifest()
        session=m.common.seal({'schema':'mfm-public-conditional-foundation-session-v1','session_index':0,
            'allocation_ref':'mkrayanyan/prior-fit/allocation-fixture'})
        history=m.common.seal({'schema':'mfm-public-conditional-foundation-history-v1','session_index':0,'sessions':[session]})
        with tempfile.TemporaryDirectory() as temp:
            mount=Path(temp);root=mount/'private-mount'/'fit'/'trainer';root.mkdir(parents=True)
            path=root/'history-00006864.json';m.common.write_record(path,history);m.common.write_record(root/'session.json',session)
            d['session']={'stop_step':13728,'session_index':1,'resume_history':{'path':'fit/trainer/'+path.name,
                'sha256':m.common.digest(path),'record_sha256':history['record_sha256'],'source_session_index':0},
                'source_roots':[{'session_index':0,'kernel_id':'mkrayanyan/prior-fit','kernel_numeric_id':100,'version_number':1,
                    'trainer_root':'fit/trainer','allocation_ref':session['allocation_ref']}]}
            # Mocking this OS flag checks adapter logic only; plain writable
            # rejection above is real, actual Linux mount proof is separate.
            with patch.object(os,'ST_RDONLY',1,create=True),patch.object(os,'statvfs',return_value=SimpleNamespace(f_flag=1),create=True):
                actual,roots=m.resolve_history(d,mount)
                self.assertEqual(actual,path.resolve());self.assertEqual(roots,{0:root.resolve()})
                bad=deepcopy(d);bad['session']['source_roots'][0]['allocation_ref']='mkrayanyan/prior-fit/allocation-other'
                with self.assertRaises(ValueError):m.resolve_history(bad,mount)
                second=mount/'second'/'fit'/'trainer';second.mkdir(parents=True)
                (second/path.name).write_bytes(path.read_bytes())
                with self.assertRaises(ValueError):m.resolve_history(d,mount)

    def test_code_extraction_verifies_actual_archive_roster_and_source_bytes(self):
        m=self.module();import zipfile
        with tempfile.TemporaryDirectory() as temp:
            blob=Path(temp)/'code.mfmblob'
            with zipfile.ZipFile(blob,'w') as z:z.writestr('src/fixture.py',b'print("public")\n')
            descriptor=m.common.describe_archive(blob,'zip');destination=Path(temp)/'code'
            m.common.extract_verified(blob,destination,descriptor)
            manifest={'source_files':m.common.file_inventory(destination)}
            m.verify_source(destination,manifest)
            (destination/'src'/'fixture.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):m.verify_source(destination,manifest)

    def test_prepare_parser_has_no_bank_blob_or_timer_input_and_fixes_full_recipe(self):
        m=self.module();p=importlib.import_module('scripts.mfm.kaggle_public_foundation_fit_20261008')
        argv=['--source-root','source','--source-revision','a'*40,'--output-dir','new','--input-manifest','small.json',
            '--dataset-id','mkrayanyan/new-code','--kernel-id','mkrayanyan/new-fit']
        for name,value in self.recipe().items():argv+=['--'+name.replace('_','-'),str(value)]
        parsed=p.arguments(argv)
        self.assertEqual(parsed.epochs,20);self.assertEqual(parsed.stop_step,6864)
        self.assertFalse(any('archive' in key or 'timeout' in key or 'session_seconds' in key for key in vars(parsed)))


if __name__=='__main__':unittest.main()
