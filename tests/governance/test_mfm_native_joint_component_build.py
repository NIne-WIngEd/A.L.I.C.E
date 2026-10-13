"""Actual tiny safetensor migration/export controls; no full-size fit claim."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist
from cognitive_kernel.formation_joint_learning import _interface_pins
from cognitive_kernel.formation_v1_specialist import SpecialistConfig


REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / 'scripts/mfm/build_native_joint_component.py'


def seal(row):
    return {**row, 'record_sha256': canonical_sha256(row)}


class NativeJointBuildTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / 'parent'
        self.parent.mkdir()
        config = SpecialistConfig(4, 32, 8, 1, 2, 16, 0, 2, 1, 4)
        torch.random.default_generator.manual_seed(81)
        model = FormationDecisionSpecialist(config)
        with torch.no_grad():
            model.output_bias.copy_(torch.arange(32, dtype=torch.float32) / 17)
        self.lexical = {k: v.clone() for k, v in model.state_dict().items()}
        weight = self.parent / 'component/specialist.safetensors'
        weight.parent.mkdir()
        save_file(self.lexical, str(weight))
        weight_pin = sha256(weight.read_bytes()).hexdigest()
        run = seal(dict(schema='mfm-public-conditional-foundation-run-v1',
            task='conditional_meaning_to_text', specialist_config=config.record(),
            initialization='fresh-token-position-scale-0.02-v1', seed=81,
            qualified_for_product=False))
        curriculum = seal(dict(schema='mfm-public-conditional-foundation-curriculum-v1', pairs=[]))
        cursor = {'epoch': 1, 'next_case': 0, 'step': 1}
        binding = dict(run=run['record_sha256'], seed='1'*64, train_cases=1, batch_size=1, epochs=1)
        checkpoint = seal(dict(schema='mfm-public-conditional-foundation-checkpoint-v1',
            binding=binding, cursor=cursor, specialist_sha256=weight_pin, optimizer_sha256='2'*64))
        entry = seal(dict(schema='mfm-public-conditional-foundation-history-entry-v1',
            path='checkpoint-00000001', previous_entry_sha256=None, session_index=0,
            receipt=checkpoint, files=[dict(path='checkpoint-00000001/specialist.safetensors',
                sha256=weight_pin, size=weight.stat().st_size)]))
        session = seal(dict(schema='mfm-public-conditional-foundation-session-v1',
            session_index=0, run_sha256=run['record_sha256'], parent_entry_count=0,
            parent_history_sha256=None, start_cursor={'epoch':0,'next_case':0,'step':0},
            allocation_ref='tiny-genuine-fixture'))
        history = seal(dict(schema='mfm-public-conditional-foundation-history-v1',
            binding=binding, run=run, curriculum=curriculum, entries=[entry], sessions=[session],
            origin={}, session_index=0, purpose='full-roster-public-fit'))
        component = seal(dict(schema='mfm-public-conditional-foundation-component-v1',
            task='conditional_meaning_to_text', specialist_config=config.record(), cursor=cursor,
            specialist_sha256=weight_pin, checkpoint_sha256=checkpoint['record_sha256'],
            checkpoint_lineage=[checkpoint['record_sha256']],
            checkpoint_history_sha256=history['record_sha256'], run_sha256=run['record_sha256'],
            curriculum_sha256=curriculum['record_sha256'], seed_sha256='1'*64,
            qualified_for_product=False))
        records = {}
        for name, row, path in [('component',component,'component/component.json'),
                               ('run',run,'run.json'),('checkpoint',checkpoint,'checkpoint-00000001/checkpoint.json'),
                               ('history',history,'history-00000001.json')]:
            file = self.parent / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(canonical_json_bytes(row))
            records[name] = dict(path=path, bytes=file.stat().st_size,
                file_sha256=sha256(file.read_bytes()).hexdigest(), record_sha256=row['record_sha256'])
        self.plan = seal(dict(schema='mfm-native-joint-build-plan-v1',
            purpose='tiny-mechanics-fixture', component_id='tiny-native-joint-seed',
            parent_kernel=dict(ref='tiny-genuine-fixture', numeric_id=1, version=1, source_revision='3'*40),
            parent=dict(weights=dict(path='component/specialist.safetensors', bytes=weight.stat().st_size,
                file_sha256=weight_pin), records=records),
            expected_parent=dict(config=config.record(), config_canonical_sha256=canonical_sha256(config.record()),
                cursor=cursor, lexical_parameter_count=sum(v.numel() for v in self.lexical.values()),
                history_entry_count=1, history_session_count=1),
            source=dict(builder_file_sha256=sha256(BUILDER.read_bytes()).hexdigest() if BUILDER.exists() else '4'*64,
                interface_pins=_interface_pins(), reviewed_joint_file_sha256=_interface_pins()['src/cognitive_kernel/formation_joint_learning.py']),
            initialization=dict(reference_seed=2026101301, copy_seed=2026101302),
            runtime=dict(device='cpu', threads=1, torch_version_family='.'.join(torch.__version__.split('.')[:2]),
                safetensors_required=True), joint_updates=0, training_authorized=False, full_MFM_qualified=False))

    def builder(self):
        self.assertTrue(BUILDER.is_file(), 'actual native joint component builder is missing')
        return importlib.import_module('scripts.mfm.build_native_joint_component')

    def run_build(self, plan=None, output=None):
        plan = self.plan if plan is None else plan
        return self.builder().build_component(plan, expected_plan_sha256=plan['record_sha256'],
            parent_root=self.parent, output_dir=self.root/'out' if output is None else output,
            fixture_only=True)

    def reseal(self, plan):
        return seal({k:v for k,v in plan.items() if k!='record_sha256'})

    def rewrite_parent(self, plan, *, weight_pin=None, cursor=None, binding=None, parent_entry_count=None):
        rows={key:json.loads((self.parent/desc['path']).read_bytes())
              for key,desc in plan['parent']['records'].items()}
        if weight_pin is not None:
            rows['component']['specialist_sha256']=weight_pin
            rows['checkpoint']['specialist_sha256']=weight_pin
        if cursor is not None:
            rows['component']['cursor']=cursor
            rows['checkpoint']['cursor']=cursor
        if binding is not None:
            rows['checkpoint']['binding']=binding
            rows['history']['binding']=binding
        if parent_entry_count is not None:
            rows['history']['sessions'][0]['parent_entry_count']=parent_entry_count
            rows['history']['sessions'][0]=self.reseal(rows['history']['sessions'][0])
        rows['checkpoint']=self.reseal(rows['checkpoint'])
        rows['history']['entries'][0]['receipt']=rows['checkpoint']
        rows['history']['entries'][0]=self.reseal(rows['history']['entries'][0])
        rows['history']=self.reseal(rows['history'])
        rows['component']['checkpoint_sha256']=rows['checkpoint']['record_sha256']
        rows['component']['checkpoint_lineage']=[rows['checkpoint']['record_sha256']]
        rows['component']['checkpoint_history_sha256']=rows['history']['record_sha256']
        rows['component']=self.reseal(rows['component'])
        for key,row in rows.items():
            desc=plan['parent']['records'][key];file=self.parent/desc['path']
            file.write_bytes(canonical_json_bytes(row))
            desc.update(bytes=file.stat().st_size,file_sha256=sha256(file.read_bytes()).hexdigest(),
                        record_sha256=row['record_sha256'])
        return self.reseal(plan)

    def test_complete_actual_chain_preserves_language_and_fresh_heads_and_reload(self):
        rng = torch.get_rng_state().clone()
        result = self.run_build()
        self.assertEqual(result['status'], 'native-joint-seed-built-reloaded-unqualified')
        self.assertEqual(result['joint_updates'], 0)
        self.assertFalse(result['training_authorized'])
        self.assertFalse(result['full_MFM_qualified'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        reference = load_file(str(self.root/'out/reference/specialist.safetensors'))
        joint = load_file(str(self.root/'out/joint/specialist.safetensors'))
        for key, original in self.lexical.items():
            self.assertTrue(torch.equal(original, reference[key]), key)
            self.assertTrue(torch.equal(original, joint[key]), key)
        self.assertTrue(all(torch.equal(reference[k], joint[k]) for k in reference))
        self.assertTrue(any(k.startswith('copy_') for k in joint))
        self.assertTrue(result['reference_fresh_heads_preserved'])
        self.assertTrue(result['copy_fresh_heads_preserved'])
        self.assertEqual(result['reloaded_tensor_count'], len(joint))
        self.assertEqual(result['inherited_lexical_tensor_count'], len(self.lexical))
        self.assertEqual(result['inherited_reference_tensor_count'], len(reference))
        self.assertEqual(result['parent_before'], result['parent_after'])
        self.assertEqual(result['source_before'], result['source_after'])
        from cognitive_kernel.formation_reference_specialist import FormationReferenceSpecialist
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026101301)
            fresh=FormationReferenceSpecialist(SpecialistConfig(**self.plan['expected_parent']['config']))
            for key,value in fresh.state_dict().items():
                if key not in self.lexical:self.assertTrue(torch.equal(value,reference[key]),key)
            torch.random.default_generator.manual_seed(2026101302)
            fresh=FormationJointLearning(SpecialistConfig(**self.plan['expected_parent']['config']))
            for key,value in fresh.state_dict().items():
                if key not in reference:self.assertTrue(torch.equal(value,joint[key]),key)

    def test_bad_weight_pin_retains_failure_and_never_replays_namespace(self):
        plan = deepcopy(self.plan)
        plan['parent']['weights']['file_sha256'] = '9'*64
        with self.assertRaises(ValueError): self.run_build(self.reseal(plan))
        failure = json.loads((self.root/'out/failure.json').read_text())
        self.assertEqual(failure['failure_stage'], 'parent_preflight')
        self.assertFalse((self.root/'out/reference').exists())
        before = (self.root/'out/failure.json').read_bytes()
        with self.assertRaises(FileExistsError): self.run_build()
        self.assertEqual(before, (self.root/'out/failure.json').read_bytes())

    def test_resealed_wrong_config_types_cursor_and_schema_refuse(self):
        for field, value in [('width', True), ('width', 8.0), ('max_target_tokens', 0)]:
            plan = deepcopy(self.plan)
            plan['expected_parent']['config'][field] = value
            plan['expected_parent']['config_canonical_sha256'] = canonical_sha256(plan['expected_parent']['config'])
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                self.run_build(self.reseal(plan), self.root/f'bad-{field}-{str(value)}')
        plan = deepcopy(self.plan)
        plan['expected_parent']['cursor']['step'] = True
        with self.assertRaises(ValueError): self.run_build(self.reseal(plan),self.root/'bad-cursor')
        plan = deepcopy(self.plan); plan['schema']='wrong'
        with self.assertRaises(ValueError): self.run_build(self.reseal(plan),self.root/'bad-schema')

    def test_original_json_duplicate_and_nonfinite_refuse_before_model(self):
        for label,raw in [('duplicate',b'{"schema":"x","schema":"y"}'),('nonfinite',b'{"x":NaN}')]:
            plan=deepcopy(self.plan); descriptor=plan['parent']['records']['component']
            file=self.parent/descriptor['path']; file.write_bytes(raw)
            descriptor['bytes']=len(raw); descriptor['file_sha256']=sha256(raw).hexdigest()
            with self.subTest(label=label),self.assertRaises(ValueError):
                self.run_build(self.reseal(plan),self.root/label)

    def test_numeric_overflow_json_refuses_at_original_reader(self):
        with self.assertRaises(ValueError):
            self.builder().read_json_bytes(b'{"objective":1e400}')

    def test_changed_current_builder_or_interface_refuses(self):
        for key in ('builder_file_sha256','interface_pins'):
            plan=deepcopy(self.plan)
            if key=='builder_file_sha256':plan['source'][key]='8'*64
            else:plan['source'][key][next(iter(plan['source'][key]))]='8'*64
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.run_build(self.reseal(plan),self.root/key)

    def test_nonfinite_and_non_fp32_original_state_refuse(self):
        for label in ('nonfinite','dtype'):
            state={k:v.clone() for k,v in self.lexical.items()}
            name=next(iter(state))
            if label=='nonfinite':state[name].reshape(-1)[0]=float('nan')
            else:state[name]=state[name].to(torch.float16)
            file=self.parent/'component/specialist.safetensors';save_file(state,str(file))
            plan=deepcopy(self.plan);pin=sha256(file.read_bytes()).hexdigest()
            plan['parent']['weights'].update(file_sha256=pin,bytes=file.stat().st_size)
            plan=self.rewrite_parent(plan,weight_pin=pin)
            with self.subTest(label=label),self.assertRaises(ValueError):
                self.run_build(plan,self.root/label)
            failure=json.loads((self.root/label/'failure.json').read_bytes())
            self.assertEqual(failure['failure_stage'],'reference_parent_load')

    def test_coherently_resealed_noninteger_original_cursor_refuses(self):
        for label,value in [('float',1.0),('bool',True)]:
            plan=self.rewrite_parent(deepcopy(self.plan),cursor={'epoch':1,'next_case':0,'step':value})
            with self.subTest(label=label),self.assertRaises(ValueError):
                self.run_build(plan,self.root/label)

    def test_coherently_resealed_binding_and_session_numeric_aliases_refuse(self):
        checkpoint=json.loads((self.parent/self.plan['parent']['records']['checkpoint']['path']).read_bytes())
        binding=deepcopy(checkpoint['binding']);binding['batch_size']=True
        plan=self.rewrite_parent(deepcopy(self.plan),binding=binding)
        with self.subTest(kind='binding'),self.assertRaises(ValueError):self.run_build(plan,self.root/'binding-alias')
        # Restore original binding while changing the actual original session.
        plan=self.rewrite_parent(deepcopy(self.plan),binding=checkpoint['binding'],parent_entry_count=0.0)
        with self.subTest(kind='session'),self.assertRaises(ValueError):self.run_build(plan,self.root/'session-alias')

    def test_actual_alice_memory_import_origin_is_bracketed(self):
        from unittest.mock import patch
        module=importlib.import_module('alice_memory')
        foreign=self.root/'foreign.py';foreign.write_text('foreign=1\n',encoding='utf-8')
        with patch.object(module,'__file__',str(foreign)),self.assertRaises(ValueError):
            self.builder().source_signature(self.plan['source'])

    def test_fixture_cannot_be_used_by_production_admission(self):
        builder=self.builder()
        with self.assertRaises(ValueError):
            builder.build_component(self.plan,expected_plan_sha256=self.plan['record_sha256'],
                parent_root=self.parent,output_dir=self.root/'production')

    def test_in_parent_output_refuses_before_any_parent_mutation(self):
        before={p.relative_to(self.parent).as_posix():sha256(p.read_bytes()).hexdigest()
                for p in self.parent.rglob('*') if p.is_file()}
        rejected=self.parent/'rejected-output'
        with self.assertRaises(ValueError):self.run_build(output=rejected)
        self.assertFalse(rejected.exists())
        after={p.relative_to(self.parent).as_posix():sha256(p.read_bytes()).hexdigest()
               for p in self.parent.rglob('*') if p.is_file()}
        self.assertEqual(before,after)


class NativeJointSourceMigrationTests(unittest.TestCase):
    """Whole-state identity, foreign-source refusal, and explicit cold migration."""
    def setUp(self):
        import io,shutil,subprocess,sys,tarfile
        NativeJointBuildTests.setUp(self)
        self.builder=importlib.import_module('scripts.mfm.build_native_joint_component')
        self.old_source=self.root/'old-source';self.old_source.mkdir()
        files={**_interface_pins(),'scripts/mfm/build_native_joint_component.py':sha256(BUILDER.read_bytes()).hexdigest()}
        for relative in files:
            target=self.old_source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/relative,target)
        # Semantically identical historical source with a distinct physical pin.
        changed=self.old_source/'src/cognitive_kernel/formation_reference_decision.py'
        with changed.open('ab') as stream:stream.write(b'\n# historical source fixture, arithmetic unchanged\n')
        interfaces={name:sha256((self.old_source/name).read_bytes()).hexdigest() for name in _interface_pins()}
        old_source=dict(builder_file_sha256=sha256((self.old_source/'scripts/mfm/build_native_joint_component.py').read_bytes()).hexdigest(),interface_pins=interfaces,reviewed_joint_file_sha256=interfaces['src/cognitive_kernel/formation_joint_learning.py'])
        old_plan=deepcopy(self.plan);old_plan['source']=old_source;old_plan=seal({k:v for k,v in old_plan.items() if k!='record_sha256'})
        self.package=self.root/'package';self.package.mkdir()
        (self.package/'build-plan.json').write_bytes(canonical_json_bytes(old_plan))
        roster={name:dict(bytes=(self.old_source/name).stat().st_size,file_sha256=sha256((self.old_source/name).read_bytes()).hexdigest()) for name in files}
        (self.package/'files.json').write_bytes(canonical_json_bytes(roster))
        with tarfile.open(self.package/'source.tar.xz','w:xz') as archive:
            for name in sorted(files):archive.add(self.old_source/name,arcname=name,recursive=False)
        self.complete_parent=self.root/'complete-parent';self.complete_parent.mkdir()
        child="""import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src'),str(root)]
from scripts.mfm import build_native_joint_component as b
from cognitive_kernel.formation_joint_learning import FormationJointLearning
import torch
torch.set_num_threads(1)
plan=b.read_json_bytes(Path(sys.argv[2]).read_bytes());cfg=b.SpecialistConfig(**plan['expected_parent']['config'])
source=b.source_signature(plan['source']);model=FormationJointLearning(cfg)
with torch.no_grad():
 for index,tensor in enumerate(model.state_dict().values()):tensor.fill_((index+1)/97)
out=Path(sys.argv[3]);provenance=dict(kind='zero-joint-update-seed',language_parent_cursor=plan['expected_parent']['cursor'],joint_updates=0,new_reference_heads_trained=False,new_copy_heads_trained=False,training_authorized=False,full_MFM_qualified=False)
m=model.export_component(out/'specialist.safetensors',out/'config.json',out/'component.json',component_id='historical-complete-tiny',provenance=provenance,provenance_sha256=b.canonical_sha256(provenance),source_pins=source,source_pins_sha256=b.canonical_sha256(source),evidence_scope='untrained_mechanics')
b._write_new(out/'result.json',b._seal(dict(schema='mfm-native-joint-component-build-result-v1',joint_manifest_file_sha256=b._digest(out/'component.json'),joint_weight_file_sha256=m['weight_file_sha256'],reloaded_tensor_count=len(model.state_dict()),original_language_cursor=plan['expected_parent']['cursor'],source_before=source,source_after=source,joint_updates=0,optimizer_created=False,Gemma_or_features_loaded=False,training_authorized=False,full_MFM_qualified=False)))
"""
        child_result=subprocess.run([sys.executable,'-I','-B','-c',child,str(self.old_source),str(self.package/'build-plan.json'),str(self.complete_parent)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        self.assertEqual(child_result.returncode,0,child_result.stdout.decode('utf-8'))
        def descriptor(root,name):
            p=root/name;return dict(path=name,bytes=p.stat().st_size,file_sha256=sha256(p.read_bytes()).hexdigest())
        state=load_file(str(self.complete_parent/'specialist.safetensors'));self.original_state={k:v.clone() for k,v in state.items()}
        manifest=json.loads((self.complete_parent/'component.json').read_bytes())
        self.migration_plan=seal(dict(schema='mfm-native-joint-source-migration-plan-v1',purpose='tiny-mechanics-fixture',component_id='migrated-complete-tiny',parent_kernel=dict(ref='tiny-genuine-fixture',numeric_id=1,version=1,source_revision='3'*40),
            parent={key:descriptor(self.complete_parent,name) for key,name in [('weights','specialist.safetensors'),('config','config.json'),('manifest','component.json'),('result','result.json')]},
            parent_source={key:descriptor(self.package,name) for key,name in [('capsule','source.tar.xz'),('files','files.json'),('build_plan','build-plan.json')]},
            expected_parent=dict(config=self.plan['expected_parent']['config'],cursor=self.plan['expected_parent']['cursor'],tensor_count=len(state),parameter_count=sum(x.numel() for x in state.values()),provenance_sha256=manifest['provenance_sha256'],source_pins_sha256=manifest['source_pins_sha256']),
            source=self.plan['source'],runtime=self.plan['runtime'],joint_updates=0,training_authorized=False,full_MFM_qualified=False))

    def migrate(self,plan=None,output=None):
        plan=plan or self.migration_plan
        self.assertTrue(callable(getattr(self.builder,'migrate_component',None)), 'explicit complete-state source migration is missing')
        return self.builder.migrate_component(plan,expected_plan_sha256=plan['record_sha256'],parent_root=self.complete_parent,source_package_root=self.package,output_dir=output or self.root/'migrated',fixture_only=True)

    def test_complete_state_cold_migration_preserves_every_tensor_and_rng(self):
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        parent_manifest=json.loads((self.complete_parent/'component.json').read_bytes());cfg=SpecialistConfig(**self.plan['expected_parent']['config'])
        with self.assertRaises(ValueError):
            FormationJointLearning(cfg).load_component(self.complete_parent/'specialist.safetensors',self.complete_parent/'config.json',self.complete_parent/'component.json',weight_sha256=parent_manifest['weight_file_sha256'],config_sha256=parent_manifest['config_file_sha256'],manifest_sha256=sha256((self.complete_parent/'component.json').read_bytes()).hexdigest(),expected_provenance_sha256=parent_manifest['provenance_sha256'],expected_source_pins=parent_manifest['source_pins'],expected_evidence_scope='untrained_mechanics')
        rng=torch.get_rng_state().clone();result=self.migrate();self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        loaded=load_file(str(self.root/'migrated/joint/specialist.safetensors'))
        self.assertEqual(set(loaded),set(self.original_state))
        for name,value in self.original_state.items():self.assertTrue(torch.equal(value,loaded[name]),name)
        self.assertEqual(result['reloaded_tensor_count'],len(self.original_state));self.assertEqual(result['joint_updates'],0)
        self.assertTrue(result['complete_parent_state_preserved']);self.assertFalse(result['optimizer_created'])
        self.assertEqual(result['original_language_cursor'],{'epoch':1,'next_case':0,'step':1})
        self.assertEqual(result['parent_before'],result['parent_after']);self.assertEqual(result['source_before'],result['source_after'])
        self.assertTrue(result['old_cold_load']['isolated_process']);self.assertEqual(result['old_cold_load']['interface_pins'],parent_manifest['interface_pins'])

    def test_migration_rejects_corrupt_parent_before_state_transfer(self):
        p=self.complete_parent/'specialist.safetensors';raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
        with self.assertRaises(ValueError):self.migrate()
        self.assertFalse((self.root/'migrated/joint').exists())

    def test_migration_rejects_resealed_config_or_current_source(self):
        for label in ('config','source'):
            plan=deepcopy(self.migration_plan)
            if label=='config':plan['expected_parent']['config']['width']=8.0
            else:plan['source']['interface_pins'][next(iter(plan['source']['interface_pins']))]='0'*64
            plan=seal({k:v for k,v in plan.items() if k!='record_sha256'})
            with self.subTest(label=label),self.assertRaises(ValueError):self.migrate(plan,self.root/label)

    def test_migration_rejects_changed_old_capsule_and_parent_output(self):
        original=(self.package/'source.tar.xz').read_bytes();(self.package/'source.tar.xz').write_bytes(original+b'x')
        with self.assertRaises(ValueError):self.migrate()
        with self.assertRaises(ValueError):self.migrate(output=self.complete_parent/'forbidden')
        self.assertFalse((self.complete_parent/'forbidden').exists())

    def test_coherently_repinned_fp16_parent_fails_original_strict_cold_gate(self):
        state={k:v.clone() for k,v in self.original_state.items()};state[next(iter(state))]=state[next(iter(state))].half()
        weight=self.complete_parent/'specialist.safetensors';save_file(state,str(weight))
        plan=deepcopy(self.migration_plan)
        manifest=json.loads((self.complete_parent/'component.json').read_bytes());manifest['weight_file_sha256']=sha256(weight.read_bytes()).hexdigest();manifest['weight_bytes']=weight.stat().st_size
        (self.complete_parent/'component.json').write_bytes(canonical_json_bytes(manifest))
        result=json.loads((self.complete_parent/'result.json').read_bytes());result.pop('record_sha256');result['joint_weight_file_sha256']=manifest['weight_file_sha256'];result['joint_manifest_file_sha256']=sha256((self.complete_parent/'component.json').read_bytes()).hexdigest()
        (self.complete_parent/'result.json').write_bytes(canonical_json_bytes(seal(result)))
        for key in ('weights','manifest','result'):
            d=plan['parent'][key];p=self.complete_parent/d['path'];d.update(bytes=p.stat().st_size,file_sha256=sha256(p.read_bytes()).hexdigest())
        plan=seal({k:v for k,v in plan.items() if k!='record_sha256'})
        with self.assertRaises(ValueError):self.migrate(plan)
        failure=json.loads((self.root/'migrated/failure.json').read_bytes())
        self.assertEqual(failure['failure_stage'],'isolated_original_cold_load')
        self.assertFalse((self.root/'migrated/joint').exists())

    def test_fixture_migration_cannot_grant_production_construction(self):
        with self.assertRaises(ValueError):
            self.builder.migrate_component(self.migration_plan,expected_plan_sha256=self.migration_plan['record_sha256'],parent_root=self.complete_parent,source_package_root=self.package,output_dir=self.root/'production')


if __name__=='__main__':
    unittest.main()
