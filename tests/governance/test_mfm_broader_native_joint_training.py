"""Actual broader originals and refusal before processor/feature/seed loading."""
from hashlib import sha256
from contextlib import ExitStack
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import canonical_json_bytes
from cognitive_kernel.canonical import canonical_sha256
from tests.governance.test_mfm_broader_paired_teaching import (
    ROOT,actual_roster,NAMES,ORIGINAL_INTEGRATION,ORIGINAL_INTEGRATION_REASON)

HERE=ROOT/'work/mfm-broader-joint-learning-20261013'
PRIVATE=ROOT/'work/mfm-joint-original-inventory-roundtrip-design-20261011/material-read-export-construction-001/dependency-tree-001/scripts/mfm'

def j5_module():
    name='scripts.mfm.formation_joint_input_v5'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,PRIVATE/'formation_joint_input_v5.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def ref(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path.read_bytes()).hexdigest()}

@unittest.skipUnless(ORIGINAL_INTEGRATION,ORIGINAL_INTEGRATION_REASON)
class BroaderSharedPayloadAuditTests(unittest.TestCase):
    """Physical path audit only; constructing these records grants no rights."""
    def admission(self):
        from cognitive_kernel.formation_dataset_admission import (
            CorpusAdmission,AdmittedCase,PayloadRef)
        original=ROOT/'work/mfm-broader-foundation-material-20261013'
        shared=ref(original/'world-source-successor-002/artifacts/actor-capsule.bin')
        source=PayloadRef(shared['path'],shared['sha256'])
        cases=[]
        for name in ('shared-session','assistant-revision'):
            target=ref(original/'actual-004/targets'/(name+'.json'))
            cases.append(AdmittedCase('audit-only-'+name,'train',(source,),
                PayloadRef(target['path'],target['sha256']),()))
        return CorpusAdmission('physical-path-audit-only','1'*64,
            tuple(cases),(),(),ROOT)

    def test_connected_cases_audit_shared_physical_original_once(self):
        from scripts.mfm import train_native_joint_formation as t
        admission=self.admission()
        # Preserve the original failing condition at the unchanged audit API.
        flattened=tuple(path for row in admission.train
            for path in (*(r.path for r in row.source_payloads),row.target_payload.path))
        with self.assertRaisesRegex(ValueError,'duplicate gradient payload'):
            admission.audit_handoff(gradient_paths=flattened,development_paths=())
        self.assertTrue(hasattr(t,'_audit_admitted_payloads'),
            'ordered physical-path handoff helper is missing')
        audit=t._audit_admitted_payloads(admission)
        expected=list(dict.fromkeys(flattened))
        self.assertEqual([r['path'] for r in audit['gradient_inputs']],expected)
        self.assertEqual(len(audit['gradient_inputs']),3)
        self.assertEqual(len(admission.train),2)
        self.assertTrue(audit['final_payloads_excluded'])

    def test_shared_path_union_keeps_conflicting_digest_and_split_refusals(self):
        from dataclasses import replace
        from cognitive_kernel.formation_dataset_admission import PayloadRef
        from scripts.mfm import train_native_joint_formation as t
        admission=self.admission();first,second=admission.train
        self.assertTrue(hasattr(t,'_audit_admitted_payloads'),
            'ordered physical-path handoff helper is missing')
        bad=replace(second,source_payloads=(PayloadRef(first.source_payloads[0].path,'0'*64),))
        with self.assertRaisesRegex(ValueError,'conflicting digests'):
            t._audit_admitted_payloads(replace(admission,train=(first,bad)))
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            t._audit_admitted_payloads(replace(admission,train=(first,),
                development=(replace(second,split='development'),)))


@unittest.skipUnless(ORIGINAL_INTEGRATION,ORIGINAL_INTEGRATION_REASON)
class BroaderPhysicalAuthorProvenanceTests(unittest.TestCase):
    def test_model_author_admission_binds_exact_original_metadata_for_both_profiles(self):
        from copy import deepcopy
        from scripts.mfm import train_native_joint_formation as t
        self.assertTrue(hasattr(t,'_validate_model_author_originals'),
            'physical original author-admission binding is missing')
        fragment=json.loads((ROOT/'work/mfm-broader-corpus-assembly-20261015/actual-001/'
            'program-metadata.json').read_bytes())
        originals=tuple((row['profile'],ROOT/row['accepted']['path']) for row in fragment['cases'])
        self.assertEqual(len(originals),9)
        for profile,path in originals:
            accepted=json.loads(path.read_bytes())
            t._validate_model_author_originals(accepted,profile)
            for key in ('raw_author_draft_file_sha256','projected_view_file_sha256',
                        'support_file_sha256','annotation_file_sha256','author_id','author_producer_version'):
                bad=deepcopy(accepted);bad[key]='f'*64
                bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
                with self.subTest(profile=profile,key=key):
                    with self.assertRaisesRegex(ValueError,'physical model-author'):
                        t._validate_model_author_originals(bad,profile)
            if profile=='broader-firstparty-successor':
                bad=deepcopy(accepted)
                bad['original_material_descriptors']['source_view']['file_sha256']='e'*64
                bad['record_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='record_sha256'})
                with self.assertRaisesRegex(ValueError,'physical model-author'):
                    t._validate_model_author_originals(bad,profile)


@unittest.skipUnless(ORIGINAL_INTEGRATION,ORIGINAL_INTEGRATION_REASON)
class BroaderPhysicalMaterialTests(unittest.TestCase):
    def test_later_world_metadata_cannot_change_cut35_native_model_input(self):
        from scripts.mfm import export_joint_source_features as f
        from tests.governance.test_mfm_broader_paired_teaching import NEW
        native=ROOT/'work/mfm-broader-native-geometry-20261013/actual-002'
        obs=json.loads((native/'source-observation-receipt.json').read_bytes());entry=obs['source_files'][0]
        original=json.loads((native/entry['path']).read_bytes())
        world=json.loads((NEW/'world.json').read_bytes())
        # Mutate only future row37's artifact descriptor and reseal metadata.
        # This is not admitted connected-world ancestry. Source feature
        # preflight must still derive exclusively the already frozen cut35.
        world['branches'][0]['records'][36]['artifact']['file_sha256']='0'*64
        world['world_sha256']=canonical_sha256({k:v for k,v in world.items() if k!='world_sha256'})
        with TemporaryDirectory(dir=HERE) as temp:
            path=Path(temp)/'future-metadata.json';raw=canonical_json_bytes(world);path.write_bytes(raw)
            descriptor={'case_id':'j6-reading-room-shared-session','profile':'broader-source-only',
              'source_file':str(native/entry['path']),'source_file_sha256':entry['file_sha256'],
              'source_record_sha256':entry['record_sha256'],'j5_file_sha256':obs['original_j5_file_sha256'],
              'proposed_max_source_tokens':59303,'observation_file':str(native/'source-observation-receipt.json'),
              'observation_file_sha256':sha256((native/'source-observation-receipt.json').read_bytes()).hexdigest(),
              'observation_record_sha256':obs['record_sha256'],'whole_world_file':str(path),
              'whole_world_file_sha256':sha256(raw).hexdigest(),'whole_world_sha256':world['world_sha256']}
            _,_,alignment,plain=f._preflight_case(descriptor,None,j5_module())
            self.assertEqual(plain,original['native_batch'])
            self.assertEqual(alignment,original['alignment'])
            self.assertEqual(len(plain['input_ids'][0]),56189)
            self.assertEqual(alignment['candidate_inventory_sha256'],original['alignment']['candidate_inventory_sha256'])

    def test_actual_broader_fifteen_metadata_roles_and_saved_native_fields(self):
        from scripts.mfm import train_native_joint_formation as t,export_joint_source_features as f
        import torch
        torch.set_num_threads(1)
        manifest=HERE/'material-002/shared-session-material15.json'
        material,descriptor=t.read_broader_material15(ROOT,ref(manifest),j5_source=j5_module())
        self.assertEqual(len(material['metadata']),15)
        self.assertEqual(descriptor['profile'],'broader-source-only')
        self.assertEqual(descriptor['proposed_max_source_tokens'],59303)
        _,batch,alignment,plain=f._preflight_case(descriptor,None,j5_module())
        source=json.loads(Path(descriptor['source_file']).read_bytes())
        self.assertEqual(plain,source['native_batch'])
        self.assertEqual(set(plain),{'input_ids','attention_mask','mm_token_type_ids'})
        self.assertEqual(tuple(batch['input_ids'].shape),(1,56189))
        self.assertEqual(alignment,source['alignment'])
        self.assertEqual(source['meaning']['episode_snapshot']['historical_prefix_proof']['cut']['sequence'],35)
        self.assertEqual(len(json.loads(Path(descriptor['whole_world_file']).read_bytes())['branches'][0]['records']),37)
        self.assertNotIn('target',descriptor)
        with self.assertRaises(ValueError):f._preflight_case({**descriptor,'teacher_target':'forbidden'},None,j5_module())
        with self.assertRaises(ValueError):f._preflight_case({**descriptor,'observation_file_sha256':'0'*64},None,j5_module())

    def test_original_six_material_reader_still_accepts_unchanged_physical_case(self):
        from scripts.mfm.train_native_joint_formation import read_j6_material15
        path=ROOT/'work/mfm-first-party-full-source-successor-plan-20261013/j6-material15-001/actual-001/initial-state-material15.json'
        material,descriptor=read_j6_material15(ROOT,ref(path),j5_source=j5_module())
        self.assertEqual(descriptor['profile'],'j6-source-only')
        self.assertEqual(descriptor['proposed_max_source_tokens'],52488)
        self.assertEqual(material['case_id'],'j6-reading-room-initial-state')

    def test_genuine_nine_case_program_refuses_missing_rights_before_heavy_loads(self):
        from scripts.mfm import train_native_joint_formation as t,export_joint_source_features as f
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        from transformers import AutoProcessor
        roster=actual_roster()
        private={role:{'path':str(PRIVATE/name),'sha256':sha256((PRIVATE/name).read_bytes()).hexdigest()}
          for role,name in {'source':'formation_joint_input_v5.py','paired':'formation_paired_teaching_v2.py','target':'formation_joint_target_alignment_v2.py'}.items()}
        migration=ROOT/'work/mfm-first-party-full-source-successor-plan-20261013/j6-source-binding-migration-001/actual-001/receipt.json'
        # Pair/target FileRefs deliberately remain unopened by this refusal.
        # They represent no observed native geometry or training admission.
        cases=[{'case_id':r['case_id'],'profile':r['profile'],'variant_key':'current-cut-test',
          'frozen':{'path':'unopened-frozen.json','sha256':'1'*64},
          'accepted':{'path':'unopened-target.json','sha256':'2'*64},
          'pair':{'path':'unopened-pair.json','sha256':'3'*64},
          'material15':{'path':'unopened-material.json','sha256':'4'*64},'max_target_tokens':8192} for r in roster['cases']]
        with TemporaryDirectory(dir=ROOT.parent) as temp:
            outside=Path(temp);(outside/'features').mkdir();(outside/'prepared').mkdir()
            # Only the path exists. Its invalid body must never be read before
            # the missing corpus is refused; it grants no preparation claim.
            (outside/'prepared/absent.json').write_bytes(b'{}')
            connected=HERE/'material-002/connected-roster.json'
            obs=ROOT/'work/mfm-broader-native-geometry-20261013/actual-002/source-observation-receipt.json'
            program={'schema':t.BROADER_PROGRAM_SCHEMA,'cases':cases,'private_original_modules':private,
              'source_migration':ref(migration),'connected_roster':ref(connected),'source_observation':ref(obs)}
            program_path=outside/'program.json';program_path.write_bytes(canonical_json_bytes(program))
            with patch.object(AutoProcessor,'from_pretrained') as processor,patch.object(f,'read_case') as feature, \
                 patch.object(FormationJointLearning,'load_component') as seed:
                with self.assertRaisesRegex(ValueError,'manifest cannot be read'):
                    t.fit_joint_from_program(program_path=program_path,
                      program_file_sha256=sha256(program_path.read_bytes()).hexdigest(),material_root=ROOT,
                      admission_manifest_path=outside/'absent-corpus.json',admission_manifest_sha256='5'*64,
                      owner_authorization_ref='owner-directed-mfm-v16-synthetic',feature_root=outside/'features',
                      feature_roster_sha256='6'*64,prepared_base_dir=outside/'prepared',prepared_receipt_path=outside/'prepared/absent.json',
                      prepared_receipt_file_sha256='7'*64,prepared_receipt_sha256='8'*64,parent_source_receipt_sha256='9'*64,
                      seed_weight_path=outside/'absent.safetensors',seed_config_path=outside/'absent-config.json',
                      seed_manifest_path=outside/'absent-manifest.json',seed_weight_sha256='a'*64,seed_config_sha256='b'*64,
                      seed_manifest_sha256='c'*64,seed_provenance_sha256='d'*64,seed_source_pins={},
                      recipe_path=outside/'absent-recipe.json',recipe_file_sha256='e'*64,output_dir=outside/'output')
                processor.assert_not_called();feature.assert_not_called();seed.assert_not_called()


@unittest.skipUnless(ORIGINAL_INTEGRATION,ORIGINAL_INTEGRATION_REASON)
class BroaderCompleteAuthorProgramTests(unittest.TestCase):
    def test_actual_nine_conversion_and_rights_refuse_before_heavy_loads(self):
        """One complete source join, reused only while its originals stay fixed."""
        import torch
        from transformers import AutoProcessor
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        from scripts.mfm import train_native_joint_formation as t
        from scripts.mfm import export_joint_source_features as f
        corpus_root=ROOT/'work/mfm-broader-corpus-assembly-20261015/actual-001'
        corpus_raw=(corpus_root/'corpus.json').read_bytes()
        fragment=json.loads((corpus_root/'program-metadata.json').read_bytes())
        names={'source':'formation_joint_input_v5.py','paired':'formation_paired_teaching_v2.py',
            'target':'formation_joint_target_alignment_v2.py'}
        program={'schema':t.BROADER_PROGRAM_SCHEMA,'cases':fragment['cases'],
            'connected_roster':fragment['connected_roster'],
            'source_observation':fragment['source_observation'],
            'private_original_modules':{k:{'path':str(PRIVATE/name),
                'sha256':sha256((PRIVATE/name).read_bytes()).hexdigest()} for k,name in names.items()},
            'source_migration':ref(ROOT/'work/mfm-first-party-full-source-successor-plan-20261013/'
                'j6-source-binding-migration-001/actual-001/receipt.json')}
        captured=[];joined=[]
        real_author=t.preflight_admitted_author_cases
        real_roster=t.validate_broader_material_roster
        def author(**kwargs):
            captured.append(kwargs['cases'])
            return real_author(**kwargs)
        def roster(*args,**kwargs):
            result=real_roster(*args,**kwargs);joined.append(result);return result
        with TemporaryDirectory(dir=ROOT.parent) as temp:
            outside=Path(temp);copy=outside/'corpus'
            # Copy only the small corpus payloads; no native source arrays,
            # processor preparation, feature bank or weights are copied.
            shutil.copytree(corpus_root,copy)
            manifest=copy/'corpus.json'
            (outside/'features').mkdir();(outside/'prepared').mkdir()
            prepared_receipt=outside/'prepared/absent.json';prepared_receipt.write_bytes(b'{}')
            program_path=outside/'program.json';program_path.write_bytes(canonical_json_bytes(program))
            program_pin=sha256(program_path.read_bytes()).hexdigest()
            recipe_path=outside/'recipe.json'
            def arguments():
                admission_pin=sha256(manifest.read_bytes()).hexdigest()
                recipe={'schema':t.RECIPE_SCHEMA,'program_sha256':program_pin,
                    'admission_sha256':admission_pin,'feature_roster_sha256':'6'*64,
                    'seed_manifest_sha256':'c'*64,'case_order':[r['case_id'] for r in program['cases']],
                    'update_budget':1,'accumulation_cases':1,'learning_rate':0.0001,
                    'weight_decay':0.0,'adam_betas':[0.9,0.999],'adam_eps':1e-8,
                    'device':'cpu','seed':0,'checkpoint_every_updates':1,
                    'precision':'owned-fp32-source-bf16','term_weights':{'lexical':1,'reference':1,'copy':1},
                    'mass_policy':'equal-world-equal-cut-equal-variant',
                    'update_policy':'frozen-order-case-window-admitted-mass',
                    'runtime':{'torch_version':torch.__version__,'transformers_version':version('transformers')},
                    'trainer_file_sha256':sha256(Path(t.__file__).read_bytes()).hexdigest(),
                    'generation_max_new_tokens':1}
                recipe_path.write_bytes(canonical_json_bytes(recipe))
                return dict(program_path=program_path,program_file_sha256=program_pin,material_root=ROOT,
                    admission_manifest_path=manifest,admission_manifest_sha256=admission_pin,
                    owner_authorization_ref=json.loads(corpus_raw)['authorization_id'],
                    feature_root=outside/'features',feature_roster_sha256='6'*64,
                    prepared_base_dir=outside/'prepared',prepared_receipt_path=prepared_receipt,
                    prepared_receipt_file_sha256='7'*64,prepared_receipt_sha256='8'*64,
                    parent_source_receipt_sha256='9'*64,seed_weight_path=outside/'absent.safetensors',
                    seed_config_path=outside/'absent-config.json',seed_manifest_path=outside/'absent-manifest.json',
                    seed_weight_sha256='a'*64,seed_config_sha256='b'*64,seed_manifest_sha256='c'*64,
                    seed_provenance_sha256='d'*64,seed_source_pins={},recipe_path=recipe_path,
                    recipe_file_sha256=sha256(recipe_path.read_bytes()).hexdigest(),output_dir=outside/'output')
            with ExitStack() as stack:
                blocked=[stack.enter_context(patch.object(obj,name)) for obj,name in (
                    (AutoProcessor,'from_pretrained'),(f,'_read_record'),(f,'_verified_prepared'),
                    (f,'read_case'),(FormationJointLearning,'load_component'),(t,'load_file'))]
                # A fully repinned revoked physical receipt must refuse before
                # per-case source/target/native lowering or any heavy loader.
                changed=json.loads(corpus_raw);rights_name=changed['cases'][0]['sources'][0]['rights_path']
                rights=copy/rights_name;rights_raw=rights.read_bytes();record=json.loads(rights_raw)
                record['revoked']=True;rights.write_bytes(canonical_json_bytes(record))
                rights_pin=sha256(rights.read_bytes()).hexdigest()
                for row in changed['cases']:
                    for source in row['sources']:
                        if source['rights_path']==rights_name:source['rights_sha256']=rights_pin
                manifest.write_bytes(canonical_json_bytes(changed))
                with patch.object(t,'preflight_broader_author_metadata') as metadata:
                    with self.assertRaisesRegex(ValueError,'revoked or unknown'):
                        t.fit_joint_from_program(**arguments())
                    metadata.assert_not_called()
                rights.write_bytes(rights_raw);manifest.write_bytes(corpus_raw)
                # The corpus reader treats converter contents as opaque. This
                # exact native binding must be rejected by the early real join.
                changed=json.loads(corpus_raw)
                conversion_ref=changed['cases'][0]['target_provenance']['conversion']
                conversion=copy/conversion_ref['path'];conversion_raw=conversion.read_bytes()
                record=json.loads(conversion_raw);record['source_native_sha256']='f'*64
                conversion.write_bytes(canonical_json_bytes(record))
                conversion_ref['sha256']=sha256(conversion.read_bytes()).hexdigest()
                manifest.write_bytes(canonical_json_bytes(changed))
                with patch.object(t,'preflight_admitted_author_cases',side_effect=author), \
                     patch.object(t,'validate_broader_material_roster',side_effect=roster):
                    with self.assertRaisesRegex(ValueError,'author conversion does not bind'):
                        t.fit_joint_from_program(**arguments())
                for loader in blocked:loader.assert_not_called()
            self.assertEqual(len(captured),1);self.assertEqual(len(joined),1)
            self.assertEqual(joined[0]['sequences'],[8,14,18,22,28,30,35,36,37])
            self.assertTrue(joined[0]['original_prefix_preserved'])
            cases=captured[0]
            self.assertEqual(len(cases),9)
            self.assertEqual([r['frozen']['source_tokens'] for r in cases[-3:]],[56189,57911,59303])
            self.assertEqual(set(t._case_masses(cases).values()),{1/9})
            # Repair only our copied negative converter. Reuse the already
            # validated unchanged source/native/material joins, then execute
            # the complete real positive author and physical-file audit.
            conversion.write_bytes(conversion_raw);manifest.write_bytes(corpus_raw)
            admission,audit=real_author(admission_manifest_path=manifest,
                admission_manifest_sha256=sha256(corpus_raw).hexdigest(),
                owner_authorization_ref=json.loads(corpus_raw)['authorization_id'],cases=cases)
            self.assertEqual(len(admission.train),9)
            self.assertEqual(len(admission.development),9)
            paths=[r['path'] for r in audit['gradient_inputs']]
            self.assertEqual(len(paths),len(set(paths)))
            self.assertTrue(audit['final_payloads_excluded'])
            self.assertEqual((corpus_root/'corpus.json').read_bytes(),corpus_raw)

if __name__=='__main__':unittest.main()
