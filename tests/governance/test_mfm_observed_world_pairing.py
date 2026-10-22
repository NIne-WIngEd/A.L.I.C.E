"""Observed-world admission using retained full originals, never model tensors."""
from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256

ROOT = Path(os.environ.get('ALICE_MFM_OBSERVED_ORIGINAL_ROOT', str(Path(__file__).resolve().parents[4])))
INTEGRATION = os.environ.get('ALICE_MFM_OBSERVED_ORIGINAL_INTEGRATION') == '1'
WORLD = ROOT/'work/mfm-rill-world-composition-20261022/originals-002'
NATIVE = ROOT/'work/mfm-rill-native-material-20261022/actual-003'
AUTHOR = ROOT/'work/mfm-rill-target-authoring-20261022'
BOUND = AUTHOR/'bound-001'
GEOMETRY = AUTHOR/'native-geometry-001/actual-001'

def load(path): return json.loads(path.read_bytes())
def ref(path): return {'path':path.relative_to(ROOT).as_posix(), 'bytes':path.stat().st_size, 'file_sha256':sha256(path.read_bytes()).hexdigest()}
def seal(value): return {**value, 'record_sha256':canonical_sha256(value)}
def reseal(value): return seal({k:v for k,v in value.items() if k != 'record_sha256'})
def raw(value): return canonical_json_bytes(value)+b'\n'

def source_manifest():
    composed=load(WORLD/'manifest.json')
    cases=[]
    for c in composed['cuts']:
        cases.append({'case_id':c['cut_id'].replace('rill-cut-','rill-training-',1),'cut_id':c['cut_id'],'variant_key':'original',
          'split':'train','lineage':{'world_id':composed['world_id'],
            'generator_family':composed['construction_author']['generator_family'],
            'source_author_family':composed['construction_author']['world_author_family'],
            'parent_case_id':cases[-1]['case_id'] if cases else None},
          'metadata':{k:ref(WORLD/v['path']) for k,v in c['files'].items()}})
    return seal({'schema':'mfm-observed-world-source-manifest-v1',
      'source_contract':'complete-new-world-native-material-v1',
      'composition_contract':'rill-target-free-composed-originals-v1',
      'world':ref(WORLD/'world.json'),'composition_manifest':ref(WORLD/'manifest.json'),
      'composition_result':ref(WORLD/'result.json'),'native_receipt':ref(NATIVE/'receipt.json'),
      'native_runtime':ref(NATIVE/'runtime.json'),
      'native_producer':ref(ROOT/'work/mfm-rill-native-material-20261022/prepare_sources_003.py'),
      'cases':cases,'source_only':True,'independent_gold':False,'training_authorized':False})

def author_manifest(case):
    a=next(r for r in load(BOUND/'READBACK_001.json')['cases'] if r['case_id']==case['case_id'])
    original=load(BOUND/a['case']['path'])
    raw_paths={sha256(p.read_bytes()).hexdigest():p for p in [*AUTHOR.glob('RAW_*.json'),AUTHOR/'REFERENCE_SELECTIONS_001.json']}
    materials={k:ref(BOUND/a[k]['path']) for k in ('case','target','support','choices','source_view','canonical_field_geometry','decision')}
    materials.update({k:ref(raw_paths[original['lineage'][key]]) for k,key in (
      ('raw_semantic','raw_model_author_output_sha256'),('raw_reference','raw_reference_author_output_sha256'),('raw_literal','raw_literal_author_output_sha256'))})
    return seal({'schema':'mfm-observed-world-author-manifest-v1',
      'input_contract':'rill-complete-authored-v1','case_id':case['case_id'],
      'author_producer_version':'codex-assistant-rill-authoring-20261022',
      'conversion_id':'original-rill-v16-observed-binding-v1','materials':materials,
      'native_geometry':ref(GEOMETRY/a['cut_id']/'canonical-target-geometry.json'),
      'native_original':ref(GEOMETRY/a['cut_id']/'canonical-native-original.json'),
      'native_receipt':ref(GEOMETRY/'receipt.json'),
      'native_producer':ref(AUTHOR/'native-geometry-001/materialize.py'),
      'independent_gold':False,'training_authorized':False})

class ObservedWorldContractTests(unittest.TestCase):
    def test_profile_is_a_separate_source_and_target_contract(self):
        try:
            p=importlib.import_module('scripts.mfm.formation_observed_world_paired_teaching')
            t=importlib.import_module('scripts.mfm.formation_observed_world_target_alignment')
        except ImportError:
            self.fail('observed-world material admission is not implemented')
        self.assertEqual(p.INPUT,'mfm-paired-observed-world-input-v1')
        self.assertEqual(p.TARGET,'mfm-paired-observed-world-original-target-v1')
        self.assertEqual(t.PAIR,'mfm-paired-observed-world-target-geometry-v1')

@unittest.skipUnless(INTEGRATION,'set ALICE_MFM_OBSERVED_ORIGINAL_INTEGRATION=1 for exact retained Rill originals')
class ObservedWorldOriginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=importlib.import_module('scripts.mfm.formation_observed_world_paired_teaching')
        cls.t=importlib.import_module('scripts.mfm.formation_observed_world_target_alignment')
        cls.v6=importlib.import_module('scripts.mfm.formation_joint_input_v6')
        name='scripts.mfm.formation_joint_input_v5'
        path=ROOT/'work/mfm-joint-original-inventory-roundtrip-design-20261011/material-read-export-construction-001/dependency-tree-001/scripts/mfm/formation_joint_input_v5.py'
        spec=importlib.util.spec_from_file_location(name,path)
        cls.j5=importlib.util.module_from_spec(spec);sys.modules[name]=cls.j5;spec.loader.exec_module(cls.j5)
        cls.manifest=source_manifest()

    def freeze(self,index=0,manifest=None):
        m=self.manifest if manifest is None else manifest; b=raw(m)
        return self.p.freeze_input(self.j5,self.v6,material_root=ROOT,
          manifest_reader=lambda:b,expected_manifest_file_sha256=sha256(b).hexdigest(),case_id=m['cases'][index]['case_id'])

    def convert(self,frozen,manifest=None,reader=None):
        case=next(c for c in self.manifest['cases'] if c['case_id']==frozen['split_row']['case_id'])
        m=author_manifest(case) if manifest is None else manifest;b=raw(m)
        return self.p.convert_original_target(self.j5,self.v6,frozen,material_root=ROOT,
          expected_input_sha256=frozen['record_sha256'],author_manifest_reader=reader or (lambda:b),
          expected_author_manifest_file_sha256=sha256(b).hexdigest())

    def test_all_nine_original_sources_targets_and_native_geometry(self):
        total_labels=total_references=total_literals=0
        for i,c in enumerate(self.manifest['cases']):
            with self.subTest(case_id=c['case_id']):
                frozen=self.freeze(i); target=self.convert(frozen); pair=self.t.bind_observed_target(frozen,target)
                self.assertEqual(frozen['whole_world_record_count'],51)
                self.assertFalse(frozen['target_accessed'])
                self.assertFalse(target['independent_gold'])
                self.assertFalse(pair['processor_reexecuted_for_target'])
                self.assertEqual(pair['labels'][-1],frozen['processor_binding']['tokenizer_wrapper']['eos_token_id'])
                self.t.validate_pair(pair,frozen,target,expected_pair_sha256=pair['record_sha256'])
                total_labels+=len(pair['labels'])
                total_references+=sum(r['responsibility']=='external_selection' and r['annotation_status']=='authored' for r in pair['fields'])
                total_literals+=len(pair['literal_segments'])
        self.assertEqual((total_labels,total_references,total_literals),(17052,329,9))

    def test_changed_cut_and_relabelled_family_refused(self):
        for field in ('cut_id','lineage'):
            changed=deepcopy(self.manifest)
            if field=='cut_id': changed['cases'][0]['cut_id']=changed['cases'][1]['cut_id']
            else: changed['cases'][0]['lineage']['generator_family']='independent-new-family'
            changed=reseal(changed)
            with self.assertRaises(ValueError):self.freeze(manifest=changed)

    def test_author_contract_split_and_exact_candidate_path(self):
        # This isolated author-consumer control reuses retained originals. The
        # full source admission and all-nine path are exercised separately.
        case=self.manifest['cases'][0];m=author_manifest(case)
        observed=load(NATIVE/'receipt.json')['source_files'][0]
        source=load(NATIVE/observed['source']['path'])
        composed=load(WORLD/'manifest.json')
        self.p._physical_boundaries(ROOT,self.manifest,composed,case,source)
        withheld=deepcopy(composed)
        withheld['cuts'][0]['withheld_source_ids']=[source['caller']['opened_sources'][0]['ref_id']]
        with self.assertRaises(ValueError):
            self.p._physical_boundaries(ROOT,self.manifest,withheld,case,source)
        frozen={'source':source,'split_row':{'case_id':case['case_id'],
          'split':case['split'],'lineage':deepcopy(case['lineage'])},
          'whole_world_sha256':load(WORLD/'world.json')['world_sha256']}
        materials={k:load(ROOT/v['path']) for k,v in m['materials'].items()}
        self.p._author_material(frozen,m,materials)
        wrong=deepcopy(materials)
        row=wrong['choices']['reference_choices'][0]
        candidate=source['meaning']['candidate_inventory'][row['candidate_ordinal']]
        row['candidate_path']=next(path for path in candidate['eligible_paths'] if path!=row['candidate_path'])
        with self.assertRaises(ValueError):self.p._author_material(frozen,m,wrong)
        # The source/author contract permits development. Its codec call must
        # receive that actual split explicitly, without promoting independence.
        development=deepcopy(materials);development['case']['split']='development'
        dev_frozen=deepcopy(frozen);dev_frozen['split_row']['split']='development'
        example,_=self.p._author_material(dev_frozen,m,development)
        self.assertEqual(example.split,'development')

    def test_final_and_changed_subject_time_refuse_before_author_reader(self):
        frozen=self.freeze(); opened=[]
        for change in ('final','subject','time'):
            bad=deepcopy(frozen)
            if change=='final':
                manifest=deepcopy(self.manifest)
                for c in manifest['cases']:c['split']='final'
                bad=self.freeze(manifest=reseal(manifest))
            elif change=='subject':bad['source']['caller']['context']['owner_ref']='another-owner'
            else:bad['source']['meaning']['episode_snapshot']['historical_prefix_proof']['cut']['sequence']+=1
            bad=reseal(bad)
            with self.assertRaises(ValueError):self.convert(bad,reader=lambda:opened.append('author'))
        self.assertEqual(opened,[])

    def test_changed_author_pin_and_native_geometry_refused(self):
        frozen=self.freeze(); author=author_manifest(self.manifest['cases'][0])
        bad=deepcopy(author);bad['materials']['raw_semantic']['file_sha256']='0'*64
        with self.assertRaises(ValueError):self.convert(frozen,manifest=reseal(bad))
        target=self.convert(frozen);pair=self.t.bind_observed_target(frozen,target)
        bad=deepcopy(pair);bad['tokens'][0]['canonical_chars']=[1,2];bad=reseal(bad)
        with self.assertRaises(ValueError):self.t.validate_pair(bad,frozen,target,expected_pair_sha256=bad['record_sha256'])
        bad=deepcopy(target);bad['source_cache']['source_tokens']=1;bad=reseal(bad)
        with self.assertRaises(ValueError):self.t.bind_observed_target(frozen,bad)

if __name__=='__main__':unittest.main()
