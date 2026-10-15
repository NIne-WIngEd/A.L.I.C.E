"""Broader profile checks on physical world/cut originals; no model fixtures."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cognitive_kernel.canonical import canonical_json_bytes,canonical_sha256

ROOT=Path(os.environ.get('ALICE_MFM_BROADER_ORIGINAL_ROOT',str(Path(__file__).resolve().parents[4])))
ORIGINAL_INTEGRATION=os.environ.get('ALICE_MFM_BROADER_ORIGINAL_INTEGRATION')=='1'
ORIGINAL_INTEGRATION_REASON='set ALICE_MFM_BROADER_ORIGINAL_INTEGRATION=1 for staged exact-original integration'
OLD=ROOT/'work/mfm-first-party-longitudinal-corpus-design-20261011/construction-001/public-connected-originals-005/originals'
NEW=ROOT/'work/mfm-broader-foundation-material-20261013/world-source-successor-002'
NAMES=('initial-state','question-open','actor-report','contrary-observation',
 'supported-revision','withdrawal-intent','shared-session','assistant-revision','schedule-revocation')
SEQUENCES=(8,14,18,22,28,30,35,36,37)

def ref(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path.read_bytes()).hexdigest()}
def reseal(row):return {**{k:v for k,v in row.items() if k!='record_sha256'},'record_sha256':canonical_sha256({k:v for k,v in row.items() if k!='record_sha256'})}
def actual_roster():
    rows=[{'case_id':'j6-reading-room-'+name,'cut_id':name,
      'profile':'j6-firstparty-successor' if i<6 else 'broader-firstparty-successor',
      'cut':ref((OLD if i<6 else NEW)/name/'cut.json')} for i,name in enumerate(NAMES)]
    return reseal({'schema':'mfm-broader-connected-world-roster-v1',
      'world_id':'reading-room-rain-gauge-world-001','legacy_world':ref(OLD/'world.json'),
      'successor_world':ref(NEW/'world.json'),'cases':rows,'construction_only':True,
      'legacy_view':ref(OLD/'withdrawal-intent/view.json'),
      'successor_view':ref(NEW/'schedule-revocation/view.json'),
      'training_authorized':False,'capacity_admitted':False})

@unittest.skipUnless(ORIGINAL_INTEGRATION,ORIGINAL_INTEGRATION_REASON)
class BroaderConnectedOriginalTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('scripts.mfm.formation_broader_paired_teaching')
        except ImportError:self.fail('broader source-first connected-world validator is not implemented')

    def test_actual_old30_is_exact_prefix_of_new37_with_nine_chronological_cuts(self):
        broad=self.module();roster=actual_roster()
        rows=[{'case_id':r['case_id'],'profile':r['profile']} for r in roster['cases']]
        result=broad.validate_connected_roster(ROOT,roster,
          expected_roster_sha256=roster['record_sha256'],program_cases=rows)
        self.assertEqual(result['sequences'],list(SEQUENCES))
        self.assertEqual(result['legacy_world_sha256'],'a3c0e4e17fa61bcf961de137aaeeab4d56f3df88f7cb779dc24cb14d2f688c3c')
        self.assertEqual(result['successor_world_sha256'],'4ca3ce4846491ba7b199d7f131977c62548d719433fa104d16722bcbba433895')
        self.assertEqual(result['world_id'],'reading-room-rain-gauge-world-001')
        self.assertTrue(result['original_prefix_preserved'])
        self.assertFalse(result['training_authorized'])

    def test_actual_cut_order_and_legacy_profile_cannot_be_relabelled(self):
        broad=self.module();roster=actual_roster()
        rows=[{'case_id':r['case_id'],'profile':r['profile']} for r in roster['cases']]
        for changed in (rows[:5]+rows[6:7]+rows[5:6]+rows[7:],
                        [{**r,'profile':'broader-firstparty-successor'} if i==0 else r for i,r in enumerate(rows)]):
            with self.assertRaises(ValueError):broad.validate_connected_roster(ROOT,roster,
              expected_roster_sha256=roster['record_sha256'],program_cases=changed)

    def test_resealed_changed_prefix_or_cut_body_cannot_join_actual_world(self):
        broad=self.module();roster=actual_roster()
        rows=[{'case_id':r['case_id'],'profile':r['profile']} for r in roster['cases']]
        scratch=ROOT/'work/mfm-broader-joint-learning-20261013'
        with TemporaryDirectory(dir=scratch) as temp:
            path=Path(temp)/'changed-world.json';world=json.loads((NEW/'world.json').read_bytes())
            world['branches'][0]['records'][0]['artifact']['file_sha256']='0'*64
            world['world_sha256']=canonical_sha256({k:v for k,v in world.items() if k!='world_sha256'})
            path.write_bytes(canonical_json_bytes(world));bad=deepcopy(roster);bad['successor_world']=ref(path);bad=reseal(bad)
            with self.assertRaises(ValueError):broad.validate_connected_roster(ROOT,bad,
              expected_roster_sha256=bad['record_sha256'],program_cases=rows)

    def test_resealed_extension_descriptor_with_stale_record_pin_is_refused(self):
        broad=self.module();roster=actual_roster()
        rows=[{'case_id':r['case_id'],'profile':r['profile']} for r in roster['cases']]
        scratch=ROOT/'work/mfm-broader-joint-learning-20261013'
        with TemporaryDirectory(dir=scratch) as temp:
            path=Path(temp)/'changed-extension.json';world=json.loads((NEW/'world.json').read_bytes())
            # No original prefix or selected cut head changes. The broken row31
            # still needs its own actual bytes/record seal/previous chain proof.
            world['branches'][0]['records'][30]['artifact']['file_sha256']='0'*64
            world['world_sha256']=canonical_sha256({k:v for k,v in world.items() if k!='world_sha256'})
            path.write_bytes(canonical_json_bytes(world));bad=deepcopy(roster)
            bad['successor_world']=ref(path);bad=reseal(bad)
            with self.assertRaises(ValueError):broad.validate_connected_roster(ROOT,bad,
              expected_roster_sha256=bad['record_sha256'],program_cases=rows)
            cut=json.loads((NEW/'shared-session/cut.json').read_bytes());cut['head_sha256']='0'*64;cut=reseal(cut)
            path=Path(temp)/'changed-cut.json';path.write_bytes(canonical_json_bytes(cut));bad=deepcopy(roster);bad['cases'][6]['cut']=ref(path);bad=reseal(bad)
            with self.assertRaises(ValueError):broad.validate_connected_roster(ROOT,bad,
              expected_roster_sha256=bad['record_sha256'],program_cases=rows)


@unittest.skipUnless(ORIGINAL_INTEGRATION,ORIGINAL_INTEGRATION_REASON)
class BroaderActualSourceTests(unittest.TestCase):
    def test_actual_complete_target_geometry_keeps_authored_references_and_copies(self):
        from scripts.mfm import formation_broader_paired_teaching as b,formation_broader_target_alignment as t
        from scripts.mfm import formation_joint_input_v6 as v6
        import importlib.util,sys
        name='scripts.mfm.formation_joint_input_v5'
        spec=importlib.util.spec_from_file_location(name,ROOT/'work/mfm-joint-original-inventory-roundtrip-design-20261011/material-read-export-construction-001/dependency-tree-001/scripts/mfm/formation_joint_input_v5.py')
        j5=importlib.util.module_from_spec(spec);sys.modules[name]=j5;spec.loader.exec_module(j5)
        native=ROOT/'work/mfm-broader-native-geometry-20261013/actual-002'
        obs=(native/'source-observation-receipt.json').read_bytes();entry=json.loads(obs)['source_files'][0]
        reader=lambda descriptor:(native/descriptor['path']).read_bytes()
        frozen=b.freeze_input(j5,v6,lambda:(native/entry['path']).read_bytes(),lambda:obs,
          expected_source_file_sha256=entry['file_sha256'],expected_observation_file_sha256=sha256(obs).hexdigest(),
          native_reader=reader,whole_world_reader=lambda:(NEW/'world.json').read_bytes(),expected_whole_world_file_sha256=sha256((NEW/'world.json').read_bytes()).hexdigest(),split_row={'case_id':'j6-reading-room-shared-session','split':'train','lineage':{'world':'reading-room-rain-gauge-world-001'}})
        author=(native/'shared-session/complete-author-choices.json').read_bytes()
        geometry=(native/'shared-session/canonical-target-native-geometry.json').read_bytes()
        target=b.convert_original_target(j5,v6,frozen,expected_input_sha256=frozen['record_sha256'],
          expected_source_file_sha256=entry['file_sha256'],observation_reader=lambda:obs,native_reader=reader,
          original_root=ROOT,authoring_reader=lambda:author,expected_authoring_file_sha256=sha256(author).hexdigest(),
          geometry_reader=lambda:geometry,expected_geometry_file_sha256=sha256(geometry).hexdigest())
        pair=t.bind_observed_target(frozen,target)
        self.assertEqual(len(pair['labels']),1620)
        self.assertEqual(sum(r['responsibility']=='external_selection' and r['annotation_status']=='authored' for r in pair['fields']),29)
        self.assertEqual(len(pair['literal_segments']),7)
        self.assertFalse(pair['processor_reexecuted_for_target'])
        t.validate_pair(pair,frozen,target,expected_pair_sha256=pair['record_sha256'])
        bad=deepcopy(pair);bad['labels'][0]+=1;bad=reseal(bad)
        with self.assertRaises(ValueError):t.validate_pair(bad,frozen,target,expected_pair_sha256=bad['record_sha256'])
        bad=deepcopy(target);bad['unknown_dimensions']=['episode'];bad=reseal(bad)
        with self.assertRaises(ValueError):t.bind_observed_target(frozen,bad)

    def test_saved_complete_native_source_is_bound_before_target_access(self):
        from scripts.mfm import formation_broader_paired_teaching as b
        from scripts.mfm import formation_joint_input_v6 as v6
        import importlib.util,sys
        name='scripts.mfm.formation_joint_input_v5'
        spec=importlib.util.spec_from_file_location(name,ROOT/'work/mfm-joint-original-inventory-roundtrip-design-20261011/material-read-export-construction-001/dependency-tree-001/scripts/mfm/formation_joint_input_v5.py')
        j5=importlib.util.module_from_spec(spec);sys.modules[name]=j5;spec.loader.exec_module(j5)
        native=ROOT/'work/mfm-broader-native-geometry-20261013/actual-002'
        observation=(native/'source-observation-receipt.json').read_bytes()
        obs=json.loads(observation);entry=obs['source_files'][0];source=(native/entry['path']).read_bytes()
        frozen=b.freeze_input(j5,v6,lambda:source,lambda:observation,
          expected_source_file_sha256=entry['file_sha256'],
          expected_observation_file_sha256=sha256(observation).hexdigest(),
          native_reader=lambda descriptor:(native/descriptor['path']).read_bytes(),
          whole_world_reader=lambda:(NEW/'world.json').read_bytes(),expected_whole_world_file_sha256=sha256((NEW/'world.json').read_bytes()).hexdigest(),
          split_row={'case_id':'j6-reading-room-shared-session','split':'train','lineage':{'world':'reading-room-rain-gauge-world-001'}})
        self.assertEqual(frozen['source_tokens'],56189)
        self.assertEqual(frozen['proposed_max_source_tokens'],59303)
        self.assertFalse(frozen['target_accessed'])
        with self.assertRaises(ValueError):b.freeze_input(j5,v6,lambda:source,lambda:observation,
          expected_source_file_sha256=entry['file_sha256'],expected_observation_file_sha256=sha256(observation).hexdigest(),
          native_reader=lambda descriptor:b'{}',whole_world_reader=lambda:(NEW/'world.json').read_bytes(),expected_whole_world_file_sha256=sha256((NEW/'world.json').read_bytes()).hexdigest(),split_row=frozen['split_row'])

if __name__=='__main__':unittest.main()
