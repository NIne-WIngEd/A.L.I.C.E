"""Observed profile dispatch and exact saved Rill choice lowering, without a model.

Small records exercise only the dispatch/binding boundary. Opt-in originals
compare all nine complete decisions to the previously observed native outputs.
Neither path admits a corpus or establishes learned behavior.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel import formation_decision_v16 as local
from cognitive_kernel import formation_original_copy_decision as copies
from scripts.mfm import formation_reference_target_alignment as reference
from scripts.mfm import formation_original_copy_target_alignment as copy_target
from scripts.mfm import formation_observed_world_target_alignment as observed


def reseal(row):
    core={k:v for k,v in row.items() if k!='record_sha256'}
    return {**core,'record_sha256':canonical_sha256(core)}


def nested_binding_fixture():
    """Explicit binding-only fixture; not a valid admitted world or native source."""
    caller=reseal({'schema':'binding-control-caller'})
    joint=reseal({'schema':'mfm-joint-caller-input-alignment-v6',
      'cache':{'processed_view_sha256':'a'*64,'provenance_sha256':'b'*64},
      'processor_binding':{'fixture':True},'input_ids':[1,2],
      'attention_mask':[1,1],'mm_token_type_ids':[0,0]})
    batch={k:[joint[k]] for k in ('input_ids','attention_mask','mm_token_type_ids')}
    source=reseal({'schema':'mfm-full-source-text-construction-freeze-v6',
      'caller':caller,'alignment':joint,'native_batch':batch})
    frozen=reseal({'schema':observed.paired.INPUT,'source':source,
      'source_file_sha256':sha256(canonical_json_bytes(source)+b'\n').hexdigest(),
      'source_cache':joint['cache'],'processor_binding':joint['processor_binding'],'native_batch':batch})
    choice=reseal({'schema':observed.CHOICES,'input_freeze_sha256':frozen['record_sha256'],
      'target_record_sha256':'c'*64,'author_id':'explicit-fixture-author',
      'author_provenance_file_sha256':'d'*64,'field_annotations':[],
      'literal_declarations':[],'whole_literal_inventory_complete':True})
    raw=canonical_json_bytes(choice)+b'\n'
    pair=reseal({'schema':observed.PAIR,'original_choice_hex':raw.hex(),
      'choice_file_sha256':sha256(raw).hexdigest(),'choice_packet':choice,
      'target_record_sha256':choice['target_record_sha256'],
      'author_provenance_file_sha256':choice['author_provenance_file_sha256'],
      'source_file_sha256':frozen['source_file_sha256'],'source_cache':frozen['source_cache'],
      'native_batch':batch,'processor_reexecuted_for_target':False,
      'native_target_observation_consumed':True,'original_author_contract':observed.paired.AUTHOR_CONTRACT})
    return frozen,pair,choice


class ObservedWorldDispatchTests(unittest.TestCase):
    def test_real_observed_schema_selects_nested_v6_source_without_relabelling(self):
        frozen,_,_=nested_binding_fixture()
        self.assertEqual(reference.PROFILES.get(observed.paired.INPUT),
          ('mfm-joint-caller-input-alignment-v6',observed.PAIR))
        before=canonical_json_bytes(frozen)
        self.assertEqual(reference.source_parts(frozen),
          (frozen['source']['caller'],frozen['source']['alignment']))
        self.assertEqual(canonical_json_bytes(frozen),before)

    def test_nested_source_file_batch_cache_and_processor_are_bound(self):
        frozen,_,_=nested_binding_fixture()
        self.assertIn(observed.paired.INPUT,reference.PROFILES)
        for field,value in (('source_file_sha256','0'*64),
          ('native_batch',{'input_ids':[[9]]}),('source_cache',{}),('processor_binding',{})):
            bad=deepcopy(frozen);bad[field]=value;bad=reseal(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):reference.source_parts(bad)

    def test_external_choice_envelope_has_exact_profile_fields_and_ancestry(self):
        reader=getattr(reference,'observed_choices',None)
        self.assertTrue(callable(reader),'observed choice dispatch is missing')
        frozen,pair,choice=nested_binding_fixture()
        self.assertEqual(reader(pair,frozen),choice)
        for field,value in (('schema',reference.j6_target.PAIR),
          ('author_provenance_file_sha256','0'*64),('source_file_sha256','0'*64),
          ('target_record_sha256','0'*64),('source_cache',{}),('native_batch',{}),
          ('processor_reexecuted_for_target',True),('native_target_observation_consumed',False),
          ('original_author_contract','historical-author')):
            bad=deepcopy(pair);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):reader(reseal(bad),frozen)
        for field,value in (('schema',reference.j6_target.CHOICES),
          ('input_freeze_sha256','0'*64),('whole_literal_inventory_complete',False),
          ('unexpected_author_choice',True)):
            changed=reseal({**choice,field:value});raw=canonical_json_bytes(changed)+b'\n'
            bad={**pair,'choice_packet':changed,'original_choice_hex':raw.hex(),
                 'choice_file_sha256':sha256(raw).hexdigest()}
            with self.subTest(choice_field=field),self.assertRaises(ValueError):reader(reseal(bad),frozen)
        bad=deepcopy(pair);bad['choice_packet']['field_annotations']=[{'inferred':True}]
        with self.assertRaises(ValueError):reader(reseal(bad),frozen)
        bad=deepcopy(frozen);bad['schema']=reference.broader_target.paired.INPUT
        with self.assertRaises(ValueError):reader(pair,reseal(bad))

    def test_historical_nested_source_profiles_keep_their_existing_dispatch(self):
        frozen,_,_=nested_binding_fixture()
        for module in (reference.j6_target,reference.broader_target):
            old=reseal({**frozen,'schema':module.paired.INPUT})
            self.assertEqual(reference.PROFILES[module.paired.INPUT],
              ('mfm-joint-caller-input-alignment-v6',module.PAIR))
            self.assertEqual(reference.source_parts(old),
              (frozen['source']['caller'],frozen['source']['alignment']))


ROOT=Path(os.environ.get('ALICE_MFM_OBSERVED_ORIGINAL_ROOT',str(Path(__file__).resolve().parents[4])))
INTEGRATION=os.environ.get('ALICE_MFM_OBSERVED_ORIGINAL_INTEGRATION')=='1'
PAIRS=ROOT/'work/mfm-rill-observed-pair-admission-20261022/actual-001'
NATIVE=ROOT/'work/mfm-rill-target-authoring-20261022/native-geometry-001/actual-001'


def read_ref(root,ref):
    path=(root/ref['path']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('original leaves its root')
    raw=path.read_bytes()
    if len(raw)!=ref['bytes'] or sha256(raw).hexdigest()!=ref['file_sha256']:
        raise ValueError('original physical FILE differs')
    return json.loads(raw)


@unittest.skipUnless(INTEGRATION,'set ALICE_MFM_OBSERVED_ORIGINAL_INTEGRATION=1 for retained nine-case originals')
class ObservedWorldOriginalLoweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=json.loads((PAIRS/'result.json').read_bytes())
        observed.paired._record(cls.result,'mfm-observed-world-pair-construction-result-v1')
        if cls.result['status']!='completed' or cls.result['case_count']!=9:
            raise ValueError('completed full observed pair construction required')
        cls.native=json.loads((NATIVE/'receipt.json').read_bytes())
        observed.paired._record(cls.native,observed.paired.NATIVE_TARGET_RECEIPT)
        if cls.native['status']!='completed':raise ValueError('completed native target originals required')
        cls.native_files={r['path']:r for r in cls.native['files']}

    def material(self,row):
        frozen=read_ref(PAIRS,row['source_input']);pair=read_ref(PAIRS,row['pair'])
        target=read_ref(PAIRS,row['target'])
        example,_=observed.paired.require_complete_labels(frozen,target)
        self.assertEqual(pair['target_record_sha256'],target['record_sha256'])
        caller,joint=reference.source_parts(frozen)
        inventory=read_ref(NATIVE,self.native_files[row['cut_id']+'/source-copy-inventory.json'])
        self.assertEqual(inventory['joint_alignment_sha256'],joint['record_sha256'])
        self.assertEqual(inventory['reference_inventory']['binding']['caller_sha256'],caller['record_sha256'])
        return frozen,pair,example,inventory

    def lower(self,frozen,pair,example,inventory):
        return copy_target.original_copy_decision_view(example,pair,frozen,
          pair_sha256=pair['record_sha256'],frozen_input_sha256=frozen['record_sha256'],inventory=inventory)

    def test_all_nine_exact_native_choice_goldens_and_semantic_roundtrip(self):
        counts=[0,0,0]
        for row in self.result['cases']:
            with self.subTest(case_id=row['case_id']):
                self.assertEqual(row['profile'],'observed-world-firstparty')
                frozen,pair,example,inventory=self.material(row)
                before=tuple(canonical_sha256(v) for v in (frozen,pair,inventory))
                view=self.lower(frozen,pair,example,inventory)
                copy_target._view(view,view['record_sha256'])
                golden=read_ref(NATIVE,self.native_files[row['cut_id']+'/choice-view.json'])
                for key in ('decision','binding','copy_binding','processor_binding','reference_leaves',
                            'authored_ordinals','copy_leaves','authored_copy_tuples'):
                    self.assertEqual(view[key],golden[key],key)
                self.assertEqual(view['source_profile'],{'input_schema':observed.paired.INPUT,
                  'joint_schema':'mfm-joint-caller-input-alignment-v6','pair_schema':observed.PAIR})
                caller,joint=reference.source_parts(frozen)
                chosen=[{'pointer':r['pointer'],**r['alternatives'][0]} for r in view['authored_copy_tuples']['rows']]
                compiled=copies.compile_original_copy_decision(example.context,example.opened_sources,view['decision'],
                  {'binding':view['copy_binding'],'rows':chosen},
                  {'binding':view['binding'],'rows':view['authored_ordinals']},
                  caller=caller,caller_sha256=caller['record_sha256'],joint_alignment=joint,
                  joint_alignment_sha256=joint['record_sha256'],expected_binding=view['copy_binding'],
                  artifact_sha256='a'*64,inference_run_id='observed-world-original-lowering-control')
                self.assertEqual(local.decision_from_bundle(compiled['bundle']),local.decision_record(example))
                self.assertEqual(tuple(canonical_sha256(v) for v in (frozen,pair,inventory)),before)
                self.assertFalse(view['training_authorized']);self.assertFalse(view['authority_granted'])
                counts=[counts[0]+len(view['decision']['proposals']),
                        counts[1]+len(view['authored_ordinals']),counts[2]+len(view['copy_leaves'])]
        self.assertEqual(counts,[41,329,9])

    def test_actual_incompatible_profiles_resealed_field_and_copy_changes_refuse(self):
        frozen,pair,example,inventory=self.material(self.result['cases'][0])
        for mode in ('historical_pair','choice_schema','choice_bytes','field_ordinal','literal_occurrence','native_flag'):
            bad=deepcopy(pair)
            if mode=='historical_pair':bad['schema']=reference.broader_target.PAIR
            elif mode=='choice_schema':
                choice=reseal({**bad['choice_packet'],'schema':reference.broader_target.CHOICES})
                raw=canonical_json_bytes(choice)+b'\n';bad.update(choice_packet=choice,
                  original_choice_hex=raw.hex(),choice_file_sha256=sha256(raw).hexdigest())
            elif mode=='choice_bytes':bad['original_choice_hex']=b'{}\n'.hex()
            elif mode=='field_ordinal':
                field=next(r for r in bad['fields'] if 'selected_candidate_ordinal' in r)
                field['selected_candidate_ordinal']+=1
            elif mode=='literal_occurrence':bad['literal_segments'][0]['selected_occurrences'][0]['original_utf8'][0]+=1
            else:bad['native_target_observation_consumed']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.lower(frozen,reseal(bad),example,inventory)


@unittest.skipUnless(INTEGRATION,'set ALICE_MFM_OBSERVED_ORIGINAL_INTEGRATION=1 for retained historical originals')
class HistoricalOriginalLoweringTests(unittest.TestCase):
    def test_j6_and_broader_lowered_views_remain_byte_identical(self):
        old=ROOT/'work/mfm-first-party-full-source-successor-plan-20261013'
        broad=ROOT/'work/mfm-broader-joint-learning-20261013/metadata-successor-002/shared-session'
        j6=old/'j6-material-native-pair-binding-001/actual-001/initial-state'
        controls=(
          (reference.j6_target.paired,j6/'paired-input.json',j6/'accepted-target.json',
           j6/'native-target-pair.json',old/'j6-native-mixed-wire-geometry-observation-001/actual-001/initial-state/lowered-view.json'),
          (reference.broader_target.paired,broad/'broader-paired-input.json',
           broad/'broader-accepted-target.json',broad/'broader-native-target-pair.json',broad/'lowered-view.json'))
        for paired,source_path,target_path,pair_path,golden_path in controls:
            with self.subTest(profile=paired.INPUT):
                original_bytes={p:p.read_bytes() for p in (source_path,target_path,pair_path,golden_path)}
                frozen,target,pair,golden=(json.loads(original_bytes[p]) for p in
                  (source_path,target_path,pair_path,golden_path))
                example,_=paired.require_complete_labels(frozen,target)
                caller,joint=reference.source_parts(frozen)
                binding={'caller_sha256':caller['record_sha256'],
                  'candidate_inventory_sha256':joint['candidate_inventory_sha256'],**joint['cache']}
                inventory=copies.bind_original_copy_inventory(caller,joint,
                  caller_sha256=caller['record_sha256'],joint_alignment_sha256=joint['record_sha256'],
                  expected_binding=binding)
                view=copy_target.original_copy_decision_view(example,pair,frozen,
                  pair_sha256=pair['record_sha256'],frozen_input_sha256=frozen['record_sha256'],inventory=inventory)
                self.assertEqual(canonical_json_bytes(view),canonical_json_bytes(golden))
                self.assertEqual(original_bytes,{p:p.read_bytes() for p in original_bytes})


if __name__=='__main__':unittest.main()
