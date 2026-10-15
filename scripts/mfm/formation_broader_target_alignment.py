"""Bind separately observed complete targets; never execute a processor here."""
from . import formation_broader_paired_teaching as paired
from . import formation_joint_target_alignment_v6 as original
from . import formation_target_alignment as geometry_api
from cognitive_kernel.canonical import canonical_sha256
from hashlib import sha256
from copy import deepcopy

PAIR='mfm-paired-broader-full-source-joint-target-geometry-v1'
CHOICES='mfm-broader-external-field-and-literal-choices-v1'
field_rows=original.field_rows
literal_rows=original.literal_rows

def _choice_packet(raw,expected,frozen,target):
    choice=paired.pinned(raw,expected,CHOICES)
    paired.need(set(choice)=={'schema','input_freeze_sha256','target_record_sha256','author_id',
      'author_provenance_file_sha256','field_annotations','literal_declarations',
      'whole_literal_inventory_complete','record_sha256'} and
      choice['input_freeze_sha256']==frozen['record_sha256'] and
      choice['target_record_sha256']==target['record_sha256'] and
      choice['author_provenance_file_sha256']==target['annotation_file_sha256'] and
      choice['whole_literal_inventory_complete'] is True,
      'broader complete authored choice binding differs')
    return choice

def _observed(frozen,target):
    _,output=paired.require_complete_labels(frozen,target)
    author=paired.pinned(bytes.fromhex(target['raw_annotation_hex']),
      target['annotation_file_sha256'],paired.AUTHORING)
    raw=bytes.fromhex(target['raw_native_geometry_hex'])
    component=paired.pinned(raw,target['native_geometry_file_sha256'],
      'mfm-broader-canonical-target-native-geometry-v1')
    geometry=geometry_api.canonical_field_geometry(output,
      expected_target_sha256=target['canonical_output_sha256'])
    tokens=component['tokens'];native=component['native_return']
    paired.need(component['source']['file_sha256']==frozen['source_file_sha256'] and
      component['complete_author_choices']['file_sha256']==target['annotation_file_sha256'] and
      component['author_originals']==target['original_material_descriptors'] and
      component['processor_binding']==frozen['processor_binding'] and component['geometry']==geometry and
      component['case_id']==frozen['split_row']['case_id'] and component['cut_id']==frozen['cut_id'] and
      component['capacity_admitted'] is False and component['model_forward'] is False and
      component['old_six_pairing_ancestry_claimed'] is False and component['training_corpus_admitted'] is False and
      set(native)=={'input_ids','attention_mask','offset_mapping','special_tokens_mask'},
      'broader actual observed target/source/authority geometry differs')
    ids,mask,offsets,special=(native[k] for k in
      ('input_ids','attention_mask','offset_mapping','special_tokens_mask'))
    paired.need(all(type(row) is list for row in (ids,mask,offsets,special)) and
      len(ids)==len(mask)==len(offsets)==len(special) and
      all(type(i) is int for row in (ids,mask,special) for i in row) and
      all(type(span) is list and len(span)==2 and all(type(i) is int for i in span) for span in offsets),
      'broader actual target native integer shape differs')
    rebuilt=geometry_api._token_rows(geometry,ids,mask,offsets,special)
    inputs,labels,maximum=(component[k] for k in ('decoder_inputs','labels','max_target_tokens'))
    wrapper=frozen['processor_binding']['tokenizer_wrapper']
    paired.need(rebuilt==tokens and type(maximum) is int and maximum>0 and
      type(inputs) is list and type(labels) is list and
      all(type(i) is int and i>=0 for row in (inputs,labels) for i in row) and
      len(inputs)==len(labels)==len(ids)+1<=maximum and inputs[1:]==labels[:-1]==ids and
      inputs[0]==wrapper['bos_token_id'] and labels[-1]==wrapper['eos_token_id'],
      'broader observed complete target shift/BOS/EOS differs')
    fields=field_rows(geometry,frozen,author['field_annotations'])
    literals=literal_rows(geometry,tokens,frozen,author['literal_copy_segments_declared'])
    paired.need(component['fields']==fields and component['literal_segments']==literals,
      'broader observed target field/whole-copy originals differ')
    choice=paired.seal({'schema':CHOICES,'input_freeze_sha256':frozen['record_sha256'],
      'target_record_sha256':target['record_sha256'],'author_id':author['author_id'],
      'author_provenance_file_sha256':target['annotation_file_sha256'],
      'field_annotations':deepcopy(author['field_annotations']),
      'literal_declarations':deepcopy(author['literal_copy_segments_declared']),
      'whole_literal_inventory_complete':True})
    choice_raw=paired.canonical_json_bytes(choice)+b'\n'
    core=original._core(frozen,target,sha256(choice_raw).hexdigest(),choice_raw,choice,
      geometry,inputs,labels,tokens,maximum,fields,literals)
    core.update(schema=PAIR,processor_reexecuted_for_target=False,
      observed_native_target_file_sha256=target['native_geometry_file_sha256'],
      original_native_target_hex=raw.hex(),native_target_observation_consumed=True)
    return core

def bind_observed_target(frozen,target):
    """Consume exact separately captured native fields; no tokenizer/model."""
    return paired.seal(_observed(frozen,target))

def validate_pair(pair,frozen,target,*,expected_pair_sha256):
    paired.need(pair.get('schema')==PAIR and pair.get('record_sha256')==expected_pair_sha256 and
      canonical_sha256({k:v for k,v in pair.items() if k!='record_sha256'})==expected_pair_sha256,
      'broader complete target pair seal differs')
    rebuilt=bind_observed_target(frozen,target)
    paired.need(pair==rebuilt,'broader whole source/target/field/copy/native pair differs')
    return deepcopy(pair)
