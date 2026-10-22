"""Bind complete saved native target geometry to observed-world author choices.

No tokenization, semantic authoring, source generation or lowerer dispatch is
performed. The supported original author/observer contract is explicit.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import PurePosixPath

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from . import formation_observed_world_paired_teaching as paired
from . import formation_joint_target_alignment_v6 as original
from . import formation_target_alignment as geometry_api

PAIR='mfm-paired-observed-world-target-geometry-v1'
CHOICES='mfm-observed-world-external-field-and-literal-choices-v1'
field_rows=original.field_rows
literal_rows=original.literal_rows


def _observed(frozen,target):
    _,output=paired.require_complete_labels(frozen,target)
    manifest=paired.pinned(bytes.fromhex(target['author_manifest_hex']),target['author_manifest_file_sha256'],paired.AUTHOR_MANIFEST)
    component=paired.pinned(bytes.fromhex(target['native_geometry_hex']),manifest['native_geometry']['file_sha256'],paired.NATIVE_TARGET)
    receipt=paired.pinned(bytes.fromhex(target['native_receipt_hex']),manifest['native_receipt']['file_sha256'],paired.NATIVE_TARGET_RECEIPT)
    native_raw=bytes.fromhex(target['native_original_hex'])
    paired.need(len(native_raw)==manifest['native_original']['bytes'] and sha256(native_raw).hexdigest()==manifest['native_original']['file_sha256'],
                'observed original target native FILE differs')
    native=paired.legacy.original_json(native_raw)
    choices=paired.legacy.original_json(bytes.fromhex(target['original_material_hex']['choices']))
    geometry=geometry_api.canonical_field_geometry(output,expected_target_sha256=target['canonical_output_sha256'])
    source_ref=component['source'];native_ref=component['native_original']
    paired.need(source_ref['file_sha256']==frozen['source_file_sha256'] and
      source_ref['bytes']==frozen['source_originals']['source']['bytes'] and
      component['source_receipt_FILE_sha256']==receipt['source_receipt_FILE_sha256']==frozen['source_observation_file_sha256'] and
      receipt['status']=='completed' and receipt['all_originals_preserved'] is True and receipt['no_padding_or_truncation'] is True and
      receipt['producer_FILE_sha256']==manifest['native_producer']['file_sha256'] and receipt['same_author_training_material'] is True and
      all(receipt[k] is False for k in ('training_admitted','independent_gold','learned_forward','model_weights_loaded','output_repair','historical_pair_profile_claimed')) and
      component['processor_binding']==frozen['processor_binding'] and component['geometry']==geometry and
      component['case_id']==frozen['split_row']['case_id'] and component['cut_id']==frozen['cut_id'] and
      all(component[k] is False for k in ('training_admitted','independent_gold','capacity_admitted')) and
      paired._relative(manifest['native_receipt'],native_ref)==manifest['native_original'] and
      set(native)=={'input_ids','attention_mask','offset_mapping','special_tokens_mask'},
      'observed native target/source/processor/authority binding differs')
    for key in ('native_geometry','native_original'):
        paired.need(manifest[key] in [paired._relative(manifest['native_receipt'],r) for r in receipt['files']],
                    'observed target FILE absent from original terminal manifest')
    author_root=PurePosixPath(manifest['materials']['case']['path']).parent.parent
    for key in ('case','target','support','choices','source_view','canonical_field_geometry','decision'):
        ref=component['author_originals'][key]
        paired.need(manifest['materials'][key]=={**ref,'path':str(author_root/ref['path'])},
                    'observed original target author descriptors differ')
    ids,mask,offsets,special=(native[k] for k in ('input_ids','attention_mask','offset_mapping','special_tokens_mask'))
    paired.need(all(type(row) is list for row in (ids,mask,offsets,special)) and
      len(ids)==len(mask)==len(offsets)==len(special) and
      all(type(i) is int for row in (ids,mask,special) for i in row) and
      all(type(span) is list and len(span)==2 and all(type(i) is int for i in span) for span in offsets),
      'observed complete target native shape differs')
    tokens=geometry_api._token_rows(geometry,ids,mask,offsets,special)
    inputs,labels=component['decoder_inputs'],component['labels']
    maximum=receipt['observed_maximum_canonical_target_labels']
    wrapper=frozen['processor_binding']['tokenizer_wrapper']
    paired.need(component['tokens']==tokens and type(maximum) is int and maximum>0 and
      type(inputs) is list and type(labels) is list and
      all(type(i) is int and i>=0 for row in (inputs,labels) for i in row) and
      len(inputs)==len(labels)==len(ids)+1==component['observed_target_labels'] and len(labels)<=maximum and
      inputs[1:]==labels[:-1]==ids and inputs[0]==wrapper['bos_token_id'] and labels[-1]==wrapper['eos_token_id'],
      'observed whole target labels/BOS/EOS/native geometry differs')
    summary=[r for r in receipt['cases'] if r['cut_id']==frozen['cut_id']]
    paired.need(len(summary)==1 and summary[0]['canonical_target_labels']==len(labels) and
      maximum==max(r['canonical_target_labels'] for r in receipt['cases']), 'observed target length roster differs')
    fields=field_rows(geometry,frozen,choices['field_annotations'])
    literals=literal_rows(geometry,tokens,frozen,choices['literal_declarations'])
    paired.need(component['fields']==fields and component['literal_segments']==literals,
                'observed actual target field/literal choices differ')
    choice=paired.seal({'schema':CHOICES,'input_freeze_sha256':frozen['record_sha256'],
      'target_record_sha256':target['record_sha256'],'author_id':target['author_id'],
      'author_provenance_file_sha256':target['annotation_file_sha256'],
      'field_annotations':deepcopy(choices['field_annotations']),
      'literal_declarations':deepcopy(choices['literal_declarations']),
      'whole_literal_inventory_complete':True})
    choice_raw=canonical_json_bytes(choice)+b'\n'
    core=original._core(frozen,target,sha256(choice_raw).hexdigest(),choice_raw,choice,
      geometry,inputs,labels,tokens,maximum,fields,literals)
    core.update(schema=PAIR,processor_reexecuted_for_target=False,
      observed_native_target_file_sha256=target['native_geometry_file_sha256'],
      native_target_observation_consumed=True,original_author_contract=paired.AUTHOR_CONTRACT)
    return core


def bind_observed_target(frozen,target):
    return paired.seal(_observed(frozen,target))


def validate_pair(pair,frozen,target,*,expected_pair_sha256):
    paired._record(pair,PAIR)
    paired.need(pair['record_sha256']==expected_pair_sha256,'observed pair external record pin differs')
    paired.need(pair==bind_observed_target(frozen,target),'observed whole source/target/native pair differs')
    return deepcopy(pair)
