"""Source-first admission of explicit observed worlds and authored v1.6 material.

This profile does not inherit either historical migration/roster contract.
Its manifest declares the supported original producer formats explicitly.
Rights, feature capacity, trainer integration and learned qualification remain
separate. No tokenizer, model or target reader participates in freeze_input.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path, PurePosixPath

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, require_identifier
from cognitive_kernel import formation_learning_v16 as learning
from . import formation_paired_teaching_v6 as legacy
from . import formation_world_replay_v3 as replay
from . import formation_target_alignment as geometry_api
from . import formation_joint_target_alignment_v6 as fields_api

MANIFEST = 'mfm-observed-world-source-manifest-v1'
INPUT = 'mfm-paired-observed-world-input-v1'
TARGET = 'mfm-paired-observed-world-original-target-v1'
AUTHOR_MANIFEST = 'mfm-observed-world-author-manifest-v1'
AUTHOR_CONTRACT = 'rill-complete-authored-v1'
COMPOSITION_CONTRACT = 'rill-target-free-composed-originals-v1'
NATIVE_RECEIPT = 'mfm-complete-new-world-native-material-v1'
NATIVE_TARGET_RECEIPT = 'mfm-rill-complete-native-target-observation-v1'
NATIVE_TARGET = 'mfm-rill-full-canonical-target-native-geometry-v1'
need, seal, pinned = legacy.need, legacy.seal, legacy.pinned
ROLES = {'cut','request','verification_boundary','delivery_boundary','view',
         'verification_receipt','bundle','caller','meaning'}


def _record(value, schema=None):
    need(type(value) is dict and (schema is None or value.get('schema') == schema) and
         value.get('record_sha256') == canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'}),
         'observed original record seal/schema differs')
    return value


def _name(name):
    need(type(name) is str and name and '\\' not in name and
         not PurePosixPath(name).is_absolute() and ':' not in name and
         all(p not in ('','.','..') for p in name.split('/')), 'observed original relative path differs')
    return name


def read_original(root, ref):
    need(type(ref) is dict and set(ref)=={'path','bytes','file_sha256'} and
         type(ref['bytes']) is int and ref['bytes']>=0, 'observed physical descriptor differs')
    root=Path(root).resolve(strict=True)
    path=legacy._ordinary(root/_name(ref['path']))
    need(path.resolve(strict=True).is_relative_to(root), 'observed original leaves material root')
    raw=path.read_bytes()
    need(len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['file_sha256'], 'observed original physical FILE differs')
    return raw


def json_original(root, ref):
    raw=read_original(root,ref); value=legacy.original_json(raw)
    need(type(value) is dict and raw in (canonical_json_bytes(value),canonical_json_bytes(value)+b'\n'),
         'observed original JSON must retain canonical bytes')
    return value


def _relative(parent_ref, child):
    return {**child,'path':str(PurePosixPath(parent_ref['path']).parent/child['path'])}


def paired_code_binding():
    return seal({'schema':'mfm-observed-world-code-binding-v1',
      'paired_source_file_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
      'target_geometry_file_sha256':sha256(Path(__file__).with_name('formation_observed_world_target_alignment.py').read_bytes()).hexdigest(),
      'current_j6_source_file_sha256':legacy.SOURCE_V6_FILE,
      'original_j5_source_file_sha256':legacy.SOURCE_J5_FILE})


def _manifest(root, raw, expected):
    m=pinned(raw,expected,MANIFEST)
    need(set(m)=={'schema','source_contract','composition_contract','world','composition_manifest',
      'composition_result','native_receipt','native_runtime','native_producer','cases','source_only',
      'independent_gold','training_authorized','record_sha256'} and
      m['source_contract']=='complete-new-world-native-material-v1' and m['composition_contract']==COMPOSITION_CONTRACT and
      m['source_only'] is True and m['independent_gold'] is False and m['training_authorized'] is False,
      'observed source manifest contract/status differs')
    composed=_record(json_original(root,m['composition_manifest']),COMPOSITION_CONTRACT)
    result=_record(json_original(root,m['composition_result']),'rill-target-free-composition-result-v1')
    observation=_record(json_original(root,m['native_receipt']),NATIVE_RECEIPT)
    runtime=json_original(root,m['native_runtime'])
    need(result['status']=='completed' and result['input_and_code_brackets_passed'] is True and
      result['stores_closed'] is True and result['targets_read_or_authored'] is False and result['training_admitted'] is False and
      _relative(m['composition_result'],result['manifest_file'])==m['composition_manifest'] and
      composed['source_only'] is True and composed['targets_read_or_authored'] is False and
      composed['training_admitted'] is False and composed['independent_gold'] is False,
      'actual target-free composition is incomplete')
    need(observation['status']=='completed' and observation['source_only'] is True and
      all(observation[k] is False for k in ('target_accessed','model_weights_loaded','learned_forward','training_admitted','independent_gold')) and
      observation['no_padding_or_truncation'] is True and observation['originals_unchanged'] is True and
      observation['world_manifest_FILE_sha256']==m['composition_manifest']['file_sha256'] and
      observation['world_result_FILE_sha256']==m['composition_result']['file_sha256'] and
      observation['producer_FILE_sha256']==m['native_producer']['file_sha256'] and
      observation['current_j6_source_FILE_sha256']==legacy.SOURCE_V6_FILE and observation['original_j5_FILE_sha256']==legacy.SOURCE_J5_FILE,
      'completed native source/producer/world ancestry differs')
    read_original(root,m['native_producer'])
    need(runtime['processor_binding']==observation['processor_binding'] and
      runtime['processor_settings']=={'padding':None,'truncation':None} and
      type(runtime['runtime']) is dict and type(runtime['module_origins']) is list,
      'observed actual processor/runtime binding differs')
    for module,pin in (('scripts.mfm.formation_joint_input_v5',legacy.SOURCE_J5_FILE),
                       ('scripts.mfm.formation_joint_input_v6',legacy.SOURCE_V6_FILE)):
        need(len([r for r in runtime['module_origins'] if r['module']==module and r['file_sha256']==pin])==1,
             'observed native runtime source origin differs')
    cases=m['cases']; entries=observation['source_files']; cuts=composed['cuts']
    need(type(cases) is list and cases and len(cases)==len(entries)==len(cuts) and
      len({r['case_id'] for r in cases})==len(cases) and len({r['cut_id'] for r in cases})==len(cases) and
      [r['cut_id'] for r in cases]==[r['cut_id'] for r in entries]==[r['cut_id'] for r in cuts],
      'complete observed case/cut roster differs')
    need([r['sequence'] for r in entries]==[r['sequence'] for r in cuts] and
      [r['sequence'] for r in entries]==sorted({r['sequence'] for r in entries}) and
      observation['observed_maximum_source_tokens']==max(r['source_tokens'] for r in entries) and
      observation['source_tokens']==sum(r['source_tokens'] for r in entries), 'observed source order/lengths differ')
    partitions={r['split'] for r in cases}
    need(len(partitions)==1 and partitions<={'train','development','final'}, 'one connected world cannot cross partitions')
    for index,(c,cut) in enumerate(zip(cases,cuts,strict=True)):
        need(set(c)=={'case_id','cut_id','variant_key','split','lineage','metadata'} and set(c['metadata'])==ROLES,
             'observed case metadata roles differ')
        require_identifier(c['case_id'],'observed case');require_identifier(c['variant_key'],'observed variant')
        need(c['lineage']['world_id']==composed['world_id'] and
          all(type(c['lineage'].get(k)) is str and c['lineage'][k] for k in ('generator_family','source_author_family')),
          'explicit connected world/author ancestry required')
        need(c['lineage']['generator_family']==composed['construction_author']['generator_family'] and
          c['lineage']['source_author_family']==composed['construction_author']['world_author_family'] and
          c['lineage']['parent_case_id']==(cases[index-1]['case_id'] if index else None),
          'observed connected author/parent ancestry cannot be relabelled')
        need(c['metadata']=={k:_relative(m['composition_manifest'],v) for k,v in cut['files'].items()},
             'observed case physical composition roles differ')
    need(m['world']==_relative(m['composition_manifest'],composed['world_file']), 'exact whole-world FILE differs')
    return m,composed,observation


def _world_prefix(world, source):
    refs=[r['artifact'] for b in world['branches'] for r in b['records']]
    need(bool(refs),'empty observed world')
    limits=replay.C.ConstructionLimits(len(refs),max(r['bytes'] for r in refs),sum(r['bytes'] for r in refs))
    _,_,branches=replay._manifest(world,world['world_sha256'],limits)
    proof=source['meaning']['episode_snapshot']['historical_prefix_proof'];cut=proof['cut']
    replay._cut(cut,world['world_id'],source['cut_sha256'])
    need(cut['branch_id'] in branches and 0<cut['sequence']<=len(branches[cut['branch_id']]), 'cut leaves admitted world')
    prefix={k:deepcopy(v) for k,v in world.items() if k not in ('branches','world_sha256')}
    prefix['branches']=[{'branch_id':cut['branch_id'],'parent_cut':None,'records':branches[cut['branch_id']][:cut['sequence']]}]
    prefix=replay.C.seal(prefix,'world_sha256')
    need(prefix==proof['world'] and prefix['world_sha256']==source['world_sha256'], 'observed exact world prefix differs')
    return len(refs)


def _physical_boundaries(root,m,composed,case,source):
    d={k:json_original(root,v) for k,v in case['metadata'].items()}; b=d['bundle']
    need(d['caller']==source['caller']==b['caller'] and d['meaning']==source['meaning']==b['meaning'] and
      d['cut']==b['cut']==d['request']['cut']==d['view']['cut']==d['view']['historical_prefix_proof']['cut'] and
      d['cut']['record_sha256']==source['cut_sha256'] and
      d['request']['read_boundary']==d['delivery_boundary'] and
      b['input_request_sha256']==d['request']['record_sha256'] and
      b['replay_sha256']==d['view']['replay_sha256']==d['verification_receipt']['replay_sha256'] and
      b['verification_receipt']==d['verification_receipt'] and
      b['verification_recheck_reads']==d['verification_receipt']['reads'] and
      d['verification_receipt']['world_sha256']==composed['world_sha256'] and
      d['verification_receipt']['cut_sha256']==source['cut_sha256'], 'observed caller/cut/subject/time/boundary join differs')
    proof=b['original_artifact_pins']; delivery=b['delivery_read_receipt']; verified=d['verification_receipt']
    need([r['artifact'] for r in verified['reads']]==proof and len({r['artifact_id'] for r in proof})==len(proof),
         'observed proof roster differs')
    union={r['artifact_id']:r for r in proof}
    for calls,boundary in ((verified,d['verification_boundary']),(delivery,d['delivery_boundary'])):
        need(calls['boundary_sha256']==boundary['record_sha256'], 'observed receipt boundary differs')
        for call in calls['reads']:
            a=call['artifact'];aid=a['artifact_id']
            need(aid not in union or union[aid]==a,'observed delivery/proof identity differs');union[aid]=a
            ref=_relative(m['composition_manifest'],composed['artifacts'][aid]); data=read_original(root,ref)
            need(a['bytes']==len(data)==call['body_bytes'] and a['file_sha256']==sha256(data).hexdigest()==call['body_sha256'] and
              call['pre_boundary_sha256']==call['post_boundary_sha256']==boundary['record_sha256'],
              'observed proof/delivery original body differs')
    c=next(c for c in composed['cuts'] if c['cut_id']==case['cut_id'])
    need(c['cut_sha256']==source['cut_sha256'] and c['as_of']==d['delivery_boundary']['as_of'] and
      not set(c['withheld_source_ids']) & {s['ref_id'] for s in source['caller']['opened_sources']},
      'observed cut time or withheld delivery differs')
    return {'metadata':deepcopy(case['metadata']),
      'ordered_proof_delivery_union':[{'artifact':a,'original':_relative(m['composition_manifest'],composed['artifacts'][aid])} for aid,a in union.items()],
      'delivery':[{'artifact':c['artifact'],'family':c['family'],'key':c['key'],
        'original':_relative(m['composition_manifest'],composed['artifacts'][c['artifact']['artifact_id']])} for c in delivery['reads']]}


def freeze_input(j5,v6,*,material_root,manifest_reader,expected_manifest_file_sha256,case_id):
    """Admit source-only originals; no author or target reader is accepted."""
    legacy._source_modules(j5,v6)
    manifest_raw=manifest_reader();m,composed,obs=_manifest(material_root,manifest_raw,expected_manifest_file_sha256)
    selected=[(c,s) for c,s in zip(m['cases'],obs['source_files'],strict=True) if c['case_id']==case_id]
    need(len(selected)==1,'observed case absent from exact roster');case,entry=selected[0]
    refs={key:_relative(m['native_receipt'],entry[key]) for key in ('source','default_native','offset_native','rendered_source')}
    source=pinned(read_original(material_root,refs['source']),refs['source']['file_sha256'],v6.FROZEN)
    maximum=obs['observed_maximum_source_tokens']
    v6.validate_frozen_source(j5,source,j5_file_sha256=legacy.SOURCE_J5_FILE,proposed_max_source_tokens=maximum)
    alignment=source['alignment'];default=json_original(material_root,refs['default_native']);offsets=json_original(material_root,refs['offset_native'])
    need(set(default)==set(legacy.NATIVE) and set(offsets)==set(legacy.NATIVE)|{'offset_mapping','special_tokens_mask'} and
      default==source['native_batch'] and all(offsets[k]==default[k] for k in legacy.NATIVE) and
      offsets['offset_mapping']==[[t['rendered_chars'] for t in alignment['tokens']]] and
      offsets['special_tokens_mask']==[[t['special_mask'] for t in alignment['tokens']]] and
      read_original(material_root,refs['rendered_source'])==alignment['rendered_text'].encode('utf-8') and
      source['cut_id']==case['cut_id'] and alignment['source_tokens']==entry['source_tokens'] and
      alignment['processor_binding']==obs['processor_binding'] and alignment['source_identity']==obs['source_identity'] and
      source['meaning']['episode_snapshot']['historical_prefix_proof']['cut']['sequence']==entry['sequence'],
      'observed complete default/offset/rendered/source identity differs')
    world=json_original(material_root,m['world']);count=_world_prefix(world,source)
    need(world['world_sha256']==composed['world_sha256'] and count==composed['world_record_count'], 'observed whole-world record count differs')
    material=_physical_boundaries(material_root,m,composed,case,source)
    return seal({'schema':INPUT,'paired_code_binding':paired_code_binding(),'source':source,
      'source_manifest_hex':manifest_raw.hex(),'source_manifest_file_sha256':expected_manifest_file_sha256,
      'source_file_sha256':refs['source']['file_sha256'],'source_originals':refs,
      'source_observation_file_sha256':m['native_receipt']['file_sha256'],
      'source_cache':deepcopy(alignment['cache']),'source_identity':deepcopy(alignment['source_identity']),
      'processor_binding':deepcopy(alignment['processor_binding']),'native_batch':deepcopy(source['native_batch']),
      'cut_id':source['cut_id'],'cut_sha256':source['cut_sha256'],'world_sha256':source['world_sha256'],
      'whole_world_sha256':world['world_sha256'],'whole_world_file_sha256':m['world']['file_sha256'],
      'whole_world_record_count':count,'source_tokens':alignment['source_tokens'],'proposed_max_source_tokens':maximum,
      'split_row':{'case_id':case_id,'split':case['split'],'lineage':deepcopy(case['lineage'])},
      'variant_key':case['variant_key'],'material':material,'target_accessed':False,'capacity_admitted':False,
      'training_authorized':False,'current_service_authenticated':False,'independent_gold':False})


def validate_input(j5,v6,row,*,material_root,expected_input_sha256):
    _record(row,INPUT);need(row['record_sha256']==expected_input_sha256,'observed input external pin differs')
    rebuilt=freeze_input(j5,v6,material_root=material_root,manifest_reader=lambda:bytes.fromhex(row['source_manifest_hex']),
      expected_manifest_file_sha256=row['source_manifest_file_sha256'],case_id=row['split_row']['case_id'])
    need(row==rebuilt,'observed whole input/subject/time/split differs')
    return row


def _author_material(frozen, manifest, materials):
    need(manifest['input_contract']==AUTHOR_CONTRACT,'unsupported observed original author contract')
    case,target,support,choices,view=(materials[k] for k in ('case','target','support','choices','source_view'))
    source=frozen['source'];caller=source['caller'];meaning=source['meaning'];lineage=case['lineage']
    need(choices['schema']=='mfm-rill-complete-authored-field-reference-literal-choices-001' and
      case['case_id']==manifest['case_id']==view['case_id']==support['case_id']==frozen['split_row']['case_id'] and
      case['split']==frozen['split_row']['split'] and case['context']==view['context']==caller['context'] and
      case['target']==target and view['candidate_inventory']==meaning['candidate_inventory'] and
      view['caller_record_sha256']==caller['record_sha256'] and view['meaning_record_sha256']==meaning['record_sha256'] and
      view['world_sha256']==lineage['world_sha256']==frozen['whole_world_sha256'] and
      view['cut']==meaning['episode_snapshot']['historical_prefix_proof']['cut'] and
      lineage['current_world_cut_sequence']==view['cut']['sequence'] and
      all(lineage[k]==v for k,v in frozen['split_row']['lineage'].items()),
      'observed author/caller/subject/time/world lineage differs')
    expected_sources=[{'ref_id':r['ref_id'],'content_sha256':r['content_sha256'],'payload_base64':r['payload_base64']} for r in caller['opened_sources']]
    need(view['opened_sources']==expected_sources and
      [r['ref_id'] for r in case['sources']]==[r['ref_id'] for r in caller['opened_sources']] and
      all(a['content_b64']==b['payload_base64'] for a,b in zip(case['sources'],caller['opened_sources'],strict=True)),
      'observed target source delivery differs')
    refs=manifest['materials'];view_pin=refs['source_view']['file_sha256'];target_pin=refs['target']['file_sha256']
    need(lineage['current_source_view_file_sha256']==support['source_view_file_sha256']==choices['source_view_file_sha256']==view_pin and
      support['target_file_sha256']==choices['target_file_sha256']==target_pin and
      all(lineage[key]==refs[role]['file_sha256'] for role,key in (
        ('raw_semantic','raw_model_author_output_sha256'),('raw_reference','raw_reference_author_output_sha256'),('raw_literal','raw_literal_author_output_sha256'))),
      'observed actual raw author/source view FILE ancestry differs')
    need(set(support['dimensions'])==set(learning.FULL_ROLE_DIMENSIONS) and
      all(support['dimensions'][k]['state']==v and v!='unknown' for k,v in target['adjudications'].items()) and
      support['all_ten_dimensions_explicit'] is True and choices['all_nonnull_external_references_bound'] is True and
      choices['whole_literal_inventory_complete'] is True and choices['independent_gold'] is False and
      support['independent_gold'] is False and view['independent_gold'] is False and
      choices['author_id']==support['author']==lineage['target_author'], 'complete actual model-authored supervision required')
    example=learning.learning_example_v16_from_record(case,split=case['split']);output=learning.supervised_output_record_v16(example)
    geometry=geometry_api.canonical_field_geometry(output,expected_target_sha256=canonical_sha256(output))
    need(materials['canonical_field_geometry']==geometry,'original authored canonical field geometry differs')
    fields=fields_api.field_rows(geometry,frozen,choices['field_annotations'])
    selected={r['pointer']:r['selected_candidate_ordinal'] for r in fields if 'selected_candidate_ordinal' in r}
    need(len(choices['reference_choices'])==len(selected) and selected=={r['pointer']:r['candidate_ordinal'] for r in choices['reference_choices']} and
      choices['candidate_inventory_sha256']==canonical_sha256(meaning['candidate_inventory']) and
      choices['policy_sha256']==fields_api.field_api.normative_policy()['record_sha256'], 'complete explicit reference choices differ')
    by_pointer={r['pointer']:r for r in fields}
    for row in choices['reference_choices']:
        candidate=meaning['candidate_inventory'][row['candidate_ordinal']]
        need(set(row)=={'pointer','candidate_ordinal','candidate_path','candidate_sha256','ref_id','kind','origin'} and
          row['candidate_sha256']==canonical_sha256(candidate) and
          all(row[k]==candidate[k] for k in ('ref_id','kind','origin')) and
          row['candidate_path']==by_pointer[row['pointer']]['candidate_path'] and
          row['candidate_path'] in candidate['eligible_paths'], 'authored eligible candidate identity differs')
    from cognitive_kernel import formation_decision_v16 as local
    need(materials['decision']==local.decision_record(example),'original local-ID decision differs')
    return example,output


def convert_original_target(j5,v6,frozen,*,material_root,expected_input_sha256,
                            author_manifest_reader,expected_author_manifest_file_sha256):
    """Recheck full source before the first author-manifest read; FINAL stays shut."""
    validate_input(j5,v6,frozen,material_root=material_root,expected_input_sha256=expected_input_sha256)
    need(frozen['split_row']['split'] in ('train','development'),'final observed target remains unopened')
    raw=author_manifest_reader();m=pinned(raw,expected_author_manifest_file_sha256,AUTHOR_MANIFEST)
    need(set(m)=={'schema','input_contract','case_id','author_producer_version','conversion_id','materials',
      'native_geometry','native_original','native_receipt','native_producer','independent_gold','training_authorized','record_sha256'} and
      m['input_contract']==AUTHOR_CONTRACT and m['independent_gold'] is False and m['training_authorized'] is False and
      all(type(m[k]) is str and m[k] for k in ('author_producer_version','conversion_id')), 'observed explicit author conversion contract differs')
    need(set(m['materials'])=={'case','target','support','choices','source_view','canonical_field_geometry','decision',
      'raw_semantic','raw_reference','raw_literal'}, 'complete observed original author files required')
    originals={k:read_original(material_root,v) for k,v in m['materials'].items()}
    materials={k:legacy.original_json(v) for k,v in originals.items()}
    example,output=_author_material(frozen,m,materials)
    native={k:read_original(material_root,m[k]) for k in ('native_geometry','native_original','native_receipt','native_producer')}
    receipt=_record(legacy.original_json(native['native_receipt']),NATIVE_TARGET_RECEIPT)
    need(receipt['status']=='completed' and receipt['producer_FILE_sha256']==m['native_producer']['file_sha256'] and
      receipt['source_receipt_FILE_sha256']==frozen['source_observation_file_sha256'] and receipt['all_originals_preserved'] is True and
      receipt['no_padding_or_truncation'] is True and receipt['same_author_training_material'] is True and
      all(receipt[k] is False for k in ('training_admitted','independent_gold','learned_forward','model_weights_loaded','output_repair','historical_pair_profile_claimed')),
      'actual observed target receipt/producer/source differs')
    for key in ('native_geometry','native_original'):
        need(m[key] in [_relative(m['native_receipt'],r) for r in receipt['files']], 'target original absent from complete observation manifest')
    return seal({'schema':TARGET,'input_freeze_sha256':frozen['record_sha256'],'source_cache':deepcopy(frozen['source_cache']),
      'author_manifest_hex':raw.hex(),'author_manifest_file_sha256':expected_author_manifest_file_sha256,
      'original_material_hex':{k:v.hex() for k,v in originals.items()},
      'original_target_file_sha256':m['materials']['target']['file_sha256'],
      'annotation_file_sha256':m['materials']['choices']['file_sha256'],
      'native_original_hex':native['native_original'].hex(),'native_geometry_hex':native['native_geometry'].hex(),
      'native_receipt_hex':native['native_receipt'].hex(),
      'native_geometry_file_sha256':m['native_geometry']['file_sha256'],
      'canonical_output_sha256':canonical_sha256(output),'unknown_dimensions':[],
      'author_origin':'model_author','author_id':materials['case']['lineage']['target_author'],
      'author_producer_version':m['author_producer_version'],'conversion_id':m['conversion_id'],
      'state':'complete','independent_gold':False,'training_authorized':False,'authenticated_service':False})


def require_complete_labels(frozen,target):
    _record(frozen,INPUT);_record(target,TARGET)
    need(target['input_freeze_sha256']==frozen['record_sha256'] and target['source_cache']==frozen['source_cache'] and
      target['state']=='complete' and target['unknown_dimensions']==[] and
      all(target[k] is False for k in ('independent_gold','training_authorized','authenticated_service')),
      'observed complete target/source status differs')
    m=pinned(bytes.fromhex(target['author_manifest_hex']),target['author_manifest_file_sha256'],AUTHOR_MANIFEST)
    materials={}
    need(set(target['original_material_hex'])==set(m['materials']), 'observed retained author roster differs')
    for key,encoded in target['original_material_hex'].items():
        raw=bytes.fromhex(encoded);ref=m['materials'][key]
        need(len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['file_sha256'],'observed retained authored bytes differ')
        materials[key]=legacy.original_json(raw)
    example,output=_author_material(frozen,m,materials)
    need(target['original_target_file_sha256']==m['materials']['target']['file_sha256'] and
      target['annotation_file_sha256']==m['materials']['choices']['file_sha256'] and
      target['canonical_output_sha256']==canonical_sha256(output) and target['author_origin']=='model_author' and
      target['author_id']==materials['case']['lineage']['target_author'] and
      target['author_producer_version']==m['author_producer_version'] and target['conversion_id']==m['conversion_id'],
      'observed original target/author identity differs')
    return example,output
