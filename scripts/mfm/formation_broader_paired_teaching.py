"""Explicit broader source-first material, preserving the original J6 profile.

Construction admission binds physical observations and author choices. It does
not admit corpus rights, model capacity, learned behavior or write authority.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from cognitive_kernel.canonical import (
    canonical_json_bytes,canonical_sha256,require_sha256)
from . import formation_paired_teaching_v6 as legacy
from . import formation_world_replay_v3 as world_replay

INPUT='mfm-paired-broader-full-source-input-v1'
TARGET='mfm-paired-broader-full-source-original-target-v1'
SAVED='mfm-paired-broader-full-source-saved-input-v1'
OBSERVATION='mfm-broader-three-source-native-observation-v1'
AUTHORING='mfm-broader-complete-native-author-choices-v1'
CONNECTED='mfm-broader-connected-world-roster-v1'
NEW_CUTS=('shared-session','assistant-revision','schedule-revocation')
LEGACY_CUTS=('initial-state','question-open','actor-report',
    'contrary-observation','supported-revision','withdrawal-intent')
ORDERED_CUTS=(*LEGACY_CUTS,*NEW_CUTS)
SEQUENCES=(8,14,18,22,28,30,35,36,37)
need=legacy.need
seal=legacy.seal
pinned=legacy.pinned
original_json=legacy.original_json

def physical_json(raw,ref):
    need(type(ref) is dict and set(ref)=={'path','bytes','file_sha256'} and
        type(ref['bytes']) is int and type(raw) is bytes and
        len(raw)==ref['bytes'] and sha256(raw).hexdigest()==ref['file_sha256'],
        'broader original physical FILE differs')
    row=original_json(raw)
    need(raw in (canonical_json_bytes(row),canonical_json_bytes(row)+b'\n'),
        'broader original JSON encoding differs')
    return row

def paired_code_binding():
    return seal({'schema':'mfm-paired-broader-source-code-binding-v1',
      'paired_source_file_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
      'target_geometry_file_sha256':sha256((Path(__file__).parent/'formation_broader_target_alignment.py').read_bytes()).hexdigest(),
      'current_j6_source_file_sha256':legacy.SOURCE_V6_FILE,
      'original_j5_source_file_sha256':legacy.SOURCE_J5_FILE})

def observation(raw,expected):
    row=pinned(raw,expected,OBSERVATION)
    need(set(row)=={'schema','current_j6_source_file_sha256','original_j5_file_sha256',
      'native_processor_reexecuted','new_native_observation','original_six_processor_reexecuted',
      'processor_binding','proposed_max_source_tokens','source_files','source_only',
      'capacity_admitted','training_authorized','record_sha256'} and
      row['current_j6_source_file_sha256']==legacy.SOURCE_V6_FILE and
      row['original_j5_file_sha256']==legacy.SOURCE_J5_FILE and
      row['native_processor_reexecuted'] is True and row['new_native_observation'] is True and
      row['original_six_processor_reexecuted'] is False and row['source_only'] is True and
      row['capacity_admitted'] is False and row['training_authorized'] is False,
      'broader whole native observation ancestry/status differs')
    rows=row['source_files']
    need(type(rows) is list and len(rows)==3 and [r['cut_id'] for r in rows]==list(NEW_CUTS) and
      [r['sequence'] for r in rows]==[35,36,37] and
      all(set(r)=={'cut_id','sequence','path','bytes','file_sha256','record_sha256',
        'source_tokens','default_native','offset_native'} and
        type(r['source_tokens']) is int and r['source_tokens']>0 for r in rows) and
      type(row['proposed_max_source_tokens']) is int and
      row['proposed_max_source_tokens']==max(r['source_tokens'] for r in rows),
      'broader actual three-source observation roster/maximum differs')
    return row

def _native_originals(source,entry,native_reader):
    default=physical_json(native_reader(entry['default_native']),entry['default_native'])
    offsets=physical_json(native_reader(entry['offset_native']),entry['offset_native'])
    need(set(default)==set(legacy.NATIVE) and
      set(offsets)==set(legacy.NATIVE)|{'offset_mapping','special_tokens_mask'} and
      default==source['native_batch'] and
      all(offsets[k]==default[k] for k in legacy.NATIVE),
      'broader observed default/offset native fields differ')
    length=entry['source_tokens'];tokens=source['alignment']['tokens']
    for key in legacy.NATIVE:
        need(type(default[key]) is list and len(default[key])==1 and
          type(default[key][0]) is list and len(default[key][0])==length and
          all(type(i) is int for i in default[key][0]),'broader native integer batch differs')
    for key,expected in (('offset_mapping',[r['rendered_chars'] for r in tokens]),
                         ('special_tokens_mask',[r['special_mask'] for r in tokens])):
        need(type(offsets[key]) is list and len(offsets[key])==1 and offsets[key][0]==expected,
          'broader original offset/native geometry differs')
    need(all(type(i) is int for i in offsets['special_tokens_mask'][0]) and
      all(type(span) is list and len(span)==2 and all(type(i) is int for i in span)
          for span in offsets['offset_mapping'][0]),'broader offset integer values differ')

def whole_world_prefix(raw,expected_file_sha256,source):
    need(type(raw) is bytes and sha256(raw).hexdigest()==expected_file_sha256,
      'broader exact whole-world FILE differs')
    world=original_json(raw);proof=source['meaning']['episode_snapshot']['historical_prefix_proof'];cut=proof['cut']
    need(raw in (canonical_json_bytes(world),canonical_json_bytes(world)+b'\n'),
      'broader whole-world canonical original differs')
    refs=[row['artifact'] for branch in world['branches'] for row in branch['records']]
    limits=world_replay.C.ConstructionLimits(37,max(r['bytes'] for r in refs),sum(r['bytes'] for r in refs))
    _,_,branches=world_replay._manifest(world,world['world_sha256'],limits)
    world_replay._cut(cut,world['world_id'],source['cut_sha256'])
    rows=branches[cut['branch_id']][:cut['sequence']]
    prefix={k:deepcopy(v) for k,v in world.items() if k not in ('branches','world_sha256')}
    prefix['branches']=[{'branch_id':cut['branch_id'],'parent_cut':None,'records':rows}]
    prefix=world_replay.C.seal(prefix,'world_sha256')
    need(prefix==proof['world'] and source['world_sha256']==prefix['world_sha256'] and
      len(branches[cut['branch_id']])==37 and cut['sequence'] in (35,36,37),
      'broader native source proof is not exact selected whole-world prefix')
    return world

def input_core(j5,v6,source,source_file_sha256,obs,observation_file_sha256,native_reader,
               whole_world_raw,whole_world_file_sha256):
    legacy._source_modules(j5,v6)
    need(sha256(canonical_json_bytes(source)+b'\n').hexdigest()==source_file_sha256,
      'broader complete source FILE differs')
    entries=[r for r in obs['source_files'] if r['cut_id']==source['cut_id']]
    need(len(entries)==1,'broader source absent from actual three-source observation')
    entry=entries[0];maximum=obs['proposed_max_source_tokens']
    need(entry['file_sha256']==source_file_sha256 and entry['record_sha256']==source['record_sha256'] and
      entry['source_tokens']==source['alignment']['source_tokens'] and
      entry['sequence']==source['meaning']['episode_snapshot']['historical_prefix_proof']['cut']['sequence'] and
      obs['processor_binding']==source['alignment']['processor_binding'],
      'broader exact observed source/processor/cut differs')
    _native_originals(source,entry,native_reader)
    world=whole_world_prefix(whole_world_raw,whole_world_file_sha256,source)
    v6.validate_frozen_source(j5,source,j5_file_sha256=legacy.SOURCE_J5_FILE,
      proposed_max_source_tokens=maximum)
    alignment=source['alignment']
    return {'paired_code_binding':paired_code_binding(),'source':deepcopy(source),
      'source_file_sha256':source_file_sha256,'source_observation_file_sha256':observation_file_sha256,
      'source_observation_record_sha256':obs['record_sha256'],'source_observation_entry':deepcopy(entry),
      'current_j6_source_file_sha256':legacy.SOURCE_V6_FILE,'original_j5_source_file_sha256':legacy.SOURCE_J5_FILE,
      'source_cache':deepcopy(alignment['cache']),'source_identity':deepcopy(alignment['source_identity']),
      'processor_binding':deepcopy(alignment['processor_binding']),'cut_id':source['cut_id'],
      'cut_sha256':source['cut_sha256'],'world_sha256':source['world_sha256'],
      'whole_world_sha256':world['world_sha256'],'whole_world_file_sha256':whole_world_file_sha256,
      'whole_world_hex':whole_world_raw.hex(),
      'native_batch':deepcopy(source['native_batch']),'source_tokens':alignment['source_tokens'],
      'proposed_max_source_tokens':maximum,'capacity_admitted':False,'target_accessed':False,
      'training_authorized':False,'current_service_authenticated':False}

def freeze_input(j5,v6,source_reader,observation_reader,*,expected_source_file_sha256,
                 expected_observation_file_sha256,native_reader,whole_world_reader,
                 expected_whole_world_file_sha256,split_row):
    """Validate actual source/default/offset originals before target access."""
    legacy._split(split_row)
    obs=observation(observation_reader(),expected_observation_file_sha256)
    raw=source_reader();source=pinned(raw,expected_source_file_sha256,v6.FROZEN)
    need(raw==canonical_json_bytes(source)+b'\n','broader complete source encoding differs')
    return seal({'schema':INPUT,'split_row':deepcopy(split_row),
      **input_core(j5,v6,source,expected_source_file_sha256,obs,
        expected_observation_file_sha256,native_reader,whole_world_reader(),expected_whole_world_file_sha256)})

def validate_input(j5,v6,row,*,expected_input_sha256,expected_source_file_sha256,
                   observation_reader,native_reader):
    need(row.get('schema')==INPUT and row.get('record_sha256')==expected_input_sha256 and
      canonical_sha256({k:v for k,v in row.items() if k!='record_sha256'})==expected_input_sha256,
      'broader paired input seal differs')
    legacy._split(row['split_row'])
    obs=observation(observation_reader(),row['source_observation_file_sha256'])
    rebuilt=seal({'schema':INPUT,'split_row':row['split_row'],**input_core(j5,v6,row['source'],
      expected_source_file_sha256,obs,row['source_observation_file_sha256'],native_reader,
      bytes.fromhex(row['whole_world_hex']),row['whole_world_file_sha256'])})
    need(row==rebuilt,'broader complete source/native observation join differs')
    return row

def _author_material(frozen,author,materials):
    from cognitive_kernel import formation_learning_v16 as learning
    from . import formation_target_alignment as geometry_api
    from . import formation_joint_target_alignment_v6 as field_api
    need(set(materials)=={'case','target','support','choices','source_view'},
      'broader complete original author material roster differs')
    case,target,support,choices,view=(materials[k] for k in
      ('case','target','support','choices','source_view'))
    source=frozen['source'];caller=source['caller'];meaning=source['meaning']
    need(author['schema']==AUTHORING and author['source_freeze_file_sha256']==frozen['source_file_sha256'] and
      author['source_freeze_record_sha256']==source['record_sha256'] and
      author['case_id']==case['case_id']==view['case_id']==frozen['split_row']['case_id'] and
      author['cut_id']==frozen['cut_id'] and case['split']==frozen['split_row']['split'] and
      case['context']==view['context']==caller['context'] and case['target']==target and
      view['candidate_inventory']==meaning['candidate_inventory'] and
      view['caller_record_sha256']==caller['record_sha256'] and
      view['meaning_record_sha256']==meaning['record_sha256'] and
      view['world_sha256']==frozen['whole_world_sha256'] and
      view['cut']==meaning['episode_snapshot']['historical_prefix_proof']['cut'] and
      view['opened_sources']==[{'ref_id':r['ref_id'],'content_sha256':r['content_sha256'],
        'payload_base64':r['payload_base64']} for r in caller['opened_sources']] and
      case['lineage']['genuine_source_world_sha256']==frozen['whole_world_sha256'] and
      case['lineage']['current_world_cut_sequence']==view['cut']['sequence'],
      'broader original target/caller/cut/source subject join differs')
    need([r['ref_id'] for r in case['sources']]==[r['ref_id'] for r in caller['opened_sources']] and
      all(a['content_b64']==b['payload_base64'] for a,b in zip(case['sources'],caller['opened_sources'],strict=True)) and
      support['dimensions']==author['dimension_decisions'] and
      all(author['dimension_decisions'][k]['state']==v for k,v in target['adjudications'].items()) and
      set(author['dimension_decisions'])==set(learning.FULL_ROLE_DIMENSIONS) and
      author['whole_copy_alternatives_authored'] is True and
      author['unresolved_selection_fields']==[] and author['independent_gold'] is False and
      author['training_corpus_admitted'] is False,
      'broader complete dimension/source-delivery author decisions differ')
    need(all(v!='unknown' for v in target['adjudications'].values()),
      'broader unknown dimension cannot produce a negative teacher label')
    example=learning.learning_example_v16_from_record(case)
    output=learning.supervised_output_record_v16(example)
    geometry=geometry_api.canonical_field_geometry(output,expected_target_sha256=canonical_sha256(output))
    need(author['canonical_target_sha256']==canonical_sha256(output),
      'broader complete authored output differs')
    field_api.field_rows(geometry,frozen,author['field_annotations'])
    selected={r['pointer']:r['candidate_ordinal'] for r in author['field_annotations'] if r['mode']=='external_selection'}
    need(selected=={r['pointer']:r['selected_candidate_ordinal'] for r in author['authored_selection_fields']} and
      selected=={r['pointer']:r['candidate_ordinal'] for r in choices['reference_choices']} and
      len(selected)==len(choices['reference_choices']) and
      author['semantic_field_annotations']==[r for r in author['field_annotations'] if r['mode']=='semantic_authoring'],
      'broader separately authored semantic/reference fields differ')
    return example,output

def convert_original_target(j5,v6,frozen,*,expected_input_sha256,expected_source_file_sha256,
      observation_reader,native_reader,original_root,authoring_reader,
      expected_authoring_file_sha256,geometry_reader,expected_geometry_file_sha256):
    """Revalidate complete source before opening any separately authored target."""
    validate_input(j5,v6,frozen,expected_input_sha256=expected_input_sha256,
      expected_source_file_sha256=expected_source_file_sha256,observation_reader=observation_reader,native_reader=native_reader)
    need(frozen['split_row']['split'] in ('train','development'),'final broader target remains unopened')
    author_raw=authoring_reader();author=pinned(author_raw,expected_authoring_file_sha256,AUTHORING)
    geometry_raw=geometry_reader();component=pinned(geometry_raw,expected_geometry_file_sha256,
      'mfm-broader-canonical-target-native-geometry-v1')
    material_raw={};materials={}
    for key,ref in component['author_originals'].items():
        path=(Path(original_root)/ref['path']).resolve(strict=True)
        need(path.is_relative_to(Path(original_root).resolve(strict=True)), 'broader authored original escapes root')
        raw=path.read_bytes();materials[key]=physical_json(raw,ref);material_raw[key]=raw
    for key in ('case','target','support','choices'):
        author_key={'case':'case_original','target':'target_original','support':'support_original','choices':'reference_choices_original'}[key]
        need(author[author_key]==component['author_originals'][key], 'broader original author descriptor differs')
    view_pin=component['author_originals']['source_view']['file_sha256'];target_pin=component['author_originals']['target']['file_sha256']
    need(materials['case']['lineage']['current_source_view_file_sha256']==view_pin and
      materials['support']['source_view_file_sha256']==materials['choices']['source_view_file_sha256']==view_pin and
      materials['support']['target_file_sha256']==materials['choices']['target_file_sha256']==target_pin,
      'broader physical original view/target author pins differ')
    _,output=_author_material(frozen,author,materials)
    case=materials['case'];original={'schema':case['target']['schema'],'context':case['context'],
      'source_ids':[r['ref_id'] for r in case['sources']],**{k:case['target'][k] for k in ('proposals','dispositions','adjudications')}}
    original_raw=canonical_json_bytes(original)+b'\n'
    return seal({'schema':TARGET,'input_freeze_sha256':frozen['record_sha256'],
      'original_target_hex':original_raw.hex(),'original_target_file_sha256':sha256(original_raw).hexdigest(),
      'raw_author_draft_hex':material_raw['case'].hex(),'raw_author_draft_file_sha256':author['case_original']['file_sha256'],
      'raw_support_hex':material_raw['support'].hex(),'support_file_sha256':author['support_original']['file_sha256'],
      'raw_annotation_hex':author_raw.hex(),'annotation_file_sha256':expected_authoring_file_sha256,
      'projected_view_file_sha256':view_pin,'original_material_hex':{k:v.hex() for k,v in material_raw.items()},
      'original_material_descriptors':component['author_originals'],
      'native_geometry_file_sha256':expected_geometry_file_sha256,'raw_native_geometry_hex':geometry_raw.hex(),
      'canonical_output_sha256':canonical_sha256(output),'unknown_dimensions':[],
      'unresolved_selection_fields':[],'whole_copy_alternatives_authored':True,
      'source_cache':deepcopy(frozen['source_cache']),'state':'complete',
      'author_origin':'model_author','author_id':case['lineage']['target_author'],
      'author_producer_version':case['lineage']['producer_family'],
      'native_choice_author_id':author['author_id'],'native_choice_producer_version':author['producer_version'],
      'author_model_reinvoked_for_projection':False,'independent_gold':False,
      'authenticated_service':False,'training_authorized':False})

def require_complete_labels(frozen,target):
    need(target.get('schema')==TARGET and target.get('record_sha256')==canonical_sha256(
      {k:v for k,v in target.items() if k!='record_sha256'}) and
      target['input_freeze_sha256']==frozen['record_sha256'] and target['source_cache']==frozen['source_cache'] and
      target['state']=='complete' and target['unknown_dimensions']==[] and target['unresolved_selection_fields']==[] and
      target['whole_copy_alternatives_authored'] is True and target['training_authorized'] is False and
      target['authenticated_service'] is False and target['independent_gold'] is False,
      'broader complete target/source seal/status differs')
    author=pinned(bytes.fromhex(target['raw_annotation_hex']),target['annotation_file_sha256'],AUTHORING)
    materials={k:physical_json(bytes.fromhex(raw),target['original_material_descriptors'][k])
      for k,raw in target['original_material_hex'].items()}
    need(bytes.fromhex(target['raw_author_draft_hex'])==bytes.fromhex(target['original_material_hex']['case']) and
      bytes.fromhex(target['raw_support_hex'])==bytes.fromhex(target['original_material_hex']['support']),
      'broader original target draft/support differs')
    example,output=_author_material(frozen,author,materials)
    raw=bytes.fromhex(target['original_target_hex']);original=original_json(raw);case=materials['case']
    need(target['author_origin']=='model_author' and target['author_id']==case['lineage']['target_author'] and
      target['author_producer_version']==case['lineage']['producer_family'] and
      target['native_choice_author_id']==author['author_id'] and
      target['native_choice_producer_version']==author['producer_version'],
      'broader original teacher/native choice author identities differ')
    need(sha256(raw).hexdigest()==target['original_target_file_sha256'] and
      original=={'schema':case['target']['schema'],'context':case['context'],
        'source_ids':[r['ref_id'] for r in case['sources']],**{k:case['target'][k] for k in ('proposals','dispositions','adjudications')}} and
      canonical_sha256(output)==target['canonical_output_sha256'],
      'broader exact standalone original target differs')
    return example,output

def read_json_ref(root,ref):
    """Physical metadata FileRef reader shared by this explicit profile."""
    need(type(ref) is dict and set(ref) in (
        {'path','sha256'},{'path','bytes','file_sha256','record_sha256'}),
        'broader physical FileRef shape differs')
    name=ref['path']
    need(type(name) is str and name and '\\' not in name and
        not Path(name).is_absolute() and
        all(part not in ('','.','..') for part in name.split('/')),
        'broader physical FileRef path differs')
    root=Path(root).resolve(strict=True);path=(root/name).resolve(strict=True)
    need(path.is_relative_to(root) and path.is_file(),
        'broader physical FileRef escapes material root')
    pin=ref.get('file_sha256',ref.get('sha256'));require_sha256(pin,'broader metadata FILE')
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==pin and ('bytes' not in ref or
        type(ref['bytes']) is int and len(raw)==ref['bytes']),
        'broader physical metadata FILE differs')
    row=original_json(raw)
    need(type(row) is dict and raw in (canonical_json_bytes(row),canonical_json_bytes(row)+b'\n') and
        ('record_sha256' not in ref or row.get('record_sha256')==ref['record_sha256']),
        'broader physical metadata record differs')
    return path,row

def validate_connected_roster(material_root,roster,*,expected_roster_sha256,program_cases):
    """Bind nine ordered cuts to actual old30→new37 prefix, without rights."""
    require_sha256(expected_roster_sha256,'broader connected roster')
    need(type(roster) is dict and roster.get('schema')==CONNECTED and
        roster.get('record_sha256')==expected_roster_sha256 and
        canonical_sha256({k:v for k,v in roster.items() if k!='record_sha256'})==expected_roster_sha256 and
        set(roster)=={'schema','world_id','legacy_world','successor_world','legacy_view','successor_view','cases',
          'construction_only','training_authorized','capacity_admitted','record_sha256'} and
        roster['construction_only'] is True and roster['training_authorized'] is False and
        roster['capacity_admitted'] is False,
        'broader connected roster seal/status differs')
    _,old=read_json_ref(material_root,roster['legacy_world'])
    _,new=read_json_ref(material_root,roster['successor_world'])
    for world in (old,new):
        need(world.get('schema')=='mfm-longitudinal-world-v3' and
            world.get('world_sha256')==canonical_sha256({k:v for k,v in world.items() if k!='world_sha256'}) and
            len(world['branches'])==1 and world['branches'][0]['branch_id']=='main' and
            world['branches'][0]['parent_cut'] is None,
            'broader physical world seal/branch differs')
    before=old['branches'][0]['records'];after=new['branches'][0]['records']
    need(old['world_id']==new['world_id']==roster['world_id'] and
        old['scope']==new['scope'] and old['namespace']==new['namespace'] and
        len(before)==30 and len(after)==37 and after[:30]==before and
        all(type(row['sequence']) is int and row['sequence']==i for i,row in enumerate(after,1)),
        'original chronological world prefix differs')
    rows=roster['cases']
    need(type(rows) is list and len(rows)==9 and type(program_cases) is list and
        len(program_cases)==9 and
        [(r['case_id'],r['profile']) for r in program_cases]==[(r['case_id'],r['profile']) for r in rows] and
        [r['cut_id'] for r in rows]==list(ORDERED_CUTS) and
        [r['case_id'] for r in rows]==['j6-reading-room-'+name for name in ORDERED_CUTS] and
        [r['profile'] for r in rows]==['j6-firstparty-successor']*6+['broader-firstparty-successor']*3,
        'broader nine-case chronological program/profile differs')
    cuts=[]
    for item,sequence in zip(rows,SEQUENCES,strict=True):
        need(set(item)=={'case_id','cut_id','profile','cut'},
            'broader exact cut roster differs')
        _,cut=read_json_ref(material_root,item['cut'])
        need(cut.get('schema')=='mfm-world-cut-v1' and
            cut.get('record_sha256')==canonical_sha256({k:v for k,v in cut.items() if k!='record_sha256'}) and
            type(cut['sequence']) is int and cut['sequence']==sequence and
            cut['world_id']==roster['world_id'] and cut['branch_id']=='main' and
            cut['head_sha256']==after[sequence-1]['record_sha256'],
            'broader physical cut/head/prefix differs')
        cuts.append(cut)
    need(len({cut['record_sha256'] for cut in cuts})==9,
        'broader cut identity duplicated')
    for key,world,cut in (('legacy_view',old,cuts[5]),('successor_view',new,cuts[8])):
        _,view=read_json_ref(material_root,roster[key])
        proof=view['historical_prefix_proof']
        need(proof['world']==world and proof['cut']==cut,
            'connected WorldView differs from exact physical world/terminal cut')
        rebuilt,_=world_replay.prefix_proof_replay_v3(proof,cut=cut)
        need(rebuilt==view,'connected original WorldView differs from canonical R3 replay')
    return {'world_id':roster['world_id'],'legacy_world_sha256':old['world_sha256'],
        'successor_world_sha256':new['world_sha256'],'sequences':[c['sequence'] for c in cuts],
        'cut_sha256s':[c['record_sha256'] for c in cuts],
        'original_prefix_preserved':True,'training_authorized':False}
