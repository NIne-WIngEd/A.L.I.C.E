"""Versioned supplied-record teaching preparation, with no read-service authority.

Complete J4 input is frozen before a separately supplied original v1.6 target.
This profile does not implement FileRef/current-rights/corpus acquisition or
grant teacher, training, native, full-role or production qualification.
"""
from copy import deepcopy
from hashlib import sha256
import json
from cognitive_kernel.canonical import canonical_json_bytes,canonical_sha256,CognitiveKernelContractError,require_sha256,require_identifier
from cognitive_kernel import formation_learning_v16 as L
from . import formation_joint_input_v4 as J

INPUT='mfm-paired-supplied-input-v1'
TARGET='mfm-paired-supplied-original-target-v1'
EXPECTED={'caller_sha256','meaning_sha256','joint_sha256','batch_sha256','joint_code_binding','processor_binding','source_identity','max_source_tokens'}
LINEAGE={'world_id','branch_id','case_family','world_author_family','generator_family','rendering_family','source_system_family','ancestry_group_ids','parent_case_ids'}

def need(ok,message):
    if not ok:raise CognitiveKernelContractError(message)

def seal(row):return {**deepcopy(row),'record_sha256':canonical_sha256(row)}

def exact(row,keys,name):need(type(row) is dict and set(row)==set(keys),name+' shape differs')

def pin(row,expected,schema):
    require_sha256(expected,'external record pin')
    need(type(row) is dict and row.get('schema')==schema and row.get('record_sha256')==expected and
        canonical_sha256({k:v for k,v in row.items() if k!='record_sha256'})==expected,'external record differs')

def identifier(value):need(type(value) is str and require_identifier(value,'identity')==value,'noncanonical identity')

def validate_split(row):
    exact(row,{'case_id','split','lineage'},'split');identifier(row['case_id']);need(row['split'] in ('train','development','final'),'unsupported split')
    exact(row['lineage'],LINEAGE,'lineage')
    for key,value in row['lineage'].items():
        if key in ('ancestry_group_ids','parent_case_ids'):
            need(type(value) is list,'lineage IDs must be array')
            for item in value:identifier(item)
            need(len(set(value))==len(value),'duplicate lineage IDs')
        else:identifier(value)

def validate_saved_joint(caller,meaning,row,batch,*,expected):
    """Recompute all saved render/joins/cache; no tokenizer or read service."""
    exact(expected,EXPECTED,'independent expectations')
    budget=expected['max_source_tokens'];need(type(budget) is int and budget>0,'explicit source budget required')
    pin(caller,expected['caller_sha256'],J.caller_api.SCHEMA);pin(meaning,expected['meaning_sha256'],J.meaning_api.SCHEMA);pin(row,expected['joint_sha256'],J.SCHEMA)
    need(row['code_binding']==expected['joint_code_binding']==J.joint_code_binding_v4(),'expected frozen joint code differs')
    need(row['processor_binding']==expected['processor_binding'] and row['source_identity']==expected['source_identity'],'source/processor expectation differs')
    J._identity(row['source_identity'],row['source_identity']['record_sha256'])
    need(row['max_source_tokens']==budget,'whole source budget differs');exact(batch,{'input_ids','attention_mask'},'saved batch')
    need(canonical_sha256(batch)==expected['batch_sha256'] and batch=={'input_ids':[row['input_ids']],'attention_mask':[row['attention_mask']]},'independent full batch differs')
    ids,masks=row['input_ids'],row['attention_mask'];need(type(ids) is list and type(masks) is list and 0<len(ids)<=budget and len(ids)==len(masks),'complete source budget overflow or shape differs')
    need(all(type(x) is int and x>=0 for x in ids) and all(type(x) is int and x==1 for x in masks),'invalid saved IDs/masks')
    _,text,blocks,fields,sources,parts=J._render(caller,meaning,expected['caller_sha256'],expected['meaning_sha256'],row['render_policy'],row['render_policy']['record_sha256'])
    need(type(row['tokens']) is list and len(row['tokens'])==len(ids),'complete token roster differs')
    tokens,candidates=J._joins(text,sources,ids,masks,[r['rendered_chars'] for r in row['tokens']],[r['special_mask'] for r in row['tokens']],
        set(expected['processor_binding']['tokenizer_wrapper']['all_special_ids']),fields,blocks,meaning['candidate_inventory'])
    # Canonical saved JSON sorts dictionaries used by J4's part iteration.
    # Verify every complete row, then retain the original observed roster order.
    saved_parts=row['prior_original_parts'];need(type(saved_parts) is list,'original part roster must be array')
    saved_pins=[canonical_sha256(p) for p in saved_parts];part_pins=[canonical_sha256(p) for p in parts]
    need(len(set(saved_pins))==len(saved_pins) and len(set(part_pins))==len(part_pins) and sorted(saved_pins)==sorted(part_pins),'saved original part roster differs')
    parts=deepcopy(saved_parts)
    codes=expected['joint_code_binding'];rp=row['render_policy']['record_sha256'];ci=canonical_sha256(meaning['candidate_inventory']);rt=sha256(text.encode()).hexdigest()
    view=canonical_sha256({'task':J.TASK,'profile':J.PROFILE,'codes':codes,'render_policy_sha256':rp,'source_identity':row['source_identity'],
        'processor_binding':row['processor_binding'],'rendered_utf8_sha256':rt,'input_ids':ids,'attention_mask':masks,'max_source_tokens':budget})
    provenance=canonical_sha256({'caller_sha256':expected['caller_sha256'],'meaning_sha256':expected['meaning_sha256'],'render_policy_sha256':rp,
        'code_binding_sha256':codes['record_sha256'],'candidate_inventory_sha256':ci,'claim_snapshot_sha256':meaning['claim_snapshot']['record_sha256'],
        'episode_snapshot_sha256':meaning['episode_snapshot']['record_sha256'],'task_snapshot_sha256':meaning['task_snapshot']['record_sha256'],
        'source_identity_sha256':row['source_identity']['record_sha256'],'original_part_inventory_sha256':canonical_sha256(parts)})
    rebuilt=seal({'schema':J.SCHEMA,'task':J.TASK,'profile':J.PROFILE,'code_binding':codes,'render_policy':row['render_policy'],
        'caller_sha256':expected['caller_sha256'],'meaning_sha256':expected['meaning_sha256'],'processor_binding':expected['processor_binding'],'source_identity':expected['source_identity'],
        'max_source_tokens':budget,'candidate_inventory_sha256':ci,'rendered_text':text,'rendered_utf8_sha256':rt,'input_ids':ids,'attention_mask':masks,
        'blocks':blocks,'fields':fields,'sources':sources,'prior_original_parts':parts,'candidates':candidates,'tokens':tokens,
        'cache':{'processed_view_sha256':view,'provenance_sha256':provenance},'text_attention':'causal_full_or_sliding_as_configured',
        'teacher_accessed':False,'native_locator_alignment':False,'learning_qualified':False,'authority_granted':False})
    need(row==rebuilt,'saved original/field/token/cache geometry differs');return deepcopy(row)

def freeze_input(caller,meaning,joint_alignment,batch,*,expected,split_row):
    validate_split(split_row);validate_saved_joint(caller,meaning,joint_alignment,batch,expected=expected)
    return seal({'schema':INPUT,'split_row':split_row,'caller':caller,'meaning':meaning,'joint':joint_alignment,'batch':batch,
        'expected_sha256':canonical_sha256(expected),'teacher_accessed':False,'current_service_authenticated':False,'training_authorized':False})

def validate_input(row,*,expected_freeze_sha256,expected):
    pin(row,expected_freeze_sha256,INPUT);exact(row,{'schema','split_row','caller','meaning','joint','batch','expected_sha256','teacher_accessed','current_service_authenticated','training_authorized','record_sha256'},'frozen input')
    validate_split(row['split_row']);need(row['expected_sha256']==canonical_sha256(expected) and row['teacher_accessed'] is False and row['current_service_authenticated'] is False and row['training_authorized'] is False,'input expectation/claim differs')
    validate_saved_joint(row['caller'],row['meaning'],row['joint'],row['batch'],expected=expected);return row

def original_json(raw):
    need(type(raw) is bytes,'target reader must return original bytes')
    def pairs(rows):
        out={}
        for key,value in rows:need(key not in out,'duplicate original JSON key');out[key]=value
        return out
    try:return json.loads(raw.decode('utf-8','strict'),object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))
    except (UnicodeError,ValueError) as exc:raise CognitiveKernelContractError('invalid original target JSON') from exc

def example(frozen,row):
    exact(row,{'schema','context','source_ids','proposals','dispositions','adjudications'},'standalone target')
    context,opened,_=J.caller_api.caller_input_from_record(frozen['caller'])
    need(row['schema']==L.TARGET_SCHEMA_V16 and row['context']==context.record() and row['source_ids']==[ref for ref,_ in opened],'target differs from independently frozen caller')
    bundle,status=L._target_from_record(context,{k:row[k] for k in ('schema','proposals','dispositions','adjudications')},case_id=frozen['split_row']['case_id'])
    result=L.FormationLearningExampleV16(frozen['split_row']['case_id'],context,opened,bundle,status,frozen['split_row']['split']);result.validate();return result

def target_core(frozen,raw):
    ex=example(frozen,original_json(raw));unknown=[name for name,state in ex.adjudications if state=='unknown']
    return {'schema':TARGET,'input_freeze_sha256':frozen['record_sha256'],'target_file_sha256':sha256(raw).hexdigest(),'original_target_hex':raw.hex(),
        'state':'non_supervised_unknown' if unknown else 'complete','unknown_dimensions':unknown,
        'canonical_output_sha256':None if unknown else sha256(canonical_json_bytes(L.supervised_output_record_v16(ex))).hexdigest(),
        **frozen['joint']['cache'],'author_authenticated':False,'training_authorized':False}

def admit_target(frozen,*,expected_freeze_sha256,expected,target_reader,expected_target_file_sha256):
    validate_input(frozen,expected_freeze_sha256=expected_freeze_sha256,expected=expected)
    need(frozen['split_row']['split'] in ('train','development'),'FINAL target remains unopened');require_sha256(expected_target_file_sha256,'external target FILE')
    raw=target_reader();need(type(raw) is bytes and sha256(raw).hexdigest()==expected_target_file_sha256,'original target FILE differs');return seal(target_core(frozen,raw))

def admitted(row,frozen,expected):
    pin(row,expected,TARGET);raw=bytes.fromhex(row['original_target_hex']);need(row==seal(target_core(frozen,raw)),'target original/state/ancestry differs');return example(frozen,original_json(raw))

def validate_splits(rows):
    need(type(rows) is list,'split roster must be array');[validate_split(r) for r in rows];ids=[r['case_id'] for r in rows];need(len(set(ids))==len(ids),'duplicate case')
    by_id={r['case_id']:r for r in rows};parents={i:i for i in ids};visiting=set();visited=set()
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    def union(a,b):parents[root(a)]=root(b)
    def walk(i):
        need(i not in visiting,'ancestry cycle')
        if i in visited:return
        visiting.add(i)
        for p in by_id[i]['lineage']['parent_case_ids']:need(p in by_id,'absent parent');walk(p);union(i,p)
        visiting.remove(i);visited.add(i)
    for i in ids:walk(i)
    seen={}
    for row in rows:
        line=row['lineage'];keys=[(k,line[k]) for k in ('world_id','case_family','world_author_family','generator_family','rendering_family','source_system_family')]+[('ancestry',v) for v in line['ancestry_group_ids']]
        for key in keys:
            if key in seen:union(row['case_id'],seen[key])
            else:seen[key]=row['case_id']
    groups={}
    for row in rows:groups.setdefault(root(row['case_id']),[]).append(row)
    need(all(len({r['split'] for r in group})==1 for group in groups.values()),'cross-split ancestry component');return {'component_count':len(groups),'groups':[[r['case_id'] for r in group] for group in groups.values()]}

def export_pair(frozen,accepted,pair,*,expected,expected_pair_sha256):
    from .formation_joint_target_alignment import validate_pair
    validate_pair(pair,frozen,accepted,expected=expected,expected_pair_sha256=expected_pair_sha256);need(frozen['split_row']['split'] in ('train','development'),'FINAL cannot be exported')
    return seal({'schema':'mfm-paired-supplied-packet-v1','split_row':frozen['split_row'],'input_freeze_sha256':frozen['record_sha256'],'target_record_sha256':accepted['record_sha256'],
        'pair_sha256':pair['record_sha256'],'batch':frozen['batch'],'decoder_inputs':pair['decoder_inputs'],'labels':pair['labels'],'fields':pair['fields'],'literal_segments':pair['literal_segments'],
        'source_cache':frozen['joint']['cache'],'training_authorized':False,'current_service_authenticated':False,'independent_teacher_qualified':False,'full_role_qualified':False})

