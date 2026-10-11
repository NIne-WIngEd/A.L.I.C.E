"""Complete semantic originals with exact custody kept in the validated companion.

This separate input profile does not acquire memory or change learning/authority.
Only extra proof-roster and original-parts verification projections leave text.
"""
from base64 import b64decode
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from cognitive_kernel.canonical import canonical_json_bytes,canonical_sha256
from . import formation_joint_input_v3 as previous,formation_caller_input as caller_api,formation_caller_meaning_v3 as meaning_api
from . import formation_source_alignment as source,formation_target_alignment as geometry_api,train_v1_formation_specialist as shared

SCHEMA='mfm-joint-caller-input-alignment-v4'
TASK='formation-complete-semantic-caller-input-v4'
PROFILE='text-causal-complete-semantic-companion-v1'
POLICY_SCHEMA='mfm-complete-semantic-joint-render-policy-v1'
_require=previous._require
_seal=previous._seal
_identity=previous._identity
_compose=previous._compose
_joins=previous._joins


def normative_joint_render_policy_v4():
    """Closed code-owned classification; caller/teacher cannot choose omissions."""
    return _seal({'schema':POLICY_SCHEMA,'task':TASK,'profile':PROFILE,'code_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'complete_meaning_policy':meaning_api.normative_view_policy_v3(),
        'previous_joint_binding':previous.joint_code_binding_v3(),
        'proof_projection':'exact_schema_record_cut_world_count_ordered_artifact_roster_sha256',
        'candidate_original_parts':'validated_companion_inventory_only',
        'semantic_originals':'complete_original_content_metadata_and_all_authorized_parts',
        'historical_roles':'preserved_not_current_source_or_selection_authority',
        'order':['context_snapshots','observations','candidate_meanings','candidate_index'],
        'teacher_accessed':False,'authority_granted':False})


def _policy(row,expected):
    source._pinned_record(row,expected,POLICY_SCHEMA)
    _require(row==normative_joint_render_policy_v4(),'normative complete semantic render policy differs')


def joint_code_binding_v4():
    return _seal({'schema':'mfm-joint-input-code-binding-v4','task':TASK,'profile':PROFILE,
        'render_policy':normative_joint_render_policy_v4(),'authority_granted':False})


def _proof_descriptor(snapshot):
    row=deepcopy(snapshot);proof=row['historical_prefix_proof']
    artifacts=[deepcopy(p['artifact']) for p in proof['original_artifacts']]
    row['historical_prefix_proof']={'schema':proof['schema'],'record_sha256':proof['record_sha256'],
        'cut_sha256':proof['cut']['record_sha256'],'world_sha256':proof['world']['world_sha256'],
        'artifact_count':len(artifacts),'artifact_roster_sha256':canonical_sha256(artifacts)}
    return row


def _part_inventory(semantic):
    """Map approved original parts into complete semantic body, before projection."""
    inventories={}
    for row in semantic:
        custody=row.get('original_parts');parts=[]
        if type(custody) is dict:
            for name,part in custody.items():
                parts.append({'pointer':'/content/'+previous._pointer(name),'family':'episode','entity_id':row['candidate']['ref_id'],
                    'content_version':None,'part_name':name,'original':deepcopy(part)})
        elif type(custody) is list:
            _require(len(custody)==len(row['content']['contents']),'complete semantic original inventory differs')
            for i,(owner,item) in enumerate(zip(custody,row['content']['contents'])):
                if item['availability']!='present':continue
                for name,part in owner['parts'].items():
                    if name=='content-metadata':pointer='/content/contents/'+str(i)+'/record'
                    elif name in item['parts']:pointer='/content/contents/'+str(i)+'/parts/'+previous._pointer(name)
                    else:continue # result opaque bytes have custody, never fabricated semantic text
                    parts.append({'pointer':pointer,**{k:item[k] for k in ('family','entity_id','content_version')},
                        'part_name':name,'original':deepcopy(part)})
        inventories[row['slot']]=parts
    return inventories


def _render(caller,meaning,caller_sha256,meaning_sha256,render_policy,render_policy_sha256):
    _policy(render_policy,render_policy_sha256)
    context,opened,registry=caller_api.caller_input_from_record(caller)
    _require(all(r.modality in ('text','code','structured') for r in context.base.evidence),'native evidence requires complete native caller adapter')
    semantic=meaning_api.semantic_view_v3(meaning,caller,meaning_sha256=meaning_sha256,caller_sha256=caller_sha256)
    inventories=_part_inventory(semantic)
    # Only the separately generated verification projection is absent from text.
    values=[{'section':'task_context_snapshot','task':TASK,'profile':PROFILE,'context':context.record(),
        'claim_snapshot':deepcopy(meaning['claim_snapshot']),'episode_snapshot':_proof_descriptor(meaning['episode_snapshot']),
        'task_snapshot':_proof_descriptor(meaning['task_snapshot'])}]
    values.extend({'section':'observation','source_ref':ref,'content':raw.decode('utf-8','strict')} for ref,raw in opened)
    values.extend({'section':'candidate_meaning',**{k:deepcopy(v) for k,v in row.items() if k!='original_parts'}} for row in semantic)
    values.append({'section':'candidate_index','candidates':[{'slot':slot,'ref_id':r['ref_id'],'kind':r['kind'],'origin':r['origin'],
        'namespace':r['namespace'],'eligible_paths':deepcopy(r['eligible_paths'])} for slot,r in enumerate(meaning['candidate_inventory'])]})
    blocks=[];fields=[];originals=[];prior_parts=[];position=0
    for block_index,value in enumerate(values):
        raw=canonical_json_bytes(value);text=raw.decode('utf-8');quoted=shared._escaped_text(text);block_text='json_text='+quoted
        atoms,end=source._character_map(text,position+len('json_text=')+1,ascii_source=True)
        geometry=geometry_api.canonical_field_geometry(value,expected_target_sha256=sha256(raw).hexdigest())
        nodes={n['pointer']:n for n in geometry['nodes']};slot=value.get('slot');content=nodes.get('/content');body_pin=value.get('body_file_sha256')
        if body_pin is not None:
            _require(content is not None and sha256(raw[slice(*content['canonical_utf8'])]).hexdigest()==body_pin,'complete semantic body differs')
        parts=[]
        for inventory in inventories.get(slot,[]):
            node=nodes.get(inventory['pointer']);part=inventory['original'];ref=part['artifact'];payload=b64decode(part['base64'],validate=True)
            _require(node is not None and len(payload)==ref['bytes'] and sha256(payload).hexdigest()==ref['file_sha256'],'complete original descriptor differs')
            if ref['media_type']=='application/json':_require(raw[slice(*node['canonical_utf8'])]==payload,'canonical semantic original differs')
            else:_require(node['kind']=='string' and node['value'].encode('utf-8')==payload,'exact semantic original text differs')
            characters=[] if ref['media_type']=='application/json' else [{'original_utf8':a['value_utf8'],
                'rendered_chars':_compose(atoms,*a['canonical_chars'],end)} for a in node['characters']]
            original={k:deepcopy(inventory[k]) for k in ('family','entity_id','content_version','part_name','pointer')}
            original.update(candidate_slot=slot,artifact_id=ref['artifact_id'],bytes=ref['bytes'],file_sha256=ref['file_sha256'],media_type=ref['media_type'],
                original_base64=part['base64'],rendered_chars=_compose(atoms,*node['canonical_chars'],end),characters=characters)
            prior_parts.append(original);parts.append((node,inventory,ref,len(payload)))
        spans=[*geometry['lexemes'],*({**n,'role':'container'} for n in geometry['nodes'] if n['kind'] in ('object','array'))]
        for span in spans:
            node=nodes[span['pointer']]
            field={'block_index':block_index,'section':value['section'],'candidate_slot':slot,'pointer':span['pointer'],'role':span['role'],
                'kind':'string' if span['role']=='key' else node['kind'],'canonical_chars':span['canonical_chars'],'canonical_utf8':span['canonical_utf8'],
                'rendered_chars':_compose(atoms,*span['canonical_chars'],end),'body_file_sha256':None,'body_original_utf8':None,
                'original_part':None,'original_part_file_sha256':None,'original_part_utf8':None,'original_family':None,'original_entity_id':None,'original_content_version':None}
            if span['role']=='value':
                field['value']=deepcopy(node['value'])
                if node['kind']=='string':field['characters']=[{'value_utf8':a['value_utf8'],
                    'rendered_chars':_compose(atoms,*a['canonical_chars'],end)} for a in node['characters']]
            if body_pin is not None and span['pointer'].startswith('/content') and content['canonical_utf8'][0]<=span['canonical_utf8'][0]<=span['canonical_utf8'][1]<=content['canonical_utf8'][1]:
                field.update(body_file_sha256=body_pin,body_original_utf8=[x-content['canonical_utf8'][0] for x in span['canonical_utf8']])
            for part,inventory,ref,length in parts:
                pointer=inventory['pointer']
                if (span['pointer']==pointer or span['pointer'].startswith(pointer+'/')) and part['canonical_utf8'][0]<=span['canonical_utf8'][0]<=span['canonical_utf8'][1]<=part['canonical_utf8'][1]:
                    offsets=[x-part['canonical_utf8'][0] for x in span['canonical_utf8']] if ref['media_type']=='application/json' else [0,length]
                    field.update(original_part=inventory['part_name'],original_part_file_sha256=ref['file_sha256'],original_part_utf8=offsets,
                        original_family=inventory['family'],original_entity_id=inventory['entity_id'],original_content_version=inventory['content_version'])
            fields.append(field)
        if value['section']=='observation':
            source_index=block_index-1;ref,payload=opened[source_index]
            _require(content is not None and content['kind']=='string','observed original not a string')
            characters=[{'original_utf8':a['value_utf8'],'rendered_chars':_compose(atoms,*a['canonical_chars'],end)} for a in content['characters']]
            originals.append({'ref_id':ref,'modality':context.base.evidence[source_index].modality,'content_sha256':sha256(payload).hexdigest(),
                'original_bytes':len(payload),'rendered_content_chars':_compose(atoms,content['canonical_chars'][0]+1,content['canonical_chars'][1]-1,end),
                'characters':characters,'block_index':block_index})
        blocks.append({'section':value['section'],'candidate_slot':slot,'json_string':quoted,'block_utf8_sha256':sha256(raw).hexdigest(),
            'rendered_chars':[position,position+len(block_text)],'canonical_geometry_sha256':geometry['record_sha256']})
        position+=len(block_text)+1
    messages=[{'role':'user','content':[{'type':'text','text':'json_text='+b['json_string']} for b in blocks]}]
    return messages,''.join('json_text='+b['json_string']+'\n' for b in blocks),blocks,fields,originals,prior_parts


def joint_source_batch_with_alignment_v4(processor,caller,meaning,*,caller_sha256,meaning_sha256,render_policy,render_policy_sha256,
        processor_binding,source_identity,source_identity_sha256,max_source_tokens):
    """Full default processor input/offset observation; complete capacity refusal."""
    _policy(render_policy,render_policy_sha256)
    meaning_api.caller_meaning_v3_from_record(meaning,caller,meaning_sha256=meaning_sha256,caller_sha256=caller_sha256)
    _identity(source_identity,source_identity_sha256)
    _require(type(max_source_tokens) is int and max_source_tokens>0,'explicit complete source budget required')
    binding=source.processor_binding(processor);_require(binding==processor_binding,'external semantic processor binding differs')
    messages,assembled,blocks,fields,originals,prior_parts=_render(caller,meaning,caller_sha256,meaning_sha256,render_policy,render_policy_sha256)
    rendered=processor.apply_chat_template(messages,chat_template=shared.SOURCE_TEMPLATE,tokenize=False,add_generation_prompt=False)
    _require(type(rendered) is str and rendered==assembled,'actual complete semantic template differs')
    with source._processor_execution_settings(processor):
        batch=processor.apply_chat_template(messages,chat_template=shared.SOURCE_TEMPLATE,tokenize=True,add_generation_prompt=False,
            return_dict=True,return_tensors='pt',processor_kwargs={'do_sample_frames':False})
        _require(all(k in batch and hasattr(batch[k],'tolist') and len(batch[k].shape)==2 and batch[k].shape[0]==1
            for k in ('input_ids','attention_mask')),'complete processor batch missing')
        ids,masks=batch['input_ids'][0].tolist(),batch['attention_mask'][0].tolist()
        _require(0<len(ids)<=max_source_tokens and len(ids)==len(masks),'complete source budget overflow or shape differs')
        encoded=processor.apply_chat_template(messages,chat_template=shared.SOURCE_TEMPLATE,tokenize=True,add_generation_prompt=False,
            return_dict=True,return_tensors='pt',processor_kwargs={'do_sample_frames':False,'return_offsets_mapping':True,
                'return_special_tokens_mask':True,'return_attention_mask':True})
        shapes={'input_ids':(1,len(ids)),'attention_mask':(1,len(ids)),'offset_mapping':(1,len(ids),2),'special_tokens_mask':(1,len(ids))}
        _require(all(k in encoded and hasattr(encoded[k],'tolist') and tuple(encoded[k].shape)==shape for k,shape in shapes.items()),'complete semantic processor geometry malformed')
        _require(encoded['input_ids'][0].tolist()==ids and encoded['attention_mask'][0].tolist()==masks,'complete IDs/masks differ from offset observation')
    _require(source.processor_binding(processor)==binding,'semantic processor binding changed')
    tokens,candidates=_joins(rendered,originals,ids,masks,encoded['offset_mapping'][0].tolist(),encoded['special_tokens_mask'][0].tolist(),
        set(processor.tokenizer.all_special_ids),fields,blocks,meaning['candidate_inventory'])
    codes=joint_code_binding_v4()
    view=canonical_sha256({'task':TASK,'profile':PROFILE,'codes':codes,'render_policy_sha256':render_policy_sha256,
        'source_identity':source_identity,'processor_binding':binding,'rendered_utf8_sha256':sha256(rendered.encode()).hexdigest(),
        'input_ids':ids,'attention_mask':masks,'max_source_tokens':max_source_tokens})
    provenance=canonical_sha256({'caller_sha256':caller_sha256,'meaning_sha256':meaning_sha256,
        'render_policy_sha256':render_policy_sha256,'code_binding_sha256':codes['record_sha256'],
        'candidate_inventory_sha256':canonical_sha256(meaning['candidate_inventory']),
        'claim_snapshot_sha256':meaning['claim_snapshot']['record_sha256'],'episode_snapshot_sha256':meaning['episode_snapshot']['record_sha256'],
        'task_snapshot_sha256':meaning['task_snapshot']['record_sha256'],'source_identity_sha256':source_identity_sha256,
        'original_part_inventory_sha256':canonical_sha256(prior_parts)})
    return batch,_seal({'schema':SCHEMA,'task':TASK,'profile':PROFILE,'code_binding':codes,'render_policy':deepcopy(render_policy),
        'caller_sha256':caller_sha256,'meaning_sha256':meaning_sha256,'processor_binding':binding,'source_identity':deepcopy(source_identity),
        'max_source_tokens':max_source_tokens,'candidate_inventory_sha256':canonical_sha256(meaning['candidate_inventory']),
        'rendered_text':rendered,'rendered_utf8_sha256':sha256(rendered.encode()).hexdigest(),'input_ids':ids,'attention_mask':masks,
        'blocks':blocks,'fields':fields,'sources':originals,'prior_original_parts':prior_parts,'candidates':candidates,'tokens':tokens,
        'cache':{'processed_view_sha256':view,'provenance_sha256':provenance},
        'text_attention':'causal_full_or_sliding_as_configured','teacher_accessed':False,'native_locator_alignment':False,
        'learning_qualified':False,'authority_granted':False})


def verify_joint_alignment_v4(processor,caller,meaning,sidecar,*,alignment_sha256,**kwargs):
    source._pinned_record(sidecar,alignment_sha256,SCHEMA)
    _,actual=joint_source_batch_with_alignment_v4(processor,caller,meaning,**kwargs)
    _require(actual==sidecar,'actual complete semantic alignment differs')
    return actual
