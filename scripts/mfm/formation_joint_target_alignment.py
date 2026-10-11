"""Whole target geometry over an externally frozen supplied J4 input.

Selection/literal/semantic annotations are explicit fixture or author inputs;
the geometry does not infer teacher truth or current read/write authority.
"""
from copy import deepcopy
from cognitive_kernel.canonical import canonical_sha256
from cognitive_kernel import formation_learning_v16 as L
from .formation_paired_teaching import need,exact,seal,pin,validate_input,admitted
from . import formation_joint_input_v4 as J,formation_source_alignment as S
from . import formation_target_alignment as T,formation_field_responsibility as F
from . import train_v16_formation_specialist as V

PAIR='mfm-paired-supplied-joint-target-geometry-v1'

def field_rows(geometry,frozen,annotations):
    policy=F.normative_policy();candidates=frozen['meaning']['candidate_inventory'];rows=[];need(type(annotations) is list,'annotations must be array')
    for node in geometry['nodes']:
        rules=[r for r in policy['rules'] if F._match(node['path'],r['path'])];need(len(rules)==1 and node['kind'] in rules[0]['kinds'],'unknown normative path/type')
        rule=rules[0];eligible=[];matching=[]
        if rule['candidate_path'] is not None and node['kind']=='string':
            eligible=[i for i,c in enumerate(candidates) if rule['candidate_path'] in c['eligible_paths']];matching=[i for i in eligible if candidates[i]['ref_id']==node['value']]
        rows.append({'pointer':node['pointer'],'kind':node['kind'],'responsibility':rule['responsibility'],'candidate_path':rule['candidate_path'],
            'eligible_candidate_indices':eligible,'matching_candidate_indices':matching,'candidate_status':'not_applicable' if rule['candidate_path'] is None or node['kind']=='null' else 'missing' if not matching else 'ambiguous' if len(matching)>1 else 'unique','annotation_status':'unannotated'})
    by_pointer={r['pointer']:r for r in rows};seen=set()
    for entry in annotations:
        need(type(entry) is dict and entry.get('pointer') in by_pointer and entry['pointer'] not in seen,'invalid authored pointer');seen.add(entry['pointer']);row=by_pointer[entry['pointer']]
        need(entry.get('mode')==row['responsibility'],'annotation cannot reclassify field')
        if entry['mode']=='external_selection':
            exact(entry,{'pointer','mode','candidate_ordinal'},'selection annotation');need(type(entry['candidate_ordinal']) is int and entry['candidate_ordinal'] in row['matching_candidate_indices'],'authored ordinal is not eligible exact identity');row['selected_candidate_ordinal']=entry['candidate_ordinal']
        else:exact(entry,{'pointer','mode'},'semantic annotation');need(entry['mode']=='semantic_authoring','unsupported annotation mode')
        row['annotation_status']='authored'
    return rows

def literal_rows(geometry,tokens,frozen,declarations):
    need(type(declarations) is list,'literal declarations must be array');nodes={n['pointer']:n for n in geometry['nodes']};seen=set();out=[]
    _,opened,_=J.caller_api.caller_input_from_record(frozen['caller']);sources={r['ref_id']:r for r in frozen['joint']['sources']}
    for entry in declarations:
        exact(entry,{'pointer','target_value_utf8','source_occurrences'},'literal declaration');node=nodes.get(entry['pointer'])
        need(node is not None and len(node['path'])==4 and node['path'][0]=='proposals' and node['path'][2:]==['proposal','value_text'],'only explicit value_text literals supported')
        span=entry['target_value_utf8'];need(type(span) is list and len(span)==2 and all(type(x) is int for x in span),'invalid literal span')
        raw=node['value'].encode();boundaries={0,len(raw),*(a['value_utf8'][0] for a in node['characters'])};need(0<=span[0]<span[1]<=len(raw) and all(x in boundaries for x in span),'literal splits original UTF8 atom')
        key=(entry['pointer'],*span);need(key not in seen,'duplicate literal');seen.add(key);needle=raw[slice(*span)];occurrences=[]
        for ref,data in opened:
            sr=sources[ref];bounds={a['original_utf8'][0]:a['rendered_chars'][0] for a in sr['characters']};bounds[len(data)]=sr['rendered_content_chars'][1];start=data.find(needle)
            while start>=0:
                end=start+len(needle);need(start in bounds and end in bounds,'source literal splits UTF8 atom');a,b=bounds[start],bounds[end]
                indices=[t['token_index'] for t in frozen['joint']['tokens'] if t['source_ref']==ref and t['observed_source_eligible'] and t['rendered_chars'][0]<b and t['rendered_chars'][1]>a]
                occurrences.append({'source_ref':ref,'original_utf8':[start,end],'rendered_chars':[a,b],'supporting_token_indices':indices});start=data.find(needle,start+1)
        need(bool(occurrences),'declared literal has no original observation alternative');selected=entry['source_occurrences']
        if selected is not None:
            need(type(selected) is list and bool(selected) and len({canonical_sha256(r) for r in selected})==len(selected),'invalid selected occurrences')
            for choice in selected:
                exact(choice,{'source_ref','original_utf8'},'selected occurrence')
                need(type(choice['source_ref']) is str and type(choice['original_utf8']) is list and len(choice['original_utf8'])==2 and all(type(v) is int for v in choice['original_utf8']),'selected occurrence requires exact integer coordinates')
                need(any(choice=={'source_ref':o['source_ref'],'original_utf8':o['original_utf8']} for o in occurrences),'authored occurrence is not an original alternative')
        atoms=[a for a in node['characters'] if span[0]<=a['value_utf8'][0]<a['value_utf8'][1]<=span[1]];a,b=atoms[0]['canonical_chars'][0],atoms[-1]['canonical_chars'][1]
        indices=[t['body_index'] for t in tokens if not t['special_mask'] and t['canonical_chars'][0]<b and t['canonical_chars'][1]>a]
        out.append({'pointer':entry['pointer'],'target_value_utf8':span,'literal_utf8_hex':needle.hex(),'canonical_chars':[a,b],'target_body_indices':indices,
            'occurrences':occurrences,'selected_occurrences':selected,'token_id_copy_claimed':False})
    return out

def require_target_parent(joint_alignment,target_alignment):
    """Metadata comparison only; no conversion of old source234 to J4."""
    pin(joint_alignment,joint_alignment['record_sha256'],J.SCHEMA)
    T.validate_target_alignment_record(target_alignment,target_alignment_sha256=target_alignment['record_sha256'])
    need(target_alignment['caller_sha256']==joint_alignment['caller_sha256'] and target_alignment['processor_binding']==joint_alignment['processor_binding'],'actual source/target parents differ')

def pair_core(frozen,accepted,geometry,inputs,labels,tokens,max_target_tokens,annotations,declarations):
    return {'schema':PAIR,'input_freeze_sha256':frozen['record_sha256'],'target_record_sha256':accepted['record_sha256'],'target_file_sha256':accepted['target_file_sha256'],
        'canonical_output_sha256':accepted['canonical_output_sha256'],'target_code_binding':T.target_code_binding(),'processor_binding':frozen['joint']['processor_binding'],
        'max_target_tokens':max_target_tokens,'geometry':geometry,'decoder_inputs':inputs,'labels':labels,'tokens':tokens,
        'scaffolding':{'bos':{'decoder_input_index':0,'token_id':inputs[0]},'eos':{'label_index':len(tokens),'query_index':len(tokens),'token_id':labels[-1]}},
        'policy':F.normative_policy(),'fields':field_rows(geometry,frozen,annotations),'annotations':annotations,'literal_declarations':declarations,
        'literal_segments':literal_rows(geometry,tokens,frozen,declarations),'learning_qualified':False,'authority_granted':False,'processor_reexecuted':True}

def align_target(processor,frozen,accepted,*,expected,expected_freeze_sha256,expected_target_record_sha256,max_target_tokens,annotations=None,literal_declarations=None):
    validate_input(frozen,expected_freeze_sha256=expected_freeze_sha256,expected=expected);ex=admitted(accepted,frozen,expected_target_record_sha256)
    need(accepted['state']=='complete','unknown adjudication cannot generate labels');need(type(max_target_tokens) is int and max_target_tokens>0,'explicit complete target budget required')
    binding=S.processor_binding(processor);need(binding==expected['processor_binding'],'actual target processor differs')
    geometry=T.canonical_field_geometry(L.supervised_output_record_v16(ex),expected_target_sha256=accepted['canonical_output_sha256']);text=bytes.fromhex(geometry['canonical_utf8_hex']).decode()
    try:
        with S._processor_execution_settings(processor):
            inputs,labels=V._target_ids(processor.tokenizer,ex,max_target_tokens)
            observed=processor.tokenizer(text,add_special_tokens=False,padding=False,truncation=False,return_offsets_mapping=True,return_attention_mask=True,return_special_tokens_mask=True)
            tokens=T._token_rows(geometry,observed['input_ids'],observed['attention_mask'],observed['offset_mapping'],observed['special_tokens_mask']);need(observed['input_ids']==inputs[1:]==labels[:-1],'complete target IDs differ from unchanged v16')
    finally:need(S.processor_binding(processor)==binding,'processor semantic binding changed')
    return seal(pair_core(frozen,accepted,geometry,inputs,labels,tokens,max_target_tokens,deepcopy([] if annotations is None else annotations),deepcopy([] if literal_declarations is None else literal_declarations)))

def validate_pair(pair,frozen,accepted,*,expected,expected_pair_sha256):
    validate_input(frozen,expected_freeze_sha256=frozen['record_sha256'],expected=expected);ex=admitted(accepted,frozen,accepted['record_sha256']);pin(pair,expected_pair_sha256,PAIR);need(accepted['state']=='complete','unknown cannot have paired labels')
    geometry=T.canonical_field_geometry(L.supervised_output_record_v16(ex),expected_target_sha256=accepted['canonical_output_sha256']);rows=pair['tokens'];need(type(rows) is list,'target tokens must be array')
    rebuilt=T._token_rows(geometry,[r['token_id'] for r in rows],[r['attention_mask'] for r in rows],[r['canonical_chars'] for r in rows],[r['special_mask'] for r in rows])
    inputs,labels,budget=pair['decoder_inputs'],pair['labels'],pair['max_target_tokens'];need(type(budget) is int and budget>0 and type(inputs) is list and type(labels) is list and
        len(inputs)==len(labels)==len(rows)+1<=budget and all(type(x) is int and x>=0 for x in inputs+labels) and rows==rebuilt and inputs[1:]==labels[:-1]==[r['token_id'] for r in rows],'whole decoder shift differs')
    binding=expected['processor_binding']['tokenizer_wrapper'];need(inputs[0]==binding['bos_token_id'] and labels[-1]==binding['eos_token_id'],'BOS/EOS differs')
    need(pair==seal(pair_core(frozen,accepted,geometry,inputs,labels,rows,budget,pair['annotations'],pair['literal_declarations'])),'whole paired ancestry/geometry differs');return deepcopy(pair)

