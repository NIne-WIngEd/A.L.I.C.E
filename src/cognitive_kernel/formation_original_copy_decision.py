"""Exact opened UTF8 copying beside the unchanged native reference consumer.

This input-only geometry join does not admit a private input profile, establish
source rights, prove semantic truth, or grant memory-write authority.
"""
from bisect import bisect_left, bisect_right
from copy import deepcopy
from hashlib import sha256
import math

from .canonical import canonical_json_bytes, canonical_sha256
from . import formation_reference_decision as references
from scripts.mfm import formation_source_alignment as source
from scripts.mfm import formation_target_alignment as geometry
from scripts.mfm import formation_joint_input as joint_geometry

SCHEMA = 'mfm-formation-decision-v1.6-caller-references-original-spans-v1'
INVENTORY_SCHEMA = 'mfm-original-copy-input-inventory-v1'
COPY_BINDING_FIELDS = {'reference_binding','joint_alignment_sha256','joint_schema','copy_inventory_sha256'}
INVENTORY_FIELDS = {'schema','reference_inventory','joint_alignment_sha256','joint_schema','source_identity',
                    'source_geometry','sources','record_sha256'}
_require = references._require


def _seal(core):
    return {**deepcopy(core), 'record_sha256': canonical_sha256(core)}


def _copy_binding(binding):
    _require(type(binding) is dict and set(binding)==COPY_BINDING_FIELDS, 'copy binding fields differ')
    references._binding(binding['reference_binding'])
    _require(binding['joint_schema'] in references.JOINT_SCHEMAS, 'copy original joint profile differs')
    for key in ('joint_alignment_sha256','copy_inventory_sha256'):
        from .canonical import require_sha256
        _require(require_sha256(binding[key],key)==binding[key], 'copy binding pin is noncanonical')
    return binding


def _inventory(inventory):
    _require(type(inventory) is dict and set(inventory)==INVENTORY_FIELDS and inventory['schema']==INVENTORY_SCHEMA,
             'copy inventory packet differs')
    references._pinned(inventory,inventory['record_sha256'])
    references._inventory(inventory['reference_inventory'])
    _require(inventory['joint_schema']==inventory['reference_inventory']['source_profile'], 'copy inventory profile differs')
    from .canonical import require_sha256
    _require(require_sha256(inventory['joint_alignment_sha256'],'original joint pin')==inventory['joint_alignment_sha256'],
             'copy original joint pin differs')
    identity=inventory['source_identity']
    _require(type(identity) is dict and type(identity.get('record_sha256')) is str,'copy source identity differs')
    joint_geometry._identity(identity,identity['record_sha256'])
    geometry_packet=inventory['source_geometry']
    _require(type(geometry_packet) is dict and set(geometry_packet)=={'token_count','attention_mask'} and
             type(geometry_packet['token_count']) is int and geometry_packet['token_count']>0 and
             type(geometry_packet['attention_mask']) is list and len(geometry_packet['attention_mask'])==geometry_packet['token_count'] and
             all(type(x) is int and x in (0,1) for x in geometry_packet['attention_mask']), 'copy source geometry packet differs')
    count=geometry_packet['token_count'];mask=geometry_packet['attention_mask'];seen=set()
    _require(type(inventory['sources']) is list,'copy source roster differs')
    for slot,s in enumerate(inventory['sources']):
        _require(type(s) is dict and set(s)=={'source_slot','source_ref','content_sha256','original_bytes','modality',
             'source_token_indices','boundaries'} and type(s['source_slot']) is int and s['source_slot']==slot and
             type(s['source_ref']) is str and bool(s['source_ref']) and s['source_ref'] not in seen and
             type(s['original_bytes']) is int and s['original_bytes']>=0 and s['modality'] in ('text','code','structured'),
             'copy source row fields/identity differ')
        seen.add(s['source_ref']);length=s['original_bytes'];require_sha256(s['content_sha256'],'opened content pin')
        rows=s['boundaries'];_require(type(rows) is list and bool(rows),'copy boundary roster missing')
        offsets=[];byte_values={};union=set()
        for k,b in enumerate(rows):
            _require(type(b) is dict and set(b)=={'byte_offset','left_token_indices','right_token_indices',
                'original_byte_window','coordinate_features'} and type(b['byte_offset']) is int and 0<=b['byte_offset']<=length,
                'copy boundary fields/byte coordinate differ')
            off=b['byte_offset'];offsets.append(off)
            for name,endpoint in (('left_token_indices',k==0),('right_token_indices',k==len(rows)-1)):
                indices=b[name]
                _require(type(indices) is list and all(type(i) is int and 0<=i<count and mask[i]==1 for i in indices) and
                     indices==sorted(set(indices)) and (not indices if endpoint else bool(indices)),
                     'copy adjacent actual token context differs')
                union.update(indices)
            window=b['original_byte_window'];features=b['coordinate_features']
            _require(type(window) is list and len(window)==8 and all(type(v) is int and 0<=v<=256 for v in window),
                     'copy original byte window differs')
            _require(type(features) is list and len(features)==8 and all(type(v) is float and math.isfinite(v) and 0<=v<=1 for v in features)
                and features[0]==off/max(1,length) and features[6:]==[float(k==0),float(k==len(rows)-1)],
                'copy coordinate features differ')
            for pos,value in zip(range(off-4,off+4),window):
                if 0<=pos<length:
                    _require(value<256 and (pos not in byte_values or byte_values[pos]==value),'copy byte windows disagree')
                    byte_values[pos]=value
                else:_require(value==256,'copy byte window padding differs')
        _require(offsets==sorted(set(offsets)) and offsets[0]==0 and offsets[-1]==length and len(byte_values)==length,
                 'copy complete original boundary/byte roster differs')
        raw=bytes(byte_values[i] for i in range(length))
        _require(sha256(raw).hexdigest()==s['content_sha256'],'copy byte windows content digest differs')
        try:text=raw.decode('utf-8','strict')
        except UnicodeDecodeError:_require(False,'copy original bytes are not strict UTF8')
        actual=[0];position=0
        for char in text:position+=len(char.encode());actual.append(position)
        _require(offsets==actual,'copy boundary roster differs from all actual UTF8 boundaries')
        for k,b in enumerate(rows):
            _require(b['coordinate_features'][4]==(offsets[k]-offsets[k-1])/4 if k else b['coordinate_features'][4]==0.,
                     'copy left adjacent UTF8 length differs')
            _require(b['coordinate_features'][5]==(offsets[k+1]-offsets[k])/4 if k+1<len(rows) else b['coordinate_features'][5]==0.,
                     'copy right adjacent UTF8 length differs')
        _require(type(s['source_token_indices']) is list and all(type(i) is int for i in s['source_token_indices']) and
                 s['source_token_indices']==sorted(union),'copy source actual context union differs')
    return inventory


def copy_binding(inventory):
    _inventory(inventory)
    return _copy_binding({'reference_binding':deepcopy(inventory['reference_inventory']['binding']),
        'joint_alignment_sha256':inventory['joint_alignment_sha256'],'joint_schema':inventory['joint_schema'],
        'copy_inventory_sha256':inventory['record_sha256']})


def _range(value, limit, label):
    _require(type(value) is list and len(value)==2 and all(type(x) is int for x in value)
             and 0<=value[0]<=value[1]<=limit, label+' geometry differs')
    return value


def _opened_geometry(joint, row, ref, raw, modality):
    """Rebuild only actual opened observation atoms; not a whole-profile reader."""
    _require(type(row) is dict and set(row)=={'ref_id','modality','content_sha256','original_bytes',
              'rendered_content_chars','characters','block_index'}, 'opened source geometry fields differ')
    rendered=joint['rendered_text']; index=row['block_index']
    _require(type(row['original_bytes']) is int and row['original_bytes']==len(raw) and type(row['characters']) is list,
             'opened original byte count or atoms differ')
    _range(row['rendered_content_chars'],len(rendered),'opened original content')
    for char in row['characters']:
        _require(type(char) is dict and set(char)=={'original_utf8','rendered_chars'},'opened original atom fields differ')
        _range(char['original_utf8'],len(raw),'opened original UTF8 atom')
        _range(char['rendered_chars'],len(rendered),'opened original rendered atom')
    _require(type(index) is int and 0<=index<len(joint['blocks']), 'opened source block differs')
    block=joint['blocks'][index]; a,b=_range(block['rendered_chars'],len(rendered),'opened block')
    actual={'section':'observation','source_ref':ref,'content':raw.decode('utf-8','strict')}
    text=canonical_json_bytes(actual).decode('utf-8')
    from scripts.mfm.train_v1_formation_specialist import _escaped_text
    quoted=_escaped_text(text)
    _require(block['section']=='observation' and block['json_string']==quoted and
             rendered[a:b]=='json_text='+quoted and block['block_utf8_sha256']==sha256(text.encode()).hexdigest(),
             'actual opened observation rendered bytes differ')
    atoms,end=source._character_map(text,a+len('json_text=')+1,ascii_source=True)
    g=geometry.canonical_field_geometry(actual,expected_target_sha256=sha256(text.encode()).hexdigest())
    _require(block['canonical_geometry_sha256']==g['record_sha256'], 'opened canonical block geometry differs')
    node=next(n for n in g['nodes'] if n['pointer']=='/content')
    chars=[{'original_utf8':x['value_utf8'], 'rendered_chars':joint_geometry._compose(atoms,*x['canonical_chars'],end)}
           for x in node['characters']]
    complete={'ref_id':ref,'modality':modality,'content_sha256':sha256(raw).hexdigest(),'original_bytes':len(raw),
        'rendered_content_chars':joint_geometry._compose(atoms,node['canonical_chars'][0]+1,node['canonical_chars'][1]-1,end),
        'characters':chars,'block_index':index}
    _require(row==complete, 'complete opened original character geometry differs')
    return chars


def bind_original_copy_inventory(caller,joint_alignment,*,caller_sha256,joint_alignment_sha256,expected_binding):
    reference=references.bind_reference_inventory(caller,joint_alignment,caller_sha256=caller_sha256,
        joint_alignment_sha256=joint_alignment_sha256,expected_binding=expected_binding)
    context,opened,_=references.caller_api.caller_input_from_record(caller)
    joint=joint_alignment; ids,masks,tokens=joint['input_ids'],joint['attention_mask'],joint['tokens']
    _require(bool(ids) and all(type(x) is int and x>=0 for x in ids) and
             all(type(x) is int and x in (0,1) for x in masks), 'copy actual complete source IDs/mask differ')
    rendered=joint['rendered_text']
    _require(type(rendered) is str and sha256(rendered.encode()).hexdigest()==joint['rendered_utf8_sha256'],
             'copy actual rendered input pin differs')
    special_ids=set(reference['processor_binding']['tokenizer_wrapper']['all_special_ids'])
    valid=[]
    for i,t in enumerate(tokens):
        _require(type(t) is dict and t['token_index']==i and type(t['token_index']) is int and
                 t['token_id']==ids[i] and type(t['token_id']) is int and t['attention_mask']==masks[i] and
                 type(t['attention_mask']) is int and type(t['special_mask']) is int and t['special_mask'] in (0,1),
                 'copy actual token identity or masks differ')
        a,b=_range(t['rendered_chars'],len(rendered),'actual token')
        if a<b and masks[i]==1 and t['special_mask']==0 and ids[i] not in special_ids: valid.append((a,b,i))
    valid.sort();starts=[];ends=[];maximum=0
    for a,b,_ in valid:
        starts.append(a);maximum=max(maximum,b);ends.append(maximum)
    def overlaps(span):
        a,b=span
        result=sorted(i for x,y,i in valid[bisect_right(ends,a):bisect_left(starts,b)] if y>a)
        _require(bool(result),'actual original atom has no active nonspecial contextual token overlap')
        return result
    def relative(indices,boundary):
        return 0.0 if not indices else sum(max(0.,min(1.,(boundary-tokens[i]['rendered_chars'][0])/
            (tokens[i]['rendered_chars'][1]-tokens[i]['rendered_chars'][0]))) for i in indices)/len(indices)
    _require(type(joint['sources']) is list and len(joint['sources'])==len(opened), 'copy complete opened source roster differs')
    sources=[]
    for slot,((ref,raw),row) in enumerate(zip(opened,joint['sources'])):
        chars=_opened_geometry(joint,row,ref,raw,context.base.evidence[slot].modality)
        supports=[overlaps(c['rendered_chars']) for c in chars]
        offsets=[0,*[c['original_utf8'][1] for c in chars]]; boundaries=[]
        start,end=row['rendered_content_chars']; all_indices=set()
        for k,offset in enumerate(offsets):
            left=[] if k==0 else supports[k-1];right=[] if k==len(chars) else supports[k]
            rb=chars[k]['rendered_chars'][0] if k<len(chars) else end
            all_indices.update(left);all_indices.update(right)
            boundaries.append({'byte_offset':offset,'left_token_indices':list(left),'right_token_indices':list(right),
                'original_byte_window':[raw[j] if 0<=j<len(raw) else 256 for j in range(offset-4,offset+4)],
                'coordinate_features':[float(offset/max(1,len(raw))),float((rb-start)/(end-start)) if end>start else 0.,
                    float(relative(left,rb)),float(relative(right,rb)),
                    float((chars[k-1]['original_utf8'][1]-chars[k-1]['original_utf8'][0])/4) if k else 0.,
                    float((chars[k]['original_utf8'][1]-chars[k]['original_utf8'][0])/4) if k<len(chars) else 0.,
                    float(k==0),float(k==len(chars))]})
        sources.append({'source_slot':slot,'source_ref':ref,'content_sha256':sha256(raw).hexdigest(),'original_bytes':len(raw),
            'modality':row['modality'],'source_token_indices':sorted(all_indices),'boundaries':boundaries})
    return _seal({'schema':INVENTORY_SCHEMA,'reference_inventory':reference,
        'joint_alignment_sha256':joint_alignment_sha256,'joint_schema':joint['schema'],
        'source_identity':deepcopy(joint['source_identity']),
        'source_geometry':{'token_count':len(ids),'attention_mask':list(masks)},'sources':sources})


def reference_requests(decision,inventory):
    clone=deepcopy(decision);clone['schema']=references.SCHEMA
    requests=references.reference_requests(clone,inventory['reference_inventory'])
    _require([{k:r[k] for k in ('pointer','candidate_path')} for r in requests]==_reference_shape_requests(decision),
             'complete mixed-wire reference request shape differs')
    return requests


def _reference_shape_requests(decision):
    """Pure complete-wire roles, without inventing caller eligibility or labels."""
    _part_requests(decision)
    rules=[r for r in references.fields.normative_policy()['rules'] if r['responsibility']=='external_selection']
    requests=[]
    def visit(value,path):
        matches=[r for r in rules if references.fields._match(path,r['path'])]
        rule=matches[0] if len(matches)==1 else None
        if type(value) is dict and 'caller_reference' in value:
            _require(set(value)=={'caller_reference'} and value['caller_reference'] is True and rule is not None,
                     'reference marker is malformed or has unsupported mixed-wire path')
            requests.append({'pointer':references._pointer(path),'candidate_path':rule['candidate_path']})
            return
        if rule is not None:
            if value is None:
                _require('null' in rule['kinds'],'mandatory mixed-wire reference cannot be null')
                return
            tags={'local_uncertainty'} if path[-1]=='uncertainty_ref' else \
                {'local_proposal','local_value','local_uncertainty'} if rule['candidate_path'].endswith(('.contradicts','.target_refs')) else set()
            _require(type(value) is dict and len(value)==1 and next(iter(value)) in tags and
                type(next(iter(value.values()))) is int and next(iter(value.values()))>0,
                'external mixed-wire reference must be an explicit marker or valid local reference')
            return
        if type(value) is dict:
            for key,child in value.items():visit(child,[*path,key])
        elif type(value) is list:
            for i,child in enumerate(value):visit(child,[*path,i])
    visit(decision,[])
    return sorted(requests,key=lambda r:r['pointer'])


def _part_requests(decision):
    _require(type(decision) is dict and set(decision)=={'schema','proposals','dispositions'} and
             decision['schema']==SCHEMA and type(decision['proposals']) is list and type(decision['dispositions']) is list,
             'unsupported complete copy decision')
    allowed=set();requests=[]
    for n,p in enumerate(decision['proposals']):
        _require(type(p) is dict and type(p.get('proposal')) is dict,'copy proposal structure differs')
        value=p['proposal'].get('value_text')
        _require(type(value) is dict and set(value)=={'parts'} and type(value['parts']) is list and bool(value['parts']),
                 'copy value must contain exact nonempty parts')
        semantic=False
        for k,part in enumerate(value['parts']):
            pointer=f'/proposals/{n}/proposal/value_text/parts/{k}'
            _require(type(part) is dict, 'copy part is malformed')
            if set(part)=={'semantic_text'}:
                _require(type(part['semantic_text']) is str and bool(part['semantic_text']) and not semantic,
                         'empty or adjacent semantic parts are noncanonical')
                part['semantic_text'].encode('utf-8','strict');semantic=True
            else:
                _require(set(part)=={'original_copy'} and part['original_copy'] is True,'copy marker is malformed')
                requests.append({'pointer':pointer});allowed.add(pointer);semantic=False
    def visit(value,path):
        if type(value) is dict:
            if 'original_copy' in value:
                _require(references._pointer(path) in allowed,'copy marker has unsupported path')
            for key,child in value.items():visit(child,[*path,key])
        elif type(value) is list:
            for i,child in enumerate(value):visit(child,[*path,i])
    visit(decision,[])
    return sorted(requests,key=lambda r:r['pointer'])


def copy_requests(decision,inventory):
    _inventory(inventory)
    requests=_part_requests(decision)
    reference_requests(decision,inventory)
    return requests


def compile_original_copy_decision(context,opened_sources,decision,copy_predictions,reference_predictions,*,caller,
        caller_sha256,joint_alignment,joint_alignment_sha256,expected_binding,artifact_sha256,inference_run_id):
    _copy_binding(expected_binding)
    inventory=bind_original_copy_inventory(caller,joint_alignment,caller_sha256=caller_sha256,
        joint_alignment_sha256=joint_alignment_sha256,expected_binding=expected_binding['reference_binding'])
    _require(copy_binding(inventory)==expected_binding,'copy consumer complete binding differs')
    supplied,opened,_=references.caller_api.caller_input_from_record(caller)
    _require(context.record()==supplied.record() and tuple(opened_sources)==opened,'copy actual context/opened bytes differ')
    requests={r['pointer'] for r in copy_requests(decision,inventory)}
    _require(type(copy_predictions) is dict and set(copy_predictions)=={'binding','rows'} and
             _copy_binding(copy_predictions['binding'])==expected_binding and type(copy_predictions['rows']) is list,
             'copy prediction binding or packet differs')
    chosen={}
    for row in copy_predictions['rows']:
        _require(type(row) is dict and set(row)=={'pointer','source_slot','start_byte','end_byte'} and
                 type(row['pointer']) is str and row['pointer'] in requests and row['pointer'] not in chosen,
                 'copy prediction set contains extra/duplicate/missing request')
        s,a,b=row['source_slot'],row['start_byte'],row['end_byte']
        _require(all(type(x) is int for x in (s,a,b)) and 0<=s<len(inventory['sources']), 'copy source/range malformed')
        sr=inventory['sources'][s];valid={r['byte_offset'] for r in sr['boundaries']}
        _require(a in valid and b in valid and a<b,'copy predicted range is not a nonempty original UTF8 span')
        chosen[row['pointer']]=row
    _require(set(chosen)==requests,'copy prediction set is incomplete')
    resolved=deepcopy(decision);provenance=[];sources=dict(opened)
    for n,p in enumerate(resolved['proposals']):
        pieces=[];position=0
        for k,part in enumerate(p['proposal']['value_text']['parts']):
            pointer=f'/proposals/{n}/proposal/value_text/parts/{k}'
            if 'semantic_text' in part:raw=part['semantic_text'].encode('utf-8','strict')
            else:
                r=chosen[pointer];sr=inventory['sources'][r['source_slot']];a,b=r['start_byte'],r['end_byte']
                raw=sources[sr['source_ref']][a:b];raw.decode('utf-8','strict')
                provenance.append({'pointer':pointer,'source_ref':sr['source_ref'],'content_sha256':sr['content_sha256'],
                    'original_utf8':[a,b],'value_utf8':[position,position+len(raw)],'literal_sha256':sha256(raw).hexdigest()})
            pieces.append(raw);position+=len(raw)
        p['proposal']['value_text']=b''.join(pieces).decode('utf-8','strict')
    resolved['schema']=references.SCHEMA
    bundle=references.compile_reference_decision(context,opened_sources,resolved,reference_predictions,
        caller=caller,caller_sha256=caller_sha256,joint_alignment=joint_alignment,joint_alignment_sha256=joint_alignment_sha256,
        expected_binding=expected_binding['reference_binding'],artifact_sha256=artifact_sha256,inference_run_id=inference_run_id)
    for row in provenance:
        n=int(row['pointer'].split('/')[2]);proposal=bundle.proposals[n].base;a,b=row['original_utf8']
        _require(row['source_ref'] in proposal.evidence_refs and any(anchor.ref_id==row['source_ref'] and
            anchor.locator is None and type(anchor.start_byte) is int and type(anchor.end_byte) is int and
            anchor.start_byte<=a<b<=anchor.end_byte for anchor in proposal.anchors),
            'copy provenance source is not actually cited and covered by its owning proposal anchor')
    return {'bundle':bundle,'copy_provenance':provenance}
