"""Explicit original target partitions and one actual mixed-wire geometry.

Original profile admission/readers remain upstream; private J5 is not admitted
by the public J4 reader here. No undeclared occurrence becomes a teacher label.
"""
from copy import deepcopy
from hashlib import sha256

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel import formation_original_copy_decision as copies
from cognitive_kernel import formation_reference_decision as references
from . import formation_reference_target_alignment as reference_alignment
from . import formation_joint_target_alignment as original
from . import formation_source_alignment as source

VIEW_SCHEMA='mfm-original-copy-target-view-v1'
ALIGNMENT_SCHEMA='mfm-original-copy-decision-token-alignment-v1'
VIEW_FIELDS={'schema','decision','binding','copy_binding','source_profile','processor_binding','original_pair_sha256',
    'original_frozen_input_sha256','original_target_sha256','reference_leaves','authored_ordinals','copy_leaves',
    'authored_copy_tuples','training_authorized','learning_qualified','authority_granted','record_sha256'}
need=references._require


def _seal(row): return {**deepcopy(row),'record_sha256':canonical_sha256(row)}


def original_copy_decision_view(example,complete_pair,frozen_input,*,pair_sha256,frozen_input_sha256,inventory,
                                local_uncertainty_refs=()):
    copies._inventory(inventory)
    _,joint=reference_alignment.source_parts(frozen_input)
    need(joint['record_sha256']==inventory['joint_alignment_sha256'], 'copy actual original joint pin differs')
    references._pinned(joint,inventory['joint_alignment_sha256'])
    ref=reference_alignment.reference_decision_view(example,complete_pair,frozen_input,pair_sha256=pair_sha256,
        frozen_input_sha256=frozen_input_sha256,inventory=inventory['reference_inventory'],
        local_uncertainty_refs=local_uncertainty_refs)
    if frozen_input['schema'] in (reference_alignment.j6_target.paired.INPUT,
                                  reference_alignment.broader_target.paired.INPUT):
        choices=(reference_alignment.broader_choices if frozen_input['schema']==
          reference_alignment.broader_target.paired.INPUT else reference_alignment.j6_choices)
        declarations=choices(complete_pair,frozen_input)['literal_declarations']
        literals=reference_alignment.j6_target.literal_rows(
            complete_pair['geometry'],complete_pair['tokens'],frozen_input,declarations)
    else:
        declarations=complete_pair.get('literal_declarations')
        # The original J4/P1 and private J5/P2 pure geometry remains unchanged.
        literals=original.literal_rows(complete_pair['geometry'],complete_pair['tokens'],frozen_input,declarations)
    need(complete_pair.get('literal_segments')==literals,'original complete literal declarations/geometry differ')
    decision=deepcopy(ref['decision']);decision['schema']=copies.SCHEMA
    grouped={}
    for row in literals:grouped.setdefault(row['pointer'],[]).append(row)
    opened=dict(example.opened_sources);slots={s['source_ref']:s['source_slot'] for s in inventory['sources']}
    authored=[]
    for n,p in enumerate(decision['proposals']):
        pointer=f'/proposals/{n}/proposal/value_text';raw=p['proposal']['value_text'].encode('utf-8','strict')
        parts=[];position=0
        for row in sorted(grouped.get(pointer,[]),key=lambda r:r['target_value_utf8']):
            a,b=row['target_value_utf8'];need(a>=position,'overlapping original target copy declarations are unresolved')
            if position<a:parts.append({'semantic_text':raw[position:a].decode('utf-8','strict')})
            k=len(parts);parts.append({'original_copy':True});part_pointer=pointer+f'/parts/{k}'
            selected=row['selected_occurrences'];alternatives=None
            if selected is not None:
                alternatives=[]
                for choice in selected:
                    source_ref=choice['source_ref'];x,y=choice['original_utf8']
                    need(source_ref in slots,'authored copy occurrence is not an actual opened source')
                    sr=inventory['sources'][slots[source_ref]];bounds={r['byte_offset'] for r in sr['boundaries']}
                    need(x in bounds and y in bounds and x<y and opened[source_ref][x:y]==raw[a:b],
                         'authored copy tuple differs from actual target/opened original bytes')
                    alternatives.append({'source_slot':slots[source_ref],'start_byte':x,'end_byte':y})
                need(bool(alternatives) and len({canonical_sha256(t) for t in alternatives})==len(alternatives),
                     'authored complete copy tuple alternatives differ')
            authored.append({'pointer':part_pointer,'alternatives':alternatives});position=b
        if position<len(raw):parts.append({'semantic_text':raw[position:].decode('utf-8','strict')})
        need(bool(parts),'empty original value cannot become complete copy parts')
        p['proposal']['value_text']={'parts':parts}
    leaves=copies.copy_requests(decision,inventory)
    need({r['pointer'] for r in leaves}=={r['pointer'] for r in authored},'complete authored copy partition differs')
    return _seal({**{k:deepcopy(v) for k,v in ref.items() if k not in {'schema','record_sha256','decision'}},
        'schema':VIEW_SCHEMA,'decision':decision,'copy_binding':copies.copy_binding(inventory),'copy_leaves':leaves,
        'authored_copy_tuples':{'binding':copies.copy_binding(inventory),'rows':sorted(authored,key=lambda r:r['pointer'])}})


def _align(processor,decision,reference_leaves,copy_leaves,binding,copy_binding,*,max_target_tokens,mode,generated_ids=None):
    aligned=reference_alignment._align(processor,decision,reference_leaves,binding,
        max_target_tokens=max_target_tokens,mode=mode,generated_ids=generated_ids)
    aligned.pop('record_sha256');aligned['schema']=ALIGNMENT_SCHEMA;aligned['copy_binding']=deepcopy(copy_binding)
    nodes={n['pointer']:n for n in aligned['geometry']['nodes']};requests=[]
    for leaf in copy_leaves:
        a,b=nodes[leaf['pointer']]['canonical_chars']
        overlaps=[t['body_index'] for t in aligned['tokens'] if t['attention_mask']==1 and t['special_mask']==0
            and t['canonical_chars'][0]<b and t['canonical_chars'][1]>a]
        need(bool(overlaps),'copy marker has no actual complete mixed-wire token query')
        j=min(overlaps);requests.append({'pointer':leaf['pointer'],'query_index':j,'decoder_input_index':j+1})
    aligned['copy_requests']=requests
    return _seal(aligned)


def _view(view,view_sha256):
    references._pinned(view,view_sha256)
    need(type(view) is dict and set(view)==VIEW_FIELDS and view['schema']==VIEW_SCHEMA,'copy target view packet differs')
    references._binding(view['binding']);copies._copy_binding(view['copy_binding'])
    need(view['binding']==view['copy_binding']['reference_binding'],'copy target nested reference binding differs')
    leaves=view['reference_leaves']
    need(type(leaves) is list,'teacher reference leaves must be a complete array')
    for row in leaves:
        need(type(row) is dict and set(row)=={'pointer','candidate_path','eligible_candidate_indices'} and
             type(row['pointer']) is str and type(row['candidate_path']) is str and
             type(row['eligible_candidate_indices']) is list and bool(row['eligible_candidate_indices']) and
             all(type(i) is int and i>=0 for i in row['eligible_candidate_indices']) and
             row['eligible_candidate_indices']==sorted(set(row['eligible_candidate_indices'])),
             'teacher reference request row fields or eligible ordinals differ')
    need([{k:r[k] for k in ('pointer','candidate_path')} for r in leaves]==copies._reference_shape_requests(view['decision']),
         'teacher reference leaves differ from complete actual mixed-wire marker set')
    ordinals=view['authored_ordinals'];need(type(ordinals) is list,'authored reference ordinals must be an array')
    by_pointer={r['pointer']:r for r in leaves};seen_ordinals=set()
    for row in ordinals:
        need(type(row) is dict and set(row)=={'pointer','candidate_ordinal'} and type(row['pointer']) is str and
             row['pointer'] in by_pointer and row['pointer'] not in seen_ordinals and type(row['candidate_ordinal']) is int and
             row['candidate_ordinal'] in by_pointer[row['pointer']]['eligible_candidate_indices'],
             'teacher authored reference ordinal fields/identity/eligibility differ')
        seen_ordinals.add(row['pointer'])
    need([r['pointer'] for r in ordinals]==[r['pointer'] for r in leaves],
         'teacher complete authored reference ordinal set differs')
    packet=view['authored_copy_tuples']
    need(type(packet) is dict and set(packet)=={'binding','rows'} and packet['binding']==view['copy_binding']
         and type(packet['rows']) is list,'authored copy tuple packet differs')
    seen=set()
    for row in packet['rows']:
        need(type(row) is dict and set(row)=={'pointer','alternatives'} and type(row['pointer']) is str
             and row['pointer'] not in seen,'authored copy tuple row differs');seen.add(row['pointer'])
        alternatives=row['alternatives']
        if alternatives is not None:
            need(type(alternatives) is list and bool(alternatives),'authored copy tuple alternatives missing')
            tuples=set()
            for t in alternatives:
                need(type(t) is dict and set(t)=={'source_slot','start_byte','end_byte'} and
                    all(type(v) is int for v in t.values()) and t['source_slot']>=0 and 0<=t['start_byte']<t['end_byte'],
                    'authored copy tuple coordinate differs')
                key=tuple(t[k] for k in ('source_slot','start_byte','end_byte'));need(key not in tuples,'duplicate authored copy tuple');tuples.add(key)
    need(type(view['copy_leaves']) is list and all(type(r) is dict and set(r)=={'pointer'} and
         type(r['pointer']) is str for r in view['copy_leaves']) and
         len(view['copy_leaves'])==len(seen) and {r['pointer'] for r in view['copy_leaves']}==seen,
         'copy target leaves/whole tuple set differs')
    need(view['copy_leaves']==copies._part_requests(view['decision']),
         'teacher copy leaves differ from complete actual mixed-wire marker set')


def align_original_copy_decision_target(processor,view,*,view_sha256,max_target_tokens):
    _view(view,view_sha256)
    need(source.processor_binding(processor)==view['processor_binding'],'copy target processor differs')
    result=_align(processor,view['decision'],view['reference_leaves'],view['copy_leaves'],view['binding'],view['copy_binding'],
                  max_target_tokens=max_target_tokens,mode='teacher_forced')
    need(source.processor_binding(processor)==view['processor_binding'],'copy target processor changed')
    return result


def align_generated_original_copy_decision(processor,decision,generated_ids,inventory,*,max_target_tokens,eos_observed):
    copies._inventory(inventory)
    need(eos_observed is True and source.processor_binding(processor)==inventory['reference_inventory']['processor_binding'],
         'generated copy decision requires actual EOS/pinned processor')
    leaves=copies.copy_requests(decision,inventory)
    result=_align(processor,decision,copies.reference_requests(decision,inventory),leaves,
        inventory['reference_inventory']['binding'],copies.copy_binding(inventory),max_target_tokens=max_target_tokens,
        mode='generated',generated_ids=generated_ids)
    need(source.processor_binding(processor)==inventory['reference_inventory']['processor_binding'],'copy processor changed')
    return result
