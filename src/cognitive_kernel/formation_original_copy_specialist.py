"""Owned UTF8 occurrence selection on the complete native MFM decoder.

Source states are frozen inputs. New bytes/context keys do not use target text,
source identity embeddings, pretrained bytes or all-boundary-pairs tensors.
Mean pooled observation context may dilute long sources. Greedy staged choice
is not global tuple MAP. Construction mechanics do not establish learning.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from .canonical import canonical_sha256, require_identifier, require_sha256
from . import formation_reference_specialist as reference
from .formation_reference_specialist import FormationReferenceSpecialist, _binding, _file_sha256, CONDITIONING


TASK = 'mfm-native-original-copy-producer-v1'
_COPY_BINDING = {'reference_binding','joint_alignment_sha256','joint_schema','copy_inventory_sha256'}
_SOURCE = {'source_slot','source_ref','content_sha256','original_bytes','modality','source_token_indices','boundaries'}
_BOUNDARY = {'byte_offset','left_token_indices','right_token_indices','original_byte_window','coordinate_features'}
_POINTER = re.compile(r'/proposals/(?:0|[1-9][0-9]*)/proposal/value_text/parts/(?:0|[1-9][0-9]*)')


def _copy_binding(value):
    if type(value) is not dict or set(value) != _COPY_BINDING:
        raise ValueError('copy binding requires exact nested reference/joint/inventory fields')
    _binding(value['reference_binding'])
    for name in ('joint_alignment_sha256','copy_inventory_sha256'):
        if require_sha256(value[name], name) != value[name]:
            raise ValueError('copy pins must be canonical')
    if value['joint_schema'] not in ('mfm-joint-caller-input-alignment-v4','mfm-joint-caller-input-alignment-v5',
                                   'mfm-joint-caller-input-alignment-v6'):
        raise ValueError('copy binding requires the actual J4, J5 or J6 original joint schema')
    return deepcopy(value)


def _indices(values, length, mask):
    if (type(values) is not list or any(type(i) is not int or not 0 <= i < length for i in values)
            or values != sorted(set(values)) or any(mask[i] != 1 for i in values)):
        raise ValueError('copy context support requires sorted unique active input indices')
    return values


def _sources(rows, *, length=None, mask=None):
    if type(rows) is not list:
        raise ValueError('copy keys require complete ordered opened sources')
    seen = set()
    for slot,row in enumerate(rows):
        if type(row) is not dict or set(row) != _SOURCE or type(row['source_slot']) is not int or row['source_slot'] != slot:
            raise ValueError('copy sources must retain every original opened slot')
        if require_identifier(row['source_ref'],'source_ref') != row['source_ref'] or row['source_ref'] in seen:
            raise ValueError('copy source identity invalid or duplicated')
        seen.add(row['source_ref'])
        if require_sha256(row['content_sha256'],'content_sha256') != row['content_sha256']:
            raise ValueError('copy content pin must be canonical')
        n = row['original_bytes']; boundaries = row['boundaries']
        if type(n) is not int or n < 0 or type(row['modality']) is not str or not row['modality']:
            raise ValueError('copy source size/modality invalid')
        if type(boundaries) is not list or not boundaries:
            raise ValueError('copy source must retain all UTF8 boundaries including endpoints')
        offsets = []
        union = set()
        original=bytearray(n) if length is not None else None
        covered=bytearray(n) if length is not None else None
        for index,boundary in enumerate(boundaries):
            if type(boundary) is not dict or set(boundary) != _BOUNDARY:
                raise ValueError('copy boundary metadata differs')
            offset = boundary['byte_offset']
            if type(offset) is not int or not 0 <= offset <= n or (offsets and not 1 <= offset-offsets[-1] <= 4):
                raise ValueError('copy boundary offsets must be ordered UTF8 atom endpoints')
            offsets.append(offset)
            window = boundary['original_byte_window']; features = boundary['coordinate_features']
            if (type(window) is not list or len(window) != 8 or
                    any(type(b) is not int or not 0 <= b <= 256 for b in window)):
                raise ValueError('copy boundary needs eight exact original bytes/padding')
            for byte_index,b in zip(range(offset-4,offset+4),window):
                if (b == 256) != (not 0 <= byte_index < n):
                    raise ValueError('copy byte padding disagrees with source endpoints')
                if original is not None and 0 <= byte_index < n:
                    if covered[byte_index] and original[byte_index] != b:
                        raise ValueError('copy boundary windows disagree on actual original byte')
                    original[byte_index]=b;covered[byte_index]=1
            if (type(features) is not list or len(features) != 8 or
                    any(type(v) is not float or not 0 <= v <= 1 for v in features)):
                raise ValueError('copy coordinates require eight finite normalized floats')
            if features[0] != offset/max(1,n) or features[6:] != [float(index == 0),float(index == len(boundaries)-1)]:
                raise ValueError('copy coordinate endpoints disagree with actual source')
            for side, absent in (('left_token_indices',index == 0),('right_token_indices',index == len(boundaries)-1)):
                indices = boundary[side]
                if length is not None:
                    _indices(indices,length,mask)
                elif type(indices) is not list or any(type(i) is not int or i < 0 for i in indices) or indices != sorted(set(indices)):
                    raise ValueError('copy support indices malformed')
                if (not indices) != absent:
                    raise ValueError('actual neighboring original atom lacks complete token support')
                union.update(indices)
        if offsets[0] != 0 or offsets[-1] != n or (n == 0 and len(offsets) != 1):
            raise ValueError('copy boundaries omit a source endpoint')
        if row['source_token_indices'] != sorted(union):
            raise ValueError('copy source pooling must include complete adjacent supports')
        if original is not None:
            _indices(row['source_token_indices'],length,mask)
            if not all(covered) or sha256(original).hexdigest() != row['content_sha256']:
                raise ValueError('copy original bytes do not match complete opened content hash')
            try: text=original.decode('utf-8',errors='strict')
            except UnicodeDecodeError as exc:raise ValueError('copy original bytes must be strict UTF8') from exc
            actual=[0]
            for atom in text:actual.append(actual[-1]+len(atom.encode('utf-8')))
            if offsets != actual:raise ValueError('copy boundaries differ from actual UTF8 codepoint endpoints')
            for i,boundary in enumerate(boundaries):
                features=boundary['coordinate_features']
                left=(offsets[i]-offsets[i-1])/4 if i else 0.
                right=(offsets[i+1]-offsets[i])/4 if i+1<len(offsets) else 0.
                if features[4:6] != [left,right]:
                    raise ValueError('copy adjacent atom lengths disagree with actual original bytes')
    return rows


class FormationOriginalCopySpecialist(FormationReferenceSpecialist):
    def __init__(self, config):
        super().__init__(config)
        self._reference_parent_keys = tuple(self.state_dict())
        width = config.width
        self.copy_byte_embedding = nn.Embedding(257,32)
        self.copy_boundary = nn.Sequential(nn.Linear(2*width+256+8,width),nn.GELU(),nn.Linear(width,width))
        self.copy_source_key = nn.Linear(width,width,bias=False)
        self.copy_marker_query = nn.Linear(2*width,width)
        self.copy_source_query = nn.Linear(width,width)
        self.copy_start_query = nn.Linear(2*width,width)
        self.copy_end_query = nn.Linear(3*width,width)
        self.copy_start_key = nn.Linear(width,width,bias=False)
        self.copy_end_key = nn.Linear(width,width,bias=False)

    @classmethod
    def from_parent_safetensors(cls, *args, **kwargs):
        raise ValueError('original-copy migration requires the complete native-reference parent')

    def load_parent_safetensors(self, *args, **kwargs):
        raise ValueError('lexical-only migration cannot bypass complete reference-parent validation')

    @classmethod
    def from_reference_parent_safetensors(cls,config,path,*,parent_sha256,parent_config_path,
            parent_config_file_sha256,parent_task,parent_provenance,parent_provenance_sha256):
        model = cls(config)
        model.load_reference_parent_safetensors(path,parent_sha256=parent_sha256,
            parent_config_path=parent_config_path,parent_config_file_sha256=parent_config_file_sha256,
            parent_task=parent_task,parent_provenance=parent_provenance,parent_provenance_sha256=parent_provenance_sha256)
        return model

    def load_reference_parent_safetensors(self,path,*,parent_sha256,parent_config_path,
            parent_config_file_sha256,parent_task,parent_provenance,parent_provenance_sha256):
        """Validate actual complete FP32 ancestry before a single tensor changes."""
        for name,value in (('parent_sha256',parent_sha256),('parent_config_file_sha256',parent_config_file_sha256),
                           ('parent_provenance_sha256',parent_provenance_sha256)):
            if require_sha256(value,name) != value:
                raise ValueError('reference parent pins must be canonical')
        fields={'kind','task','producer_sha256','weight_file_sha256','config_file_sha256',
                'config_canonical_sha256','lexical_parent_binding'}
        if (parent_task != reference.TASK or type(parent_provenance) is not dict or set(parent_provenance) != fields or
                canonical_sha256(parent_provenance) != parent_provenance_sha256 or
                parent_provenance['task'] != reference.TASK or
                parent_provenance['kind'] not in ('untrained_fixture','unqualified_parent')):
            raise ValueError('parent requires externally pinned actual native-reference ancestry')
        if (_file_sha256(path) != parent_sha256 or
                _file_sha256(parent_config_path) != parent_config_file_sha256):
            raise ValueError('reference parent weight/config FILE mismatch')
        def unique(pairs):
            record={}
            for key,value in pairs:
                if key in record: raise ValueError('duplicate parent config key')
                record[key]=value
            return record
        config_record=json.loads(Path(parent_config_path).read_text(encoding='utf-8'),object_pairs_hook=unique)
        config_pin=canonical_sha256(self.config.record())
        if canonical_sha256(config_record) != config_pin:
            raise ValueError('bound actual config differs from this model config')
        expected={'producer_sha256':_file_sha256(reference.__file__),'weight_file_sha256':parent_sha256,
                  'config_file_sha256':parent_config_file_sha256,'config_canonical_sha256':config_pin}
        if any(parent_provenance[k] != v for k,v in expected.items()):
            raise ValueError('native-reference ancestry FILE/config differs')
        lexical=parent_provenance['lexical_parent_binding']
        if (type(lexical) is not dict or set(lexical) != {'parent_weight_file_sha256','parent_config_canonical_sha256',
                'new_task','reference_initialization','reference_updates_performed_by_migration'} or
                lexical['new_task'] != reference.TASK or
                lexical['reference_initialization'] != 'fresh-native-typed-reference-head-v1' or
                type(lexical['reference_updates_performed_by_migration']) is not int or
                lexical['reference_updates_performed_by_migration'] != 0 or
                lexical['parent_config_canonical_sha256'] != config_pin):
            raise ValueError('complete actual lexical-to-reference migration ancestry required')
        if require_sha256(lexical['parent_weight_file_sha256'],'lexical parent pin') != lexical['parent_weight_file_sha256']:
            raise ValueError('lexical parent FILE must be canonical')
        try: parent=load_file(str(path),device='cpu')
        except Exception as exc: raise ValueError('reference parent safetensors cannot be read') from exc
        current=self.state_dict()
        if set(parent) != set(self._reference_parent_keys):
            raise ValueError('complete lexical and reference parent tensor names differ')
        for name,tensor in parent.items():
            if (tensor.shape != current[name].shape or tensor.dtype != torch.float32 or
                    current[name].dtype != torch.float32 or not torch.isfinite(tensor).all().item()):
                raise ValueError('reference parent tensor shape/finite FP32 differs: '+name)
        if (_file_sha256(path) != parent_sha256 or
                _file_sha256(parent_config_path) != parent_config_file_sha256 or
                _file_sha256(reference.__file__) != expected['producer_sha256']):
            raise ValueError('reference parent/config/producer changed during validation')
        self.load_state_dict({**current,**parent},strict=True)
        self.parent_binding={'parent_weight_file_sha256':parent_sha256,
            'parent_config_file_sha256':parent_config_file_sha256,'parent_config_canonical_sha256':config_pin,
            'parent_task':parent_task,'parent_provenance':deepcopy(parent_provenance),
            'parent_provenance_sha256':parent_provenance_sha256,'new_task':TASK,
            'copy_initialization':'fresh-native-byte-context-occurrence-head-v1','copy_updates_performed_by_migration':0}
        return deepcopy(self.parent_binding)

    def copy_keys(self,*,base_states,source_mask,inventory):
        source=self._source(base_states,source_mask)
        fields={'schema','reference_inventory','joint_alignment_sha256','joint_schema','source_identity',
                'source_geometry','sources','record_sha256'}
        if (type(inventory) is not dict or set(inventory) != fields or
                inventory['schema'] != 'mfm-original-copy-input-inventory-v1' or
                canonical_sha256({k:v for k,v in inventory.items() if k != 'record_sha256'}) != inventory['record_sha256']):
            raise ValueError('copy keys require a sealed exact input-only inventory')
        geometry=inventory['source_geometry']
        if (type(geometry) is not dict or set(geometry) != {'token_count','attention_mask'} or
                type(geometry['token_count']) is not int or geometry['token_count'] != source.shape[1] or
                type(geometry['attention_mask']) is not list or
                any(type(v) is not int or v not in (0,1) for v in geometry['attention_mask']) or
                len(geometry['attention_mask']) != source.shape[1] or source_mask[0].tolist() != geometry['attention_mask']):
            raise ValueError('copy source requires exact complete admitted input length/mask')
        binding=_copy_binding(dict(reference_binding=inventory['reference_inventory']['binding'],
            joint_alignment_sha256=inventory['joint_alignment_sha256'],joint_schema=inventory['joint_schema'],
            copy_inventory_sha256=inventory['record_sha256']))
        rows=_sources(inventory['sources'],length=source.shape[1],mask=geometry['attention_mask'])
        projected=self.source_projection(source.to(self.source_projection.weight.dtype))[0]
        zero=projected.new_zeros(self.config.width)
        source_keys=[]; boundary_keys=[]
        for row in rows:
            indices=row['source_token_indices']
            source_keys.append(self.copy_source_key(projected[indices].mean(0)) if indices else zero)
            boundaries=[]
            for boundary in row['boundaries']:
                left=boundary['left_token_indices'];right=boundary['right_token_indices']
                window=torch.tensor(boundary['original_byte_window'],dtype=torch.long,device=source.device)
                features=projected.new_tensor(boundary['coordinate_features'])
                boundaries.append(self.copy_boundary(torch.cat((projected[left].mean(0) if left else zero,
                    projected[right].mean(0) if right else zero,self.copy_byte_embedding(window).flatten(),features))))
            boundary_keys.append(torch.stack(boundaries))
        return {'binding':binding,'sources':deepcopy(rows),
            'source_keys':torch.stack(source_keys) if rows else projected.new_empty((0,self.config.width)),
            'boundary_keys':boundary_keys}

    def _copy_queries(self,hidden,requests,keys):
        if type(keys) is not dict or set(keys) != {'binding','sources','source_keys','boundary_keys'}:
            raise ValueError('copy key Tensor packet differs')
        binding=_copy_binding(keys['binding'])
        if _binding(hidden['binding']) != binding['reference_binding']:
            raise ValueError('copy hidden/reference/key binding mismatch')
        states=hidden['hidden']; rows=_sources(keys['sources']); source_keys=keys['source_keys']
        if (hidden['mode'] not in ('generated','teacher_forced') or hidden['conditioning'] != CONDITIONING or
                not isinstance(states,torch.Tensor) or states.ndim != 3 or states.shape[0] != 1 or
                states.shape[2] != self.config.width or hidden['completion_index'] != states.shape[1]-1 or
                not torch.isfinite(states).all().item() or not isinstance(source_keys,torch.Tensor) or
                source_keys.shape != (len(rows),self.config.width) or source_keys.device != states.device or
                not torch.isfinite(source_keys).all().item() or type(keys['boundary_keys']) is not list or
                len(keys['boundary_keys']) != len(rows)):
            raise ValueError('copy hidden/key packet geometry invalid')
        for row,boundaries in zip(rows,keys['boundary_keys']):
            if (not isinstance(boundaries,torch.Tensor) or boundaries.shape != (len(row['boundaries']),self.config.width) or
                    boundaries.device != states.device or not torch.isfinite(boundaries).all().item()):
                raise ValueError('copy boundary Tensor geometry invalid')
        if type(requests) is not list:
            raise ValueError('copy requests require explicit concrete markers')
        seen=set();queries=[]
        for row in requests:
            if type(row) is not dict or set(row) != {'pointer','query_index','decoder_input_index'}:
                raise ValueError('copy request fields differ')
            pointer=row['pointer']; j=row['query_index']; operand=row['decoder_input_index']
            if type(pointer) is not str or not _POINTER.fullmatch(pointer) or pointer in seen:
                raise ValueError('copy pointer is malformed or duplicated')
            seen.add(pointer)
            if type(j) is not int or type(operand) is not int or j < 0 or operand != j+1 or operand >= states.shape[1]:
                raise ValueError('copy query must predict actual whole-wire marker at j+1')
            queries.append(self.copy_marker_query(torch.cat((states[0,j],states[0,hidden['completion_index']]))))
        return torch.stack(queries) if queries else states.new_empty((0,self.config.width))

    def copy_logits(self,hidden,requests,keys,*,source_slot=None,start_byte=None):
        queries=self._copy_queries(hidden,requests,keys)
        rows=keys['sources']; width=self.config.width
        if source_slot is None:
            if start_byte is not None: raise ValueError('end conditioning requires an actual source')
            coordinates=[row['source_slot'] for row in rows]
            valid=[row['original_bytes'] > 0 for row in rows]
            if requests and not any(valid): raise ValueError('copy request has no nonempty opened source')
            logits=self.copy_source_query(queries) @ keys['source_keys'].T/width**.5
            stage='source'
        else:
            if type(source_slot) is not int or not 0 <= source_slot < len(rows) or rows[source_slot]['original_bytes'] == 0:
                raise ValueError('copy source selection is not an eligible opened slot')
            row=rows[source_slot];coordinates=[b['byte_offset'] for b in row['boundaries']]
            boundary_keys=keys['boundary_keys'][source_slot]
            selected=keys['source_keys'][source_slot].expand(len(requests),-1)
            if start_byte is None:
                valid=[offset < row['original_bytes'] for offset in coordinates]
                logits=self.copy_start_query(torch.cat((queries,selected),dim=1)) @ self.copy_start_key(boundary_keys).T/width**.5
                stage='start'
            else:
                if type(start_byte) is not int or start_byte not in coordinates or start_byte >= row['original_bytes']:
                    raise ValueError('copy start must be an actual nonterminal UTF8 boundary')
                start=boundary_keys[coordinates.index(start_byte)].expand(len(requests),-1)
                valid=[offset > start_byte for offset in coordinates]
                logits=self.copy_end_query(torch.cat((queries,selected,start),dim=1)) @ self.copy_end_key(boundary_keys).T/width**.5
                stage='end'
        if not torch.isfinite(logits).all().item():raise ValueError('copy scoring produced nonfinite logits')
        mask=torch.tensor(valid,dtype=torch.bool,device=logits.device)
        return {'stage':stage,'logits':logits.masked_fill(~mask,-torch.inf),'coordinates':coordinates}

    def copy_loss(self,hidden,requests,keys,*,authored_copy_tuples):
        if hidden['mode'] != 'teacher_forced':raise ValueError('copy loss requires explicit teacher-forced states')
        self._copy_queries(hidden,requests,keys)
        if (type(authored_copy_tuples) is not dict or set(authored_copy_tuples) != {'binding','rows'} or
                _copy_binding(authored_copy_tuples['binding']) != _copy_binding(keys['binding']) or
                type(authored_copy_tuples['rows']) is not list):
            raise ValueError('authored copy tuple packet binding/fields differ')
        labels={}
        for row in authored_copy_tuples['rows']:
            if type(row) is not dict or set(row) != {'pointer','alternatives'} or type(row['pointer']) is not str or row['pointer'] in labels:
                raise ValueError('copy labels require unique exact request rows')
            labels[row['pointer']]=row['alternatives']
        if set(labels) != {row['pointer'] for row in requests}:raise ValueError('complete authored copy pointer set differs')
        if not requests:return None
        source_scores=F.log_softmax(self.copy_logits(hidden,requests,keys)['logits'].float(),dim=1)
        losses=[]
        for i,request in enumerate(requests):
            alternatives=labels[request['pointer']]
            if type(alternatives) is not list or not alternatives:raise ValueError('copy labels are unresolved or empty')
            seen=set();probabilities=[]
            for alternative in alternatives:
                if type(alternative) is not dict or set(alternative) != {'source_slot','start_byte','end_byte'} or any(type(v) is not int for v in alternative.values()):
                    raise ValueError('copy alternative requires a complete integer occurrence tuple')
                s,a,b=(alternative[k] for k in ('source_slot','start_byte','end_byte'))
                if (s,a,b) in seen:raise ValueError('duplicate authored occurrence tuple')
                seen.add((s,a,b))
                starts=self.copy_logits(hidden,[request],keys,source_slot=s)
                ends=self.copy_logits(hidden,[request],keys,source_slot=s,start_byte=a)
                if b not in ends['coordinates'] or b <= a:raise ValueError('copy end is not a later same-source UTF8 boundary')
                start_index=starts['coordinates'].index(a);end_index=ends['coordinates'].index(b)
                probabilities.append(source_scores[i,s]+F.log_softmax(starts['logits'].float(),dim=1)[0,start_index]+
                    F.log_softmax(ends['logits'].float(),dim=1)[0,end_index])
            losses.append(-torch.logsumexp(torch.stack(probabilities),dim=0))
        return torch.stack(losses).mean()

    def predict_copies(self,hidden,requests,keys):
        if hidden['mode'] != 'generated':raise ValueError('copy predictions require actual generated states')
        source=self.copy_logits(hidden,requests,keys);rows=[]
        for i,request in enumerate(requests):
            s=source['coordinates'][int(source['logits'][i].argmax().item())]
            start=self.copy_logits(hidden,[request],keys,source_slot=s)
            a=start['coordinates'][int(start['logits'][0].argmax().item())]
            end=self.copy_logits(hidden,[request],keys,source_slot=s,start_byte=a)
            b=end['coordinates'][int(end['logits'][0].argmax().item())]
            rows.append(dict(pointer=request['pointer'],source_slot=s,start_byte=a,end_byte=b))
        return {'binding':deepcopy(keys['binding']),'rows':rows}

    def generate_original_copy_decision(self,*,processor,base_states,source_mask,caller,
            caller_sha256,joint_alignment,joint_alignment_sha256,expected_binding,
            artifact_sha256,inference_run_id,max_new_tokens):
        """Own generation→same-wire queries→real native heads→exact consumer.

        Full reference decoding uses FP32 and carries no cached performance or
        learned qualification. Refused actions stay raw; no repair or retry.
        """
        from . import formation_original_copy_decision as consumer
        from .formation_reference_decision import caller_api
        from scripts.mfm import formation_original_copy_target_alignment as alignment
        from scripts.mfm import formation_source_alignment as source_api

        result={'generated_token_ids':[],'raw_output_text':'','eos_observed':False,
            'validation_status':'invalid','validation_error':None,'failure_stage':None,
            'reference_predictions':None,'copy_predictions':None,'copy_provenance':None,'bundle':None}
        stage,raw_decoded='preflight',False
        modes=[(module,module.training) for module in self.modules()]
        device=self.source_projection.weight.device
        devices=[device.index] if device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices),torch.inference_mode(),torch.autocast(device.type,enabled=False):
                if device.type not in ('cpu','cuda') or any(p.dtype != torch.float32 or p.device != device for p in self.parameters()):
                    raise ValueError('native copy inference requires one CPU/CUDA device and owned FP32 weights')
                self.config.validate()
                if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.config.max_target_tokens:
                    raise ValueError('native copy inference budget exceeds full target geometry')
                for name,value in (('artifact_sha256',artifact_sha256),('caller_sha256',caller_sha256),
                                   ('joint_alignment_sha256',joint_alignment_sha256)):
                    if require_sha256(value,name) != value:raise ValueError('native inference pins must be canonical')
                if require_identifier(inference_run_id,'inference_run_id') != inference_run_id:
                    raise ValueError('native inference run identity must be canonical')
                expected=_copy_binding(expected_binding)
                inventory=consumer.bind_original_copy_inventory(caller,joint_alignment,
                    caller_sha256=caller_sha256,joint_alignment_sha256=joint_alignment_sha256,
                    expected_binding=expected['reference_binding'])
                if consumer.copy_binding(inventory) != expected:raise ValueError('complete copy input binding differs')
                ref_inventory=inventory['reference_inventory']
                if source_api.processor_binding(processor) != ref_inventory['processor_binding']:
                    raise ValueError('native copy inference processor binding differs')
                tokenizer=processor.tokenizer
                if (len(tokenizer) != self.config.vocabulary_size or tokenizer.bos_token_id != self.config.start_token_id or
                        tokenizer.eos_token_id != self.config.end_token_id or tokenizer.pad_token_id != self.config.pad_token_id):
                    raise ValueError('native copy inference processor/config geometry differs')
                source=self._source(base_states,source_mask)
                if source.shape[1] != len(joint_alignment['input_ids']) or source_mask[0].tolist() != joint_alignment['attention_mask']:
                    raise ValueError('native copy inference needs complete admitted source length/mask')
                # Key construction validates full target-free copy geometry.
                self.eval()
                copy_keys=self.copy_keys(base_states=source,source_mask=source_mask,inventory=inventory)
                context,opened,_=caller_api.caller_input_from_record(caller)
                stage='generation';prefix=[self.config.start_token_id]
                for _ in range(max_new_tokens):
                    inputs=torch.tensor([prefix],dtype=torch.long,device=device)
                    logits=self.next_token_logits(base_states=source,source_mask=source_mask,input_ids=inputs)
                    if logits.shape != (1,self.config.vocabulary_size) or not torch.isfinite(logits).all().item():
                        raise ValueError('native lexical logits have invalid geometry or nonfinite values')
                    token=int(logits[0].argmax().item())
                    if token == self.config.end_token_id:
                        result['eos_observed']=True;break
                    result['generated_token_ids'].append(token);prefix.append(token)
                if not result['eos_observed']:raise ValueError('no EOS inside the declared native inference budget')
                stage='decode'
                result['raw_output_text']=tokenizer.decode(result['generated_token_ids'],skip_special_tokens=False,
                    clean_up_tokenization_spaces=False)
                raw_decoded=True;stage='parse'
                def unique(pairs):
                    record={}
                    for key,value in pairs:
                        if key in record:raise ValueError('duplicate generated JSON key')
                        record[key]=value
                    return record
                def nonfinite(value):raise ValueError('nonfinite generated JSON constant: '+value)
                decision=json.loads(result['raw_output_text'],object_pairs_hook=unique,parse_constant=nonfinite)
                stage='alignment'
                geometry=alignment.align_generated_original_copy_decision(processor,decision,
                    result['generated_token_ids'],inventory,max_target_tokens=self.config.max_target_tokens,eos_observed=True)
                stage='reference'
                hidden=self.decision_states(base_states=source,source_mask=source_mask,
                    input_ids=torch.tensor([geometry['decoder_inputs']],dtype=torch.long,device=device),
                    binding=ref_inventory['binding'],mode='generated')
                ref_keys=self.candidate_keys(base_states=source,source_mask=source_mask,
                    candidates=ref_inventory['candidates'],binding=ref_inventory['binding'])
                requests=geometry['requests']
                eligibility=torch.tensor([[slot in request['eligible_candidate_indices'] for slot in ref_keys['slots']]
                    for request in requests],dtype=torch.bool,device=device).reshape(len(requests),len(ref_keys['slots']))
                result['reference_predictions']=self.predict_references(hidden,requests,ref_keys,eligibility)
                stage='copy'
                result['copy_predictions']=self.predict_copies(hidden,geometry['copy_requests'],copy_keys)
                stage='compile_ground'
                compiled=consumer.compile_original_copy_decision(context,opened,decision,
                    result['copy_predictions'],result['reference_predictions'],caller=caller,caller_sha256=caller_sha256,
                    joint_alignment=joint_alignment,joint_alignment_sha256=joint_alignment_sha256,
                    expected_binding=expected,artifact_sha256=artifact_sha256,inference_run_id=inference_run_id)
                result['bundle']=compiled['bundle'];result['copy_provenance']=compiled['copy_provenance']
                result['validation_status']='grounded_proposal_only'
        except (ValueError,TypeError,KeyError,IndexError,AttributeError,RuntimeError) as exc:
            result['failure_stage']=stage;result['validation_error']=f'{type(exc).__name__}: {exc}'
            if stage == 'decode':result['raw_output_text']=None
            elif not raw_decoded and stage == 'generation':
                try:
                    with torch.random.fork_rng(devices=devices),torch.inference_mode(),torch.autocast(device.type,enabled=False):
                        result['raw_output_text']=processor.tokenizer.decode(result['generated_token_ids'],
                            skip_special_tokens=False,clean_up_tokenization_spaces=False)
                except (ValueError,TypeError,AttributeError,RuntimeError) as decode_error:
                    result['raw_output_text']=None;result['raw_decode_error']=f'{type(decode_error).__name__}: {decode_error}'
        finally:
            for module,training in modes:module.training=training
        return result
