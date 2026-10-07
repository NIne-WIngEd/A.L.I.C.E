"""Bounded read-only field, rank and source-conditioning measurements.

No admission, optimizer, generation or memory authority lives here. The caller
supplies admitted examples and an explicit task-bound specialist view. Ranks use
competition ranking: one plus the number of strictly greater logits; top1 uses
the actual argmax, so tied ranks do not imply the same greedy token.
"""
from cognitive_kernel.canonical import CognitiveKernelContractError
from . import analyze_v16_budget_failure as ancestry
from . import formation_sequence_curriculum as curriculum


def task_targets(tokenizer, example, maximum, task):
    if task=='formation':
        return ancestry.precise_target_groups(tokenizer,example,maximum)
    if task!='source_reconstruction':
        raise CognitiveKernelContractError('unregistered failure-measurement task')
    inputs,labels=curriculum.lexical_target_ids(tokenizer,example,maximum)
    text,spans=ancestry.precise_scalar_spans(curriculum.lexical_source_record(example))
    spans=[(start,end,'source_text:'+kind.split(':',1)[1]
            if kind.startswith('value:') and kind.endswith('.text') else kind)
           for start,end,kind in spans]
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=encoded['offset_mapping']
    if (list(encoded['input_ids'])!=labels[:-1] or inputs[1:]!=labels[:-1]
            or inputs[0]!=tokenizer.bos_token_id or labels[-1]!=tokenizer.eos_token_id
            or len(offsets)!=len(labels)-1):
        raise CognitiveKernelContractError('source field target offset/shift/EOS differs')
    groups=[]
    for offset in offsets:
        if (len(offset)!=2 or any(type(x) is not int for x in offset)
                or not 0<=offset[0]<offset[1]<=len(text)):
            raise CognitiveKernelContractError('source field tokenizer offsets are invalid')
        left,right=offset
        size,kind=max(((max(0,min(right,end)-max(left,start)),kind)
                       for start,end,kind in spans),default=(0,'structure'))
        groups.append(kind if size else 'structure')
    return labels,[*groups,'eos']


def _source_pair(payload):
    import torch
    states,mask=payload.get('base_states'),payload.get('source_mask')
    if (not isinstance(states,torch.Tensor) or not isinstance(mask,torch.Tensor)
            or states.ndim!=3 or mask.ndim!=2 or mask.shape!=states.shape[:2]
            or not states.shape[1] or not torch.isfinite(states).all()
            or not torch.isfinite(mask).all() or not mask.bool().any(dim=1).all()):
        raise CognitiveKernelContractError('source intervention needs a complete finite states/mask pair')
    return states,mask


def source_intervention_payload(payload, foreign_payload=None, mode='zero'):
    import torch
    states,mask=_source_pair(payload)
    if mode=='zero':
        if foreign_payload is not None:
            raise CognitiveKernelContractError('zero-source intervention does not consume foreign states')
        return {**payload,'base_states':torch.zeros_like(states)}
    if mode!='foreign_case' or foreign_payload is None:
        raise CognitiveKernelContractError('intervention must zero states or replace a foreign case pair')
    foreign,foreign_mask=_source_pair(foreign_payload)
    if foreign.shape[0]!=states.shape[0] or foreign.shape[2]!=states.shape[2]:
        raise CognitiveKernelContractError('foreign source batch/hidden geometry differs')
    return {**payload,'base_states':foreign.to(device=states.device,dtype=states.dtype),
            'source_mask':foreign_mask.to(device=mask.device,dtype=mask.dtype)}


def rank_measure(model_view, payload, position, candidate=None):
    import torch
    from torch.nn import functional as F
    states,mask=_source_pair(payload)
    inputs,labels=payload.get('input_ids'),payload.get('labels')
    if (any(p.dtype!=torch.float32 for p in model_view.parameters())
            or not isinstance(inputs,torch.Tensor) or not isinstance(labels,torch.Tensor)
            or inputs.ndim!=2 or labels.shape!=inputs.shape or inputs.shape[0]!=1
            or states.shape[0]!=1 or inputs.dtype!=torch.long or labels.dtype!=torch.long
            or type(position) is not int or not 0<=position<inputs.shape[1]):
        raise CognitiveKernelContractError('rank measurement requires one FP32 case and a valid teacher position')
    teacher=int(labels[0,position]);vocabulary=model_view.config.vocabulary_size
    if (not 0<=teacher<vocabulary or (candidate is not None
            and (type(candidate) is not int or not 0<=candidate<vocabulary))):
        raise CognitiveKernelContractError('rank measurement teacher/candidate token is invalid')
    model_view.eval()
    with torch.inference_mode(),torch.autocast(states.device.type,enabled=False):
        # The supplied labels never enter decoding. Causal prefix contains the
        # same current token as a full teacher-prefix pass at this position.
        hidden=model_view._decode(base_states=states,source_mask=mask,input_ids=inputs[:,:position+1])
        logits=F.linear(hidden[0,-1],model_view.token_embedding.weight,model_view.output_bias)
        if logits.dtype!=torch.float32 or not torch.isfinite(logits).all():
            raise CognitiveKernelContractError('rank measurement projection is not finite FP32')
        logs=logits.log_softmax(-1);probability=logs.exp();argmax=int(logits.argmax())
        if not torch.isfinite(logs).all():
            raise CognitiveKernelContractError('rank measurement log probabilities are nonfinite')
        def token_values(token):
            return (float(probability[token]),1+int((logits>logits[token]).sum()),argmax==token)
        tp,tr,tt=token_values(teacher)
        cp,cr,ct=token_values(candidate) if candidate is not None else (None,None,None)
        return {'teacher_token_id':teacher,'teacher_probability':tp,'teacher_rank':tr,'teacher_top1':tt,
            'argmax_token_id':argmax,'candidate_token_id':candidate,'candidate_probability':cp,
            'candidate_rank':cr,'candidate_top1':ct,'dtype':'float32','logits_finite':True,
            'logit_min':float(logits.min()),'logit_max':float(logits.max()),
            'logit_mean':float(logits.mean()),'logit_std':float(logits.std(unbiased=False)),
            'entropy':float(-(probability*logs).sum())}
