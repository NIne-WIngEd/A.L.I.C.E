"""Read-only final natural branch replay and selected AdamW state attribution.

This is not a complete native packet reconstruction or a historical task-gradient
replay. Saved disabled candidates belong only to the final treatment core.
"""
from types import SimpleNamespace
import math
import torch
from src.alice_personality.identity.contracts import CONCEPT_VIEWS
from src.alice_personality.identity.model import masked_softmax
from src.alice_personality.teaching.contracts import ReviewedTarget
from src.alice_personality.teaching.objectives import _distribution, _probability


def require(condition,message='saved branch evidence differs'):
    if not condition:raise ValueError(message)


def tensor(value,*,boolean=False):
    result=torch.tensor(value,dtype=torch.bool if boolean else torch.float32)
    require(boolean or bool(torch.isfinite(result).all()),'nonfinite saved tensor')
    return result


def representation(decision):
    """Recover the exact authorized source union; never sum duplicated views."""
    saved=decision['native_latent'];count=len(decision['sources'][0])
    require(len(decision['sources'])==1 and set(saved['views'])==set(saved['view_masks'])==set(CONCEPT_VIEWS))
    views={k:tensor(saved['views'][k]) for k in CONCEPT_VIEWS}
    masks={k:tensor(saved['view_masks'][k],boolean=True) for k in CONCEPT_VIEWS}
    concepts=tensor(saved['concepts']);concept_mask=tensor(saved['concept_mask'],boolean=True)
    states=concepts.new_zeros((1,count,concepts.shape[-1]));seen=torch.zeros((1,count),dtype=torch.bool)
    for name in CONCEPT_VIEWS:
        values,mask=views[name],masks[name]
        require(values.shape==(1,count+concepts.shape[1],concepts.shape[-1]) and mask.shape==values.shape[:2])
        require(bool((values[~mask]==0).all()),'masked native view carries data')
        allowed=mask[:,:count];copy=values[:,:count]
        require(torch.equal(states[seen & allowed],copy[seen & allowed]),'authorized source copies disagree')
        states[allowed & ~seen]=copy[allowed & ~seen];seen |= allowed
    conditional={k:tensor(v,boolean=k.endswith('_mask')) for k,v in decision['conditional_concepts'].items()}
    return SimpleNamespace(source_states=states,concepts=concepts,concept_mask=concept_mask,
                           views=views,view_masks=masks,**conditional)


def close(actual,expected,name):
    require(actual.shape==expected.shape,'replay shape differs: '+name)
    require(bool(torch.isfinite(actual).all()),'nonfinite replay: '+name)
    # Autograd and inference MHA can select different CPU kernels. Public native
    # fixtures validate this tolerance; report the actual error, never round it away.
    require(torch.allclose(actual,expected,rtol=2e-5,atol=2e-6),'saved native replay differs: '+name)
    return float((actual.detach()-expected).abs().max()) if actual.numel() else 0.


def rms(value):
    return float(value.detach().double().square().mean().sqrt()) if value.numel() else 0.


def replay(model,disabled,natural):
    require(next(model.parameters()).device.type=='cpu' and next(model.parameters()).dtype==torch.float32,
            'diagnostic supports original CPU float32 runtime only')
    for key in ('candidate_ids','candidate_mask','sources','concept_ids','concepts','conditional_concepts'):
        require(disabled[key]==natural[key],'paired saved input differs: '+key)
    for key in ('concepts','concept_mask','views','view_masks'):
        require(disabled['native_latent'][key]==natural['native_latent'][key],'paired N1 differs')
    require(bool(model.n2.conditioning.enabled),'final treatment branch must be enabled')
    candidates=tensor(disabled['native_latent']['candidates']);mask=tensor(natural['candidate_mask'],boolean=True)
    rep=representation(natural)
    disabled_logits=model.n3.scale('preference',model.n2.scalar(candidates)[...,0])*mask
    errors={'disabled_logit_max_error':close(disabled_logits,tensor(disabled['preference_logits']),'disabled logits'),
            'disabled_probability_max_error':close(masked_softmax(disabled_logits,mask),tensor(disabled['preferences']),'disabled probabilities')}
    memory,memory_mask=model.n2.conditioning_context(rep)
    residual=model.n2.conditioning(candidates,mask,memory,memory_mask)
    enabled=candidates+residual
    logits=model.n3.scale('preference',model.n2.scalar(enabled)[...,0])*mask
    probabilities=masked_softmax(logits,mask)
    errors.update(enabled_candidate_max_error=close(enabled,tensor(natural['native_latent']['candidates']),'enabled candidates'),
                  enabled_logit_max_error=close(logits,tensor(natural['preference_logits']),'enabled logits'),
                  enabled_probability_max_error=close(probabilities,tensor(natural['preferences']),'enabled probabilities'))
    effect=(logits-disabled_logits)[mask]
    return logits,{**errors,'grad_enabled':torch.is_grad_enabled(),
        'candidate_rms':rms(candidates[mask]),'effective_residual_rms':rms(residual[mask]),
        'logit_effect_rms':rms(effect),'centered_logit_effect_rms':rms(effect-effect.mean()) if effect.numel() else 0.,
        'probability_effect_rms':rms((probabilities-masked_softmax(disabled_logits,mask))[mask])}


def order_targets(case,decision):
    """Select original named facts in saved candidate coordinates; no new labels."""
    case.validate();require(case.phase=='n2' and case.split=='TRAIN')
    require(len(decision['candidate_ids'])==1)
    ids=decision['candidate_ids'][0];require(len(set(ids))==len(ids))
    require(ids==[c.candidate_id for c in case.document.candidates],'case candidate order differs')
    mask=tensor(decision['candidate_mask'],boolean=True);targets={}
    for annotation in case.targets:
        if annotation.field not in ('preferences','candidate_pair_probabilities'):continue
        pair=annotation.field=='candidate_pair_probabilities'
        require(annotation.mode==('independent' if pair else 'closed_distribution'),'reviewed order mode differs')
        shape=(1,len(ids),len(ids)) if pair else (1,len(ids))
        values=torch.full(shape,float('nan'));uncertainty=values.clone();available=torch.zeros(shape,dtype=torch.bool)
        for fact in annotation.facts:
            require(len(fact.pointers)==(2 if pair else 1))
            position=(0,*(ids.index(p) for p in fact.pointers))
            if fact.available:values[position],uncertainty[position],available[position]=fact.value,fact.uncertainty,True
        allowed=(mask[:,:,None]&mask[:,None,:]&~torch.eye(len(ids),dtype=torch.bool)) if pair else mask
        target=ReviewedTarget(values,available,uncertainty);target.validate(shape,allowed)
        if not pair:require(torch.equal(available,allowed),'closed targets omit allowed candidate')
        targets[annotation.field]=target
    require('preferences' in targets,'missing reviewed preference objective')
    return targets


def order_loss(logits,mask,targets,weights=None):
    """Same primitives and reductions as native preference/pair objectives."""
    require(set(targets)<={'preferences','candidate_pair_probabilities'})
    losses={'preferences':_distribution(logits,targets['preferences'],mask)[0]}
    if 'candidate_pair_probabilities' in targets:
        target=targets['candidate_pair_probabilities'];both=target.available&target.available.transpose(-1,-2)
        require(bool(((target.values+target.values.transpose(-1,-2))[both]-1).abs().le(1e-5).all()),'noncomplementary pair facts')
        margins=logits[:,:,None]-logits[:,None,:]
        allowed=mask[:,:,None]&mask[:,None,:]&~torch.eye(mask.shape[-1],dtype=torch.bool)
        losses['candidate_pair_probabilities']=_probability(margins,target,allowed,logits=True)[0]
    return sum(v*(weights or {}).get(k,1.) for k,v in losses.items()),losses


def adam_transition(before,after,parameter_names):
    """Recover clipped aggregate gradients; validate moments and actual AdamW delta.

    The result is not an unpreconditioned-update estimate and carries no
    historical task attribution. Float64 inversion reduces additional arithmetic
    error; original rounded moments still make recovered gradients approximate.
    """
    a,b=before['optimizer_state'],after['optimizer_state']
    require(len(a['param_groups'])==len(b['param_groups'])==1)
    group=a['param_groups'][0];require(group==b['param_groups'][0],'Adam settings changed')
    ids=group['params'];require(len(ids)==len(parameter_names)==len(set(parameter_names)) and len(set(ids))==len(ids))
    require(not group.get('amsgrad') and not group.get('maximize') and not group.get('fused'), 'unsupported Adam variant')
    require(set(a['state'])<=set(ids) and set(b['state'])<=set(ids),'unknown optimizer parameter')
    beta1,beta2=group['betas'];lr,eps,decay=(group[k] for k in ('lr','eps','weight_decay'))
    gradients={};details={};unused=0;steps=[]
    for name,identifier in zip(parameter_names,ids):
        p,q=(x['model_state'][name].detach().double() for x in (before,after))
        old,new=a['state'].get(identifier),b['state'].get(identifier)
        step0=int(old['step']) if old else 0;step=int(new['step']) if new else 0
        if step==step0:
            require(torch.equal(p,q),'parameter changed without Adam step')
            require((old is None and new is None) or all(torch.equal(old[k],new[k]) for k in old),'state changed without step')
            unused+=1;continue
        require(step==step0+1,'nonconsecutive parameter step');steps.append(step)
        m0=old['exp_avg'].double() if old else torch.zeros_like(p)
        v0=old['exp_avg_sq'].double() if old else torch.zeros_like(p)
        m,v=new['exp_avg'].double(),new['exp_avg_sq'].double()
        require(all(bool(torch.isfinite(t).all()) for t in (p,q,m0,v0,m,v)) and bool((v>=0).all()))
        g=(m-beta1*m0)/(1-beta1)
        predicted_v=beta2*v0+(1-beta2)*g.square()
        require(torch.allclose(v,predicted_v,rtol=2e-5,atol=1e-12),'Adam second moment inconsistent')
        expected=p*(1-lr*decay)-lr*(m/(1-beta1**step))/(v.sqrt()/math.sqrt(1-beta2**step)+eps)
        error=float((expected-q).abs().max())
        require(torch.allclose(q,expected,rtol=2e-6,atol=2e-8),'actual AdamW parameter delta differs')
        gradients[name]=g
        details[name]={'step':step,'gradient_norm':float(g.norm()),'actual_delta_norm':float((q-p).norm()),
                       'decay_delta_norm':float((p*lr*decay).norm()),'parameter_max_reconstruction_error':error,
                       'second_moment_max_error':float((v-predicted_v).abs().max())}
    require(steps,'transition has no optimizer updates')
    return {'step':max(steps),'parameters_without_gradient':unused,'parameters':details,'recovered_gradients':gradients,
            'clipped_aggregate_gradient_norm':math.sqrt(math.fsum(float(g.square().sum()) for g in gradients.values()))}
