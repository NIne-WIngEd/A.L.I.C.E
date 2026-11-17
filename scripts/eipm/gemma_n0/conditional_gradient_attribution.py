"""Read-only derivatives of complete native objectives; never an optimizer."""
import math
import torch

RTOL=1.3e-6
ATOL=1e-5
COMPONENTS=('preference','concept_pointer','remaining')


def require(condition,message='gradient attribution contract differs'):
    if not condition:raise ValueError(message)


def _finite(vectors):
    vectors=tuple(vectors)
    require(vectors and all(isinstance(v,torch.Tensor) and v.is_floating_point()
            and bool(torch.isfinite(v).all()) for v in vectors),'invalid gradient vector')
    return vectors


def add(*vectors):
    require(vectors,'no vectors supplied')
    length=len(_finite(vectors[0]))
    require(all(len(_finite(v))==length for v in vectors),'gradient membership differs')
    require(all(all(a.shape==b.shape and a.dtype==b.dtype and a.device==b.device
                   for a,b in zip(vectors[0],v)) for v in vectors),'gradient geometry differs')
    return tuple(sum((v[i] for v in vectors[1:]),vectors[0][i].clone()) for i in range(length))


def scale(vector,coefficient):
    require(type(coefficient) in (int,float) and math.isfinite(coefficient))
    return tuple(v*coefficient for v in _finite(vector))


def dot(first,second):
    first,second=_finite(first),_finite(second)
    require(len(first)==len(second) and all(a.shape==b.shape for a,b in zip(first,second)))
    return math.fsum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(first,second))


def vector_norm(vector):return math.sqrt(max(0.,dot(vector,vector)))


def native_gradient_norm(vector):
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v,2) for v in _finite(vector)]),2))


def parameter_partitions(names):
    require(names and len(set(names))==len(names) and all(n.startswith('n2.') for n in names))
    result={key:[] for key in ('conditional_projections','shared_readout','scalar_head','other')}
    for index,name in enumerate(names):
        key=('conditional_projections' if name.startswith('n2.conditional_') else
             'shared_readout' if name.startswith(('n2.readout.','n2.layers.')) else
             'scalar_head' if name.startswith('n2.scalar.') else 'other')
        result[key].append(index)
    return result


def measure_objectives(total,losses,named_parameters,*,loss_weights=None,coefficient=1.):
    """Return A/B/C and independent total gradients; leave parameters/grad untouched."""
    require(losses and total.ndim==0 and bool(torch.isfinite(total)) and total.requires_grad)
    names,parameters=zip(*named_parameters)
    require(type(coefficient) in (int,float) and math.isfinite(coefficient) and coefficient>0)
    parameter_partitions(names)
    require(all(p.requires_grad and p.grad is None for p in parameters),'clean trainable N2 parameters required')
    weights=loss_weights or {}
    require(set(weights)<=set(losses),'weight for absent objective')
    require(all(type(w) in (int,float) and math.isfinite(w) and w>=0 for w in weights.values()))
    components={key:[] for key in COMPONENTS}
    for name,value in losses.items():
        require(value.ndim==0 and bool(torch.isfinite(value)),'nonfinite native objective')
        key='preference' if name in ('preferences','candidate_pair_probabilities') else 'concept_pointer' if name=='concept_pointers' else 'remaining'
        components[key].append(value*weights.get(name,1.))
    terms={key:sum(values) if values else total*0 for key,values in components.items()}
    torch.testing.assert_close(sum(terms.values()).detach(),total.detach(),rtol=RTOL,atol=ATOL)
    gradients={}
    for key,value in (*terms.items(),('total',total)):
        values=torch.autograd.grad(value*coefficient,parameters,allow_unused=True,retain_graph=key!='total') if value.requires_grad else (None,)*len(parameters)
        gradients[key]=tuple(v.detach().clone() if v is not None else torch.zeros_like(p) for p,v in zip(parameters,values))
        _finite(gradients[key])
    reconstructed=add(*(gradients[k] for k in COMPONENTS))
    errors=[]
    for actual,expected in zip(reconstructed,gradients['total']):
        torch.testing.assert_close(actual,expected,rtol=RTOL,atol=ATOL)
        errors.append(float((actual-expected).abs().max()))
    require(all(p.grad is None for p in parameters),'measurement accumulated parameter gradients')
    return {'gradients':gradients,'values':{k:float(v.detach()) for k,v in (*terms.items(),('total',total))},
            'decomposition_max_abs_error':max(errors),'tolerance':{'rtol':RTOL,'atol':ATOL}}


def measure_group(model,objectives,recipe,event,post_state,*,batch_loader,validate_batch,
                  correction_forward,on_endpoint=lambda count:None):
    """Replay one actual pre-update group; post_state supplies the real update."""
    from src.alice_personality.identity.codec import fingerprint
    require(model.phase=='n2' and event['arm']=='intervention')
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    require(all(p.grad is None for p in model.parameters()))
    names=[name for name,p in named];partitions=parameter_partitions(names)
    before=fingerprint(model.state_dict());objective_before=fingerprint(objectives.state_dict())
    require(model.state_dict().keys()==post_state.keys())
    for name,value in model.state_dict().items():
        require(value.shape==post_state[name].shape and value.dtype==post_state[name].dtype)
        if not name.startswith('n2.'):require(torch.equal(value,post_state[name]),'saved frozen stage changed')
    delta=tuple(post_state[name].to(p.device)-p.detach() for name,p in named)
    _finite(delta)
    size=event['group_size'];require(size==len(event['endpoints'])==len(event['batch_sha256']) and size>=2)
    require([e['batch_sha256'] for e in event['endpoints']]==event['batch_sha256'])
    require(len(set(event['batch_sha256']))==size)
    coefficient=1/(2*size)
    aggregate={k:tuple(torch.zeros_like(p) for _,p in named) for k in (*COMPONENTS,'total')}
    endpoints=[];pair_views={'natural':[],'replay':[]};residual=0.
    for index,expected in enumerate(event['endpoints']):
        batch=batch_loader(expected['batch_sha256'])
        require(validate_batch(batch)==expected['batch_sha256'])
        views={};gradients={}
        for view in ('natural','replay'):
            packet=model(batch.frame) if view=='natural' else correction_forward(model,batch)
            total,losses,coverage=objectives(packet,batch,loss_weights=recipe.loss_weights)
            saved=expected['views'][view]
            require(saved['coverage']==coverage and saved['objectives'].keys()==losses.keys(),'saved objective coverage differs')
            require(float(total.detach())==saved['loss'] and all(float(value.detach())==saved['objectives'][key] for key,value in losses.items()),
                    'original native loss did not reproduce exactly')
            measured=measure_objectives(total,losses,named,loss_weights=recipe.loss_weights,coefficient=coefficient)
            gradients[view]=measured['gradients'];residual=max(residual,measured['decomposition_max_abs_error'])
            for key,value in gradients[view].items():aggregate[key]=add(aggregate[key],value)
            a,b,c=(gradients[view][k] for k in COMPONENTS)
            views[view]={'loss':saved['loss'],'field_loss':dict(saved['objectives']),
                         'component_values':measured['values'],'preference_geometry':geometry(a,add(b,c),delta,partitions)}
            if index<2:pair_views[view].append(a)
            del packet,total,losses,measured
        difference=add(gradients['replay']['preference'],scale(gradients['natural']['preference'],-1))
        changes={name:{'corrected_minus_natural_norm':vector_norm(tuple(difference[i] for i in indices)),
                       'natural_preference_norm':vector_norm(tuple(gradients['natural']['preference'][i] for i in indices))}
                 for name,indices in {'all':list(range(len(names))),**partitions}.items() if indices}
        endpoints.append({'batch_sha256':expected['batch_sha256'],'views':views,'preference_correction_signal':changes,
            'natural_replay_preference_cancellation':cancellation([gradients['natural']['preference'],gradients['replay']['preference']])})
        require(validate_batch(batch)==expected['batch_sha256'])
        del batch,gradients,difference
        on_endpoint(index+1)
    norm=native_gradient_norm(aggregate['total'])
    require(math.isclose(norm,event['gradient_norm_before_clip'],rel_tol=RTOL,abs_tol=ATOL),'original accumulated gradient norm differs')
    for actual,expected in zip(add(*(aggregate[k] for k in COMPONENTS)),aggregate['total']):
        torch.testing.assert_close(actual,expected,rtol=RTOL,atol=ATOL)
        residual=max(residual,float((actual-expected).abs().max()))
    require(fingerprint(model.state_dict())==before and fingerprint(objectives.state_dict())==objective_before)
    require(all(p.grad is None for p in model.parameters()),'measurement accumulated gradients')
    a,b,c=(aggregate[k] for k in COMPONENTS)
    return {'group':event['update'],'model_state_sha256':before,'post_state_sha256':fingerprint(post_state),
        'endpoints':endpoints,'endpoint_view_coefficient':coefficient,'original_losses_reproduced_exactly':True,
        'summed_gradient_norm':norm,'original_gradient_norm':event['gradient_norm_before_clip'],
        'decomposition_max_abs_error':residual,'tolerance':{'rtol':RTOL,'atol':ATOL},
        'parameter_partitions':{k:[names[i] for i in indices] for k,indices in partitions.items()},
        'preference_vs_other_geometry':geometry(a,add(b,c),delta,partitions),
        'concept_vs_remaining_geometry':geometry(b,c,delta,partitions),
        'broader_pair_cancellation':paired_cancellation(pair_views['natural'],pair_views['replay']),
        'new_optimizer_steps':0,'derivatives_are_local_sensitivity_not_actual_loss_changes':True}


def cancellation(vectors):
    require(vectors,'empty cancellation comparison')
    individual=math.fsum(vector_norm(v) for v in vectors)
    aggregate=vector_norm(add(*vectors))
    return {'sum_individual_norms':individual,'aggregate_norm':aggregate,
            'retained_fraction':aggregate/individual if individual else None}


def paired_cancellation(natural,replay):
    require(len(natural)==len(replay)==2,'two actual context endpoints required')
    return {'natural':cancellation(natural),'replay':cancellation(replay),
            'combined':cancellation([add(a,b) for a,b in zip(natural,replay)]),
            'correction_difference':cancellation([add(b,scale(a,-1)) for a,b in zip(natural,replay)])}


def geometry(task,remainder,actual_delta,partitions):
    """Positive task dot actual delta means a first-order increase, not observed loss."""
    _finite(task);_finite(remainder);_finite(actual_delta)
    require(sorted(i for indices in partitions.values() for i in indices)==list(range(len(task))),
            'parameter partitions do not cover the complete vector exactly once')
    result={}
    for name,indices in {'all':list(range(len(task))),**partitions}.items():
        if not indices:
            result[name]={'parameter_tensors':0,'task_norm':0.,'remainder_norm':0.,'actual_delta_norm':0.,
                          'task_remainder_cosine':None,'task_dot_actual_delta':0.,'local_actual_update_increases_task_loss':False}
            continue
        a,b,d=(tuple(v[i] for i in indices) for v in (task,remainder,actual_delta))
        an,bn,dn=(vector_norm(v) for v in (a,b,d));direction=dot(a,d)
        result[name]={'parameter_tensors':len(indices),'task_norm':an,'remainder_norm':bn,'actual_delta_norm':dn,
            'task_remainder_cosine':dot(a,b)/(an*bn) if an and bn else None,
            'task_dot_actual_delta':direction,'local_actual_update_increases_task_loss':direction>0}
    return result
