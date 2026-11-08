"""Actual native execution/derivatives and actual AdamW, not fabricated results."""
import copy
import importlib
import json
from dataclasses import replace
import pytest
import torch
from src.alice_personality.identity import IdentityModel
from src.alice_personality.identity.codec import decision_to_mapping, fingerprint
from src.alice_personality.teaching import IdentityObjectives
from src.alice_personality.teaching.cases import NamedTarget, TargetFact
from scripts.eipm.gemma_n0.conditional_judgment_residual import migrate
from tests.personality.test_conditional_teaching import public_fixture


def api():
    return importlib.import_module('scripts.eipm.gemma_n0.conditional_residual_saved_attribution')


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('gate',[0.,.0001,.6])
def test_actual_native_saved_replay_targets_and_branch_derivatives(gate):
    torch.manual_seed(73)
    config,_,case,_,_,_,materialize=public_fixture()
    ids=[c.candidate_id for c in case.document.candidates]
    source=case.document.sources[0].record.record_id
    pair=NamedTarget('candidate_pair_probabilities','independent',(
        TargetFact((ids[0],ids[1]),.2,.3,True,(source,),()),
        TargetFact((ids[1],ids[0]),.8,.7,True,(source,),())))
    case=replace(case,targets=(*case.targets,pair))
    batch=materialize(case)
    model,_=migrate(IdentityModel(config),seed=713)
    model.n2.conditioning.gate.data.fill_(gate)
    with torch.no_grad():
        natural=decision_to_mapping(model(batch.frame))
        model.n2.conditioning.enabled.fill_(False)
        disabled=decision_to_mapping(model(batch.frame))
        model.n2.conditioning.enabled.fill_(True)
    # Canonical persistence sorts views; that must not change the native order.
    natural=json.loads(json.dumps(natural,sort_keys=True))
    disabled=json.loads(json.dumps(disabled,sort_keys=True))
    rep=api().representation(natural)
    with torch.no_grad(): expected_rep,_=model.n1(batch.frame)
    torch.testing.assert_close(rep.source_states,expected_rep.source_states,rtol=0,atol=0)
    before=fingerprint(model.state_dict())
    model.n2.conditioning.requires_grad_(True)
    params=tuple(model.n2.conditioning.parameters())
    logits,details=api().replay(model,disabled,natural)
    targets=api().order_targets(case,natural)
    loss,fields=api().order_loss(logits,torch.tensor(natural['candidate_mask']),targets)
    reconstructed=torch.autograd.grad(loss,params)
    packet=model(batch.frame)
    actual,actual_fields,_=IdentityObjectives(config.learned_width)(packet,batch)
    actual_grad=torch.autograd.grad(actual,params)
    torch.testing.assert_close(loss,actual)
    for key in fields:torch.testing.assert_close(fields[key],actual_fields[key])
    for a,b in zip(reconstructed,actual_grad):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
    assert fingerprint(model.state_dict())==before
    assert details['enabled_candidate_max_error']<1e-5
    assert details['grad_enabled'] is True
    corrupted=copy.deepcopy(natural)
    # A masked source copy must not carry data even when another view admits it.
    found=False
    for view,mask in corrupted['native_latent']['view_masks'].items():
        for i,m in enumerate(mask[0][:len(corrupted['sources'][0])]):
            if not m:
                corrupted['native_latent']['views'][view][0][i][0]=99.;found=True;break
        if found:break
    assert found
    with pytest.raises(ValueError):api().representation(corrupted)
    changed=copy.deepcopy(natural);changed['candidate_ids'][0].reverse()
    with pytest.raises(ValueError):api().replay(model,disabled,changed)


def test_adam_moments_recover_actual_clipped_gradient_decay_and_step():
    torch.manual_seed(21)
    params={name:torch.nn.Parameter(torch.randn(3,4)) for name in ('n2.core','n2.conditioning.weight','n2.unused')}
    optimizer=torch.optim.AdamW(list(params.values()),lr=.0003,weight_decay=.01)
    snapshots=[];gradients=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        params['n2.core'].square().sum().backward()
        params['n2.conditioning.weight'].grad=torch.zeros_like(params['n2.conditioning.weight']) if step==1 else torch.randn(3,4)
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        gradients.append({n:p.grad.clone() if p.grad is not None else None for n,p in params.items()})
        optimizer.step()
        snapshots.append({'model_state':{n:p.detach().clone() for n,p in params.items()},'optimizer_state':copy.deepcopy(optimizer.state_dict())})
    for i in (1,2):
        detail=api().adam_transition(snapshots[i-1],snapshots[i],list(params))
        assert detail['parameters_without_gradient']==1
        assert detail['step']==i+1
        for name,value in gradients[i].items():
            if value is not None:torch.testing.assert_close(detail['recovered_gradients'][name],value.double(),atol=2e-7,rtol=2e-5)
    damaged=copy.deepcopy(snapshots[1]);damaged['optimizer_state']['state'][0]['exp_avg_sq'].add_(1.)
    with pytest.raises(ValueError):api().adam_transition(snapshots[0],damaged,list(params))
    damaged=copy.deepcopy(snapshots[1]);damaged['optimizer_state']['state'][0]['step'].add_(1)
    with pytest.raises(ValueError):api().adam_transition(snapshots[0],damaged,list(params))


def test_saved_replay_preserves_native_blocked_candidate_zero_logits():
    torch.manual_seed(78)
    config,_,_,_,batch,_,_=public_fixture()
    mask=batch.frame.candidate_authority_mask.clone();mask[0,-1]=False
    frame=replace(batch.frame,candidate_authority_mask=mask)
    model,_=migrate(IdentityModel(config),seed=713)
    model.n2.conditioning.gate.data.fill_(.6)
    with torch.no_grad():
        model.n2.scalar[-1].bias[0]=1.7
        natural=decision_to_mapping(model(frame))
        model.n2.conditioning.enabled.fill_(False)
        disabled=decision_to_mapping(model(frame))
        model.n2.conditioning.enabled.fill_(True)
    model.n2.conditioning.requires_grad_(True)
    logits,metrics=api().replay(model,disabled,natural)
    assert logits[0,-1]==0 and natural['preferences'][0][-1]==0
    torch.testing.assert_close(logits,torch.tensor(natural['preference_logits']))


def test_worker_isolates_before_private_read_and_never_steps_optimizer():
    import ast
    from pathlib import Path
    from unittest.mock import patch
    module=importlib.import_module('scripts.eipm.gemma_n0.magnolia_residual_saved_attribution_20261108')
    class Refused(Exception):pass
    class Namespace:
        def _p2_guard(self):raise Refused()
    with patch.object(Path,'open',side_effect=AssertionError('pre-isolation')),patch.object(Path,'stat',side_effect=AssertionError('pre-isolation')):
        with pytest.raises(Refused):module.run(Namespace())
    tree=ast.parse(Path(module.__file__).read_text())
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert calls.count('_p2_guard')==1
    assert 'step' not in calls


def test_actual_experiment_snapshot_final_binding_and_adam_reconstruction(tmp_path):
    from tests.personality.test_conditional_residual_experiment import run
    from tests.personality.test_paired_learning_experiment import PairedLearningExperimentTests
    worker=importlib.import_module('scripts.eipm.gemma_n0.magnolia_residual_saved_attribution_20261108')
    fixture=PairedLearningExperimentTests();fixture.setUp()
    fixture.model,fixture.migration=migrate(fixture.model,seed=713)
    result=run(fixture,tmp_path/'progress',destination=tmp_path/'export')
    final=torch.load(tmp_path/'export/experimental.pt',weights_only=True)
    names=[n for n,_ in fixture.model.named_parameters() if n.startswith('n2.')]
    for tag,offset in (('control',0),('intervention',2)):
        arms=[]
        for number in (1,2):
            saved=torch.load(tmp_path/'progress'/f'group-{offset+number:04d}'/'experimental-progress.pt',weights_only=True)
            worker.verify_saved_binding(saved['binding'],final,names)
            changed=copy.deepcopy(saved['binding']);changed['recipe']['learning_rate']*=2
            with pytest.raises(ValueError):worker.verify_saved_binding(changed,final,names)
            arms.append(saved['arms'][tag])
        measured=api().adam_transition(*arms,names)
        expected=arms[1]['events'][-1]['gradient_norm_before_clip']
        expected*=min(1.,fixture.recipe.max_grad_norm/(expected+1e-6))
        assert measured['clipped_aggregate_gradient_norm']==pytest.approx(expected,rel=2e-5,abs=1e-7)
        assert fingerprint(arms[1]['model_state'])==result['arms'][tag]['model_state_sha256']
