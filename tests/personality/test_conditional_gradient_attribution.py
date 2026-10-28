"""Gradient attribution must preserve actual native objectives and update meaning."""
from copy import deepcopy
import pytest
import torch
from scripts.eipm.gemma_n0 import conditional_gradient_attribution as api


@pytest.fixture(autouse=True)
def single_thread_native_fixture():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_weighted_components_cover_total_without_accumulating_or_mutating():
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    before=p.detach().clone()
    losses={'preferences':p[0]**2,'candidate_pair_probabilities':p[1]**2,
            'concept_pointers':(p[0]-p[1])**2,'head:values':p.sum()**2}
    weights={'preferences':2.,'concept_pointers':.3}
    total=sum(v*weights.get(k,1.) for k,v in losses.items())
    m=api.measure_objectives(total,losses,[('n2.scalar.weight',p)],loss_weights=weights)
    assert m['values']['preference']==pytest.approx(6.)
    assert m['values']['concept_pointer']==pytest.approx(.3)
    assert m['values']['remaining']==pytest.approx(9.)
    torch.testing.assert_close(m['gradients']['preference'][0],torch.tensor([4.,4.]))
    assert p.grad is None and torch.equal(p,before)
    assert m['decomposition_max_abs_error']<1e-5


def test_pair_cancellation_is_distinct_from_weak_individual_signal():
    first=(torch.tensor([1.,0.]),);second=(torch.tensor([-1.,0.]),)
    value=api.cancellation([first,second])
    assert value['sum_individual_norms']==2 and value['aggregate_norm']==0
    assert value['retained_fraction']==0
    assert api.cancellation([(torch.zeros(2),)])['retained_fraction'] is None


def test_combined_endpoint_cancellation_can_be_hidden_by_separate_views():
    natural=[(torch.tensor([1.,0.]),),(torch.tensor([0.,1.]),)]
    replay=[(torch.tensor([0.,-1.]),),(torch.tensor([-1.,0.]),)]
    result=api.paired_cancellation(natural,replay)
    assert result['natural']['retained_fraction']==pytest.approx(2**-.5)
    assert result['replay']['retained_fraction']==pytest.approx(2**-.5)
    assert result['combined']['retained_fraction']==0


def test_actual_parameter_change_is_not_replaced_with_negative_gradient():
    task=(torch.tensor([1.,0.]),);remaining=(torch.tensor([-2.,0.]),)
    actual=(torch.tensor([.2,0.]),)
    result=api.geometry(task,remaining,actual,{'scalar_head':[0]})
    assert result['all']['task_remainder_cosine']==pytest.approx(-1)
    assert result['all']['task_dot_actual_delta']==pytest.approx(.2)
    assert result['all']['local_actual_update_increases_task_loss'] is True


def test_zero_unused_gradients_and_bad_values_are_explicit():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));unused=torch.nn.Parameter(torch.ones(3))
    loss=p.sum()**2
    m=api.measure_objectives(loss,{'head:stance':loss},[('n2.scalar.weight',p),('n2.other',unused)])
    assert api.vector_norm(m['gradients']['preference'])==0
    assert torch.equal(m['gradients']['total'][1],torch.zeros(3))
    with pytest.raises(ValueError):api.measure_objectives(loss,{'preferences':loss},[('n1.weight',p)])
    with pytest.raises(ValueError):api.geometry((torch.tensor([float('nan')]),),(torch.ones(1),),(torch.ones(1),),{'other':[0]})


def test_actual_native_objective_and_all_parameter_partitions():
    from tests.personality.test_paired_learning_experiment import PairedLearningExperimentTests
    fixture=PairedLearningExperimentTests();fixture.setUp()
    model=deepcopy(fixture.model);model.set_phase('n2',adapt_n1=False)
    fixture.objectives.requires_grad_(False)
    before=deepcopy(model.state_dict())
    batch=fixture.batches[fixture.digests[0]]
    total,losses,_=fixture.objectives(model(batch.frame),batch,loss_weights=fixture.recipe.loss_weights)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    measured=api.measure_objectives(total,losses,named,loss_weights=fixture.recipe.loss_weights)
    partitions=api.parameter_partitions([n for n,p in named])
    assert sorted(i for values in partitions.values() for i in values)==list(range(len(named)))
    assert api.native_gradient_norm(measured['gradients']['total'])>0
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[n],v) for n,v in model.state_dict().items())


def test_saved_real_group_reproduces_losses_norm_without_update(tmp_path):
    from tests.personality.test_paired_learning_experiment import PairedLearningExperimentTests
    from tests.personality.test_conditional_learning_experiment import run
    fixture=PairedLearningExperimentTests();fixture.setUp()
    saved=run(fixture,tmp_path/'source')
    post=torch.load(tmp_path/'source/group-0003/experimental-progress.pt',weights_only=True)['arms']['intervention']
    model=deepcopy(fixture.model);model.set_phase('n2',adapt_n1=False)
    objective=deepcopy(fixture.objectives);objective.requires_grad_(False)
    result=api.measure_group(model,objective,fixture.recipe,saved['arms']['intervention']['events'][0],post['model_state'],
        batch_loader=fixture.batches.__getitem__,validate_batch=lambda b:b.fingerprint(),
        correction_forward=lambda m,b:m(b.frame))
    assert result['original_losses_reproduced_exactly'] and result['new_optimizer_steps']==0
    assert set(result['broader_pair_cancellation'])=={'natural','replay','combined','correction_difference'}
    assert all(row['natural_replay_preference_cancellation']['retained_fraction']==pytest.approx(1.) for row in result['endpoints'])
    assert result['summed_gradient_norm']==pytest.approx(result['original_gradient_norm'],rel=api.RTOL,abs=api.ATOL)
    assert all(p.grad is None for p in model.parameters())
    bad=deepcopy(saved['arms']['intervention']['events'][0]);bad['endpoints'][0]['views']['natural']['loss']+=1.
    with pytest.raises(ValueError,match='did not reproduce'):
        api.measure_group(model,objective,fixture.recipe,bad,post['model_state'],batch_loader=fixture.batches.__getitem__,
            validate_batch=lambda b:b.fingerprint(),correction_forward=lambda m,b:m(b.frame))


def test_actual_admitted_correction_replays_original_group(tmp_path):
    from dataclasses import replace
    from tests.personality.test_conditional_teaching import public_fixture
    from scripts.eipm.gemma_n0 import conditional_teaching as teaching
    from scripts.eipm.gemma_n0 import conditional_learning_experiment as experiment
    from src.alice_personality.identity import IdentityModel
    from src.alice_personality.teaching import IdentityObjectives,TrainingRecipe
    config,n1,n2,b1,_,binding,materialize=public_fixture()
    model=IdentityModel(config);objectives=IdentityObjectives(config.learned_width)
    recipe=TrainingRecipe('n2',.003,1.,61);batches={};corrections={}
    for index in range(4):
        case=replace(n2,case_id='corrected-public-'+str(index))
        batch=materialize(case);digest=batch.fingerprint();batches[digest]=batch
        corrections[digest]=teaching.build_correction_record(n1,case,b1,batch,frame_binding=binding,
            reviewed_n1_batches={b1.fingerprint():{'split':'TRAIN','case_ids':list(b1.case_ids),
                'source_family_ids':[list(row) for row in b1.source_family_ids]}})
    digests=list(batches)
    plan=experiment.make_plan(model,objectives,recipe,batch_sha256=digests,matched_groups=[digests[:2],digests[2:]],
        correction_sha256={d:teaching.record_sha256(r) for d,r in corrections.items()},bindings={'public_fixture':True})
    result=experiment.run_experiment(model,objectives,recipe,plan=plan,expected_plan_sha256=experiment.plan_sha256(plan),
        batch_loader=batches.__getitem__,validate_batch=lambda b:b.fingerprint(),correction_loader=corrections.get,
        custody_guard=lambda:None,progress_directory=tmp_path/'corrected')
    post=torch.load(tmp_path/'corrected/group-0003/experimental-progress.pt',weights_only=True)['arms']['intervention']
    model.set_phase('n2',adapt_n1=False);objectives.requires_grad_(False)
    measured=api.measure_group(model,objectives,recipe,result['arms']['intervention']['events'][0],post['model_state'],
        batch_loader=batches.__getitem__,validate_batch=lambda b:b.fingerprint(),
        correction_forward=lambda m,b:teaching.forward_with_correction(m,b,corrections[b.fingerprint()],
            expected_record_sha256=plan['correction_sha256'][b.fingerprint()],apply_correction=True))
    assert measured['original_losses_reproduced_exactly'] and measured['new_optimizer_steps']==0
    assert any(row['preference_correction_signal']['all']['corrected_minus_natural_norm']>0 for row in measured['endpoints'])


def test_readonly_context_guard_and_worker_never_step():
    import ast
    from pathlib import Path
    from unittest.mock import patch
    from scripts.eipm.gemma_n0 import conditional_learning_readonly_context as context
    from scripts.eipm.gemma_n0 import magnolia_conditional_gradient_attribution_20261028 as worker
    class Refused(Exception):pass
    class Namespace:
        def _p2_guard(self):raise Refused()
    with patch.object(Path,'stat',side_effect=AssertionError('private access before isolation')),patch.object(Path,'open',side_effect=AssertionError('private access before isolation')):
        with pytest.raises(Refused):
            with context.open_context(Namespace()):pass
    for path in (Path(context.__file__),Path(worker.__file__),Path(api.__file__)):
        tree=ast.parse(path.read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','backward') for n in ast.walk(tree))
    tree=ast.parse(Path(context.__file__).read_text())
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_p2_guard' for n in ast.walk(tree))==1
