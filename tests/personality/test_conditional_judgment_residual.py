"""Experimental conditional access must preserve the complete native contract."""
from copy import deepcopy
from dataclasses import replace
import importlib
import pytest
import torch

from src.alice_personality.identity.codec import fingerprint
from tests.personality.test_paired_learning_experiment import PairedLearningExperimentTests


@pytest.fixture
def native():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    fixture = PairedLearningExperimentTests()
    fixture.setUp()
    yield fixture
    torch.set_num_threads(threads)


def api():
    return importlib.import_module('scripts.eipm.gemma_n0.conditional_judgment_residual')


def test_zero_gate_preserves_full_packet_existing_tensors_rng_and_gradients(native):
    before = fingerprint(native.model.state_dict())
    rng = torch.get_rng_state().clone()
    migrated, receipt = api().migrate(native.model, seed=713)
    assert torch.equal(rng, torch.get_rng_state())
    assert fingerprint(native.model.state_dict()) == before
    assert receipt['native_checkpoint'] is False
    assert receipt['parent_model_state_sha256'] == before
    assert not any(p.requires_grad for p in migrated.parameters())
    for name, value in native.model.state_dict().items():
        assert torch.equal(value, migrated.state_dict()[name]), name
    batch = native.batches[native.digests[0]]
    with torch.no_grad():
        assert fingerprint(native.model(batch.frame)) == fingerprint(migrated(batch.frame))
    original = deepcopy(native.model)
    original.set_phase('n2')
    migrated.set_phase('n2')
    objectives = deepcopy(native.objectives).requires_grad_(False)
    for model in (original,migrated):
        total,_,_ = objectives(model(batch.frame),batch)
        total.backward()
    for name,p in original.named_parameters():
        q = dict(migrated.named_parameters())[name]
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_masks_empty_memory_permutation_and_magnitude(native):
    reader = api().ConditionalResidual(8,2)
    with torch.no_grad(): reader.gate.fill_(.5)
    candidates = torch.randn(2,3,8)
    memory = torch.randn(2,4,8)
    mask = torch.tensor([[True,False,True],[True,True,False]])
    allowed = torch.tensor([[True,True,False,False],[False,False,False,False]])
    actual = reader(candidates,mask,memory,allowed)
    changed = memory.clone(); changed[~allowed] = 1e5
    torch.testing.assert_close(actual,reader(candidates,mask,changed,allowed),rtol=0,atol=0)
    assert torch.equal(actual[1],torch.zeros_like(actual[1]))
    assert torch.equal(actual[~mask],torch.zeros_like(actual[~mask]))
    order = torch.tensor([2,0,3,1])
    torch.testing.assert_close(actual,reader(candidates,mask,memory[:,order],allowed[:,order]))
    assert not torch.allclose(actual,reader(candidates,mask,memory*3,allowed))
    assert torch.isfinite(actual).all()


def test_gate_then_branch_learns_and_disabled_control_cannot(native):
    reader = api().ConditionalResidual(8,2)
    candidates = torch.randn(1,3,8); memory = torch.randn(1,4,8)
    cm = torch.ones(1,3,dtype=torch.bool); mm = torch.ones(1,4,dtype=torch.bool)
    optimizer = torch.optim.AdamW(reader.parameters(),lr=.01)
    reader(candidates,cm,memory,mm).sum().backward()
    assert reader.gate.grad.abs().item() > 0
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for n,p in reader.named_parameters() if n!='gate')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    reader(candidates,cm,memory,mm).sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for n,p in reader.named_parameters() if n!='gate')
    reader.enabled.fill_(False);reader.zero_grad(set_to_none=True)
    reader(candidates,cm,memory,mm).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in reader.parameters())


def test_authorized_context_changes_every_native_head_and_reload_is_exact(native):
    model,_=api().migrate(native.model,seed=713)
    frame=native.batches[native.digests[0]].frame
    with torch.no_grad():
        representation,query=model.n1(frame)
        baseline=model.n2(frame,representation,query)
        model.n2.conditioning.gate.fill_(.8)
        changed=model.n2(frame,representation,query)
        for key in ('scalar','evidence_logits','concept_logits','voice_confidence'):
            assert not torch.equal(baseline[key],changed[key]),key
        for family in baseline['heads']:
            assert not torch.equal(baseline['heads'][family],changed['heads'][family]),family
        assert torch.equal(baseline['candidate_mask'],changed['candidate_mask'])
        assert torch.equal(baseline['source_mask'],changed['source_mask'])
        # Native view masks, not raw concept/source presence, govern added access.
        unavailable=replace(representation,view_masks={k:torch.zeros_like(v) for k,v in representation.view_masks.items()})
        assert torch.equal(model.n2.conditioning_context(unavailable)[1],torch.zeros_like(model.n2.conditioning_context(unavailable)[1]))
        restored,_=api().migrate(native.model,seed=713)
        restored.load_state_dict(model.state_dict(),strict=True)
        assert fingerprint(model(frame))==fingerprint(restored(frame))
        with pytest.raises(RuntimeError): native.model.load_state_dict(model.state_dict(),strict=True)


def test_different_condition_can_be_learned_without_changing_source_model(native):
    reader=api().ConditionalResidual(8,2)
    candidate=torch.randn(2,1,8)
    candidate[1]=candidate[0]
    memory=torch.randn(2,2,8);memory[1]=-memory[0]
    cm=torch.ones(2,1,dtype=torch.bool);mm=torch.ones(2,2,dtype=torch.bool)
    target=torch.tensor([1.,-1.])
    optimizer=torch.optim.AdamW(reader.parameters(),lr=.02)
    initial=None
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        prediction=reader(candidate,cm,memory,mm)[:,0,0]
        loss=(prediction-target).square().mean()
        if initial is None: initial=float(loss.detach())
        loss.backward();optimizer.step()
    assert loss.item()<initial*.05
    assert prediction[0]>0 and prediction[1]<0


@pytest.mark.parametrize('method',[
    'test_candidate_permutation_is_equivariant_across_packet_heads',
    'test_authority_blocked_candidate_cannot_influence_allowed_choices',
    'test_host_and_self_views_do_not_rewrite_source_identity_or_history',
    'test_synthetic_prior_never_promotes_itself_to_history',
    'test_no_available_candidate_has_zero_mass_and_no_fabricated_decision',
    'test_dynamic_candidates_labels_and_token_lengths_are_not_fixed_classes',
])
def test_active_residual_preserves_existing_native_authority_and_dynamic_contract(native,method):
    from tests.personality.test_identity_model import IdentityModelMechanicalTests
    checks=IdentityModelMechanicalTests();checks.setUp()
    checks.model,_=api().migrate(checks.model,seed=713)
    with torch.no_grad():checks.model.n2.conditioning.gate.fill_(.8)
    getattr(checks,method)()
