"""Whole-step gradients and reducer behavior against the registered N0 fixture."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn

from alice_personality.n0.full_envelope_joint_ddp_route_v2 import (
    FullEnvelopeJointDDPRouteV2,
)
from alice_personality.n0.full_envelope_joint_step_v1 import (
    execute_full_envelope_joint_step,
)
from alice_personality.n0.full_envelope_stage_policy_v1 import (
    J1, J2, J3, apply_stage_trainability, resolve_stage_policy,
)
from alice_personality.n0.full_envelope_training_objective_v1 import (
    FullEnvelopeJointTrainingObjectiveV1,
)


def _registered_fixture():
    path=Path(__file__).with_name("test_n0_full_envelope_trainable_system_v1.py")
    spec=importlib.util.spec_from_file_location("n0_registered_fixture",path)
    assert spec is not None and spec.loader is not None
    fixture=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return fixture


def _trainer():
    script=Path(__file__).resolve().parents[2]/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py"
    scripts=str(script.parent)
    sys.path.insert(0,scripts)
    try:
        spec=importlib.util.spec_from_file_location("n0_joint_trainer_route_v2",script)
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def _batches(fixture, system, *, token_shift=0):
    full=fixture._full_batch()
    with torch.no_grad():
        shape=system(task="full_envelope",batch=full)
    operator_targets,behavioral_targets=fixture._training_targets(shape)
    semantic_keys=(
        "query_input_ids","query_attention_mask","relation_input_ids",
        "relation_attention_mask","relation_domain_type_mask",
        "relation_range_type_mask","relation_symmetric","relation_candidate_mask",
        "factor_input_ids","factor_attention_mask","factor_candidate_masks",
        "factor_opcodes","max_reasoning_steps",
    )
    natural_keys=(
        "query_input_ids","query_attention_mask","relation_input_ids",
        "relation_attention_mask","relation_domain_type_mask",
        "relation_range_type_mask","relation_symmetric","relation_candidate_mask",
    )
    mlm_ids,mlm_mask=fixture._tokens(1,length=6,offset=73+token_shift)
    candidate_ids,candidate_mask=fixture._tokens(4,length=7,offset=90+token_shift)
    rationale_ids,rationale_mask=fixture._tokens(2,length=7,offset=120+token_shift)
    return {
        "mlm_batch":{
            "input_ids":mlm_ids,"attention_mask":mlm_mask,"labels":mlm_ids.clone(),
        },
        "teacher_batch":{
            "candidate_input_ids":candidate_ids,
            "candidate_attention_mask":candidate_mask,
            "rationale_input_ids":rationale_ids,
            "rationale_attention_mask":rationale_mask,
            "candidate_rationale_index":torch.tensor([0,0,1,1]),
            "group_sizes":[2,2],
            "preferred_masks":[torch.tensor([True,False]),torch.tensor([False,True])],
            "principle_tags":["principle-a","principle-b"],
            "ids":["teacher-a","teacher-b"],
        },
        "semantic_operator_compiled":{
            "batch":{key:full[key] for key in semantic_keys},
            "operator_targets":operator_targets,
        },
        "full_fabric_compiled":{
            "primary_batch":full,"decisive_ablated_batch":full,
            "irrelevant_removed_batch":full,"permuted_batch":full,
            "behavioral_targets":behavioral_targets,
        },
        "natural_relation_compiled":{
            "batch":{**{key:full[key] for key in natural_keys},"max_reasoning_steps":1},
            "target_relation_index":torch.zeros(1,dtype=torch.long),
        },
    }


def _stage_batches(batches,stage):
    result=dict(batches)
    if stage==J1:
        result["full_fabric_compiled"]=None
    return {**result,"stage":stage}


@pytest.mark.parametrize("stage",[J1,J2,J3])
def test_real_joint_objective_loss_gradients_and_reporting_match(stage):
    fixture=_registered_fixture()
    torch.manual_seed(2901)
    reference_system=fixture._system().train()
    candidate_system=copy.deepcopy(reference_system)
    apply_stage_trainability(reference_system,stage=stage)
    apply_stage_trainability(candidate_system,stage=stage)
    batches=_stage_batches(_batches(fixture,reference_system),stage)
    original_objective=FullEnvelopeJointTrainingObjectiveV1()
    route=FullEnvelopeJointDDPRouteV2(
        candidate_system,FullEnvelopeJointTrainingObjectiveV1(),
    ).train()
    original=execute_full_envelope_joint_step(
        system=reference_system,objective=original_objective,
        update_ema=False,**batches,
    )
    loss=route(**batches)
    observation=route.take_observation()
    assert observation["active_families"]==resolve_stage_policy(stage).active_macro_families
    assert observation["all_active_stage_lanes_executed"] is True
    assert observation["placeholder_losses_used"] is False
    assert observation["requires_full_fabric_counterfactuals"] is (stage==J3)
    assert torch.allclose(loss,original["loss"],rtol=1e-5,atol=1e-6)
    for family in observation["active_families"]:
        raw=f"raw/{family}"
        assert observation["balanced"][raw].grad_fn is None
        assert torch.allclose(observation["balanced"][raw],original["balanced"][raw])
    original["loss"].backward()
    loss.backward()
    for (name,left),(other,right) in zip(
        reference_system.named_parameters(),candidate_system.named_parameters(),strict=True,
    ):
        assert name==other
        assert left.requires_grad==right.requires_grad
        assert (left.grad is None)==(right.grad is None),name
        if left.grad is not None:
            torch.testing.assert_close(left.grad,right.grad,rtol=1e-4,atol=1e-5)
    assert list(reference_system.state_dict())==list(route.system.state_dict())
    assert list(original_objective.state_dict())==list(route.objective.state_dict())


def test_optimizer_groups_and_split_checkpoint_resume_are_preserved(tmp_path):
    fixture=_registered_fixture()
    trainer=_trainer()
    torch.manual_seed(2901)
    system=fixture._system().train()
    apply_stage_trainability(system,stage=J3)
    route=FullEnvelopeJointDDPRouteV2(
        system,FullEnvelopeJointTrainingObjectiveV1(),
    ).train()
    original=copy.deepcopy(route)
    optimizer_kwargs={
        "backbone_lr":1e-5,"interface_lr":5e-5,
        "backbone_weight_decay":0.1,"interface_weight_decay":0.05,
    }
    candidate_opt=trainer.build_optimizer(route.system,**optimizer_kwargs)
    original_opt=trainer.build_optimizer(original.system,**optimizer_kwargs)
    def group_names(model,opt):
        by_id={id(parameter):name for name,parameter in model.named_parameters()}
        return [
            (group["group_role"],group["lr"],group["weight_decay"],
             tuple(by_id[id(p)] for p in group["params"]))
            for group in opt.param_groups
        ]
    assert group_names(route.system,candidate_opt)==group_names(original.system,original_opt)
    assert sum(len(group["params"]) for group in candidate_opt.param_groups)==len(list(route.system.parameters()))
    batches=_stage_batches(_batches(fixture,route.system),J3)
    def step(model,opt):
        opt.zero_grad(set_to_none=True)
        loss=model(**batches)
        observation=model.take_observation()
        loss.backward()
        opt.step()
        model.objective.balancer.observe_detached_family_means(
            {family:observation["balanced"][f"raw/{family}"] for family in observation["active_families"]},
            active_families=observation["active_families"],
        )
    step(route,candidate_opt)
    step(original,original_opt)
    assert list(route.system.state_dict())==list(original.system.state_dict())
    assert list(route.objective.state_dict())==list(original.objective.state_dict())
    checkpoint=tmp_path/"split-checkpoint.pt"
    torch.save({
        "system":route.system.state_dict(),
        "objective":route.objective.state_dict(),
        "optimizer":candidate_opt.state_dict(),
    },checkpoint)
    resumed=FullEnvelopeJointDDPRouteV2(
        fixture._system(),FullEnvelopeJointTrainingObjectiveV1(),
    ).train()
    apply_stage_trainability(resumed.system,stage=J3)
    resumed_opt=trainer.build_optimizer(resumed.system,**optimizer_kwargs)
    snapshot=torch.load(checkpoint,weights_only=True)
    resumed.system.load_state_dict(snapshot["system"],strict=True)
    resumed.objective.load_state_dict(snapshot["objective"],strict=True)
    resumed_opt.load_state_dict(snapshot["optimizer"])
    step(original,original_opt)
    step(resumed,resumed_opt)
    for key,expected in original.system.state_dict().items():
        torch.testing.assert_close(resumed.system.state_dict()[key],expected)
    for key,expected in original.objective.state_dict().items():
        torch.testing.assert_close(resumed.objective.state_dict()[key],expected)


def test_split_predecessor_state_opens_j2_and_j3_without_changing_key_space():
    fixture=_registered_fixture()
    torch.manual_seed(2901)
    predecessor=FullEnvelopeJointDDPRouteV2(
        fixture._system(),FullEnvelopeJointTrainingObjectiveV1(),
    ).train()
    for stage in (J1,J2,J3):
        current=FullEnvelopeJointDDPRouteV2(
            fixture._system(),FullEnvelopeJointTrainingObjectiveV1(),
        ).train()
        current.system.load_state_dict(predecessor.system.state_dict(),strict=True)
        current.objective.load_state_dict(predecessor.objective.state_dict(),strict=True)
        report=apply_stage_trainability(current.system,stage=stage)
        assert report["all_parameters_owned"] is True
        batches=_stage_batches(_batches(fixture,current.system),stage)
        loss=current(**batches)
        observation=current.take_observation()
        assert len(observation["active_families"])==len(resolve_stage_policy(stage).active_macro_families)
        assert observation["requires_full_fabric_counterfactuals"] is (stage==J3)
        loss.backward()
        if stage==J3:
            judgment=next(current.system.stack.public_judgment_probe.parameters())
            assert judgment.grad is not None
            assert bool(torch.isfinite(judgment.grad).all())
        predecessor=current


def _gloo_worker(rank, rendezvous, stage):
    dist.init_process_group("gloo",init_method=f"file://{rendezvous}",rank=rank,world_size=2)
    try:
        fixture=_registered_fixture()
        torch.manual_seed(2901)
        reference=fixture._system().train()
        candidate=copy.deepcopy(reference)
        apply_stage_trainability(reference,stage=stage)
        apply_stage_trainability(candidate,stage=stage)
        ref_objective=FullEnvelopeJointTrainingObjectiveV1()
        route=FullEnvelopeJointDDPRouteV2(
            candidate,FullEnvelopeJointTrainingObjectiveV1(),
        ).train()
        ddp=DDP(route,find_unused_parameters=True)
        batches=_stage_batches(_batches(fixture,reference,token_shift=rank*5),stage)
        for micro in range(2):
            ref_result=execute_full_envelope_joint_step(
                system=reference,objective=ref_objective,
                update_ema=False,**batches,
            )
            (ref_result["loss"]/2).backward()
            if micro==0:
                with ddp.no_sync():
                    distributed_loss=ddp(**batches)
                    (distributed_loss/2).backward()
            else:
                distributed_loss=ddp(**batches)
                (distributed_loss/2).backward()
            torch.testing.assert_close(distributed_loss,ref_result["loss"],rtol=1e-4,atol=1e-5)
            observation=route.take_observation()
            assert observation["all_active_stage_lanes_executed"] is True
            assert observation["placeholder_losses_used"] is False
        for (name,local),(other,distributed) in zip(
            reference.named_parameters(),candidate.named_parameters(),strict=True,
        ):
            assert name==other
            assert (local.grad is None)==(distributed.grad is None),name
            if local.grad is not None:
                averaged=local.grad.clone()
                dist.all_reduce(averaged)
                averaged/=2
                torch.testing.assert_close(distributed.grad,averaged,rtol=1e-4,atol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("stage",[J1,J2,J3])
def test_two_rank_gloo_complete_joint_step_with_accumulation(tmp_path,stage):
    rendezvous=tmp_path/(stage+".rendezvous")
    spawn(_gloo_worker,args=(str(rendezvous),stage),nprocs=2,join=True)
