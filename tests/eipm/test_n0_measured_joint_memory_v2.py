"""Exercise the production step boundary and reject incomplete measured receipts."""
from __future__ import annotations

import contextlib
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from alice_personality.n0.full_envelope_joint_ddp_route_v2 import FullEnvelopeJointDDPRouteV2
from alice_personality.n0.full_envelope_stage_policy_v1 import J3,apply_stage_trainability,resolve_stage_policy
from alice_personality.n0.full_envelope_training_objective_v1 import FullEnvelopeJointTrainingObjectiveV1
from test_n0_full_envelope_joint_ddp_route_v2 import _batches,_registered_fixture,_trainer


ROOT=Path(__file__).resolve().parents[2]


class _SingleProcessAccelerator:
    def no_sync(self,model):
        return contextlib.nullcontext()

    def backward(self,loss):
        loss.backward()

    def clip_grad_norm_(self,parameters,norm):
        return torch.nn.utils.clip_grad_norm_(parameters,norm)

    def reduce(self,value,*,reduction):
        assert reduction=="sum"
        return value


def test_shared_production_optimizer_step_preserves_full_objective_and_state():
    trainer=_trainer()
    fixture=_registered_fixture()
    torch.manual_seed(2901)
    reference_system=fixture._system()
    candidate_system=copy.deepcopy(reference_system)
    apply_stage_trainability(reference_system,stage=J3)
    apply_stage_trainability(candidate_system,stage=J3)
    reference=FullEnvelopeJointDDPRouteV2(
        reference_system,FullEnvelopeJointTrainingObjectiveV1(),
    ).train()
    candidate=FullEnvelopeJointDDPRouteV2(
        candidate_system,FullEnvelopeJointTrainingObjectiveV1(),
    ).train()
    kwargs=dict(backbone_lr=1e-5,interface_lr=5e-5,
                backbone_weight_decay=0.1,interface_weight_decay=0.05)
    ref_opt=trainer.build_optimizer(reference_system,**kwargs)
    new_opt=trainer.build_optimizer(candidate_system,**kwargs)
    ref_sched=trainer.cosine_with_warmup(
        ref_opt,warmup_steps=1,horizon_steps=8,minimum_lr_scale=0.1,
    )
    new_sched=trainer.cosine_with_warmup(
        new_opt,warmup_steps=1,horizon_steps=8,minimum_lr_scale=0.1,
    )
    accelerator=_SingleProcessAccelerator()
    batches=_batches(fixture,reference_system)
    # The registered compiler supplies these counts; the route fixture omits
    # metadata because its earlier tests only exercise the forward boundary.
    for key in ("semantic_operator_compiled","full_fabric_compiled","natural_relation_compiled"):
        batches[key]["metadata"]={"batch_size":1}
    for _ in range(2):
        ref_opt.zero_grad(set_to_none=True)
        numerator={family:torch.zeros(()) for family in resolve_stage_policy(J3).active_macro_families}
        denominator={family:torch.zeros(()) for family in numerator}
        for micro in range(2):
            (reference(**batches,stage=J3)/2).backward()
            observation=reference.take_observation()
            weights=trainer.family_sample_weights(
                observation,mlm_batch=batches["mlm_batch"],
                teacher_batch=batches["teacher_batch"],
                semantic_compiled=batches["semantic_operator_compiled"],
                full_compiled=batches["full_fabric_compiled"],
                natural_compiled=batches["natural_relation_compiled"],
            )
            for family in numerator:
                numerator[family]+=observation["balanced"][f"raw/{family}"].float()*weights[family]
                denominator[family]+=weights[family]
        accelerator.clip_grad_norm_(reference_system.parameters(),1.0)
        ref_opt.step()
        ref_sched.step()
        trainer.globally_observe_effective_batch(
            accelerator=accelerator,objective=reference.objective,
            raw_numerators=numerator,raw_denominators=denominator,
            active_families=resolve_stage_policy(J3).active_macro_families,
        )
        trainer.execute_registered_optimizer_step(
            accelerator=accelerator,route=candidate,unwrapped_route=candidate,
            system=candidate_system,objective=candidate.objective,
            optimizer=new_opt,lr_scheduler=new_sched,next_batch=lambda:batches,
            stage=J3,device=torch.device("cpu"),accumulation_steps=2,
            gradient_clip_norm=1.0,
        )
        for name,expected in reference.state_dict().items():
            torch.testing.assert_close(expected,candidate.state_dict()[name],msg=name)
        for name,expected in reference_system.named_parameters():
            observed=dict(candidate_system.named_parameters())[name]
            if expected.grad is not None:
                torch.testing.assert_close(expected.grad,observed.grad,msg=name)
        assert ref_sched.state_dict()==new_sched.state_dict()


def _complete_memory_receipt():
    semantic=("max_runtime_axes","max_factor_cardinality","long_context_semantic")
    fabric=(
        "max_candidate_cardinality","max_field_cardinality","max_edge_cardinality",
        "max_view_cardinality","max_reasoning_depth","long_additional_view_source",
    )
    step={
        "total_memory_bytes":80_000_000_000,"safety_margin_bytes":1_073_741_824,
        "max_allocated_bytes":16_000_000_000,"max_reserved_bytes":18_000_000_000,
        "max_sampled_device_used_bytes":19_000_000_000,
        "conservative_measured_bytes":20_073_741_824,
        "capacity_85_percent_pass":True,"finite_joint_loss":True,
        "finite_nonzero_gradients":True,"nonzero_public_judgment_gradient":True,
        "weight_and_optimizer_state_changed":True,"microbatch_count":8,
    }
    pairs=[{
        "pair":f"{s}__{f}","all_ten_j3_families_executed":True,
        "complete_backward":True,
        "state_reset_to_identical_initialization":True,
        "steps":[dict(step,optimizer_step_index=1),dict(step,optimizer_step_index=2)],
    } for s in semantic for f in fabric]
    pairs[-1]["checkpoint_resume_memory"]=[
        {**step,"stage":stage}
        for stage in ("checkpoint_save","checkpoint_load","resumed_optimizer_step")
    ]
    return {
        "schema":"alice.eipm.n0.full-envelope-measured-joint-memory.v2",
        "status":"PASS_N0_FULL_ENVELOPE_MEASURED_JOINT_MEMORY_V2",
        "world_size":2,"ddp_replica_wrapped":True,
        "ddp_boundary":"complete_joint_step_v2",
        "stress_pair_count":18,"all_ten_j3_families_executed":True,
        "complete_backward_per_rank":True,"finite_nonzero_gradients_per_rank":True,
        "first_and_later_adamw_steps_measured":True,
        "state_reset_to_identical_initialization_per_pair":True,
        "eight_microbatch_accumulation_measured":True,
        "same_stage_resume_verified":True,
        "checkpoint_resume_capacity_pass_per_rank":True,
        "predecessor_state_transfer_verified":True,
        "stage_dev_selection_claimed":False,
        "capacity_85_percent_pass_per_rank":True,
        "predecessor_no_gradient_forward_verified":True,
        "old_projection_remains_independent":True,
        "projection_is_training_authorization":False,
        "measurement_is_training_authorization":False,
        "diagnostic_weight_updates":True,
        "production_model_training_performed":False,
        "gpu_training_authorized":False,
        "private_identity_data":False,
        "final_results_observed":False,
        "old_projection_status":"FAIL_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1",
        "no_gradient_receipt_sha256":"0"*64,
        "microbatch_size":1,"teacher_batch_size":2,
        "replay_sequence_length":512,"candidate_gradient_accumulation":8,
        "training_mixed_precision":"fp16",
        "ranks":[
            {"rank":rank,"pair_count":18,"same_stage_resume_verified":True,
             "total_memory_bytes":80_000_000_000,"pairs":copy.deepcopy(pairs)}
            for rank in range(2)
        ],
    }


def test_measured_capacity_gate_rejects_bad_rank_step_and_preserves_failed_projection():
    trainer=_trainer()
    receipt=_complete_memory_receipt()
    trainer.require_measured_joint_memory(receipt)
    altered=copy.deepcopy(receipt)
    step=altered["ranks"][1]["pairs"][5]["steps"][1]
    step["max_sampled_device_used_bytes"]=68_000_000_000
    step["conservative_measured_bytes"]=69_073_741_824
    with pytest.raises(SystemExit,match="85%"):
        trainer.require_measured_joint_memory(altered)
    altered=copy.deepcopy(receipt)
    altered["ranks"][0]["pairs"].pop()
    with pytest.raises(SystemExit,match="stress-pair"):
        trainer.require_measured_joint_memory(altered)
    altered=copy.deepcopy(receipt)
    altered["ranks"][1]["pairs"][8]["steps"][0]["nonzero_public_judgment_gradient"]=False
    with pytest.raises(SystemExit,match="incomplete step"):
        trainer.require_measured_joint_memory(altered)
    altered=copy.deepcopy(receipt)
    altered["ranks"][1]["pairs"][-1]["checkpoint_resume_memory"][2]["max_reserved_bytes"]=70_000_000_000
    with pytest.raises(SystemExit,match="checkpoint/resume memory arithmetic"):
        trainer.require_measured_joint_memory(altered)
    altered=copy.deepcopy(receipt)
    altered["ranks"][0]["pairs"][-1].pop("checkpoint_resume_memory")
    with pytest.raises(SystemExit,match="checkpoint/resume memory coverage"):
        trainer.require_measured_joint_memory(altered)
    altered=copy.deepcopy(receipt)
    altered["old_projection_status"]="PASS_N0_FULL_ENVELOPE_MEASURED_JOINT_MEMORY_V2"
    with pytest.raises(SystemExit,match="old projection"):
        trainer.require_measured_joint_memory(altered)


def test_full_route_config_keeps_old_no_gradient_result_independent():
    config=json.loads((ROOT/"configs/eipm/n0/n0_v02_full_envelope_measured_joint_memory_v2.json").read_text())
    assert config["required_world_size"]==2
    assert config["required_stress_pairs"]==18
    assert config["gradient_accumulation_steps"]==8
    assert config["measured_optimizer_steps_per_pair"]==2
    assert math.isclose(config["memory_fraction_max"],0.85)
    assert config["no_gradient_forward_before_backward"] is True
    assert config["stage_dev_selection_claimed"] is False
    assert config["diagnostic_weight_updates_are_training_authorization"] is False
