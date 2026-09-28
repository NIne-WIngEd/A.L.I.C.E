#!/usr/bin/env python3
"""Measure the intact J3 joint step; write no training authority on partial proof.

Run after the separately recorded, two-rank, no-gradient P43 diagnostic on the
same immutable source and public input closure. Its failed *projection* remains
a failure. Actual optimizer-step capacity is qualified independently at the
same 85% per-rank limit. All weight updates here are disposable diagnostics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
from alice_personality.n0.full_envelope_behavioral_batch_v1 import compile_behavioral_batch
from alice_personality.n0.full_envelope_joint_ddp_route_v2 import FullEnvelopeJointDDPRouteV2
from alice_personality.n0.full_envelope_runtime_factory_v1 import load_registered_full_envelope_system
from alice_personality.n0.full_envelope_stage_policy_v1 import J1, J2, J3, apply_stage_trainability, resolve_stage_policy
from alice_personality.n0.full_envelope_training_objective_v1 import FullEnvelopeJointTrainingObjectiveV1
from alice_personality.n0.natural_relation_batch_v1 import compile_natural_relation_batch
from alice_personality.n0.semantic_operator_batch_v1 import compile_semantic_operator_batch
from alice_personality.n0.source_authority_v1 import require_canonical_source_file
from alice_personality.n0.v02_training import (
    TeacherMultitaskCollator,sha256_file,verify_public_corpus_v021,verify_teacher_registry,
)
from qualify_n0_v02_full_envelope_gpu_memory_v1 import (
    choose_full_fabric_cases,choose_long_semantic,choose_max_factor_semantic,
    choose_natural,choose_semantic,read_jsonl,train_rows,
)
from train_n0_v02_full_envelope_joint_v1 import (
    build_optimizer,cosine_with_warmup,current_git_revision,
    enable_precommitted_gradient_checkpointing,execute_registered_optimizer_step,
    read_json,recursive_to_device,require_clean_worktree,
)


PASS_MEMORY="PASS_N0_FULL_ENVELOPE_MEASURED_JOINT_MEMORY_V2"
PASS_ROUTE="PASS_N0_COMPLETE_JOINT_ROUTE_GRADIENT_OPTIMIZER_RESUME_V2"
CONFIG="configs/eipm/n0/n0_v02_full_envelope_measured_joint_memory_v2.json"
OLD_CONFIG="configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json"


def required_pairs(config: Mapping[str,Any]) -> tuple[str,...]:
    semantic=config.get("required_semantic_cases")
    fabric=config.get("required_full_fabric_cases")
    if semantic!=["max_runtime_axes","max_factor_cardinality","long_context_semantic"]:
        raise SystemExit("measured route lost a semantic stress case")
    if fabric!=[
        "max_candidate_cardinality","max_field_cardinality","max_edge_cardinality",
        "max_view_cardinality","max_reasoning_depth","long_additional_view_source",
    ]:
        raise SystemExit("measured route lost a full-fabric stress case")
    pairs=tuple(f"{s}__{f}" for s in semantic for f in fabric)
    if len(pairs)!=18 or config.get("required_stress_pairs")!=18:
        raise SystemExit("measured route needs every semantic × fabric pair")
    if int(config.get("required_world_size",0))!=2:
        raise SystemExit("measured route needs two independent DDP replicas")
    if int(config.get("gradient_accumulation_steps",0))!=8:
        raise SystemExit("measured route needs the full eight-microbatch step")
    if int(config.get("measured_optimizer_steps_per_pair",0))!=2:
        raise SystemExit("measure first and later AdamW steps for every pair")
    if float(config.get("memory_fraction_max",0))!=0.85:
        raise SystemExit("original 85% per-rank capacity threshold drift")
    if int(config.get("device_memory_safety_margin_bytes",0))!=1073741824:
        raise SystemExit("one-GiB memory safety margin drift")
    for field in (
        "no_gradient_forward_before_backward","first_and_later_adamw_required",
        "same_stage_optimizer_scheduler_resume_required","predecessor_state_transfer_required",
    ):
        if config.get(field) is not True:
            raise SystemExit(f"measured route lost {field}")
    for field in (
        "stage_dev_selection_claimed","private_identity_data","final_results_observed",
        "production_checkpoint_selection","diagnostic_weight_updates_are_training_authorization",
    ):
        if config.get(field) is not False:
            raise SystemExit(f"measured route cannot claim {field}")
    if len(resolve_stage_policy(J3).active_macro_families)!=10:
        raise SystemExit("measured route lost a registered J3 family")
    return pairs


def validate_no_gradient_receipt(
    receipt: Mapping[str,Any],*,config: Mapping[str,Any],args: argparse.Namespace,
    revision: str,pairs: tuple[str,...],
) -> dict[str,Any]:
    """Admit original projection FAIL as data, never as authorization."""
    if receipt.get("schema")!="alice.eipm.n0.full-envelope-gpu-memory-result.v1":
        raise SystemExit("predecessor no-gradient receipt schema drift")
    if receipt.get("status") not in {
        "PASS_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1",
        "FAIL_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1",
    }:
        raise SystemExit("predecessor no-gradient result missing")
    expected={
        "source_revision":revision,
        "qualifier_sha256":sha256_file(Path(__file__).resolve().with_name(
            "qualify_n0_v02_full_envelope_gpu_memory_v1.py")),
        "qualification_config_sha256":sha256_file(args.old_qualification_config),
        "registered_topology_sha256":sha256_file(args.topology_config),
        "semantic_config_sha256":sha256_file(args.semantic_config),
        "semantic_checkpoint_sha256":sha256_file(args.semantic_checkpoint),
        "tokenizer_json_sha256":sha256_file(Path(args.tokenizer_dir)/"tokenizer.json"),
        "source_config_sha256":sha256_file(args.source_config),
        "corpus_receipt_sha256":sha256_file(Path(args.corpus_dir)/"corpus_receipt.json"),
        "mixture_manifest_sha256":sha256_file(args.mixture_manifest),
        "mixture_audit_sha256":sha256_file(args.mixture_audit),
        "teacher_registry_sha256":sha256_file(args.teacher_registry),
        "teacher_audit_sha256":sha256_file(args.teacher_audit),
        "stage":J3,"world_size":2,"ddp_replica_wrapped":True,
        "ddp_boundary":"complete_joint_step_v2",
        "stress_pair_count":18,"finite_joint_loss":True,
        "full_j3_counterfactual_path":True,
        "real_optimizer_facing_public_lanes":True,
        "gradient":False,"backward":False,"optimizer_object_created":False,
        "weight_update":False,"projection_is_training_authorization":False,
        "gpu_training_authorized":False,"private_identity_data":False,
        "final_results_observed":False,
    }
    for name,value in expected.items():
        if receipt.get(name)!=value:
            raise SystemExit(f"predecessor no-gradient {name} drift")
    if set(receipt.get("stress_pair_receipts") or {})!=set(pairs):
        raise SystemExit("predecessor no-gradient stress pair coverage drift")
    if receipt.get("training_mixed_precision")!="fp16":
        raise SystemExit("predecessor precision drift")
    old=read_json(args.old_qualification_config)
    if old["route"]["initial_gradient_accumulation_operating_point"]!=8:
        raise SystemExit("predecessor accumulation drift")
    if old["memory_projection"]["projected_training_fraction_max"]!=0.85:
        raise SystemExit("predecessor 85% threshold drift")
    ranks=receipt.get("ranks") or []
    if len(ranks)!=2 or sorted(int(row.get("rank",-1)) for row in ranks)!=[0,1]:
        raise SystemExit("predecessor no-gradient rank coverage drift")
    for row in ranks:
        total=int(row["total_memory_bytes"])
        if total<=0 or int(row["peak_reserved_bytes"])>math.floor(0.85*total):
            raise SystemExit("no-gradient forward itself exceeds 85% on a rank")
        if int(row["peak_allocated_bytes"])>int(row["peak_reserved_bytes"]):
            raise SystemExit("predecessor allocator peaks inconsistent")
    return {"old_projection_status":receipt["status"],"ranks":ranks}


def digest_state(model: torch.nn.Module,optimizer: Any,scheduler: Any) -> str:
    """Stream state through CPU one tensor at a time, avoiding an extra GPU copy."""
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value=tensor.detach().contiguous().cpu()
        digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    for group in optimizer.param_groups:
        digest.update(str((group["group_role"],group["lr"],group["weight_decay"])).encode())
        for parameter in group["params"]:
            for key,value in sorted(optimizer.state.get(parameter,{}).items()):
                digest.update(str(key).encode())
                if isinstance(value,torch.Tensor):
                    cpu=value.detach().contiguous().cpu()
                    digest.update(cpu.numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
    digest.update(json.dumps(scheduler.state_dict(),sort_keys=True,default=str).encode())
    return digest.hexdigest()


class MemoryObserver:
    def __init__(self,*,device: torch.device,margin: int) -> None:
        self.device=device
        self.total=int(torch.cuda.get_device_properties(device).total_memory)
        self.margin=margin
        self.stages=[]
        self.nonzero_public_judgment=False
        self.finite_nonzero_gradients=False

    def __call__(self,stage: str,*,micro: int,loss: float,result: Mapping[str,Any]) -> None:
        torch.cuda.synchronize(self.device)
        if len(result["active_families"])!=10 or result["placeholder_losses_used"] is not False:
            raise RuntimeError("J3 joint observation did not cover all ten families")
        if not math.isfinite(loss):
            raise RuntimeError("nonfinite J3 joint loss")
        if stage=="pre_clip":
            # The named public judgment weight failed on both old DDP ranks.
            route=self.model
            finite=False
            judgment=False
            for name,parameter in route.named_parameters():
                grad=parameter.grad
                if not parameter.requires_grad or grad is None:
                    continue
                if not bool(torch.isfinite(grad).all()):
                    raise RuntimeError(f"nonfinite named gradient: {name}")
                if bool(torch.count_nonzero(grad).item()):
                    finite=True
                    if "stack.public_judgment_probe.score.2.weight" in name:
                        judgment=True
            if not finite or not judgment:
                raise RuntimeError("missing nonzero gradient or named public judgment gradient")
            self.finite_nonzero_gradients=True
            self.nonzero_public_judgment=True
        free,_=torch.cuda.mem_get_info(self.device)
        self.stages.append({
            "stage":stage,"micro":micro,
            "allocated_bytes":int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes":int(torch.cuda.memory_reserved(self.device)),
            "device_used_bytes":self.total-int(free),
        })

    def summary(self) -> dict[str,Any]:
        torch.cuda.synchronize(self.device)
        free,_=torch.cuda.mem_get_info(self.device)
        allocated=int(torch.cuda.max_memory_allocated(self.device))
        reserved=int(torch.cuda.max_memory_reserved(self.device))
        used=max(self.total-int(free),*(p["device_used_bytes"] for p in self.stages))
        measured=max(allocated,reserved,used+self.margin)
        return {
            "max_allocated_bytes":allocated,"max_reserved_bytes":reserved,
            "max_sampled_device_used_bytes":used,"safety_margin_bytes":self.margin,
            "conservative_measured_bytes":measured,
            "total_memory_bytes":self.total,
            "capacity_fraction":measured/self.total,
            "capacity_85_percent_pass":measured<=math.floor(0.85*self.total),
            "finite_nonzero_gradients":self.finite_nonzero_gradients,
            "nonzero_public_judgment_gradient":self.nonzero_public_judgment,
            "stages":self.stages,
        }


def checkpoint_memory_snapshot(device: torch.device,margin: int,stage: str) -> dict[str,Any]:
    """Include checkpoint transport in the same device capacity contract."""
    torch.cuda.synchronize(device)
    total=int(torch.cuda.get_device_properties(device).total_memory)
    free,_=torch.cuda.mem_get_info(device)
    allocated=int(torch.cuda.max_memory_allocated(device))
    reserved=int(torch.cuda.max_memory_reserved(device))
    used=total-int(free)
    conservative=max(allocated,reserved,used+margin)
    return {
        "stage":stage,"total_memory_bytes":total,
        "max_allocated_bytes":allocated,"max_reserved_bytes":reserved,
        "max_sampled_device_used_bytes":used,"safety_margin_bytes":margin,
        "conservative_measured_bytes":conservative,
        "capacity_85_percent_pass":conservative<=math.floor(0.85*total),
    }


def runtime_build(device: torch.device) -> dict[str,Any]:
    packages={}
    for package in ("accelerate","transformers","tokenizers","safetensors","datasets"):
        try:
            packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package]=None
    try:
        driver=subprocess.run(
            ["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
            capture_output=True,text=True,check=False,timeout=10,
        )
        driver_versions=driver.stdout.strip().splitlines() if driver.returncode==0 else []
    except (FileNotFoundError,subprocess.TimeoutExpired):
        driver_versions=[]
    return {
        "python":platform.python_version(),"torch":torch.__version__,
        "cuda_build":torch.version.cuda,"nccl":str(torch.cuda.nccl.version()),
        "cudnn":torch.backends.cudnn.version(),
        "device_name":torch.cuda.get_device_name(device),
        "device_compute_capability":list(torch.cuda.get_device_capability(device)),
        "driver_versions":driver_versions,
        "packages":packages,
    }


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    for name in (
        "qualification-config","old-qualification-config","no-gradient-receipt",
        "training-plan","topology-config","semantic-config","semantic-checkpoint",
        "tokenizer-dir","corpus-dir","source-config","teacher-registry","teacher-audit",
        "semantic-rows","semantic-long-rows","behavioral-rows","runtime-view-rows",
        "long-context-rows","fewrel-rows","fewrel-bank","mixture-manifest",
        "mixture-audit","output-dir",
    ):
        p.add_argument("--"+name,required=True)
    return p


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str,Any],dict[str,Any],tuple[str,...],str]:
    require_clean_worktree()
    for path,relative in (
        (args.qualification_config,CONFIG),
        (args.old_qualification_config,OLD_CONFIG),
        (args.training_plan,"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json"),
        (args.topology_config,"configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json"),
        (args.semantic_config,"configs/eipm/n0/alice_n0_semantic_v0.2.json"),
        (args.source_config,"configs/eipm/n0/public_corpus_v0.2.1.activated.json"),
    ):
        require_canonical_source_file(path,relative,label=relative)
    revision=current_git_revision()
    config=read_json(args.qualification_config)
    if config.get("schema")!="alice.eipm.n0.full-envelope-measured-joint-memory.v2":
        raise SystemExit("measured joint memory config schema drift")
    pairs=required_pairs(config)
    mixture=read_json(args.mixture_manifest)
    audit=read_json(args.mixture_audit)
    if mixture.get("source_revision")!=revision or audit.get("source_revision")!=revision:
        raise SystemExit("measured route mixture not bound to checked-out source")
    if mixture.get("status")!="MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT":
        raise SystemExit("measured route public mixture not materialized")
    if audit.get("status")!="PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1":
        raise SystemExit("measured route mixture audit not PASS")
    if audit.get("manifest_sha256")!=sha256_file(args.mixture_manifest):
        raise SystemExit("measured route mixture audit hash drift")
    if mixture.get("private_identity_data") is not False or mixture.get("final_results_observed") is not False:
        raise SystemExit("measured route cannot access private identity or FINAL")
    if int(mixture.get("final_rows_in_training",-1))!=0:
        raise SystemExit("FINAL rows entered measured route mixture")
    lanes=mixture.get("training_lanes") or {}
    lane_paths={
        "semantic_operator_intervention":args.semantic_rows,
        "semantic_operator_long_context":args.semantic_long_rows,
        "full_envelope_behavioral":args.behavioral_rows,
        "runtime_view_supplement":args.runtime_view_rows,
        "long_context_supplement":args.long_context_rows,
        "natural_relation":args.fewrel_rows,
    }
    for name,path in lane_paths.items():
        if (lanes.get(name) or {}).get("rows_sha256")!=sha256_file(path):
            raise SystemExit(f"measured route lane hash drift: {name}")
    if (lanes.get("natural_relation") or {}).get("bank_sha256")!=sha256_file(args.fewrel_bank):
        raise SystemExit("measured route natural bank hash drift")
    if (lanes.get("broad_semantic_replay") or {}).get("corpus_receipt_sha256")!=sha256_file(
        Path(args.corpus_dir)/"corpus_receipt.json"
    ):
        raise SystemExit("measured route broad replay corpus receipt drift")
    teacher_lane=lanes.get("governed_judgment_replay") or {}
    if teacher_lane.get("teacher_registry_sha256")!=sha256_file(args.teacher_registry):
        raise SystemExit("measured route teacher registry drift")
    if teacher_lane.get("teacher_audit_sha256")!=sha256_file(args.teacher_audit):
        raise SystemExit("measured route teacher audit drift")
    old_receipt=read_json(args.no_gradient_receipt)
    no_grad=validate_no_gradient_receipt(
        old_receipt,config=config,args=args,revision=revision,pairs=pairs,
    )
    plan=read_json(args.training_plan)
    operating=(plan.get("optimization_strategy") or {})
    if plan.get("schema")!="alice.eipm.n0.semantic-operator-joint-training-plan.v1":
        raise SystemExit("measured route training plan schema drift")
    if operating.get("gradient_checkpointing") is not True:
        raise SystemExit("precommitted gradient checkpointing required")
    if operating.get("runtime_hyperparameters_fixed_before_gradient") is not True:
        raise SystemExit("precommitted optimizer operating points missing")
    old_cfg=read_json(args.old_qualification_config)["route"]
    if old_cfg["teacher_batch_size"]!=2 or old_cfg["microbatch_size"]!=1:
        raise SystemExit("teacher or lane microbatch drift")
    if old_cfg["replay_sequence_length"]!=512 or old_cfg["training_mixed_precision"]!="fp16":
        raise SystemExit("replay length or precision drift")
    return config,no_grad,pairs,revision


def prepare_case_inputs(args: argparse.Namespace,config: Mapping[str,Any]) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any],Any]:
    """Select the identical public case geometry used in P43 without training data substitution."""
    from alice_personality.n0.v02_training import verify_tokenizer_v021
    verify_tokenizer_v021(args.tokenizer_dir)
    tokenizer=load_tokenizer(args.tokenizer_dir)
    corpus_receipt,corpus_paths=verify_public_corpus_v021(args.corpus_dir,args.source_config)
    teacher_paths,teacher_report=verify_teacher_registry(
        Path(__file__).resolve().parents[3],args.teacher_registry,args.teacher_audit,
    )
    if corpus_receipt.get("status")!="PASS" or teacher_report.get("status")!="PASS":
        raise SystemExit("measured route public input revalidation failed")
    semantic_rows=train_rows(args.semantic_rows)
    semantic_long=train_rows(args.semantic_long_rows)
    behavioral=train_rows(args.behavioral_rows)
    runtime=train_rows(args.runtime_view_rows)
    long_rows=train_rows(args.long_context_rows)
    natural_rows=train_rows(args.fewrel_rows)
    relation_bank=read_json(args.fewrel_bank)
    semantic_cases={
        "max_runtime_axes":choose_semantic(semantic_rows),
        "max_factor_cardinality":choose_max_factor_semantic(semantic_rows),
        "long_context_semantic":choose_long_semantic(semantic_long),
    }
    full_cases=choose_full_fabric_cases(behavioral,runtime,long_rows)
    if set(semantic_cases)!=set(config["required_semantic_cases"]):
        raise SystemExit("semantic stress set drift")
    if set(full_cases)!=set(config["required_full_fabric_cases"]):
        raise SystemExit("fabric stress set drift")
    semantic_compiled={name:compile_semantic_operator_batch(
        rows=[row],tokenizer=tokenizer,
    ) for name,row in semantic_cases.items()}
    full_compiled={name:compile_behavioral_batch(
        rows=[row],tokenizer=tokenizer,
    ) for name,row in full_cases.items()}
    natural_compiled=compile_natural_relation_batch(
        rows=[choose_natural(natural_rows)],relation_bank=relation_bank,tokenizer=tokenizer,
    )
    rank=int(os.environ["RANK"])
    mlm=PackedJSONLIterableDataset(
        paths=corpus_paths,tokenizer=tokenizer,sequence_length=512,
        split="train",shuffle_seed=20260922+rank,
    )
    mlm_loader=DataLoader(
        mlm,batch_size=1,collate_fn=SpanMLMCollator(
            tokenizer=tokenizer,mlm_probability=0.30,mean_span=3.0,max_span=10,
            seed=20260922+rank,
        ),num_workers=0,
    )
    teacher_loader=DataLoader(
        CurriculumDataset(teacher_paths,"train"),batch_size=2,shuffle=False,
        collate_fn=TeacherMultitaskCollator(tokenizer,256),num_workers=0,
    )
    common={
        "mlm_batch":next(iter(mlm_loader)),
        "teacher_batch":next(iter(teacher_loader)),
        "natural_relation_compiled":natural_compiled,
    }
    return semantic_compiled,full_compiled,common,semantic_cases,full_cases


def predecessor_state_transfer(args: argparse.Namespace,checkpoint: Mapping[str,torch.Tensor],objective_state: Mapping[str,torch.Tensor]) -> dict[str,Any]:
    """Prove full key-space transfer; production DEV selection remains a later gate."""
    fresh,_=load_registered_full_envelope_system(
        topology_path=args.topology_config,semantic_config_path=args.semantic_config,
        semantic_checkpoint_path=args.semantic_checkpoint,device="cpu",runtime_profile=None,
    )
    original=set(checkpoint)
    reports={}
    for stage in (J1,J2,J3):
        fresh.load_state_dict(checkpoint,strict=True)
        report=apply_stage_trainability(fresh,stage=stage)
        if report["all_parameters_owned"] is not True or set(fresh.state_dict())!=original:
            raise RuntimeError(f"predecessor state transfer lost full topology at {stage}")
        objective=FullEnvelopeJointTrainingObjectiveV1()
        objective.load_state_dict(objective_state,strict=True)
        opt=build_optimizer(
            fresh,backbone_lr=1e-5,interface_lr=5e-5,
            backbone_weight_decay=0.1,interface_weight_decay=0.05,
        )
        if opt.state:
            raise RuntimeError("new stage inherited predecessor optimizer moments")
        reports[stage]={
            "topology_parameter_count":report["topology_parameter_count"],
            "trainable_parameter_count":report["trainable_parameter_count"],
            "all_parameters_owned":True,
            "optimizer_moments_carried":False,
            "full_state_load_strict":True,
        }
    return reports


def cpu_state(value: Any) -> Any:
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:cpu_state(item) for key,item in value.items()}
    if isinstance(value,list):
        return [cpu_state(item) for item in value]
    if isinstance(value,tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def assert_state_close(left: Any,right: Any,*,label: str) -> None:
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left.cpu(),right.cpu(),rtol=1e-4,atol=1e-5,msg=label)
    elif isinstance(left,dict):
        if set(left)!=set(right):
            raise RuntimeError(f"{label}: checkpoint state key drift")
        for key in left:
            assert_state_close(left[key],right[key],label=f"{label}/{key}")
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):
            raise RuntimeError(f"{label}: checkpoint state length drift")
        for index,(a,b) in enumerate(zip(left,right,strict=True)):
            assert_state_close(a,b,label=f"{label}/{index}")
    elif left!=right:
        raise RuntimeError(f"{label}: checkpoint state scalar drift")


def normalized_optimizer(optimizer: Any) -> Any:
    return getattr(optimizer,"optimizer",optimizer)


def pair_batch(
    common: Mapping[str,Any],semantic: Mapping[str,Any],full: Mapping[str,Any],
    device: torch.device,
) -> dict[str,Any]:
    return recursive_to_device({
        **common,"semantic_operator_compiled":semantic,
        "full_fabric_compiled":full,
    },device)


def read_plan_optimizer(args: argparse.Namespace) -> dict[str,Any]:
    optimization=read_json(args.training_plan)["optimization_strategy"]
    groups=optimization["optimizer_parameter_groups"]
    backbone=groups["pretrained_semantic_backbone"]
    interface=groups["new_or_reopened_interfaces"]
    horizon=int(optimization["scheduler_policy"]["operating_horizon_steps"])
    warmup=int(round(horizon*float(optimization["warmup_fraction"])))
    return {
        "backbone_lr":float(backbone["learning_rate"]),
        "interface_lr":float(interface["learning_rate"]),
        "backbone_weight_decay":float(backbone["weight_decay"]),
        "interface_weight_decay":float(interface["weight_decay"]),
        "gradient_clip_norm":float(optimization["gradient_clip_norm"]),
        "warmup_steps":warmup,"horizon_steps":horizon,
        "minimum_lr_scale":float(optimization["scheduler_policy"]["minimum_lr_scale"]),
        "seed":int(optimization["training_seed"]),
    }


def append_rank_row(path: Path,row: Mapping[str,Any]) -> None:
    with path.open("a",encoding="utf-8") as file:
        file.write(json.dumps(row,sort_keys=True)+"\n")
        file.flush()
        os.fsync(file.fileno())


def atomic_json(path: Path,row: Mapping[str,Any]) -> None:
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def main() -> None:
    args=parser().parse_args()
    config,no_grad,pairs,revision=validate_inputs(args)
    output=Path(args.output_dir).resolve()
    if int(os.environ.get("RANK","-1"))==0 and output.exists():
        raise SystemExit(f"preserve existing measured route evidence root: {output}")
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE","0"))!=2:
        raise SystemExit("measured full route requires two CUDA ranks")

    from accelerate import Accelerator, DistributedDataParallelKwargs
    from safetensors.torch import load_file,save_file

    options=read_plan_optimizer(args)
    random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    torch.cuda.manual_seed_all(options["seed"])
    accelerator=Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes!=2 or accelerator.device.type!="cuda":
        raise SystemExit("measured route lost two-rank CUDA DDP")
    if accelerator.is_main_process:
        output.mkdir(parents=True)
        atomic_json(output/"result.json",{
            "status":"INCOMPLETE_N0_MEASURED_JOINT_ROUTE_NOT_AUTHORITY",
            "source_revision":revision,"gpu_training_authorized":False,
            "final_results_observed":False,"private_identity_data":False,
        })
    accelerator.wait_for_everyone()
    rank_file=output/f"rank-{accelerator.process_index}.jsonl"
    semantic,full,common,semantic_rows,full_rows=prepare_case_inputs(args,config)

    system,load_receipt=load_registered_full_envelope_system(
        topology_path=args.topology_config,semantic_config_path=args.semantic_config,
        semantic_checkpoint_path=args.semantic_checkpoint,device="cpu",runtime_profile=None,
    )
    stage_report=apply_stage_trainability(system,stage=J3)
    if stage_report["all_parameters_owned"] is not True:
        raise RuntimeError("measured route does not own the full J3 topology")
    enable_precommitted_gradient_checkpointing(system)
    objective=FullEnvelopeJointTrainingObjectiveV1()
    optimizer=build_optimizer(
        system,backbone_lr=options["backbone_lr"],
        interface_lr=options["interface_lr"],
        backbone_weight_decay=options["backbone_weight_decay"],
        interface_weight_decay=options["interface_weight_decay"],
    )
    scheduler=cosine_with_warmup(
        optimizer,warmup_steps=options["warmup_steps"],
        horizon_steps=options["horizon_steps"],
        minimum_lr_scale=options["minimum_lr_scale"],
    )
    route=FullEnvelopeJointDDPRouteV2(system,objective)
    route,optimizer,scheduler=accelerator.prepare(route,optimizer,scheduler)
    unwrapped=accelerator.unwrap_model(route)
    system=unwrapped.system
    objective=unwrapped.objective
    route.train()
    if not hasattr(route,"module") or route.module is not unwrapped:
        raise RuntimeError("two-rank measured route is not a complete-step DDP wrapper")
    device=accelerator.device
    initial_route_state=cpu_state(unwrapped.state_dict())
    initial_scheduler_state=cpu_state(scheduler.state_dict())
    initial_group_lrs=[group["lr"] for group in optimizer.param_groups]
    if normalized_optimizer(optimizer).state:
        raise RuntimeError("diagnostic AdamW state exists before first measured step")
    total=int(torch.cuda.get_device_properties(device).total_memory)
    old_rank=no_grad["ranks"][accelerator.process_index]
    if int(old_rank["total_memory_bytes"])!=total:
        raise RuntimeError("GPU capacity changed since source-bound no-gradient proof")
    if old_rank["device_name"]!=torch.cuda.get_device_name(device):
        raise RuntimeError("GPU model changed since source-bound no-gradient proof")
    rank_rows=[]
    checkpoint_pair="long_context_semantic__long_additional_view_source"
    checkpoint_root=output/"resume_diagnostic"
    strict_transfer=None
    same_stage_resume=False
    checkpoint_memory=[]
    margin=int(config["device_memory_safety_margin_bytes"])
    for pair_number,pair in enumerate(pairs):
        semantic_name,full_name=pair.split("__",1)
        if pair_number:
            unwrapped.load_state_dict(initial_route_state,strict=True)
            normalized_optimizer(optimizer).state.clear()
            scheduler.load_state_dict(initial_scheduler_state)
            for group,lr in zip(optimizer.param_groups,initial_group_lrs,strict=True):
                group["lr"]=lr
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        if normalized_optimizer(optimizer).state:
            raise RuntimeError(f"first AdamW step is not fresh: {pair}")
        print(f"MEASURED_J3_PAIR_START rank={accelerator.process_index} pair={pair}",flush=True)
        torch.cuda.synchronize(device)
        observers=[]
        losses=[]
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(device)
            observer=MemoryObserver(
                device=device,margin=int(config["device_memory_safety_margin_bytes"]),
            )
            observer.model=system
            before=digest_state(unwrapped,normalized_optimizer(optimizer),scheduler)
            step_losses=execute_registered_optimizer_step(
                accelerator=accelerator,route=route,unwrapped_route=unwrapped,
                system=system,objective=objective,optimizer=optimizer,
                lr_scheduler=scheduler,
                next_batch=lambda:pair_batch(common,semantic[semantic_name],full[full_name],device),
                stage=J3,device=device,accumulation_steps=8,
                gradient_clip_norm=options["gradient_clip_norm"],observe=observer,
            )
            if accelerator.optimizer_step_was_skipped:
                raise RuntimeError(f"fp16 scaler skipped an optimizer step: {pair}/{index}")
            if not normalized_optimizer(optimizer).state:
                raise RuntimeError(f"AdamW moments were not created: {pair}/{index}")
            after=digest_state(unwrapped,normalized_optimizer(optimizer),scheduler)
            if after==before:
                raise RuntimeError(f"optimizer step produced no state change: {pair}/{index}")
            measured=observer.summary()
            measured["optimizer_step_index"]=index+1
            measured["finite_joint_loss"]=all(math.isfinite(v) for v in step_losses)
            measured["microbatch_count"]=len(step_losses)
            measured["weight_and_optimizer_state_changed"]=True
            if not measured["capacity_85_percent_pass"]:
                append_rank_row(rank_file,{"pair":pair,"step":index+1,"status":"FAIL_85_PERCENT_CAPACITY","measurement":measured})
                raise RuntimeError(f"measured full step exceeds 85% per-rank capacity: {pair}/{index}")
            if measured["microbatch_count"]!=8 or not measured["finite_nonzero_gradients"]:
                raise RuntimeError("complete full step lacks accumulation or named gradients")
            observers.append(measured)
            losses.append(step_losses)

            if pair==checkpoint_pair and index==0:
                accelerator.wait_for_everyone()
                torch.cuda.reset_peak_memory_stats(device)
                accelerator.save_state(str(checkpoint_root))
                if accelerator.is_main_process:
                    save_file({key:value.detach().cpu().contiguous() for key,value in objective.state_dict().items()},str(output/"objective_diagnostic.safetensors"))
                accelerator.wait_for_everyone()
                snapshot=checkpoint_memory_snapshot(device,margin,"checkpoint_save")
                checkpoint_memory.append(snapshot)
                if not snapshot["capacity_85_percent_pass"]:
                    append_rank_row(rank_file,{"pair":pair,"status":"FAIL_85_PERCENT_CHECKPOINT_SAVE","measurement":snapshot})
                    raise RuntimeError("checkpoint save exceeded original 85% per-rank capacity")
                first_digest=digest_state(unwrapped,normalized_optimizer(optimizer),scheduler)

            if pair==checkpoint_pair and index==1:
                continuous={
                    "route":cpu_state(unwrapped.state_dict()),
                    "optimizer":cpu_state(optimizer.state_dict()),
                    "scheduler":cpu_state(scheduler.state_dict()),
                }
                accelerator.wait_for_everyone()
                torch.cuda.reset_peak_memory_stats(device)
                accelerator.load_state(str(checkpoint_root))
                objective.load_state_dict(load_file(str(output/"objective_diagnostic.safetensors"),device="cpu"),strict=True)
                snapshot=checkpoint_memory_snapshot(device,margin,"checkpoint_load")
                checkpoint_memory.append(snapshot)
                if not snapshot["capacity_85_percent_pass"]:
                    append_rank_row(rank_file,{"pair":pair,"status":"FAIL_85_PERCENT_CHECKPOINT_LOAD","measurement":snapshot})
                    raise RuntimeError("checkpoint load exceeded original 85% per-rank capacity")
                if digest_state(unwrapped,normalized_optimizer(optimizer),scheduler)!=first_digest:
                    raise RuntimeError("same-stage restore changed model/optimizer/scheduler")
                torch.cuda.reset_peak_memory_stats(device)
                resumed_observer=MemoryObserver(device=device,margin=margin)
                resumed_observer.model=system
                resumed=execute_registered_optimizer_step(
                    accelerator=accelerator,route=route,unwrapped_route=unwrapped,
                    system=system,objective=objective,optimizer=optimizer,
                    lr_scheduler=scheduler,
                    next_batch=lambda:pair_batch(common,semantic[semantic_name],full[full_name],device),
                    stage=J3,device=device,accumulation_steps=8,
                    gradient_clip_norm=options["gradient_clip_norm"],observe=resumed_observer,
                )
                resumed_memory=resumed_observer.summary()
                resumed_memory["stage"]="resumed_optimizer_step"
                checkpoint_memory.append(resumed_memory)
                if not resumed_memory["capacity_85_percent_pass"]:
                    append_rank_row(rank_file,{"pair":pair,"status":"FAIL_85_PERCENT_RESUMED_STEP","measurement":resumed_memory})
                    raise RuntimeError("resumed optimizer step exceeded original 85% per-rank capacity")
                if accelerator.optimizer_step_was_skipped or not resumed_memory["finite_nonzero_gradients"]:
                    raise RuntimeError("resumed step skipped AdamW or missing named gradients")
                for expected,observed in zip(step_losses,resumed,strict=True):
                    if not math.isclose(expected,observed,rel_tol=1e-4,abs_tol=1e-5):
                        raise RuntimeError("same-stage resumed joint loss drift")
                assert_state_close(continuous["route"],unwrapped.state_dict(),label="route")
                assert_state_close(continuous["optimizer"],optimizer.state_dict(),label="optimizer")
                assert_state_close(continuous["scheduler"],scheduler.state_dict(),label="scheduler")
                same_stage_resume=True
                del continuous

        row={
            "pair":pair,"semantic_row_id":semantic_rows[semantic_name].get("id"),
            "full_fabric_row_id":full_rows[full_name].get("id"),
            "steps":observers,"all_ten_j3_families_executed":True,
            "complete_backward":True,"same_ddp_boundary_as_trainer":True,
            "first_and_later_adamw_measured":True,
            "state_reset_to_identical_initialization":True,
        }
        if pair==checkpoint_pair:
            if [entry["stage"] for entry in checkpoint_memory]!=[
                "checkpoint_save","checkpoint_load","resumed_optimizer_step",
            ]:
                raise RuntimeError("checkpoint and resumed memory coverage incomplete")
            row["checkpoint_resume_memory"]=checkpoint_memory
        append_rank_row(rank_file,row)
        rank_rows.append(row)
        print(f"MEASURED_J3_PAIR_PASS rank={accelerator.process_index} pair={pair}",flush=True)

    if not same_stage_resume:
        raise RuntimeError("same-stage full route resume not exercised")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        strict_transfer=predecessor_state_transfer(
            args,{name:value.detach().cpu() for name,value in system.state_dict().items()},
            {name:value.detach().cpu() for name,value in objective.state_dict().items()},
        )
        atomic_json(output/"predecessor_transfer.json",strict_transfer)
    accelerator.wait_for_everyone()
    strict_transfer=read_json(output/"predecessor_transfer.json")
    if set(strict_transfer)!={J1,J2,J3}:
        raise RuntimeError("J1/J2/J3 strict predecessor transfer missing")

    local={
        "rank":accelerator.process_index,"local_rank":accelerator.local_process_index,
        "device_name":torch.cuda.get_device_name(device),"total_memory_bytes":total,
        "runtime_build":runtime_build(device),
        "pair_count":len(rank_rows),"pairs":rank_rows,
        "same_stage_resume_verified":same_stage_resume,
        "predecessor_state_transfer_verified":True,
    }
    gathered=[None,None]
    torch.distributed.all_gather_object(gathered,local)
    if accelerator.is_main_process:
        if sorted(row["rank"] for row in gathered)!=[0,1]:
            raise RuntimeError("measured full route rank receipts incomplete")
        for row in gathered:
            if row["pair_count"]!=18 or {p["pair"] for p in row["pairs"]}!=set(pairs):
                raise RuntimeError("measured full route omitted a pair on a rank")
            if row["same_stage_resume_verified"] is not True:
                raise RuntimeError("rank did not verify same-stage resume")
            resume_pair=next(p for p in row["pairs"] if p["pair"]==checkpoint_pair)
            if len(resume_pair.get("checkpoint_resume_memory",[]))!=3 or any(
                sample["capacity_85_percent_pass"] is not True
                for sample in resume_pair["checkpoint_resume_memory"]
            ):
                raise RuntimeError("checkpoint transport or resumed step exceeded 85%")
            for pair in row["pairs"]:
                if len(pair["steps"])!=2 or any(
                    step["capacity_85_percent_pass"] is not True
                    or step["finite_nonzero_gradients"] is not True
                    or step["nonzero_public_judgment_gradient"] is not True
                    for step in pair["steps"]
                ):
                    raise RuntimeError("incomplete actual optimizer and memory evidence")
        gpu={
            "schema":"alice.eipm.n0.full-envelope-measured-joint-memory.v2",
            "status":PASS_MEMORY,"source_revision":revision,
            "qualifier_sha256":sha256_file(Path(__file__).resolve()),
            "registered_topology_sha256":sha256_file(args.topology_config),
            "qualification_config_sha256":sha256_file(args.qualification_config),
            "semantic_config_sha256":sha256_file(args.semantic_config),
            "semantic_checkpoint_sha256":sha256_file(args.semantic_checkpoint),
            "tokenizer_json_sha256":sha256_file(Path(args.tokenizer_dir)/"tokenizer.json"),
            "source_config_sha256":sha256_file(args.source_config),
            "corpus_receipt_sha256":sha256_file(Path(args.corpus_dir)/"corpus_receipt.json"),
            "mixture_manifest_sha256":sha256_file(args.mixture_manifest),
            "mixture_audit_sha256":sha256_file(args.mixture_audit),
            "teacher_registry_sha256":sha256_file(args.teacher_registry),
            "teacher_audit_sha256":sha256_file(args.teacher_audit),
            "no_gradient_receipt_sha256":sha256_file(args.no_gradient_receipt),
            "old_projection_status":no_grad["old_projection_status"],
            "old_projection_remains_independent":True,
            "predecessor_no_gradient_forward_verified":True,
            "registered_topology_load_receipt":load_receipt,
            "world_size":2,"ddp_replica_wrapped":True,
            "ddp_boundary":"complete_joint_step_v2",
            "microbatch_size":1,"teacher_batch_size":2,
            "replay_sequence_length":512,"candidate_gradient_accumulation":8,
            "training_mixed_precision":"fp16",
            "stress_pair_count":18,"all_ten_j3_families_executed":True,
            "complete_backward_per_rank":True,
            "finite_nonzero_gradients_per_rank":True,
            "first_and_later_adamw_steps_measured":True,
            "state_reset_to_identical_initialization_per_pair":True,
            "eight_microbatch_accumulation_measured":True,
            "same_stage_resume_verified":True,
            "checkpoint_resume_capacity_pass_per_rank":True,
            "predecessor_state_transfer_verified":True,
            "stage_dev_selection_claimed":False,
            "capacity_85_percent_pass_per_rank":True,
            "ranks":gathered,
            "projection_is_training_authorization":False,
            "measurement_is_training_authorization":False,
            "diagnostic_weight_updates":True,"production_model_training_performed":False,
            "gpu_training_authorized":False,"private_identity_data":False,
            "final_results_observed":False,"final_opening_authorized":False,
            "n0_complete":False,
        }
        gpu_file=output/"measured_gpu_memory_result.json"
        atomic_json(gpu_file,gpu)
        route_receipt={
            "schema":"alice.eipm.n0.complete-joint-route-qualification.v2",
            "status":PASS_ROUTE,"source_revision":revision,
            "qualifier_sha256":sha256_file(Path(__file__).resolve()),
            "trainer_sha256":sha256_file(Path(__file__).resolve().with_name("train_n0_v02_full_envelope_joint_v1.py")),
            "route_sha256":sha256_file(Path(__file__).resolve().parents[3]/"src/alice_personality/n0/full_envelope_joint_ddp_route_v2.py"),
            "mixture_manifest_sha256":sha256_file(args.mixture_manifest),
            "gpu_memory_receipt_sha256":sha256_file(gpu_file),
            "topology_config_sha256":sha256_file(args.topology_config),
            "ddp_boundary":"complete_joint_step_v2",
            "world_size":2,"j3_stress_pair_count":18,
            "all_ten_j3_families_executed":True,"complete_backward_per_rank":True,
            "finite_nonzero_gradients_per_rank":True,
            "first_and_later_adamw_steps_measured":True,
            "eight_microbatch_accumulation_measured":True,
            "same_stage_resume_verified":True,
            "checkpoint_resume_capacity_pass_per_rank":True,
            "predecessor_state_transfer_verified":True,
            "stage_dev_selection_claimed":False,
            "capacity_85_percent_pass_per_rank":True,
            "diagnostic_weight_updates_are_production_training":False,
            "final_results_observed":False,"private_identity_data":False,
        }
        atomic_json(output/"complete_joint_route_result.json",route_receipt)
        print(json.dumps({
            "status":PASS_ROUTE,"source_revision":revision,
            "gpu_receipt":str(gpu_file),
            "joint_route_receipt":str(output/"complete_joint_route_result.json"),
            "old_projection_status":no_grad["old_projection_status"],
            "production_training_performed":False,
        },sort_keys=True),flush=True)
    accelerator.wait_for_everyone()


if __name__=="__main__":
    main()
