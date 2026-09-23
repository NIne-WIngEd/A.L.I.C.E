#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
    compile_behavioral_batch,
)
from alice_personality.n0.full_envelope_joint_step_v1 import (
    execute_full_envelope_joint_step,
)
from alice_personality.n0.full_envelope_runtime_factory_v1 import (
    load_registered_full_envelope_system,
)
from alice_personality.n0.full_envelope_stage_policy_v1 import (
    J1,
    J2,
    J3,
    apply_stage_trainability,
    resolve_stage_policy,
)
from alice_personality.n0.full_envelope_training_batch_scheduler_v1 import (
    FullEnvelopeTrainingBatchSchedulerV1,
)
from alice_personality.n0.full_envelope_training_objective_v1 import (
    FullEnvelopeJointTrainingObjectiveV1,
)
from alice_personality.n0.natural_relation_batch_v1 import (
    compile_natural_relation_batch,
)
from alice_personality.n0.semantic_operator_batch_v1 import (
    compile_semantic_operator_batch,
)
from alice_personality.n0.v02_training import (
    TeacherMultitaskCollator,
    sha256_file,
    verify_public_corpus_v021,
    verify_teacher_registry,
)


PASS_MIXTURE="PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1"
PASS_CPU="PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1"
PASS_GPU="PASS_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1"
PASS_LONG_BOUNDARY="PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_TOKEN_BOUNDARY_ALIGNMENT_V1"
PASS_SEMANTIC_LONG_TOKEN="PASS_N0_SEMANTIC_OPERATOR_LONG_TOKEN_ALIGNMENT_V1"

STAGES=(J1,J2,J3)
PREDECESSOR={J1:None,J2:J1,J3:J2}


def read_json(path: str | Path) -> dict[str,Any]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def sha256_tree(path: str | Path) -> str:
    root=Path(path)
    if not root.is_dir():
        raise SystemExit(f"checkpoint state directory missing: {root}")
    files=sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item:item.relative_to(root).as_posix(),
    )
    if not files:
        raise SystemExit(f"checkpoint state directory empty: {root}")
    digest=hashlib.sha256()
    for item in files:
        relative=item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8,"big"))
        digest.update(relative)
        size=item.stat().st_size
        digest.update(int(size).to_bytes(8,"big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path, *, split: str) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected=[]
    for row in rows:
        if str(row.get("split",""))!=split:
            continue
        if row.get("private_identity_data") is not False:
            raise SystemExit(f"private identity row forbidden: {row.get('id')}")
        if row.get("final_validation_only") is True:
            raise SystemExit(f"FINAL row entered optimizer lane: {row.get('id')}")
        if split=="train" and row.get("training_authorized") is not True:
            raise SystemExit(f"TRAIN authority drift: {row.get('id')}")
        selected.append(row)
    if not selected:
        raise SystemExit(f"no {split.upper()} rows in {path}")
    return selected


def current_git_revision() -> str:
    return subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True
    ).strip()


def require_clean_worktree() -> None:
    status=subprocess.check_output(
        ["git","status","--porcelain"],text=True
    )
    if status.strip():
        raise SystemExit("successor training requires a clean exact-source worktree")


def require_status(path: str | Path, expected: str, *, label: str) -> dict[str,Any]:
    value=read_json(path)
    if value.get("status")!=expected:
        raise SystemExit(
            f"{label} gate not passed: expected={expected!r} "
            f"observed={value.get('status')!r}"
        )
    return value


def verify_pre_gradient_runtime(
    *,
    mixture_manifest_path: str | Path,
    mixture_audit_path: str | Path,
    tokenizer_audit_path: str | Path,
    cpu_runtime_receipt_path: str | Path,
    gpu_memory_receipt_path: str | Path,
    long_boundary_receipt_path: str | Path,
    semantic_long_token_receipt_path: str | Path,
    static_proof_receipt_path: str | Path,
) -> tuple[dict[str,Any],dict[str,Any]]:
    mixture=read_json(mixture_manifest_path)
    mixture_audit=require_status(
        mixture_audit_path,PASS_MIXTURE,label="full public mixture"
    )
    if mixture.get("status")!="MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT":
        raise SystemExit("full public mixture manifest status drift")
    source_revision=str(mixture.get("source_revision",""))
    if source_revision!=current_git_revision():
        raise SystemExit(
            "mixture source revision does not match exact checked-out successor head"
        )
    if mixture_audit.get("source_revision")!=source_revision:
        raise SystemExit("mixture audit source revision drift")
    if mixture_audit.get("manifest_sha256")!=sha256_file(
        mixture_manifest_path
    ):
        raise SystemExit("full public mixture audit/manifest hash drift")
    if mixture.get("final_results_observed") is not False:
        raise SystemExit("FINAL results were observed before gradient")
    if int(mixture.get("final_rows_in_training",-1))!=0:
        raise SystemExit("FINAL rows present in public mixture")
    if mixture.get("private_identity_data") is not False:
        raise SystemExit("private identity data present in public mixture")

    tokenizer=require_status(
        tokenizer_audit_path,"PASS",label="exact tokenizer stress"
    )
    cpu=require_status(
        cpu_runtime_receipt_path,PASS_CPU,label="CPU full-envelope runtime"
    )
    gpu=require_status(
        gpu_memory_receipt_path,PASS_GPU,label="GPU no-gradient memory"
    )
    boundary=require_status(
        long_boundary_receipt_path,PASS_LONG_BOUNDARY,
        label="long-context tokenizer boundary"
    )
    semantic_long=require_status(
        semantic_long_token_receipt_path,PASS_SEMANTIC_LONG_TOKEN,
        label="semantic long-context token evidence"
    )
    static_proof=require_status(
        static_proof_receipt_path,
        "PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1",
        label="exact-head static proof matrix",
    )
    if static_proof.get("source_revision")!=source_revision:
        raise SystemExit("static proof receipt source revision drift")
    for label,receipt in (
        ("CPU",cpu),("GPU",gpu),("boundary",boundary),("semantic-long",semantic_long),
        ("static-proof",static_proof)
    ):
        observed=receipt.get("source_revision")
        if observed is not None and observed!=source_revision:
            raise SystemExit(f"{label} runtime receipt source revision drift")
        if receipt.get("final_results_observed") not in (None,False):
            raise SystemExit(f"{label} runtime receipt observed FINAL")
        if receipt.get("private_identity_data") not in (None,False):
            raise SystemExit(f"{label} runtime receipt contains private identity data")
    if gpu.get("projection_is_training_authorization") is not False:
        raise SystemExit("GPU memory projection may not authorize training by itself")
    if boundary.get("final_opening_authorized") not in (None,False):
        raise SystemExit("boundary audit unexpectedly authorizes FINAL")
    if tokenizer.get("model_training_performed") not in (None,False):
        raise SystemExit("tokenizer audit unexpectedly records training")
    return mixture,mixture_audit


def recursive_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value,torch.Tensor):
        return value.to(device,non_blocking=True)
    if isinstance(value,dict):
        return {k:recursive_to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):
        return [recursive_to_device(v,device) for v in value]
    if isinstance(value,tuple):
        return tuple(recursive_to_device(v,device) for v in value)
    return value


def cycle_next(iterator: Any, loader: Any) -> tuple[Any,Any]:
    try:
        return next(iterator),iterator
    except StopIteration:
        iterator=iter(loader)
        return next(iterator),iterator


def build_optimizer(
    system: torch.nn.Module,
    *,
    backbone_lr: float,
    interface_lr: float,
    backbone_weight_decay: float,
    interface_weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_ids={id(p) for p in system.backbone.parameters()}
    groups={
        ("backbone","decay"):[],
        ("backbone","no_decay"):[],
        ("interface","decay"):[],
        ("interface","no_decay"):[],
    }
    for name,p in system.named_parameters():
        owner="backbone" if id(p) in backbone_ids else "interface"
        no_decay=("bias" in name.lower() or "norm" in name.lower())
        groups[(owner,"no_decay" if no_decay else "decay")].append(p)

    params=[]
    for (owner,decay_kind),values in groups.items():
        if not values:
            continue
        params.append({
            "params":values,
            "lr":backbone_lr if owner=="backbone" else interface_lr,
            "weight_decay":(
                0.0
                if decay_kind=="no_decay"
                else backbone_weight_decay if owner=="backbone"
                else interface_weight_decay
            ),
            "group_role":f"{owner}_{decay_kind}",
        })
    return torch.optim.AdamW(
        params,betas=(0.9,0.95),eps=1e-8
    )


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    horizon_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps<0 or horizon_steps<=max(warmup_steps,0):
        raise ValueError("invalid scheduler operating horizon")
    def scale(step: int) -> float:
        if warmup_steps and step<warmup_steps:
            return float(step+1)/float(warmup_steps)
        progress=(
            float(step-warmup_steps)
            / float(max(1,horizon_steps-warmup_steps))
        )
        progress=min(max(progress,0.0),1.0)
        return 0.5*(1.0+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,scale)


def family_sample_weights(
    result: Mapping[str,Any],
    *,
    mlm_batch: Mapping[str,Any],
    teacher_batch: Mapping[str,Any],
    semantic_compiled: Mapping[str,Any],
    full_compiled: Mapping[str,Any] | None,
    natural_compiled: Mapping[str,Any],
) -> dict[str,float]:
    active=tuple(map(str,result["active_families"]))
    semantic_n=float(semantic_compiled["metadata"]["batch_size"])
    natural_n=float(natural_compiled["metadata"]["batch_size"])
    full_n=(
        float(full_compiled["metadata"]["batch_size"])
        if full_compiled is not None else 0.0
    )
    mlm_n=float(mlm_batch["input_ids"].shape[0])
    teacher_n=float(len(teacher_batch["ids"]))
    out={}
    for family in active:
        if family=="broad_semantic_replay":
            out[family]=mlm_n
        elif family=="governed_judgment_replay":
            out[family]=teacher_n
        elif family=="natural_relation_semantics":
            out[family]=natural_n
        elif family in {
            "relation_program_semantics",
            "dynamic_factor_semantics",
            "uncertainty_and_control",
            "token_evidence_grounding",
        }:
            out[family]=semantic_n
        else:
            if full_n<=0:
                raise RuntimeError(
                    f"active full-fabric family lacks full batch: {family}"
                )
            out[family]=full_n
    return out


def globally_observe_effective_batch(
    *,
    accelerator: Any,
    objective: FullEnvelopeJointTrainingObjectiveV1,
    raw_numerators: Mapping[str,torch.Tensor],
    raw_denominators: Mapping[str,torch.Tensor],
    active_families: tuple[str,...],
) -> None:
    means={}
    for family in active_families:
        numerator=accelerator.reduce(
            raw_numerators[family],reduction="sum"
        )
        denominator=accelerator.reduce(
            raw_denominators[family],reduction="sum"
        )
        if float(denominator.item())<=0.0:
            raise RuntimeError(f"effective batch has zero samples: {family}")
        means[family]=numerator/denominator
    objective.balancer.observe_detached_family_means(
        means,active_families=active_families
    )


def save_checkpoint(
    *,
    accelerator: Any,
    system: torch.nn.Module,
    objective: FullEnvelopeJointTrainingObjectiveV1,
    output_root: Path,
    stage: str,
    step: int,
    source_revision: str,
    topology_path: Path,
    semantic_initialization_path: Path,
    mixture_manifest_path: Path,
    mixture_audit_path: Path,
    tokenizer_dir: Path,
    corpus_receipt_path: Path,
    teacher_audit_path: Path,
    static_proof_receipt_path: Path,
    stage_report: Mapping[str,Any],
    scheduler_horizon_steps: int,
    warmup_steps: int,
    gradient_accumulation_steps: int,
) -> None:
    from safetensors.torch import save_file

    checkpoint=output_root/f"{stage}-step-{step:08d}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint/"accelerator_state"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint.mkdir(parents=True,exist_ok=True)
        unwrapped=accelerator.unwrap_model(system)
        system_path=checkpoint/"full_system.safetensors"
        save_file(
            {
                name:tensor.detach().cpu().contiguous()
                for name,tensor in unwrapped.state_dict().items()
            },
            str(system_path),
        )
        objective_path=checkpoint/"full_envelope_objective.safetensors"
        save_file(
            {
                name:tensor.detach().cpu().contiguous()
                for name,tensor in objective.state_dict().items()
            },
            str(objective_path),
        )
        receipt={
            "schema":"alice.eipm.n0.full-envelope-checkpoint-receipt.v1",
            "status":"TRAINED_PUBLIC_N0_CANDIDATE_REQUIRES_DEV_SELECTION",
            "source_revision":source_revision,
            "registered_topology_sha256":sha256_file(topology_path),
            "semantic_initialization_sha256":sha256_file(
                semantic_initialization_path
            ),
            "full_public_mixture_manifest_sha256":sha256_file(
                mixture_manifest_path
            ),
            "full_public_mixture_audit_sha256":sha256_file(
                mixture_audit_path
            ),
            "stage":stage,
            "stage_policy":{
                "active_macro_families":list(stage_report["active_macro_families"]),
                "trainable_components":list(stage_report["trainable_components"]),
                "architecture_reduced":False,
                "all_parameters_owned":True,
            },
            "optimizer_contract":{
                "name":"AdamW",
                "full_topology_parameter_groups":True,
                "frozen_parameters_receive_updates":False,
                "macro_family_ema_once_per_effective_batch":True,
            },
            "scheduler_contract":{
                "family":"linear_warmup_cosine_decay",
                "warmup_steps":int(warmup_steps),
                "operating_horizon_steps":int(scheduler_horizon_steps),
                "horizon_is_stage_completion":False,
            },
            "objective_contract":{
                "class":"FullEnvelopeJointTrainingObjectiveV1",
                "equal_precommitted_macro_family_weights":True,
                "learned_task_weights":False,
            },
            "tokenizer_sha256":sha256_file(tokenizer_dir/"tokenizer.json"),
            "public_corpus_receipt_sha256":sha256_file(corpus_receipt_path),
            "teacher_audit_sha256":sha256_file(teacher_audit_path),
            "static_proof_receipt_sha256":sha256_file(
                static_proof_receipt_path
            ),
            "optimizer_step":int(step),
            "gradient_accumulation_steps":int(gradient_accumulation_steps),
            "accelerator_state_tree_sha256":sha256_tree(
                checkpoint/"accelerator_state"
            ),
            "full_system_sha256":sha256_file(system_path),
            "objective_state_sha256":sha256_file(objective_path),
            "checkpoint_selection_performed":False,
            "final_results_observed":False,
            "final_opening_authorized":False,
            "final_checkpoint_selection_allowed":False,
            "private_identity_data":False,
            "private_identity_gradient":False,
            "n0_complete":False,
        }
        (checkpoint/"receipt.json").write_text(
            json.dumps(receipt,indent=2,sort_keys=True)+"\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


def main() -> None:
    parser=argparse.ArgumentParser(
        description=(
            "Successor staged J1/J2/J3 trainer for the exact registered "
            "N0 full-envelope topology. FINAL validation is forbidden here."
        )
    )
    parser.add_argument("--stage",choices=STAGES,required=True)
    parser.add_argument("--training-plan",required=True)
    parser.add_argument("--topology-config",required=True)
    parser.add_argument("--semantic-config",required=True)
    parser.add_argument("--semantic-checkpoint",required=True)
    parser.add_argument("--tokenizer-dir",required=True)
    parser.add_argument("--source-config",required=True)
    parser.add_argument("--corpus-dir",required=True)
    parser.add_argument("--teacher-registry",required=True)
    parser.add_argument("--teacher-audit",required=True)
    parser.add_argument("--mixture-manifest",required=True)
    parser.add_argument("--mixture-audit",required=True)
    parser.add_argument("--tokenizer-audit",required=True)
    parser.add_argument("--cpu-runtime-receipt",required=True)
    parser.add_argument("--gpu-memory-receipt",required=True)
    parser.add_argument("--long-boundary-receipt",required=True)
    parser.add_argument("--semantic-long-token-receipt",required=True)
    parser.add_argument("--static-proof-receipt",required=True)
    parser.add_argument("--semantic-rows",required=True)
    parser.add_argument("--semantic-long-rows",required=True)
    parser.add_argument("--behavioral-rows",required=True)
    parser.add_argument("--runtime-view-rows",required=True)
    parser.add_argument("--long-context-rows",required=True)
    parser.add_argument("--natural-rows",required=True)
    parser.add_argument("--natural-bank",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--resume-accelerator-state")
    parser.add_argument("--resume-objective-state")
    parser.add_argument("--resume-receipt")
    parser.add_argument("--predecessor-dev-receipt")
    parser.add_argument("--execute-gradient",action="store_true")
    parser.add_argument("--max-optimizer-steps",type=int,default=100)
    parser.add_argument("--save-every",type=int,default=25)
    parser.add_argument("--gradient-accumulation-steps",type=int,default=8)
    parser.add_argument("--lane-batch-size",type=int,default=1)
    parser.add_argument("--mlm-micro-batch-size",type=int,default=1)
    parser.add_argument("--teacher-batch-size",type=int,default=2)
    parser.add_argument("--replay-sequence-length",type=int,default=512)
    parser.add_argument("--warmup-steps",type=int,default=20)
    parser.add_argument("--scheduler-horizon-steps",type=int,default=1000)
    parser.add_argument("--backbone-learning-rate",type=float,default=1e-5)
    parser.add_argument("--interface-learning-rate",type=float,default=5e-5)
    parser.add_argument("--backbone-weight-decay",type=float,default=0.1)
    parser.add_argument("--interface-weight-decay",type=float,default=0.05)
    parser.add_argument("--gradient-clip-norm",type=float,default=1.0)
    parser.add_argument("--seed",type=int,default=20260922)
    parser.add_argument(
        "--mixed-precision",choices=("no","fp16","bf16"),default="fp16"
    )
    args=parser.parse_args()

    if min(
        args.max_optimizer_steps,args.save_every,
        args.gradient_accumulation_steps,args.lane_batch_size,
        args.mlm_micro_batch_size,args.teacher_batch_size,
        args.replay_sequence_length,args.scheduler_horizon_steps,
    )<=0:
        raise SystemExit("training operating points must be positive")
    if args.warmup_steps<0:
        raise SystemExit("warmup steps cannot be negative")
    if args.scheduler_horizon_steps<=args.warmup_steps:
        raise SystemExit("scheduler horizon must exceed warmup")

    require_clean_worktree()
    training_plan=read_json(args.training_plan)
    if training_plan.get("schema")!="alice.eipm.n0.semantic-operator-joint-training-plan.v1":
        raise SystemExit("successor training-plan schema drift")
    authority=dict(training_plan.get("authorization") or {})
    if args.execute_gradient:
        required_authority=("optimizer","gradient","gpu_training")
        blocked=[
            name for name in required_authority
            if authority.get(name) is not True
        ]
        if blocked:
            raise SystemExit(
                "successor optimization remains governance-blocked; "
                "explicit training-plan authority is false for "
                + repr(blocked)
            )
    mixture,mixture_audit=verify_pre_gradient_runtime(
        mixture_manifest_path=args.mixture_manifest,
        mixture_audit_path=args.mixture_audit,
        tokenizer_audit_path=args.tokenizer_audit,
        cpu_runtime_receipt_path=args.cpu_runtime_receipt,
        gpu_memory_receipt_path=args.gpu_memory_receipt,
        long_boundary_receipt_path=args.long_boundary_receipt,
        semantic_long_token_receipt_path=args.semantic_long_token_receipt,
        static_proof_receipt_path=args.static_proof_receipt,
    )
    if not args.execute_gradient:
        print(json.dumps({
            "status":"PREGRADIENT_GATES_PASS_GRADIENT_NOT_EXECUTED",
            "stage":args.stage,
            "source_revision":mixture["source_revision"],
            "training_plan_authorization":{
                "optimizer":authority.get("optimizer"),
                "gradient":authority.get("gradient"),
                "gpu_training":authority.get("gpu_training"),
            },
            "optimizer_created":False,
            "gradient":False,
            "weight_update":False,
            "final_results_observed":False,
            "final_opening_authorized":False,
            "n0_complete":False,
        },indent=2,sort_keys=True))
        return

    predecessor=PREDECESSOR[args.stage]
    if predecessor is None:
        if any((
            args.resume_accelerator_state,
            args.resume_objective_state,
            args.resume_receipt,
            args.predecessor_dev_receipt,
        )):
            raise SystemExit("J1 starts from registered semantic initialization only")
    else:
        required_paths=(
            args.resume_accelerator_state,
            args.resume_objective_state,
            args.resume_receipt,
            args.predecessor_dev_receipt,
        )
        if not all(required_paths):
            raise SystemExit(
                f"{args.stage} requires predecessor checkpoint and DEV receipt"
            )
        prior=read_json(args.resume_receipt)
        dev=read_json(args.predecessor_dev_receipt)
        if prior.get("stage")!=predecessor:
            raise SystemExit("resume checkpoint stage is not the required predecessor")
        if prior.get("final_results_observed") is not False:
            raise SystemExit("resume checkpoint has observed FINAL")
        if dev.get("stage")!=predecessor:
            raise SystemExit("predecessor DEV receipt stage drift")
        if dev.get("checkpoint_selection_surface")!="DEV_ONLY":
            raise SystemExit("predecessor selection did not use DEV-only surface")
        if dev.get("stage_gate_pass") is not True:
            raise SystemExit("predecessor DEV stage gates did not pass")
        if dev.get("final_results_observed") is not False:
            raise SystemExit("predecessor DEV receipt observed FINAL")
        if sha256_file(args.resume_receipt)!=dev.get(
            "candidate_checkpoint_receipt_sha256"
        ):
            raise SystemExit(
                "predecessor DEV receipt/checkpoint receipt hash drift"
            )
        if prior.get("full_system_sha256")!=dev.get("candidate_system_sha256"):
            raise SystemExit("predecessor DEV receipt/system hash drift")
        if sha256_tree(args.resume_accelerator_state)!=prior.get(
            "accelerator_state_tree_sha256"
        ):
            raise SystemExit("predecessor accelerator-state hash drift")
        if sha256_file(args.resume_objective_state)!=prior.get(
            "objective_state_sha256"
        ):
            raise SystemExit("predecessor objective-state hash drift")
        if (
            prior.get("source_revision")!=mixture.get("source_revision")
            or dev.get("source_revision")!=mixture.get("source_revision")
        ):
            raise SystemExit("predecessor source revision drift")

    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("install requirements-n0.txt before successor training") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator=Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
    )
    device=accelerator.device

    system,load_receipt=load_registered_full_envelope_system(
        topology_path=args.topology_config,
        semantic_config_path=args.semantic_config,
        semantic_checkpoint_path=args.semantic_checkpoint,
        device="cpu",
        runtime_profile=None,
    )
    stage_report=apply_stage_trainability(system,stage=args.stage)
    policy=resolve_stage_policy(args.stage)
    objective=FullEnvelopeJointTrainingObjectiveV1()

    tokenizer_dir=Path(args.tokenizer_dir).resolve()
    tokenizer=load_tokenizer(tokenizer_dir)
    corpus_dir=Path(args.corpus_dir).resolve()
    corpus_receipt,corpus_paths=verify_public_corpus_v021(
        corpus_dir,args.source_config
    )
    curriculum_paths,teacher_report=verify_teacher_registry(
        Path(__file__).resolve().parents[3],
        args.teacher_registry,
        args.teacher_audit,
    )

    semantic_rows=read_jsonl(args.semantic_rows,split="train")
    semantic_long_rows=read_jsonl(args.semantic_long_rows,split="train")
    behavioral_rows=read_jsonl(args.behavioral_rows,split="train")
    runtime_view_rows=read_jsonl(args.runtime_view_rows,split="train")
    long_context_rows=read_jsonl(args.long_context_rows,split="train")
    natural_rows=read_jsonl(args.natural_rows,split="train")
    natural_bank=read_json(args.natural_bank)

    scheduler_rows={
        "semantic_operator_intervention":semantic_rows,
        "long_context_semantic":semantic_long_rows,
        "full_envelope_behavioral":behavioral_rows,
        "runtime_view_supplement":runtime_view_rows,
        "long_context_fabric":long_context_rows,
        "natural_relation":natural_rows,
    }
    lane_scheduler=FullEnvelopeTrainingBatchSchedulerV1(
        lanes=scheduler_rows,
        seed=args.seed+int(accelerator.process_index),
    )

    mlm_dataset=PackedJSONLIterableDataset(
        paths=corpus_paths,
        tokenizer=tokenizer,
        sequence_length=args.replay_sequence_length,
        split="train",
        shuffle_seed=args.seed,
    )
    mlm_loader=DataLoader(
        mlm_dataset,
        batch_size=args.mlm_micro_batch_size,
        collate_fn=SpanMLMCollator(
            tokenizer=tokenizer,
            mlm_probability=0.30,
            mean_span=3.0,
            max_span=10,
            seed=args.seed+int(accelerator.process_index),
        ),
        num_workers=0,
        pin_memory=True,
    )
    teacher_dataset=CurriculumDataset(curriculum_paths,"train")
    teacher_loader=DataLoader(
        teacher_dataset,
        batch_size=args.teacher_batch_size,
        shuffle=True,
        collate_fn=TeacherMultitaskCollator(tokenizer,256),
        num_workers=0,
        pin_memory=True,
    )

    optimizer=build_optimizer(
        system,
        backbone_lr=args.backbone_learning_rate,
        interface_lr=args.interface_learning_rate,
        backbone_weight_decay=args.backbone_weight_decay,
        interface_weight_decay=args.interface_weight_decay,
    )
    lr_scheduler=cosine_with_warmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        horizon_steps=args.scheduler_horizon_steps,
    )

    system,optimizer,mlm_loader,teacher_loader,lr_scheduler=accelerator.prepare(
        system,optimizer,mlm_loader,teacher_loader,lr_scheduler
    )
    objective=objective.to(device)

    if predecessor is not None:
        accelerator.load_state(args.resume_accelerator_state)
        objective_state=load_file(args.resume_objective_state,device="cpu")
        objective.load_state_dict(objective_state,strict=True)
        # Re-apply successor stage authority after restoring the predecessor.
        stage_report=apply_stage_trainability(
            accelerator.unwrap_model(system),stage=args.stage
        )

    if accelerator.is_main_process:
        print(json.dumps({
            "status":"N0_SUCCESSOR_TRAIN_START",
            "stage":args.stage,
            "source_revision":mixture["source_revision"],
            "registered_system":"N0FullEnvelopeTrainableSystemV1",
            "registered_topology_load_receipt":load_receipt,
            "active_macro_families":list(policy.active_macro_families),
            "architecture_reduced":False,
            "private_identity_data":False,
            "final_results_observed":False,
            "automatic_stage_transition":False,
            "max_optimizer_steps_is_stage_completion":False,
        },sort_keys=True))

    mlm_iterator=iter(mlm_loader)
    teacher_iterator=iter(teacher_loader)
    output_root=Path(args.output_dir).resolve()
    output_root.mkdir(parents=True,exist_ok=True)

    system.train()
    objective.train()
    for step in range(1,args.max_optimizer_steps+1):
        optimizer.zero_grad(set_to_none=True)
        numerators={
            family:torch.zeros((),device=device,dtype=torch.float32)
            for family in policy.active_macro_families
        }
        denominators={
            family:torch.zeros((),device=device,dtype=torch.float32)
            for family in policy.active_macro_families
        }

        for micro in range(args.gradient_accumulation_steps):
            mlm_batch,mlm_iterator=cycle_next(mlm_iterator,mlm_loader)
            teacher_batch,teacher_iterator=cycle_next(
                teacher_iterator,teacher_loader
            )
            semantic_lane,semantic_selected=lane_scheduler.next_rows(
                stage=args.stage,
                kind="semantic",
                batch_size=args.lane_batch_size,
            )
            semantic_compiled=compile_semantic_operator_batch(
                rows=semantic_selected,
                tokenizer=tokenizer,
            )
            natural_lane,natural_selected=lane_scheduler.next_rows(
                stage=args.stage,
                kind="natural",
                batch_size=args.lane_batch_size,
            )
            natural_compiled=compile_natural_relation_batch(
                rows=natural_selected,
                relation_bank=natural_bank,
                tokenizer=tokenizer,
            )
            full_compiled=None
            if lane_scheduler.active_full_fabric_lanes(stage=args.stage):
                full_lane,full_selected=lane_scheduler.next_rows(
                    stage=args.stage,
                    kind="full_fabric",
                    batch_size=args.lane_batch_size,
                )
                full_compiled=compile_behavioral_batch(
                    rows=full_selected,
                    tokenizer=tokenizer,
                )

            mlm_batch=recursive_to_device(mlm_batch,device)
            teacher_batch=recursive_to_device(teacher_batch,device)
            semantic_compiled=recursive_to_device(semantic_compiled,device)
            natural_compiled=recursive_to_device(natural_compiled,device)
            if full_compiled is not None:
                full_compiled=recursive_to_device(full_compiled,device)

            sync_context=(
                accelerator.no_sync(system)
                if micro<args.gradient_accumulation_steps-1
                else contextlib.nullcontext()
            )
            with sync_context:
                result=execute_full_envelope_joint_step(
                    system=system,
                    objective=objective,
                    mlm_batch=mlm_batch,
                    teacher_batch=teacher_batch,
                    semantic_operator_compiled=semantic_compiled,
                    full_fabric_compiled=full_compiled,
                    natural_relation_compiled=natural_compiled,
                    update_ema=False,
                    stage=args.stage,
                )
                loss=result["loss"]/float(args.gradient_accumulation_steps)
                accelerator.backward(loss)

            weights=family_sample_weights(
                result,
                mlm_batch=mlm_batch,
                teacher_batch=teacher_batch,
                semantic_compiled=semantic_compiled,
                full_compiled=full_compiled,
                natural_compiled=natural_compiled,
            )
            for family in policy.active_macro_families:
                raw=result["balanced"][f"raw/{family}"].detach().float()
                weight=float(weights[family])
                numerators[family].add_(raw*weight)
                denominators[family].add_(weight)

        accelerator.clip_grad_norm_(
            system.parameters(),args.gradient_clip_norm
        )
        optimizer.step()
        lr_scheduler.step()

        globally_observe_effective_batch(
            accelerator=accelerator,
            objective=objective,
            raw_numerators=numerators,
            raw_denominators=denominators,
            active_families=policy.active_macro_families,
        )

        if accelerator.is_main_process and (step==1 or step%10==0):
            print(json.dumps({
                "step":step,
                "stage":args.stage,
                "lr":[float(x) for x in lr_scheduler.get_last_lr()],
                "active_macro_families":list(policy.active_macro_families),
                "private_identity_data":False,
                "final_results_observed":False,
            },sort_keys=True))

        if step%args.save_every==0 or step==args.max_optimizer_steps:
            save_checkpoint(
                accelerator=accelerator,
                system=system,
                objective=objective,
                output_root=output_root,
                stage=args.stage,
                step=step,
                source_revision=str(mixture["source_revision"]),
                topology_path=Path(args.topology_config),
                semantic_initialization_path=Path(args.semantic_checkpoint),
                mixture_manifest_path=Path(args.mixture_manifest),
                mixture_audit_path=Path(args.mixture_audit),
                tokenizer_dir=tokenizer_dir,
                corpus_receipt_path=corpus_dir/"corpus_receipt.json",
                teacher_audit_path=Path(args.teacher_audit),
                static_proof_receipt_path=Path(args.static_proof_receipt),
                stage_report=stage_report,
                scheduler_horizon_steps=args.scheduler_horizon_steps,
                warmup_steps=args.warmup_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

    accelerator.wait_for_everyone()


if __name__=="__main__":
    main()
