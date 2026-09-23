#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from alice_personality.n0.curriculum_data import (
    CurriculumDataset,
    load_tokenizer,
)
from alice_personality.n0.data import (
    PackedJSONLIterableDataset,
    SpanMLMCollator,
)
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
    J3,
    apply_stage_trainability,
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
    verify_tokenizer_v021,
)


PASS="PASS_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1"


def read_json(path: str | Path) -> dict[str,Any]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str,Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train_rows(path: str | Path) -> list[dict[str,Any]]:
    rows=[row for row in read_jsonl(path) if row.get("split")=="train"]
    if not rows:
        raise ValueError(f"no TRAIN rows in {path}")
    for row in rows:
        if row.get("training_authorized") is not True:
            raise ValueError(f"TRAIN authority drift in {path}: {row.get('id')}")
        if row.get("final_validation_only") is True:
            raise ValueError(f"FINAL row entered GPU dry run: {row.get('id')}")
        if row.get("private_identity_data") is not False:
            raise ValueError(f"private identity row entered GPU dry run: {row.get('id')}")
    return rows


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value,torch.Tensor):
        return value.to(device,non_blocking=False)
    if isinstance(value,dict):
        return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):
        return [to_device(v,device) for v in value]
    if isinstance(value,tuple):
        return tuple(to_device(v,device) for v in value)
    return value


def choose_semantic(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    return dict(max(
        rows,
        key=lambda row:(
            int(row.get("runtime_relation_count",0)),
            int(row.get("runtime_operator_slots",0)),
            int(row.get("runtime_reasoning_steps",0)),
        ),
    ))


def choose_long_semantic(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    candidates=[
        dict(row) for row in rows
        if row.get("lane")=="semantic_operator_long_context"
    ]
    if not candidates:
        raise ValueError("dedicated long semantic GPU dry-run row pool empty")
    def score(row: Mapping[str,Any]) -> tuple[int,...]:
        texts=[str(row.get("query",""))]
        texts.extend(str(x.get("text","")) for x in row.get("relation_candidates") or [])
        for bank in (row.get("factor_schemas") or {}).values():
            texts.extend(str(x.get("text","")) for x in bank)
        return (
            max((len(text.split()) for text in texts),default=0),
            int(row.get("runtime_relation_count",0)),
            int(row.get("runtime_operator_slots",0)),
        )
    return dict(max(candidates,key=score))


def choose_full_fabric(
    behavioral: Sequence[Mapping[str,Any]],
    runtime_views: Sequence[Mapping[str,Any]],
    long_context: Sequence[Mapping[str,Any]],
) -> dict[str,Any]:
    all_rows=[dict(x) for x in behavioral]+[dict(x) for x in runtime_views]+[dict(x) for x in long_context]
    if not all_rows:
        raise ValueError("full-fabric GPU dry-run row pool empty")
    def score(row: Mapping[str,Any]) -> tuple[int,...]:
        return (
            1 if row.get("long_context_surface")=="additional_view_source" else 0,
            1 if row.get("long_context_surface")=="additional_view_descriptor" else 0,
            int(row.get("runtime_additional_view_count",0)),
            int(row.get("runtime_candidate_answer_count",0)),
            int(row.get("runtime_field_count",0)),
            int(row.get("runtime_edge_count",0)),
            int(row.get("runtime_reasoning_steps",0)),
        )
    return dict(max(all_rows,key=score))


def choose_natural(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    return dict(max(
        rows,
        key=lambda row:int(row.get("runtime_relation_count",0)),
    ))


def init_distributed() -> tuple[int,int,int,torch.device]:
    if not torch.cuda.is_available():
        raise SystemExit("GPU memory qualification requires CUDA")
    world_size=int(os.environ.get("WORLD_SIZE","1"))
    rank=int(os.environ.get("RANK","0"))
    local_rank=int(os.environ.get("LOCAL_RANK","0"))
    if world_size>1:
        dist.init_process_group(backend="nccl")
    if local_rank<0 or local_rank>=torch.cuda.device_count():
        raise SystemExit(
            f"invalid LOCAL_RANK={local_rank} cuda_devices={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    return rank,world_size,local_rank,torch.device("cuda",local_rank)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--qualification-config",required=True)
    p.add_argument("--topology-config",required=True)
    p.add_argument("--semantic-config",required=True)
    p.add_argument("--semantic-checkpoint",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--corpus-dir",required=True)
    p.add_argument("--source-config",required=True)
    p.add_argument("--teacher-registry",required=True)
    p.add_argument("--teacher-audit",required=True)
    p.add_argument("--semantic-rows",required=True)
    p.add_argument("--semantic-long-rows",required=True)
    p.add_argument("--behavioral-rows",required=True)
    p.add_argument("--runtime-view-rows",required=True)
    p.add_argument("--long-context-rows",required=True)
    p.add_argument("--fewrel-rows",required=True)
    p.add_argument("--fewrel-bank",required=True)
    p.add_argument("--mixture-manifest",required=True)
    p.add_argument("--mixture-audit",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    cfg=read_json(args.qualification_config)
    if cfg.get("schema")!="alice.eipm.n0.full-envelope-gpu-memory-dry-run.v1":
        raise SystemExit("GPU memory qualification config schema drift")
    q=dict(cfg["qualification"])
    if q.get("no_gradient") is not True or q.get("no_optimizer_object") is not True:
        raise SystemExit("GPU dry run must forbid gradient and optimizer")
    mixture=read_json(args.mixture_manifest)
    mixture_audit=read_json(args.mixture_audit)
    if mixture.get("status")!="MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT":
        raise SystemExit("full public mixture manifest not materialized")
    if mixture_audit.get("status")!="PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1":
        raise SystemExit("full public mixture audit not PASS")
    if mixture_audit.get("manifest_sha256")!=sha256_file(
        args.mixture_manifest
    ):
        raise SystemExit("full public mixture audit/manifest hash drift")
    if mixture.get("final_results_observed") is not False:
        raise SystemExit("FINAL results observed before GPU dry run")
    if int(mixture.get("final_rows_in_training",-1))!=0:
        raise SystemExit("FINAL rows entered public mixture")

    rank,world_size,local_rank,device=init_distributed()
    preferred=int(cfg["route"]["preferred_world_size"])
    if world_size<1:
        raise SystemExit("invalid distributed world size")

    verify_tokenizer_v021(args.tokenizer_dir)
    tokenizer=load_tokenizer(args.tokenizer_dir)
    corpus_receipt,corpus_paths=verify_public_corpus_v021(
        args.corpus_dir,
        args.source_config,
    )
    curriculum_paths,teacher_report=verify_teacher_registry(
        Path(__file__).resolve().parents[3],
        args.teacher_registry,
        args.teacher_audit,
    )

    semantic_rows=train_rows(args.semantic_rows)
    semantic_long_rows=train_rows(args.semantic_long_rows)
    behavioral_rows=train_rows(args.behavioral_rows)
    runtime_rows=train_rows(args.runtime_view_rows)
    long_rows=train_rows(args.long_context_rows)
    natural_rows=train_rows(args.fewrel_rows)
    relation_bank=read_json(args.fewrel_bank)

    mlm_dataset=PackedJSONLIterableDataset(
        paths=corpus_paths,
        tokenizer=tokenizer,
        sequence_length=512,
        split="train",
        shuffle_seed=20260922+rank,
    )
    mlm_loader=DataLoader(
        mlm_dataset,
        batch_size=1,
        collate_fn=SpanMLMCollator(
            tokenizer=tokenizer,
            mlm_probability=0.30,
            mean_span=3.0,
            max_span=10,
            seed=20260922+rank,
        ),
        num_workers=0,
    )
    teacher_dataset=CurriculumDataset(curriculum_paths,"train")
    teacher_loader=DataLoader(
        teacher_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=TeacherMultitaskCollator(tokenizer,256),
        num_workers=0,
    )
    mlm_batch=next(iter(mlm_loader))
    teacher_batch=next(iter(teacher_loader))

    semantic_compiled_cases={
        "max_runtime_axes":compile_semantic_operator_batch(
            rows=[choose_semantic(semantic_rows)],
            tokenizer=tokenizer,
        ),
        "long_context_semantic":compile_semantic_operator_batch(
            rows=[choose_long_semantic(semantic_long_rows)],
            tokenizer=tokenizer,
        ),
    }
    full_compiled=compile_behavioral_batch(
        rows=[choose_full_fabric(
            behavioral_rows,
            runtime_rows,
            long_rows,
        )],
        tokenizer=tokenizer,
    )
    natural_compiled=compile_natural_relation_batch(
        rows=[choose_natural(natural_rows)],
        relation_bank=relation_bank,
        tokenizer=tokenizer,
    )

    system,load_receipt=load_registered_full_envelope_system(
        topology_path=args.topology_config,
        semantic_config_path=args.semantic_config,
        semantic_checkpoint_path=args.semantic_checkpoint,
        device=device,
        runtime_profile=None,
    )
    stage_report=apply_stage_trainability(system,stage=J3)
    system.eval()
    objective=FullEnvelopeJointTrainingObjectiveV1().to(device).eval()

    mlm_batch=to_device(mlm_batch,device)
    teacher_batch=to_device(teacher_batch,device)
    semantic_compiled_cases={
        name:to_device(compiled,device)
        for name,compiled in semantic_compiled_cases.items()
    }
    full_compiled=to_device(full_compiled,device)
    natural_compiled=to_device(natural_compiled,device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident_before=int(torch.cuda.memory_allocated(device))
    reserved_before=int(torch.cuda.memory_reserved(device))

    semantic_case_receipts={}
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        for case_name,semantic_compiled in semantic_compiled_cases.items():
            result=execute_full_envelope_joint_step(
                system=system,
                objective=objective,
                mlm_batch=mlm_batch,
                teacher_batch=teacher_batch,
                semantic_operator_compiled=semantic_compiled,
                full_fabric_compiled=full_compiled,
                natural_relation_compiled=natural_compiled,
                update_ema=False,
                stage=J3,
            )
            loss=result["loss"]
            if (
                not isinstance(loss,torch.Tensor)
                or loss.ndim!=0
                or not bool(torch.isfinite(loss))
            ):
                raise SystemExit(
                    f"GPU no-gradient joint loss non-finite: {case_name}"
                )
            if result.get("all_public_training_lanes_executed") is not True:
                raise SystemExit(
                    f"J3 dry run did not execute every public lane: {case_name}"
                )
            if result.get("placeholder_losses_used") is not False:
                raise SystemExit(
                    f"J3 dry run used placeholder losses: {case_name}"
                )
            semantic_case_receipts[case_name]={
                "finite_joint_loss":True,
                "runtime_relation_count":int(
                    semantic_compiled["metadata"]["runtime_relation_count"]
                ),
                "max_reasoning_steps":int(
                    semantic_compiled["metadata"]["max_reasoning_steps"]
                ),
            }
    if any(p.grad is not None for p in system.parameters()):
        raise SystemExit("GPU dry run created gradients")

    peak_alloc=int(torch.cuda.max_memory_allocated(device))
    peak_reserved=int(torch.cuda.max_memory_reserved(device))
    total_memory=int(torch.cuda.get_device_properties(device).total_memory)
    trainable_params=sum(
        p.numel() for p in system.parameters() if p.requires_grad
    )
    model_bytes=sum(
        p.numel()*p.element_size() for p in system.parameters()
    )
    projection=dict(cfg["memory_projection"])
    activation_delta=max(0,peak_alloc-resident_before)
    projected=(
        model_bytes
        + trainable_params*int(projection["gradient_bytes_per_trainable_parameter"])
        + trainable_params*int(projection["adam_moment_bytes_per_trainable_parameter"])
        + int(math.ceil(
            activation_delta*float(
                projection["backward_activation_multiplier_over_no_grad_peak_delta"]
            )
        ))
        + int(projection["allocator_safety_margin_bytes"])
    )
    projected_fraction=projected/max(total_memory,1)
    local={
        "rank":rank,
        "local_rank":local_rank,
        "device_name":torch.cuda.get_device_name(device),
        "total_memory_bytes":total_memory,
        "resident_before_bytes":resident_before,
        "reserved_before_bytes":reserved_before,
        "peak_allocated_bytes":peak_alloc,
        "peak_reserved_bytes":peak_reserved,
        "no_grad_activation_delta_bytes":activation_delta,
        "model_parameter_bytes":model_bytes,
        "trainable_parameters":trainable_params,
        "projected_training_bytes":projected,
        "projected_training_fraction":projected_fraction,
        "projection_pass":projected_fraction<=float(
            projection["projected_training_fraction_max"]
        ),
    }
    gathered=[None for _ in range(world_size)]
    if world_size>1:
        dist.all_gather_object(gathered,local)
    else:
        gathered=[local]

    if rank==0:
        every_rank=all(bool(row["projection_pass"]) for row in gathered)
        output=Path(args.output)
        if output.exists():
            raise SystemExit("refusing to overwrite GPU dry-run result")
        result_json={
            "schema":"alice.eipm.n0.full-envelope-gpu-memory-result.v1",
            "status":PASS if every_rank else "FAIL_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1",
            "source_revision":mixture.get("source_revision"),
            "registered_system":"N0FullEnvelopeTrainableSystemV1",
            "registered_topology_load_receipt":load_receipt,
            "stage":J3,
            "stage_report":{
                "active_macro_families":list(stage_report["active_macro_families"]),
                "trainable_parameter_count":int(stage_report["trainable_parameter_count"]),
                "topology_parameter_count":int(stage_report["topology_parameter_count"]),
                "all_parameters_owned":bool(stage_report["all_parameters_owned"]),
            },
            "world_size":world_size,
            "preferred_world_size":preferred,
            "preferred_world_size_is_capability_ceiling":False,
            "microbatch_size":int(cfg["route"]["microbatch_size"]),
            "microbatch_size_is_capability_ceiling":False,
            "candidate_gradient_accumulation":int(
                cfg["route"]["initial_gradient_accumulation_operating_point"]
            ),
            "ranks":gathered,
            "public_corpus_sources":len(corpus_receipt.get("sources") or []),
            "teacher_registered_rows":int(teacher_report["registered_rows"]),
            "real_optimizer_facing_public_lanes":True,
            "semantic_case_receipts":semantic_case_receipts,
            "full_j3_counterfactual_path":True,
            "finite_joint_loss":True,
            "gradient":False,
            "backward":False,
            "optimizer_object_created":False,
            "weight_update":False,
            "model_training_performed":False,
            "projection_is_training_authorization":False,
            "gpu_training_authorized":False,
            "private_identity_data":False,
            "final_results_observed":False,
            "final_opening_authorized":False,
            "n0_complete":False,
        }
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(
            json.dumps(result_json,indent=2,sort_keys=True)+"\n",
            encoding="utf-8",
        )
        print(json.dumps(result_json,indent=2,sort_keys=True))
        if not every_rank:
            raise SystemExit(2)

    if world_size>1:
        dist.barrier()
        dist.destroy_process_group()


if __name__=="__main__":
    main()
