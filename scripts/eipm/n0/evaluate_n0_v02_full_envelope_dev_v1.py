#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
    compile_behavioral_batch,
)
from alice_personality.n0.full_envelope_joint_step_v1 import (
    governed_judgment_replay_loss,
)
from alice_personality.n0.full_envelope_runtime_factory_v1 import (
    load_registered_full_envelope_system,
)
from alice_personality.n0.full_envelope_stage_policy_v1 import (
    J1,J2,J3,apply_stage_trainability,resolve_stage_policy,
)
from alice_personality.n0.full_envelope_training_objective_v1 import (
    behavioral_supervision,
    semantic_operator_supervision,
)
from alice_personality.n0.natural_relation_batch_v1 import (
    compile_natural_relation_batch,
    natural_relation_semantic_loss,
)
from alice_personality.n0.semantic_operator_batch_v1 import (
    compile_semantic_operator_batch,
)
from alice_personality.n0.v02_training import (
    TeacherMultitaskCollator,
    sha256_file,
    verify_teacher_registry,
)


STAGES=(J1,J2,J3)


def read_json(path: str | Path) -> dict[str,Any]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value,dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def read_dev_rows(path: str | Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected=[]
    for row in rows:
        split=str(row.get("split",""))
        if split=="final" or row.get("final_validation_only") is True:
            raise SystemExit("FINAL rows are forbidden in successor DEV evaluator")
        if split!="dev":
            continue
        if row.get("private_identity_data") is not False:
            raise SystemExit("private identity data forbidden in N0 DEV")
        if row.get("model_selection_authorized") not in (None,True):
            raise SystemExit(f"DEV model-selection authority drift: {row.get('id')}")
        selected.append(row)
    if not selected:
        raise SystemExit(f"DEV lane is empty: {path}")
    return selected


def to_device(value: Any,device: torch.device) -> Any:
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):
        return [to_device(v,device) for v in value]
    if isinstance(value,tuple):
        return tuple(to_device(v,device) for v in value)
    return value


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty DEV metric")
    return sum(values)/len(values)


def teacher_top1_rate(
    *,
    system: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float,float]:
    correct=0
    total=0
    losses=[]
    for batch in loader:
        batch=to_device(batch,device)
        loss,_=governed_judgment_replay_loss(
            system=system,batch=batch
        )
        losses.append(float(loss.detach().cpu()))
        outputs=system(task="teacher",batch=batch)
        scores=outputs["scores"]
        offset=0
        for size,preferred in zip(
            batch["group_sizes"],batch["preferred_masks"]
        ):
            local=scores[offset:offset+int(size)]
            prediction=int(local.argmax().item())
            mask=preferred.to(device=local.device,dtype=torch.bool)
            correct+=int(bool(mask[prediction]))
            total+=1
            offset+=int(size)
    if total<=0:
        raise RuntimeError("teacher DEV produced no groups")
    return correct/total,mean(losses)


def semantic_lane_metrics(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
) -> dict[str,float]:
    totals=defaultdict(list)
    for row in rows:
        compiled=compile_semantic_operator_batch(
            rows=[row],tokenizer=tokenizer
        )
        compiled=to_device(compiled,device)
        outputs=system(
            task="semantic_operator",
            batch=compiled["batch"],
        )
        losses=semantic_operator_supervision(
            outputs=outputs,
            targets=compiled["operator_targets"],
        )
        for name,value in losses.items():
            if isinstance(value,torch.Tensor) and value.ndim==0:
                totals[name].append(float(value.detach().cpu()))
    return {name:mean(values) for name,values in sorted(totals.items())}


def natural_lane_metrics(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    bank: Mapping[str,Any],
    tokenizer: Any,
    device: torch.device,
) -> dict[str,float]:
    losses=[]
    correct=0
    total=0
    for row in rows:
        compiled=compile_natural_relation_batch(
            rows=[row],relation_bank=bank,tokenizer=tokenizer
        )
        compiled=to_device(compiled,device)
        outputs=system(
            task="natural_relation",
            batch=compiled["batch"],
        )
        loss=natural_relation_semantic_loss(
            outputs=outputs,
            target_relation_index=compiled["target_relation_index"],
        )
        losses.append(float(loss.detach().cpu()))
        logits=outputs["semantic_operator"]["relation_logits"][:,0,:]
        target=compiled["target_relation_index"]
        correct+=int((logits.argmax(dim=-1)==target).sum().item())
        total+=int(target.numel())
    return {
        "loss":mean(losses),
        "top1":correct/max(total,1),
        "rows":float(total),
    }


def fabric_lane_metrics(
    *,
    system: torch.nn.Module,
    rows: list[dict[str,Any]],
    tokenizer: Any,
    device: torch.device,
    stage: str,
) -> dict[str,float]:
    policy=resolve_stage_policy(stage)
    totals=defaultdict(list)
    for row in rows:
        compiled=compile_behavioral_batch(
            rows=[row],tokenizer=tokenizer
        )
        compiled=to_device(compiled,device)
        primary=system(
            task="full_envelope",
            batch=compiled["primary_batch"],
        )
        decisive=irrelevant=permuted=None
        if "multi_view_causal_preservation" in policy.active_macro_families:
            decisive=system(
                task="full_envelope",
                batch=compiled["decisive_ablated_batch"],
            )
            irrelevant=system(
                task="full_envelope",
                batch=compiled["irrelevant_removed_batch"],
            )
            permuted=system(
                task="full_envelope",
                batch=compiled["permuted_batch"],
            )
        losses=behavioral_supervision(
            primary_outputs=primary,
            decisive_ablated_outputs=decisive,
            irrelevant_removed_outputs=irrelevant,
            permuted_outputs=permuted,
            targets=compiled["behavioral_targets"],
            active_families=policy.active_macro_families,
        )
        for name,value in losses.items():
            if isinstance(value,torch.Tensor) and value.ndim==0:
                totals[name].append(float(value.detach().cpu()))
    return {name:mean(values) for name,values in sorted(totals.items())}


def main() -> None:
    p=argparse.ArgumentParser(
        description=(
            "DEV-only successor evaluator for N0FullEnvelopeTrainableSystemV1. "
            "FINAL rows/results are forbidden and this script cannot open FINAL."
        )
    )
    p.add_argument("--stage",choices=STAGES,required=True)
    p.add_argument("--topology-config",required=True)
    p.add_argument("--semantic-config",required=True)
    p.add_argument("--semantic-checkpoint",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--candidate-system",required=True)
    p.add_argument("--candidate-receipt",required=True)
    p.add_argument("--mixture-manifest",required=True)
    p.add_argument("--mixture-audit",required=True)
    p.add_argument("--teacher-registry",required=True)
    p.add_argument("--teacher-audit",required=True)
    p.add_argument("--semantic-rows",required=True)
    p.add_argument("--semantic-long-rows",required=True)
    p.add_argument("--behavioral-rows",required=True)
    p.add_argument("--runtime-view-rows",required=True)
    p.add_argument("--long-context-rows",required=True)
    p.add_argument("--natural-rows",required=True)
    p.add_argument("--natural-bank",required=True)
    p.add_argument("--dev-contract",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--teacher-batch-size",type=int,default=8)
    args=p.parse_args()

    if args.teacher_batch_size<=0:
        raise SystemExit("teacher DEV batch size must be positive")
    contract=read_json(args.dev_contract)
    if contract.get("schema")!="alice.eipm.n0.full-envelope-dev-validation-contract.v1":
        raise SystemExit("DEV validation contract schema drift")
    if contract.get("checkpoint_selection_surface")!="DEV_ONLY":
        raise SystemExit("DEV evaluator contract lost DEV-only authority")
    if contract.get("final_rows_allowed") is not False:
        raise SystemExit("DEV contract unexpectedly allows FINAL rows")
    if contract.get("final_results_allowed") is not False:
        raise SystemExit("DEV contract unexpectedly allows FINAL results")

    candidate=read_json(args.candidate_receipt)
    if candidate.get("stage")!=args.stage:
        raise SystemExit("candidate checkpoint stage drift")
    if candidate.get("final_results_observed") is not False:
        raise SystemExit("candidate checkpoint has observed FINAL")
    if candidate.get("private_identity_data") is not False:
        raise SystemExit("candidate checkpoint contains private identity data")

    mixture=read_json(args.mixture_manifest)
    mixture_audit=read_json(args.mixture_audit)
    if mixture_audit.get("status")!="PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1":
        raise SystemExit("full public-mixture audit not PASS")
    if mixture.get("final_results_observed") is not False:
        raise SystemExit("mixture observed FINAL")
    if candidate.get("full_public_mixture_manifest_sha256")!=sha256_file(
        args.mixture_manifest
    ):
        raise SystemExit("candidate/mixture manifest lineage drift")
    if candidate.get("full_public_mixture_audit_sha256")!=sha256_file(
        args.mixture_audit
    ):
        raise SystemExit("candidate/mixture audit lineage drift")

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("safetensors required for DEV evaluation") from exc

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system,_=load_registered_full_envelope_system(
        topology_path=args.topology_config,
        semantic_config_path=args.semantic_config,
        semantic_checkpoint_path=args.semantic_checkpoint,
        device=device,
        runtime_profile=None,
    )
    state=load_file(str(args.candidate_system),device="cpu")
    system.load_state_dict(state,strict=True)
    apply_stage_trainability(system,stage=args.stage)
    system.eval()

    tokenizer=load_tokenizer(args.tokenizer_dir)
    repo_root=Path(__file__).resolve().parents[3]
    curriculum_paths,teacher_report=verify_teacher_registry(
        repo_root,args.teacher_registry,args.teacher_audit
    )
    teacher_dev=CurriculumDataset(curriculum_paths,"dev")
    teacher_loader=DataLoader(
        teacher_dev,
        batch_size=args.teacher_batch_size,
        shuffle=False,
        collate_fn=TeacherMultitaskCollator(tokenizer,256),
        num_workers=0,
    )

    semantic=read_dev_rows(args.semantic_rows)
    semantic_long=read_dev_rows(args.semantic_long_rows)
    behavioral=read_dev_rows(args.behavioral_rows)
    runtime_view=read_dev_rows(args.runtime_view_rows)
    long_context=read_dev_rows(args.long_context_rows)
    natural=read_dev_rows(args.natural_rows)
    natural_bank=read_json(args.natural_bank)

    with torch.inference_mode():
        teacher_top1,teacher_loss=teacher_top1_rate(
            system=system,loader=teacher_loader,device=device
        )
        semantic_metrics=semantic_lane_metrics(
            system=system,rows=semantic,tokenizer=tokenizer,device=device
        )
        semantic_long_metrics=semantic_lane_metrics(
            system=system,rows=semantic_long,tokenizer=tokenizer,device=device
        )
        natural_metrics=natural_lane_metrics(
            system=system,rows=natural,bank=natural_bank,
            tokenizer=tokenizer,device=device
        )
        fabric_metrics={}
        if args.stage in {J2,J3}:
            fabric_metrics["full_envelope_behavioral"]=fabric_lane_metrics(
                system=system,rows=behavioral,tokenizer=tokenizer,
                device=device,stage=args.stage
            )
            fabric_metrics["long_context_supplement"]=fabric_lane_metrics(
                system=system,rows=long_context,tokenizer=tokenizer,
                device=device,stage=args.stage
            )
        if args.stage==J3:
            fabric_metrics["runtime_view_supplement"]=fabric_lane_metrics(
                system=system,rows=runtime_view,tokenizer=tokenizer,
                device=device,stage=args.stage
            )

    stage_plan=read_json(contract["stage_gate_source"])
    declared=list(stage_plan["stage_gates"][
        "J1" if args.stage==J1 else "J2" if args.stage==J2 else "J3"
    ])
    # This evaluator deliberately does not pretend that a finite aggregate
    # loss proves every named capability gate. It emits real DEV measurements
    # and blocks stage selection until every declared gate has an explicit
    # metric mapping. FINAL data may never be used to fill that gap.
    mapped={
        "historical_semantic_regression":{
            "metric":"teacher_dev_top1",
            "value":teacher_top1,
        },
        "natural_heldout_relation_family_semantics":{
            "metric":"natural_relation_top1",
            "value":natural_metrics["top1"],
        },
    }
    unmapped=[name for name in declared if name not in mapped]
    gate_coverage_complete=not unmapped
    stage_gate_pass=False

    result={
        "schema":"alice.eipm.n0.full-envelope-dev-evaluation.v1",
        "status":"DEV_METRICS_RECORDED_GATE_MAPPING_INCOMPLETE" if unmapped else "DEV_METRICS_RECORDED",
        "stage":args.stage,
        "checkpoint_selection_surface":"DEV_ONLY",
        "candidate_checkpoint_receipt_sha256":sha256_file(args.candidate_receipt),
        "candidate_system_sha256":sha256_file(args.candidate_system),
        "teacher_registered_rows":int(teacher_report["registered_rows"]),
        "metrics":{
            "teacher_dev_top1":teacher_top1,
            "teacher_dev_loss":teacher_loss,
            "semantic_operator":semantic_metrics,
            "semantic_operator_long_context":semantic_long_metrics,
            "natural_relation":natural_metrics,
            "full_fabric":fabric_metrics,
        },
        "declared_stage_gates":declared,
        "mapped_stage_gates":mapped,
        "unmapped_stage_gates":unmapped,
        "stage_gate_coverage_complete":gate_coverage_complete,
        "stage_gate_pass":stage_gate_pass,
        "checkpoint_selection_performed":False,
        "automatic_stage_transition":False,
        "final_validation_only":False,
        "final_results_observed":False,
        "final_opening_authorized":False,
        "final_checkpoint_selection_allowed":False,
        "private_identity_data":False,
        "n0_complete":False,
    }
    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite DEV evaluation")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
