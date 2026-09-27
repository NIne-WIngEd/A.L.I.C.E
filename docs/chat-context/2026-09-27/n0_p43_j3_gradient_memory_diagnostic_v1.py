#!/usr/bin/env python3
"""Two-rank complete-J3 backward memory diagnostic; never a P43 PASS."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from typing import Any


PINNED = "a19f8e8893422702c138182f239064385addf91c"
QUALIFIER_SHA = "a4c33e247fcb05b97cf7ea3dbd632fb8eb6e4444ea72d971a0d9e243d732a7ef"
TRAINER_SHA = "a75a35c4ded5c16939e499f6ecf18f2bcc173428888764e8508fd449d5f2c6db"
SCHEMA = "alice.n0.p43.full-j3-two-rank-gradient-memory-diagnostic.v1"
EXPECTED_SEMANTIC = (
    "max_runtime_axes", "max_factor_cardinality", "long_context_semantic",
)
EXPECTED_FABRIC = (
    "max_candidate_cardinality", "max_field_cardinality",
    "max_edge_cardinality", "max_view_cardinality",
    "max_reasoning_depth", "long_additional_view_source",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parameter_digest(model: Any) -> str:
    """Check that this backward-only probe did not change a model parameter."""
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode("utf-8"))
        tensor = parameter.detach().cpu().contiguous().numpy()
        h.update(memoryview(tensor).cast("B"))
    return h.hexdigest()


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def memory(device: Any, torch: Any) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "driver_used_snapshot_bytes": int(total - free),
        "total_bytes": int(total),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def input_paths(parser: argparse.ArgumentParser) -> None:
    for name in (
        "qualification-config", "training-plan", "topology-config",
        "semantic-config", "semantic-checkpoint", "tokenizer-dir",
        "corpus-dir", "source-config", "teacher-registry",
        "teacher-audit", "semantic-rows", "semantic-long-rows",
        "behavioral-rows", "runtime-view-rows", "long-context-rows",
        "fewrel-rows", "fewrel-bank", "mixture-manifest",
        "mixture-audit", "output-root",
    ):
        parser.add_argument("--" + name, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_paths(parser)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != PINNED:
        raise SystemExit("requires the exact pinned scientific N0 source")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("requires a clean pinned scientific N0 worktree")
    qualifier = repo / "scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py"
    trainer = repo / "scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py"
    if digest(qualifier) != QUALIFIER_SHA or digest(trainer) != TRAINER_SHA:
        raise SystemExit("original qualifier or trainer source hash drift")

    import torch
    import torch.distributed as dist
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from torch.utils.data import DataLoader
    from alice_personality.n0.curriculum_data import CurriculumDataset, load_tokenizer
    from alice_personality.n0.data import PackedJSONLIterableDataset, SpanMLMCollator
    from alice_personality.n0.full_envelope_behavioral_batch_v1 import compile_behavioral_batch
    from alice_personality.n0.full_envelope_joint_step_v1 import execute_full_envelope_joint_step
    from alice_personality.n0.full_envelope_runtime_factory_v1 import load_registered_full_envelope_system
    from alice_personality.n0.full_envelope_stage_policy_v1 import (
        ALL_FAMILIES, J3, apply_stage_trainability,
    )
    from alice_personality.n0.full_envelope_training_objective_v1 import FullEnvelopeJointTrainingObjectiveV1
    from alice_personality.n0.natural_relation_batch_v1 import compile_natural_relation_batch
    from alice_personality.n0.semantic_operator_batch_v1 import compile_semantic_operator_batch
    from alice_personality.n0.v02_training import (
        TeacherMultitaskCollator, verify_public_corpus_v021,
        verify_teacher_registry, verify_tokenizer_v021,
    )

    spec = importlib.util.spec_from_file_location("pinned_p43_qualifier", qualifier)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot import pinned original qualifier")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    for actual, canonical in (
        (args.qualification_config, "configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json"),
        (args.training_plan, "configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json"),
        (args.topology_config, "configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json"),
        (args.semantic_config, "configs/eipm/n0/alice_n0_semantic_v0.2.json"),
        (args.source_config, "configs/eipm/n0/public_corpus_v0.2.1.activated.json"),
    ):
        q.require_canonical_source_file(actual, canonical, label="gradient diagnostic source")

    cfg = q.read_json(args.qualification_config)
    plan = q.read_json(args.training_plan)
    route = cfg["route"]
    if (cfg.get("schema") != "alice.eipm.n0.full-envelope-gpu-memory-dry-run.v1"
            or plan.get("schema") != "alice.eipm.n0.semantic-operator-joint-training-plan.v1"
            or int(route.get("preferred_world_size", 0)) != 2
            or int(route.get("microbatch_size", 0)) != 1
            or int(route.get("teacher_batch_size", 0)) != 2
            or int(route.get("replay_sequence_length", 0)) != 512
            or route.get("training_mixed_precision") != "fp16"
            or route.get("ddp_replica_topology") is not True
            or float(cfg["memory_projection"]["projected_training_fraction_max"]) != 0.85
            or plan["optimization_strategy"].get("gradient_checkpointing") is not True):
        raise SystemExit("registered J3 operating geometry or trainer checkpoint policy drift")
    # The original P43 config explicitly forbids gradients for *its* authority.
    # This separate, owner-requested experiment cannot replace that P43 receipt.
    if (cfg["qualification"].get("no_gradient") is not True
            or cfg["authorization"].get("gpu_training") is not False):
        raise SystemExit("original P43 authority boundary drift")

    mixture = q.read_json(args.mixture_manifest)
    audit = q.read_json(args.mixture_audit)
    if (mixture.get("source_revision") != PINNED
            or mixture.get("status") != "MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT"
            or mixture.get("final_results_observed") is not False
            or int(mixture.get("final_rows_in_training", -1)) != 0
            or audit.get("status") != "PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1"
            or audit.get("manifest_sha256") != digest(Path(args.mixture_manifest))):
        raise SystemExit("source-bound mixture authority or audit mismatch")
    lanes = mixture["training_lanes"]
    for lane, path in (
        ("semantic_operator_intervention", args.semantic_rows),
        ("semantic_operator_long_context", args.semantic_long_rows),
        ("full_envelope_behavioral", args.behavioral_rows),
        ("runtime_view_supplement", args.runtime_view_rows),
        ("long_context_supplement", args.long_context_rows),
        ("natural_relation", args.fewrel_rows),
    ):
        if lanes[lane]["rows_sha256"] != digest(Path(path)):
            raise SystemExit("mixture row hash mismatch: " + lane)
    if lanes["natural_relation"]["bank_sha256"] != digest(Path(args.fewrel_bank)):
        raise SystemExit("natural bank hash mismatch")
    if lanes["broad_semantic_replay"]["source_config_sha256"] != digest(Path(args.source_config)):
        raise SystemExit("public source config hash mismatch")
    if lanes["broad_semantic_replay"]["corpus_receipt_sha256"] != digest(Path(args.corpus_dir) / "corpus_receipt.json"):
        raise SystemExit("public corpus receipt hash mismatch")
    if lanes["governed_judgment_replay"]["teacher_registry_sha256"] != digest(Path(args.teacher_registry)):
        raise SystemExit("teacher registry hash mismatch")
    if lanes["governed_judgment_replay"]["teacher_audit_sha256"] != digest(Path(args.teacher_audit)):
        raise SystemExit("teacher audit hash mismatch")

    if (int(os.environ.get("WORLD_SIZE", "0")) != 2
            or not torch.cuda.is_available()
            or torch.cuda.device_count() < 2):
        raise SystemExit("requires real two-rank CUDA DDP; no single-process fallback")
    seed = int(plan["optimization_strategy"]["training_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.num_processes != 2 or accelerator.device.type != "cuda":
        raise SystemExit("Accelerate did not produce two CUDA ranks")
    rank, device = int(accelerator.process_index), accelerator.device
    output = Path(args.output_root).resolve()
    if not output.is_dir():
        raise SystemExit("batch launcher must create a fresh evidence directory first")
    progress = output / f"rank-{rank}.jsonl"
    with progress.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": SCHEMA, "event": "rank_started", "rank": rank,
            "source_revision": PINNED, "gpu": torch.cuda.get_device_name(device),
            "authority": False, "weight_update": False,
        }, sort_keys=True) + "\n")
    current_case = "preparation"
    try:
        verify_tokenizer_v021(args.tokenizer_dir)
        tokenizer = load_tokenizer(args.tokenizer_dir)
        _, corpus_paths = verify_public_corpus_v021(args.corpus_dir, args.source_config)
        curriculum_paths, _ = verify_teacher_registry(
            repo, args.teacher_registry, args.teacher_audit,
        )
        semantic_rows = q.train_rows(args.semantic_rows)
        semantic_long_rows = q.train_rows(args.semantic_long_rows)
        behavioral_rows = q.train_rows(args.behavioral_rows)
        runtime_rows = q.train_rows(args.runtime_view_rows)
        long_rows = q.train_rows(args.long_context_rows)
        natural_rows = q.train_rows(args.fewrel_rows)
        bank = q.read_json(args.fewrel_bank)
        semantic_cases = {
            "max_runtime_axes": q.choose_semantic(semantic_rows),
            "max_factor_cardinality": q.choose_max_factor_semantic(semantic_rows),
            "long_context_semantic": q.choose_long_semantic(semantic_long_rows),
        }
        fabric_cases = q.choose_full_fabric_cases(behavioral_rows, runtime_rows, long_rows)
        if (tuple(semantic_cases) != EXPECTED_SEMANTIC
                or tuple(fabric_cases) != EXPECTED_FABRIC
                or set(cfg["qualification"]["required_semantic_memory_cases"]) != set(semantic_cases)
                or set(cfg["qualification"]["required_full_fabric_memory_cases"]) != set(fabric_cases)):
            raise RuntimeError("full 3 × 6 stress-case geometry drift")
        semantic_compiled = {
            name: compile_semantic_operator_batch(rows=[row], tokenizer=tokenizer)
            for name, row in semantic_cases.items()
        }
        fabric_compiled = {
            name: compile_behavioral_batch(rows=[row], tokenizer=tokenizer)
            for name, row in fabric_cases.items()
        }
        natural = compile_natural_relation_batch(
            rows=[q.choose_natural(natural_rows)],
            relation_bank=bank, tokenizer=tokenizer,
        )
        mlm_loader = DataLoader(
            PackedJSONLIterableDataset(
                paths=corpus_paths, tokenizer=tokenizer, sequence_length=512,
                split="train", shuffle_seed=seed,
            ),
            batch_size=1,
            collate_fn=SpanMLMCollator(
                tokenizer=tokenizer, mlm_probability=0.30,
                mean_span=3.0, max_span=10, seed=seed + rank,
            ),
            num_workers=0, pin_memory=True,
        )
        generator = torch.Generator()
        generator.manual_seed(seed + rank)
        teacher_loader = DataLoader(
            CurriculumDataset(curriculum_paths, "train"),
            batch_size=2, shuffle=True, generator=generator,
            collate_fn=TeacherMultitaskCollator(tokenizer, 256),
            num_workers=0, pin_memory=True,
        )
        mlm = next(iter(mlm_loader))
        teacher = next(iter(teacher_loader))

        system, _ = load_registered_full_envelope_system(
            topology_path=args.topology_config,
            semantic_config_path=args.semantic_config,
            semantic_checkpoint_path=args.semantic_checkpoint,
            device="cpu", runtime_profile=None,
        )
        stage = apply_stage_trainability(system, stage=J3)
        trainer_spec = importlib.util.spec_from_file_location("pinned_joint_trainer", trainer)
        if trainer_spec is None or trainer_spec.loader is None:
            raise RuntimeError("cannot import pinned full-envelope trainer")
        trainer_module = importlib.util.module_from_spec(trainer_spec)
        trainer_spec.loader.exec_module(trainer_module)
        trainer_module.enable_precommitted_gradient_checkpointing(system)
        checkpoint_modules = [
            module for module in system.semantic_model.mlm.modules()
            if getattr(module, "gradient_checkpointing", False) is True
        ]
        checkpoint_modes = []
        for module in checkpoint_modules:
            fn = getattr(module, "_gradient_checkpointing_func", None)
            keywords = getattr(fn, "keywords", None)
            checkpoint_modes.append(
                None if keywords is None else keywords.get("use_reentrant")
            )
        append_record(progress, {
            "event": "checkpoint_mode", "rank": rank,
            "enabled_modules": len(checkpoint_modules),
            "use_reentrant_modes": checkpoint_modes,
            "ddp_find_unused_parameters": True,
            "torch_version": torch.__version__,
            "accelerate_version": version("accelerate"),
            "transformers_version": version("transformers"),
        })
        if not checkpoint_modes or any(mode is not False for mode in checkpoint_modes):
            raise RuntimeError(
                "production checkpoint mode is not verified nonreentrant; "
                "DDP find_unused_parameters=True needs a versioned route review"
            )
        objective = FullEnvelopeJointTrainingObjectiveV1()
        system = accelerator.prepare(system)
        objective = objective.to(device)
        system.train()
        objective.train()
        if not isinstance(system, torch.nn.parallel.DistributedDataParallel):
            raise RuntimeError("trainer-style two-rank DDP wrapper absent")
        if (len(tuple(ALL_FAMILIES)) != 10
                or stage.get("stage") != J3
                or stage.get("architecture_reduced") is not False
                or stage.get("all_parameters_owned") is not True):
            raise RuntimeError("stage trainability or ten-family policy drift")
        unwrapped = accelerator.unwrap_model(system)
        before_digest = parameter_digest(unwrapped)
        mlm, teacher, natural = (q.to_device(x, device) for x in (mlm, teacher, natural))
        torch.cuda.empty_cache()
        total = int(torch.cuda.get_device_properties(device).total_memory)
        cap = int(total * 0.85)
        append_record(progress, {
            "event": "loaded", "rank": rank,
            "total_memory_bytes": total, "capacity_85_percent_bytes": cap,
            "registered_parameter_digest_before": before_digest,
            "trainable_parameters": sum(p.numel() for p in system.parameters() if p.requires_grad),
            "trainer_sha256": TRAINER_SHA,
            "qualifier_sha256": QUALIFIER_SHA,
            "training_plan_sha256": digest(Path(args.training_plan)),
            "mixture_manifest_sha256": digest(Path(args.mixture_manifest)),
            "semantic_checkpoint_sha256": digest(Path(args.semantic_checkpoint)),
            "gradient_checkpointing_enabled": bool(
                unwrapped.semantic_model.mlm.is_gradient_checkpointing
            ),
            "optimizer_object_created": False,
            "optimizer_state_measured": False,
            "baseline": memory(device, torch),
        })
        completed = 0
        worst_reserved = 0
        worst_driver_snapshot = 0
        for semantic_name in semantic_cases:
            for fabric_name in fabric_cases:
                current_case = semantic_name + "__" + fabric_name
                print(f"GRADIENT_CASE_START rank={rank} case={current_case}", flush=True)
                system.zero_grad(set_to_none=True)
                semantic = q.to_device(semantic_compiled[semantic_name], device)
                fabric = q.to_device(fabric_compiled[fabric_name], device)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline = memory(device, torch)
                result = execute_full_envelope_joint_step(
                    system=system, objective=objective,
                    mlm_batch=mlm, teacher_batch=teacher,
                    semantic_operator_compiled=semantic,
                    full_fabric_compiled=fabric,
                    natural_relation_compiled=natural,
                    update_ema=False, stage=J3,
                )
                loss = result["loss"]
                if (loss.ndim != 0 or not bool(torch.isfinite(loss))
                        or result.get("all_public_training_lanes_executed") is not True
                        or result.get("placeholder_losses_used") is not False
                        or result.get("requires_full_fabric_counterfactuals") is not True
                        or tuple(result.get("active_families", ())) != tuple(ALL_FAMILIES)):
                    raise RuntimeError("incomplete or non-finite full J3 joint step")
                forward = memory(device, torch)
                torch.cuda.reset_peak_memory_stats(device)
                accelerator.backward(loss)
                backward = memory(device, torch)
                grad_tensors = [p.grad for p in system.parameters() if p.requires_grad and p.grad is not None]
                if not grad_tensors or not all(bool(torch.isfinite(g).all()) for g in grad_tensors):
                    raise RuntimeError("missing or non-finite full J3 gradients")
                nonzero = sum(int(torch.count_nonzero(g)) for g in grad_tensors)
                if nonzero == 0:
                    raise RuntimeError("all full-J3 gradients are zero")
                worst_reserved = max(
                    worst_reserved, forward["peak_reserved_bytes"],
                    backward["peak_reserved_bytes"],
                )
                worst_driver_snapshot = max(
                    worst_driver_snapshot,
                    baseline["driver_used_snapshot_bytes"],
                    forward["driver_used_snapshot_bytes"],
                    backward["driver_used_snapshot_bytes"],
                )
                record = {
                    "event": "case_completed", "rank": rank, "case": current_case,
                    "finite_joint_loss": True, "active_macro_family_count": len(ALL_FAMILIES),
                    "full_fabric_view_count": 4,
                    "finite_gradient_tensors": len(grad_tensors),
                    "nonzero_gradient_elements": nonzero,
                    "loss": float(loss.detach().float()),
                    "baseline": baseline, "forward": forward, "backward": backward,
                    "observed_memory_within_85_percent": (
                        max(worst_reserved, worst_driver_snapshot) <= cap
                    ),
                    "optimizer_object_created": False, "weight_update": False,
                }
                append_record(progress, record)
                completed += 1
                print(f"GRADIENT_CASE_DONE rank={rank} case={current_case} peak_reserved={worst_reserved}", flush=True)
                del result, loss, semantic, fabric, grad_tensors
                system.zero_grad(set_to_none=True)
                if max(worst_reserved, worst_driver_snapshot) > cap:
                    raise RuntimeError("observed CUDA reservation or driver snapshot exceeded unchanged 85-percent limit")
        after_digest = parameter_digest(unwrapped)
        if before_digest != after_digest:
            raise RuntimeError("parameter weights changed during backward-only diagnostic")
        summary = {
            "schema": SCHEMA, "rank": rank, "status": "COMPLETE_GRADIENT_DIAGNOSTIC_NOT_P43_AUTHORITY",
            "source_revision": PINNED, "world_size": 2,
            "stress_pair_count": completed,
            "mixture_manifest_sha256": digest(Path(args.mixture_manifest)),
            "optimizer_object_created": False, "optimizer_step_measured": False,
            "weight_update": False, "parameter_sha256_before": before_digest,
            "parameter_sha256_after": after_digest,
            "backward_measured": True, "gradient_checkpointing_enabled": True,
            "max_peak_reserved_bytes": worst_reserved, "total_memory_bytes": total,
            "max_driver_used_snapshot_bytes": worst_driver_snapshot,
            "capacity_85_percent_bytes": cap,
            "observed_memory_within_85_percent": (
                max(worst_reserved, worst_driver_snapshot) <= cap
            ),
            "measured_training_peak_bytes": None, "training_authorized": False,
            "final_results_observed": False, "private_identity_data": False,
        }
        append_record(progress, {"event": "rank_completed", **summary})
        gathered = [None, None]
        dist.all_gather_object(gathered, summary)
        if rank == 0:
            if not all(
                r["status"] == "COMPLETE_GRADIENT_DIAGNOSTIC_NOT_P43_AUTHORITY"
                and r["stress_pair_count"] == 18 and r["weight_update"] is False
                for r in gathered
            ):
                raise RuntimeError("incomplete two-rank gradient diagnostic")
            result_path = output / "diagnostic.json"
            with result_path.open("x", encoding="utf-8") as stream:
                json.dump({
                    "schema": SCHEMA, "status": "COMPLETE_TWO_RANK_GRADIENT_DIAGNOSTIC_NOT_P43_AUTHORITY",
                    "ranks": gathered,
                    "gradient_probe_only": True, "optimizer_step_measured": False,
                    "measured_training_peak_bytes": None,
                    "training_authorized": False, "weight_update": False,
                }, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print("COMPLETE_TWO_RANK_GRADIENT_DIAGNOSTIC_NOT_P43_AUTHORITY", flush=True)
    except BaseException as exc:
        try:
            failure_memory = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        except Exception:
            failure_memory = None
        append_record(progress, {
            "event": "rank_failed", "rank": rank, "case": current_case,
            "error_type": type(exc).__name__, "error": str(exc),
            "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
            "failure_memory": failure_memory,
            "training_authorized": False, "weight_update": False,
        })
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
