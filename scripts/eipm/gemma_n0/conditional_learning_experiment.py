"""Matched native N2 learning with admitted same-context corrections.

The native architecture stays unchanged. Each case supplies equally weighted
natural and replay losses; control uses unchanged replay, treatment uses its
separately admitted TRAIN N1 record. Complete groups, optimizer state and RNG
are durably retained in a distinct experimental schema, never native approval.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys

import torch

from scripts.eipm.gemma_n0 import paired_learning_experiment as original
from scripts.eipm.gemma_n0 import conditional_teaching as teaching
from src.alice_personality.identity.codec import canonical, fingerprint

ORIGINAL_HELPER_SHA256 = "d3d9b52b7abbd702fd55fd39ab2c0e72c19b122d500f39953c8f5cfcba042bd2"
SNAPSHOT_SCHEMA = "alice-personality-conditional-learning-progress-v1"
PLAN_SCHEMA = "alice-personality-conditional-learning-plan-v1"
EXPERIMENT_SCHEMA = "alice-personality-conditional-learning-experiment-v1"
plan_sha256 = original.plan_sha256


def _groups(plan):
    _require(type(plan) is dict and set(plan) == {"schema", "model_state_sha256", "model_config_sha256",
        "objective_state_sha256", "recipe_sha256", "batch_sha256", "matched_groups", "correction_sha256", "bindings"}
        and plan["schema"] == PLAN_SCHEMA, "conditional learning plan schema differs")
    inventory, groups, corrections = plan["batch_sha256"], plan["matched_groups"], plan["correction_sha256"]
    _require(type(inventory) is list and inventory and len(set(inventory)) == len(inventory)
        and all(type(d) is str and original._DIGEST.fullmatch(d) for d in inventory), "unique native batch inventory required")
    _require(type(groups) is list and len(groups) >= 2 and all(type(g) is list and len(g) >= 2 for g in groups)
        and Counter(d for g in groups for d in g) == Counter(inventory), "complete identical native groups required")
    _require(type(corrections) is dict and set(corrections) <= set(inventory)
        and all(type(d) is str and original._DIGEST.fullmatch(d) for d in corrections.values()), "correction inventory differs")
    _require(type(plan["bindings"]) is dict, "explicit owner custody bindings required")
    return {"control": groups, "intervention": deepcopy(groups)}


def make_plan(model, objectives, recipe, *, batch_sha256, matched_groups, correction_sha256, bindings):
    plan = {"schema": PLAN_SCHEMA, "model_state_sha256": fingerprint(model.state_dict()),
        "model_config_sha256": fingerprint(model.config), "objective_state_sha256": fingerprint(objectives.state_dict()),
        "recipe_sha256": recipe.fingerprint(), "batch_sha256": list(batch_sha256),
        "matched_groups": [list(g) for g in matched_groups], "correction_sha256": dict(correction_sha256),
        "bindings": deepcopy(bindings)}
    _groups(plan); original._identity(model, objectives, recipe, plan)
    return plan


def _correction(plan, batch, loader):
    digest = batch.fingerprint()
    record = loader(digest)
    expected = plan["correction_sha256"].get(digest)
    _require((record is None) == (expected is None), "correction coverage differs from admitted plan")
    if record is not None:
        _require(teaching.record_sha256(record) == expected, "correction record changed")
        teaching.validate_correction_record(record, batch, expected_sha256=expected)
    return record, expected
_require = original._require
_ARM_KEYS = {"model_state", "objective_state", "optimizer_state", "model_state_sha256",
             "torch_rng", "cuda_rng", "events", "updates", "endpoint_forwards",
             "batch_occurrences", "n1_n3_unchanged"}


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _plain_path(path):
    path = Path(path)
    _require(path.is_absolute() and not any(
        item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction())
        for item in (path, *path.parents)), "experimental artifact path must be absolute without links")
    return path


def _sync_directory(path):
    # Directory fsync is available on the Linux execution host, not Windows.
    # A real Linux storage failure propagates; it is never silently called durable.
    if sys.platform.startswith("linux"):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write(path, value, *, tensor=False):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        if tensor:
            torch.save(value, stream)
        else:
            stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _runtime(model, recipe):
    return {"python": list(sys.version_info[:3]), "platform": sys.platform,
            "machine": platform.machine(), "torch": str(torch.__version__),
            "device": str(next(model.parameters()).device),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic": recipe.deterministic,
            "model_tensors": [[name, str(value.dtype), list(value.shape)]
                              for name, value in model.state_dict().items()]}


def _arm_state(clone, objectives, optimizer, events, occurrences, devices):
    return {"model_state": original._cpu(clone.state_dict()),
            "objective_state": original._cpu(objectives.state_dict()),
            "optimizer_state": original._cpu(optimizer.state_dict()),
            "model_state_sha256": fingerprint(clone.state_dict()),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": [torch.cuda.get_rng_state(device).cpu() for device in devices],
            "events": deepcopy(events), "updates": len(events),
            "endpoint_forwards": sum(occurrences.values()), "batch_occurrences": dict(occurrences),
            "n1_n3_unchanged": True}


def _save_snapshot(directory, binding, arms, arm_name, next_group):
    total = sum(arm["updates"] for arm in arms.values())
    path = directory / f"group-{total:04d}"
    path.mkdir(mode=0o700, exist_ok=False)
    cursor = {"arm": arm_name, "next_group": next_group, "total_committed_updates": total}
    state = {"schema": SNAPSHOT_SCHEMA, "binding": binding, "cursor": cursor,
             "arms": arms, "native_checkpoint": False, "native_v7_resumable": False,
             "qualification": "UNQUALIFIED", "acceptance_authority": False}
    state_path = path / "experimental-progress.pt"
    _write(state_path, state, tensor=True)
    digest = _hash(state_path)
    # A manifest is the commit record and is always published last.
    manifest = {"schema": SNAPSHOT_SCHEMA, "binding_sha256": sha256(canonical(binding)).hexdigest(),
                "cursor": cursor, "state_file_sha256": digest,
                "native_checkpoint": False, "native_v7_resumable": False,
                "qualification": "UNQUALIFIED", "acceptance_authority": False}
    _write(path / "manifest.json", manifest)
    _sync_directory(path)
    _sync_directory(directory)
    _require(_hash(state_path) == digest, "experimental snapshot readback differs")
    return {"path": str(path), "manifest_sha256": _hash(path / "manifest.json")}


def _check_arm(arm, arm_name, groups, model, objectives, optimizer, devices):
    _require(type(arm) is dict and set(arm) == _ARM_KEYS, "experimental arm schema differs")
    updates = arm["updates"]
    _require(type(updates) is int and 0 < updates <= len(groups), "experimental group cursor differs")
    events = arm["events"]
    _require(type(events) is list and len(events) == updates, "experimental event cursor differs")
    for index, (event, group) in enumerate(zip(events, groups), 1):
        _require(type(event) is dict and set(event) == {
            "arm", "update", "batch_sha256", "group_size", "endpoint_weight",
            "mean_endpoint_loss", "gradient_norm_before_clip", "endpoints"}, "experimental event schema differs")
        _require(event["arm"] == arm_name and type(event["update"]) is int and event["update"] == index
                 and event["batch_sha256"] == group and event["group_size"] == len(group)
                 and event["endpoint_weight"] == 1.0 / len(group), "experimental history is not the planned prefix")
        endpoints = event["endpoints"]
        _require(type(endpoints) is list and len(endpoints) == len(group), "experimental endpoints differ")
        for endpoint, digest in zip(endpoints, group):
            _require(type(endpoint) is dict and set(endpoint) == {"batch_sha256", "loss", "objectives", "coverage", "views"}
                     and endpoint["batch_sha256"] == digest and math.isfinite(endpoint["loss"])
                     and type(endpoint["objectives"]) is dict
                     and all(math.isfinite(value) for value in endpoint["objectives"].values()),
                     "experimental endpoint history differs")
            views = endpoint["views"]
            _require(type(views) is dict and set(views) == {"natural", "replay"}, "native view history differs")
            _require(all(type(v) is dict and set(v) == {"loss", "objectives", "coverage"}
                and math.isfinite(v["loss"]) and v["coverage"] == endpoint["coverage"]
                and set(v["objectives"]) == set(endpoint["objectives"])
                and all(math.isfinite(x) for x in v["objectives"].values()) for v in views.values()), "native view evidence differs")
            _require(endpoint["loss"] == math.fsum(v["loss"] for v in views.values()) / 2
                and all(endpoint["objectives"][key] == math.fsum(v["objectives"][key] for v in views.values()) / 2
                        for key in endpoint["objectives"]), "native view mean differs")
        _require(math.isfinite(event["gradient_norm_before_clip"]) and event["gradient_norm_before_clip"] >= 0
                 and event["mean_endpoint_loss"] == math.fsum(e["loss"] for e in endpoints) / len(group),
                 "experimental objective history differs")
    counts = Counter(digest for group in groups[:updates] for digest in group)
    _require(arm["batch_occurrences"] == dict(counts) and arm["endpoint_forwards"] == sum(counts.values())
             and arm["n1_n3_unchanged"] is True, "experimental occurrence accounting differs")
    for key, reference in (("model_state", model.state_dict()), ("objective_state", objectives.state_dict())):
        state = arm[key]
        _require(type(state) is dict and state.keys() == reference.keys(), "experimental tensor membership differs")
        for name, expected in reference.items():
            actual = state[name]
            _require(type(actual) is torch.Tensor and actual.shape == expected.shape and actual.dtype == expected.dtype
                     and actual.device.type == "cpu" and bool(torch.isfinite(actual).all()),
                     "experimental tensor geometry or values differ")
            if key == "objective_state" or name.startswith(("n1.", "n3.")):
                _require(torch.equal(actual, expected.detach().cpu()), "experimental frozen tensor changed")
    _require(fingerprint(arm["model_state"]) == arm["model_state_sha256"], "experimental learned-state hash differs")
    saved_optimizer = arm["optimizer_state"]
    expected_optimizer = optimizer.state_dict()
    _require(type(saved_optimizer) is dict and set(saved_optimizer) == {"state", "param_groups"}
             and saved_optimizer["param_groups"] == expected_optimizer["param_groups"],
             "experimental AdamW parameters or recipe differ")
    parameters = list(model.n2.parameters())
    ids = expected_optimizer["param_groups"][0]["params"]
    _require(type(saved_optimizer["state"]) is dict and saved_optimizer["state"]
             and all(type(key) is int and key in ids for key in saved_optimizer["state"]),
             "experimental AdamW state membership differs")
    for parameter_id, values in saved_optimizer["state"].items():
        _require(type(values) is dict and set(values) == {"step", "exp_avg", "exp_avg_sq"},
                 "experimental AdamW state schema differs")
        step = values["step"]
        _require(type(step) is torch.Tensor and step.numel() == 1 and bool(torch.isfinite(step).all())
                 and float(step) == int(step) and 0 < int(step) <= updates, "experimental AdamW step differs")
        parameter = parameters[ids.index(parameter_id)]
        for key in ("exp_avg", "exp_avg_sq"):
            value = values[key]
            _require(type(value) is torch.Tensor and value.shape == parameter.shape and value.dtype == parameter.dtype
                     and value.device.type == "cpu" and bool(torch.isfinite(value).all()),
                     "experimental AdamW moment differs")
        _require(bool((values["exp_avg_sq"] >= 0).all()), "experimental AdamW variance differs")
    _require(type(arm["cuda_rng"]) is list and len(arm["cuda_rng"]) == len(devices), "experimental CUDA RNG scope differs")
    for rng in (arm["torch_rng"], *arm["cuda_rng"]):
        _require(type(rng) is torch.Tensor and rng.dtype == torch.uint8 and rng.ndim == 1
                 and rng.device.type == "cpu", "experimental RNG state differs")
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(arm["torch_rng"])
        for device, rng in zip(devices, arm["cuda_rng"]):
            torch.cuda.set_rng_state(rng, device)


def _load_snapshot(path, expected_digest, binding, groups, model, objectives, recipe, devices):
    path = _plain_path(path)
    _require(type(expected_digest) is str and original._DIGEST.fullmatch(expected_digest),
             "explicit external snapshot manifest pin required")
    _require(path.is_dir() and {item.name for item in path.iterdir()} == {"manifest.json", "experimental-progress.pt"},
             "experimental snapshot is incomplete or has unexpected files")
    for name in ("manifest.json", "experimental-progress.pt"):
        item = _plain_path(path / name)
        _require(item.is_file(), "experimental snapshot member is not a regular file")
        if os.name == "posix":
            _require(stat.S_IMODE(item.stat().st_mode) == 0o600, "experimental snapshot permissions differ")
    manifest_path = path / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    _require(sha256(raw_manifest).hexdigest() == expected_digest, "external snapshot manifest changed")
    manifest = json.loads(raw_manifest)
    _require(type(manifest) is dict and set(manifest) == {
        "schema", "binding_sha256", "cursor", "state_file_sha256", "native_checkpoint",
        "native_v7_resumable", "qualification", "acceptance_authority"}
        and manifest["schema"] == SNAPSHOT_SCHEMA
        and manifest["binding_sha256"] == sha256(canonical(binding)).hexdigest(), "snapshot code, plan or runtime binding differs")
    state_path = path / "experimental-progress.pt"
    file_identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    before = file_identity(state_path.stat())
    _require(_hash(state_path) == manifest["state_file_sha256"], "snapshot tensor file changed")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    _require(file_identity(state_path.stat()) == before and _hash(state_path) == manifest["state_file_sha256"]
             and manifest_path.read_bytes() == raw_manifest, "snapshot changed during readback")
    _require(type(state) is dict and set(state) == {"schema", "binding", "cursor", "arms",
        "native_checkpoint", "native_v7_resumable", "qualification", "acceptance_authority"}
        and state["schema"] == SNAPSHOT_SCHEMA and state["binding"] == binding
        and state["cursor"] == manifest["cursor"], "snapshot binding or cursor differs")
    for value in (state, manifest):
        _require(value["native_checkpoint"] is False and value["native_v7_resumable"] is False
                 and value["qualification"] == "UNQUALIFIED" and value["acceptance_authority"] is False,
                 "experimental snapshot cannot grant native authority")
    cursor = state["cursor"]
    _require(type(cursor) is dict and set(cursor) == {"arm", "next_group", "total_committed_updates"}
             and cursor["arm"] in groups and type(cursor["next_group"]) is int
             and type(cursor["total_committed_updates"]) is int, "experimental cursor schema differs")
    order = list(groups)
    position = order.index(cursor["arm"])
    _require(type(state["arms"]) is dict and list(state["arms"]) == order[:position + 1],
             "experimental completed arm history differs")
    optimizer = torch.optim.AdamW(list(model.n2.parameters()), lr=recipe.learning_rate,
        betas=recipe.betas, eps=recipe.eps, weight_decay=recipe.weight_decay)
    for name, arm in state["arms"].items():
        _check_arm(arm, name, groups[name], model, objectives, optimizer, devices)
        _require(arm["updates"] == (cursor["next_group"] if name == cursor["arm"] else len(groups[name])),
                 "experimental arm and next-group cursors disagree")
    _require(cursor["total_committed_updates"] == sum(arm["updates"] for arm in state["arms"].values()),
             "experimental total group cursor differs")
    return state["arms"]


def _export(result, destination):
    destination = original._destination(destination)
    destination.mkdir(mode=0o700, exist_ok=False)
    state_path = destination / "experimental.pt"
    _write(state_path, result, tensor=True)
    manifest = {key: value for key, value in result.items() if key != "arms"}
    manifest["state_file_sha256"] = _hash(state_path)
    manifest["arms"] = {name: {key: value for key, value in arm.items()
        if key not in {"model_state", "objective_state", "optimizer_state", "torch_rng", "cuda_rng"}}
        for name, arm in result["arms"].items()}
    _write(destination / "manifest.json", manifest)
    _sync_directory(destination)
    _sync_directory(destination.parent)


def run_experiment(model, objectives, recipe, *, plan, expected_plan_sha256,
                   batch_loader, validate_batch, correction_loader, custody_guard, progress_directory,
                   resume_from=None, expected_resume_manifest_sha256=None,
                   destination=None, on_event=None):
    """Continue only an explicitly pinned experimental group-boundary snapshot.

    ``progress_directory`` must be new, including on recovery. ``on_event`` sees
    only newly committed groups, after durable publication. Original arm events
    and counters describe the full experiment; invocation counts are separate.
    The result's legacy ``resumable=False`` concerns native checkpoint admission;
    only the separately named experimental progress schema supports recovery.
    """
    _require(_hash(original.__file__) == ORIGINAL_HELPER_SHA256, "original scientific helper changed")
    helper_sha = _hash(__file__)
    teaching_sha = _hash(teaching.__file__)
    _require(all(callable(item) for item in (batch_loader, validate_batch, correction_loader, custody_guard))
             and (on_event is None or callable(on_event)), "explicit validation and custody callbacks required")
    _require(type(expected_plan_sha256) is str and original._DIGEST.fullmatch(expected_plan_sha256)
             and plan_sha256(plan) == expected_plan_sha256, "external experiment plan pin differs")
    _require((resume_from is None) == (expected_resume_manifest_sha256 is None), "snapshot and external pin must be supplied together")
    plan = deepcopy(plan)
    groups = _groups(plan)
    original._identity(model, objectives, recipe, plan)
    progress_directory = original._destination(progress_directory)
    if destination is not None:
        destination = original._destination(destination)
        _require(destination != progress_directory, "progress and final artifact directories must differ")
    custody_guard()
    materializations = 0
    for digest in plan["batch_sha256"]:
        batch = batch_loader(digest)
        materializations += 1
        original._check_batch(batch, digest, model.config, validate_batch)
        _correction(plan, batch, correction_loader)
        custody_guard()
        del batch

    def unchanged_parent():
        _require(fingerprint(model.state_dict()) == plan["model_state_sha256"]
                 and fingerprint(objectives.state_dict()) == plan["objective_state_sha256"],
                 "original model or objective tensors changed during experiment")
        _require(_hash(original.__file__) == ORIGINAL_HELPER_SHA256 and _hash(__file__) == helper_sha and _hash(teaching.__file__) == teaching_sha,
                 "experimental helper custody changed")

    device = next(model.parameters()).device
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    binding = {"plan": plan, "plan_sha256": expected_plan_sha256, "model_config": asdict(model.config),
               "recipe": asdict(recipe), "original_helper_sha256": ORIGINAL_HELPER_SHA256,
               "recovery_helper_sha256": helper_sha, "teaching_helper_sha256": teaching_sha, "runtime": _runtime(model, recipe),
               "optimizer_parameter_names": [name for name, _ in model.named_parameters() if name.startswith("n2.")]}
    # Canonical JSON roundtrip removes tuple/list ambiguity between manifest and tensor files.
    binding = json.loads(canonical(binding))
    arms = {} if resume_from is None else _load_snapshot(resume_from, expected_resume_manifest_sha256,
        binding, groups, model, objectives, recipe, devices)
    unchanged_parent()
    custody_guard()
    progress_directory.mkdir(mode=0o700, exist_ok=False)
    _sync_directory(progress_directory.parent)
    result = {"schema": EXPERIMENT_SCHEMA, "plan": plan, "plan_sha256": expected_plan_sha256,
              "model_config": asdict(model.config), "recipe": asdict(recipe), "helper_sha256": helper_sha,
              "original_helper_sha256": ORIGINAL_HELPER_SHA256,
              "native_checkpoint": False, "resumable": False, "qualification": "UNQUALIFIED",
              "acceptance_authority": False, "independent_qualification": False,
              "endpoint_reduction": "mean_of_complete_cases_each_half_natural_half_replay",
              "new_objective_terms": ["same_context_admitted_intervention_task_supervision"],
              "teaching_helper_sha256": teaching_sha, "native_forward_passes_per_endpoint": 2,
              "experimental_recovery_only": True, "experimental_progress_schema": SNAPSHOT_SCHEMA,
              "resumed_manifest_sha256": expected_resume_manifest_sha256,
              "new_updates_this_invocation": 0, "total_committed_updates": 0,
              "materializations_this_invocation": materializations, "latest_snapshot": None, "arms": arms}
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(recipe.deterministic)
        for arm_name, arm_groups in groups.items():
            saved = arms.get(arm_name)
            if saved is not None and saved["updates"] == len(arm_groups):
                continue
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(recipe.seed)
                clone, loss_module = deepcopy(model), deepcopy(objectives)
                clone.set_phase("n2", adapt_n1=False)
                loss_module.requires_grad_(False)
                loss_module.eval()
                parameters = list(clone.n2.parameters())
                optimizer = torch.optim.AdamW(parameters, lr=recipe.learning_rate, betas=recipe.betas,
                                              eps=recipe.eps, weight_decay=recipe.weight_decay)
                frozen = fingerprint({name: value for name, value in clone.state_dict().items()
                                      if name.startswith(("n1.", "n3."))})
                events, occurrences, start = [], Counter(), 0
                if saved is not None:
                    clone.load_state_dict(saved["model_state"], strict=True)
                    loss_module.load_state_dict(saved["objective_state"], strict=True)
                    optimizer.load_state_dict(saved["optimizer_state"])
                    events, occurrences, start = deepcopy(saved["events"]), Counter(saved["batch_occurrences"]), saved["updates"]
                    torch.set_rng_state(saved["torch_rng"])
                    for cuda_device, rng in zip(devices, saved["cuda_rng"]):
                        torch.cuda.set_rng_state(rng, cuda_device)
                for group_index in range(start, len(arm_groups)):
                    group = arm_groups[group_index]
                    custody_guard()
                    unchanged_parent()
                    _require(clone.phase == "n2" and all(parameter.requires_grad == name.startswith("n2.")
                             for name, parameter in clone.named_parameters()), "experimental trainable scope changed")
                    optimizer.zero_grad(set_to_none=True)
                    endpoints = []
                    for digest in group:
                        batch = batch_loader(digest)
                        result["materializations_this_invocation"] += 1
                        original._check_batch(batch, digest, clone.config, validate_batch)
                        custody_guard()
                        record, record_sha = _correction(plan, batch, correction_loader)
                        views = {}
                        for view in ("natural", "replay"):
                            if view == "replay" and record is not None:
                                packet = teaching.forward_with_correction(clone, batch, record,
                                    expected_record_sha256=record_sha, apply_correction=arm_name == "intervention")
                            else:
                                packet = clone(batch.frame)
                            total, losses, coverage = loss_module(packet, batch, loss_weights=recipe.loss_weights)
                            _require(bool(torch.isfinite(total)) and total.requires_grad,
                                     "native view objective must be finite with a trainable N2 path")
                            (total / (2 * len(group))).backward()
                            views[view] = {"loss": float(total.detach()),
                                "objectives": {key: float(value.detach()) for key, value in losses.items()},
                                "coverage": coverage}
                            del packet, total, losses
                        _require(views["natural"]["coverage"] == views["replay"]["coverage"]
                            and views["natural"]["objectives"].keys() == views["replay"]["objectives"].keys(),
                            "correction changed native objective coverage")
                        original._check_batch(batch, digest, clone.config, validate_batch)
                        _correction(plan, batch, correction_loader)
                        custody_guard()
                        endpoints.append({"batch_sha256": digest,
                            "loss": math.fsum(v["loss"] for v in views.values()) / 2,
                            "objectives": {key: math.fsum(v["objectives"][key] for v in views.values()) / 2
                                           for key in views["natural"]["objectives"]},
                            "coverage": views["natural"]["coverage"], "views": views})
                        occurrences[digest] += 1
                        del coverage, batch, record, views
                    _require(any(parameter.grad is not None for parameter in parameters), "group produced no native N2 gradient")
                    for name, parameter in clone.named_parameters():
                        _require(parameter.grad is None or (name.startswith("n2.") and bool(torch.isfinite(parameter.grad).all())),
                                 "group gradient reached a frozen stage or became nonfinite")
                    _require(all(parameter.grad is None for parameter in loss_module.parameters()),
                             "group gradient reached frozen auxiliary objectives")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm, error_if_nonfinite=True)
                    custody_guard()
                    unchanged_parent()
                    optimizer.step()
                    _require(all(bool(torch.isfinite(parameter).all()) for parameter in clone.parameters()),
                             "experimental update produced nonfinite weights")
                    _require(fingerprint({name: value for name, value in clone.state_dict().items()
                                          if name.startswith(("n1.", "n3."))}) == frozen, "experimental update changed N1 or N3")
                    _require(fingerprint(loss_module.state_dict()) == plan["objective_state_sha256"],
                             "experimental update changed auxiliary objectives")
                    event = {"arm": arm_name, "update": group_index + 1, "batch_sha256": list(group),
                             "group_size": len(group), "endpoint_weight": 1.0 / len(group),
                             "mean_endpoint_loss": math.fsum(item["loss"] for item in endpoints) / len(group),
                             "gradient_norm_before_clip": float(norm), "endpoints": endpoints}
                    events.append(event)
                    unchanged_parent()
                    custody_guard()
                    arms[arm_name] = _arm_state(clone, loss_module, optimizer, events, occurrences, devices)
                    result["latest_snapshot"] = _save_snapshot(progress_directory, binding, arms, arm_name, group_index + 1)
                    result["new_updates_this_invocation"] += 1
                    if on_event is not None:
                        on_event(deepcopy(event))
                _require(occurrences == Counter(plan["batch_sha256"]), "completed arm endpoint accounting differs")
                unchanged_parent()
                custody_guard()
                del clone, loss_module, optimizer, parameters
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
    unchanged_parent()
    custody_guard()
    result["total_committed_updates"] = sum(arm["updates"] for arm in arms.values())
    _require(set(arms) == set(groups) and all(arms[name]["updates"] == len(groups[name]) for name in groups),
             "experimental trial is incomplete")
    if destination is not None:
        _export(result, destination)
    return result
