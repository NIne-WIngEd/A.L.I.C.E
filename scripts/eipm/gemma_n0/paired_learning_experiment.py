"""Bounded N2 update-composition experiment, never a native checkpoint.

The caller owns source/target admission and semantic pairing. Pointer equality
does not establish a semantic pair. Its externally pinned plan and callbacks
must verify actual reviewed documents, features, targets and complete custody.
This helper adds no targets, directional objective, calibration or promotion.

Each arm starts from a separate copy of the same complete native model. A group
accumulates the mean of its complete endpoint objectives, then clips once and
takes one AdamW step. The control permutes one declared group position, keeping
every endpoint, group size, optimizer setting and update count identical.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import math
from pathlib import Path
import re

import torch

from src.alice_personality.identity import IdentityModel
from src.alice_personality.identity.codec import canonical, fingerprint
from src.alice_personality.teaching import IdentityObjectives, TeachingBatch, TeachingError, TrainingRecipe

PLAN_SCHEMA = "alice-personality-paired-group-plan-v1"
EXPERIMENT_SCHEMA = "alice-personality-paired-group-experiment-v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_KEYS = {
    "schema", "model_state_sha256", "model_config_sha256", "objective_state_sha256",
    "recipe_sha256", "batch_sha256", "matched_groups", "control_permutation",
    "controlled_slot", "bindings",
}


def _require(condition, message):
    if not condition:
        raise TeachingError(message)


def plan_sha256(plan):
    """Content binding only; a self-digest never supplies admission authority."""
    return sha256(canonical(plan)).hexdigest()


def _groups(plan):
    _require(type(plan) is dict and set(plan) == _PLAN_KEYS
             and plan["schema"] == PLAN_SCHEMA, "experimental plan schema differs")
    _require(type(plan["bindings"]) is dict, "plan requires explicit caller custody bindings")
    inventory, matched = plan["batch_sha256"], plan["matched_groups"]
    _require(type(inventory) is list and inventory
             and all(type(d) is str and _DIGEST.fullmatch(d) for d in inventory)
             and len(set(inventory)) == len(inventory), "complete unique batch inventory required")
    _require(type(matched) is list and len(matched) >= 2
             and all(type(group) is list and len(group) >= 2 for group in matched),
             "at least two complete endpoint groups required")
    flattened = [digest for group in matched for digest in group]
    _require(all(type(d) is str and _DIGEST.fullmatch(d) for d in flattened)
             and Counter(flattened) == Counter(inventory),
             "groups must contain every admitted batch exactly once")
    permutation, slot = plan["control_permutation"], plan["controlled_slot"]
    _require(type(slot) is int and slot >= 0 and all(slot < len(g) for g in matched),
             "controlled slot must address every group")
    _require(type(permutation) is list and all(type(i) is int for i in permutation)
             and sorted(permutation) == list(range(len(matched)))
             and all(i != donor for i, donor in enumerate(permutation)),
             "control must independently permute every declared pair member")
    _require(all(len(matched[index]) == len(matched[donor])
                 for index, donor in enumerate(permutation)),
             "control must preserve each endpoint's mean-objective weight")
    control = [list(group) for group in matched]
    for index, donor in enumerate(permutation):
        control[index][slot] = matched[donor][slot]
    _require(Counter(d for group in control for d in group) == Counter(inventory)
             and list(map(len, control)) == list(map(len, matched)),
             "control differs in endpoint membership or group sizes")
    return {"matched": matched, "control": control}


def _identity(model, objectives, recipe, plan):
    _require(type(model) is IdentityModel and type(objectives) is IdentityObjectives
             and type(recipe) is TrainingRecipe, "exact native model, objectives and recipe required")
    recipe.validate()
    _require(recipe.phase == "n2" and recipe.adapt_n1 is False,
             "experiment updates only N2 without calibration or N1 adaptation")
    _require(model.phase == "inference" and not model.training
             and not any(p.requires_grad for p in model.parameters()),
             "experimental parent must be frozen native inference state")
    for actual, key in (
        (fingerprint(model.state_dict()), "model_state_sha256"),
        (fingerprint(model.config), "model_config_sha256"),
        (fingerprint(objectives.state_dict()), "objective_state_sha256"),
        (recipe.fingerprint(), "recipe_sha256"),
    ):
        _require(plan[key] == actual, f"bound {key} differs")


def make_plan(model, objectives, recipe, *, batch_sha256, matched_groups,
              control_permutation, controlled_slot=1, bindings):
    """Build a reviewable candidate plan; caller must independently pin it."""
    plan = {
        "schema": PLAN_SCHEMA,
        "model_state_sha256": fingerprint(model.state_dict()),
        "model_config_sha256": fingerprint(model.config),
        "objective_state_sha256": fingerprint(objectives.state_dict()),
        "recipe_sha256": recipe.fingerprint(),
        "batch_sha256": list(batch_sha256),
        "matched_groups": [list(group) for group in matched_groups],
        "control_permutation": list(control_permutation),
        "controlled_slot": controlled_slot,
        "bindings": deepcopy(bindings),
    }
    _groups(plan)
    _identity(model, objectives, recipe, plan)
    plan_sha256(plan)
    return plan


def _banks(batch):
    frame = batch.frame
    return (frame.query, frame.sources, frame.concepts, frame.candidates,
            frame.relations, *frame.labels.values())


def _check_batch(batch, digest, config, validate_batch):
    _require(type(batch) is TeachingBatch, "loader must return a native TeachingBatch")
    batch.validate(config)
    _require(batch.phase == "n2" and batch.split == "TRAIN",
             "only admitted N2 TRAIN endpoints may update the experiment")
    _require(not batch.contrasts,
             "preexisting contrast objectives require a separately specified experimental protocol")
    _require(batch.fingerprint() == digest and validate_batch(batch) == digest,
             "endpoint differs from its exact external admission")
    _require(all(not bank.states.is_leaf or bank.states.grad is None for bank in _banks(batch)),
             "provider feature gradients must remain absent")


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def _destination(path):
    destination = Path(path)
    _require(destination.is_absolute() and not destination.exists() and destination.parent.is_dir(),
             "experiment output needs a new absolute directory with an existing parent")
    _require(not any(item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction())
                     for item in (destination, *destination.parents)),
             "experiment output cannot traverse links")
    return destination


def run_experiment(model, objectives, recipe, *, plan, expected_plan_sha256,
                   batch_loader, validate_batch, custody_guard,
                   destination=None, on_event=None):
    """Run exactly one planned pass per arm; fail closed before each update.

    ``validate_batch(batch)`` must perform the caller's native admission check
    and return the exact batch digest. ``custody_guard()`` must raise on changed
    source/checkpoint/plan custody. Neither callback grants authority itself.
    The result contains complete experimental states for separate assessment;
    it cannot be resumed or promoted through native v7 checkpoint machinery.
    """
    _require(all(callable(item) for item in (batch_loader, validate_batch, custody_guard))
             and (on_event is None or callable(on_event)), "explicit validation and custody callbacks required")
    _require(type(expected_plan_sha256) is str and _DIGEST.fullmatch(expected_plan_sha256)
             and plan_sha256(plan) == expected_plan_sha256, "external experiment plan pin differs")
    plan = deepcopy(plan)
    groups = _groups(plan)
    _identity(model, objectives, recipe, plan)
    destination = _destination(destination) if destination is not None else None
    custody_guard()
    # Validate all curriculum members before the first gradient, streaming the
    # large frozen feature frames rather than keeping the entire corpus live.
    for digest in plan["batch_sha256"]:
        batch = batch_loader(digest)
        _check_batch(batch, digest, model.config, validate_batch)
        custody_guard()
        del batch

    def unchanged_parent():
        _require(fingerprint(model.state_dict()) == plan["model_state_sha256"]
                 and fingerprint(objectives.state_dict()) == plan["objective_state_sha256"],
                 "original model or objective tensors changed during experiment")

    result = {
        "schema": EXPERIMENT_SCHEMA, "plan": plan, "plan_sha256": expected_plan_sha256,
        "model_config": asdict(model.config), "recipe": asdict(recipe),
        "helper_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_checkpoint": False, "resumable": False, "qualification": "UNQUALIFIED",
        "acceptance_authority": False, "independent_qualification": False,
        "endpoint_reduction": "mean_of_complete_native_case_objectives",
        "new_objective_terms": [], "arms": {},
    }
    device = next(model.parameters()).device
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(recipe.deterministic)
        for arm_name, arm_groups in groups.items():
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(recipe.seed)
                clone, loss_module = deepcopy(model), deepcopy(objectives)
                clone.set_phase("n2", adapt_n1=False)
                loss_module.requires_grad_(False)
                loss_module.eval()
                parameters = list(clone.n2.parameters())
                optimizer = torch.optim.AdamW(
                    parameters, lr=recipe.learning_rate, betas=recipe.betas,
                    eps=recipe.eps, weight_decay=recipe.weight_decay)
                frozen = fingerprint({name: value for name, value in clone.state_dict().items()
                                      if name.startswith(("n1.", "n3."))})
                events, occurrences = [], Counter()
                for group_index, group in enumerate(arm_groups):
                    custody_guard()
                    unchanged_parent()
                    _require(clone.phase == "n2" and all(
                        parameter.requires_grad == name.startswith("n2.")
                        for name, parameter in clone.named_parameters()),
                        "experimental trainable scope changed")
                    optimizer.zero_grad(set_to_none=True)
                    endpoints = []
                    for digest in group:
                        batch = batch_loader(digest)
                        _check_batch(batch, digest, clone.config, validate_batch)
                        custody_guard()
                        packet = clone(batch.frame)
                        total, losses, coverage = loss_module(
                            packet, batch, loss_weights=recipe.loss_weights)
                        _require(bool(torch.isfinite(total)) and total.requires_grad,
                                 "native endpoint objective must be finite with a trainable N2 path")
                        (total / len(group)).backward()
                        _check_batch(batch, digest, clone.config, validate_batch)
                        custody_guard()
                        endpoints.append({
                            "batch_sha256": digest, "loss": float(total.detach()),
                            "objectives": {key: float(value.detach()) for key, value in losses.items()},
                            "coverage": coverage,
                        })
                        occurrences[digest] += 1
                        del packet, total, losses, coverage, batch
                    _require(any(parameter.grad is not None for parameter in parameters),
                             "group produced no native N2 gradient")
                    for name, parameter in clone.named_parameters():
                        _require(parameter.grad is None or (name.startswith("n2.")
                                 and bool(torch.isfinite(parameter.grad).all())),
                                 "group gradient reached a frozen stage or became nonfinite")
                    _require(all(parameter.grad is None for parameter in loss_module.parameters()),
                             "group gradient reached frozen auxiliary objectives")
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, recipe.max_grad_norm, error_if_nonfinite=True)
                    custody_guard()
                    unchanged_parent()
                    optimizer.step()
                    _require(all(bool(torch.isfinite(p).all()) for p in clone.parameters()),
                             "experimental update produced nonfinite weights")
                    _require(fingerprint({name: value for name, value in clone.state_dict().items()
                                          if name.startswith(("n1.", "n3."))}) == frozen,
                             "experimental update changed N1 or N3")
                    _require(fingerprint(loss_module.state_dict()) == plan["objective_state_sha256"],
                             "experimental update changed auxiliary objectives")
                    event = {
                        "arm": arm_name, "update": group_index + 1,
                        "batch_sha256": list(group), "group_size": len(group),
                        "endpoint_weight": 1.0 / len(group),
                        "mean_endpoint_loss": math.fsum(item["loss"] for item in endpoints) / len(group),
                        "gradient_norm_before_clip": float(norm), "endpoints": endpoints,
                    }
                    events.append(event)
                    if on_event is not None:
                        on_event(deepcopy(event))
                _require(occurrences == Counter(plan["batch_sha256"]),
                         "completed arm endpoint accounting differs")
                unchanged_parent()
                custody_guard()
                result["arms"][arm_name] = {
                    "model_state": _cpu(clone.state_dict()),
                    "objective_state": _cpu(loss_module.state_dict()),
                    "optimizer_state": _cpu(optimizer.state_dict()),
                    "model_state_sha256": fingerprint(clone.state_dict()),
                    "torch_rng": torch.get_rng_state().clone(),
                    "cuda_rng": [torch.cuda.get_rng_state(d).cpu() for d in devices],
                    "events": events, "updates": len(events),
                    "endpoint_forwards": sum(occurrences.values()),
                    "batch_occurrences": dict(occurrences),
                    "n1_n3_unchanged": True,
                }
                del clone, loss_module, optimizer, parameters
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
    unchanged_parent()
    custody_guard()
    if destination is not None:
        destination = _destination(destination)
        destination.mkdir(mode=0o700, exist_ok=False)
        state_path = destination / "experimental.pt"
        with state_path.open("xb") as stream:
            torch.save(result, stream)
        digest = sha256()
        with state_path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        manifest = {key: value for key, value in result.items() if key != "arms"}
        manifest["state_file_sha256"] = digest.hexdigest()
        manifest["arms"] = {
            name: {key: value for key, value in arm.items()
                   if key not in {"model_state", "objective_state", "optimizer_state", "torch_rng", "cuda_rng"}}
            for name, arm in result["arms"].items()
        }
        with (destination / "manifest.json").open("xb") as stream:
            stream.write(canonical(manifest) + b"\n")
    return result
