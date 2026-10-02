"""Stage-isolated optimization with externally pinned current authorization.

Compiler eligibility, package custody and teaching-target review are distinct.
This module writes honest unqualified optimization evidence, never acceptance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import platform
from typing import Mapping

import numpy as np
import torch

from ..identity.contracts import CalibrationBatch, FRAME_SCHEMA, PACKET_SCHEMA
from ..identity.model import IdentityModel
from .contracts import (AUTH_SCHEMA, FAMILY_SCHEMA, REVIEW_SCHEMA, TARGET_SCHEMA, TeachingBatch,
                        TeachingError, TrainingRecipe, canonical, fingerprint)
from .objectives import IdentityObjectives

CHECKPOINT_SCHEMA = "alice-personality-resumable-teaching-checkpoint-v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_ROLES = {"source_package", "compiled_receipt", "prepared_receipt", "reviewed_targets",
                   "source_families", "authorization"}
_GOVERNED_ROLES = _ARTIFACT_ROLES | {"n0_qualification"}


def _digest(value, label):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise TeachingError(f"{label} requires an exact lowercase SHA256")
    return value


def _path(value, *, directory=False):
    if not isinstance(value, (str, Path)):
        raise TeachingError("artifact binding requires a path")
    path = Path(value).expanduser()
    if not path.is_absolute() or any(item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction())
                                     for item in (path, *path.parents)):
        raise TeachingError("artifact paths must be absolute and must not traverse links")
    if directory:
        if not path.is_dir():
            raise TeachingError("checkpoint directory is missing")
    elif not path.is_file():
        raise TeachingError("pinned artifact file is missing")
    return path.resolve(strict=True)


def _hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _json(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise TeachingError("admission JSON contains duplicate fields")
            output[key] = value
        return output
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError) as exc:
        raise TeachingError("admission artifact has invalid JSON") from exc
    if not isinstance(value, dict):
        raise TeachingError("admission metadata must be an object")
    return value


def current_code_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)).replace("\\", "/"): _hash(_path(path))
            for folder in ("identity", "teaching") for path in sorted((root / folder).glob("*.py"))}


def model_fingerprint(model: IdentityModel) -> str:
    return fingerprint(model.state_dict())


def recipe_fingerprint(recipe: TrainingRecipe) -> str:
    return recipe.fingerprint()


def _production(frame, binding, session, *, prepared_file_sha256, prepared_receipt_sha256, implementation):
    """Validate a reviewed frame's exact closed feature-production lineage.

    These records stay unqualified. Their complete bytes and the feature frame
    are externally admitted as part of the reviewed teaching-batch fingerprint;
    a self-digest by itself never grants private execution or gradient authority.
    """
    if not isinstance(binding, Mapping) or not isinstance(session, Mapping):
        raise TeachingError("governed frames require their actual binding and closed production session")
    for value, key in ((binding, "binding_sha256"), (session, "receipt_sha256")):
        digest = _digest(value.get(key), "production metadata")
        body = {name: item for name, item in value.items() if name != key}
        if sha256(canonical(body)).hexdigest() != digest:
            raise TeachingError("production binding/session digest changed")
    if (binding.get("schema") != "alice-personality-produced-frame-binding-v1"
            or binding.get("frame_sha256") != fingerprint(frame)
            or binding.get("source_geometry") != [frame.query.states.shape[2], frame.query.states.shape[-1]]
            or binding.get("session_source_recheck_pending") is not True
            or session.get("schema") != "alice-personality-feature-session-v1"
            or session.get("status") != "CLOSED_SOURCE_VERIFIED"
            or session.get("session_source_recheck_passed") is not True
            or session.get("mechanical_fixture_only") is not False
            or session.get("prepared_receipt_sha256") != prepared_receipt_sha256):
        raise TeachingError("exact frame provenance needs a genuinely closed source-verified session")
    for value in (binding, session):
        if (value.get("source_class") != "prepared_frozen_gemma_features_unqualified"
                or value.get("prepared_receipt_file_sha256") != prepared_file_sha256
                or value.get("implementation_sha256") != implementation
                or value.get("qualification") != "UNQUALIFIED"
                or value.get("source_acceptance_authority") is not False
                or value.get("persistent_feature_values_retained") is not False):
            raise TeachingError("production custody/source/code binding or unqualified authority differs")
    if any(binding.get(name) is not False for name in ("generation", "truncation", "chat_template")):
        raise TeachingError("production cannot derive frame features through generation or truncation")
    _digest(binding.get("document_sha256"), "produced document")
    members = session.get("frame_binding_sha256")
    if not isinstance(members, list) or not members or any(not isinstance(item, str) or not _DIGEST.fullmatch(item) for item in members) \
            or binding["binding_sha256"] not in members:
        raise TeachingError("produced frame is absent from the closed session inventory")


@dataclass(frozen=True)
class ArtifactPin:
    path: str | Path
    sha256: str

    def verify(self):
        path = _path(self.path)
        if _hash(path) != _digest(self.sha256, "artifact file pin"):
            raise TeachingError("current artifact file differs from its external SHA pin")
        return path


@dataclass(frozen=True)
class TeachingAdmission:
    artifacts: Mapping[str, ArtifactPin]
    model_config_sha256: str
    model_initial_state_sha256: str
    recipe_sha256: str
    code_sha256: Mapping[str, str]

    def binding(self):
        return {"artifacts": {role: {"path": str(_path(pin.path)), "file_sha256": pin.sha256}
                              for role, pin in sorted(self.artifacts.items())},
                "model_config_sha256": self.model_config_sha256,
                "model_initial_state_sha256": self.model_initial_state_sha256,
                "recipe_sha256": self.recipe_sha256, "code_sha256": dict(self.code_sha256),
                "frame_schema": FRAME_SCHEMA, "packet_schema": PACKET_SCHEMA,
                "target_schema": TARGET_SCHEMA, "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__), "python_version": platform.python_version(),
                "cuda_runtime": torch.version.cuda}

    def verify(self, model, recipe, *, initial=False, deep_custody=True):
        recipe.validate()
        if not isinstance(self.artifacts, Mapping) or set(self.artifacts) not in (_ARTIFACT_ROLES, _GOVERNED_ROLES) \
                or any(not isinstance(value, ArtifactPin) for value in self.artifacts.values()):
            raise TeachingError("teaching needs all six explicit pins plus personality N0 qualification for governed training")
        paths = {role: pin.verify() for role, pin in self.artifacts.items()}
        config_hash = sha256(canonical(asdict(model.config))).hexdigest()
        if config_hash != _digest(self.model_config_sha256, "model config") \
                or recipe.fingerprint() != _digest(self.recipe_sha256, "recipe") \
                or current_code_hashes() != self.code_sha256:
            raise TeachingError("model configuration, recipe or firstparty code differs from its external binding")
        _digest(self.model_initial_state_sha256, "initial model state")
        if initial and model_fingerprint(model) != self.model_initial_state_sha256:
            raise TeachingError("initial learned model state differs from the external model pin")
        reviewed, families, auth = (_json(paths[name]) for name in
                                    ("reviewed_targets", "source_families", "authorization"))
        if reviewed.get("schema") != REVIEW_SCHEMA or reviewed.get("target_schema") != TARGET_SCHEMA \
                or reviewed.get("reviewed") is not True:
            raise TeachingError("compiler rows are not a reviewed teaching target receipt")
        if families.get("schema") != FAMILY_SCHEMA or not isinstance(families.get("family_splits"), dict):
            raise TeachingError("source-family splits require their explicit bound schema")
        if any(not isinstance(key, str) or not key or value not in
               {"TRAIN", "DEV", "CALIBRATION_TRAIN", "CALIBRATION_DEV", "FINAL"}
               for key, value in families["family_splits"].items()):
            raise TeachingError("source-family split declarations are malformed")
        if auth.get("schema") != AUTH_SCHEMA or not isinstance(auth.get("authorization_id"), str) \
                or not auth["authorization_id"] or auth.get("phase") != recipe.phase:
            raise TeachingError("a separately bound current phase authorization is required")
        source_class = reviewed.get("source_class")
        if auth.get("source_class") != source_class:
            raise TeachingError("authorization and reviewed target source classes disagree")
        if type(auth.get("private_gradient_authorized")) is not bool \
                or type(auth.get("mechanical_steps_authorized")) is not bool:
            raise TeachingError("current authorization flags must be actual Booleans")
        if source_class == "public_mechanical_fixture":
            if set(self.artifacts) != _ARTIFACT_ROLES:
                raise TeachingError("public mechanics cannot consume a private-training qualification grant")
            if auth["private_gradient_authorized"] or not auth["mechanical_steps_authorized"]:
                raise TeachingError("public fixture authorization cannot grant private identity gradients")
            compiled, prepared = _json(paths["compiled_receipt"]), _json(paths["prepared_receipt"])
            if compiled.get("schema") != "public-mechanical-source-receipt-v1" \
                    or prepared.get("schema") != "public-mechanical-feature-receipt-v1":
                raise TeachingError("mechanical fixture artifacts must identify their limited evidence")
        else:
            expected_class = "governed_calibration_targets" if recipe.phase == "calibration" else "governed_identity_targets"
            if source_class != expected_class or not auth["private_gradient_authorized"]:
                raise TeachingError("governed identity gradients need current explicit private authorization")
            if set(self.artifacts) != _GOVERNED_ROLES:
                raise TeachingError("governed gradients require externally pinned QUALIFIED_PERSONALITY_N0 evidence first")
            # Custody is independently rechecked at admission/resume; source
            # compiler eligibility is never read as the authorization above.
            from ..n1.compiler import verify_compiled
            from ..gemma_n0.preparation import verify_prepared
            compiled = verify_compiled(paths["compiled_receipt"].parent) if deep_custody else _json(paths["compiled_receipt"])
            prepared = verify_prepared(paths["prepared_receipt"]) if deep_custody else _json(paths["prepared_receipt"])
            if compiled["source_package_sha256"] != self.artifacts["source_package"].sha256:
                raise TeachingError("compiled archive differs from the externally pinned source package")
            if deep_custody:
                try:
                    from ..gemma_n0.qualification import verify_personality_n0_qualification
                except ImportError as exc:
                    raise TeachingError("personality N0 qualification validator is not available; governed gradients remain blocked") from exc
                qualification = verify_personality_n0_qualification(paths["n0_qualification"],
                    expected_file_sha256=self.artifacts["n0_qualification"].sha256,
                    expected_prepared_file_sha256=self.artifacts["prepared_receipt"].sha256,
                    expected_prepared_receipt_sha256=prepared["receipt_sha256"])
            else:
                qualification = _json(paths["n0_qualification"])
            if (not isinstance(qualification, dict)
                    or qualification.get("schema") != "alice-personality-n0-role-qualification-v1"
                    or qualification.get("state") != "QUALIFIED_PERSONALITY_N0"
                    or qualification.get("role") != "personality"
                    or qualification.get("prepared_receipt_file_sha256") != self.artifacts["prepared_receipt"].sha256
                    or qualification.get("prepared_receipt_sha256") != prepared["receipt_sha256"]
                    or qualification.get("repository") != prepared["repository"]
                    or qualification.get("revision") != prepared["revision"]):
                raise TeachingError("personality N0 role qualification differs from the exact prepared source")
        geometry = prepared.get("model_geometry")
        if not isinstance(geometry, dict) or (geometry.get("hidden_size"), geometry.get("hidden_state_count")) \
                != (model.config.provider_width, model.config.provider_state_count):
            raise TeachingError("model provider geometry differs from the bound prepared feature artifact")
        expected_artifacts = {role: pin.sha256 for role, pin in self.artifacts.items() if role != "authorization"}
        if auth.get("artifact_file_sha256") != expected_artifacts \
                or auth.get("model_config_sha256") != self.model_config_sha256 \
                or auth.get("model_initial_state_sha256") != self.model_initial_state_sha256 \
                or auth.get("recipe_sha256") != self.recipe_sha256 or auth.get("code_sha256") != self.code_sha256:
            raise TeachingError("current authorization must bind the actual source, features, targets, model, recipe and code")
        if not isinstance(reviewed.get("reviewed_batches"), dict) or not isinstance(reviewed.get("batch_order"), dict):
            raise TeachingError("reviewed targets require exact batch fingerprints and explicit replay order")
        return reviewed, families, auth


class IdentityTrainer:
    """Actual stage-only AdamW, explicit data order and resumable state."""
    def __init__(self, model: IdentityModel, recipe: TrainingRecipe, admission: TeachingAdmission,
                 *, first_batch: TeachingBatch):
        if not isinstance(model, IdentityModel):
            raise TeachingError("trainer accepts only the implemented firstparty identity model")
        self.model, self.recipe, self.admission = model, recipe, admission
        self.reviewed, self.families, self.authorization = admission.verify(model, recipe, initial=True)
        self._binding = admission.binding()
        self._check_batch(first_batch)
        random.seed(recipe.seed)
        np.random.seed(recipe.seed % (2 ** 32))
        torch.manual_seed(recipe.seed)
        if next(model.parameters()).device.type == "cuda":
            if recipe.deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
                raise TeachingError("deterministic CUDA training needs its configured cuBLAS workspace")
            torch.cuda.manual_seed_all(recipe.seed)
        torch.use_deterministic_algorithms(recipe.deterministic)
        device = next(model.parameters()).device
        self.objectives = IdentityObjectives(model.config.learned_width).to(device=device)
        self._parameter_geometry = {name: (tuple(parameter.shape), str(parameter.dtype), str(parameter.device))
                                   for name, parameter in self._all_named_parameters()}
        self.steps, self.epoch, self.batch_index = 0, 0, 0
        self.failed = False
        self.events = []
        self._phase(first_batch)
        parameters = self._expected_parameters()
        self.optimizer = torch.optim.AdamW([parameter for _, parameter in parameters],
            lr=recipe.learning_rate, betas=recipe.betas, eps=recipe.eps, weight_decay=recipe.weight_decay)

    def _check_batch(self, batch):
        if not isinstance(batch, TeachingBatch):
            raise TeachingError("source compiler records cannot substitute for a TeachingBatch")
        batch.validate(self.model.config)
        if batch.phase != self.recipe.phase or batch.source_class != self.reviewed["source_class"]:
            raise TeachingError("batch phase/source differs from its current authorization")
        digest = batch.fingerprint()
        declared = self.reviewed["reviewed_batches"].get(digest)
        expected = {"split": batch.split, "case_ids": list(batch.case_ids),
                    "source_family_ids": [list(row) for row in batch.source_family_ids]}
        if declared != expected:
            raise TeachingError("teaching batch lacks exact externally reviewed facts/features/pointers")
        if batch.source_class != "public_mechanical_fixture":
            prepared_pin = self.admission.artifacts["prepared_receipt"]
            prepared = _json(_path(prepared_pin.path))
            code = {name: self.admission.code_sha256["identity/" + name]
                    for name in ("codec.py", "feature_producer.py")}
            arguments = {"prepared_file_sha256": prepared_pin.sha256,
                         "prepared_receipt_sha256": prepared.get("receipt_sha256"), "implementation": code}
            _production(batch.frame, batch.production_binding, batch.production_session, **arguments)
            for contrast in batch.contrasts:
                _production(contrast.other_frame, contrast.production_binding, contrast.production_session, **arguments)
        for row in batch.source_family_ids:
            if any(self.families["family_splits"].get(family) != batch.split for family in row):
                raise TeachingError("shared source family crossed its held-out teaching split")
        for contrast in batch.contrasts:
            contrast.other_frame.validate(self.model.config)
            other_families = {source.source_family_id for row in contrast.other_frame.source_records
                              for source in row if source is not None}
            if any(self.families["family_splits"].get(family) != batch.split for family in other_families):
                raise TeachingError("contrast frame crosses source-family holdouts")
        return digest

    def _phase(self, batch):
        if self.recipe.phase == "calibration":
            batch_data = CalibrationBatch(batch.frame, {name: target.values for name, target in batch.targets.items()},
                torch.stack([target.available for target in batch.targets.values()]).any(0), batch.source_class,
                fingerprint(batch.source_family_ids), batch.fingerprint(), target_masks={
                    name: target.available for name, target in batch.targets.items()})
            self.model.set_phase("calibration", calibration_batch=batch_data)
        else:
            self.model.set_phase(self.recipe.phase, adapt_n1=self.recipe.adapt_n1)
        self.objectives.requires_grad_(self.recipe.phase == "n1")
        self.objectives.train(self.recipe.phase == "n1")

    def _expected_parameters(self):
        prefixes = ("n1.",) if self.recipe.phase == "n1" else ("n2.", "n1.") \
            if self.recipe.phase == "n2" and self.recipe.adapt_n1 else ("n2.",) \
            if self.recipe.phase == "n2" else ("n3.",)
        selected = [("model." + name, parameter) for name, parameter in self.model.named_parameters()
                    if name.startswith(prefixes)]
        if self.recipe.phase == "n1":
            selected += [("teaching." + name, parameter) for name, parameter in self.objectives.named_parameters()]
        return selected

    def _all_named_parameters(self):
        return [("model." + name, parameter) for name, parameter in self.model.named_parameters()] + \
               [("teaching." + name, parameter) for name, parameter in self.objectives.named_parameters()]

    def _check_optimizer(self):
        expected = self._expected_parameters()
        actual = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        if len(actual) != len(expected) or len({id(value) for value in actual}) != len(actual) \
                or {id(value) for value in actual} != {id(value) for _, value in expected}:
            raise TeachingError("optimizer must contain exact firstparty stage parameters only")
        allowed = {id(value) for _, value in expected}
        all_parameters = [*self.model.parameters(), *self.objectives.parameters()]
        if any(parameter.requires_grad != (id(parameter) in allowed) for parameter in all_parameters):
            raise TeachingError("trainable stage parameter flags changed")
        if self._parameter_geometry != {name: (tuple(parameter.shape), str(parameter.dtype), str(parameter.device))
                                        for name, parameter in self._all_named_parameters()}:
            raise TeachingError("learned parameter geometry/dtype/device changed")
        if torch.are_deterministic_algorithms_enabled() != self.recipe.deterministic:
            raise TeachingError("runtime deterministic algorithm setting changed")
        for group in self.optimizer.param_groups:
            if (group["lr"] != self.recipe.learning_rate or tuple(group["betas"]) != self.recipe.betas
                    or group["eps"] != self.recipe.eps or group["weight_decay"] != self.recipe.weight_decay
                    or group.get("amsgrad") is not False or group.get("maximize") is not False):
                raise TeachingError("optimizer hyperparameters differ from the admitted recipe")
        for state in self.optimizer.state.values():
            if any(isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()) for value in state.values()):
                raise TeachingError("nonfinite optimizer state cannot continue training")
        for parameter, state in self.optimizer.state.items():
            if state:
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise TeachingError("AdamW resume state schema differs")
                for name in ("exp_avg", "exp_avg_sq"):
                    value = state[name]
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or value.dtype != parameter.dtype or value.device != parameter.device):
                        raise TeachingError("AdamW moments differ from learned parameter geometry")
                step = state["step"]
                if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) < 0 \
                        or not float(step).is_integer() or bool((state["exp_avg_sq"] < 0).any()):
                    raise TeachingError("AdamW step or squared moment state is invalid")
        return expected

    def _loss(self, batch):
        packet = self.model(batch.frame)
        total, losses, coverage = self.objectives(packet, batch, loss_weights=self.recipe.loss_weights)
        if batch.contrasts:
            if self.recipe.phase == "calibration":
                raise TeachingError("N3 owner calibration cannot consume N1/N2 contrast rehearsal")
            contrast_loss = sum(self.objectives.contrast_loss(packet, self.model(item.other_frame), item, batch)
                                for item in batch.contrasts) / len(batch.contrasts)
            total = total + contrast_loss * (self.recipe.loss_weights or {}).get("contrast", 1.0)
            losses["contrast"] = contrast_loss
            coverage["contrast"] = sum(int(item.available.sum()) for item in batch.contrasts)
        if not bool(torch.isfinite(total)):
            raise TeachingError("nonfinite reviewed objective cannot drive optimization")
        return total, losses, coverage

    def step(self, batch: TeachingBatch) -> dict:
        if self.failed:
            raise TeachingError("failed optimization state requires an admitted checkpoint restart")
        self.admission.verify(self.model, self.recipe, deep_custody=False)
        digest = self._check_batch(batch)
        split = "CALIBRATION_TRAIN" if self.recipe.phase == "calibration" else "TRAIN"
        order = self.reviewed["batch_order"].get(split)
        if batch.split != split or not isinstance(order, list) or not order \
                or self.batch_index >= len(order) or order[self.batch_index] != digest:
            raise TeachingError("optimization must follow the explicitly reviewed training replay order")
        self._phase(batch)
        expected = self._check_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            total, losses, coverage = self._loss(batch)
            if not total.requires_grad:
                raise TeachingError("reviewed objective has no trainable firstparty stage path")
            total.backward()
            allowed = {id(parameter) for _, parameter in expected}
            for parameter in [*self.model.parameters(), *self.objectives.parameters()]:
                if parameter.grad is not None and (id(parameter) not in allowed or not bool(torch.isfinite(parameter.grad).all())):
                    raise TeachingError("gradient reached a frozen stage or became nonfinite")
            frames = (batch.frame, *(item.other_frame for item in batch.contrasts))
            banks = tuple(bank for frame in frames for bank in (frame.query, frame.sources, frame.concepts,
                           frame.candidates, frame.relations, *frame.labels.values()))
            if any(bank.states.grad is not None for bank in banks):
                raise TeachingError("gradient reached frozen provider features")
            if not any(parameter.grad is not None for _, parameter in expected):
                raise TeachingError("reviewed objective produced no firstparty gradient")
            norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in expected],
                                                  self.recipe.max_grad_norm, error_if_nonfinite=True)
            # A persistent target/source/code mutation during the forward or
            # backward must fail before any learned weight update is applied.
            self.admission.verify(self.model, self.recipe, deep_custody=False)
            self._check_batch(batch)
            self.optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for _, parameter in expected):
                raise TeachingError("optimizer produced nonfinite firstparty weights")
        except Exception:
            self.failed = True
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.steps += 1
        self.batch_index += 1
        if self.batch_index == len(order):
            self.epoch += 1
            self.batch_index = 0
        event = self._event(batch, losses, coverage, float(total.detach()))
        event["gradient_norm_before_clip"] = float(norm)
        self.events.append(json.loads(canonical(event)))
        return event

    def evaluate(self, batch: TeachingBatch) -> dict:
        if self.failed:
            raise TeachingError("failed optimization state cannot produce development evidence")
        self.admission.verify(self.model, self.recipe, deep_custody=False)
        self._check_batch(batch)
        split = "CALIBRATION_DEV" if self.recipe.phase == "calibration" else "DEV"
        if batch.split != split:
            raise TeachingError("development evidence requires separately held-out source families")
        self._phase(batch)
        self.model.eval()
        self.objectives.eval()
        with torch.no_grad():
            total, losses, coverage = self._loss(batch)
        return self._event(batch, losses, coverage, float(total))

    def _event(self, batch, losses, coverage, total):
        return {"schema": "alice-personality-unqualified-teaching-evidence-v1", "phase": self.recipe.phase,
                "split": batch.split, "step": self.steps, "epoch": self.epoch, "loss": total,
                "objectives": {name: float(value.detach()) for name, value in losses.items()},
                "reviewed_target_coverage": coverage, "source_class": batch.source_class,
                "teaching_batch_sha256": batch.fingerprint(),
                "admission_binding_sha256": sha256(canonical(self._binding)).hexdigest(),
                "qualification": "UNQUALIFIED", "behavior_qualification": None,
                "acceptance_authority": False,
                "private_gradient_authorized": self.authorization["private_gradient_authorized"]}

    def save_checkpoint(self, destination: str | Path) -> dict:
        self.admission.verify(self.model, self.recipe, deep_custody=False)
        if self.failed:
            raise TeachingError("failed optimization cannot publish a resumable checkpoint")
        path = Path(destination)
        if not path.is_absolute() or path.exists() or not path.parent.is_dir():
            raise TeachingError("checkpoint requires a new absolute directory")
        if any(item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()) for item in (path, *path.parents)):
            raise TeachingError("checkpoint cannot traverse links")
        self._check_optimizer()
        self._check_events(self.events, self.steps)
        numpy_state = np.random.get_state()
        state = {"model": self.model.state_dict(), "objectives": self.objectives.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if next(self.model.parameters()).is_cuda else [],
                 "python_rng": random.getstate(), "numpy_rng": {
                     "algorithm": numpy_state[0], "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                     "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gauss": numpy_state[4]},
                 "steps": self.steps, "epoch": self.epoch, "batch_index": self.batch_index}
        path.mkdir(exist_ok=False)
        torch.save(state, path / "resume.pt")
        manifest = {"schema": CHECKPOINT_SCHEMA, "phase": self.recipe.phase,
                    "training_state": "OPTIMIZED_UNQUALIFIED" if self.steps else "IMPLEMENTED_UNQUALIFIED",
                    "qualification": "UNQUALIFIED", "acceptance_authority": False,
                    "binding": self._binding, "recipe": asdict(self.recipe),
                    "optimizer_parameters": [name for name, _ in self._expected_parameters()],
                    "resume_sha256": _hash(path / "resume.pt"), "step": self.steps,
                    "epoch": self.epoch, "batch_index": self.batch_index,
                    "events": json.loads(canonical(self.events)), "device": str(next(self.model.parameters()).device)}
        manifest["manifest_sha256"] = sha256(canonical(manifest)).hexdigest()
        with (path / "manifest.json").open("xb") as stream:
            stream.write(canonical(manifest) + b"\n")
        return manifest

    def _check_events(self, events, steps):
        split = "CALIBRATION_TRAIN" if self.recipe.phase == "calibration" else "TRAIN"
        if not isinstance(events, list) or len(events) != steps:
            raise TeachingError("optimization evidence must match the exact update count")
        binding_hash = sha256(canonical(self._binding)).hexdigest()
        for index, event in enumerate(events, 1):
            if (not isinstance(event, dict) or event.get("phase") != self.recipe.phase
                    or event.get("split") != split or type(event.get("step")) is not int or event["step"] != index
                    or event.get("qualification") != "UNQUALIFIED" or event.get("behavior_qualification") is not None
                    or event.get("acceptance_authority") is not False
                    or event.get("private_gradient_authorized") is not self.authorization["private_gradient_authorized"]
                    or event.get("admission_binding_sha256") != binding_hash):
                raise TeachingError("optimization evidence cannot claim acceptance or a different admitted run")

    def resume_checkpoint(self, source: str | Path, *, expected_manifest_sha256: str) -> None:
        self.admission.verify(self.model, self.recipe)
        path = _path(source, directory=True)
        if {item.name for item in path.iterdir()} != {"manifest.json", "resume.pt"}:
            raise TeachingError("resume checkpoint membership differs")
        manifest = _json(_path(path / "manifest.json"))
        supplied = manifest.pop("manifest_sha256", None)
        if supplied != _digest(expected_manifest_sha256, "externally pinned checkpoint manifest") \
                or supplied != sha256(canonical(manifest)).hexdigest():
            raise TeachingError("resume manifest differs from the externally admitted checkpoint")
        if manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("phase") != self.recipe.phase \
                or manifest.get("qualification") != "UNQUALIFIED" or manifest.get("acceptance_authority") is not False \
                or manifest.get("binding") != self._binding or manifest.get("recipe") != json.loads(canonical(asdict(self.recipe))) \
                or manifest.get("optimizer_parameters") != [name for name, _ in self._expected_parameters()] \
                or manifest.get("device") != str(next(self.model.parameters()).device):
            raise TeachingError("resume checkpoint source/model/code/recipe/runtime/phase binding differs")
        payload = _path(path / "resume.pt")
        if _hash(payload) != _digest(manifest.get("resume_sha256"), "resume payload"):
            raise TeachingError("resume tensor/optimizer/RNG payload changed")
        state = torch.load(payload, map_location=next(self.model.parameters()).device, weights_only=True)
        if not isinstance(state, dict) or set(state) != {"model", "objectives", "optimizer", "torch_rng", "cuda_rng",
                                                        "python_rng", "numpy_rng", "steps", "epoch", "batch_index"}:
            raise TeachingError("resume payload schema differs")
        for key in ("model", "objectives"):
            if not isinstance(state[key], dict) or any(not isinstance(value, torch.Tensor)
                    or not bool(torch.isfinite(value).all()) for value in state[key].values()):
                raise TeachingError("resume learned tensors must be finite")
        for key, manifest_key in (("steps", "step"), ("epoch", "epoch"), ("batch_index", "batch_index")):
            if type(state[key]) is not int or state[key] < 0 or state[key] != manifest.get(manifest_key):
                raise TeachingError("resume data cursor does not match bound evidence")
        split = "CALIBRATION_TRAIN" if self.recipe.phase == "calibration" else "TRAIN"
        order = self.reviewed["batch_order"].get(split)
        if not isinstance(order, list) or not order or state["batch_index"] >= len(order) \
                or state["steps"] != state["epoch"] * len(order) + state["batch_index"]:
            raise TeachingError("resume cursor is inconsistent with its reviewed replay order")
        if manifest.get("training_state") != ("OPTIMIZED_UNQUALIFIED" if state["steps"] else "IMPLEMENTED_UNQUALIFIED") \
                or not isinstance(manifest.get("events"), list) or len(manifest["events"]) != state["steps"]:
            raise TeachingError("resume evidence does not match the actual optimization count")
        self._check_events(manifest["events"], state["steps"])
        # Any failure after mutable state starts loading poisons this trainer;
        # another admitted checkpoint can restore it, but no step/evidence may.
        self.failed = True
        self.model.load_state_dict(state["model"], strict=True)
        self.objectives.load_state_dict(state["objectives"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self._check_optimizer()
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        rng = state["numpy_rng"]
        np.random.set_state((rng["algorithm"], rng["keys"].cpu().numpy().astype(np.uint32),
                             rng["position"], rng["has_gauss"], rng["cached_gauss"]))
        self.steps, self.epoch, self.batch_index = state["steps"], state["epoch"], state["batch_index"]
        self.events = manifest["events"]
        self.failed = False
