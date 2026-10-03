"""Versioned teaching facts, distinct from source rows and model outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import torch
from torch import Tensor

from ..identity.contracts import IdentityFrame
from ..identity.codec import fingerprint as _frame_fingerprint, tensor_digest

TARGET_SCHEMA = "alice-personality-reviewed-teaching-targets-v2"
REVIEW_SCHEMA = "alice-personality-reviewed-teaching-batches-v1"
FAMILY_SCHEMA = "alice-personality-teaching-family-splits-v1"
AUTH_SCHEMA = "alice-personality-current-training-authorization-v1"


class TeachingError(ValueError):
    """Teaching facts, admission or optimization state violate their contract."""


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def fingerprint(value: object) -> str:
    """Bind complete feature/target bytes and IDs without exposing their values."""
    return _frame_fingerprint(value)


@dataclass(frozen=True)
class ReviewedTarget:
    """Known values with exact availability and separately reviewed uncertainty.

    Missing positions may use NaN placeholders, but never carry loss. Values
    do not come from candidate order, output scores or unordered competitors.
    """
    values: Tensor
    available: Tensor
    uncertainty: Tensor

    def validate(self, shape: tuple, allowed: Tensor, *, probability: bool = True) -> None:
        if (not isinstance(self.values, Tensor) or not self.values.is_floating_point()
                or tuple(self.values.shape) != shape or self.values.device != allowed.device
                or not isinstance(self.available, Tensor) or self.available.dtype != torch.bool
                or tuple(self.available.shape) != shape or self.available.device != allowed.device
                or not isinstance(self.uncertainty, Tensor) or not self.uncertainty.is_floating_point()
                or tuple(self.uncertainty.shape) != shape or self.uncertainty.device != allowed.device):
            raise TeachingError("reviewed values, Boolean availability and uncertainty must match output geometry")
        if bool((self.available & ~allowed).any()):
            raise TeachingError("reviewed target addresses unavailable or unauthorized output positions")
        for value in (self.values, self.uncertainty):
            if value.requires_grad:
                raise TeachingError("reviewed targets cannot be model gradients")
            if not bool(torch.isfinite(value[self.available]).all()):
                raise TeachingError("available reviewed targets must have finite facts")
        uncertainty = self.uncertainty[self.available]
        if not bool(((uncertainty >= 0) & (uncertainty <= 1)).all()):
            raise TeachingError("reviewed uncertainty must be a probability")
        if probability:
            values = self.values[self.available]
            if not bool(((values >= 0) & (values <= 1)).all()):
                raise TeachingError("reviewed probability targets must lie in zero-to-one")


@dataclass(frozen=True)
class ContrastConstraint:
    other_frame: IdentityFrame
    field: str
    relation: str
    position_mask: Tensor
    available: Tensor
    distance_bound: Tensor
    uncertainty: Tensor
    production_binding: Mapping | None = None
    production_session: Mapping | None = None


@dataclass(frozen=True)
class TeachingBatch:
    frame: IdentityFrame
    targets: Mapping[str, ReviewedTarget]
    case_ids: tuple[str, ...]
    source_family_ids: tuple[tuple[str, ...], ...]
    phase: str
    split: str
    source_class: str
    contrasts: tuple[ContrastConstraint, ...] = ()
    schema: str = TARGET_SCHEMA
    production_binding: Mapping | None = None
    production_session: Mapping | None = None

    def validate(self, config) -> None:
        self.frame.validate(config)
        count = self.frame.query.states.shape[0]
        if self.schema != TARGET_SCHEMA or self.phase not in {"n1", "n2", "calibration"}:
            raise TeachingError("unsupported teaching target schema or phase")
        expected = {"TRAIN", "DEV"} if self.phase != "calibration" else {"CALIBRATION_TRAIN", "CALIBRATION_DEV"}
        if self.split not in expected:
            raise TeachingError("teaching cannot consume FINAL or a different phase split")
        classes = {"public_mechanical_fixture", "governed_calibration_targets"} if self.phase == "calibration" \
            else {"public_mechanical_fixture", "governed_identity_targets"}
        if self.source_class not in classes:
            raise TeachingError("teaching source class does not match its phase")
        if type(self.case_ids) is not tuple or len(self.case_ids) != count \
                or len(set(self.case_ids)) != count or any(not isinstance(value, str) or not value for value in self.case_ids):
            raise TeachingError("teaching needs exact unique per-row case IDs")
        if type(self.source_family_ids) is not tuple or len(self.source_family_ids) != count:
            raise TeachingError("teaching needs explicit per-row source family IDs")
        for row, declared in enumerate(self.source_family_ids):
            actual = {source.source_family_id for source in self.frame.source_records[row] if source is not None}
            if type(declared) is not tuple or len(set(declared)) != len(declared) or set(declared) != actual:
                raise TeachingError("reviewed source families must match the complete declared frame lineage")
        if not isinstance(self.targets, Mapping) or not self.targets \
                or any(not isinstance(name, str) or not isinstance(value, ReviewedTarget) for name, value in self.targets.items()):
            raise TeachingError("source rows are not reviewed teaching targets")
        if type(self.contrasts) is not tuple or any(not isinstance(value, ContrastConstraint) for value in self.contrasts):
            raise TeachingError("contrasts require explicitly reviewed typed constraints")
        for value in (self.production_binding, self.production_session):
            if value is not None and not isinstance(value, Mapping):
                raise TeachingError("frame production provenance must be explicit metadata objects")

    def fingerprint(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True)
class TrainingRecipe:
    phase: str
    learning_rate: float
    max_grad_norm: float
    seed: int
    adapt_n1: bool = False
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    deterministic: bool = True
    loss_weights: Mapping[str, float] | None = None

    def validate(self) -> None:
        if self.phase not in {"n1", "n2", "calibration"} or type(self.adapt_n1) is not bool \
                or (self.adapt_n1 and self.phase != "n2") or type(self.deterministic) is not bool:
            raise TeachingError("recipe must select an implemented phase and explicit adaptation strategy")
        for name in ("learning_rate", "max_grad_norm", "eps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise TeachingError(f"recipe {name} must be finite and positive")
        if type(self.weight_decay) not in (int, float) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise TeachingError("recipe weight decay must be finite and nonnegative")
        if type(self.seed) is not int or self.seed < 0 or type(self.betas) is not tuple or len(self.betas) != 2 \
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1 for value in self.betas):
            raise TeachingError("recipe seed/betas are invalid")
        if self.loss_weights is not None and (not isinstance(self.loss_weights, Mapping)
                or any(not isinstance(key, str) or type(value) not in (int, float)
                       or not math.isfinite(value) or value <= 0 for key, value in self.loss_weights.items())):
            raise TeachingError("recipe loss weights must be finite positive declared values")

    def fingerprint(self) -> str:
        self.validate()
        return sha256(canonical(asdict(self))).hexdigest()
