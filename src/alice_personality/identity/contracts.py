"""Implemented identity tensor protocol v1; not a source acceptance codec.

The producer remains responsible for evidence permissions and lineage receipts.
This validator enforces their declared boundaries; model scores never grant
authority. Pointer IDs are returned for audit and never embedded as meanings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Mapping

import torch
from torch import Tensor

FRAME_SCHEMA = "alice-personality-identity-tensor-frame-v1"
PACKET_SCHEMA = "alice-personality-identity-tensor-packet-v1"
SYNTHETIC_KINDS = ("ASYN_DIRECT", "ASYN_BASE", "ASYN_TARGETED", "ASYN_CONTEXT")
SOURCE_KINDS = ("E0", "EINF", *SYNTHETIC_KINDS, "UNKNOWN", "HOST", "RELATIONSHIP", "AEXP", "ASELF")
CONCEPT_VIEWS = ("identity", "host", "relationship", "self", "context")
EDGE_ROLES = ("identity_support", "context_only", "excluded_context", "inference_support")
HEAD_FAMILIES = ("stance", "values", "relationship", "emotion", "communication", "voice", "drift")


class IdentityError(ValueError):
    """An unsupported or inconsistent identity frame was refused."""


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise IdentityError(f"{label} must be a nonempty string")


def _boolean(value: object, label: str) -> None:
    if type(value) is not bool:
        raise IdentityError(f"{label} must be an actual boolean")


def _bool_tensor(value: Tensor, shape: tuple, label: str) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or tuple(value.shape) != shape:
        raise IdentityError(f"{label} requires a boolean tensor of shape {shape}")


@dataclass(frozen=True)
class IdentityModelConfig:
    provider_width: int
    provider_state_count: int
    learned_width: int = 256
    attention_heads: int = 8
    graph_layers: int = 3
    readout_layers: int = 2

    def validate(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise IdentityError(f"{name} must be a positive checkpoint operating dimension")
        if self.provider_state_count < 2:
            raise IdentityError("provider must expose embedding and actual transformer layers")
        if self.learned_width % self.attention_heads:
            raise IdentityError("learned width must be divisible by attention heads")


@dataclass(frozen=True)
class FrozenTokenBank:
    """[batch, entries, actual layers, tokens, provider width], with masks."""
    states: Tensor
    token_mask: Tensor
    entry_mask: Tensor
    entry_ids: tuple[tuple[str | None, ...], ...]

    def validate(self, config: IdentityModelConfig, label: str, batch: int | None = None) -> None:
        value = self.states
        if (not isinstance(value, Tensor) or value.ndim != 5 or not value.is_floating_point()
                or any(dim < 1 for dim in value.shape) or not bool(torch.isfinite(value).all())):
            raise IdentityError(f"{label} requires finite nonempty all-layer token features")
        b, n, layers, tokens, width = value.shape
        if (batch is not None and b != batch) or (layers, width) != (config.provider_state_count, config.provider_width):
            raise IdentityError(f"{label} differs from the actual provider geometry/batch")
        _bool_tensor(self.token_mask, (b, n, tokens), f"{label} token mask")
        _bool_tensor(self.entry_mask, (b, n), f"{label} entry mask")
        if self.token_mask.device != value.device or self.entry_mask.device != value.device:
            raise IdentityError(f"{label} masks and features must share a device")
        if not torch.equal(self.token_mask.any(-1), self.entry_mask):
            raise IdentityError(f"{label} availability must match complete attended entries")
        if type(self.entry_ids) is not tuple or len(self.entry_ids) != b:
            raise IdentityError(f"{label} needs per-row pointer IDs")
        for row, ids in enumerate(self.entry_ids):
            if type(ids) is not tuple or len(ids) != n:
                raise IdentityError(f"{label} pointer dimensions differ")
            active = []
            for col, pointer in enumerate(ids):
                if bool(self.entry_mask[row, col]):
                    _text(pointer, f"{label} pointer")
                    active.append(pointer)
                elif pointer is not None:
                    raise IdentityError(f"{label} unavailable entry must have no pointer")
            if len(set(active)) != len(active):
                raise IdentityError(f"{label} pointers must be unique within a row")


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    source_family_id: str
    source_kind: str
    provenance_class: str
    identity_core_allowed: bool
    context_allowed: bool
    historical_truth_allowed: bool
    alice_lived_memory: bool
    autobiographical_recall_allowed: bool
    runtime_behavioral_prior_allowed: bool

    def validate(self) -> None:
        _text(self.record_id, "source ID")
        _text(self.source_family_id, "source family")
        if self.source_kind not in SOURCE_KINDS:
            raise IdentityError("unimplemented source kind requires an explicit protocol migration")
        expected = {"E0": "E0", "EINF": "E-INF", "AEXP": "A-EXP", "ASELF": "A-SELF"}
        provenance = "A-SYN" if self.source_kind in SYNTHETIC_KINDS else expected.get(self.source_kind, self.source_kind)
        if self.provenance_class != provenance:
            raise IdentityError("source kind and provenance class disagree")
        for flag in ("identity_core_allowed", "context_allowed", "historical_truth_allowed", "alice_lived_memory",
                     "autobiographical_recall_allowed", "runtime_behavioral_prior_allowed"):
            _boolean(getattr(self, flag), flag)
        if self.identity_core_allowed and self.source_kind not in {"E0", "EINF", *SYNTHETIC_KINDS}:
            raise IdentityError("host/relationship/self/unknown cannot become source-person identity authority")
        if self.historical_truth_allowed and self.source_kind != "E0":
            raise IdentityError("only canonical E0 can be source-person historical truth")
        if self.alice_lived_memory and self.source_kind != "AEXP":
            raise IdentityError("only actual Alice experience can be Alice lived memory")
        if self.autobiographical_recall_allowed and self.source_kind != "AEXP":
            raise IdentityError("source-person history and synthetic policy cannot become Alice autobiography")
        if self.runtime_behavioral_prior_allowed and self.source_kind not in SYNTHETIC_KINDS:
            raise IdentityError("runtime synthetic-prior authority requires explicit A-SYN provenance")
        if self.identity_core_allowed and self.source_kind in SYNTHETIC_KINDS and not self.runtime_behavioral_prior_allowed:
            raise IdentityError("synthetic identity policy requires explicit runtime prior permission")
        if self.autobiographical_recall_allowed and not self.alice_lived_memory:
            raise IdentityError("Alice autobiography requires actual lived-memory permission")


@dataclass(frozen=True)
class ConceptRecord:
    concept_id: str
    view: str
    residual: bool = False

    def validate(self) -> None:
        _text(self.concept_id, "concept ID")
        if self.view not in CONCEPT_VIEWS:
            raise IdentityError("unimplemented concept view requires an explicit protocol migration")
        _boolean(self.residual, "residual concept flag")


@dataclass(frozen=True)
class LabelSpec:
    label_id: str
    name: str
    description: str
    unit: str = "categorical"
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, family: str) -> None:
        for name in ("label_id", "name", "description", "unit"):
            _text(getattr(self, name), name)
        if family == "voice":
            if (self.unit == "categorical" or type(self.minimum) not in (int, float)
                    or type(self.maximum) not in (int, float) or not math.isfinite(self.minimum)
                    or not math.isfinite(self.maximum) or self.minimum >= self.maximum):
                raise IdentityError("voice controls require explicit portable units and finite ranges")
        elif self.minimum is not None or self.maximum is not None:
            raise IdentityError("numeric ranges are implemented only for voice controls")


@dataclass(frozen=True)
class SupportGraph:
    """Directed pointers into [source entries, concept entries].

    Sources are immutable input nodes; only concept receivers are updated.
    Relation meanings come from dynamic descriptions. Edge roles enforce
    authority independently of the learned relation representation.
    """
    senders: Tensor
    receivers: Tensor
    relation_indices: Tensor
    edge_mask: Tensor
    roles: tuple[tuple[str | None, ...], ...]

    def validate(self, batch: int, sources: int, concepts: int, relations: int,
                 device: torch.device) -> None:
        if not isinstance(self.senders, Tensor) or self.senders.ndim != 2 or self.senders.shape[0] != batch:
            raise IdentityError("graph endpoints require a [batch, edges] tensor")
        shape = tuple(self.senders.shape)
        for name in ("senders", "receivers", "relation_indices"):
            value = getattr(self, name)
            if value.dtype != torch.long or tuple(value.shape) != shape or value.device != device:
                raise IdentityError("graph pointers must be same-device int64 tensors of equal shape")
        _bool_tensor(self.edge_mask, shape, "graph edge mask")
        if self.edge_mask.device != device or type(self.roles) is not tuple or len(self.roles) != batch:
            raise IdentityError("graph role/mask dimensions differ")
        for row, roles in enumerate(self.roles):
            if type(roles) is not tuple or len(roles) != shape[1]:
                raise IdentityError("graph role dimensions differ")
            for edge, role in enumerate(roles):
                if bool(self.edge_mask[row, edge]):
                    if role not in EDGE_ROLES:
                        raise IdentityError("unimplemented edge role requires protocol migration")
                    sender, receiver, relation = (int(getattr(self, name)[row, edge])
                                                   for name in ("senders", "receivers", "relation_indices"))
                    if not (0 <= sender < sources + concepts and sources <= receiver < sources + concepts
                            and 0 <= relation < relations):
                        raise IdentityError("graph must point to actual nodes/relations without rewriting sources")
                elif role is not None:
                    raise IdentityError("unavailable graph edge cannot carry an authority role")


@dataclass(frozen=True)
class IdentityFrame:
    query: FrozenTokenBank
    sources: FrozenTokenBank
    concepts: FrozenTokenBank
    candidates: FrozenTokenBank
    relations: FrozenTokenBank
    labels: Mapping[str, FrozenTokenBank]
    source_records: tuple[tuple[SourceRecord | None, ...], ...]
    concept_records: tuple[tuple[ConceptRecord | None, ...], ...]
    label_specs: Mapping[str, tuple[tuple[LabelSpec | None, ...], ...]]
    graph: SupportGraph
    candidate_authority_mask: Tensor
    candidate_context_mask: Tensor
    schema: str = FRAME_SCHEMA

    def validate(self, config: IdentityModelConfig) -> None:
        config.validate()
        if self.schema != FRAME_SCHEMA:
            raise IdentityError("unsupported identity tensor frame schema")
        self.query.validate(config, "query")
        batch = self.query.states.shape[0]
        if not bool(self.query.entry_mask.any(-1).all()):
            raise IdentityError("every frame needs an actual situation/query")
        device = self.query.states.device
        for name in ("sources", "concepts", "candidates", "relations"):
            bank = getattr(self, name)
            bank.validate(config, name, batch)
            if bank.states.device != device or bank.states.dtype != self.query.states.dtype:
                raise IdentityError("all feature banks must share provider dtype/device")
        if set(self.labels) != set(HEAD_FAMILIES) or set(self.label_specs) != set(HEAD_FAMILIES):
            raise IdentityError("all implemented multi-head label families are required")
        for family, bank in self.labels.items():
            bank.validate(config, family, batch)
            if bank.states.device != device or bank.states.dtype != self.query.states.dtype:
                raise IdentityError("label features must share provider dtype/device")
            if not bool(bank.entry_mask.any(-1).all()):
                raise IdentityError(f"{family} needs an explicit runtime semantic label/control schema")
            self._records(bank, self.label_specs[family], LabelSpec, family)
        self._records(self.sources, self.source_records, SourceRecord, "sources")
        self._records(self.concepts, self.concept_records, ConceptRecord, "concepts")
        candidate_shape = tuple(self.candidates.entry_mask.shape)
        for name in ("candidate_authority_mask", "candidate_context_mask"):
            _bool_tensor(getattr(self, name), candidate_shape, name)
            if getattr(self, name).device != device:
                raise IdentityError("candidate masks must share the source device")
        s, k, r = (bank.states.shape[1] for bank in (self.sources, self.concepts, self.relations))
        self.graph.validate(batch, s, k, r, device)
        for row in range(batch):
            for edge in range(self.graph.senders.shape[1]):
                if not bool(self.graph.edge_mask[row, edge]):
                    continue
                sender, receiver, relation = (int(getattr(self.graph, name)[row, edge])
                                               for name in ("senders", "receivers", "relation_indices"))
                role = self.graph.roles[row][edge]
                sender_active = self.sources.entry_mask[row, sender] if sender < s else self.concepts.entry_mask[row, sender - s]
                if (not bool(sender_active) or not bool(self.concepts.entry_mask[row, receiver - s])
                        or not bool(self.relations.entry_mask[row, relation])):
                    raise IdentityError("active graph edges cannot address unavailable entries")
                target_view = self.concept_records[row][receiver - s].view
                if sender < s:
                    source = self.source_records[row][sender]
                    if role in {"identity_support", "inference_support"}:
                        if not source.identity_core_allowed or target_view != "identity":
                            raise IdentityError("identity graph support requires identity-authorized source and receiver")
                        if role == "inference_support" and source.source_kind != "EINF":
                            raise IdentityError("inference support requires declared EINF lineage")
                    elif role == "context_only":
                        expected_view = {"HOST": "host", "RELATIONSHIP": "relationship", "AEXP": "self", "ASELF": "self"}.get(source.source_kind, "context")
                        if not source.context_allowed or target_view != expected_view:
                            raise IdentityError("context-only support cannot enter source identity or a different runtime view")
                elif (role != "excluded_context"
                      and self.concept_records[row][sender - s].view != target_view):
                    raise IdentityError("concept graph messages cannot silently cross authority views")
                elif sender >= s and role != "excluded_context":
                    if (role in {"identity_support", "inference_support"}) != (target_view == "identity"):
                        raise IdentityError("concept graph authority role must match its separate view")

    @staticmethod
    def _records(bank: FrozenTokenBank, rows: tuple, kind: type, label: str) -> None:
        if type(rows) is not tuple or len(rows) != bank.states.shape[0]:
            raise IdentityError(f"{label} requires exact metadata dimensions")
        for row, records in enumerate(rows):
            if type(records) is not tuple or len(records) != bank.states.shape[1]:
                raise IdentityError(f"{label} metadata dimensions differ")
            for col, record in enumerate(records):
                if bool(bank.entry_mask[row, col]):
                    if type(record) is not kind:
                        raise IdentityError(f"{label} requires typed active metadata")
                    if kind is LabelSpec:
                        record.validate(label)
                        pointer = record.label_id
                    else:
                        record.validate()
                        pointer = record.record_id if kind is SourceRecord else record.concept_id
                    if pointer != bank.entry_ids[row][col]:
                        raise IdentityError(f"{label} metadata pointer disagrees with token bank")
                elif record is not None:
                    raise IdentityError(f"{label} unavailable entry must have no metadata")


@dataclass(frozen=True)
class IdentityRepresentation:
    source_tokens: Tensor
    source_states: Tensor
    concepts: Tensor
    concept_mask: Tensor
    views: Mapping[str, Tensor]
    view_masks: Mapping[str, Tensor]
    query_tokens: Tensor
    query_mask: Tensor
    identity_available: Tensor
    layer_readout_weights: Tensor
    concept_source_logits: Tensor
    concept_source_probabilities: Tensor
    concept_source_mask: Tensor


@dataclass(frozen=True)
class CalibrationBatch:
    """Explicit calibration-only labels; provenance is declared, not accepted.

    Mechanical public targets establish fit isolation only. A future governed
    producer must independently verify lineage, source-family holdouts and
    owner-reviewed fidelity before real calibration data is admitted.
    """
    frame: IdentityFrame
    targets: Mapping[str, Tensor]
    target_mask: Tensor
    source_class: str
    source_family_id: str
    target_lineage_sha256: str
    split: str = "CALIBRATION"
    target_masks: Mapping[str, Tensor] | None = None

    def validate(self, config: IdentityModelConfig) -> None:
        self.frame.validate(config)
        if self.split != "CALIBRATION" or self.source_class not in {"public_mechanical_fixture", "governed_calibration_targets"}:
            raise IdentityError("N3 needs explicitly separate calibration data")
        _text(self.source_family_id, "calibration source family")
        if not isinstance(self.target_lineage_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.target_lineage_sha256) is None:
            raise IdentityError("calibration target lineage needs a SHA256 binding")
        mask = self.frame.candidates.entry_mask & self.frame.candidate_authority_mask & self.frame.candidate_context_mask
        _bool_tensor(self.target_mask, tuple(mask.shape), "calibration target mask")
        if self.target_mask.device != mask.device or bool((self.target_mask & ~mask).any()) or not bool(self.target_mask.any()):
            raise IdentityError("calibration labels require available authorized candidates")
        if not self.targets or not set(self.targets) <= {"uncertainty", "failure_tail_risk", "owner_fidelity_uncertainty"}:
            raise IdentityError("calibration requires provided implemented target families")
        if self.target_masks is not None:
            if set(self.target_masks) != set(self.targets):
                raise IdentityError("per-target calibration availability must match provided families")
            union = torch.zeros_like(self.target_mask)
            for name, family_mask in self.target_masks.items():
                _bool_tensor(family_mask, tuple(mask.shape), f"{name} calibration availability")
                if family_mask.device != mask.device or bool((family_mask & ~self.target_mask).any()):
                    raise IdentityError("per-target calibration mask exceeds admitted aggregate availability")
                union |= family_mask
            if not torch.equal(union, self.target_mask):
                raise IdentityError("aggregate calibration mask must equal provided target availability union")
        identity_available = torch.tensor([any(record is not None and record.identity_core_allowed for record in row)
                                            for row in self.frame.source_records], device=mask.device)
        for name, value in self.targets.items():
            family_mask = self.target_mask if self.target_masks is None else self.target_masks[name]
            if (not isinstance(value, Tensor) or tuple(value.shape) != tuple(mask.shape)
                    or not value.is_floating_point() or value.device != mask.device
                    or not bool(torch.isfinite(value[family_mask]).all())
                    or not bool(((value[family_mask] >= 0) & (value[family_mask] <= 1)).all())):
                raise IdentityError(f"{name} available calibration labels must be finite probabilities matching candidates")
            unavailable = family_mask & ~identity_available[:, None]
            if bool((value[unavailable] != 1).any()):
                raise IdentityError("calibration cannot overrule deterministic missing identity evidence")


@dataclass(frozen=True)
class HeadOutput:
    scores: Tensor
    activations: Tensor
    mask: Tensor
    specs: tuple[tuple[LabelSpec | None, ...], ...]


@dataclass(frozen=True)
class IdentityDecisionPacket:
    candidate_ids: tuple[tuple[str | None, ...], ...]
    source_records: tuple[tuple[SourceRecord | None, ...], ...]
    concept_ids: tuple[tuple[str | None, ...], ...]
    concept_records: tuple[tuple[ConceptRecord | None, ...], ...]
    candidate_mask: Tensor
    preference_logits: Tensor
    preference_margins: Tensor
    preferences: Tensor
    co_valid_probabilities: Tensor
    heads: Mapping[str, HeadOutput]
    value_tradeoff_margins: Tensor
    voice_control_values: Tensor
    voice_control_confidence: Tensor
    evidence_pointers: Tensor
    historical_evidence_pointers: Tensor
    alice_experience_pointers: Tensor
    concept_pointers: Tensor
    evidence_sufficiency: Tensor
    uncertainty: Tensor
    contraindications: Tensor
    failure_tail_risk: Tensor
    owner_fidelity_uncertainty: Tensor
    identity_grounding_available: Tensor
    latent_candidates: Tensor
    representation: IdentityRepresentation
    schema: str = field(default=PACKET_SCHEMA, init=False)
    training_state: str = field(default="UNTRAINED", init=False)
    qualification: str = field(default="UNQUALIFIED", init=False)

    def validate(self) -> None:
        """Validate the implemented tensor packet before it leaves the core."""
        shape = tuple(self.candidate_mask.shape)
        _bool_tensor(self.candidate_mask, shape, "packet candidate mask")
        if len(shape) != 2 or set(self.heads) != set(HEAD_FAMILIES):
            raise IdentityError("packet must contain the complete dynamic multi-head decision")
        for name in ("preference_logits", "preference_margins", "preferences", "co_valid_probabilities",
                     "evidence_sufficiency", "uncertainty", "contraindications", "failure_tail_risk", "owner_fidelity_uncertainty"):
            if tuple(getattr(self, name).shape) != shape:
                raise IdentityError(f"packet {name} dimensions differ from candidates")
        for family, head in self.heads.items():
            if (head.scores.ndim != 3 or tuple(head.scores.shape[:2]) != shape
                    or head.activations.shape != head.scores.shape):
                raise IdentityError(f"packet {family} head dimensions differ")
            _bool_tensor(head.mask, tuple(head.scores.shape), f"packet {family} mask")
            if bool((head.mask & ~self.candidate_mask[..., None]).any()):
                raise IdentityError("packet head cannot grant blocked candidate authority")
        def check(value: object) -> None:
            if isinstance(value, Tensor):
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise IdentityError("nonfinite learned outputs cannot become a decision packet")
            elif isinstance(value, Mapping):
                for item in value.values():
                    check(item)
            elif isinstance(value, (IdentityRepresentation, HeadOutput)):
                for item in vars(value).values():
                    check(item)
        for value in vars(self).values():
            check(value)
