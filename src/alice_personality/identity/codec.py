"""Versioned world/evidence input and complete decision output codecs.

This is an implemented packaging protocol, not source acceptance or a teacher.
Targets are deliberately outside the input document. Arbitrary source text
still requires producer review: structural separation cannot prove its truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import json
from typing import Mapping

import torch

from .contracts import (ConceptRecord, HEAD_FAMILIES, IdentityDecisionPacket,
                        IdentityError, LabelSpec, SourceRecord)

DOCUMENT_SCHEMA = "alice-personality-cognitive-frame-document-v1"
OUTPUT_SCHEMA = "alice-personality-decision-document-v1"
# Explicit absence is accepted. These families declare full-frame coverage;
# entries and additional semantic roles remain dynamic, without numeric caps.
REQUIRED_ROLES = frozenset({"actors_roles", "situation", "goals_constraints",
    "stakes_consequences", "evidence_time_uncertainty", "selected_spans",
    "source_identity", "host", "relationship", "alice_self", "social_emotional"})


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def tensor_digest(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    digest = sha256(canonical({"shape": list(data.shape), "dtype": str(data.dtype)}))
    digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    """Exact tensor, typed-metadata and pointer binding without their contents."""
    def describe(item):
        if isinstance(item, torch.Tensor):
            return {"tensor_sha256": tensor_digest(item)}
        if is_dataclass(item):
            return {field.name: describe(getattr(item, field.name)) for field in fields(item)}
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise IdentityError("fingerprinted mappings require string keys")
            return {key: describe(value) for key, value in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [describe(value) for value in item]
        if item is None or type(item) in (bool, int, float, str):
            return item
        raise IdentityError("unsupported identity fingerprint value")
    return sha256(canonical(describe(value))).hexdigest()


def _text(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise IdentityError("semantic text and pointer names must be nonempty strings")


def _exact(value: object, fields: set[str]) -> dict:
    if type(value) is not dict or set(value) != fields:
        raise IdentityError("document fields differ from the implemented target-free schema")
    return value


@dataclass(frozen=True)
class SituationField:
    field_id: str
    semantic_role: str
    name: str
    text: str | None
    available: bool

    def validate(self) -> None:
        for value in (self.field_id, self.semantic_role, self.name):
            _text(value)
        if type(self.available) is not bool:
            raise IdentityError("field availability must be an actual boolean")
        if self.available:
            _text(self.text)
        elif self.text is not None:
            raise IdentityError("unavailable situation fields cannot carry hidden content")

    def semantic_text(self) -> str:
        self.validate()
        return f"{self.semantic_role}\n{self.name}\n{self.text if self.available else '[not supplied]'}"


@dataclass(frozen=True)
class SourceEntry:
    text: str
    record: SourceRecord


@dataclass(frozen=True)
class ConceptEntry:
    text: str
    record: ConceptRecord


@dataclass(frozen=True)
class SemanticEntry:
    entry_id: str
    text: str


@dataclass(frozen=True)
class CandidateEntry:
    candidate_id: str
    text: str
    authority_allowed: bool
    context_allowed: bool


@dataclass(frozen=True)
class GraphEdge:
    sender_id: str
    receiver_id: str
    relation_id: str
    role: str


@dataclass(frozen=True)
class CognitiveFrameDocument:
    fields: tuple[SituationField, ...]
    sources: tuple[SourceEntry, ...]
    concepts: tuple[ConceptEntry, ...]
    candidates: tuple[CandidateEntry, ...]
    relations: tuple[SemanticEntry, ...]
    labels: Mapping[str, tuple[LabelSpec, ...]]
    edges: tuple[GraphEdge, ...]
    schema: str = DOCUMENT_SCHEMA

    def validate(self) -> None:
        if self.schema != DOCUMENT_SCHEMA:
            raise IdentityError("unsupported cognitive document schema")
        groups = (self.fields, self.sources, self.concepts, self.candidates, self.relations, self.edges)
        if any(type(group) is not tuple for group in groups):
            raise IdentityError("document collections require immutable ordered tuples")
        pointers = []
        for field in self.fields:
            if type(field) is not SituationField:
                raise IdentityError("situation fields require implemented typed semantics")
            field.validate()
        if not self.fields or not REQUIRED_ROLES <= {field.semantic_role for field in self.fields}:
            raise IdentityError("full frame roles need explicit supplied or unavailable fields")
        if not any(field.available and field.semantic_role == "situation" for field in self.fields):
            raise IdentityError("an actual situation is required")
        self._unique([field.field_id for field in self.fields])
        for entry in self.sources:
            if type(entry) is not SourceEntry or type(entry.record) is not SourceRecord:
                raise IdentityError("source entries require typed provenance and permission flags")
            _text(entry.text)
            entry.record.validate()
            pointers.append(entry.record.record_id)
        for entry in self.concepts:
            if type(entry) is not ConceptEntry or type(entry.record) is not ConceptRecord:
                raise IdentityError("concept entries require typed authority views")
            _text(entry.text)
            entry.record.validate()
            pointers.append(entry.record.concept_id)
        self._unique(pointers)
        for entry in self.candidates:
            if type(entry) is not CandidateEntry:
                raise IdentityError("candidate entries require explicit admission masks")
            _text(entry.candidate_id)
            _text(entry.text)
            if type(entry.authority_allowed) is not bool or type(entry.context_allowed) is not bool:
                raise IdentityError("candidate admission flags must be actual booleans")
        self._unique([entry.candidate_id for entry in self.candidates])
        for entry in self.relations:
            if type(entry) is not SemanticEntry:
                raise IdentityError("relations require actual semantic descriptions")
            _text(entry.entry_id)
            _text(entry.text)
        self._unique([entry.entry_id for entry in self.relations])
        if set(self.labels) != set(HEAD_FAMILIES):
            raise IdentityError("the complete multi-head semantic label schema is required")
        for family, labels in self.labels.items():
            if type(labels) is not tuple or not labels:
                raise IdentityError("each head needs dynamic explicit semantic labels")
            for label in labels:
                if type(label) is not LabelSpec:
                    raise IdentityError("head labels must use implemented portable specs")
                label.validate(family)
            self._unique([label.label_id for label in labels])
        nodes, concepts = set(pointers), {entry.record.concept_id for entry in self.concepts}
        relations = {entry.entry_id for entry in self.relations}
        source_records = {entry.record.record_id: entry.record for entry in self.sources}
        concept_records = {entry.record.concept_id: entry.record for entry in self.concepts}
        for edge in self.edges:
            if type(edge) is not GraphEdge or edge.sender_id not in nodes or edge.receiver_id not in concepts:
                raise IdentityError("graph must connect actual source/concept pointers to concepts")
            if edge.relation_id not in relations:
                raise IdentityError("graph relations require provided descriptions")
            # The tensor-frame validator additionally checks source/view rights.
            from .contracts import EDGE_ROLES
            if edge.role not in EDGE_ROLES:
                raise IdentityError("unknown graph authority role requires protocol migration")
            view = concept_records[edge.receiver_id].view
            if edge.sender_id in source_records:
                source = source_records[edge.sender_id]
                if edge.role in {"identity_support", "inference_support"}:
                    if not source.identity_core_allowed or view != "identity":
                        raise IdentityError("identity graph support requires identity-authorized source and receiver")
                    if edge.role == "inference_support" and source.source_kind != "EINF":
                        raise IdentityError("inference support requires EINF lineage")
                elif edge.role == "context_only":
                    expected = {"HOST": "host", "RELATIONSHIP": "relationship", "AEXP": "self", "ASELF": "self"}.get(source.source_kind, "context")
                    if not source.context_allowed or view != expected:
                        raise IdentityError("context graph cannot cross source authority views")
            elif edge.role != "excluded_context":
                if concept_records[edge.sender_id].view != view:
                    raise IdentityError("concept graph cannot cross source authority views")
                if (edge.role in {"identity_support", "inference_support"}) != (view == "identity"):
                    raise IdentityError("concept graph role must match its source authority view")

    @staticmethod
    def _unique(values: list[str]) -> None:
        if len(set(values)) != len(values):
            raise IdentityError("pointer names must be unique within their namespace")

    def to_mapping(self) -> dict:
        self.validate()
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256(canonical(self.to_mapping())).hexdigest()

    @classmethod
    def from_mapping(cls, raw: dict) -> "CognitiveFrameDocument":
        _exact(raw, {"schema", "fields", "sources", "concepts", "candidates", "relations", "labels", "edges"})
        if any(type(raw[key]) is not list for key in ("fields", "sources", "concepts", "candidates", "relations", "edges")):
            raise IdentityError("serialized document collections must be JSON lists")
        def create(kind, value):
            _exact(value, set(kind.__dataclass_fields__))
            return kind(**value)
        fields = tuple(create(SituationField, value) for value in raw["fields"])
        def records(values, entry_type, record_type):
            out = []
            for value in values:
                _exact(value, {"text", "record"})
                out.append(entry_type(value["text"], create(record_type, value["record"])))
            return tuple(out)
        if type(raw["labels"]) is not dict or any(type(value) is not list for value in raw["labels"].values()):
            raise IdentityError("serialized semantic labels need explicit JSON list families")
        document = cls(fields, records(raw["sources"], SourceEntry, SourceRecord),
            records(raw["concepts"], ConceptEntry, ConceptRecord),
            tuple(create(CandidateEntry, value) for value in raw["candidates"]),
            tuple(create(SemanticEntry, value) for value in raw["relations"]),
            {family: tuple(create(LabelSpec, value) for value in labels) for family, labels in raw["labels"].items()},
            tuple(create(GraphEdge, value) for value in raw["edges"]), raw["schema"])
        document.validate()
        return document

    @classmethod
    def from_json(cls, payload: str | bytes) -> "CognitiveFrameDocument":
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise IdentityError("duplicate JSON document keys are forbidden")
                value[key] = item
            return value
        raw = json.loads(payload, object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(IdentityError("nonfinite JSON is forbidden")))
        return cls.from_mapping(raw)


def decision_to_mapping(packet: IdentityDecisionPacket) -> dict:
    """Complete downstream packet, with native latents and audit pointers.

    Returns protected runtime data to the caller; never prints or saves it.
    Runtime tensor output and JSON output retain the same unqualified status.
    """
    packet.validate()
    tensor = lambda value: value.detach().cpu().tolist()
    result = {"schema": OUTPUT_SCHEMA, "tensor_schema": packet.schema,
              "training_state": packet.training_state, "qualification": packet.qualification,
              "candidate_ids": packet.candidate_ids,
              "sources": [[asdict(record) if record else None for record in row] for row in packet.source_records],
              "concept_ids": packet.concept_ids,
              "concepts": [[asdict(record) if record else None for record in row] for row in packet.concept_records]}
    for name in ("candidate_mask", "preference_logits", "preference_margins", "preferences", "co_valid_probabilities",
        "value_tradeoff_margins", "voice_control_values", "voice_control_confidence", "evidence_pointers",
        "historical_evidence_pointers", "alice_experience_pointers", "concept_pointers", "evidence_sufficiency",
        "uncertainty", "contraindications", "failure_tail_risk", "owner_fidelity_uncertainty", "identity_grounding_available"):
        result[name] = tensor(getattr(packet, name))
    result["heads"] = {family: {"scores": tensor(head.scores), "activations": tensor(head.activations),
        "mask": tensor(head.mask), "specs": [[asdict(spec) if spec else None for spec in row] for row in head.specs]}
        for family, head in packet.heads.items()}
    result["native_latent"] = {"candidates": tensor(packet.latent_candidates),
        "concepts": tensor(packet.representation.concepts), "concept_mask": tensor(packet.representation.concept_mask),
        "views": {name: tensor(value) for name, value in packet.representation.views.items()},
        "view_masks": {name: tensor(value) for name, value in packet.representation.view_masks.items()}}
    canonical(result)  # Refuse any nonfinite serialized values, including latents.
    return result
