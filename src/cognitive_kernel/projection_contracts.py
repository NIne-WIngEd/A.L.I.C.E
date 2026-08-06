"""Memory M2.3 episode and projection-version contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_confidence,
    require_identifier,
    require_sha256,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope

PROJECTION_CONTRACT_SCHEMA_VERSION = "1.0.0"

EPISODE_KINDS = frozenset(
    {
        "interaction",
        "research_session",
        "mission_segment",
        "life_event",
        "learning_session",
        "relationship_event",
        "source_history_segment",
        "system_event",
        "other",
    }
)

EPISODE_STATES = frozenset(
    {
        "candidate",
        "accepted",
        "disputed",
        "superseded",
        "retired",
        "deleted",
    }
)

PROJECTION_TYPES = frozenset(
    {
        "graph",
        "vector",
        "lexical",
        "owner_model",
        "source_person_model",
        "self_model",
        "world_model",
        "social_model",
        "causal_model",
        "relationship_model",
        "preference_model",
        "goal_model",
        "mission_model",
        "temporal",
        "prediction",
        "episode_index",
    }
)

PROJECTION_STATES = frozenset(
    {
        "candidate",
        "shadow",
        "active",
        "stale",
        "superseded",
        "disputed",
        "quarantined",
        "retired",
        "deleted",
    }
)

PROJECTION_MODALITIES = frozenset(
    {
        "symbolic",
        "graph",
        "vector",
        "lexical",
        "multimodal",
        "neural",
        "relational",
        "temporal",
    }
)

PROJECTION_SUBJECT_TYPES = frozenset(
    {
        "owner",
        "source_person",
        "alice_self",
        "relationship",
        "mission",
        "claim",
        "episode",
        "world_entity",
        "social_entity",
        "causal_entity",
        "prediction",
        "skill",
    }
)


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CognitiveKernelContractError(
            f"{field} must be a positive integer"
        )
    return value


def _optional_identifier(
    value: object | None,
    field: str,
) -> str | None:
    return (
        require_identifier(value, field)
        if value is not None
        else None
    )


def _optional_timestamp(
    value: object | None,
    field: str,
) -> str | None:
    return (
        normalize_timestamp(value, field)
        if value is not None
        else None
    )


def _sorted_identifiers(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(normalize_identifier_sequence(values, field))
    )


@dataclass(frozen=True)
class EpisodeRecord:
    """Authority-aware episode metadata around full content held by a runtime."""

    envelope: MemoryUnitEnvelope
    episode_id: str
    episode_kind: str
    episode_state: str
    member_evidence_ids: tuple[str, ...]
    member_claim_version_ids: tuple[str, ...]
    member_candidate_ids: tuple[str, ...]
    mission_node_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    valid_from: str
    valid_to: str | None
    formed_at: str
    formation_component_id: str
    formation_version: str
    summary_content_digest: str
    full_content_digest: str
    confidence: float | None
    supersedes_episode_id: str | None
    generation: int
    episode_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        episode_id: object,
        episode_kind: object,
        episode_state: object,
        member_evidence_ids: Iterable[object] = (),
        member_claim_version_ids: Iterable[object] = (),
        member_candidate_ids: Iterable[object] = (),
        mission_node_ids: Iterable[object] = (),
        participant_ids: Iterable[object] = (),
        valid_from: object,
        valid_to: object | None,
        formed_at: object,
        formation_component_id: object,
        formation_version: object,
        summary_content_digest: object,
        full_content_digest: object,
        confidence: object | None,
        supersedes_episode_id: object | None = None,
        generation: object = 1,
    ) -> "EpisodeRecord":
        draft = cls(
            envelope=envelope,
            episode_id=require_identifier(
                episode_id,
                "episode_id",
            ),
            episode_kind=require_identifier(
                episode_kind,
                "episode_kind",
            ),
            episode_state=require_identifier(
                episode_state,
                "episode_state",
            ),
            member_evidence_ids=_sorted_identifiers(
                member_evidence_ids,
                "member_evidence_ids",
            ),
            member_claim_version_ids=_sorted_identifiers(
                member_claim_version_ids,
                "member_claim_version_ids",
            ),
            member_candidate_ids=_sorted_identifiers(
                member_candidate_ids,
                "member_candidate_ids",
            ),
            mission_node_ids=_sorted_identifiers(
                mission_node_ids,
                "mission_node_ids",
            ),
            participant_ids=_sorted_identifiers(
                participant_ids,
                "participant_ids",
            ),
            valid_from=normalize_timestamp(
                valid_from,
                "valid_from",
            ),
            valid_to=_optional_timestamp(
                valid_to,
                "valid_to",
            ),
            formed_at=normalize_timestamp(
                formed_at,
                "formed_at",
            ),
            formation_component_id=require_identifier(
                formation_component_id,
                "formation_component_id",
            ),
            formation_version=require_identifier(
                formation_version,
                "formation_version",
            ),
            summary_content_digest=require_sha256(
                summary_content_digest,
                "summary_content_digest",
            ),
            full_content_digest=require_sha256(
                full_content_digest,
                "full_content_digest",
            ),
            confidence=require_confidence(
                confidence,
                "confidence",
            ),
            supersedes_episode_id=_optional_identifier(
                supersedes_episode_id,
                "supersedes_episode_id",
            ),
            generation=_positive_integer(
                generation,
                "generation",
            ),
            episode_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "episode_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def _validate_material(self) -> None:
        self.envelope.validate()
        if self.envelope.record_type != "episode":
            raise CognitiveKernelContractError(
                "episode envelope must use record_type episode"
            )
        if self.envelope.authority_role != "registered_projection":
            raise CognitiveKernelContractError(
                "episode envelope must use registered_projection authority"
            )
        if self.envelope.record_id != self.episode_id:
            raise CognitiveKernelContractError(
                "episode_id must equal envelope.record_id"
            )
        if require_identifier(
            self.episode_id,
            "episode_id",
        ) != self.episode_id:
            raise CognitiveKernelContractError(
                "episode_id is not canonical"
            )
        if self.episode_kind not in EPISODE_KINDS:
            raise CognitiveKernelContractError(
                "episode_kind is not approved"
            )
        if self.episode_state not in EPISODE_STATES:
            raise CognitiveKernelContractError(
                "episode_state is not approved"
            )
        for field, values in (
            ("member_evidence_ids", self.member_evidence_ids),
            (
                "member_claim_version_ids",
                self.member_claim_version_ids,
            ),
            ("member_candidate_ids", self.member_candidate_ids),
            ("mission_node_ids", self.mission_node_ids),
            ("participant_ids", self.participant_ids),
        ):
            if _sorted_identifiers(values, field) != values:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if not (
            self.member_evidence_ids
            or self.member_claim_version_ids
            or self.member_candidate_ids
        ):
            raise CognitiveKernelContractError(
                "episode requires evidence, claims, or candidates"
            )
        valid_from = normalize_timestamp(
            self.valid_from,
            "valid_from",
        )
        if valid_from != self.valid_from:
            raise CognitiveKernelContractError(
                "valid_from is not canonical"
            )
        if self.valid_to is not None:
            valid_to = normalize_timestamp(
                self.valid_to,
                "valid_to",
            )
            if valid_to != self.valid_to:
                raise CognitiveKernelContractError(
                    "valid_to is not canonical"
                )
            if valid_to < valid_from:
                raise CognitiveKernelContractError(
                    "valid_to may not precede valid_from"
                )
        if normalize_timestamp(
            self.formed_at,
            "formed_at",
        ) != self.formed_at:
            raise CognitiveKernelContractError(
                "formed_at is not canonical"
            )
        for field, value in (
            (
                "formation_component_id",
                self.formation_component_id,
            ),
            ("formation_version", self.formation_version),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        require_sha256(
            self.summary_content_digest,
            "summary_content_digest",
        )
        require_sha256(
            self.full_content_digest,
            "full_content_digest",
        )
        require_confidence(self.confidence, "confidence")
        generation = _positive_integer(
            self.generation,
            "generation",
        )
        if generation == 1 and self.supersedes_episode_id is not None:
            raise CognitiveKernelContractError(
                "first episode generation may not supersede another"
            )
        if generation > 1 and self.supersedes_episode_id is None:
            raise CognitiveKernelContractError(
                "later episode generations require a predecessor"
            )
        if self.supersedes_episode_id == self.episode_id:
            raise CognitiveKernelContractError(
                "episode may not supersede itself"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "schema_version": PROJECTION_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "episode_id": self.episode_id,
            "episode_kind": self.episode_kind,
            "episode_state": self.episode_state,
            "member_evidence_ids": list(self.member_evidence_ids),
            "member_claim_version_ids": list(
                self.member_claim_version_ids
            ),
            "member_candidate_ids": list(
                self.member_candidate_ids
            ),
            "mission_node_ids": list(self.mission_node_ids),
            "participant_ids": list(self.participant_ids),
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "formed_at": self.formed_at,
            "formation_component_id": self.formation_component_id,
            "formation_version": self.formation_version,
            "summary_content_digest": self.summary_content_digest,
            "full_content_digest": self.full_content_digest,
            "confidence": self.confidence,
            "supersedes_episode_id": self.supersedes_episode_id,
            "generation": self.generation,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["episode_sha256"] = self.episode_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(self.episode_sha256, "episode_sha256")
        if (
            canonical_sha256(self.material_record())
            != self.episode_sha256
        ):
            raise CognitiveKernelContractError(
                "episode digest mismatch"
            )


@dataclass(frozen=True)
class ProjectionVersion:
    """Immutable derived projection version with explicit lineage."""

    envelope: MemoryUnitEnvelope
    projection_id: str
    version_id: str
    projection_type: str
    subject_type: str
    subject_id: str
    modalities: tuple[str, ...]
    generation: int
    source_episode_ids: tuple[str, ...]
    source_claim_version_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    valid_from: str
    valid_to: str | None
    produced_at: str
    projection_state: str
    responsible_component: str
    model_id: str | None
    model_version: str | None
    content_digest: str
    vector_space_id: str | None
    graph_namespace_id: str | None
    supersedes_version_id: str | None
    confidence: float | None
    projection_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        projection_id: object,
        version_id: object,
        projection_type: object,
        subject_type: object,
        subject_id: object,
        modalities: Iterable[object],
        generation: object,
        source_episode_ids: Iterable[object] = (),
        source_claim_version_ids: Iterable[object] = (),
        source_evidence_ids: Iterable[object] = (),
        valid_from: object,
        valid_to: object | None,
        produced_at: object,
        projection_state: object,
        responsible_component: object,
        model_id: object | None,
        model_version: object | None,
        content_digest: object,
        vector_space_id: object | None = None,
        graph_namespace_id: object | None = None,
        supersedes_version_id: object | None = None,
        confidence: object | None = None,
    ) -> "ProjectionVersion":
        draft = cls(
            envelope=envelope,
            projection_id=require_identifier(
                projection_id,
                "projection_id",
            ),
            version_id=require_identifier(
                version_id,
                "version_id",
            ),
            projection_type=require_identifier(
                projection_type,
                "projection_type",
            ),
            subject_type=require_identifier(
                subject_type,
                "subject_type",
            ),
            subject_id=require_identifier(
                subject_id,
                "subject_id",
            ),
            modalities=tuple(
                sorted(
                    normalize_identifier_sequence(
                        modalities,
                        "modalities",
                    )
                )
            ),
            generation=_positive_integer(
                generation,
                "generation",
            ),
            source_episode_ids=_sorted_identifiers(
                source_episode_ids,
                "source_episode_ids",
            ),
            source_claim_version_ids=_sorted_identifiers(
                source_claim_version_ids,
                "source_claim_version_ids",
            ),
            source_evidence_ids=_sorted_identifiers(
                source_evidence_ids,
                "source_evidence_ids",
            ),
            valid_from=normalize_timestamp(
                valid_from,
                "valid_from",
            ),
            valid_to=_optional_timestamp(
                valid_to,
                "valid_to",
            ),
            produced_at=normalize_timestamp(
                produced_at,
                "produced_at",
            ),
            projection_state=require_identifier(
                projection_state,
                "projection_state",
            ),
            responsible_component=require_identifier(
                responsible_component,
                "responsible_component",
            ),
            model_id=_optional_identifier(
                model_id,
                "model_id",
            ),
            model_version=_optional_identifier(
                model_version,
                "model_version",
            ),
            content_digest=require_sha256(
                content_digest,
                "content_digest",
            ),
            vector_space_id=_optional_identifier(
                vector_space_id,
                "vector_space_id",
            ),
            graph_namespace_id=_optional_identifier(
                graph_namespace_id,
                "graph_namespace_id",
            ),
            supersedes_version_id=_optional_identifier(
                supersedes_version_id,
                "supersedes_version_id",
            ),
            confidence=require_confidence(
                confidence,
                "confidence",
            ),
            projection_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "projection_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def _validate_material(self) -> None:
        self.envelope.validate()
        if self.envelope.record_type != "projection_version":
            raise CognitiveKernelContractError(
                "projection envelope must use projection_version"
            )
        if self.envelope.authority_role != "registered_projection":
            raise CognitiveKernelContractError(
                "projection envelope must use registered_projection authority"
            )
        if self.envelope.record_id != self.version_id:
            raise CognitiveKernelContractError(
                "version_id must equal envelope.record_id"
            )
        for field, value in (
            ("projection_id", self.projection_id),
            ("version_id", self.version_id),
            ("subject_id", self.subject_id),
            (
                "responsible_component",
                self.responsible_component,
            ),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.projection_type not in PROJECTION_TYPES:
            raise CognitiveKernelContractError(
                "projection_type is not approved"
            )
        if self.subject_type not in PROJECTION_SUBJECT_TYPES:
            raise CognitiveKernelContractError(
                "subject_type is not approved"
            )
        if not self.modalities:
            raise CognitiveKernelContractError(
                "modalities may not be empty"
            )
        if any(
            modality not in PROJECTION_MODALITIES
            for modality in self.modalities
        ):
            raise CognitiveKernelContractError(
                "modalities contain an unapproved value"
            )
        if tuple(sorted(self.modalities)) != self.modalities:
            raise CognitiveKernelContractError(
                "modalities are not canonical"
            )
        for field, values in (
            ("source_episode_ids", self.source_episode_ids),
            (
                "source_claim_version_ids",
                self.source_claim_version_ids,
            ),
            ("source_evidence_ids", self.source_evidence_ids),
        ):
            if _sorted_identifiers(values, field) != values:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if not (
            self.source_episode_ids
            or self.source_claim_version_ids
            or self.source_evidence_ids
        ):
            raise CognitiveKernelContractError(
                "projection requires source lineage"
            )
        valid_from = normalize_timestamp(
            self.valid_from,
            "valid_from",
        )
        if valid_from != self.valid_from:
            raise CognitiveKernelContractError(
                "valid_from is not canonical"
            )
        if self.valid_to is not None:
            valid_to = normalize_timestamp(
                self.valid_to,
                "valid_to",
            )
            if valid_to != self.valid_to:
                raise CognitiveKernelContractError(
                    "valid_to is not canonical"
                )
            if valid_to < valid_from:
                raise CognitiveKernelContractError(
                    "valid_to may not precede valid_from"
                )
        if normalize_timestamp(
            self.produced_at,
            "produced_at",
        ) != self.produced_at:
            raise CognitiveKernelContractError(
                "produced_at is not canonical"
            )
        if self.projection_state not in PROJECTION_STATES:
            raise CognitiveKernelContractError(
                "projection_state is not approved"
            )
        if (self.model_id is None) != (self.model_version is None):
            raise CognitiveKernelContractError(
                "model_id and model_version must appear together"
            )
        if "vector" in self.modalities and self.vector_space_id is None:
            raise CognitiveKernelContractError(
                "vector projections require vector_space_id"
            )
        if "graph" in self.modalities and self.graph_namespace_id is None:
            raise CognitiveKernelContractError(
                "graph projections require graph_namespace_id"
            )
        if (
            self.vector_space_id is not None
            and "vector" not in self.modalities
        ):
            raise CognitiveKernelContractError(
                "vector_space_id requires vector modality"
            )
        if (
            self.graph_namespace_id is not None
            and "graph" not in self.modalities
        ):
            raise CognitiveKernelContractError(
                "graph_namespace_id requires graph modality"
            )
        require_sha256(self.content_digest, "content_digest")
        require_confidence(self.confidence, "confidence")
        generation = _positive_integer(
            self.generation,
            "generation",
        )
        if generation == 1 and self.supersedes_version_id is not None:
            raise CognitiveKernelContractError(
                "first projection generation may not supersede another"
            )
        if generation > 1 and self.supersedes_version_id is None:
            raise CognitiveKernelContractError(
                "later projection generations require a predecessor"
            )
        if self.supersedes_version_id == self.version_id:
            raise CognitiveKernelContractError(
                "projection version may not supersede itself"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "schema_version": PROJECTION_CONTRACT_SCHEMA_VERSION,
            "envelope": self.envelope.metadata_record(),
            "projection_id": self.projection_id,
            "version_id": self.version_id,
            "projection_type": self.projection_type,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "modalities": list(self.modalities),
            "generation": self.generation,
            "source_episode_ids": list(self.source_episode_ids),
            "source_claim_version_ids": list(
                self.source_claim_version_ids
            ),
            "source_evidence_ids": list(self.source_evidence_ids),
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "produced_at": self.produced_at,
            "projection_state": self.projection_state,
            "responsible_component": self.responsible_component,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "content_digest": self.content_digest,
            "vector_space_id": self.vector_space_id,
            "graph_namespace_id": self.graph_namespace_id,
            "supersedes_version_id": self.supersedes_version_id,
            "confidence": self.confidence,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["projection_sha256"] = self.projection_sha256
        return record

    def validate(self) -> None:
        self._validate_material()
        require_sha256(
            self.projection_sha256,
            "projection_sha256",
        )
        if (
            canonical_sha256(self.material_record())
            != self.projection_sha256
        ):
            raise CognitiveKernelContractError(
                "projection digest mismatch"
            )
