"""Additive, non-authoritative MFM semantics contract (1.6.0).

This is a separate wire format. Frozen 1.5.0 examples, hashes, trainer targets
and scores cannot be silently upgraded: missing labels mean *unknown*, not an
adjudicated absence. Registry metadata here is a snapshot, and the independent
memory gate must re-read current source policy and target registration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CognitiveKernelContractError, canonical_sha256, normalize_identifier_sequence,
    normalize_timestamp, require_identifier,
)
from .contracts import ProductHostScope
from .formation_contracts import (
    FormationContextPacket, FormationDisposition, FormationProposal,
    MemoryProposalBundle, validate_formation_grounding,
)
from .formation_evaluation import FormationAssessment, FormationGoldCase, assess_formation
from .formation_learning import (
    OUTPUT_SCHEMA, bundle_from_output, context_from_record,
)


SCHEMA_VERSION = "1.6.0"
CONTEXT_SCHEMA = "mfm-formation-context-v1.6"
OUTPUT_SCHEMA_V16 = "mfm-formation-output-v1.6"
SENSITIVITY_ORDER = ("public", "internal", "private", "highly_sensitive")
TARGET_KINDS = frozenset({"entity", "scene", "mission", "workspace"})
ADJUDICATION_DIMENSIONS = frozenset({
    "sensitivity", "episode", "relationship", "mission", "workspace",
})
FULL_ROLE_ADJUDICATION_DIMENSIONS = ADJUDICATION_DIMENSIONS | frozenset({
    "correction", "source_person", "contradiction", "outcome", "abstention",
})


def _canonical_id(value: str, name: str) -> None:
    if require_identifier(value, name) != value:
        raise CognitiveKernelContractError(f"{name} must be canonical")


def _ids(values: tuple[str, ...], name: str) -> None:
    if normalize_identifier_sequence(values, name) != values:
        raise CognitiveKernelContractError(f"{name} must be canonical")


def _record(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise CognitiveKernelContractError(f"{name} has unsupported fields")
    return value


def _array(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise CognitiveKernelContractError(f"{name} must be an array")
    return value


@dataclass(frozen=True)
class RegisteredSourceSensitivity:
    """A source registry's *minimum* sensitivity; never model authority."""

    ref_id: str
    minimum: str

    def validate(self) -> None:
        _canonical_id(self.ref_id, "sensitivity ref_id")
        if self.minimum not in SENSITIVITY_ORDER:
            raise CognitiveKernelContractError("unsupported registered sensitivity")

    def record(self) -> dict[str, object]:
        self.validate()
        return {"ref_id": self.ref_id, "minimum": self.minimum}


@dataclass(frozen=True)
class RegisteredFormationTarget:
    """A scoped snapshot of a target resolved by an authority-owned service."""

    ref_id: str
    kind: str
    scope: ProductHostScope
    authority_namespace_id: str
    supporting_evidence_refs: tuple[str, ...]

    def validate(self) -> None:
        _canonical_id(self.ref_id, "target ref_id")
        if self.kind not in TARGET_KINDS:
            raise CognitiveKernelContractError("unsupported target kind")
        self.scope.validate()
        _canonical_id(self.authority_namespace_id, "target authority_namespace_id")
        _ids(self.supporting_evidence_refs, "supporting_evidence_refs")
        if not self.supporting_evidence_refs:
            raise CognitiveKernelContractError("target needs registered supporting evidence")

    def record(self) -> dict[str, object]:
        self.validate()
        return {"ref_id": self.ref_id, "kind": self.kind,
                "scope": self.scope.metadata_record(),
                "authority_namespace_id": self.authority_namespace_id,
                "supporting_evidence_refs": list(self.supporting_evidence_refs)}


@dataclass(frozen=True)
class FormationContextV16:
    base: FormationContextPacket
    source_sensitivities: tuple[RegisteredSourceSensitivity, ...]
    targets: tuple[RegisteredFormationTarget, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        self.base.validate()
        if self.schema_version != SCHEMA_VERSION:
            raise CognitiveKernelContractError("unsupported v1.6 context schema")
        evidence_ids = {ref.ref_id for ref in self.base.evidence}
        sensitivity_ids = [item.ref_id for item in self.source_sensitivities]
        for item in self.source_sensitivities:
            item.validate()
        if len(sensitivity_ids) != len(set(sensitivity_ids)) or set(sensitivity_ids) != evidence_ids:
            raise CognitiveKernelContractError("registered sensitivity must cover each source exactly once")
        seen_targets: set[str] = set()
        for target in self.targets:
            target.validate()
            if target.ref_id in seen_targets:
                raise CognitiveKernelContractError("duplicate registered target")
            seen_targets.add(target.ref_id)
            if (target.scope != self.base.scope or
                    target.authority_namespace_id != self.base.authority_namespace_id):
                raise CognitiveKernelContractError("foreign-scope registered target")
            if not set(target.supporting_evidence_refs).issubset(evidence_ids):
                raise CognitiveKernelContractError("registered target cites absent evidence")

    def record(self) -> dict[str, object]:
        self.validate()
        return {"schema": CONTEXT_SCHEMA, "schema_version": self.schema_version,
                "base_context": self.base.metadata_record(),
                "source_sensitivities": [item.record() for item in self.source_sensitivities],
                "targets": [target.record() for target in self.targets]}

    def content_digest(self) -> str:
        return canonical_sha256(self.record())


def context_v16_from_record(value: object) -> FormationContextV16:
    row = _record(value, {"schema", "schema_version", "base_context",
                          "source_sensitivities", "targets"}, "v1.6 context")
    if row["schema"] != CONTEXT_SCHEMA or row["schema_version"] != SCHEMA_VERSION:
        raise CognitiveKernelContractError("unsupported v1.6 context schema")
    base = context_from_record(row["base_context"])
    context = FormationContextV16(
        base,
        tuple(RegisteredSourceSensitivity(**_record(item, {"ref_id", "minimum"},
                                                  "source sensitivity"))
              for item in _array(row["source_sensitivities"], "source_sensitivities")),
        tuple(RegisteredFormationTarget(
            ref_id=item["ref_id"], kind=item["kind"],
            scope=ProductHostScope.create(**item["scope"]),
            authority_namespace_id=item["authority_namespace_id"],
            supporting_evidence_refs=tuple(_array(item["supporting_evidence_refs"],
                                                  "supporting_evidence_refs")),
        ) for value in _array(row["targets"], "targets")
              for item in [_record(value, {"ref_id", "kind", "scope",
                                          "authority_namespace_id",
                                          "supporting_evidence_refs"}, "registered target")]),
        row["schema_version"],
    )
    if context.record() != row:
        raise CognitiveKernelContractError("v1.6 context is not canonical")
    return context


@dataclass(frozen=True)
class EpisodeSemantics:
    """A proposed event interval; none at one end means that end is unknown."""

    event_start: str | None
    event_end: str | None
    participant_refs: tuple[str, ...]
    scene_refs: tuple[str, ...]

    def validate(self) -> None:
        if self.event_start is None and self.event_end is None:
            raise CognitiveKernelContractError("episode needs at least one event boundary")
        for name in ("event_start", "event_end"):
            value = getattr(self, name)
            if value is not None and normalize_timestamp(value, name) != value:
                raise CognitiveKernelContractError(f"{name} must be canonical")
        if self.event_start and self.event_end and self.event_end < self.event_start:
            raise CognitiveKernelContractError("episode end precedes start")
        _ids(self.participant_refs, "episode participant_refs")
        _ids(self.scene_refs, "episode scene_refs")

    def record(self) -> dict[str, object]:
        self.validate()
        return {"event_start": self.event_start, "event_end": self.event_end,
                "participant_refs": list(self.participant_refs),
                "scene_refs": list(self.scene_refs)}


@dataclass(frozen=True)
class FormationProposalV16:
    base: FormationProposal
    sensitivity_hint: str | None = None
    episode: EpisodeSemantics | None = None
    relationship_counterpart_ref: str | None = None
    mission_target_refs: tuple[str, ...] = ()
    workspace_target_refs: tuple[str, ...] = ()

    def validate(self) -> None:
        self.base.validate()
        if self.sensitivity_hint is not None and self.sensitivity_hint not in SENSITIVITY_ORDER:
            raise CognitiveKernelContractError("unsupported sensitivity hint")
        if self.episode is not None:
            self.episode.validate()
        if (self.base.kind == "episode") != (self.episode is not None):
            raise CognitiveKernelContractError("v1.6 episode kind requires event semantics")
        if self.relationship_counterpart_ref is not None:
            _canonical_id(self.relationship_counterpart_ref, "relationship counterpart")
            if (self.base.domain != "relationship" or
                    self.relationship_counterpart_ref == self.base.subject_ref):
                raise CognitiveKernelContractError("invalid relationship counterpart")
        if self.base.kind in {"relationship", "relationship_norm"} and self.base.domain != "relationship":
            raise CognitiveKernelContractError("relationship kind needs relationship domain")
        if self.base.domain == "relationship" and self.relationship_counterpart_ref is None:
            raise CognitiveKernelContractError("v1.6 relationship needs counterpart")
        _ids(self.mission_target_refs, "mission_target_refs")
        _ids(self.workspace_target_refs, "workspace_target_refs")

    def record(self) -> dict[str, object]:
        self.validate()
        return {"proposal": self.base.record(), "sensitivity_hint": self.sensitivity_hint,
                "episode": self.episode.record() if self.episode else None,
                "relationship_counterpart_ref": self.relationship_counterpart_ref,
                "mission_target_refs": list(self.mission_target_refs),
                "workspace_target_refs": list(self.workspace_target_refs)}


@dataclass(frozen=True)
class MemoryProposalBundleV16:
    base: MemoryProposalBundle
    context_v16_digest: str
    proposals: tuple[FormationProposalV16, ...]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        self.base.validate()
        if self.schema_version != SCHEMA_VERSION:
            raise CognitiveKernelContractError("unsupported v1.6 output schema")
        if (not isinstance(self.context_v16_digest, str) or
                len(self.context_v16_digest) != 64 or
                any(c not in "0123456789abcdef" for c in self.context_v16_digest)):
            raise CognitiveKernelContractError("invalid v1.6 context digest")
        if len(self.base.proposals) != len(self.proposals):
            raise CognitiveKernelContractError("v1.6 proposal count differs from base")
        for old, new in zip(self.base.proposals, self.proposals):
            new.validate()
            if new.base != old:
                raise CognitiveKernelContractError("v1.6 proposal order or base changed")

    def output_record(self) -> dict[str, object]:
        self.validate()
        return {"schema": OUTPUT_SCHEMA_V16,
                "proposals": [proposal.record() for proposal in self.proposals],
                "dispositions": [d.record() for d in self.base.dispositions]}

    def content_digest(self) -> str:
        return canonical_sha256({"context_v16_digest": self.context_v16_digest,
                                 "output": self.output_record(),
                                 "base_bundle": self.base.metadata_record()})


def bundle_v16_from_output(context: FormationContextV16, value: object, *,
                           artifact_sha256: str, inference_run_id: str) -> MemoryProposalBundleV16:
    row = _record(value, {"schema", "proposals", "dispositions"}, "v1.6 output")
    if row["schema"] != OUTPUT_SCHEMA_V16:
        raise CognitiveKernelContractError("unsupported v1.6 output schema")
    proposal_rows = [_record(item, {"proposal", "sensitivity_hint", "episode",
                                    "relationship_counterpart_ref", "mission_target_refs",
                                    "workspace_target_refs"}, "v1.6 proposal")
                     for item in _array(row["proposals"], "v1.6 proposals")]
    base = bundle_from_output(context.base, {
        "schema": OUTPUT_SCHEMA,
        "proposals": [item["proposal"] for item in proposal_rows],
        "dispositions": _array(row["dispositions"], "v1.6 dispositions"),
    }, artifact_sha256=artifact_sha256, inference_run_id=inference_run_id)
    proposals = []
    for old, item in zip(base.proposals, proposal_rows):
        episode = item["episode"]
        if episode is not None:
            episode = _record(episode,
                              {"event_start", "event_end", "participant_refs", "scene_refs"},
                              "episode")
        proposals.append(FormationProposalV16(
            base=old, sensitivity_hint=item["sensitivity_hint"],
            episode=(EpisodeSemantics(
                event_start=episode["event_start"], event_end=episode["event_end"],
                participant_refs=tuple(_array(episode["participant_refs"], "participant_refs")),
                scene_refs=tuple(_array(episode["scene_refs"], "scene_refs")),
            ) if episode is not None else None),
            relationship_counterpart_ref=item["relationship_counterpart_ref"],
            mission_target_refs=tuple(_array(item["mission_target_refs"], "mission_target_refs")),
            workspace_target_refs=tuple(_array(item["workspace_target_refs"], "workspace_target_refs")),
        ))
    bundle = MemoryProposalBundleV16(base, context.content_digest(), tuple(proposals))
    if bundle.output_record() != row:
        raise CognitiveKernelContractError("v1.6 output is not canonical")
    return bundle


def validate_formation_grounding_v16(
    context: FormationContextV16,
    bundle: MemoryProposalBundleV16,
    opened_sources: tuple[tuple[str, bytes], ...],
) -> None:
    """Check v1.5 byte anchors plus all added links and policy floors.

    This is preflight only; gate must re-resolve source and target policies at
    write time, especially after revocation or permission changes.
    """
    context.validate()
    bundle.validate()
    if bundle.context_v16_digest != context.content_digest():
        raise CognitiveKernelContractError("v1.6 context digest differs")
    validate_formation_grounding(context.base, bundle.base, opened_sources)
    policies = {item.ref_id: SENSITIVITY_ORDER.index(item.minimum)
                for item in context.source_sensitivities}
    targets = {item.ref_id: item for item in context.targets}
    for proposal in bundle.proposals:
        base = proposal.base
        if proposal.sensitivity_hint is not None and SENSITIVITY_ORDER.index(
                proposal.sensitivity_hint) < max(policies[ref] for ref in base.evidence_refs):
            raise CognitiveKernelContractError("model sensitivity hint lowers source policy")
        links = []
        if proposal.episode is not None:
            links.extend((ref, "entity") for ref in proposal.episode.participant_refs)
            links.extend((ref, "scene") for ref in proposal.episode.scene_refs)
        if proposal.relationship_counterpart_ref is not None:
            links.append((proposal.relationship_counterpart_ref, "entity"))
        links.extend((ref, "mission") for ref in proposal.mission_target_refs)
        links.extend((ref, "workspace") for ref in proposal.workspace_target_refs)
        for ref_id, kind in links:
            target = targets.get(ref_id)
            if target is None or target.kind != kind:
                raise CognitiveKernelContractError("v1.6 target is unregistered or wrong kind")
            if not set(base.evidence_refs).intersection(target.supporting_evidence_refs):
                raise CognitiveKernelContractError("v1.6 target has no cited supporting evidence")


@dataclass(frozen=True)
class FormationGoldCaseV16:
    """Every dimension has an independent adjudication; absence is explicit."""

    case_id: str
    context: FormationContextV16
    expected: tuple[FormationProposalV16, ...]
    expected_dispositions: tuple[FormationDisposition, ...]
    critical_forbidden: tuple[tuple[str, str, str, str, str], ...]
    adjudicated_dimensions: frozenset[str]
    reviewer_refs: tuple[str, ...]
    opened_sources: tuple[tuple[str, bytes], ...]

    def validate(self) -> None:
        _canonical_id(self.case_id, "case_id")
        if self.adjudicated_dimensions != FULL_ROLE_ADJUDICATION_DIMENSIONS:
            raise CognitiveKernelContractError("v1.6 gold has unadjudicated dimensions")
        _ids(self.reviewer_refs, "reviewer_refs")
        if len(self.reviewer_refs) < 2:
            raise CognitiveKernelContractError("v1.6 gold needs independent reviewer records")
        base_gold = FormationGoldCase(self.case_id, self.context.base,
                                      tuple(p.base for p in self.expected),
                                      self.critical_forbidden,
                                      self.expected_dispositions)
        base_gold.validate()
        expected_base = MemoryProposalBundle(
            scope=self.context.base.scope,
            authority_namespace_id=self.context.base.authority_namespace_id,
            bundle_id=f"gold-{self.case_id}",
            experience_refs=self.context.base.experience_refs,
            context_digest=self.context.base.content_digest(),
            model_artifact_digest="0" * 64, inference_run_id="gold-validation",
            proposals=tuple(p.base for p in self.expected),
            dispositions=self.expected_dispositions,
        )
        validate_formation_grounding_v16(
            self.context, MemoryProposalBundleV16(
                expected_base, self.context.content_digest(), self.expected),
            self.opened_sources)


def _semantic_identity(p: FormationProposalV16) -> str:
    record = p.record()
    record["proposal"].pop("proposal_id")
    # These are artifact-local IDs and uncalibrated numbers, not adjudicated
    # semantic content. Confidence receives a separate calibration evaluation.
    record["proposal"].pop("value_ref")
    record["proposal"].pop("confidence")
    return canonical_sha256(record)


def assess_formation_v16(gold: FormationGoldCaseV16,
                         output: MemoryProposalBundleV16) -> FormationAssessment:
    """Full v1.6 semantic identity, after explicit label-coverage admission."""
    gold.validate()
    validate_formation_grounding_v16(gold.context, output, gold.opened_sources)
    legacy_gold = FormationGoldCase(gold.case_id, gold.context.base,
                                    tuple(p.base for p in gold.expected),
                                    gold.critical_forbidden,
                                    gold.expected_dispositions)
    baseline = assess_formation(legacy_gold, output.base)
    expected = {_semantic_identity(p) for p in gold.expected}
    predicted = {_semantic_identity(p) for p in output.proposals}
    if len(predicted) != len(output.proposals):
        raise CognitiveKernelContractError("duplicate v1.6 semantic output")
    return FormationAssessment(
        gold.case_id, len(expected & predicted), len(predicted - expected),
        len(expected - predicted), baseline.critical_failures,
        baseline.disposition_true_positives,
        baseline.disposition_false_positives,
        baseline.disposition_false_negatives,
    )


def require_full_role_structural_coverage_v16(
    *, corpus_schema: str, split_roster: dict[str, int],
    observed_label_counts: dict[str, dict[str, int]],
    sealed_final_sha256: str | None,
    independent_adjudication_receipt_sha256: str | None,
    authenticated_rights_receipt_sha256: str | None,
) -> None:
    """Reject missing labels, held-out splits or receipt references.

    This is only a structural check of *declared* metadata, never an admission
    to optimize. The data steward must authenticate exact reviewed targets,
    rights, split separation and seal against the bytes independently.
    """
    if corpus_schema != "mfm-full-role-corpus-v1.6":
        raise CognitiveKernelContractError("historical v1.5 corpus lacks full-role labels")
    required = FULL_ROLE_ADJUDICATION_DIMENSIONS
    if set(split_roster) != {"train", "development", "final"}:
        raise CognitiveKernelContractError("full-role corpus needs sealed train/dev/FINAL splits")
    for split, count in split_roster.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CognitiveKernelContractError(f"empty or invalid {split} split")
        labels = observed_label_counts.get(split)
        if (not isinstance(labels, dict) or set(labels) != required or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > count
                    for n in labels.values())):
            raise CognitiveKernelContractError(f"{split} lacks observed full-role label coverage")
    if set(observed_label_counts) != set(split_roster):
        raise CognitiveKernelContractError("full-role coverage has unsupported split")
    for name, digest in (("FINAL seal", sealed_final_sha256),
                         ("independent adjudication", independent_adjudication_receipt_sha256),
                         ("authenticated rights", authenticated_rights_receipt_sha256)):
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(c not in "0123456789abcdef" for c in digest)):
            raise CognitiveKernelContractError(f"full-role admission needs {name} digest")
