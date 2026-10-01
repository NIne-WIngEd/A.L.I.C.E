"""Versioned MFM 1.6 training rows and exact-source target serialization.

This codec can parse author-labelled synthetic rows for CPU experiments. It
does not authenticate a data steward or turn them into independent gold. The
admitted path uses the existing corpus manifest's rights/reviewer receipts and
sealed FINAL metadata; its declarative receipts still need external custody
authentication before a full capability claim.

Unknown dimensions are never serialized as negative supervised targets. The
1.5 curriculum/target format remains separate and is rejected here.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterator

from .canonical import CognitiveKernelContractError, canonical_sha256, require_identifier
from .formation_dataset_admission import CorpusAdmission, _unique_json
from .formation_semantics_v16 import (
    FULL_ROLE_ADJUDICATION_DIMENSIONS, FormationContextV16, MemoryProposalBundleV16,
    OUTPUT_SCHEMA_V16, bundle_v16_from_output, context_v16_from_record,
    validate_formation_grounding_v16,
)


CURRICULUM_SCHEMA_V16 = "mfm-full-role-curriculum-case-v1.6"
TARGET_SCHEMA_V16 = "mfm-formation-target-v1.6"
OBJECTIVE_VERSION_V16 = "mfm-grounded-full-role-objective-v1.6"
ADJUDICATION_STATES = frozenset({"present", "negative", "unknown"})
FULL_ROLE_DIMENSIONS = FULL_ROLE_ADJUDICATION_DIMENSIONS


def _record(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise CognitiveKernelContractError(f"{name} has unsupported or missing fields")
    return value


def _array(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise CognitiveKernelContractError(f"{name} must be an array")
    return value


def _identity(value: object, name: str) -> str:
    identifier = require_identifier(value, name)
    if identifier != value:
        raise CognitiveKernelContractError(f"{name} must be canonical")
    return identifier


def _positive_dimensions(target: MemoryProposalBundleV16) -> set[str]:
    present: set[str] = set()
    for proposal in target.proposals:
        if proposal.sensitivity_hint is not None:
            present.add("sensitivity")
        if proposal.episode is not None:
            present.add("episode")
        if proposal.relationship_counterpart_ref is not None:
            present.add("relationship")
        if proposal.mission_target_refs:
            present.add("mission")
        if proposal.workspace_target_refs:
            present.add("workspace")
        base = proposal.base
        if base.kind in {"correction_request", "deletion_request", "revocation_request"}:
            present.add("correction")
        if (base.domain == "source_person" or base.kind == "source_person_evidence" or
                base.epistemic_status == "source_person_attestation"):
            present.add("source_person")
        if base.kind == "contradiction" or base.contradicts:
            present.add("contradiction")
        if base.kind == "outcome":
            present.add("outcome")
    if any(disposition.action == "abstain" for disposition in target.base.dispositions):
        present.add("abstention")
    return present


@dataclass(frozen=True)
class FormationLearningExampleV16:
    case_id: str
    context: FormationContextV16
    opened_sources: tuple[tuple[str, bytes], ...]
    target: MemoryProposalBundleV16
    adjudications: tuple[tuple[str, str], ...]
    split: str

    def validate(self) -> None:
        _identity(self.case_id, "case_id")
        if self.split not in {"train", "development"}:
            raise CognitiveKernelContractError("FINAL cannot be opened as a learning example")
        if not self.target.base.dispositions:
            raise CognitiveKernelContractError("learning target needs scoped dispositions")
        expected_refs = tuple(ref.ref_id for ref in self.context.base.evidence)
        if tuple(ref for ref, _ in self.opened_sources) != expected_refs:
            raise CognitiveKernelContractError("learning sources must follow context evidence order")
        if any(not isinstance(raw, bytes) for _, raw in self.opened_sources):
            raise CognitiveKernelContractError("learning sources require exact bytes")
        if tuple(sorted(self.adjudications)) != self.adjudications or (
                {name for name, _ in self.adjudications} != FULL_ROLE_DIMENSIONS or
                len(self.adjudications) != len(FULL_ROLE_DIMENSIONS)):
            raise CognitiveKernelContractError("v1.6 adjudications must cover every full-role dimension")
        actual = _positive_dimensions(self.target)
        for name, state in self.adjudications:
            if not isinstance(state, str) or state not in ADJUDICATION_STATES:
                raise CognitiveKernelContractError("unsupported v1.6 adjudication state")
            if state == "present" and name not in actual:
                raise CognitiveKernelContractError(f"{name} is marked present but has no target")
            if state == "negative" and name in actual:
                raise CognitiveKernelContractError(f"{name} is marked negative but has a target")
            if state == "unknown" and name in actual:
                raise CognitiveKernelContractError(f"{name} has an unadjudicated target")
        validate_formation_grounding_v16(self.context, self.target, self.opened_sources)


def model_input_sha256_v16(example: FormationLearningExampleV16) -> str:
    """Hash versioned context plus the exact opened source byte digests."""
    example.validate()
    return canonical_sha256({
        "context": example.context.record(),
        "sources": [{"ref_id": ref_id, "sha256": sha256(raw).hexdigest()}
                    for ref_id, raw in example.opened_sources],
    })


def supervised_output_record_v16(example: FormationLearningExampleV16) -> dict[str, object]:
    """Return a complete v1.6 token target, refusing all unknown dimensions."""
    example.validate()
    if any(state == "unknown" for _, state in example.adjudications):
        raise CognitiveKernelContractError(
            "unknown adjudication cannot become a supervised negative target")
    return example.target.output_record()


def _target_from_record(context: FormationContextV16, value: object,
                        *, case_id: str) -> tuple[MemoryProposalBundleV16, tuple[tuple[str, str], ...]]:
    row = _record(value, {"schema", "proposals", "dispositions", "adjudications"},
                  "v1.6 learning target")
    if row["schema"] != TARGET_SCHEMA_V16:
        raise CognitiveKernelContractError("unsupported v1.6 learning target schema")
    statuses = _record(row["adjudications"], set(FULL_ROLE_DIMENSIONS),
                       "v1.6 adjudications")
    if any(not isinstance(status, str) or status not in ADJUDICATION_STATES
           for status in statuses.values()):
        raise CognitiveKernelContractError("unsupported v1.6 adjudication state")
    output = {"schema": OUTPUT_SCHEMA_V16,
              "proposals": row["proposals"], "dispositions": row["dispositions"]}
    bundle = bundle_v16_from_output(context, output, artifact_sha256="0" * 64,
                                    inference_run_id=f"target-{case_id}")
    return bundle, tuple(sorted(statuses.items()))


def learning_example_v16_from_record(value: object, *,
                                     split: str = "train") -> FormationLearningExampleV16:
    """Parse a v1.6 row without promoting synthetic labels to independent gold."""
    if split not in {"train", "development"}:
        raise CognitiveKernelContractError("FINAL cannot be opened by the trainer")
    if not isinstance(value, dict) or not {"schema", "case_id", "split", "context",
                                           "sources", "target"}.issubset(value) or (
            set(value) - {"schema", "case_id", "split", "context", "sources", "target",
                          "lineage", "status", "authorization_id"}):
        raise CognitiveKernelContractError("v1.6 curriculum row has unsupported or missing fields")
    if value["schema"] != CURRICULUM_SCHEMA_V16 or value["split"] != split:
        raise CognitiveKernelContractError("curriculum row has wrong v1.6 schema or split")
    case_id = _identity(value["case_id"], "case_id")
    if "authorization_id" in value:
        _identity(value["authorization_id"], "authorization_id")
    context = context_v16_from_record(value["context"])
    sources: list[tuple[str, bytes]] = []
    for source_value in _array(value["sources"], "sources"):
        source = _record(source_value, {"ref_id", "content_b64"}, "source")
        ref_id = _identity(source["ref_id"], "source ref_id")
        encoded = source["content_b64"]
        if not isinstance(encoded, str):
            raise CognitiveKernelContractError("source content_b64 must be text")
        try:
            raw = b64decode(encoded, validate=True)
        except (ValueError, Base64Error) as exc:
            raise CognitiveKernelContractError("invalid source base64") from exc
        if b64encode(raw).decode("ascii") != encoded:
            raise CognitiveKernelContractError("noncanonical source base64")
        sources.append((ref_id, raw))
    target, adjudications = _target_from_record(context, value["target"], case_id=case_id)
    example = FormationLearningExampleV16(
        case_id, context, tuple(sources), target, adjudications, split)
    example.validate()
    return example


def _unique_input(example: FormationLearningExampleV16,
                  seen: dict[str, str]) -> None:
    digest = model_input_sha256_v16(example)
    target_digest = canonical_sha256({"output": example.target.output_record(),
                                     "adjudications": list(example.adjudications)})
    previous = seen.get(digest)
    if previous is not None:
        if previous != target_digest:
            raise CognitiveKernelContractError("conflicting targets for identical v1.6 input")
        raise CognitiveKernelContractError("duplicate v1.6 model input")
    seen[digest] = target_digest


def admitted_rows_v16(admission: CorpusAdmission, *, split: str) -> Iterator[FormationLearningExampleV16]:
    """Read admitted train/dev rows after exact byte audit; never open FINAL."""
    if split not in {"train", "development"}:
        raise CognitiveKernelContractError("FINAL is sealed from model training and selection")
    def digests(cases) -> set[str]:
        return {ref.sha256 for case in cases
                for ref in (*case.source_payloads, case.target_payload)}
    if digests(admission.train) & digests(admission.development):
        raise CognitiveKernelContractError("train and development share payload bytes")
    if {case.case_id for case in admission.train} & {
            case.case_id for case in admission.development}:
        raise CognitiveKernelContractError("train and development share a case ID")
    selected = admission.train if split == "train" else admission.development
    # One source may legitimately support multiple cases in this split. The
    # handoff list names each physical payload once, then each case still
    # reopens and checks its own exact ref below. The public handoff API keeps
    # rejecting duplicate caller paths as an accidental optimizer input.
    paths = tuple(dict.fromkeys(item.path for case in selected for item in (
        *case.source_payloads, case.target_payload)))
    admission.audit_handoff(gradient_paths=paths if split == "train" else (),
                            development_paths=paths if split == "development" else ())
    seen: dict[str, str] = {}
    def exact_bytes(ref) -> bytes:
        # The handoff audit and the consumer read are separate operations.
        # Verify the very buffer we parse, even if a file changes between them.
        raw = (admission.root / ref.path).read_bytes()
        if sha256(raw).hexdigest() != ref.sha256:
            raise CognitiveKernelContractError("admitted v1.6 consumed payload changed")
        return raw
    for case in selected:
        target_row = _unique_json(exact_bytes(case.target_payload), "admitted v1.6 target")
        if not isinstance(target_row, dict) or set(target_row) != {
                "schema", "context", "source_ids", "proposals", "dispositions", "adjudications"}:
            raise CognitiveKernelContractError("admitted v1.6 target has unsupported fields")
        if target_row["schema"] != TARGET_SCHEMA_V16:
            raise CognitiveKernelContractError("admitted target has wrong v1.6 schema")
        sources: list[tuple[str, bytes]] = []
        hosts: set[str] = set()
        for source, rights_ref in zip(case.source_payloads, case.rights_receipts, strict=True):
            rights = _unique_json(exact_bytes(rights_ref), "admitted v1.6 rights")
            hosts.add(rights["host_family"])
            sources.append((rights["source_id"], exact_bytes(source)))
        if target_row["source_ids"] != [ref for ref, _ in sources]:
            raise CognitiveKernelContractError("admitted target source list differs")
        row = {"schema": CURRICULUM_SCHEMA_V16, "case_id": case.case_id,
               "split": split, "context": target_row["context"],
               "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode("ascii")}
                           for ref, raw in sources],
               "target": {key: target_row[key] for key in (
                   "schema", "proposals", "dispositions", "adjudications")}}
        example = learning_example_v16_from_record(row, split=split)
        if hosts != {example.context.base.scope.host_instance_id}:
            raise CognitiveKernelContractError("admitted case host differs from context scope")
        _unique_input(example, seen)
        yield example
