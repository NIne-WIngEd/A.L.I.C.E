"""Learned formation I/O, with exact source custody and a supervised objective.

The causal-language backbone implemented here accepts UTF-8 text, code and
structured sources. Other modalities require an actual registered multimodal
processor; a digest, caption, or base64 string is not a substitute for seeing
their content. No output becomes authoritative merely by passing this module.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterator

from .canonical import (
    CognitiveKernelContractError, canonical_json_bytes, canonical_sha256, require_identifier,
    require_sha256, require_text,
)
from .contracts import ProductHostScope
from .formation_contracts import (
    FormationContextPacket, FormationDisposition, FormationEvidenceAnchor,
    FormationEvidenceRef, FormationProposal, MemoryProposalBundle,
    validate_formation_grounding,
)
from .formation_dataset_admission import CorpusAdmission


CURRICULUM_SCHEMA = "mfm-curriculum-case-v1"
TARGET_SCHEMA = "mfm-formation-target-v1"
OUTPUT_SCHEMA = "mfm-formation-output-v1"
OBJECTIVE_VERSION = "mfm-grounded-disposition-objective-v1"
MIXTURE_SCHEMA = "mfm-training-mixture-v1"


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise CognitiveKernelContractError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise CognitiveKernelContractError(f"{name} must be an array")
    return value


def context_from_record(value: object) -> FormationContextPacket:
    row = _mapping(value, "context")
    scope = ProductHostScope.create(**_mapping(row["scope"], "scope"))
    refs = []
    for item in _array(row["evidence"], "context.evidence"):
        ref = _mapping(item, "evidence")
        refs.append(FormationEvidenceRef(
            ref_id=ref["ref_id"], scope=scope,
            authority_namespace_id=ref["authority_namespace_id"],
            content_digest=ref["content_digest"], role=ref["role"],
            modality=ref["modality"], subject_ref=ref.get("subject_ref"),
            speaker_ref=ref.get("speaker_ref"),
            source_item_ref=ref.get("source_item_ref"),
            observed_at=ref.get("observed_at"), recorded_at=ref.get("recorded_at"),
            temporal_granularity=ref.get("temporal_granularity", "instant"),
            duplicate_group_ref=ref.get("duplicate_group_ref"),
            parent_refs=tuple(ref.get("parent_refs", ())),
        ))
    context = FormationContextPacket(
        scope, row["authority_namespace_id"], tuple(row["experience_refs"]),
        tuple(refs), row["schema_version"],
    )
    context.validate()
    if context.metadata_record() != row:
        raise CognitiveKernelContractError("context is not canonical contract metadata")
    return context


def _proposal(value: object) -> FormationProposal:
    row = _mapping(value, "proposal")
    proposal = FormationProposal(
        proposal_id=row["proposal_id"], kind=row["kind"], domain=row["domain"],
        subject_ref=row["subject_ref"], value_ref=row["value_ref"],
        evidence_refs=tuple(row["evidence_refs"]), value_text=row["value_text"],
        anchors=tuple(FormationEvidenceAnchor(**_mapping(a, "anchor"))
                      for a in _array(row["anchors"], "anchors")),
        epistemic_status=row["epistemic_status"], valid_from=row.get("valid_from"),
        valid_to=row.get("valid_to"), confidence=row.get("confidence"),
        uncertainty_ref=row.get("uncertainty_ref"),
        contradicts=tuple(row.get("contradicts", ())),
        target_refs=tuple(row.get("target_refs", ())),
        disposition_scope_ref=row.get("disposition_scope_ref"),
        temporal_granularity=row.get("temporal_granularity", "instant"),
    )
    proposal.validate()
    if proposal.record() != row:
        raise CognitiveKernelContractError("proposal is not canonical contract metadata")
    return proposal


def _disposition(value: object) -> FormationDisposition:
    row = _mapping(value, "disposition")
    disposition = FormationDisposition(
        scope_ref=row["scope_ref"], action=row["action"],
        evidence_refs=tuple(row["evidence_refs"]),
        target_refs=tuple(row.get("target_refs", ())),
    )
    disposition.validate()
    if disposition.record() != row:
        raise CognitiveKernelContractError("disposition is not canonical contract metadata")
    return disposition


def bundle_from_output(
    context: FormationContextPacket, output: object, *,
    artifact_sha256: str, inference_run_id: str,
) -> MemoryProposalBundle:
    row = _mapping(output, "model output")
    if set(row) != {"schema", "proposals", "dispositions"} or row["schema"] != OUTPUT_SCHEMA:
        raise CognitiveKernelContractError("model output schema is invalid")
    bundle = MemoryProposalBundle(
        scope=context.scope, authority_namespace_id=context.authority_namespace_id,
        bundle_id=f"formation-{require_identifier(inference_run_id, 'inference_run_id')}",
        experience_refs=context.experience_refs, context_digest=context.content_digest(),
        model_artifact_digest=require_sha256(artifact_sha256, "artifact_sha256"),
        inference_run_id=inference_run_id,
        proposals=tuple(_proposal(x) for x in _array(row["proposals"], "proposals")),
        dispositions=tuple(_disposition(x) for x in _array(row["dispositions"], "dispositions")),
    )
    bundle.validate()
    return bundle


def output_record(bundle: MemoryProposalBundle) -> dict[str, object]:
    if not isinstance(bundle, MemoryProposalBundle):
        raise CognitiveKernelContractError("v1.5 output serializer only accepts v1.5 formation records")
    bundle.validate()
    return {"schema": OUTPUT_SCHEMA,
            "proposals": [p.record() for p in bundle.proposals],
            "dispositions": [d.record() for d in bundle.dispositions]}


@dataclass(frozen=True)
class FormationLearningExample:
    case_id: str
    context: FormationContextPacket
    opened_sources: tuple[tuple[str, bytes], ...]
    target: MemoryProposalBundle

    def validate(self) -> None:
        require_identifier(self.case_id, "case_id")
        if not self.target.dispositions:
            raise CognitiveKernelContractError("learning target needs scoped dispositions")
        if tuple(ref for ref, _ in self.opened_sources) != tuple(
                ref.ref_id for ref in self.context.evidence):
            raise CognitiveKernelContractError("learning sources must follow context evidence order")
        validate_formation_grounding(self.context, self.target, self.opened_sources)


def model_input_sha256(example: FormationLearningExample) -> str:
    """Fingerprint the actual context and exact opened sources, excluding labels.

    The same metadata and byte streams render the same model input. Source
    payloads are represented by their exact SHA-256, already verified against
    the context; source order and IDs remain part of the fingerprint.
    """
    example.validate()
    return canonical_sha256({
        "context": example.context.metadata_record(),
        "sources": [{"ref_id": ref_id, "sha256": sha256(raw).hexdigest()}
                    for ref_id, raw in example.opened_sources],
    })


def _check_unique_input(example: FormationLearningExample,
                        seen: dict[str, str]) -> None:
    digest = model_input_sha256(example)
    target_digest = canonical_sha256(output_record(example.target))
    previous = seen.get(digest)
    if previous is not None:
        if previous != target_digest:
            raise CognitiveKernelContractError(
                "conflicting targets for identical formation model input")
        raise CognitiveKernelContractError("duplicate formation model input")
    seen[digest] = target_digest


def learning_example_from_record(value: object, *, split: str = "train") -> FormationLearningExample:
    """Parse a private, owner-authorized curriculum row, never a public gold fixture."""
    row = _mapping(value, "curriculum row")
    if row.get("schema") != CURRICULUM_SCHEMA or row.get("split", "train") != split:
        raise CognitiveKernelContractError("curriculum row has wrong schema or split")
    if split not in {"train", "development"}:
        raise CognitiveKernelContractError("FINAL cannot be opened by the trainer")
    case_id = require_identifier(row["case_id"], "case_id")
    if case_id != row["case_id"]:
        raise CognitiveKernelContractError("case_id must be canonical")
    context = context_from_record(row["context"])
    opened: list[tuple[str, bytes]] = []
    for source_value in _array(row["sources"], "sources"):
        source = _mapping(source_value, "source")
        if set(source) != {"ref_id", "content_b64"}:
            raise CognitiveKernelContractError("source requires ref_id and content_b64")
        try:
            content = b64decode(source["content_b64"], validate=True)
        except (TypeError, ValueError, Base64Error) as exc:
            raise CognitiveKernelContractError("invalid source base64") from exc
        opened.append((source["ref_id"], content))
    target = _mapping(row["target"], "target")
    if target.get("schema") != TARGET_SCHEMA:
        raise CognitiveKernelContractError("unsupported formation target schema")
    output = {"schema": OUTPUT_SCHEMA, "proposals": target["proposals"],
              "dispositions": target["dispositions"]}
    bundle = bundle_from_output(context, output, artifact_sha256="0" * 64,
                                inference_run_id=f"target-{case_id}")
    example = FormationLearningExample(case_id, context, tuple(opened), bundle)
    example.validate()
    return example


def curriculum_rows(path: str | Path, *, expected_sha256: str,
                    owner_authorization_ref: str,
                    expected_generator_family: str | None = None) -> Iterator[FormationLearningExample]:
    """Require a frozen owner-authorized training curriculum; this is not a qualification."""
    require_text(owner_authorization_ref, "owner_authorization_ref", maximum=128)
    if not owner_authorization_ref or not owner_authorization_ref.strip():
        raise CognitiveKernelContractError("curriculum needs owner authorization")
    actual = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(chunk)
    if actual.hexdigest() != require_sha256(expected_sha256, "expected_sha256"):
        raise CognitiveKernelContractError("curriculum bytes differ from frozen digest")
    count = 0
    seen_inputs: dict[str, str] = {}
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("authorization_id") != owner_authorization_ref:
                raise CognitiveKernelContractError("curriculum row lacks matching owner authorization")
            if (expected_generator_family is not None and
                    row.get("generator_family") != expected_generator_family):
                raise CognitiveKernelContractError("curriculum row generator differs from manifest")
            example = learning_example_from_record(row)
            _check_unique_input(example, seen_inputs)
            count += 1
            yield example
    if not count:
        raise CognitiveKernelContractError("curriculum is empty")


def mixture_rows(manifest_path: str | Path, *, expected_sha256: str,
                 owner_authorization_ref: str) -> Iterator[FormationLearningExample]:
    """Freeze each constituent and reject cross-corpus case identity collisions."""
    file = Path(manifest_path)
    raw = file.read_bytes()
    if sha256(raw).hexdigest() != require_sha256(expected_sha256, "expected_sha256"):
        raise CognitiveKernelContractError("mixture manifest differs from frozen digest")
    manifest = _mapping(json.loads(raw), "mixture manifest")
    if manifest.get("schema") != MIXTURE_SCHEMA:
        raise CognitiveKernelContractError("unsupported training mixture schema")
    require_identifier(manifest.get("corpus_id"), "corpus_id")
    inputs = _array(manifest.get("inputs"), "mixture inputs")
    if len(inputs) < 2:
        raise CognitiveKernelContractError("training mixture needs two or more frozen inputs")
    base = file.parent.resolve()
    seen_paths: set[Path] = set()
    seen_cases: set[str] = set()
    seen_inputs: dict[str, str] = {}
    for item in inputs:
        entry = _mapping(item, "mixture input")
        name = entry.get("path")
        if (not isinstance(name, str) or not name or "\\" in name or
                any(part in {"", ".", ".."} for part in name.split("/"))):
            raise CognitiveKernelContractError("mixture input path escapes its root")
        path = (base / name).resolve()
        if not path.is_relative_to(base) or path in seen_paths:
            raise CognitiveKernelContractError("mixture input path escapes or repeats")
        seen_paths.add(path)
        if entry.get("authorization_id") != owner_authorization_ref:
            raise CognitiveKernelContractError("mixture authorization differs from owner instruction")
        generator_family = require_identifier(entry.get("generator_family"), "generator_family")
        if not isinstance(entry.get("case_count"), int) or entry["case_count"] < 1:
            raise CognitiveKernelContractError("mixture case count is invalid")
        count = 0
        for example in curriculum_rows(path, expected_sha256=entry["sha256"],
                                       owner_authorization_ref=owner_authorization_ref,
                                       expected_generator_family=generator_family):
            if example.case_id in seen_cases:
                raise CognitiveKernelContractError("case ID repeats across training inputs")
            seen_cases.add(example.case_id)
            _check_unique_input(example, seen_inputs)
            count += 1
            yield example
        if count != entry["case_count"]:
            raise CognitiveKernelContractError("mixture input count differs from frozen manifest")


def admitted_rows(admission: CorpusAdmission, *, split: str) -> Iterator[FormationLearningExample]:
    """Read only train or development bytes after admission's exact handoff audit."""
    if split not in {"train", "development"}:
        raise CognitiveKernelContractError("FINAL is sealed from model training and selection")
    selected = admission.train if split == "train" else admission.development
    paths = tuple(item.path for case in selected for item in (*case.source_payloads, case.target_payload))
    admission.audit_handoff(gradient_paths=paths if split == "train" else (),
                            development_paths=paths if split == "development" else ())
    seen_inputs: dict[str, str] = {}
    for case in selected:
        target = json.loads((admission.root / case.target_payload.path).read_bytes())
        row = _mapping(target, "admitted target")
        if row.get("schema") != TARGET_SCHEMA:
            raise CognitiveKernelContractError("admitted target has wrong schema")
        sources = []
        declared_hosts = set()
        for source, rights_ref in zip(case.source_payloads, case.rights_receipts, strict=True):
            # Rights receipts bind source ID to admitted bytes and host. The
            # target must use those IDs in the identical order.
            rights = json.loads((admission.root / rights_ref.path).read_bytes())
            declared_hosts.add(rights["host_family"])
            sources.append((rights["source_id"], (admission.root / source.path).read_bytes()))
        source_ids = _array(row.get("source_ids"), "target.source_ids")
        if source_ids != [ref for ref, _ in sources]:
            raise CognitiveKernelContractError("admitted target source list differs")
        record = {"schema": CURRICULUM_SCHEMA, "case_id": case.case_id,
                  "split": split, "context": row["context"],
                  "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode("ascii")}
                              for ref, raw in sources],
                  "target": {"schema": TARGET_SCHEMA, "proposals": row["proposals"],
                             "dispositions": row["dispositions"]}}
        example = learning_example_from_record(record, split=split)
        if declared_hosts != {example.context.scope.host_instance_id}:
            raise CognitiveKernelContractError("admitted case host differs from context scope")
        _check_unique_input(example, seen_inputs)
        yield example


SYSTEM_INSTRUCTION = (
    "Form source-grounded memory proposals for the scoped host. Distinguish host, "
    "source person, outside claims, and assistant self. Separate observation, "
    "inference, prediction, hypothetical, uncertainty, and correction. Preserve "
    "event time and validity intervals. Anchor each proposal to exact source "
    "bytes. Give independent scoped propose/defer/retain_raw/abstain decisions; "
    "carry deletion and correction target references. Never claim that a "
    "requested deletion was already executed. Return one JSON object exactly "
    "matching the output schema with proposals and dispositions; no commentary."
)


def prompt_for_text_backbone(context: FormationContextPacket,
                             opened_sources: tuple[tuple[str, bytes], ...]) -> str:
    context.validate()
    sources = dict(opened_sources)
    if len(sources) != len(opened_sources) or set(sources) != {r.ref_id for r in context.evidence}:
        raise CognitiveKernelContractError("prompt requires every exact source")
    rendered = []
    for ref in context.evidence:
        raw = sources[ref.ref_id]
        if sha256(raw).hexdigest() != ref.content_digest:
            raise CognitiveKernelContractError("prompt source digest mismatch")
        if ref.modality not in {"text", "code", "structured"}:
            raise CognitiveKernelContractError("text backbone cannot consume non-text source")
        try:
            content = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CognitiveKernelContractError("text source is not UTF-8") from exc
        rendered.append({"ref_id": ref.ref_id, "content": content})
    body = canonical_json_bytes({"objective": OBJECTIVE_VERSION,
                                 "context": context.metadata_record(),
                                 "opened_sources": rendered}).decode("utf-8")
    return body


def chat_messages_for_text_backbone(body: str) -> list[dict[str, str]]:
    """The tokenizer owns role delimiters and end-of-turn tokens.

    One user message works across templates that have no separate system role.
    Train and inference both call this function before apply_chat_template.
    """
    return [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{body}"}]


@dataclass
class PretrainedFormationCandidate:
    """Inference adapter over a real learned generator with strict output binding."""

    artifact_sha256: str
    generate: Callable[[str], str]
    inference_run_id: str

    def infer(self, *, context: FormationContextPacket,
              opened_sources: tuple[tuple[str, bytes], ...]) -> MemoryProposalBundle:
        prompt = prompt_for_text_backbone(context, opened_sources)
        generated = self.generate(prompt)
        if not isinstance(generated, str):
            raise CognitiveKernelContractError("learned model did not return text")
        try:
            output = json.loads(generated)
        except (ValueError, TypeError) as exc:
            raise CognitiveKernelContractError("learned formation output is not exact JSON") from exc
        bundle = bundle_from_output(context, output, artifact_sha256=self.artifact_sha256,
                                    inference_run_id=self.inference_run_id)
        validate_formation_grounding(context, bundle, opened_sources)
        return bundle
