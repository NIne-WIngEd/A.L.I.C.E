"""Compile human-authored fictional histories into source-bound MFM gold.

The corpus is public method/evaluation material. It cannot establish that a
private person's history is true or qualify a trained formation model by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .canonical import CognitiveKernelContractError, normalize_timestamp, require_identifier
from .contracts import ProductHostScope
from .formation_contracts import (
    FormationContextPacket, FormationEvidenceRef, FormationProposal,
    MemoryProposalBundle, validate_formation_binding,
)
from .formation_evaluation import FormationGoldCase


@dataclass(frozen=True)
class CompiledFormationCase:
    gold: FormationGoldCase
    split: str
    host_family: str
    source_family: str
    generator_family: str
    texts: tuple[tuple[str, str], ...]
    values: tuple[tuple[str, str], ...]
    reason: str


def _proposal(row: dict[str, object]) -> FormationProposal:
    return FormationProposal(
        proposal_id=row["proposal_id"], kind=row["kind"], domain=row["domain"],
        subject_ref=row["subject_ref"], value_ref=row["value_ref"],
        evidence_refs=tuple(row["evidence_refs"]),
        epistemic_status=row["epistemic_status"],
        valid_from=(normalize_timestamp(row["valid_from"]) if row.get("valid_from") else None),
        valid_to=(normalize_timestamp(row["valid_to"]) if row.get("valid_to") else None),
    )


def compile_formation_case(row: dict[str, object]) -> CompiledFormationCase:
    case_id = require_identifier(row["case_id"], "case_id")
    host = require_identifier(row["host_family"], "host_family")
    split = require_identifier(row["split"], "split")
    source_family = require_identifier(row["source_family"], "source_family")
    generator = require_identifier(row["generator_family"], "generator_family")
    if split not in {"train", "development", "challenge"}:
        raise CognitiveKernelContractError("unknown gold split")
    scope = ProductHostScope.create(
        product_id="alice", host_instance_id=host,
        schema_version="1.0.0", encryption_domain=f"fictional-{host}",
    )
    namespace = f"gold-{host}"
    texts: list[tuple[str, str]] = []
    refs: list[FormationEvidenceRef] = []
    for source in row["sources"]:
        ref_id = require_identifier(source["ref_id"], "ref_id")
        text = source["text"]
        if not isinstance(text, str) or not text.strip():
            raise CognitiveKernelContractError("gold source text is empty")
        texts.append((ref_id, text))
        refs.append(FormationEvidenceRef(
            ref_id=ref_id, scope=scope, authority_namespace_id=namespace,
            content_digest=sha256(text.encode("utf-8")).hexdigest(),
            role=source["role"], modality=source.get("modality", "text"),
            subject_ref=source.get("subject_ref"), speaker_ref=source.get("speaker_ref"),
            source_item_ref=source.get("source_item_ref", ref_id),
            observed_at=normalize_timestamp(source["observed_at"]),
            recorded_at=normalize_timestamp(source.get("recorded_at", source["observed_at"])),
            duplicate_group_ref=source.get("duplicate_group_ref", ref_id),
            parent_refs=tuple(source.get("parent_refs", ())),
        ))
    if len({ref.ref_id for ref in refs}) != len(refs):
        raise CognitiveKernelContractError("duplicate source in gold case")
    context = FormationContextPacket(
        scope=scope, authority_namespace_id=namespace,
        experience_refs=tuple(row["experience_refs"]), evidence=tuple(refs),
    )
    context.validate()
    values = row["values"]
    if not isinstance(values, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in values.items()
    ):
        raise CognitiveKernelContractError("gold value map is invalid")
    expected = tuple(_proposal(p) for p in row["expected"])
    if not all(p.value_ref in values for p in expected):
        raise CognitiveKernelContractError("gold proposal value has no declared meaning")
    forbidden = tuple(tuple(label) for label in row["critical_forbidden"])
    if not all(len(label) == 5 and label[3] in values for label in forbidden):
        raise CognitiveKernelContractError("critical forbidden value has no declared meaning")
    gold = FormationGoldCase(case_id, context, expected, forbidden)
    gold.validate()
    # The same provenance gate used at inference rejects mislabeled gold.
    validate_formation_binding(context, MemoryProposalBundle(
        scope=scope, authority_namespace_id=namespace,
        bundle_id=f"gold-{case_id}", experience_refs=context.experience_refs,
        context_digest=context.content_digest(), model_artifact_digest="0" * 64,
        inference_run_id="gold-validation", proposals=expected,
    ))
    reason = row["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise CognitiveKernelContractError("gold needs checkable label reason")
    return CompiledFormationCase(gold, split, host, source_family, generator,
                                 tuple(texts), tuple(sorted(values.items())), reason)


def load_formation_gold(path: str | Path, *, expected_sha256: str | None = None) -> tuple[CompiledFormationCase, ...]:
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise CognitiveKernelContractError("gold corpus bytes differ from frozen digest")
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise CognitiveKernelContractError("gold corpus is empty")
    cases = tuple(compile_formation_case(row) for row in rows)
    if len({case.gold.case_id for case in cases}) != len(cases):
        raise CognitiveKernelContractError("duplicate gold case_id")
    for field in ("host_family", "source_family"):
        split_by_family: dict[str, str] = {}
        for case in cases:
            family = getattr(case, field)
            if family in split_by_family and split_by_family[family] != case.split:
                raise CognitiveKernelContractError(f"{field} leaks across splits")
            split_by_family[family] = case.split
    # Identical evidence items cannot be counted as independent support across
    # the split boundary, even when exported under a different source ID.
    split_by_content: dict[str, str] = {}
    for case in cases:
        for _, text in case.texts:
            digest = sha256(text.encode("utf-8")).hexdigest()
            if digest in split_by_content and split_by_content[digest] != case.split:
                raise CognitiveKernelContractError("source text leaks across splits")
            split_by_content[digest] = case.split
    return cases


def load_frozen_formation_gold(manifest_path: str | Path) -> tuple[CompiledFormationCase, ...]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema") != "mfm-fictional-gold-manifest-v1":
        raise CognitiveKernelContractError("unsupported gold manifest")
    name = manifest.get("corpus")
    if not isinstance(name, str) or Path(name).name != name:
        raise CognitiveKernelContractError("gold corpus must be adjacent to manifest")
    cases = load_formation_gold(manifest_file.parent / name,
                                expected_sha256=manifest["sha256"])
    if len(cases) != manifest["case_count"]:
        raise CognitiveKernelContractError("gold case count differs from freeze")
    split_hosts: dict[str, set[str]] = {}
    for case in cases:
        split_hosts.setdefault(case.split, set()).add(case.host_family)
    if {key: sorted(value) for key, value in split_hosts.items()} != manifest["splits"]:
        raise CognitiveKernelContractError("gold split roster differs from freeze")
    return cases
