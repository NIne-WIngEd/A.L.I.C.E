#!/usr/bin/env python3
"""Prepare and convert bounded, fictional MFM v1.6 teacher excerpts.

The private source and teacher answer are separate immutable inputs. Exact
quotation anchors and model-contract checks are mechanical; neither makes the
teacher's interpretation true, authenticates rights, or admits a training row.
No private source or answer belongs in this repository.
"""

from __future__ import annotations

import argparse
from base64 import b64decode, b64encode
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FormationContextPacket, FormationEvidenceRef,
)
from cognitive_kernel.formation_learning_v16 import (
    CURRICULUM_SCHEMA_V16, FULL_ROLE_DIMENSIONS, TARGET_SCHEMA_V16,
    learning_example_v16_from_record, supervised_output_record_v16,
)
from cognitive_kernel.formation_semantics_v16 import (
    FormationContextV16, RegisteredFormationTarget, RegisteredSourceSensitivity,
)
from scripts.mfm.build_v16_source_review_packets import _canonical, _load
from scripts.mfm.prepare_v16_annotation_drafts import write_private_new


SOURCE_SCHEMA = "mfm-v16-fictional-role-excerpts-v1"
VIEW_SCHEMA = "mfm-v16-fictional-role-teacher-view-v1"
RESPONSE_SCHEMA = "mfm-v16-fictional-role-teacher-response-v1"
PROMPT = ROOT / "scripts/mfm/prompts/v16_role_excerpt_teacher_v1.txt"
AUTHORIZATION = "owner-directed-mfm-v16-synthetic"


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _raw(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _exact(value: object, names: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{label} has missing or extra fields")
    return value


def _private_input(path: Path) -> None:
    if path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("source and teacher answers belong outside Git")


def source_view(source: dict) -> dict:
    """Expose only the intentionally selected source bytes and registrations."""
    _exact(source, {"schema", "case_id", "host_id", "self_id", "authority_id",
                    "sources", "targets", "rights_status", "training_admitted"},
           "private source envelope")
    if (source["schema"] != SOURCE_SCHEMA or source["rights_status"] != "unverified" or
            source["training_admitted"] is not False):
        raise ValueError("source requires unverified, unadmitted provenance")
    if not source["sources"] or not isinstance(source["sources"], list):
        raise ValueError("source view needs bounded selected evidence")
    if len(source["sources"]) > 4:
        raise ValueError("source view is not a bounded excerpt")
    for item in source["sources"]:
        _exact(item, {"ref_id", "role", "modality", "subject_ref", "speaker_ref",
                      "source_item_ref", "observed_at", "recorded_at",
                      "temporal_granularity", "minimum_sensitivity", "content_b64"},
               "source item")
        raw = b64decode(item["content_b64"], validate=True)
        if not raw or b64encode(raw).decode("ascii") != item["content_b64"]:
            raise ValueError("source bytes need nonempty canonical base64")
        if item["role"] not in {"historical_experience", "assistant_self_event",
                                "tool_or_action_observation", "outside_source",
                                "hypothetical_training"}:
            raise ValueError("unverified fictional excerpt cannot claim an authenticated role")
        if item["role"] == "assistant_self_event" and (
                item["subject_ref"] != source["self_id"] or
                item["speaker_ref"] != source["self_id"]):
            raise ValueError("assistant-self event differs from registered self")
        if item["role"] == "historical_experience" and (
                item["subject_ref"] != source["host_id"] or
                item["speaker_ref"] != source["host_id"]):
            raise ValueError("fictional historical report differs from registered host")
    if not isinstance(source["targets"], list):
        raise ValueError("target registrations must be a list")
    # This source-only metadata is intentionally visible to the teacher.
    return {"schema": VIEW_SCHEMA, "case_id": source["case_id"],
            "host_id": source["host_id"], "self_id": source["self_id"],
            "authority_id": source["authority_id"],
            "rights_status": "unverified", "training_admitted": False,
            "sources": source["sources"], "targets": source["targets"]}


def rendered_prompt(view_raw: bytes) -> bytes:
    return PROMPT.read_bytes().rstrip() + b"\n\nSOURCE_VIEW_JSONL:\n" + view_raw


def _quote_anchor(ref_id: str, raw: bytes, anchor: dict, modality: str) -> dict:
    """Resolve a unique exact UTF-8 excerpt, never a plausible substring."""
    _exact(anchor, {"ref_id", "quote", "locator"}, "teacher anchor")
    if anchor["ref_id"] != ref_id:
        raise ValueError("anchor cites a different source")
    if modality in {"text", "structured", "code"}:
        if anchor["locator"] is not None or not isinstance(anchor["quote"], str):
            raise ValueError("text needs one exact quotation")
        quote = anchor["quote"].encode("utf-8")
        if not quote or raw.count(quote) != 1:
            raise ValueError("teacher quote is absent or ambiguous in original bytes")
        start = raw.index(quote)
        return {"ref_id": ref_id, "start_byte": start,
                "end_byte": start + len(quote), "locator": None}
    # This excerpt converter has no media-native range/semantic verifier.
    # Mere locator syntax would falsely imply checked image/audio grounding.
    raise ValueError("non-text proposal needs a separate modality-aware verifier")


def convert(source: dict, response: dict, *, source_sha256: str,
            response_sha256: str, prompt_sha256: str) -> tuple[dict, dict]:
    view = source_view(source)
    view_raw = _raw(view)
    _exact(response, {"schema", "case_id", "source_view_sha256",
                      "rendered_prompt_sha256", "proposals", "dispositions",
                      "adjudications"}, "teacher response")
    if (response["schema"] != RESPONSE_SCHEMA or
            response["case_id"] != source["case_id"] or
            response["source_view_sha256"] != _digest(view_raw) or
            response["rendered_prompt_sha256"] != prompt_sha256):
        raise ValueError("teacher response does not bind to the exact source view and prompt")
    sources = source["sources"]
    if len({item["ref_id"] for item in sources}) != len(sources):
        raise ValueError("source references are not unique")
    scope = ProductHostScope.create(product_id="alice",
                                    host_instance_id=source["host_id"],
                                    schema_version="1.0.0",
                                    encryption_domain="mfm-fictional-teacher")
    evidence = []
    sensitivities = []
    opened = []
    for item in sources:
        raw = b64decode(item["content_b64"], validate=True)
        evidence.append(FormationEvidenceRef(
            ref_id=item["ref_id"], scope=scope,
            authority_namespace_id=source["authority_id"],
            content_digest=_digest(raw), role=item["role"],
            modality=item["modality"], subject_ref=item["subject_ref"],
            speaker_ref=item["speaker_ref"], source_item_ref=item["source_item_ref"],
            observed_at=item["observed_at"], recorded_at=item["recorded_at"],
            temporal_granularity=item["temporal_granularity"]))
        sensitivities.append(RegisteredSourceSensitivity(
            item["ref_id"], item["minimum_sensitivity"]))
        opened.append((item["ref_id"], raw))
    targets = []
    for item in source["targets"]:
        _exact(item, {"ref_id", "kind", "supporting_evidence_refs"},
               "synthetic target registration")
        targets.append(RegisteredFormationTarget(
            item["ref_id"], item["kind"], scope, source["authority_id"],
            tuple(item["supporting_evidence_refs"])))
    context = FormationContextV16(
        FormationContextPacket(scope, source["authority_id"],
                               tuple(item["ref_id"] for item in sources),
                               tuple(evidence)), tuple(sensitivities), tuple(targets))
    context.validate()
    raw_by_ref = dict(opened)
    modality_by_ref = {item["ref_id"]: item["modality"] for item in sources}
    proposals = []
    for item in response["proposals"]:
        _exact(item, {"proposal", "sensitivity_hint", "episode",
                      "relationship_counterpart_ref", "mission_target_refs",
                      "workspace_target_refs"}, "teacher proposal")
        base = dict(item["proposal"])
        if set(base) != {"proposal_id", "kind", "domain", "subject_ref", "value_ref",
                        "evidence_refs", "value_text", "anchors", "epistemic_status",
                        "valid_from", "valid_to", "temporal_granularity",
                        "confidence", "uncertainty_ref", "contradicts", "target_refs",
                        "disposition_scope_ref"}:
            raise ValueError("teacher base proposal has unsupported fields")
        anchors = []
        for anchor in base["anchors"]:
            ref = anchor["ref_id"]
            if ref not in raw_by_ref:
                raise ValueError("teacher cites an unavailable source")
            anchors.append(_quote_anchor(ref, raw_by_ref[ref], anchor,
                                         modality_by_ref[ref]))
        base["anchors"] = anchors
        proposals.append({**item, "proposal": base})
    states = response["adjudications"]
    if not isinstance(states, dict) or set(states) != FULL_ROLE_DIMENSIONS or \
            set(states.values()) - {"present", "negative"}:
        raise ValueError("teacher did not review every full-role dimension within scope")
    case = {"schema": CURRICULUM_SCHEMA_V16, "case_id": source["case_id"],
            "split": "train", "authorization_id": AUTHORIZATION,
            "context": context.record(),
            "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode("ascii")}
                        for ref, raw in opened],
            "target": {"schema": TARGET_SCHEMA_V16, "proposals": proposals,
                       "dispositions": response["dispositions"],
                       "adjudications": states},
            "lineage": {"source_family": "owner-directed-fictional-role-excerpts",
                        "generator_family": "openai-codex-teacher-2026-10-01",
                        "host_family": source["host_id"],
                        "parent_source_sha256": source_sha256},
            "status": {"training_admitted": False, "rights": "unverified",
                       "review": "single-teacher-unreviewed",
                       "qualification": "source-grounded-candidate-only"}}
    example = learning_example_v16_from_record(case)
    supervised_output_record_v16(example)
    provenance = {"schema": "mfm-v16-fictional-role-teacher-provenance-v1",
                  "case_id": source["case_id"], "training_admitted": False,
                  "source_sha256": source_sha256,
                  "source_view_sha256": _digest(view_raw),
                  "rendered_prompt_sha256": prompt_sha256,
                  "teacher_response_sha256": response_sha256,
                  "prompt_template_sha256": _digest(PROMPT.read_bytes()),
                  "converter_sha256": _digest(Path(__file__).read_bytes()),
                  "case_sha256": _digest(_raw(case)),
                  "semantic_entailment_reviewed_independently": False,
                  "rights_status": "unverified"}
    return case, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--view", type=Path, required=True)
    prepare.add_argument("--prompt", type=Path, required=True)
    converted = commands.add_parser("convert")
    for option in ("source", "view", "prompt", "response", "case", "provenance"):
        converted.add_argument("--" + option, type=Path, required=True)
    for option in ("source-sha256", "view-sha256", "prompt-sha256", "response-sha256"):
        converted.add_argument("--" + option, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        _private_input(args.source)
        source = _load(args.source.read_bytes())
        view_raw = _raw(source_view(source))
        prompt_raw = rendered_prompt(view_raw)
        write_private_new(args.view, view_raw)
        write_private_new(args.prompt, prompt_raw)
        print(json.dumps({"source_sha256": _digest(args.source.read_bytes()),
                          "view_sha256": _digest(view_raw),
                          "prompt_sha256": _digest(prompt_raw),
                          "training_admitted": False}, sort_keys=True))
        return
    files = ((args.source, args.source_sha256), (args.view, args.view_sha256),
             (args.prompt, args.prompt_sha256), (args.response, args.response_sha256))
    for path, expected in files:
        _private_input(path)
        if _digest(path.read_bytes()) != expected:
            raise ValueError("source, prompt or response differs from external SHA-256 pin")
    source = _load(args.source.read_bytes())
    view_raw = _raw(source_view(source))
    if view_raw != args.view.read_bytes() or \
            rendered_prompt(view_raw) != args.prompt.read_bytes():
        raise ValueError("teacher view or prompt differs from exact source and template")
    case, provenance = convert(source, _load(args.response.read_bytes()),
                               source_sha256=args.source_sha256,
                               response_sha256=args.response_sha256,
                               prompt_sha256=args.prompt_sha256)
    write_private_new(args.case, _raw(case))
    write_private_new(args.provenance, _raw(provenance))
    print(json.dumps({"case_id": case["case_id"],
                      "case_sha256": provenance["case_sha256"],
                      "provenance_sha256": _digest(_raw(provenance)),
                      "training_admitted": False,
                      "v16_parser_and_grounding": True}, sort_keys=True))


if __name__ == "__main__":
    main()
