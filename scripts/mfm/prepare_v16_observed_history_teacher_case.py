#!/usr/bin/env python3
"""Prepare and convert one exact, as-of public-history MFM teacher view.

The source envelope names already selected source bytes outside Git. The
converter checks hashes and unique text anchors, then verifies a v1.6 learning
candidate. Rights, speaker identity, source selection, semantic entailment and
split independence are separate steward/reviewer decisions. No meeting speaker
is silently promoted to the Fable owner or Alice's assistant self.
"""

from __future__ import annotations

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.canonical import require_identifier
from cognitive_kernel.formation_dataset_admission import _unique_json
from cognitive_kernel.formation_contracts import FormationContextPacket, FormationEvidenceRef
from cognitive_kernel.formation_learning_v16 import (
    CURRICULUM_SCHEMA_V16, FULL_ROLE_DIMENSIONS, TARGET_SCHEMA_V16,
    learning_example_v16_from_record, supervised_output_record_v16,
)
from cognitive_kernel.formation_semantics_v16 import (
    FormationContextV16, RegisteredFormationTarget, RegisteredSourceSensitivity,
)
from scripts.mfm.prepare_v16_annotation_drafts import write_private_new
from scripts.mfm.prepare_v16_role_excerpt_teacher_cases import _exact, _quote_anchor, _raw


ENVELOPE_SCHEMA = "mfm-v16-observed-history-envelope-v1"
VIEW_SCHEMA = "mfm-v16-observed-history-teacher-view-v1"
RESPONSE_SCHEMA = "mfm-v16-observed-history-teacher-response-v1"
PROMPT = ROOT / "scripts/mfm/prompts/v16_observed_history_teacher_v1.txt"


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _outside_repo(path: Path) -> None:
    if path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("source and teacher payloads belong outside Git")


def _load(path: Path, expected_sha256: str) -> dict:
    _outside_repo(path)
    raw = path.read_bytes()
    if _digest(raw) != expected_sha256:
        raise ValueError(f"source or teacher bytes differ: {path}")
    return _unique_json(raw, "source or teacher JSON")


def _sources(envelope: dict, envelope_path: Path) -> list[tuple[dict, bytes]]:
    _exact(envelope, {"schema", "case_id", "split", "authorization_id", "host_id",
                  "self_id", "authority_id", "source_family", "as_of",
                  "source_window", "sources", "targets", "training_admitted", "rights_status"},
           "source envelope")
    if (envelope["schema"] != ENVELOPE_SCHEMA or
            envelope["training_admitted"] is not False or
            envelope["rights_status"] != "unverified" or
            envelope["split"] not in {"train", "development"} or
            not isinstance(envelope["sources"], list) or not envelope["sources"]):
        raise ValueError("source envelope needs bounded, unadmitted exact evidence")
    if len(envelope["sources"]) > 32:
        raise ValueError("source view is too large; select an as-of window")
    cutoff = envelope["as_of"]
    _exact(cutoff, {"timeline_ref", "cutoff_seconds"}, "relative as-of")
    if (require_identifier(cutoff["timeline_ref"], "timeline_ref") !=
            cutoff["timeline_ref"] or
            isinstance(cutoff["cutoff_seconds"], bool) or
            not isinstance(cutoff["cutoff_seconds"], (int, float)) or
            cutoff["cutoff_seconds"] < 0):
        raise ValueError("invalid relative as-of cutoff")
    for field in ("case_id", "authorization_id", "host_id", "self_id",
                  "authority_id", "source_family"):
        if require_identifier(envelope[field], field) != envelope[field]:
            raise ValueError(f"{field} is not canonical")
    _exact(envelope["source_window"], {"path", "sha256"}, "upstream source window")
    upstream_ref = envelope["source_window"]
    upstream_path = Path(upstream_ref["path"])
    if (upstream_path.is_absolute() or not upstream_ref["path"] or
            any(part in {"", ".", ".."} for part in upstream_ref["path"].split("/"))):
        raise ValueError("source window must stay under the envelope directory")
    original = (envelope_path.parent / upstream_path).resolve()
    if (not original.is_relative_to(envelope_path.parent.resolve()) or
            _digest(original.read_bytes()) != upstream_ref["sha256"]):
        raise ValueError("original source window differs from the pin")
    window = _unique_json(original.read_bytes(), "original source window")
    if window.get("schema") == "mfm-ami-original-word-window-v1":
        prefix, family, spans = "ami", "ami-manual-2017", window["derived_turns"]
        original_cutoff = window["as_of"]["window_end_seconds"]
        speaker = lambda span: span["participant_id"]
        end = lambda span: span["end_seconds"]
    elif window.get("schema") == "mfm-v16-icsi-derived-source-excerpt-v1":
        prefix, family, spans = "icsi", "icsi-core-nxt-2016", window["spans"]
        original_cutoff = window["clip_end_seconds"]
        speaker = lambda span: span["participant_id"]
        end = lambda span: float(span["end_seconds"])
    else:
        raise ValueError("unsupported original meeting source window")
    meeting = window["meeting_id"].lower()
    if envelope["source_family"] != family or cutoff != {
            "timeline_ref": f"{prefix}-{meeting}",
            "cutoff_seconds": original_cutoff}:
        raise ValueError("teacher source cutoff differs from original window")
    opened = []
    seen = set()
    for item in envelope["sources"]:
        _exact(item, {"ref_id", "role", "modality", "subject_ref", "speaker_ref",
                      "source_item_ref", "observed_at", "recorded_at",
                      "temporal_granularity", "minimum_sensitivity", "path", "sha256",
                      "relative_time"},
               "source item")
        if item["ref_id"] in seen or item["modality"] != "text" or item["role"] not in {
                "outside_source", "tool_or_action_observation", "mission_state"}:
            raise ValueError("public history cannot claim an owner/self event or non-text interpretation")
        relative = item["relative_time"]
        _exact(relative, {"timeline_ref", "end_seconds"}, "relative source time")
        if (item["observed_at"] is not None or
                relative["timeline_ref"] != cutoff["timeline_ref"] or
                isinstance(relative["end_seconds"], bool) or
                not isinstance(relative["end_seconds"], (int, float)) or
                relative["end_seconds"] < 0 or
                relative["end_seconds"] > cutoff["cutoff_seconds"]):
            raise ValueError("source event occurs after the relative as-of cutoff")
        seen.add(item["ref_id"])
        rel = Path(item["path"])
        if (rel.is_absolute() or not item["path"] or
                any(part in {"", ".", ".."} for part in item["path"].split("/"))):
            raise ValueError("source path must stay under the envelope directory")
        location = (envelope_path.parent / rel).resolve()
        if not location.is_relative_to(envelope_path.parent.resolve()):
            raise ValueError("source symlink escapes the envelope directory")
        raw = location.read_bytes()
        if not raw or _digest(raw) != item["sha256"] or b"\0" in raw:
            raise ValueError("text evidence differs from pinned exact bytes")
        raw.decode("utf-8")
        ref_prefix = f"{prefix}-{meeting}-turn-"
        if not item["ref_id"].startswith(ref_prefix) or \
                not item["ref_id"][len(ref_prefix):].isdigit():
            raise ValueError("teacher source ref does not resolve to original turn")
        index = int(item["ref_id"][len(ref_prefix):])
        if index >= len(spans):
            raise ValueError("teacher turn index exceeds original window")
        span = spans[index]
        if (raw != span["text"].encode("utf-8") or
                item["speaker_ref"] != f"{prefix}-{speaker(span).lower()}" or
                item["subject_ref"] != item["speaker_ref"] or
                item["source_item_ref"] != item["ref_id"] + "-derived-transcript" or
                item["relative_time"] != {
                    "timeline_ref": f"{prefix}-{meeting}", "end_seconds": end(span)}):
            raise ValueError("teacher text/speaker/time differs from original turn")
        opened.append((item, raw))
    if not isinstance(envelope["targets"], list):
        raise ValueError("target registrations must be an array")
    return opened


def source_view(envelope: dict, opened: list[tuple[dict, bytes]]) -> dict:
    return {"schema": VIEW_SCHEMA, "case_id": envelope["case_id"],
            "split": envelope["split"], "host_id": envelope["host_id"],
            "self_id": envelope["self_id"], "authority_id": envelope["authority_id"],
            "source_family": envelope["source_family"], "as_of": envelope["as_of"],
            "source_window_sha256": envelope["source_window"]["sha256"],
            "training_admitted": False, "rights_status": "unverified",
            "sources": [
                {**{key: value for key, value in item.items() if key not in {"path", "sha256"}},
                 "content_b64": b64encode(raw).decode("ascii")}
                for item, raw in opened], "targets": envelope["targets"]}


def rendered_prompt(view_raw: bytes) -> bytes:
    return PROMPT.read_bytes().rstrip() + b"\n\nSOURCE_VIEW_JSONL:\n" + view_raw


def convert(envelope: dict, opened: list[tuple[dict, bytes]], response: dict, *,
            envelope_sha256: str, response_sha256: str,
            prompt_sha256: str) -> tuple[dict, dict]:
    view = source_view(envelope, opened)
    view_raw = _raw(view)
    _exact(response, {"schema", "case_id", "source_view_sha256",
                      "rendered_prompt_sha256", "proposals", "dispositions",
                      "adjudications"}, "teacher response")
    if (response["schema"] != RESPONSE_SCHEMA or
            response["case_id"] != envelope["case_id"] or
            response["source_view_sha256"] != _digest(view_raw) or
            response["rendered_prompt_sha256"] != prompt_sha256):
        raise ValueError("teacher answer does not bind to exact source view and prompt")
    scope = ProductHostScope.create(product_id="alice",
                                    host_instance_id=envelope["host_id"],
                                    schema_version="1.0.0",
                                    encryption_domain="mfm-public-history-teacher")
    authority = envelope["authority_id"]
    evidence, sensitivities = [], []
    for item, raw in opened:
        evidence.append(FormationEvidenceRef(
            ref_id=item["ref_id"], scope=scope, authority_namespace_id=authority,
            content_digest=_digest(raw), role=item["role"], modality="text",
            subject_ref=item["subject_ref"], speaker_ref=item["speaker_ref"],
            source_item_ref=item["source_item_ref"],
            observed_at=item["observed_at"], recorded_at=item["recorded_at"],
            temporal_granularity=item["temporal_granularity"]))
        sensitivities.append(RegisteredSourceSensitivity(
            item["ref_id"], item["minimum_sensitivity"]))
    targets = []
    for item in envelope["targets"]:
        _exact(item, {"ref_id", "kind", "supporting_evidence_refs"},
               "registered target")
        targets.append(RegisteredFormationTarget(
            item["ref_id"], item["kind"], scope, authority,
            tuple(item["supporting_evidence_refs"])))
    base = FormationContextPacket(scope, authority,
                                  tuple(item["ref_id"] for item, _ in opened),
                                  tuple(evidence))
    context = FormationContextV16(base, tuple(sensitivities), tuple(targets))
    context.validate()
    bytes_by_ref = {item["ref_id"]: raw for item, raw in opened}
    proposals = []
    for item in response["proposals"]:
        _exact(item, {"proposal", "sensitivity_hint", "episode",
                      "relationship_counterpart_ref", "mission_target_refs",
                      "workspace_target_refs"}, "teacher proposal")
        proposal = dict(item["proposal"])
        if set(proposal) != {"proposal_id", "kind", "domain", "subject_ref",
                             "value_ref", "evidence_refs", "value_text", "anchors",
                             "epistemic_status", "valid_from", "valid_to",
                             "temporal_granularity", "confidence", "uncertainty_ref",
                             "contradicts", "target_refs", "disposition_scope_ref"}:
            raise ValueError("teacher proposal fields differ from v1.6 contract")
        anchors = []
        for anchor in proposal["anchors"]:
            ref = anchor["ref_id"]
            if ref not in bytes_by_ref:
                raise ValueError("teacher cited an unavailable source")
            anchors.append(_quote_anchor(ref, bytes_by_ref[ref], anchor, "text"))
        proposal["anchors"] = anchors
        proposals.append({**item, "proposal": proposal})
    states = response["adjudications"]
    if not isinstance(states, dict) or set(states) != FULL_ROLE_DIMENSIONS or \
            set(states.values()) - {"present", "negative"}:
        raise ValueError("teacher must adjudicate all ten scoped dimensions")
    case = {"schema": CURRICULUM_SCHEMA_V16,
            "case_id": envelope["case_id"], "split": envelope["split"],
            "authorization_id": envelope["authorization_id"],
            "context": context.record(),
            "sources": [{"ref_id": item["ref_id"],
                         "content_b64": b64encode(raw).decode("ascii")}
                        for item, raw in opened],
            "target": {"schema": TARGET_SCHEMA_V16, "proposals": proposals,
                       "dispositions": response["dispositions"],
                       "adjudications": states},
            "lineage": {"source_family": envelope["source_family"],
                        "parent_source_sha256": envelope_sha256},
            "status": {"training_admitted": False, "rights": "unverified",
                       "review": "single-teacher-unreviewed",
                       "qualification": "source-grounded-candidate-only"}}
    example = learning_example_v16_from_record(case, split=envelope["split"])
    supervised_output_record_v16(example)
    provenance = {"schema": "mfm-v16-observed-history-teacher-provenance-v1",
                  "case_id": envelope["case_id"], "training_admitted": False,
                  "envelope_sha256": envelope_sha256,
                  "source_window_sha256": envelope["source_window"]["sha256"],
                  "source_sha256s": {item["ref_id"]: _digest(raw)
                                     for item, raw in opened},
                  "source_view_sha256": _digest(view_raw),
                  "rendered_prompt_sha256": prompt_sha256,
                  "teacher_response_sha256": response_sha256,
                  "prompt_template_sha256": _digest(PROMPT.read_bytes()),
                  "converter_sha256": _digest(Path(__file__).read_bytes()),
                  "case_sha256": _digest(_raw(case)),
                  "rights_status": "unverified", "independent_review": False,
                  "semantic_entailment_reviewed_independently": False}
    return case, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--envelope", type=Path, required=True)
    prepare.add_argument("--envelope-sha256", required=True)
    prepare.add_argument("--view", type=Path, required=True)
    prepare.add_argument("--prompt", type=Path, required=True)
    converted = commands.add_parser("convert")
    for name in ("envelope", "view", "prompt", "response", "case", "provenance"):
        converted.add_argument("--" + name, type=Path, required=True)
    for name in ("envelope", "view", "prompt", "response"):
        converted.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    envelope = _load(args.envelope, args.envelope_sha256)
    opened = _sources(envelope, args.envelope)
    view_raw = _raw(source_view(envelope, opened))
    prompt_raw = rendered_prompt(view_raw)
    if args.mode == "prepare":
        write_private_new(args.view, view_raw)
        write_private_new(args.prompt, prompt_raw)
        print(json.dumps({"case_id": envelope["case_id"],
                          "view_sha256": _digest(view_raw),
                          "prompt_sha256": _digest(prompt_raw),
                          "training_admitted": False}, sort_keys=True))
        return
    for path, expected, raw in ((args.view, args.view_sha256, view_raw),
                                (args.prompt, args.prompt_sha256, prompt_raw)):
        _outside_repo(path)
        if _digest(path.read_bytes()) != expected or path.read_bytes() != raw:
            raise ValueError("teacher view/prompt differs from pinned source envelope")
    response = _load(args.response, args.response_sha256)
    case, provenance = convert(envelope, opened, response,
                               envelope_sha256=args.envelope_sha256,
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
